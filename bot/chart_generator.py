import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import mplfinance as mpf
from datetime import datetime
from typing import Optional, Tuple
import json
import asyncio
import gc
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from bot.logger import setup_logger

logger = setup_logger('ChartGenerator')

FALLBACK_CHART_TIMEOUT = 60.0
FALLBACK_SHUTDOWN_TIMEOUT = 30.0
TIMEOUT_WARNING_THRESHOLD = 0.8
MAX_DATAFRAME_ROWS = 5000
LARGE_DATAFRAME_THRESHOLD = 2000
FILE_CLEANUP_MAX_RETRIES = 3
FILE_CLEANUP_RETRY_DELAY = 0.5

class ChartError(Exception):
    """Base exception untuk chart generation errors"""
    pass

class ChartGeneratorError(ChartError):
    """Exception untuk chart generator errors"""
    pass

class DataValidationError(ChartError):
    """Chart data validation error"""
    pass

class ChartTimeoutError(ChartError):
    """Exception untuk chart generation timeout"""
    pass

class ChartCleanupError(ChartError):
    """Exception untuk chart cleanup errors"""
    pass

def validate_chart_data(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """Validate DataFrame untuk chart generation"""
    try:
        if df is None:
            return False, "DataFrame adalah None"
        
        if not isinstance(df, pd.DataFrame):
            return False, f"Tipe tidak valid: {type(df)}. Diharapkan pandas.DataFrame"
        
        if len(df) < 10:
            return False, f"Data tidak cukup: {len(df)} baris (minimum 10 diperlukan)"
        
        if len(df) > MAX_DATAFRAME_ROWS:
            return False, f"DataFrame terlalu besar: {len(df)} baris (maksimum {MAX_DATAFRAME_ROWS} baris)"
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return False, f"Kolom yang diperlukan tidak ada: {missing_cols}"
        
        for col in ['open', 'high', 'low', 'close']:
            if bool(df[col].isnull().any()):
                null_count = int(df[col].isnull().sum())
                return False, f"Kolom '{col}' mengandung {null_count} nilai null"
            
            if (df[col] <= 0).any():
                return False, f"Kolom '{col}' mengandung nilai non-positif"
        
        if (df['high'] < df['low']).any():
            invalid_count = (df['high'] < df['low']).sum()
            return False, f"Ditemukan {invalid_count} candle dimana high < low"
        
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' not in df.columns:
                return False, f"Index harus DatetimeIndex atau DataFrame harus memiliki kolom 'timestamp'. Ditemukan {type(df.index)} dengan kolom: {df.columns.tolist()}"
        
        return True, None
        
    except Exception as e:
        return False, f"Error validasi: {str(e)}"

class ChartGenerator:
    def __init__(self, config):
        self.config = config
        self.chart_dir = 'charts'
        os.makedirs(self.chart_dir, exist_ok=True)
        max_workers = 1 if self.config.FREE_TIER_MODE else 2
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="chart_gen")
        self._shutdown_requested = False
        self._pending_charts: set = set()
        self._pending_tasks: set = set()
        self._active_futures: dict = {}
        self._chart_lock = asyncio.Lock()
        self._task_lock = asyncio.Lock()
        self._future_lock = asyncio.Lock()
        self._file_lock = asyncio.Lock()
        self._sync_file_lock = threading.Lock()
        self._sync_chart_lock = threading.Lock()
        self._timed_out_tasks: set = set()
        
        self.chart_timeout = getattr(config, 'DEFAULT_CHART_TIMEOUT', None)
        if self.chart_timeout is None:
            self.chart_timeout = FALLBACK_CHART_TIMEOUT
            logger.debug(f"DEFAULT_CHART_TIMEOUT tidak ada di config, menggunakan fallback: {FALLBACK_CHART_TIMEOUT}s")
        
        self.shutdown_timeout = getattr(config, 'DEFAULT_SHUTDOWN_TIMEOUT', None)
        if self.shutdown_timeout is None:
            self.shutdown_timeout = FALLBACK_SHUTDOWN_TIMEOUT
            logger.debug(f"DEFAULT_SHUTDOWN_TIMEOUT tidak ada di config, menggunakan fallback: {FALLBACK_SHUTDOWN_TIMEOUT}s")
        
        logger.info(f"ChartGenerator diinisialisasi: max_workers={max_workers}, chart_timeout={self.chart_timeout}s, shutdown_timeout={self.shutdown_timeout}s (FREE_TIER_MODE={self.config.FREE_TIER_MODE})")
    
    def _check_dataframe_size(self, df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """Periksa ukuran dataframe dan berikan warning jika terlalu besar"""
        if df is None:
            return False, "DataFrame adalah None"
        
        row_count = len(df)
        
        if row_count > MAX_DATAFRAME_ROWS:
            return False, f"DataFrame melebihi batas maksimum: {row_count} baris > {MAX_DATAFRAME_ROWS} baris"
        
        if row_count > LARGE_DATAFRAME_THRESHOLD:
            logger.warning(f"⚠️ DataFrame besar terdeteksi: {row_count} baris - proses mungkin memakan waktu lebih lama")
        
        return True, None
    
    def generate_chart(self, df: pd.DataFrame, signal: Optional[dict] = None,
                      timeframe: str = 'M1') -> Optional[str]:
        """Generate chart dengan validasi komprehensif dan error handling"""
        chart_path = None
        start_time = time.monotonic()
        
        try:
            size_ok, size_error = self._check_dataframe_size(df)
            if not size_ok:
                logger.error(f"Ukuran dataframe tidak valid: {size_error}")
                return None
            
            is_valid, error_msg = validate_chart_data(df)
            if not is_valid:
                logger.warning(f"Validasi data chart gagal: {error_msg}")
                return None
            
            with self._sync_chart_lock:
                df_copy = df.copy()
            
            if not isinstance(df_copy.index, pd.DatetimeIndex):
                if 'timestamp' in df_copy.columns:
                    try:
                        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
                        df_copy.set_index('timestamp', inplace=True)
                    except Exception as e:
                        logger.error(f"Error mengkonversi timestamp ke DatetimeIndex: {e}")
                        return None
                else:
                    logger.error("DataFrame tidak memiliki DatetimeIndex dan tidak ada kolom 'timestamp'")
                    return None
            
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df_copy.columns for col in required_cols):
                logger.error(f"Kolom yang diperlukan tidak ada. Dimiliki: {df_copy.columns.tolist()}, Diperlukan: {required_cols}")
                return None
            
            elapsed = time.monotonic() - start_time
            chart_timeout_value = self.chart_timeout if self.chart_timeout is not None else FALLBACK_CHART_TIMEOUT
            if elapsed > chart_timeout_value * TIMEOUT_WARNING_THRESHOLD:
                logger.warning(f"⚠️ Persiapan chart sudah menggunakan {elapsed:.2f}s ({(elapsed/chart_timeout_value)*100:.1f}% dari timeout)")
            
            addplot = []
            
            from bot.indicators import IndicatorEngine
            indicator_engine = IndicatorEngine(self.config)
            
            ema_5 = df_copy['close'].ewm(span=self.config.EMA_PERIODS[0], adjust=False).mean().bfill().ffill()
            ema_10 = df_copy['close'].ewm(span=self.config.EMA_PERIODS[1], adjust=False).mean().bfill().ffill()
            ema_20 = df_copy['close'].ewm(span=self.config.EMA_PERIODS[2], adjust=False).mean().bfill().ffill()
            
            addplot.append(mpf.make_addplot(ema_5, color='blue', width=1.5, panel=0, label=f'EMA {self.config.EMA_PERIODS[0]}'))
            addplot.append(mpf.make_addplot(ema_10, color='orange', width=1.5, panel=0, label=f'EMA {self.config.EMA_PERIODS[1]}'))
            addplot.append(mpf.make_addplot(ema_20, color='red', width=1.5, panel=0, label=f'EMA {self.config.EMA_PERIODS[2]}'))
            
            delta = df_copy['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.config.RSI_PERIOD).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.RSI_PERIOD).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            if isinstance(rsi, pd.Series):
                rsi = rsi.fillna(50)
            else:
                rsi = pd.Series([50] * len(df_copy), index=df_copy.index)
            
            addplot.append(mpf.make_addplot(rsi, color='purple', width=1.5, panel=1, ylabel='RSI', ylim=(0, 100)))
            
            rsi_70 = pd.Series([70] * len(df_copy), index=df_copy.index)
            rsi_30 = pd.Series([30] * len(df_copy), index=df_copy.index)
            addplot.append(mpf.make_addplot(rsi_70, color='red', width=0.8, panel=1, linestyle='--', alpha=0.5))
            addplot.append(mpf.make_addplot(rsi_30, color='green', width=0.8, panel=1, linestyle='--', alpha=0.5))
            
            low_min = df_copy['low'].rolling(window=self.config.STOCH_K_PERIOD).min()
            high_max = df_copy['high'].rolling(window=self.config.STOCH_K_PERIOD).max()
            stoch_k_raw = 100 * (df_copy['close'] - low_min) / (high_max - low_min)
            if isinstance(stoch_k_raw, pd.Series):
                stoch_k_smooth = stoch_k_raw.rolling(window=self.config.STOCH_SMOOTH_K).mean()
                if isinstance(stoch_k_smooth, pd.Series):
                    stoch_d_raw = stoch_k_smooth.rolling(window=self.config.STOCH_D_PERIOD).mean()
                    stoch_k = stoch_k_smooth.fillna(50)
                    if isinstance(stoch_d_raw, pd.Series):
                        stoch_d = stoch_d_raw.fillna(50)
                    else:
                        stoch_d = pd.Series([50] * len(df_copy), index=df_copy.index)
                else:
                    stoch_k = pd.Series([50] * len(df_copy), index=df_copy.index)
                    stoch_d = pd.Series([50] * len(df_copy), index=df_copy.index)
            else:
                stoch_k = pd.Series([50] * len(df_copy), index=df_copy.index)
                stoch_d = pd.Series([50] * len(df_copy), index=df_copy.index)
            
            addplot.append(mpf.make_addplot(stoch_k, color='blue', width=1.5, panel=2, ylabel='Stochastic', ylim=(0, 100)))
            addplot.append(mpf.make_addplot(stoch_d, color='orange', width=1.5, panel=2))
            
            stoch_80 = pd.Series([80] * len(df_copy), index=df_copy.index)
            stoch_20 = pd.Series([20] * len(df_copy), index=df_copy.index)
            addplot.append(mpf.make_addplot(stoch_80, color='red', width=0.8, panel=2, linestyle='--', alpha=0.5))
            addplot.append(mpf.make_addplot(stoch_20, color='green', width=0.8, panel=2, linestyle='--', alpha=0.5))
            
            trf_upper, trf_lower, trf_trend = indicator_engine.calculate_twin_range_filter(df_copy, period=27, multiplier=2.0)
            trf_upper = trf_upper.bfill().ffill()
            trf_lower = trf_lower.bfill().ffill()
            addplot.append(mpf.make_addplot(trf_upper, color='cyan', width=1.5, panel=0, linestyle='--', alpha=0.7, label='TRF Upper'))
            addplot.append(mpf.make_addplot(trf_lower, color='magenta', width=1.5, panel=0, linestyle='--', alpha=0.7, label='TRF Lower'))
            
            cerebr_value, cerebr_signal, cerebr_bias = indicator_engine.calculate_market_bias_cerebr(df_copy, period=60, smoothing=10)
            cerebr_value = cerebr_value.fillna(50)
            cerebr_signal = cerebr_signal.fillna(50)
            addplot.append(mpf.make_addplot(cerebr_value, color='teal', width=1.5, panel=3, ylabel='CEREBR', ylim=(0, 100)))
            addplot.append(mpf.make_addplot(cerebr_signal, color='orange', width=1.0, panel=3, linestyle='--', alpha=0.7))
            cerebr_60 = pd.Series([60] * len(df_copy), index=df_copy.index)
            cerebr_40 = pd.Series([40] * len(df_copy), index=df_copy.index)
            addplot.append(mpf.make_addplot(cerebr_60, color='red', width=0.8, panel=3, linestyle=':', alpha=0.5))
            addplot.append(mpf.make_addplot(cerebr_40, color='green', width=0.8, panel=3, linestyle=':', alpha=0.5))
            
            if signal:
                entry_price = signal.get('entry_price')
                stop_loss = signal.get('stop_loss')
                take_profit = signal.get('take_profit')
                signal_type = signal.get('signal')
                
                if entry_price and signal_type:
                    marker_color = 'lime' if signal_type == 'BUY' else 'red'
                    marker_symbol = '^' if signal_type == 'BUY' else 'v'
                    
                    import numpy as np
                    marker_series = pd.Series(index=df_copy.index, data=[np.nan] * len(df_copy))
                    marker_series.iloc[-1] = entry_price
                    
                    addplot.append(
                        mpf.make_addplot(marker_series, type='scatter', 
                                        markersize=150, marker=marker_symbol, 
                                        color=marker_color, panel=0)
                    )
                
                if stop_loss:
                    sl_line = pd.Series(index=df_copy.index, data=[stop_loss] * len(df_copy))
                    addplot.append(
                        mpf.make_addplot(sl_line, type='line', 
                                        color='darkred', linestyle='--', 
                                        width=2.5, panel=0, 
                                        alpha=0.8)
                    )
                
                if take_profit:
                    tp_line = pd.Series(index=df_copy.index, data=[take_profit] * len(df_copy))
                    addplot.append(
                        mpf.make_addplot(tp_line, type='line', 
                                        color='darkgreen', linestyle='--', 
                                        width=2.5, panel=0, 
                                        alpha=0.8)
                    )
            
            mc = mpf.make_marketcolors(
                up='lime', down='red',
                edge='inherit',
                wick='inherit',
                volume='in'
            )
            
            style = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle=':',
                gridcolor='gray',
                gridaxis='both',
                y_on_right=False,
                rc={'font.size': 10}
            )
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'xauusd_{timeframe}_{timestamp}.png'
            filepath = os.path.join(self.chart_dir, filename)
            
            title = f'XAUUSD {timeframe} - Analisis Teknikal'
            if signal:
                title += f" ({signal.get('signal', 'SIGNAL')} Signal)"
            
            try:
                with self._sync_file_lock:
                    fig, axes = mpf.plot(
                        df_copy,
                        type='candle',
                        style=style,
                        title=title,
                        ylabel='Price (USD)',
                        volume=True,
                        addplot=addplot if addplot else None,
                        savefig=filepath,
                        figsize=(14, 14),
                        returnfig=True,
                        panel_ratios=(3, 1, 1, 1)
                    )
                chart_path = filepath
                
                elapsed_total = time.monotonic() - start_time
                logger.info(f"✅ Chart berhasil dibuat: {filepath} ({len(df_copy)} candle) dalam {elapsed_total:.2f}s")
                
            except Exception as plot_error:
                logger.error(f"Error plotting: {type(plot_error).__name__}: {plot_error}")
                return None
            finally:
                try:
                    import matplotlib.pyplot as plt
                    plt.close('all')
                    gc.collect()
                except Exception as cleanup_error:
                    logger.warning(f"Error saat cleanup matplotlib: {cleanup_error}")
            
            return chart_path
            
        except MemoryError:
            logger.error("Memory error saat generate chart - memori tidak cukup")
            try:
                gc.collect()
            except Exception:
                pass
            return None
        except Exception as e:
            logger.error(f"Error tidak terduga saat generate chart: {type(e).__name__}: {e}", exc_info=True)
            return None
    
    async def generate_chart_async(self, df: pd.DataFrame, signal: Optional[dict] = None,
                                   timeframe: str = 'M1', timeout: Optional[float] = None) -> Optional[str]:
        """Generate chart secara asynchronous dengan timeout dan proper executor cleanup"""
        task_id = None
        future = None
        future_id = None
        effective_timeout = timeout if timeout is not None else (self.chart_timeout if self.chart_timeout is not None else FALLBACK_CHART_TIMEOUT)
        start_time = time.monotonic()
        warning_threshold_seconds = effective_timeout * TIMEOUT_WARNING_THRESHOLD
        
        try:
            if self._shutdown_requested:
                logger.warning("Pembuatan chart dilewati - shutdown sedang berlangsung")
                return None
            
            if df is None or len(df) < 10:
                logger.warning(f"Data tidak cukup untuk async chart: {len(df) if df is not None else 0} candle")
                return None
            
            size_ok, size_error = self._check_dataframe_size(df)
            if not size_ok:
                logger.error(f"Ukuran dataframe tidak valid untuk async chart: {size_error}")
                return None
            
            if len(df) > LARGE_DATAFRAME_THRESHOLD:
                if effective_timeout is not None:
                    adjusted_timeout = effective_timeout * 1.5
                    logger.info(f"DataFrame besar ({len(df)} baris) - menyesuaikan timeout ke {adjusted_timeout:.1f}s")
                    effective_timeout = adjusted_timeout
                    warning_threshold_seconds = effective_timeout * TIMEOUT_WARNING_THRESHOLD
            
            loop = asyncio.get_running_loop()
            
            current_task = asyncio.current_task()
            task_id = id(current_task) if current_task else None
            
            if task_id is not None:
                async with self._task_lock:
                    self._pending_tasks.add(task_id)
                    logger.debug(f"Melacak chart task {task_id}, total pending: {len(self._pending_tasks)}")
            
            future = loop.run_in_executor(
                self.executor,
                self.generate_chart,
                df,
                signal,
                timeframe
            )
            future_id = id(future)
            
            async with self._future_lock:
                self._active_futures[future_id] = future
            
            result = await asyncio.wait_for(future, timeout=effective_timeout)
            
            elapsed_time = time.monotonic() - start_time
            if effective_timeout is not None and elapsed_time > warning_threshold_seconds:
                pct = (elapsed_time / effective_timeout) * 100
                logger.warning(f"⚠️ Pembuatan chart memakan waktu {elapsed_time:.2f}s ({pct:.1f}% dari {effective_timeout}s timeout) - pertimbangkan optimasi atau tingkatkan timeout")
            
            return result
            
        except asyncio.TimeoutError:
            elapsed_time = time.monotonic() - start_time
            logger.warning(f"Pembuatan chart timeout setelah {elapsed_time:.2f}s (batas: {effective_timeout}s)")
            
            if future is not None and future_id is not None:
                async with self._future_lock:
                    self._timed_out_tasks.add(future_id)
                    logger.debug(f"Menambahkan future {future_id} ke timed_out_tasks, total: {len(self._timed_out_tasks)}")
                
                try:
                    if hasattr(future, 'cancel'):
                        future.cancel()
                        logger.debug(f"Membatalkan executor future yang timeout {future_id}")
                except Exception as cancel_err:
                    logger.debug(f"Tidak dapat membatalkan future {future_id}: {cancel_err}")
                
                try:
                    gc.collect()
                except Exception:
                    pass
            
            return None
        except asyncio.CancelledError:
            logger.info("Pembuatan chart dibatalkan")
            if future is not None:
                try:
                    if hasattr(future, 'cancel'):
                        future.cancel()
                except Exception:
                    pass
            raise
        except RuntimeError as e:
            if "no running event loop" in str(e).lower():
                logger.warning("Tidak ada event loop yang berjalan, fallback ke sync generation")
                return self.generate_chart(df, signal, timeframe)
            logger.error(f"Runtime error dalam async chart generation: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Error dalam async chart generation: {type(e).__name__}: {e}", exc_info=True)
            return None
        finally:
            if task_id is not None:
                try:
                    async with self._task_lock:
                        self._pending_tasks.discard(task_id)
                        logger.debug(f"Berhenti melacak chart task {task_id}, tersisa: {len(self._pending_tasks)}")
                except Exception as e:
                    logger.warning(f"Error saat berhenti melacak task {task_id}: {e}")
            
            if future_id is not None:
                try:
                    async with self._future_lock:
                        self._active_futures.pop(future_id, None)
                except Exception as e:
                    logger.debug(f"Error menghapus future {future_id}: {e}")
    
    def _delete_file_with_retry(self, filepath: str, max_retries: int = FILE_CLEANUP_MAX_RETRIES) -> bool:
        """Hapus file dengan retry mechanism untuk handle concurrent access"""
        if not filepath or not isinstance(filepath, str):
            return False
        
        for attempt in range(max_retries):
            try:
                if not os.path.exists(filepath):
                    return True
                
                with self._sync_file_lock:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        logger.debug(f"File berhasil dihapus: {filepath} (percobaan {attempt + 1})")
                        return True
                    return True
                    
            except PermissionError as e:
                if attempt < max_retries - 1:
                    delay = FILE_CLEANUP_RETRY_DELAY * (2 ** attempt)
                    logger.debug(f"File sedang digunakan, retry dalam {delay:.2f}s: {filepath}")
                    time.sleep(delay)
                else:
                    logger.warning(f"Gagal menghapus file setelah {max_retries} percobaan (PermissionError): {filepath}")
                    return False
            except FileNotFoundError:
                return True
            except OSError as e:
                if attempt < max_retries - 1:
                    delay = FILE_CLEANUP_RETRY_DELAY * (2 ** attempt)
                    logger.debug(f"OSError saat menghapus file, retry dalam {delay:.2f}s: {filepath} - {e}")
                    time.sleep(delay)
                else:
                    logger.warning(f"Gagal menghapus file setelah {max_retries} percobaan (OSError): {filepath} - {e}")
                    return False
            except Exception as e:
                logger.error(f"Error tidak terduga saat menghapus file {filepath}: {type(e).__name__}: {e}")
                return False
        
        return False
    
    async def _delete_file_with_retry_async(self, filepath: str, max_retries: int = FILE_CLEANUP_MAX_RETRIES) -> bool:
        """Hapus file secara async dengan retry mechanism untuk handle concurrent access"""
        if not filepath or not isinstance(filepath, str):
            return False
        
        for attempt in range(max_retries):
            try:
                if not os.path.exists(filepath):
                    return True
                
                async with self._file_lock:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        logger.debug(f"File berhasil dihapus (async): {filepath} (percobaan {attempt + 1})")
                        return True
                    return True
                    
            except PermissionError as e:
                if attempt < max_retries - 1:
                    delay = FILE_CLEANUP_RETRY_DELAY * (2 ** attempt)
                    logger.debug(f"File sedang digunakan, retry async dalam {delay:.2f}s: {filepath}")
                    await asyncio.sleep(delay)
                else:
                    logger.warning(f"Gagal menghapus file async setelah {max_retries} percobaan (PermissionError): {filepath}")
                    return False
            except FileNotFoundError:
                return True
            except OSError as e:
                if attempt < max_retries - 1:
                    delay = FILE_CLEANUP_RETRY_DELAY * (2 ** attempt)
                    logger.debug(f"OSError saat menghapus file async, retry dalam {delay:.2f}s: {filepath} - {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.warning(f"Gagal menghapus file async setelah {max_retries} percobaan (OSError): {filepath} - {e}")
                    return False
            except Exception as e:
                logger.error(f"Error tidak terduga saat menghapus file async {filepath}: {type(e).__name__}: {e}")
                return False
        
        return False
    
    def delete_chart(self, filepath: str) -> bool:
        """Hapus file chart dengan validasi dan graceful error handling"""
        try:
            if not filepath or not isinstance(filepath, str):
                logger.warning(f"Filepath tidak valid untuk penghapusan: {filepath}")
                return False
            
            if not filepath.endswith('.png'):
                logger.warning(f"Mencoba menghapus file non-PNG: {filepath}")
                return False
            
            return self._delete_file_with_retry(filepath)
            
        except Exception as e:
            logger.error(f"Error menghapus chart {filepath}: {type(e).__name__}: {e}")
            return False
    
    async def delete_chart_async(self, filepath: str) -> bool:
        """Hapus file chart secara async dengan validasi dan graceful error handling"""
        try:
            if not filepath or not isinstance(filepath, str):
                logger.warning(f"Filepath tidak valid untuk penghapusan async: {filepath}")
                return False
            
            if not filepath.endswith('.png'):
                logger.warning(f"Mencoba menghapus file non-PNG (async): {filepath}")
                return False
            
            return await self._delete_file_with_retry_async(filepath)
            
        except Exception as e:
            logger.error(f"Error menghapus chart async {filepath}: {type(e).__name__}: {e}")
            return False
    
    def shutdown(self, timeout: Optional[float] = None):
        """Graceful synchronous shutdown dengan cleanup semua pending charts"""
        effective_timeout = timeout if timeout is not None else self.shutdown_timeout
        
        try:
            logger.info(f"Mematikan ChartGenerator executor (timeout={effective_timeout}s)...")
            self._shutdown_requested = True
            
            pending_task_count = len(self._pending_tasks)
            if pending_task_count > 0:
                logger.info(f"Membatalkan {pending_task_count} pending chart tasks...")
                self._pending_tasks.clear()
            
            timed_out_count = len(self._timed_out_tasks)
            if timed_out_count > 0:
                logger.warning(f"⚠️ Ditemukan {timed_out_count} task yang timeout saat shutdown: {list(self._timed_out_tasks)[:10]}{'...' if timed_out_count > 10 else ''}")
                self._timed_out_tasks.clear()
                logger.debug("Set timed out tasks dibersihkan")
            
            active_futures_count = len(self._active_futures)
            if active_futures_count > 0:
                logger.info(f"Membersihkan {active_futures_count} active futures...")
                for future_id, future in list(self._active_futures.items()):
                    try:
                        if hasattr(future, 'cancel'):
                            future.cancel()
                            logger.debug(f"Membatalkan active future {future_id}")
                    except Exception as cancel_err:
                        logger.debug(f"Tidak dapat membatalkan future {future_id}: {cancel_err}")
                self._active_futures.clear()
                logger.debug("Dict active futures dibersihkan")
            
            try:
                self.executor.shutdown(wait=True, cancel_futures=True)
            except Exception as executor_error:
                logger.warning(f"Error saat executor shutdown: {executor_error}")
                try:
                    self.executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
            
            logger.info("ChartGenerator executor berhasil dimatikan")
        except Exception as e:
            logger.error(f"Error saat mematikan executor: {e}")
        finally:
            try:
                self._cleanup_pending_charts()
            except Exception as cleanup_error:
                logger.error(f"Error saat final cleanup: {cleanup_error}")
            
            try:
                gc.collect()
                logger.debug("gc.collect() selesai setelah shutdown")
            except Exception as gc_error:
                logger.debug(f"gc.collect() error: {gc_error}")
    
    async def shutdown_async(self, timeout: Optional[float] = None):
        """Graceful async shutdown dengan timeout dan cleanup semua pending charts"""
        effective_timeout = timeout if timeout is not None else self.shutdown_timeout
        
        try:
            logger.info(f"Mematikan ChartGenerator (async, timeout={effective_timeout}s)...")
            self._shutdown_requested = True
            
            async with self._task_lock:
                pending_count = len(self._pending_tasks)
                if pending_count > 0:
                    logger.info(f"Membatalkan {pending_count} pending chart tasks...")
                self._pending_tasks.clear()
            
            async with self._future_lock:
                timed_out_count = len(self._timed_out_tasks)
                if timed_out_count > 0:
                    logger.warning(f"⚠️ Ditemukan {timed_out_count} task yang timeout saat shutdown: {list(self._timed_out_tasks)[:10]}{'...' if timed_out_count > 10 else ''}")
                    self._timed_out_tasks.clear()
                    logger.debug("Set timed out tasks dibersihkan (async)")
                
                active_futures_count = len(self._active_futures)
                if active_futures_count > 0:
                    logger.info(f"Membersihkan {active_futures_count} active futures (async)...")
                    for future_id, future in list(self._active_futures.items()):
                        try:
                            if hasattr(future, 'cancel'):
                                future.cancel()
                                logger.debug(f"Membatalkan active future {future_id}")
                        except Exception as cancel_err:
                            logger.debug(f"Tidak dapat membatalkan future {future_id}: {cancel_err}")
                    self._active_futures.clear()
                    logger.debug("Dict active futures dibersihkan (async)")
            
            loop = asyncio.get_running_loop()
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self.executor.shutdown(wait=True, cancel_futures=True)
                    ),
                    timeout=effective_timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"Executor shutdown timeout setelah {effective_timeout}s, memaksa shutdown")
                try:
                    self.executor.shutdown(wait=False, cancel_futures=True)
                except Exception as force_error:
                    logger.error(f"Error saat memaksa executor shutdown: {force_error}")
            except Exception as executor_error:
                logger.error(f"Error saat async executor shutdown: {executor_error}")
                try:
                    self.executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
            
            logger.info("ChartGenerator executor berhasil dimatikan (async)")
        except Exception as e:
            logger.error(f"Error saat mematikan executor (async): {e}")
        finally:
            try:
                await self._cleanup_pending_charts_async()
            except Exception as cleanup_error:
                logger.error(f"Error saat final cleanup (async): {cleanup_error}")
            
            try:
                gc.collect()
                logger.debug("gc.collect() selesai setelah async shutdown")
            except Exception as gc_error:
                logger.debug(f"gc.collect() error (async): {gc_error}")
    
    def _cleanup_pending_charts(self):
        """Cleanup semua pending chart files dengan try-finally untuk memastikan cleanup"""
        cleaned = 0
        failed = 0
        failed_details = []
        charts_to_cleanup = list(self._pending_charts)
        
        try:
            for chart_path in charts_to_cleanup:
                try:
                    if self._delete_file_with_retry(chart_path):
                        cleaned += 1
                        logger.debug(f"Berhasil membersihkan pending chart: {chart_path}")
                    else:
                        failed += 1
                        failed_details.append(f"Retry gagal: {chart_path}")
                except Exception as e:
                    failed += 1
                    error_detail = f"{type(e).__name__}: {chart_path} - {e}"
                    failed_details.append(error_detail)
                    logger.warning(f"Gagal cleanup pending chart {chart_path}: {type(e).__name__}: {e}")
        except Exception as e:
            logger.error(f"Error kritis saat cleanup pending charts: {type(e).__name__}: {e}")
        finally:
            self._pending_charts.clear()
            
            if cleaned > 0 or failed > 0:
                logger.info(f"🗑️ Cleanup pending charts: {cleaned} berhasil, {failed} gagal")
            
            if failed_details:
                logger.debug(f"Detail kegagalan cleanup: {failed_details}")
    
    async def _cleanup_pending_charts_async(self):
        """Cleanup semua pending chart files secara async dengan try-finally"""
        cleaned = 0
        failed = 0
        failed_details = []
        charts_to_cleanup = list(self._pending_charts)
        
        try:
            for chart_path in charts_to_cleanup:
                try:
                    if await self._delete_file_with_retry_async(chart_path):
                        cleaned += 1
                        logger.debug(f"Berhasil membersihkan pending chart (async): {chart_path}")
                    else:
                        failed += 1
                        failed_details.append(f"Retry gagal (async): {chart_path}")
                except Exception as e:
                    failed += 1
                    error_detail = f"{type(e).__name__}: {chart_path} - {e}"
                    failed_details.append(error_detail)
                    logger.warning(f"Gagal cleanup pending chart async {chart_path}: {type(e).__name__}: {e}")
        except Exception as e:
            logger.error(f"Error kritis saat cleanup pending charts async: {type(e).__name__}: {e}")
        finally:
            self._pending_charts.clear()
            
            if cleaned > 0 or failed > 0:
                logger.info(f"🗑️ Cleanup pending charts (async): {cleaned} berhasil, {failed} gagal")
            
            if failed_details:
                logger.debug(f"Detail kegagalan cleanup (async): {failed_details}")
    
    def track_chart(self, filepath: str):
        """Track chart file untuk cleanup nanti"""
        if filepath:
            with self._sync_chart_lock:
                self._pending_charts.add(filepath)
    
    async def track_chart_async(self, filepath: str):
        """Track chart file secara async untuk cleanup nanti"""
        if filepath:
            async with self._chart_lock:
                self._pending_charts.add(filepath)
    
    def untrack_chart(self, filepath: str):
        """Untrack chart file setelah berhasil dikirim"""
        with self._sync_chart_lock:
            self._pending_charts.discard(filepath)
    
    async def untrack_chart_async(self, filepath: str):
        """Untrack chart file secara async setelah berhasil dikirim"""
        async with self._chart_lock:
            self._pending_charts.discard(filepath)
    
    async def cleanup_orphan_charts(self, max_age_minutes: int = 30) -> int:
        """Cleanup orphan chart files yang lebih tua dari max_age"""
        try:
            now = datetime.now()
            cleaned = 0
            
            async with self._chart_lock:
                try:
                    files_to_check = os.listdir(self.chart_dir)
                except OSError as e:
                    logger.error(f"Tidak dapat membaca direktori chart: {e}")
                    return 0
                
                for filename in files_to_check:
                    filepath = os.path.join(self.chart_dir, filename)
                    if os.path.isfile(filepath) and filename.endswith('.png'):
                        try:
                            file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                            age_minutes = (now - file_time).total_seconds() / 60
                            
                            if age_minutes > max_age_minutes:
                                if await self._delete_file_with_retry_async(filepath):
                                    async with self._chart_lock:
                                        self._pending_charts.discard(filepath)
                                    cleaned += 1
                        except Exception as e:
                            logger.warning(f"Error memeriksa umur chart {filepath}: {e}")
            
            if cleaned > 0:
                logger.info(f"🗑️ Berhasil membersihkan {cleaned} orphan chart files (lebih tua dari {max_age_minutes} menit)")
            
            return cleaned
        except Exception as e:
            logger.error(f"Error membersihkan orphan charts: {e}")
            return 0
    
    def cleanup_old_charts(self, days: int = 7):
        """Cleanup chart files yang lebih tua dari X hari"""
        try:
            now = datetime.now()
            cleaned = 0
            
            try:
                files_to_check = os.listdir(self.chart_dir)
            except OSError as e:
                logger.error(f"Tidak dapat membaca direktori chart: {e}")
                return
            
            for filename in files_to_check:
                filepath = os.path.join(self.chart_dir, filename)
                if os.path.isfile(filepath):
                    try:
                        file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                        if (now - file_time).days > days:
                            if self._delete_file_with_retry(filepath):
                                with self._sync_chart_lock:
                                    self._pending_charts.discard(filepath)
                                cleaned += 1
                                logger.debug(f"Menghapus chart lama: {filename}")
                    except FileNotFoundError:
                        pass
                    except Exception as e:
                        logger.warning(f"Error menghapus chart {filename}: {e}")
            
            if cleaned > 0:
                logger.info(f"🗑️ Berhasil membersihkan {cleaned} old chart files (lebih tua dari {days} hari)")
        except Exception as e:
            logger.error(f"Error membersihkan old charts: {e}")
    
    def get_stats(self) -> dict:
        """Dapatkan statistik chart generator"""
        try:
            chart_count = len([f for f in os.listdir(self.chart_dir) if f.endswith('.png')])
            pending_chart_count = len(self._pending_charts)
            pending_task_count = len(self._pending_tasks)
            timed_out_count = len(self._timed_out_tasks)
            active_futures_count = len(self._active_futures)
            
            return {
                'chart_dir': self.chart_dir,
                'total_charts': chart_count,
                'pending_charts': pending_chart_count,
                'pending_tasks': pending_task_count,
                'timed_out_tasks': timed_out_count,
                'active_futures': active_futures_count,
                'shutdown_requested': self._shutdown_requested,
                'free_tier_mode': self.config.FREE_TIER_MODE,
                'chart_timeout': self.chart_timeout,
                'shutdown_timeout': self.shutdown_timeout,
                'max_dataframe_rows': MAX_DATAFRAME_ROWS,
                'large_dataframe_threshold': LARGE_DATAFRAME_THRESHOLD
            }
        except Exception as e:
            logger.error(f"Error mendapatkan statistik chart: {e}")
            return {'error': str(e)}
