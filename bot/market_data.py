import asyncio
import threading
import websockets
import json
import math
import time
import gc
import aiohttp
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import pandas as pd
import pytz
import random
from typing import Optional, Dict, List, Tuple, Any, Union, Callable
from bot.logger import setup_logger
from bot.resilience import CircuitBreaker
from config import Config

logger = setup_logger('MarketData')


class ConnectionState(Enum):
    """WebSocket connection state machine states"""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"


class MarketDataError(Exception):
    """Base exception for market data errors"""
    pass


class WebSocketConnectionError(MarketDataError):
    """WebSocket connection error"""
    pass


class DataValidationError(MarketDataError):
    """Data validation error"""
    pass


class MarketDataTimeoutError(MarketDataError):
    """Operation timeout error for market data operations"""
    pass


def is_valid_price(price: Any) -> bool:
    """Check if price is a valid numeric value (not None, not NaN, positive)
    
    Args:
        price: Value to check
        
    Returns:
        True if price is valid, False otherwise
    """
    if price is None:
        return False
    
    if not isinstance(price, (int, float)):
        return False
    
    if math.isnan(price) or math.isinf(price):
        return False
    
    if price <= 0:
        return False
    
    return True


def sanitize_price_data(data: Dict) -> Tuple[bool, Dict, Optional[str]]:
    """Sanitize price data dictionary, removing or flagging NaN values
    
    Args:
        data: Dictionary containing price data
        
    Returns:
        Tuple of (is_valid, sanitized_data, error_message)
    """
    if not isinstance(data, dict):
        return False, {}, "Data is not a dictionary"
    
    sanitized = {}
    price_fields = ['bid', 'ask', 'quote', 'open', 'high', 'low', 'close']
    
    for key, value in data.items():
        if key in price_fields:
            if not is_valid_price(value):
                return False, {}, f"Invalid price value for {key}: {value}"
            sanitized[key] = float(value)
        else:
            sanitized[key] = value
    
    return True, sanitized, None


def validate_ohlc_integrity(open_val: float, high_val: float, low_val: float, close_val: float) -> Tuple[bool, Optional[str]]:
    """Validasi integritas nilai OHLC (Open, High, Low, Close)
    
    Validasi yang dilakukan:
    - High >= Low (high harus lebih besar atau sama dengan low)
    - High >= Open (high harus lebih besar atau sama dengan open)
    - High >= Close (high harus lebih besar atau sama dengan close)
    - Low <= Open (low harus lebih kecil atau sama dengan open)
    - Low <= Close (low harus lebih kecil atau sama dengan close)
    
    Args:
        open_val: Harga pembukaan
        high_val: Harga tertinggi
        low_val: Harga terendah
        close_val: Harga penutupan
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not all(is_valid_price(p) for p in [open_val, high_val, low_val, close_val]):
        return False, "Satu atau lebih nilai OHLC tidak valid (NaN/Inf/negatif)"
    
    if high_val < low_val:
        return False, f"High ({high_val:.5f}) lebih kecil dari Low ({low_val:.5f})"
    
    if high_val < open_val:
        return False, f"High ({high_val:.5f}) lebih kecil dari Open ({open_val:.5f})"
    
    if high_val < close_val:
        return False, f"High ({high_val:.5f}) lebih kecil dari Close ({close_val:.5f})"
    
    if low_val > open_val:
        return False, f"Low ({low_val:.5f}) lebih besar dari Open ({open_val:.5f})"
    
    if low_val > close_val:
        return False, f"Low ({low_val:.5f}) lebih besar dari Close ({close_val:.5f})"
    
    return True, None


class OHLCBuilder:
    """Builder untuk membangun candle OHLC dari tick data secara real-time.
    
    Hybrid threading/asyncio pattern:
    - threading.Lock digunakan untuk operasi sync (add_tick, get_dataframe, clear)
    - asyncio.Lock tersedia untuk operasi async jika diperlukan di masa depan
    - Ini memungkinkan OHLCBuilder dipanggil dari sync dan async context
    
    Thread-safety:
    - _tick_lock: Melindungi akses ke candle data (sync operations)
    - _callback_lock: Melindungi akses ke callback list (sync operations)
    - Kedua lock menggunakan threading.Lock karena dipanggil dari sync context
    
    Memory Optimization:
    - maxlen uses Config.MAX_CANDLE_HISTORY (default 150 in FREE_TIER_MODE)
    - lazy_load_charts parameter untuk menunda loading chart data hingga dibutuhkan
    """
    
    def __init__(self, timeframe_minutes: int = 1, maxlen: Optional[int] = None, lazy_load_charts: bool = False):
        """Initialize OHLCBuilder with configurable memory optimization.
        
        Args:
            timeframe_minutes: Timeframe dalam menit (default: 1)
            maxlen: Maximum candle history length (default: Config.MAX_CANDLE_HISTORY)
            lazy_load_charts: Jika True, chart data tidak di-load hingga diperlukan (default: False)
        """
        if timeframe_minutes <= 0:
            raise ValueError(f"Invalid timeframe_minutes: {timeframe_minutes}. Must be > 0")
        
        self.timeframe_minutes = timeframe_minutes
        self.timeframe_seconds = timeframe_minutes * 60
        self.current_candle = None
        
        candle_maxlen = maxlen if maxlen is not None else Config.MAX_CANDLE_HISTORY
        self.candles = deque(maxlen=candle_maxlen)
        self._candle_maxlen = candle_maxlen
        
        self.lazy_load_charts = lazy_load_charts
        self._chart_data_loaded = not lazy_load_charts
        
        self.tick_count = 0
        self.nan_scrub_count = 0
        self.invalid_ohlc_count = 0
        self.last_completed_candle_timestamp: Optional[datetime] = None
        self.candle_close_callbacks: List[Callable] = []
        self.last_update_time: Optional[float] = None
        
        self._tick_data: deque = deque(maxlen=3600)
        self._last_gc_time: float = time.time()
        
        self._tick_lock = threading.Lock()
        self._callback_lock = threading.Lock()
        
        self._async_lock: Optional[asyncio.Lock] = None
        
        logger.debug(f"OHLCBuilder diinisialisasi untuk M{timeframe_minutes} dengan maxlen={candle_maxlen}, lazy_load={lazy_load_charts}")
    
    def _scrub_nan_prices(self, prices: Dict) -> Tuple[bool, Dict]:
        """Scrub NaN values from price dictionary at builder boundary.
        
        THREAD-SAFETY NOTE: This method modifies self.nan_scrub_count and MUST
        only be called while holding self._tick_lock. All current callers
        (add_tick, get_dataframe) properly acquire the lock before calling.
        
        Args:
            prices: Dictionary with open, high, low, close values
            
        Returns:
            Tuple of (is_valid, scrubbed_prices)
        """
        scrubbed = {}
        for key in ['open', 'high', 'low', 'close', 'volume']:
            value = prices.get(key)
            if value is None:
                if key == 'volume':
                    scrubbed[key] = 0
                else:
                    return False, {}
            elif isinstance(value, (int, float)):
                if math.isnan(value) or math.isinf(value):
                    self.nan_scrub_count += 1
                    logger.warning(f"NaN/Inf scrubbed from {key} in M{self.timeframe_minutes} builder (total scrubs: {self.nan_scrub_count})")
                    return False, {}
                scrubbed[key] = float(value)
            else:
                return False, {}
        
        if 'timestamp' in prices:
            scrubbed['timestamp'] = prices['timestamp']
        
        return True, scrubbed
    
    def _get_async_lock(self) -> asyncio.Lock:
        """Lazy initialization of asyncio.Lock for async operations.
        
        Ini diperlukan karena asyncio.Lock() hanya bisa dibuat dalam event loop context.
        Lock diinisialisasi saat pertama kali dibutuhkan dalam async context.
        
        Returns:
            asyncio.Lock instance untuk operasi async
        """
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        return self._async_lock
        
    def _validate_tick_data(self, bid: float, ask: float, timestamp: datetime) -> Tuple[bool, Optional[str]]:
        """Validate tick data before processing with NaN check"""
        try:
            if bid is None or ask is None:
                return False, "Bid or Ask is None"
            
            if not isinstance(bid, (int, float)) or not isinstance(ask, (int, float)):
                return False, f"Invalid bid/ask type: bid={type(bid)}, ask={type(ask)}"
            
            if math.isnan(bid) or math.isnan(ask):
                return False, f"NaN detected in bid/ask: bid={bid}, ask={ask}"
            
            if math.isinf(bid) or math.isinf(ask):
                return False, f"Inf detected in bid/ask: bid={bid}, ask={ask}"
            
            if bid <= 0 or ask <= 0:
                return False, f"Invalid bid/ask values: bid={bid}, ask={ask}"
            
            if ask < bid:
                return False, f"Ask < Bid: ask={ask}, bid={bid}"
            
            spread = ask - bid
            spread_percent = (spread / bid) * 100 if bid > 0 else 0
            if spread_percent > 50.0:
                return False, f"Spread too wide: {spread:.2f} ({spread_percent:.2f}%)"
            
            if timestamp is None:
                return False, "Timestamp is None"
            
            if not isinstance(timestamp, datetime):
                return False, f"Invalid timestamp type: {type(timestamp)}"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
        
    def add_tick(self, bid: float, ask: float, timestamp: datetime):
        """Menambahkan tick data dengan validasi, NaN scrubbing, dan thread-safety.
        
        Hybrid Pattern Note:
        - Method ini sync dan menggunakan threading.Lock (_tick_lock)
        - Dipanggil dari sync context (WebSocket handler, simulator)
        - Thread-safe untuk concurrent tick processing dari multiple sources
        
        Args:
            bid: Harga bid
            ask: Harga ask  
            timestamp: Waktu tick
        """
        is_valid, error_msg = self._validate_tick_data(bid, ask, timestamp)
        if not is_valid:
            logger.warning(f"Data tick tidak valid ditolak: {error_msg}")
            return
        
        if not is_valid_price(bid) or not is_valid_price(ask):
            logger.warning(f"Harga bid/ask tidak valid (NaN/Inf/negatif): bid={bid}, ask={ask}")
            return
        
        mid_price = (bid + ask) / 2.0
        
        if math.isnan(mid_price) or math.isinf(mid_price):
            logger.warning(f"NaN/Inf mid_price dihitung dari bid={bid}, ask={ask}")
            return
        
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=pytz.UTC)
        
        candle_start = timestamp.replace(
            second=0, 
            microsecond=0,
            minute=(timestamp.minute // self.timeframe_minutes) * self.timeframe_minutes
        )
        
        with self._tick_lock:
            try:
                if self.current_candle is None or self.current_candle['timestamp'] != candle_start:
                    if self.current_candle is not None:
                        is_valid_candle, scrubbed_candle = self._scrub_nan_prices(self.current_candle)
                        if is_valid_candle:
                            ohlc_valid, ohlc_err = validate_ohlc_integrity(
                                scrubbed_candle['open'], 
                                scrubbed_candle['high'], 
                                scrubbed_candle['low'], 
                                scrubbed_candle['close']
                            )
                            if ohlc_valid:
                                self.candles.append(scrubbed_candle.copy())
                                logger.debug(f"M{self.timeframe_minutes} candle selesai: O={scrubbed_candle['open']:.2f} H={scrubbed_candle['high']:.2f} L={scrubbed_candle['low']:.2f} C={scrubbed_candle['close']:.2f} V={scrubbed_candle['volume']}")
                                
                                self.last_completed_candle_timestamp = scrubbed_candle['timestamp']
                                logger.info(f"🕯️ Candle M{self.timeframe_minutes} ditutup pada {self.last_completed_candle_timestamp}")
                                
                                callbacks_to_call = []
                                with self._callback_lock:
                                    callbacks_to_call = list(self.candle_close_callbacks)
                                
                                for callback in callbacks_to_call:
                                    try:
                                        callback(scrubbed_candle.copy(), self.timeframe_minutes)
                                    except Exception as cb_error:
                                        logger.error(f"Error saat memanggil callback candle close: {cb_error}")
                            else:
                                self.invalid_ohlc_count += 1
                                logger.warning(f"Membuang candle M{self.timeframe_minutes} tidak valid: {ohlc_err} (total invalid: {self.invalid_ohlc_count})")
                        else:
                            logger.warning(f"Membuang candle M{self.timeframe_minutes} tidak valid karena nilai NaN")
                    
                    self.current_candle = {
                        'timestamp': candle_start,
                        'open': mid_price,
                        'high': mid_price,
                        'low': mid_price,
                        'close': mid_price,
                        'volume': 0
                    }
                    self.tick_count = 0
                
                self.current_candle['high'] = max(self.current_candle['high'], mid_price)
                self.current_candle['low'] = min(self.current_candle['low'], mid_price)
                self.current_candle['close'] = mid_price
                self.tick_count += 1
                self.current_candle['volume'] = self.tick_count
                self.last_update_time = time.time()
                
            except Exception as e:
                logger.error(f"Error menambahkan tick ke M{self.timeframe_minutes} builder: {e}")
                logger.debug(f"Tick data: bid={bid}, ask={ask}, timestamp={timestamp}")
        
    def get_dataframe(self, limit: int = 100) -> Optional[pd.DataFrame]:
        """Mendapatkan DataFrame dengan validasi, NaN filtering, OHLC integrity check.
        
        THREAD-SAFETY PATTERN:
        - Uses self._tick_lock to protect access to self.candles and self.current_candle
        - _scrub_nan_prices() is called WITHIN lock context (required for nan_scrub_count)
        - Data is copied to local 'all_candles' list while holding lock
        - Lock is released before expensive DataFrame operations (uses local copy only)
        - This pattern ensures thread-safety while minimizing lock contention
        
        Hybrid Pattern Note:
        - Method ini sync dan menggunakan threading.Lock (_tick_lock)
        - Untuk async context, gunakan get_dataframe_async() sebagai gantinya
        - Thread-safe untuk akses concurrent ke candle data
        
        Args:
            limit: Jumlah maksimum candle yang dikembalikan (default: 100)
            
        Returns:
            DataFrame dengan data OHLC atau None jika tidak ada data
        """
        try:
            if limit <= 0:
                logger.warning(f"Limit tidak valid: {limit}. Menggunakan default 100")
                limit = 100
            
            with self._tick_lock:
                all_candles = []
                ohlc_invalid_count = 0
                
                for candle in self.candles:
                    is_valid, scrubbed = self._scrub_nan_prices(candle)
                    if is_valid:
                        ohlc_valid, _ = validate_ohlc_integrity(
                            scrubbed['open'], scrubbed['high'], 
                            scrubbed['low'], scrubbed['close']
                        )
                        if ohlc_valid:
                            all_candles.append(scrubbed)
                        else:
                            ohlc_invalid_count += 1
                
                if self.current_candle:
                    is_valid, scrubbed = self._scrub_nan_prices(self.current_candle)
                    if is_valid:
                        ohlc_valid, _ = validate_ohlc_integrity(
                            scrubbed['open'], scrubbed['high'], 
                            scrubbed['low'], scrubbed['close']
                        )
                        if ohlc_valid:
                            all_candles.append(scrubbed)
                        else:
                            ohlc_invalid_count += 1
                
                if ohlc_invalid_count > 0:
                    logger.warning(f"Melewati {ohlc_invalid_count} candle dengan integritas OHLC tidak valid untuk M{self.timeframe_minutes}")
            
            if len(all_candles) == 0:
                logger.debug(f"Tidak ada candle valid tersedia untuk M{self.timeframe_minutes}")
                return None
            
            df = pd.DataFrame(all_candles)
            
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Kolom yang diperlukan tidak ditemukan dalam data candle. Ada: {df.columns.tolist()}")
                return None
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            nan_mask = df[['open', 'high', 'low', 'close']].isna().any(axis=1)
            nan_count = nan_mask.sum() if isinstance(nan_mask, pd.Series) else 0
            nan_rows = int(nan_count) if nan_count else 0
            if nan_rows > 0:
                logger.warning(f"Menghapus {nan_rows} baris dengan nilai NaN dari DataFrame M{self.timeframe_minutes}")
                df = df.dropna(subset=['open', 'high', 'low', 'close'])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.index = pd.DatetimeIndex(df.index)
            
            if len(df) > limit:
                df = df.tail(limit)
            
            return df
            
        except Exception as e:
            logger.error(f"Error membuat DataFrame untuk M{self.timeframe_minutes}: {e}")
            return None
    
    async def get_dataframe_async(self, limit: int = 100) -> Optional[pd.DataFrame]:
        """Async version of get_dataframe untuk digunakan dalam async context.
        
        Menggunakan asyncio.Lock untuk proteksi async-safe saat mengakses candle data.
        Operasi berat dilakukan di thread pool untuk menghindari blocking event loop.
        
        Args:
            limit: Jumlah maksimum candle yang dikembalikan (default: 100)
            
        Returns:
            DataFrame dengan data OHLC atau None jika tidak ada data
        """
        async with self._get_async_lock():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.get_dataframe, limit)
    
    def clear(self):
        """Membersihkan semua candle dan reset state builder (untuk reload dari DB dengan aman)
        
        Menggunakan _tick_lock untuk proteksi thread-safety.
        """
        with self._tick_lock:
            self.candles.clear()
            self.current_candle = None
            self.tick_count = 0
            self.last_completed_candle_timestamp = None
        logger.debug(f"OHLCBuilder M{self.timeframe_minutes} dibersihkan")
    
    def register_candle_close_callback(self, callback: Callable) -> None:
        """Daftarkan callback yang akan dipanggil saat candle selesai/ditutup
        
        Menggunakan _callback_lock untuk proteksi race condition pada subscriber operations.
        
        Args:
            callback: Fungsi callback dengan signature (candle_data: Dict, timeframe_minutes: int)
        """
        with self._callback_lock:
            if callback not in self.candle_close_callbacks:
                self.candle_close_callbacks.append(callback)
                logger.info(f"Callback candle close terdaftar untuk M{self.timeframe_minutes}")
    
    def unregister_candle_close_callback(self, callback: Callable) -> None:
        """Hapus callback dari daftar
        
        Menggunakan _callback_lock untuk proteksi race condition pada subscriber operations.
        
        Args:
            callback: Fungsi callback yang akan dihapus
        """
        with self._callback_lock:
            if callback in self.candle_close_callbacks:
                self.candle_close_callbacks.remove(callback)
                logger.info(f"Callback candle close dihapus dari M{self.timeframe_minutes}")
    
    def cleanup_old_tick_data(self, retention_hours: Optional[int] = None) -> int:
        """Bersihkan tick data yang lebih tua dari retention period.
        
        Thread-safe menggunakan _tick_lock.
        
        Args:
            retention_hours: Jam retensi tick data (default: Config.TICK_DATA_RETENTION_HOURS)
            
        Returns:
            Jumlah tick data yang dibersihkan
        """
        retention = retention_hours if retention_hours is not None else Config.TICK_DATA_RETENTION_HOURS
        cutoff_time = datetime.now(pytz.UTC) - timedelta(hours=retention)
        
        cleaned = 0
        with self._tick_lock:
            old_len = len(self._tick_data)
            while self._tick_data and len(self._tick_data) > 0:
                oldest = self._tick_data[0]
                if isinstance(oldest, dict) and 'timestamp' in oldest:
                    tick_time = oldest['timestamp']
                    if tick_time.tzinfo is None:
                        tick_time = tick_time.replace(tzinfo=pytz.UTC)
                    if tick_time < cutoff_time:
                        self._tick_data.popleft()
                        cleaned += 1
                    else:
                        break
                else:
                    break
            
            if cleaned > 0:
                logger.debug(f"M{self.timeframe_minutes}: Cleaned {cleaned} old tick data (retention: {retention}h)")
        
        return cleaned
    
    def run_gc_if_needed(self) -> bool:
        """Jalankan garbage collection jika sudah waktunya.
        
        Menjalankan GC setiap Config.GC_INTERVAL_SECONDS detik.
        
        Returns:
            True jika GC dijalankan, False jika belum waktunya
        """
        current_time = time.time()
        elapsed = current_time - self._last_gc_time
        
        if elapsed >= Config.GC_INTERVAL_SECONDS:
            self._last_gc_time = current_time
            
            self.cleanup_old_tick_data()
            
            gc.collect()
            
            logger.debug(f"M{self.timeframe_minutes}: GC completed after {elapsed:.1f}s interval")
            return True
        
        return False
    
    def get_stats(self) -> Dict:
        """Mendapatkan statistik builder termasuk NaN scrub count dan invalid OHLC count"""
        with self._tick_lock:
            with self._callback_lock:
                return {
                    'timeframe': f"M{self.timeframe_minutes}",
                    'candle_count': len(self.candles),
                    'candle_maxlen': self._candle_maxlen,
                    'has_current_candle': self.current_candle is not None,
                    'tick_count': self.tick_count,
                    'tick_data_count': len(self._tick_data),
                    'nan_scrub_count': self.nan_scrub_count,
                    'invalid_ohlc_count': self.invalid_ohlc_count,
                    'last_completed_candle_timestamp': self.last_completed_candle_timestamp.isoformat() if self.last_completed_candle_timestamp else None,
                    'registered_callbacks': len(self.candle_close_callbacks),
                    'last_update_time': self.last_update_time,
                    'lazy_load_charts': self.lazy_load_charts,
                    'chart_data_loaded': self._chart_data_loaded,
                    'last_gc_time': self._last_gc_time
                }
    
    def get_latest_complete_candle(self) -> Optional[Dict]:
        """Mengembalikan candle terakhir yang sudah COMPLETE (bukan current_candle yang masih di-build).
        
        Thread-safe menggunakan _tick_lock.
        
        Returns:
            Dict dengan data candle terakhir yang complete, atau None jika belum ada candle complete
        """
        with self._tick_lock:
            if len(self.candles) < 1:
                return None
            return dict(self.candles[-1])
    
    def detect_hang(self) -> bool:
        """Deteksi apakah builder mengalami hang (tidak ada update > 5 menit).
        
        Thread-safe menggunakan _tick_lock.
        
        Returns:
            True jika tidak ada update selama lebih dari 300 detik (5 menit), False jika sebaliknya
        """
        with self._tick_lock:
            if self.last_update_time is None:
                return False
            elapsed = time.time() - self.last_update_time
            return elapsed > 300
    
    def recover_from_hang(self) -> None:
        """Recovery dari kondisi hang dengan membersihkan current_candle dan reset last_update_time.
        
        Thread-safe menggunakan _tick_lock.
        """
        with self._tick_lock:
            self.current_candle = None
            self.last_update_time = None
            logger.info(f"OHLCBuilder M{self.timeframe_minutes} recovered from hang - current_candle cleared, last_update_time reset")


class SubscriberHealthMetrics:
    """Track individual subscriber health metrics for monitoring and eviction decisions"""
    
    def __init__(self, name: str):
        self.name = name
        self.created_at: float = time.time()
        self.last_success_time: float = time.time()
        self.last_activity_time: float = time.time()
        self.total_messages_sent: int = 0
        self.total_messages_dropped: int = 0
        self.consecutive_failures: int = 0
        self.last_failure_time: Optional[float] = None
        self.drop_rate_window: deque = deque(maxlen=100)
    
    def record_success(self):
        """Record a successful message delivery"""
        current_time = time.time()
        self.last_success_time = current_time
        self.last_activity_time = current_time
        self.total_messages_sent += 1
        self.consecutive_failures = 0
        self.drop_rate_window.append(True)
    
    def record_drop(self):
        """Record a dropped message"""
        current_time = time.time()
        self.last_activity_time = current_time
        self.last_failure_time = current_time
        self.total_messages_dropped += 1
        self.consecutive_failures += 1
        self.drop_rate_window.append(False)
    
    def get_drop_rate(self) -> float:
        """Calculate drop rate from recent window (0.0 to 1.0)"""
        if not self.drop_rate_window:
            return 0.0
        drops = sum(1 for success in self.drop_rate_window if not success)
        return drops / len(self.drop_rate_window)
    
    def get_inactive_seconds(self) -> float:
        """Get seconds since last successful delivery"""
        return time.time() - self.last_success_time
    
    def get_zombie_seconds(self) -> float:
        """Get seconds since last activity (success or failure)"""
        return time.time() - self.last_activity_time
    
    def is_high_drop_rate(self, threshold: float = 0.3) -> bool:
        """Check if drop rate exceeds threshold"""
        if len(self.drop_rate_window) < 10:
            return False
        return self.get_drop_rate() > threshold
    
    def get_stats(self) -> Dict:
        """Get subscriber statistics"""
        return {
            'name': self.name,
            'uptime_seconds': round(time.time() - self.created_at, 1),
            'messages_sent': self.total_messages_sent,
            'messages_dropped': self.total_messages_dropped,
            'drop_rate': round(self.get_drop_rate() * 100, 2),
            'consecutive_failures': self.consecutive_failures,
            'inactive_seconds': round(self.get_inactive_seconds(), 1),
            'zombie_seconds': round(self.get_zombie_seconds(), 1),
            'is_high_drop_rate': self.is_high_drop_rate()
        }


class ConnectionMetrics:
    """Track WebSocket connection metrics for monitoring"""
    
    def __init__(self):
        self.total_connections = 0
        self.total_disconnections = 0
        self.total_reconnect_attempts = 0
        self.successful_reconnects = 0
        self.failed_reconnects = 0
        self.last_connected_at: Optional[datetime] = None
        self.last_disconnected_at: Optional[datetime] = None
        self.connection_durations: deque = deque(maxlen=100)
        self.state_transitions: deque = deque(maxlen=50)
    
    def record_connection(self):
        """Record a successful connection"""
        self.total_connections += 1
        self.last_connected_at = datetime.now(pytz.UTC)
        self._add_transition(ConnectionState.CONNECTED)
    
    def record_disconnection(self):
        """Record a disconnection"""
        self.total_disconnections += 1
        now = datetime.now(pytz.UTC)
        self.last_disconnected_at = now
        
        if self.last_connected_at:
            duration = (now - self.last_connected_at).total_seconds()
            self.connection_durations.append(duration)
        
        self._add_transition(ConnectionState.DISCONNECTED)
    
    def record_reconnect_attempt(self, success: bool):
        """Record a reconnection attempt"""
        self.total_reconnect_attempts += 1
        if success:
            self.successful_reconnects += 1
        else:
            self.failed_reconnects += 1
    
    def _add_transition(self, state: ConnectionState):
        """Add state transition record"""
        self.state_transitions.append({
            'state': state.value,
            'timestamp': datetime.now(pytz.UTC).isoformat()
        })
    
    def get_stats(self) -> Dict:
        """Get connection metrics"""
        avg_duration = 0.0
        if self.connection_durations:
            avg_duration = sum(self.connection_durations) / len(self.connection_durations)
        
        return {
            'total_connections': self.total_connections,
            'total_disconnections': self.total_disconnections,
            'total_reconnect_attempts': self.total_reconnect_attempts,
            'successful_reconnects': self.successful_reconnects,
            'failed_reconnects': self.failed_reconnects,
            'average_connection_duration_seconds': round(avg_duration, 2),
            'last_connected_at': self.last_connected_at.isoformat() if self.last_connected_at else None,
            'last_disconnected_at': self.last_disconnected_at.isoformat() if self.last_disconnected_at else None,
            'recent_transitions': list(self.state_transitions)[-10:]
        }


class MarketDataClient:
    """Client untuk market data dengan thread-safety pada akses bid/ask.
    
    Thread-safe: protected by _market_lock
    - Property bid dan ask dilindungi oleh _market_lock untuk mencegah race condition
    - Method get_latest_candle() dan update_candle() thread-safe
    """
    SYMBOL_FALLBACKS = ["frxXAUUSD"]
    
    def __init__(self, config, db_manager=None):
        self.config = config
        self.ws_url = "wss://ws.derivws.com/websockets/v3?app_id=1089"
        self.symbol = config.DERIV_SYMBOL if hasattr(config, 'DERIV_SYMBOL') else "frxXAUUSD"
        self._current_bid = None
        self._current_ask = None
        self._current_quote = None
        self._current_timestamp = None
        self.ws = None
        self.connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.running = False
        self.use_simulator = False
        self.simulator_task = None
        self.last_ping = 0
        
        self._market_lock = threading.RLock()
        
        self._connection_state = ConnectionState.DISCONNECTED
        self._connection_state_lock = asyncio.Lock()
        self.connection_metrics = ConnectionMetrics()
        
        self.m1_builder = OHLCBuilder(timeframe_minutes=1)
        self.m5_builder = OHLCBuilder(timeframe_minutes=5)
        self.h1_builder = OHLCBuilder(timeframe_minutes=60)
        
        self.candle_lock = asyncio.Lock()
        self.db_write_lock = asyncio.Lock()
        
        self.reconnect_delay = 3
        self.max_reconnect_delay = 60
        self.base_price = 2650.0
        self.price_volatility = 2.0
        
        self.subscribers = {}
        self.subscriber_health: Dict[str, SubscriberHealthMetrics] = {}
        self.subscriber_lock = asyncio.Lock()
        self.max_consecutive_failures = 5
        self.subscriber_stale_timeout = 300
        self.subscriber_zombie_timeout = 300
        self.subscriber_cleanup_interval = 60
        self.high_drop_rate_threshold = 0.3
        self.high_drop_rate_warning_interval = 30
        self.last_drop_rate_warning: Dict[str, float] = {}
        self.tick_log_counter = 0
        
        self.simulator_price_min = 1800.0
        self.simulator_price_max = 3500.0
        self.simulator_spread_min = 0.20
        self.simulator_spread_max = 0.80
        self.simulator_last_timestamp: Optional[datetime] = None
        
        self.ws_timeout = 30
        self.fetch_timeout = 10
        self.last_data_received = None
        self.data_stale_threshold = 60
        self._http_price_cache: Optional[Tuple[datetime, float]] = None
        
        self._loading_from_db = False
        self._loaded_from_db = False
        self._shutdown_in_progress = False
        self._is_shutting_down = False
        
        self._active_ws_tasks: Dict[str, asyncio.Task] = {}
        self._ws_tasks_lock = asyncio.Lock()
        self._ws_task_cancel_timeout = 5.0
        
        self._market_closed = False
        self._market_closed_detected_at = None
        
        self._subscribed = False  # True only when subscription to symbol succeeds
        
        self._last_candle_close_time: Optional[datetime] = None
        self._candle_close_event = asyncio.Event()
        self._candle_close_callbacks: List[Callable] = []
        
        self._db_manager = db_manager
        self._pending_h1_candles: List[dict] = []
        self._pending_h1_lock = threading.Lock()
        
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=45.0,
            expected_exception=Exception,
            name="DerivWebSocket"
        )
        
        self.connection_health_score: int = 100
        self.last_heartbeat_time: Optional[float] = None
        self.reconnect_success_count: int = 0
        self.consecutive_connection_failures: int = 0
        self.max_consecutive_connection_failures: int = 20
        self.heartbeat_response_times: deque = deque(maxlen=10)
        self._pending_heartbeat_time: Optional[float] = None
        self._heartbeat_timeout_threshold: float = 60.0
        self._health_check_interval: float = 30.0
        self._extended_cooldown_duration: float = 300.0
        
        self.h1_builder.register_candle_close_callback(self._on_h1_candle_close)
        
        logger.info("MarketDataClient diinisialisasi dengan penanganan error yang ditingkatkan")
        logger.info(f"WebSocket URL: {self.ws_url}, Symbol: {self.symbol}")
        logger.info("Mekanisme Pub/Sub diinisialisasi dengan manajemen lifecycle")
        logger.info("✅ Circuit breaker diinisialisasi untuk koneksi WebSocket (threshold=3, timeout=45s)")
        logger.info("✅ State machine koneksi diinisialisasi")
        logger.info("✅ Mekanisme candle close event diinisialisasi")
        logger.info("✅ H1 candle close callback registered untuk immediate DB persistence")
        logger.info("✅ WebSocket task tracking diinisialisasi untuk graceful shutdown")
        logger.info(f"✅ Connection health monitoring diinisialisasi (max_failures={self.max_consecutive_connection_failures}, heartbeat_timeout={self._heartbeat_timeout_threshold}s)")
    
    @property
    def current_bid(self) -> Optional[float]:
        """Thread-safe: protected by _market_lock
        
        Mendapatkan harga bid saat ini dengan proteksi lock.
        """
        with self._market_lock:
            return self._current_bid
    
    @current_bid.setter
    def current_bid(self, value: Optional[float]):
        """Thread-safe setter untuk current_bid"""
        with self._market_lock:
            self._current_bid = value
    
    @property
    def current_ask(self) -> Optional[float]:
        """Thread-safe: protected by _market_lock
        
        Mendapatkan harga ask saat ini dengan proteksi lock.
        """
        with self._market_lock:
            return self._current_ask
    
    @current_ask.setter
    def current_ask(self, value: Optional[float]):
        """Thread-safe setter untuk current_ask"""
        with self._market_lock:
            self._current_ask = value
    
    @property
    def current_quote(self) -> Optional[float]:
        """Thread-safe: protected by _market_lock
        
        Mendapatkan quote saat ini dengan proteksi lock.
        """
        with self._market_lock:
            return self._current_quote
    
    @current_quote.setter
    def current_quote(self, value: Optional[float]):
        """Thread-safe setter untuk current_quote"""
        with self._market_lock:
            self._current_quote = value
    
    @property
    def current_timestamp(self) -> Optional[datetime]:
        """Thread-safe: protected by _market_lock
        
        Mendapatkan timestamp data saat ini dengan proteksi lock.
        """
        with self._market_lock:
            return self._current_timestamp
    
    @current_timestamp.setter
    def current_timestamp(self, value: Optional[datetime]):
        """Thread-safe setter untuk current_timestamp"""
        with self._market_lock:
            self._current_timestamp = value
    
    def get_latest_candle(self, timeframe: str = 'M1') -> Optional[Dict]:
        """Thread-safe: protected by _market_lock
        
        Mendapatkan candle terakhir untuk timeframe tertentu.
        
        Args:
            timeframe: Timeframe yang diinginkan ('M1', 'M5', 'H1')
            
        Returns:
            Dict candle data atau None jika tidak ada data
        """
        with self._market_lock:
            if timeframe == 'M1':
                return self.m1_builder.get_latest_complete_candle()
            elif timeframe == 'M5':
                return self.m5_builder.get_latest_complete_candle()
            elif timeframe == 'H1':
                return self.h1_builder.get_latest_complete_candle()
            else:
                logger.warning(f"Timeframe tidak dikenal: {timeframe}")
                return None
    
    def update_market_data(self, bid: float, ask: float, quote: float, timestamp: datetime):
        """Thread-safe: protected by _market_lock
        
        Update semua data market dalam satu operasi atomic.
        
        Args:
            bid: Harga bid baru
            ask: Harga ask baru
            quote: Quote baru
            timestamp: Timestamp data baru
        """
        with self._market_lock:
            self._current_bid = bid
            self._current_ask = ask
            self._current_quote = quote
            self._current_timestamp = timestamp
    
    @property
    def is_ready(self) -> bool:
        """Check if market data is ready for signal generation"""
        min_m1 = 30
        min_m5 = 10
        return len(self.m1_builder.candles) >= min_m1 or len(self.m5_builder.candles) >= min_m5

    @property
    def market_status(self) -> str:
        """Return human-readable market status"""
        if self._market_closed:
            return "TUTUP"
        elif self._subscribed and self.is_ready:
            return "AKTIF"
        elif self._subscribed:
            return "MENUNGGU DATA"
        elif self.connected:
            return "TERHUBUNG (BELUM SUBSCRIBE)"
        else:
            return "TIDAK TERHUBUNG"
    
    @property
    def is_subscribed(self) -> bool:
        """Check if successfully subscribed to market data feed"""
        return self._subscribed

    def get_candle_counts(self) -> dict:
        """Return current candle counts for all timeframes"""
        return {
            'M1': len(self.m1_builder.candles),
            'M5': len(self.m5_builder.candles),
            'H1': len(self.h1_builder.candles)
        }
    
    async def fetch_price_via_http(self) -> Optional[float]:
        """Fetch current price via HTTP REST API as fallback when WebSocket fails"""
        try:
            now = datetime.now()
            if self._http_price_cache:
                cached_time, cached_price = self._http_price_cache
                if (now - cached_time).total_seconds() < 10:
                    return cached_price
            
            async with aiohttp.ClientSession() as session:
                url = f"https://api.deriv.com/api/1/ticks?ticks={self.symbol}"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'tick' in data:
                            tick = data['tick']
                            price = float(tick.get('quote') or tick.get('bid'))
                            if price and price > 0:
                                self._http_price_cache = (now, price)
                                logger.info(f"📡 HTTP fallback price: ${price:.2f}")
                                return price
            return None
        except Exception as e:
            logger.debug(f"HTTP price fetch failed: {e}")
            return None
    
    async def _set_connection_state(self, new_state: ConnectionState):
        """Thread-safe state transition with logging"""
        async with self._connection_state_lock:
            old_state = self._connection_state
            self._connection_state = new_state
            self.connection_metrics._add_transition(new_state)
            logger.info(f"🔄 Connection state: {old_state.value} → {new_state.value}")
    
    def get_connection_state(self) -> ConnectionState:
        """Get current connection state"""
        return self._connection_state
    
    def _on_h1_candle_close(self, candle: dict, timeframe_minutes: int):
        """Callback ketika H1 candle ditutup - trigger immediate save ke DB
        
        Dipanggil oleh h1_builder ketika candle H1 selesai/ditutup.
        Ini akan menjadwalkan async task untuk menyimpan candle ke database
        untuk mencegah data loss jika bot restart.
        
        Jika _db_manager belum di-set, candle akan di-queue untuk save later.
        Jika db_manager tersedia setelah queue, akan langsung coba flush.
        
        Args:
            candle: Data candle yang baru saja ditutup
            timeframe_minutes: Timeframe dalam menit (harus 60 untuk H1)
        """
        if timeframe_minutes != 60:
            return
        
        logger.info(f"🔔 H1 candle closed at {candle['timestamp']} - scheduling immediate DB save")
        
        if self._db_manager is not None:
            try:
                asyncio.create_task(self._immediate_h1_save(candle))
            except RuntimeError as e:
                logger.warning(f"Cannot schedule H1 save (no event loop): {e}")
                with self._pending_h1_lock:
                    self._pending_h1_candles.append(candle.copy())
                    logger.info(f"📦 H1 candle queued for later save (no event loop): {candle['timestamp']}")
        else:
            with self._pending_h1_lock:
                self._pending_h1_candles.append(candle.copy())
                logger.info(f"📋 H1 candle queued for later save (db_manager not ready): {candle['timestamp']}")
            
            if self._db_manager is not None:
                try:
                    asyncio.create_task(self._flush_pending_h1_candles())
                except RuntimeError:
                    pass
    
    def _sync_h1_save(self, candle: dict):
        """Synchronous H1 candle save - dipanggil dari thread pool
        
        Method ini thread-safe dan tidak akan block event loop karena
        dipanggil via asyncio.to_thread().
        
        PENTING: Menggunakan sessionmaker langsung untuk create fresh session,
        bukan get_session() yang returns scoped session (shared across threads).
        Ini memastikan session isolation di thread pool.
        
        Args:
            candle: Data candle H1 yang akan disimpan
        """
        session = None
        try:
            from bot.database import CandleData
            from sqlalchemy.orm import sessionmaker
            
            if self._db_manager is None or self._db_manager.engine is None:
                logger.warning("Cannot save H1 candle - db_manager or engine not available")
                return
            
            Session = sessionmaker(bind=self._db_manager.engine)
            session = Session()
            
            try:
                existing = session.query(CandleData).filter(
                    CandleData.timeframe == 'H1',
                    CandleData.timestamp == candle['timestamp']
                ).first()
                
                if existing:
                    existing.open = float(candle['open'])  # type: ignore[assignment]
                    existing.high = float(candle['high'])  # type: ignore[assignment]
                    existing.low = float(candle['low'])  # type: ignore[assignment]
                    existing.close = float(candle['close'])  # type: ignore[assignment]
                    existing.volume = float(candle.get('volume', 0))  # type: ignore[assignment]
                    existing.is_partial = False  # type: ignore[assignment]
                else:
                    candle_data = CandleData(
                        timeframe='H1',
                        timestamp=candle['timestamp'],
                        open=float(candle['open']),
                        high=float(candle['high']),
                        low=float(candle['low']),
                        close=float(candle['close']),
                        volume=float(candle.get('volume', 0)),
                        is_partial=False
                    )
                    session.add(candle_data)
                
                session.commit()
                logger.info(f"✅ H1 candle saved to DB: {candle['timestamp']}")
            except Exception as e:
                if session:
                    session.rollback()
                logger.error(f"Error in H1 save commit: {e}")
                raise
        except Exception as e:
            logger.error(f"Error in sync H1 save: {e}")
        finally:
            if session:
                session.close()
                logger.debug("H1 save session closed")
    
    async def _immediate_h1_save(self, candle: dict):
        """Immediately save H1 candle ke database untuk mencegah data loss
        
        Menggunakan asyncio.to_thread() untuk wrap sync database call
        agar tidak block event loop.
        
        Args:
            candle: Data candle H1 yang akan disimpan
        """
        try:
            async with self.db_write_lock:
                await asyncio.to_thread(self._sync_h1_save, candle)
        except Exception as e:
            logger.error(f"Error in immediate H1 save: {e}")
    
    async def _flush_pending_h1_candles(self):
        """Flush pending H1 candles yang di-queue saat _db_manager belum di-set
        
        Method ini dipanggil setelah _db_manager di-set untuk menyimpan
        candle yang tertunda ke database.
        """
        with self._pending_h1_lock:
            pending_count = len(self._pending_h1_candles)
            if pending_count == 0:
                return
            
            candles_to_save = list(self._pending_h1_candles)
            self._pending_h1_candles.clear()
        
        logger.info(f"📦 Flushing {pending_count} pending H1 candles to database...")
        
        saved_count = 0
        for candle in candles_to_save:
            try:
                await self._immediate_h1_save(candle)
                saved_count += 1
            except Exception as e:
                logger.error(f"Error saving pending H1 candle {candle.get('timestamp')}: {e}")
        
        logger.info(f"✅ Flushed {saved_count}/{pending_count} pending H1 candles to database")
    
    async def flush_pending_if_ready(self):
        """Flush pending H1 candles jika db_manager sudah ready
        
        Method ini bisa dipanggil dari mana saja untuk flush pending candles
        tanpa harus menunggu load_candles_from_db.
        """
        if self._db_manager is not None:
            await self._flush_pending_h1_candles()
    
    def set_db_manager(self, db_manager):
        """Set db_manager reference dan flush pending H1 candles
        
        Method ini digunakan untuk set db_manager setelah initialization
        jika tidak diberikan saat __init__.
        
        Args:
            db_manager: DatabaseManager instance
        """
        self._db_manager = db_manager
        logger.info("✅ db_manager reference set for immediate H1 candle saves")
        
        with self._pending_h1_lock:
            pending_count = len(self._pending_h1_candles)
        
        if pending_count > 0:
            logger.info(f"📦 {pending_count} pending H1 candles will be flushed")
            try:
                asyncio.create_task(self._flush_pending_h1_candles())
            except RuntimeError:
                logger.warning("Cannot schedule pending H1 flush (no event loop) - will be flushed later")
    
    def register_candle_close_callback(self, callback: Callable) -> None:
        """Daftarkan callback yang akan dipanggil saat candle selesai/ditutup
        
        Callback akan menerima parameter: (candle_data: Dict, timeframe_minutes: int)
        Callback ini didaftarkan ke semua OHLC builders (M1 dan M5)
        
        Args:
            callback: Fungsi callback dengan signature (candle_data: Dict, timeframe_minutes: int)
        """
        if callback not in self._candle_close_callbacks:
            self._candle_close_callbacks.append(callback)
            self.m1_builder.register_candle_close_callback(callback)
            self.m5_builder.register_candle_close_callback(callback)
            logger.info("Callback candle close terdaftar di MarketDataClient")
    
    def unregister_candle_close_callback(self, callback: Callable) -> None:
        """Hapus callback dari daftar
        
        Args:
            callback: Fungsi callback yang akan dihapus
        """
        if callback in self._candle_close_callbacks:
            self._candle_close_callbacks.remove(callback)
            self.m1_builder.unregister_candle_close_callback(callback)
            self.m5_builder.unregister_candle_close_callback(callback)
            logger.info("Callback candle close dihapus dari MarketDataClient")
    
    def get_last_candle_close_time(self) -> Optional[datetime]:
        """Dapatkan timestamp candle close terakhir dari builder M1
        
        Returns:
            Timestamp candle M1 yang terakhir ditutup, atau None jika belum ada
        """
        return self.m1_builder.last_completed_candle_timestamp
    
    def get_last_candle_close_time_m5(self) -> Optional[datetime]:
        """Dapatkan timestamp candle close terakhir dari builder M5
        
        Returns:
            Timestamp candle M5 yang terakhir ditutup, atau None jika belum ada
        """
        return self.m5_builder.last_completed_candle_timestamp
    
    @property
    def candle_just_closed(self) -> bool:
        """Check apakah baru saja ada candle close dalam 2 detik terakhir
        
        Returns:
            True jika ada candle M1 yang ditutup dalam 2 detik terakhir
        """
        last_close = self.m1_builder.last_completed_candle_timestamp
        if last_close is None:
            return False
        
        now = datetime.now(pytz.UTC)
        if last_close.tzinfo is None:
            last_close = last_close.replace(tzinfo=pytz.UTC)
        
        time_since_close = (now - last_close).total_seconds()
        return time_since_close <= 2.0
    
    @property
    def candle_just_closed_m5(self) -> bool:
        """Check apakah baru saja ada candle M5 close dalam 5 detik terakhir
        
        Returns:
            True jika ada candle M5 yang ditutup dalam 5 detik terakhir
        """
        last_close = self.m5_builder.last_completed_candle_timestamp
        if last_close is None:
            return False
        
        now = datetime.now(pytz.UTC)
        if last_close.tzinfo is None:
            last_close = last_close.replace(tzinfo=pytz.UTC)
        
        time_since_close = (now - last_close).total_seconds()
        return time_since_close <= 5.0
    
    def _log_tick_sample(self, bid: float, ask: float, quote: float, spread: Optional[float] = None, mode: str = "") -> None:
        """Centralized tick logging dengan sampling - increment counter HANYA 1x per tick"""
        self.tick_log_counter += 1
        if self.tick_log_counter % self.config.TICK_LOG_SAMPLE_RATE == 0:
            if mode == "simulator":
                logger.info(f"💰 Simulator Tick Sample (setiap {self.config.TICK_LOG_SAMPLE_RATE}): Bid=${bid:.2f}, Ask=${ask:.2f}, Spread=${spread:.2f}")
            else:
                logger.info(f"💰 Tick Sample (setiap {self.config.TICK_LOG_SAMPLE_RATE}): Bid={bid:.2f}, Ask={ask:.2f}, Quote={quote:.2f}")
        else:
            if mode == "simulator":
                logger.debug(f"Simulator: Bid=${bid:.2f}, Ask=${ask:.2f}, Spread=${spread:.2f}")
            else:
                logger.debug(f"💰 Tick: Bid={bid:.2f}, Ask={ask:.2f}, Quote={quote:.2f}")
    
    async def subscribe_ticks(self, name: str) -> asyncio.Queue:
        """Subscribe to tick feed with proper lifecycle tracking and health metrics"""
        async with self.subscriber_lock:
            if self._shutdown_in_progress:
                raise RuntimeError("Cannot subscribe during shutdown")
            
            queue = asyncio.Queue(maxsize=500)
            self.subscribers[name] = queue
            self.subscriber_health[name] = SubscriberHealthMetrics(name)
            logger.info(f"✅ Subscriber '{name}' registered with health metrics tracking")
            return queue
    
    async def unsubscribe_ticks(self, name: str):
        """Unsubscribe from tick feed with proper cleanup and metrics logging"""
        async with self.subscriber_lock:
            health = self.subscriber_health.get(name)
            if health:
                stats = health.get_stats()
                logger.info(
                    f"📊 Subscriber '{name}' final stats: "
                    f"sent={stats['messages_sent']}, dropped={stats['messages_dropped']}, "
                    f"drop_rate={stats['drop_rate']}%, uptime={stats['uptime_seconds']}s"
                )
            
            if name in self.subscribers:
                try:
                    queue = self.subscribers[name]
                    while not queue.empty():
                        try:
                            queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                except Exception as e:
                    logger.debug(f"Error draining queue for '{name}': {e}")
                
                del self.subscribers[name]
            
            if name in self.subscriber_health:
                del self.subscriber_health[name]
            if name in self.last_drop_rate_warning:
                del self.last_drop_rate_warning[name]
            
            logger.debug(f"Subscriber '{name}' unregistered dari tick feed")
    
    async def cleanup_stale_subscribers(self) -> List[str]:
        """Clean up stale and zombie subscribers with detailed metrics
        
        Eviction criteria:
        1. Stale: No successful delivery for subscriber_stale_timeout seconds
        2. Zombie: No activity (success or failure) for subscriber_zombie_timeout seconds
        3. High failures: max_consecutive_failures reached
        4. High drop rate: Sustained high drop rate (logged as warning, evicted if combined with other issues)
        
        Returns:
            List of removed subscriber names with eviction reasons
        """
        removed = []
        eviction_reasons = {}
        current_time = time.time()
        
        async with self.subscriber_lock:
            stale_names = []
            
            for name in list(self.subscribers.keys()):
                health = self.subscriber_health.get(name)
                if not health:
                    stale_names.append(name)
                    eviction_reasons[name] = "missing health metrics"
                    logger.warning(f"⚠️ Subscriber '{name}' has no health metrics - marking for eviction")
                    continue
                
                inactive_time = health.get_inactive_seconds()
                zombie_time = health.get_zombie_seconds()
                failures = health.consecutive_failures
                drop_rate = health.get_drop_rate()
                is_high_drop = health.is_high_drop_rate(self.high_drop_rate_threshold)
                
                is_stale = False
                reasons = []
                
                if zombie_time > self.subscriber_zombie_timeout:
                    is_stale = True
                    reasons.append(f"zombie for {zombie_time:.0f}s (no activity)")
                    logger.warning(
                        f"🧟 ZOMBIE subscriber '{name}': no activity for {zombie_time:.0f}s "
                        f"(threshold: {self.subscriber_zombie_timeout}s)"
                    )
                
                if inactive_time > self.subscriber_stale_timeout:
                    is_stale = True
                    reasons.append(f"stale for {inactive_time:.0f}s (no success)")
                
                if failures >= self.max_consecutive_failures:
                    is_stale = True
                    reasons.append(f"{failures} consecutive failures")
                
                if is_high_drop and (inactive_time > 60 or failures >= 3):
                    is_stale = True
                    reasons.append(f"high drop rate {drop_rate*100:.1f}% with issues")
                
                if is_high_drop and not is_stale:
                    last_warning = self.last_drop_rate_warning.get(name, 0)
                    if current_time - last_warning > self.high_drop_rate_warning_interval:
                        logger.warning(
                            f"⚠️ HIGH DROP RATE for '{name}': {drop_rate*100:.1f}% "
                            f"(dropped {health.total_messages_dropped}/{health.total_messages_sent + health.total_messages_dropped} messages)"
                        )
                        self.last_drop_rate_warning[name] = current_time
                
                if is_stale:
                    stale_names.append(name)
                    eviction_reasons[name] = "; ".join(reasons)
            
            for name in stale_names:
                health = self.subscriber_health.get(name)
                reason = eviction_reasons.get(name, "unknown")
                
                if health:
                    stats = health.get_stats()
                    logger.warning(
                        f"🚫 EVICTING subscriber '{name}': {reason} | "
                        f"Stats: sent={stats['messages_sent']}, dropped={stats['messages_dropped']}, "
                        f"drop_rate={stats['drop_rate']}%, failures={stats['consecutive_failures']}"
                    )
                else:
                    logger.warning(f"🚫 EVICTING subscriber '{name}': {reason}")
                
                try:
                    queue = self.subscribers.get(name)
                    if queue:
                        drained = 0
                        while not queue.empty():
                            try:
                                queue.get_nowait()
                                drained += 1
                            except asyncio.QueueEmpty:
                                break
                        if drained > 0:
                            logger.debug(f"Drained {drained} pending messages from '{name}' queue")
                except Exception as e:
                    logger.debug(f"Error draining stale queue '{name}': {e}")
                
                if name in self.subscribers:
                    del self.subscribers[name]
                if name in self.subscriber_health:
                    del self.subscriber_health[name]
                if name in self.last_drop_rate_warning:
                    del self.last_drop_rate_warning[name]
                
                removed.append(name)
        
        if removed:
            logger.info(f"✅ Cleanup completed: evicted {len(removed)} subscribers")
            for name in removed:
                logger.info(f"   - {name}: {eviction_reasons.get(name, 'unknown')}")
        
        return removed
    
    async def get_subscriber_health_report(self) -> Dict:
        """Get health report for all subscribers
        
        Returns:
            Dictionary with subscriber health statistics
        """
        async with self.subscriber_lock:
            report = {
                'total_subscribers': len(self.subscribers),
                'healthy': 0,
                'warning': 0,
                'critical': 0,
                'subscribers': {}
            }
            
            for name, health in self.subscriber_health.items():
                stats = health.get_stats()
                
                status = 'healthy'
                if health.consecutive_failures >= self.max_consecutive_failures:
                    status = 'critical'
                elif health.is_high_drop_rate(self.high_drop_rate_threshold):
                    status = 'warning'
                elif health.get_inactive_seconds() > 60:
                    status = 'warning'
                
                stats['status'] = status
                report['subscribers'][name] = stats
                
                if status == 'healthy':
                    report['healthy'] += 1
                elif status == 'warning':
                    report['warning'] += 1
                else:
                    report['critical'] += 1
            
            return report
    
    async def _unsubscribe_all(self):
        """Unsubscribe all subscribers during shutdown with final stats"""
        async with self.subscriber_lock:
            subscriber_names = list(self.subscribers.keys())
            
            for name in subscriber_names:
                health = self.subscriber_health.get(name)
                if health:
                    stats = health.get_stats()
                    logger.info(
                        f"📊 Shutdown stats for '{name}': "
                        f"sent={stats['messages_sent']}, dropped={stats['messages_dropped']}, "
                        f"drop_rate={stats['drop_rate']}%"
                    )
                
                try:
                    queue = self.subscribers.get(name)
                    if queue:
                        while not queue.empty():
                            try:
                                queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                except Exception as e:
                    logger.debug(f"Error draining queue '{name}' during shutdown: {e}")
            
            self.subscribers.clear()
            self.subscriber_health.clear()
            self.last_drop_rate_warning.clear()
            
            if subscriber_names:
                logger.info(f"Unsubscribed all {len(subscriber_names)} subscribers during shutdown")
    
    async def _register_ws_task(self, name: str, task: asyncio.Task) -> None:
        """Register an active WebSocket task for tracking
        
        Args:
            name: Task identifier (e.g., 'heartbeat', 'data_monitor', 'stale_cleanup', 'builder_health')
            task: The asyncio.Task to track
        """
        async with self._ws_tasks_lock:
            if name in self._active_ws_tasks:
                old_task = self._active_ws_tasks[name]
                if not old_task.done():
                    logger.warning(f"Replacing existing task '{name}' - cancelling old task")
                    old_task.cancel()
                    try:
                        await asyncio.wait_for(old_task, timeout=2.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass
            
            self._active_ws_tasks[name] = task
            logger.debug(f"✅ WebSocket task '{name}' registered for tracking")
    
    async def _unregister_ws_task(self, name: str) -> None:
        """Unregister a WebSocket task after it's completed or cancelled
        
        Args:
            name: Task identifier to unregister
        """
        async with self._ws_tasks_lock:
            if name in self._active_ws_tasks:
                del self._active_ws_tasks[name]
                logger.debug(f"WebSocket task '{name}' unregistered")
    
    async def _cleanup_old_ws_tasks(self) -> int:
        """Cleanup old/zombie WebSocket tasks during reconnection
        
        Called before starting new connection to ensure no zombie tasks remain.
        
        Returns:
            Number of tasks cleaned up
        """
        cleaned = 0
        async with self._ws_tasks_lock:
            tasks_to_cleanup = list(self._active_ws_tasks.items())
        
        if not tasks_to_cleanup:
            return 0
        
        logger.info(f"🧹 Cleaning up {len(tasks_to_cleanup)} old WebSocket tasks before reconnection...")
        
        for name, task in tasks_to_cleanup:
            try:
                if task.done():
                    await self._unregister_ws_task(name)
                    cleaned += 1
                    logger.debug(f"Removed already-done task '{name}'")
                    continue
                
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=self._ws_task_cancel_timeout)
                    logger.debug(f"Task '{name}' cancelled successfully")
                except asyncio.TimeoutError:
                    logger.warning(f"Task '{name}' did not cancel within {self._ws_task_cancel_timeout}s")
                except asyncio.CancelledError:
                    pass
                
                await self._unregister_ws_task(name)
                cleaned += 1
                
            except Exception as e:
                logger.error(f"Error cleaning up task '{name}': {e}")
        
        if cleaned > 0:
            logger.info(f"✅ Cleaned up {cleaned} old WebSocket tasks")
        
        return cleaned
    
    async def _shutdown_websocket_tasks(self, timeout_per_task: float = 5.0) -> Dict[str, str]:
        """Gracefully shutdown all active WebSocket tasks with proper ordering
        
        Shutdown order (receiver first, then sender):
        1. data_monitor - Stop monitoring first
        2. stale_cleanup - Stop cleanup
        3. builder_health - Stop health check
        4. heartbeat - Stop heartbeat last (may need to send final ping)
        
        Args:
            timeout_per_task: Timeout in seconds for each task cancellation
            
        Returns:
            Dict mapping task name to final status ('completed', 'cancelled', 'timeout', 'error')
        """
        results: Dict[str, str] = {}
        
        shutdown_order = ['data_monitor', 'stale_cleanup', 'builder_health', 'gc_cleanup', 'heartbeat']
        
        async with self._ws_tasks_lock:
            tasks_snapshot = dict(self._active_ws_tasks)
        
        if not tasks_snapshot:
            logger.info("No active WebSocket tasks to shutdown")
            return results
        
        logger.info(f"🔄 Shutting down {len(tasks_snapshot)} WebSocket tasks...")
        
        ordered_tasks = []
        for name in shutdown_order:
            if name in tasks_snapshot:
                ordered_tasks.append((name, tasks_snapshot[name]))
        
        for name, task in tasks_snapshot.items():
            if name not in shutdown_order:
                ordered_tasks.append((name, task))
        
        for name, task in ordered_tasks:
            try:
                if task.done():
                    if task.cancelled():
                        results[name] = 'already_cancelled'
                    elif task.exception():
                        results[name] = f'already_failed: {task.exception()}'
                    else:
                        results[name] = 'already_completed'
                    logger.debug(f"Task '{name}' already done: {results[name]}")
                    continue
                
                logger.debug(f"Cancelling task '{name}'...")
                task.cancel()
                
                try:
                    await asyncio.wait_for(task, timeout=timeout_per_task)
                    results[name] = 'cancelled'
                    logger.info(f"✅ Task '{name}' cancelled gracefully")
                except asyncio.TimeoutError:
                    results[name] = 'timeout'
                    logger.warning(f"⚠️ Task '{name}' did not cancel within {timeout_per_task}s")
                except asyncio.CancelledError:
                    results[name] = 'cancelled'
                    logger.debug(f"Task '{name}' cancelled via CancelledError")
                    
            except Exception as e:
                results[name] = f'error: {e}'
                logger.error(f"❌ Error shutting down task '{name}': {e}")
        
        async with self._ws_tasks_lock:
            self._active_ws_tasks.clear()
        
        completed_count = sum(1 for s in results.values() if 'cancelled' in s or 'completed' in s)
        logger.info(
            f"🏁 WebSocket task shutdown complete: "
            f"{completed_count}/{len(results)} tasks cleanly terminated"
        )
        
        for name, status in results.items():
            logger.debug(f"  - {name}: {status}")
        
        return results
    
    def prepare_for_shutdown(self) -> None:
        """Prepare MarketDataClient for shutdown - call this from main.py before shutdown
        
        This method:
        1. Sets _is_shutting_down flag to disable reconnection attempts
        2. Sets _shutdown_in_progress flag
        3. Signals that no new subscriptions should be accepted
        
        After calling this, the client will:
        - Not attempt reconnection on disconnect
        - Reject new tick subscriptions
        - Allow graceful completion of current operations
        """
        logger.info("🛑 MarketDataClient preparing for shutdown...")
        self._is_shutting_down = True
        self._shutdown_in_progress = True
        
        logger.info("✅ Shutdown flags set - reconnection disabled, new subscriptions blocked")
        logger.info(f"Active WebSocket tasks: {list(self._active_ws_tasks.keys())}")
    
    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress
        
        Returns:
            True if shutdown has been initiated
        """
        return self._is_shutting_down
    
    async def _broadcast_tick(self, tick_data: Dict):
        """Broadcast tick to subscribers with NaN validation and health metrics tracking"""
        if not self.subscribers:
            return
        
        is_valid, sanitized_data, error_msg = sanitize_price_data(tick_data)
        if not is_valid:
            logger.warning(f"Tick data failed validation before broadcast: {error_msg}")
            return
        
        stale_subscribers = []
        
        async with self.subscriber_lock:
            for name, queue in list(self.subscribers.items()):
                health = self.subscriber_health.get(name)
                success = False
                max_retries = 3
                
                for attempt in range(max_retries):
                    try:
                        queue.put_nowait(sanitized_data)
                        success = True
                        if health:
                            health.record_success()
                        break
                        
                    except asyncio.QueueFull:
                        if attempt < max_retries - 1:
                            backoff_time = 0.1 * (2 ** attempt)
                            logger.debug(f"Queue full for '{name}', retry {attempt + 1}/{max_retries} after {backoff_time}s")
                            await asyncio.sleep(backoff_time)
                        else:
                            logger.debug(f"Queue full for subscriber '{name}' after {max_retries} retries, message dropped")
                            
                    except Exception as e:
                        logger.error(f"Error broadcasting tick to '{name}': {e}")
                        break
                
                if not success:
                    if health:
                        health.record_drop()
                        
                        if health.consecutive_failures >= self.max_consecutive_failures:
                            logger.warning(
                                f"⚠️ Subscriber '{name}' exceeded failure threshold: "
                                f"{health.consecutive_failures} consecutive failures, marking for removal"
                            )
                            stale_subscribers.append(name)
        
        for name in stale_subscribers:
            await self.unsubscribe_ticks(name)
    
    async def fetch_historical_candles(self, websocket, timeframe_minutes: int = 1, count: int = 100):
        """Fetch historical candles from Deriv API with timeout and validation"""
        try:
            if timeframe_minutes <= 0:
                logger.error(f"Invalid timeframe_minutes: {timeframe_minutes}")
                return False
            
            if count <= 0 or count > 5000:
                logger.warning(f"Invalid count: {count}. Using default 100")
                count = 100
            
            granularity = timeframe_minutes * 60
            
            history_request = {
                "ticks_history": self.symbol,
                "adjust_start_time": 1,
                "count": count,
                "end": "latest",
                "start": 1,
                "style": "candles",
                "granularity": granularity
            }
            
            await websocket.send(json.dumps(history_request))
            logger.debug(f"Requesting {count} historical M{timeframe_minutes} candles...")
            
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=self.fetch_timeout)
            except asyncio.TimeoutError:
                logger.error(f"Timeout fetching historical candles (timeout={self.fetch_timeout}s)")
                return False
            
            try:
                data = json.loads(response)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response: {e}")
                logger.debug(f"Response: {response[:200]}")
                return False
            
            if 'error' in data:
                logger.error(f"API error fetching candles: {data['error'].get('message', 'Unknown error')}")
                return False
            
            if 'candles' in data:
                candles = data['candles']
                if not candles or len(candles) == 0:
                    logger.warning("Received empty candles array")
                    return False
                
                logger.info(f"Received {len(candles)} historical M{timeframe_minutes} candles")
                
                if timeframe_minutes == 1:
                    builder = self.m1_builder
                elif timeframe_minutes == 5:
                    builder = self.m5_builder
                elif timeframe_minutes == 60:
                    builder = self.h1_builder
                else:
                    logger.warning(f"No builder for timeframe M{timeframe_minutes}")
                    return False
                
                valid_candles = 0
                nan_skipped = 0
                for candle in candles:
                    try:
                        if not all(k in candle for k in ['epoch', 'open', 'high', 'low', 'close']):
                            logger.warning(f"Incomplete candle data: {candle}")
                            continue
                        
                        timestamp = datetime.fromtimestamp(candle['epoch'], tz=pytz.UTC)
                        timestamp = timestamp.replace(second=0, microsecond=0)
                        
                        open_price = float(candle['open'])
                        high_price = float(candle['high'])
                        low_price = float(candle['low'])
                        close_price = float(candle['close'])
                        
                        if any(math.isnan(p) or math.isinf(p) for p in [open_price, high_price, low_price, close_price]):
                            nan_skipped += 1
                            continue
                        
                        if high_price < low_price:
                            logger.warning(f"Invalid candle: high < low ({high_price} < {low_price})")
                            continue
                        
                        if not (low_price <= open_price <= high_price and low_price <= close_price <= high_price):
                            logger.warning(f"Invalid candle: prices out of range")
                            continue
                        
                        candle_data = {
                            'timestamp': pd.Timestamp(timestamp),
                            'open': open_price,
                            'high': high_price,
                            'low': low_price,
                            'close': close_price,
                            'volume': 100
                        }
                        builder.candles.append(candle_data)
                        valid_candles += 1
                        
                    except (ValueError, KeyError, TypeError) as e:
                        logger.warning(f"Error processing candle: {e}")
                        continue
                
                if nan_skipped > 0:
                    logger.warning(f"Skipped {nan_skipped} candles with NaN/Inf values")
                
                logger.info(f"Pre-populated {valid_candles} valid M{timeframe_minutes} candles")
                return valid_candles > 0
            else:
                logger.warning(f"No 'candles' key in response: {list(data.keys())}")
                return False
                
        except asyncio.CancelledError:
            logger.info("Historical candle fetch cancelled")
            raise
        except Exception as e:
            logger.error(f"Error fetching historical candles for M{timeframe_minutes}: {e}", exc_info=True)
            return False
    
    def _calculate_backoff_with_jitter(self, attempt: int, use_extended_cooldown: bool = False) -> float:
        """Calculate exponential backoff with improved jitter for reconnection
        
        Uses multiplicative jitter (0.5-1.5x multiplier) to prevent thundering herd:
        - Jitter multiplier provides better distribution than additive jitter
        - Extended cooldown mode activates after max_consecutive_connection_failures
        
        Args:
            attempt: Current reconnection attempt number (1-based)
            use_extended_cooldown: If True, use extended cooldown duration
            
        Returns:
            Delay in seconds with jitter applied
        """
        if use_extended_cooldown:
            base_delay = self._extended_cooldown_duration
            jitter_multiplier = random.uniform(0.8, 1.2)
            final_delay = base_delay * jitter_multiplier
            logger.warning(
                f"🔄 Extended cooldown activated: base={base_delay:.0f}s, "
                f"jitter_mult={jitter_multiplier:.2f}x, final={final_delay:.1f}s"
            )
            return final_delay
        
        base_delay = self.reconnect_delay
        max_delay = self.max_reconnect_delay
        
        exponential_delay = base_delay * (2 ** (attempt - 1))
        
        jitter_multiplier = random.uniform(0.5, 1.5)
        
        delay_with_jitter = exponential_delay * jitter_multiplier
        
        final_delay = max(base_delay, min(delay_with_jitter, max_delay))
        
        logger.info(
            f"🔄 Reconnect backoff: attempt={attempt}, base={base_delay}s, "
            f"exp={exponential_delay:.1f}s, jitter_mult={jitter_multiplier:.2f}x, "
            f"final={final_delay:.1f}s, consecutive_failures={self.consecutive_connection_failures}"
        )
        
        return final_delay
        
    async def connect_websocket(self):
        """Connect to WebSocket with state machine, health monitoring, and enhanced error handling.
        
        Enhanced features:
        - Connection health scoring (0-100)
        - Automatic reset of counters after successful connection
        - Extended cooldown after max consecutive failures
        - Improved state transition logging
        """
        self.running = True
        
        while self.running:
            try:
                old_state = self._connection_state
                await self._set_connection_state(ConnectionState.CONNECTING)
                logger.info(
                    f"🔌 Connecting to Deriv WebSocket: attempt={self.reconnect_attempts + 1}, "
                    f"consecutive_failures={self.consecutive_connection_failures}, "
                    f"health_score={self.connection_health_score}, "
                    f"prev_state={old_state.value}"
                )
                
                try:
                    async with websockets.connect(
                        self.ws_url,
                        ping_interval=None,
                        close_timeout=10,
                        open_timeout=self.ws_timeout
                    ) as websocket:
                        self.ws = websocket
                        self.connected = True
                        self.reconnect_attempts = 0
                        self.last_data_received = datetime.now()
                        
                        self.reconnect_success_count += 1
                        self.consecutive_connection_failures = 0
                        self.connection_health_score = 100
                        self.heartbeat_response_times.clear()
                        self.last_heartbeat_time = None
                        self._pending_heartbeat_time = None
                        
                        await self._set_connection_state(ConnectionState.CONNECTED)
                        self.connection_metrics.record_connection()
                        logger.info(
                            f"✅ Connected to Deriv WebSocket successfully | "
                            f"total_successful_reconnects={self.reconnect_success_count}, "
                            f"health_score={self.connection_health_score}/100"
                        )
                        
                        h1_count = len(self.h1_builder.candles)
                        m1_count = len(self.m1_builder.candles)
                        m5_count = len(self.m5_builder.candles)
                        
                        need_m1_fetch = m1_count < 30
                        need_m5_fetch = m5_count < 30
                        need_h1_fetch = h1_count < 60
                        
                        if self._loaded_from_db and not need_m1_fetch and not need_m5_fetch and not need_h1_fetch:
                            logger.info("✅ Skipping historical fetch - all timeframes loaded from DB")
                            logger.info(f"Current candle counts: M1={m1_count}, M5={m5_count}, H1={h1_count}")
                        else:
                            if self._loaded_from_db:
                                missing = []
                                if need_m1_fetch: missing.append(f"M1({m1_count}<30)")
                                if need_m5_fetch: missing.append(f"M5({m5_count}<30)")
                                if need_h1_fetch: missing.append(f"H1({h1_count}<60)")
                                logger.warning(f"DB loaded but insufficient candles: {', '.join(missing)} - fetching from Deriv")
                            else:
                                logger.info("No DB load - fetching historical candles from Deriv API")
                            
                            try:
                                if need_m1_fetch or not self._loaded_from_db:
                                    m1_success = await self.fetch_historical_candles(websocket, timeframe_minutes=1, count=100)
                                else:
                                    m1_success = True
                                    
                                if need_m5_fetch or not self._loaded_from_db:
                                    m5_success = await self.fetch_historical_candles(websocket, timeframe_minutes=5, count=100)
                                else:
                                    m5_success = True
                                
                                if need_h1_fetch or not self._loaded_from_db:
                                    h1_success = await self.fetch_historical_candles(websocket, timeframe_minutes=60, count=100)
                                    
                                    if h1_success and len(self.h1_builder.candles) < 60:
                                        logger.warning(f"⚠️ H1 fetch returned only {len(self.h1_builder.candles)}/60 candles")
                                        logger.info("Retrying H1 fetch with extended count...")
                                        await self.fetch_historical_candles(websocket, timeframe_minutes=60, count=200)
                                else:
                                    h1_success = True
                                
                                if not m1_success and not m5_success:
                                    logger.warning("Failed to fetch any historical data, but continuing with live feed")
                                    
                                h1_final = len(self.h1_builder.candles)
                                if h1_final < 60:
                                    logger.warning(f"⚠️ H1 INCOMPLETE: Hanya {h1_final}/60 candles - H1 analysis akan terbatas")
                                else:
                                    logger.info(f"✅ H1 historical candles COMPLETE: {h1_final} candles")
                            except Exception as e:
                                logger.error(f"Error fetching historical data: {e}. Continuing with live feed")
                        
                        logger.info(f"Final candle counts after init: M1={len(self.m1_builder.candles)}, M5={len(self.m5_builder.candles)}, H1={len(self.h1_builder.candles)}")
                        
                        subscription_success = False
                        symbols_to_try = [self.symbol] + [s for s in self.SYMBOL_FALLBACKS if s != self.symbol]
                        
                        for symbol_candidate in symbols_to_try:
                            try:
                                subscribe_msg = {"ticks": symbol_candidate}
                                await asyncio.wait_for(
                                    websocket.send(json.dumps(subscribe_msg)),
                                    timeout=5.0
                                )
                                
                                response = await asyncio.wait_for(
                                    websocket.recv(),
                                    timeout=5.0
                                )
                                response_data = json.loads(response)
                                
                                if "error" in response_data:
                                    error_info = response_data.get("error", {})
                                    error_code = error_info.get("code", "") if isinstance(error_info, dict) else ""
                                    error_msg = error_info.get("message", "Unknown error") if isinstance(error_info, dict) else str(error_info)
                                    
                                    if error_code == "MarketIsClosed":
                                        self._market_closed = True
                                        self._market_closed_detected_at = datetime.now()
                                        logger.warning(f"🔒 Market sedang TUTUP - {error_msg}")
                                    elif "Symbol" in error_msg and "invalid" in error_msg.lower():
                                        logger.warning(f"❌ Symbol {symbol_candidate} tidak valid: {error_msg} - mencoba symbol berikutnya...")
                                    else:
                                        logger.warning(f"Symbol {symbol_candidate} subscription failed: {error_msg}")
                                    continue
                                
                                self.symbol = symbol_candidate
                                subscription_success = True
                                self._subscribed = True  # Mark as subscribed
                                logger.info(f"📡 Subscribed to {self.symbol}")
                                
                                # Reset market closed flag on successful subscription
                                self._market_closed = False
                                self._market_closed_detected_at = None
                                logger.info("✅ Market status reset - subscription successful")
                                
                                if "tick" in response_data:
                                    await self._on_message(response)
                                break
                                
                            except asyncio.TimeoutError:
                                logger.warning(f"Timeout subscribing to {symbol_candidate}, trying next symbol...")
                                continue
                            except Exception as e:
                                logger.warning(f"Error subscribing to {symbol_candidate}: {e}, trying next symbol...")
                                continue
                        
                        if not subscription_success:
                            logger.error(f"Failed to subscribe to any symbol: {symbols_to_try}")
                            raise WebSocketConnectionError("Subscribe failed for all symbols")
                        
                        heartbeat_task = asyncio.create_task(self._send_heartbeat())
                        data_monitor_task = asyncio.create_task(self._monitor_data_staleness())
                        stale_cleanup_task = asyncio.create_task(self._periodic_stale_cleanup())
                        builder_health_task = asyncio.create_task(self._monitor_builder_health())
                        gc_cleanup_task = asyncio.create_task(self._periodic_gc_cleanup())
                        
                        await self._register_ws_task('heartbeat', heartbeat_task)
                        await self._register_ws_task('data_monitor', data_monitor_task)
                        await self._register_ws_task('stale_cleanup', stale_cleanup_task)
                        await self._register_ws_task('builder_health', builder_health_task)
                        await self._register_ws_task('gc_cleanup', gc_cleanup_task)
                        
                        logger.info(f"📡 WebSocket tasks started and tracked: {list(self._active_ws_tasks.keys())}")
                        
                        try:
                            async for message in websocket:
                                if self._is_shutting_down:
                                    logger.info("🛑 Shutdown detected, stopping message processing")
                                    break
                                await self._on_message(message)
                        finally:
                            logger.info("🔄 WebSocket message loop ended, cleaning up tasks...")
                            
                            task_cleanup_results = await self._shutdown_websocket_tasks(
                                timeout_per_task=self._ws_task_cancel_timeout
                            )
                            
                            for name, status in task_cleanup_results.items():
                                logger.debug(f"Task '{name}' final status: {status}")
                                
                except asyncio.TimeoutError:
                    logger.error(f"WebSocket connection timeout ({self.ws_timeout}s)")
                    self.connected = False
                    self.connection_metrics.record_disconnection()
                    await self._handle_reconnect()
                    
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: code={e.code}, reason={e.reason}")
                self.connected = False
                self.connection_metrics.record_disconnection()
                await self._handle_reconnect()
                
            except websockets.exceptions.WebSocketException as e:
                logger.error(f"WebSocket protocol error: {e}")
                self.connected = False
                self.connection_metrics.record_disconnection()
                await self._handle_reconnect()
                
            except Exception as e:
                logger.error(f"Unexpected WebSocket error: {type(e).__name__}: {e}", exc_info=True)
                self.connected = False
                self.connection_metrics.record_disconnection()
                await self._handle_reconnect()
    
    async def _periodic_stale_cleanup(self):
        """Periodically clean up stale and zombie subscribers with configurable interval"""
        cleanup_count = 0
        try:
            logger.info(
                f"🔄 Starting periodic subscriber cleanup (interval={self.subscriber_cleanup_interval}s, "
                f"stale_timeout={self.subscriber_stale_timeout}s, zombie_timeout={self.subscriber_zombie_timeout}s)"
            )
            while self.running and self.connected:
                await asyncio.sleep(self.subscriber_cleanup_interval)
                
                try:
                    removed = await self.cleanup_stale_subscribers()
                    cleanup_count += 1
                    
                    if cleanup_count % 10 == 0:
                        health_report = await self.get_subscriber_health_report()
                        if health_report['total_subscribers'] > 0:
                            logger.info(
                                f"📊 Subscriber health report (cycle {cleanup_count}): "
                                f"total={health_report['total_subscribers']}, "
                                f"healthy={health_report['healthy']}, "
                                f"warning={health_report['warning']}, "
                                f"critical={health_report['critical']}"
                            )
                except Exception as cleanup_err:
                    logger.error(f"Error during cleanup cycle {cleanup_count}: {cleanup_err}")
                    
        except asyncio.CancelledError:
            logger.debug(f"Periodic cleanup task cancelled after {cleanup_count} cycles")
        except Exception as e:
            logger.error(f"Fatal error in periodic stale cleanup: {e}", exc_info=True)
    
    async def _monitor_builder_health(self):
        """Monitor candle builders untuk deteksi hang dan auto-recovery"""
        try:
            logger.info("🔍 Starting builder health monitor (interval=60s, hang_threshold=300s)")
            while self.running and self.connected:
                await asyncio.sleep(60)
                
                try:
                    for builder_name, builder in [
                        ('M1', self.m1_builder),
                        ('M5', self.m5_builder),
                        ('H1', self.h1_builder)
                    ]:
                        if builder.detect_hang():
                            logger.error(f"🔥 {builder_name} builder hang detected! Triggering recovery...")
                            builder.recover_from_hang()
                            logger.info(f"✅ {builder_name} builder recovered from hang")
                except Exception as check_err:
                    logger.error(f"Error during builder health check: {check_err}")
                    
        except asyncio.CancelledError:
            logger.debug("Builder health monitor cancelled")
        except Exception as e:
            logger.error(f"Fatal error in builder health monitor: {e}", exc_info=True)
    
    async def _periodic_gc_cleanup(self):
        """Periodic garbage collection and tick data cleanup.
        
        Runs every Config.GC_INTERVAL_SECONDS (default 180s) to:
        1. Clean up old tick data > Config.TICK_DATA_RETENTION_HOURS
        2. Run garbage collection on all builders
        3. Perform system-wide gc.collect()
        """
        gc_interval = Config.GC_INTERVAL_SECONDS
        retention_hours = Config.TICK_DATA_RETENTION_HOURS
        gc_cycles = 0
        
        try:
            logger.info(
                f"🧹 Starting periodic GC cleanup (interval={gc_interval}s, "
                f"tick_retention={retention_hours}h)"
            )
            
            while self.running and self.connected:
                await asyncio.sleep(gc_interval)
                
                if self._is_shutting_down:
                    logger.debug("GC cleanup skipped - shutdown in progress")
                    break
                
                try:
                    gc_cycles += 1
                    total_cleaned = 0
                    
                    for builder_name, builder in [
                        ('M1', self.m1_builder),
                        ('M5', self.m5_builder),
                        ('H1', self.h1_builder)
                    ]:
                        cleaned = builder.cleanup_old_tick_data(retention_hours)
                        total_cleaned += cleaned
                        
                        builder.run_gc_if_needed()
                    
                    gc.collect()
                    
                    if gc_cycles % 10 == 0 or total_cleaned > 0:
                        logger.info(
                            f"🧹 GC cycle {gc_cycles}: cleaned {total_cleaned} old ticks, "
                            f"M1={len(self.m1_builder.candles)}/{self.m1_builder._candle_maxlen}, "
                            f"M5={len(self.m5_builder.candles)}/{self.m5_builder._candle_maxlen}, "
                            f"H1={len(self.h1_builder.candles)}/{self.h1_builder._candle_maxlen}"
                        )
                    else:
                        logger.debug(f"GC cycle {gc_cycles} completed")
                        
                except Exception as gc_err:
                    logger.error(f"Error during GC cleanup cycle {gc_cycles}: {gc_err}")
                    
        except asyncio.CancelledError:
            logger.debug(f"Periodic GC cleanup cancelled after {gc_cycles} cycles")
        except Exception as e:
            logger.error(f"Fatal error in periodic GC cleanup: {e}", exc_info=True)
    
    async def _send_heartbeat(self):
        """Send WebSocket heartbeat ping with configurable interval and health monitoring.
        
        Uses Config.WS_HEARTBEAT_INTERVAL (default 25s) for ping frequency.
        Tracks heartbeat send times and monitors response for connection health.
        Also periodically checks connection health and triggers reconnection if needed.
        """
        heartbeat_interval = Config.WS_HEARTBEAT_INTERVAL
        health_check_counter = 0
        logger.info(f"💓 Heartbeat started: interval={heartbeat_interval}s, health_check_interval={self._health_check_interval}s")
        
        while self.running and self.ws:
            try:
                current_time = time.time()
                
                if current_time - self.last_ping >= heartbeat_interval:
                    ping_msg = {"ping": 1}
                    self._pending_heartbeat_time = current_time
                    await self.ws.send(json.dumps(ping_msg))
                    self.last_ping = current_time
                    logger.debug(f"💓 Heartbeat ping sent at {current_time:.0f}")
                
                health_check_counter += 1
                if health_check_counter >= self._health_check_interval:
                    health_check_counter = 0
                    health_score = self._check_connection_health()
                    
                    if health_score < 30:
                        logger.error(
                            f"🔴 Connection health CRITICAL: score={health_score}/100, "
                            f"triggering reconnection"
                        )
                        self.connected = False
                        if self.ws:
                            try:
                                await self.ws.close()
                            except Exception as close_err:
                                logger.debug(f"Error closing unhealthy WebSocket: {close_err}")
                        break
                    elif health_score < 60:
                        logger.warning(f"🟡 Connection health WARNING: score={health_score}/100")
                
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                logger.debug("Heartbeat task cancelled")
                raise
            except Exception as e:
                logger.warning(f"💓 Heartbeat error: {e}")
                break
    
    def _check_connection_health(self) -> int:
        """Check connection health based on heartbeat responses and data staleness.
        
        Health score factors:
        - Heartbeat response time: Fast responses = higher score
        - Time since last heartbeat: Recent = higher score
        - Data staleness: Fresh data = higher score
        - Consecutive failures: Fewer = higher score
        
        Returns:
            Health score from 0 (dead) to 100 (perfect)
        """
        score = 100
        current_time = time.time()
        
        if self.last_heartbeat_time is not None:
            time_since_heartbeat = current_time - self.last_heartbeat_time
            
            if time_since_heartbeat > self._heartbeat_timeout_threshold * 2:
                score -= 50
                logger.warning(
                    f"❌ No heartbeat response for {time_since_heartbeat:.0f}s "
                    f"(threshold: {self._heartbeat_timeout_threshold}s)"
                )
            elif time_since_heartbeat > self._heartbeat_timeout_threshold:
                score -= 30
                logger.warning(
                    f"⚠️ Heartbeat delayed: {time_since_heartbeat:.0f}s "
                    f"(threshold: {self._heartbeat_timeout_threshold}s)"
                )
            elif time_since_heartbeat > self._heartbeat_timeout_threshold / 2:
                score -= 10
        else:
            if self.connected and current_time - self.last_ping > self._heartbeat_timeout_threshold:
                score -= 20
                logger.debug("No heartbeat response received yet")
        
        if self.heartbeat_response_times:
            avg_response_time = sum(self.heartbeat_response_times) / len(self.heartbeat_response_times)
            if avg_response_time > 10.0:
                score -= 20
            elif avg_response_time > 5.0:
                score -= 10
            elif avg_response_time > 2.0:
                score -= 5
        
        if self.last_data_received:
            data_age = (datetime.now() - self.last_data_received).total_seconds()
            if data_age > 60:
                score -= 25
            elif data_age > 30:
                score -= 10
        
        failure_penalty = min(30, self.consecutive_connection_failures * 5)
        score -= failure_penalty
        
        self.connection_health_score = max(0, min(100, score))
        
        logger.debug(
            f"📊 Connection health: score={self.connection_health_score}/100, "
            f"failures={self.consecutive_connection_failures}, "
            f"last_heartbeat={self.last_heartbeat_time}"
        )
        
        return self.connection_health_score
    
    async def _monitor_data_staleness(self):
        """Monitor for stale data and trigger reconnection if needed"""
        try:
            while self.running and self.connected:
                await asyncio.sleep(10)
                
                if self.last_data_received:
                    elapsed = (datetime.now() - self.last_data_received).total_seconds()
                    
                    if elapsed > 120:
                        logger.error(f"Data stale for {elapsed:.0f}s - forcing reconnection")
                        self.connected = False
                        if self.ws:
                            try:
                                await self.ws.close()
                            except Exception as close_error:
                                logger.debug(f"Error closing stale WebSocket: {close_error}")
                        break
                    
                    elif elapsed > self.data_stale_threshold:
                        logger.warning(f"No data received for {elapsed:.0f}s (threshold: {self.data_stale_threshold}s)")
                        logger.warning("Data feed appears stale, will force reconnect if > 120s")
                        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in data staleness monitor: {e}")
    
    async def _handle_reconnect(self):
        """Handle reconnection with state machine, circuit breaker, health tracking, and enhanced backoff.
        
        Enhanced features:
        - Tracks consecutive connection failures
        - Triggers extended cooldown after max_consecutive_connection_failures (20)
        - Decrements health score on each failure
        - Improved state transition logging
        """
        self._subscribed = False
        
        self.consecutive_connection_failures += 1
        self.connection_health_score = max(0, self.connection_health_score - 10)
        
        logger.warning(
            f"🔄 Handling reconnection: consecutive_failures={self.consecutive_connection_failures}, "
            f"health_score={self.connection_health_score}/100, "
            f"max_failures_threshold={self.max_consecutive_connection_failures}"
        )
        
        if not self.running:
            logger.debug("Not running - skipping reconnection")
            return
        
        if self._is_shutting_down:
            logger.info("🛑 Shutdown in progress - skipping reconnection")
            return
        
        await self._cleanup_old_ws_tasks()
        
        if self.consecutive_connection_failures >= self.max_consecutive_connection_failures:
            logger.error(
                f"🚨 Max consecutive failures ({self.max_consecutive_connection_failures}) reached! "
                f"Triggering extended cooldown ({self._extended_cooldown_duration}s)"
            )
            delay = self._calculate_backoff_with_jitter(1, use_extended_cooldown=True)
            logger.warning(f"⏸️ Extended cooldown: waiting {delay:.0f}s before next attempt...")
            await asyncio.sleep(delay)
            
            self.consecutive_connection_failures = self.max_consecutive_connection_failures // 2
            self.connection_health_score = 30
            logger.info(f"🔄 Cooldown complete: reset failures to {self.consecutive_connection_failures}, health to 30")
        
        if self._market_closed:
            wait_time = min(300, self.reconnect_delay * 4)
            logger.info(f"🔒 Market tutup - backoff {wait_time}s sebelum retry")
            self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
            await asyncio.sleep(wait_time)
            return
        
        await self._set_connection_state(ConnectionState.RECONNECTING)
        
        cb_state = self.circuit_breaker.get_state()
        if cb_state['state'] == 'OPEN':
            remaining = self.circuit_breaker.recovery_timeout - (time.time() - (self.circuit_breaker.last_failure_time or 0))
            if remaining > 0:
                logger.warning(
                    f"⚡ Circuit breaker OPEN: waiting {remaining:.1f}s before retry | "
                    f"failures={self.consecutive_connection_failures}, health={self.connection_health_score}"
                )
                logger.warning("Falling back to simulator mode due to circuit breaker")
                await self._set_connection_state(ConnectionState.DISCONNECTED)
                self.use_simulator = True
                self.connected = False
                try:
                    self._seed_initial_tick()
                    await self._run_simulator()
                except Exception as sim_error:
                    logger.error(f"Failed to start simulator: {sim_error}", exc_info=True)
                return
        
        try:
            await self.circuit_breaker.call_async(self._attempt_reconnect)
            self.connection_metrics.record_reconnect_attempt(success=True)
            logger.info(
                f"✅ Reconnection attempt successful | "
                f"health={self.connection_health_score}, failures_reset_pending=True"
            )
        except Exception as e:
            logger.error(f"❌ Reconnection failed: {e}")
            self.connection_metrics.record_reconnect_attempt(success=False)
            
            cb_state = self.circuit_breaker.get_state()
            logger.info(
                f"📊 Circuit Breaker State: {cb_state['state']}, "
                f"failures={cb_state['failure_count']}/{cb_state['failure_threshold']}"
            )
            
            if cb_state['state'] == 'OPEN':
                logger.warning("⚡ Circuit is OPEN - falling back to simulator mode")
                await self._set_connection_state(ConnectionState.DISCONNECTED)
                self.use_simulator = True
                self.connected = False
                try:
                    self._seed_initial_tick()
                    await self._run_simulator()
                except Exception as sim_error:
                    logger.error(f"Failed to start simulator: {sim_error}", exc_info=True)
    
    async def _attempt_reconnect(self):
        """Internal reconnect logic with enhanced exponential backoff and jitter"""
        self.reconnect_attempts += 1
        
        if self.reconnect_attempts <= self.max_reconnect_attempts:
            delay = self._calculate_backoff_with_jitter(self.reconnect_attempts)
            
            logger.warning(
                f"WebSocket reconnection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts} "
                f"in {delay:.1f}s (exponential backoff with jitter)"
            )
            logger.info(f"Connection status: URL accessible check for {self.ws_url}")
            
            await asyncio.sleep(delay)
        else:
            logger.error(f"Max reconnect attempts ({self.max_reconnect_attempts}) reached after multiple failures")
            logger.warning("Gracefully degrading to SIMULATOR MODE for continued operation")
            logger.info("Simulator provides synthetic market data for testing/fallback")
            
            await self._set_connection_state(ConnectionState.DISCONNECTED)
            self.use_simulator = True
            self.connected = False
            
            try:
                self._seed_initial_tick()
                await self._run_simulator()
            except Exception as e:
                logger.error(f"Failed to start simulator: {e}", exc_info=True)
    
    def _seed_initial_tick(self):
        """Seed initial tick data with validated prices for simulator startup"""
        spread = max(self.simulator_spread_min, min(0.40, self.simulator_spread_max))
        
        mid_price = max(
            self.simulator_price_min + spread,
            min(self.base_price, self.simulator_price_max - spread)
        )
        
        self.current_bid = mid_price - (spread / 2)
        self.current_ask = mid_price + (spread / 2)
        self.current_timestamp = datetime.now(pytz.UTC)
        self.current_quote = mid_price
        self.simulator_last_timestamp = self.current_timestamp
        
        is_valid_bid, _ = self._validate_simulator_price(self.current_bid)
        is_valid_ask, _ = self._validate_simulator_price(self.current_ask)
        
        if not (is_valid_bid and is_valid_ask):
            logger.warning(
                f"Initial tick prices out of range, using safe defaults: "
                f"range=[${self.simulator_price_min:.2f}-${self.simulator_price_max:.2f}]"
            )
            safe_price = (self.simulator_price_min + self.simulator_price_max) / 2
            self.current_bid = safe_price - (spread / 2)
            self.current_ask = safe_price + (spread / 2)
            self.base_price = safe_price
        
        self.m1_builder.add_tick(self.current_bid, self.current_ask, self.current_timestamp)
        self.m5_builder.add_tick(self.current_bid, self.current_ask, self.current_timestamp)
        self.h1_builder.add_tick(self.current_bid, self.current_ask, self.current_timestamp)
        
        logger.info(
            f"Initial tick seeded: Bid=${self.current_bid:.2f}, Ask=${self.current_ask:.2f}, "
            f"Spread=${spread:.2f}, Timestamp={self.current_timestamp.isoformat()}"
        )
    
    def _validate_simulator_price(self, price: float) -> Tuple[bool, Optional[str]]:
        """Validate simulator price is within realistic XAU/USD range
        
        Args:
            price: Price to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not is_valid_price(price):
            return False, f"Invalid price value: {price}"
        
        if price < self.simulator_price_min:
            return False, f"Price ${price:.2f} below min ${self.simulator_price_min:.2f}"
        
        if price > self.simulator_price_max:
            return False, f"Price ${price:.2f} above max ${self.simulator_price_max:.2f}"
        
        return True, None
    
    def _validate_simulator_spread(self, spread: float) -> Tuple[bool, Optional[str]]:
        """Validate simulator spread is realistic
        
        Args:
            spread: Spread value to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if spread < self.simulator_spread_min:
            return False, f"Spread ${spread:.2f} below min ${self.simulator_spread_min:.2f}"
        
        if spread > self.simulator_spread_max:
            return False, f"Spread ${spread:.2f} above max ${self.simulator_spread_max:.2f}"
        
        return True, None
    
    def _validate_timestamp_monotonicity(self, new_timestamp: datetime) -> Tuple[bool, Optional[str]]:
        """Validate that timestamp is always moving forward
        
        Args:
            new_timestamp: New timestamp to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.simulator_last_timestamp is None:
            return True, None
        
        if new_timestamp <= self.simulator_last_timestamp:
            return False, (
                f"Timestamp not advancing: new={new_timestamp.isoformat()} "
                f"<= last={self.simulator_last_timestamp.isoformat()}"
            )
        
        return True, None
    
    async def _run_simulator(self):
        """Run price simulator with realistic XAU/USD data validation"""
        logger.info("Starting price simulator (fallback mode)")
        logger.info(
            f"Simulator config: base_price=${self.base_price:.2f}, volatility=±${self.price_volatility:.2f}, "
            f"price_range=[${self.simulator_price_min:.2f}-${self.simulator_price_max:.2f}], "
            f"spread_range=[${self.simulator_spread_min:.2f}-${self.simulator_spread_max:.2f}]"
        )
        
        tick_count = 0
        validation_errors = 0
        
        while self.use_simulator:
            try:
                spread = self.simulator_spread_min + random.uniform(0, self.simulator_spread_max - self.simulator_spread_min)
                
                price_change = random.uniform(-self.price_volatility, self.price_volatility)
                mid_price = self.base_price + price_change
                
                mid_price = max(self.simulator_price_min, min(mid_price, self.simulator_price_max))
                
                current_bid = mid_price - (spread / 2)
                current_ask = mid_price + (spread / 2)
                new_timestamp = datetime.now(pytz.UTC)
                
                is_valid_bid, bid_error = self._validate_simulator_price(current_bid)
                is_valid_ask, ask_error = self._validate_simulator_price(current_ask)
                is_valid_spread, spread_error = self._validate_simulator_spread(spread)
                is_valid_ts, ts_error = self._validate_timestamp_monotonicity(new_timestamp)
                
                if not is_valid_bid:
                    logger.warning(f"Simulator bid validation failed: {bid_error}")
                    validation_errors += 1
                    await asyncio.sleep(0.1)
                    continue
                
                if not is_valid_ask:
                    logger.warning(f"Simulator ask validation failed: {ask_error}")
                    validation_errors += 1
                    await asyncio.sleep(0.1)
                    continue
                
                if not is_valid_spread:
                    logger.debug(f"Simulator spread out of range, clamping: {spread_error}")
                    spread = max(self.simulator_spread_min, min(spread, self.simulator_spread_max))
                    current_bid = mid_price - (spread / 2)
                    current_ask = mid_price + (spread / 2)
                
                if not is_valid_ts:
                    logger.warning(f"Simulator timestamp validation failed: {ts_error}")
                    await asyncio.sleep(0.05)
                    continue
                
                self.current_bid = current_bid
                self.current_ask = current_ask
                self.current_timestamp = new_timestamp
                self.current_quote = mid_price
                self.simulator_last_timestamp = new_timestamp
                
                if not self._loading_from_db:
                    self.m1_builder.add_tick(self.current_bid, self.current_ask, self.current_timestamp)
                    self.m5_builder.add_tick(self.current_bid, self.current_ask, self.current_timestamp)
                    self.h1_builder.add_tick(self.current_bid, self.current_ask, self.current_timestamp)
                else:
                    logger.debug("Skipping simulator tick - loading from DB in progress")
                    await asyncio.sleep(0.1)
                    continue
                
                tick_data = {
                    'bid': self.current_bid,
                    'ask': self.current_ask,
                    'quote': self.current_quote,
                    'timestamp': self.current_timestamp
                }
                await self._broadcast_tick(tick_data)
                
                tick_count += 1
                self._log_tick_sample(self.current_bid, self.current_ask, self.current_quote, spread, mode="simulator")
                
                drift = random.uniform(-0.5, 0.5)
                self.base_price = max(
                    self.simulator_price_min + self.price_volatility,
                    min(mid_price + drift, self.simulator_price_max - self.price_volatility)
                )
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Simulator error: {e}")
                await asyncio.sleep(5)
        
        logger.info(
            f"Price simulator stopped: {tick_count} ticks generated, "
            f"{validation_errors} validation errors"
        )
    
    async def _on_message(self, message: Union[str, bytes]):
        """Process incoming WebSocket message with validation and NaN handling"""
        try:
            if not message:
                logger.warning("Received empty message")
                return
            
            try:
                data = json.loads(message)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON message: {e}")
                logger.debug(f"Raw message: {message[:500]}")
                return
            
            if not isinstance(data, dict):
                logger.warning(f"Message is not a dict: {type(data)}")
                return
            
            if "tick" in data:
                tick = data["tick"]
                
                if not isinstance(tick, dict):
                    logger.warning(f"Tick data is not a dict: {type(tick)}")
                    return
                
                try:
                    epoch = tick.get("epoch", int(datetime.now(pytz.UTC).timestamp()))
                    bid = tick.get("bid")
                    ask = tick.get("ask")
                    quote = tick.get("quote")
                    
                    if bid is None or ask is None:
                        logger.warning(f"Missing bid/ask in tick: bid={bid}, ask={ask}")
                        return
                    
                    try:
                        bid_float = float(bid)
                        ask_float = float(ask)
                    except (ValueError, TypeError) as e:
                        logger.error(f"Invalid bid/ask values: bid={bid}, ask={ask}, error={e}")
                        return
                    
                    if not is_valid_price(bid_float) or not is_valid_price(ask_float):
                        logger.warning(f"Invalid prices detected (NaN/Inf/negative): bid={bid_float}, ask={ask_float}")
                        return
                    
                    if ask_float < bid_float:
                        logger.warning(f"Ask < Bid: ask={ask_float}, bid={bid_float}")
                        return
                    
                    self.current_bid = bid_float
                    self.current_ask = ask_float
                    self.current_quote = float(quote) if quote and is_valid_price(float(quote)) else (self.current_bid + self.current_ask) / 2
                    self.current_timestamp = datetime.fromtimestamp(epoch, tz=pytz.UTC)
                    self.last_data_received = datetime.now()
                    
                    if self._loading_from_db:
                        logger.debug("Skipping tick processing - loading from DB in progress")
                        return
                    
                    self.m1_builder.add_tick(self.current_bid, self.current_ask, self.current_timestamp)
                    self.m5_builder.add_tick(self.current_bid, self.current_ask, self.current_timestamp)
                    self.h1_builder.add_tick(self.current_bid, self.current_ask, self.current_timestamp)
                    
                    self._log_tick_sample(self.current_bid, self.current_ask, self.current_quote, mode="websocket")
                    
                    tick_data = {
                        'bid': self.current_bid,
                        'ask': self.current_ask,
                        'quote': self.current_quote,
                        'timestamp': self.current_timestamp
                    }
                    await self._broadcast_tick(tick_data)
                    
                except Exception as e:
                    logger.error(f"Error processing tick data: {e}")
                    logger.debug(f"Tick content: {tick}")
                
            elif "pong" in data:
                current_time = time.time()
                self.last_heartbeat_time = current_time
                
                if self._pending_heartbeat_time is not None:
                    response_time = current_time - self._pending_heartbeat_time
                    self.heartbeat_response_times.append(response_time)
                    self._pending_heartbeat_time = None
                    
                    if response_time > 5.0:
                        logger.warning(f"💓 Pong received (SLOW): response_time={response_time:.2f}s")
                    elif response_time > 2.0:
                        logger.info(f"💓 Pong received: response_time={response_time:.2f}s")
                    else:
                        logger.debug(f"💓 Pong received: response_time={response_time:.3f}s")
                    
                    if len(self.heartbeat_response_times) >= 5:
                        avg_time = sum(self.heartbeat_response_times) / len(self.heartbeat_response_times)
                        if avg_time > 5.0:
                            self.connection_health_score = max(0, self.connection_health_score - 10)
                            logger.warning(f"⚠️ High average heartbeat latency: {avg_time:.2f}s, health={self.connection_health_score}")
                else:
                    logger.debug("💓 Pong received (no pending ping tracked)")
            
            elif "error" in data:
                error = data["error"]
                error_msg = error.get('message', 'Unknown error') if isinstance(error, dict) else str(error)
                error_code = error.get('code', 'N/A') if isinstance(error, dict) else 'N/A'
                logger.error(f"API Error (code {error_code}): {error_msg}")
                logger.debug(f"Full error data: {error}")
            
            elif "msg_type" in data:
                msg_type = data["msg_type"]
                if msg_type not in ["tick", "ping", "pong"]:
                    logger.debug(f"Received message type: {msg_type}")
                        
        except Exception as e:
            logger.error(f"Unexpected error processing message: {type(e).__name__}: {e}", exc_info=True)
            if message:
                logger.debug(f"Raw message (truncated): {message[:500]}")
    
    async def get_current_price(self) -> Optional[float]:
        """Get current mid price with validation and HTTP fallback cascade"""
        try:
            data_is_fresh = False
            if self.last_data_received:
                time_since_last_data = (datetime.now() - self.last_data_received).total_seconds()
                data_is_fresh = time_since_last_data < 30
            
            # Tier 1: WebSocket data (jika fresh)
            if data_is_fresh and self.current_bid is not None and self.current_ask is not None:
                if is_valid_price(self.current_bid) and is_valid_price(self.current_ask):
                    if self.current_ask >= self.current_bid:
                        mid_price = (self.current_bid + self.current_ask) / 2.0
                        if is_valid_price(mid_price):
                            logger.debug(f"Price from WS: ${mid_price:.2f}")
                            return mid_price
                    else:
                        logger.warning(f"Invalid bid/ask: bid={self.current_bid}, ask={self.current_ask}")
            
            # Tier 2: HTTP fallback (jika WS stale atau FREE_TIER_MODE)
            if self.config.FREE_TIER_MODE or not data_is_fresh:
                try:
                    http_price = await asyncio.wait_for(self.fetch_price_via_http(), timeout=2.0)
                    if http_price:
                        logger.info(f"🌐 Price from HTTP fallback: ${http_price:.2f} (WS {'stale' if not data_is_fresh else 'fresh but invalid'})")
                        return http_price
                except asyncio.TimeoutError:
                    logger.warning("HTTP fallback timeout")
                except Exception as e:
                    logger.warning(f"HTTP fallback error: {e}")
            
            # Tier 3: Last candle M1 close price from builder (emergency fallback)
            try:
                if self.m1_builder.candles and len(self.m1_builder.candles) > 0:
                    emergency_price = float(self.m1_builder.candles[-1]['close'])
                    if emergency_price > 0:
                        logger.warning(f"🚨 Using emergency price from M1 builder: ${emergency_price:.2f}")
                        return emergency_price
            except (ValueError, TypeError, KeyError, IndexError, AttributeError):
                pass
            
            logger.warning("No valid current price available (all fallbacks failed)")
            return None
            
        except Exception as e:
            logger.error(f"Error calculating current price: {e}")
            return None
    
    def is_websocket_healthy(self) -> bool:
        """Check if WebSocket connection is healthy and receiving data"""
        if not self.connected or not self.running:
            return False
        
        if self.last_data_received is None:
            return False
        
        time_since_last_data = (datetime.now() - self.last_data_received).total_seconds()
        return time_since_last_data < 30

    def get_connection_health(self) -> dict:
        """Get detailed connection health status"""
        time_since_last_data = None
        if self.last_data_received:
            time_since_last_data = (datetime.now() - self.last_data_received).total_seconds()
        
        return {
            'connected': self.connected,
            'running': self.running,
            'websocket_healthy': self.is_websocket_healthy(),
            'last_data_received': self.last_data_received.isoformat() if self.last_data_received else None,
            'seconds_since_last_data': time_since_last_data,
            'connection_state': self._connection_state.value,
            'reconnect_attempts': self.reconnect_attempts
        }
    
    async def get_bid_ask(self) -> Optional[Tuple[float, float]]:
        """Get current bid/ask with validation"""
        try:
            if self.current_bid is not None and self.current_ask is not None:
                if is_valid_price(self.current_bid) and is_valid_price(self.current_ask):
                    if self.current_ask >= self.current_bid:
                        return (self.current_bid, self.current_ask)
                    else:
                        logger.warning(f"Invalid bid/ask: bid={self.current_bid}, ask={self.current_ask}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting bid/ask: {e}")
            return None
    
    async def get_spread(self) -> Optional[float]:
        """Get current spread with validation"""
        try:
            if self.current_bid is not None and self.current_ask is not None:
                if is_valid_price(self.current_bid) and is_valid_price(self.current_ask):
                    if self.current_ask >= self.current_bid:
                        spread = self.current_ask - self.current_bid
                        if spread >= 0 and not math.isnan(spread) and not math.isinf(spread):
                            return spread
                        logger.warning(f"Invalid spread calculated: {spread}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating spread: {e}")
            return None
    
    async def get_historical_data(self, timeframe: str = 'M1', limit: int = 100) -> Optional[pd.DataFrame]:
        """Get historical data with validation and timeout handling
        
        Supports M1 (1 minute), M5 (5 minute), and H1 (1 hour) timeframes.
        """
        try:
            valid_timeframes = ['M1', 'M5', 'H1']
            if not timeframe or timeframe not in valid_timeframes:
                logger.warning(f"Invalid timeframe: {timeframe}. Must be one of {valid_timeframes}")
                return None
            
            if limit <= 0:
                logger.warning(f"Invalid limit: {limit}. Using default 100")
                limit = 100
            
            if timeframe == 'M1':
                df = self.m1_builder.get_dataframe(limit)
                if df is not None and len(df) > 0:
                    logger.debug(f"Retrieved {len(df)} M1 candles from tick feed")
                    return df
                else:
                    logger.debug("No M1 data available")
                    
            elif timeframe == 'M5':
                df = self.m5_builder.get_dataframe(limit)
                if df is not None and len(df) > 0:
                    logger.debug(f"Retrieved {len(df)} M5 candles from tick feed")
                    return df
                else:
                    logger.debug("No M5 data available")
            
            elif timeframe == 'H1':
                df = self.h1_builder.get_dataframe(limit)
                if df is not None and len(df) > 0:
                    logger.debug(f"Retrieved {len(df)} H1 candles from tick feed")
                    return df
                else:
                    logger.debug("No H1 data available")
            
            return None
                        
        except Exception as e:
            logger.error(f"Error fetching historical data for {timeframe}: {e}", exc_info=True)
            return None
    
    async def save_candles_to_db(self, db_manager):
        """Menyimpan 100 candle terakhir ke database untuk persistensi dengan proteksi race condition
        
        Menggunakan candle_lock dan db_write_lock untuk thread-safety dan konsistensi data.
        Menambahkan validasi OHLC integrity sebelum menyimpan.
        """
        async with self.candle_lock:
            try:
                from bot.database import CandleData
                from sqlalchemy import delete
                
                snapshots = {}
                ohlc_invalid_count = 0
                
                for timeframe, builder in [('M1', self.m1_builder), ('M5', self.m5_builder), ('H1', self.h1_builder)]:
                    valid_candles = []
                    for candle in builder.candles:
                        is_valid, scrubbed = builder._scrub_nan_prices(candle)
                        if is_valid:
                            ohlc_valid, _ = validate_ohlc_integrity(
                                scrubbed['open'], scrubbed['high'],
                                scrubbed['low'], scrubbed['close']
                            )
                            if ohlc_valid:
                                valid_candles.append(scrubbed)
                            else:
                                ohlc_invalid_count += 1
                    snapshots[timeframe] = valid_candles
                    
                    if builder.current_candle:
                        is_valid, scrubbed = builder._scrub_nan_prices(builder.current_candle)
                        if is_valid:
                            ohlc_valid, _ = validate_ohlc_integrity(
                                scrubbed['open'], scrubbed['high'],
                                scrubbed['low'], scrubbed['close']
                            )
                            if ohlc_valid:
                                snapshots[timeframe + '_current'] = scrubbed
                            else:
                                ohlc_invalid_count += 1
                                snapshots[timeframe + '_current'] = None
                        else:
                            snapshots[timeframe + '_current'] = None
                    else:
                        snapshots[timeframe + '_current'] = None
                
                if ohlc_invalid_count > 0:
                    logger.warning(f"Melewati {ohlc_invalid_count} candle dengan integritas OHLC tidak valid saat menyimpan ke DB")
                
                logger.debug("Membuat snapshot thread-safe dari data candle (termasuk current_candle) untuk disimpan ke DB")
            except Exception as e:
                logger.error(f"Error membuat snapshot candle: {e}")
                return False
        
        async with self.db_write_lock:
            session = None
            try:
                session = db_manager.get_session()
                
                try:
                    session.rollback()
                except Exception:
                    pass
                
                saved_m1 = 0
                saved_m5 = 0
                saved_h1 = 0
                
                for timeframe in ['M1', 'M5', 'H1']:
                    all_candles = []
                    for candle in snapshots[timeframe]:
                        candle_copy = candle.copy()
                        candle_copy['is_partial'] = False
                        all_candles.append(candle_copy)
                    
                    current_candle = snapshots.get(timeframe + '_current')
                    if current_candle:
                        partial_candle = current_candle.copy()
                        partial_candle['is_partial'] = True
                        all_candles.append(partial_candle)
                        logger.debug(f"Including current_candle (is_partial=True) in {timeframe} save (total: {len(all_candles)})")
                    
                    if len(all_candles) == 0:
                        logger.debug(f"No {timeframe} candles to save")
                        continue
                    
                    seen_timestamps = set()
                    deduplicated_candles = []
                    for candle in all_candles:
                        ts = candle['timestamp']
                        if ts not in seen_timestamps:
                            seen_timestamps.add(ts)
                            deduplicated_candles.append(candle)
                    
                    if len(deduplicated_candles) < len(all_candles):
                        removed = len(all_candles) - len(deduplicated_candles)
                        logger.debug(f"Removed {removed} duplicate candle(s) from {timeframe} before saving")
                    
                    seen_timestamps.clear()
                    
                    max_candles = 50 if timeframe == 'H1' else self.config.MAX_CANDLE_HISTORY
                    candles_to_save = deduplicated_candles[-max_candles:]
                    
                    session.execute(delete(CandleData).where(CandleData.timeframe == timeframe))
                    
                    for candle_dict in candles_to_save:
                        candle_record = CandleData(
                            timeframe=timeframe,
                            timestamp=candle_dict['timestamp'],
                            open=float(candle_dict['open']),
                            high=float(candle_dict['high']),
                            low=float(candle_dict['low']),
                            close=float(candle_dict['close']),
                            volume=float(candle_dict.get('volume', 0)),
                            is_partial=candle_dict.get('is_partial', False)
                        )
                        session.add(candle_record)
                    
                    if timeframe == 'M1':
                        saved_m1 = len(candles_to_save)
                    elif timeframe == 'M5':
                        saved_m5 = len(candles_to_save)
                    else:
                        saved_h1 = len(candles_to_save)
                
                session.commit()
                session.close()
                
                logger.info(f"✅ Saved candles to database: M1={saved_m1}, M5={saved_m5}, H1={saved_h1}")
                return True
                
            except Exception as e:
                logger.error(f"Error saving candles to database: {e}", exc_info=True)
                if session is not None:
                    try:
                        session.rollback()
                        session.close()
                    except Exception:
                        pass
                return False
    
    async def load_candles_from_db(self, db_manager):
        """Load candles from database on startup with race condition protection
        
        Memory optimization: Uses sliding window deque for duplicate detection
        instead of unbounded set to prevent large memory consumption.
        
        Also stores db_manager reference for immediate H1 candle saves.
        Flushes any pending H1 candles that were queued before db_manager was set.
        """
        self._db_manager = db_manager
        logger.info("✅ db_manager reference stored for immediate H1 candle saves")
        
        await self._flush_pending_h1_candles()
        
        self._loading_from_db = True
        logger.debug("Set _loading_from_db=True - blocking WebSocket tick processing")
        
        async with self.candle_lock:
            session = None
            try:
                from bot.database import CandleData
                
                self.m1_builder.clear()
                self.m5_builder.clear()
                self.h1_builder.clear()
                logger.info("Cleared existing candle builders (M1, M5, H1) before loading from database")
                
                session = db_manager.get_session()
                
                loaded_m1 = 0
                loaded_m5 = 0
                loaded_h1 = 0
                nan_skipped = 0
                
                for timeframe, builder in [('M1', self.m1_builder), ('M5', self.m5_builder), ('H1', self.h1_builder)]:
                    candles = session.query(CandleData).filter(
                        CandleData.timeframe == timeframe
                    ).order_by(CandleData.timestamp.asc()).all()
                    
                    if not candles:
                        logger.info(f"No {timeframe} candles found in database (first run?)")
                        continue
                    
                    recent_timestamps = deque(maxlen=20)
                    duplicates_skipped = 0
                    partial_restored = 0
                    
                    for candle in candles:
                        is_partial = getattr(candle, 'is_partial', False) or False
                        if is_partial:
                            logger.debug(f"Restoring partial candle at {candle.timestamp} for {timeframe}")
                            builder.current_candle = {
                                'timestamp': candle.timestamp,
                                'open': candle.open,
                                'high': candle.high,
                                'low': candle.low,
                                'close': candle.close,
                                'volume': candle.volume
                            }
                            partial_restored += 1
                            continue
                        
                        timestamp = candle.timestamp
                        
                        if timestamp.tzinfo is None:
                            ts = pd.Timestamp(timestamp).tz_localize('UTC')
                        else:
                            ts = pd.Timestamp(timestamp).tz_convert('UTC')
                        
                        if ts in recent_timestamps:
                            duplicates_skipped += 1
                            logger.debug(f"Skipping duplicate candle at {ts} for {timeframe}")
                            continue
                        
                        recent_timestamps.append(ts)
                        
                        open_val = float(candle.open)
                        high_val = float(candle.high)
                        low_val = float(candle.low)
                        close_val = float(candle.close)
                        
                        if any(math.isnan(v) or math.isinf(v) for v in [open_val, high_val, low_val, close_val]):
                            nan_skipped += 1
                            logger.warning(f"Melewati candle dengan NaN/Inf pada {ts} untuk {timeframe}")
                            continue
                        
                        is_valid_ohlc, ohlc_error = validate_ohlc_integrity(open_val, high_val, low_val, close_val)
                        if not is_valid_ohlc:
                            nan_skipped += 1
                            logger.warning(f"Melewati candle {timeframe} pada {ts} - integritas OHLC gagal: {ohlc_error}")
                            continue
                        
                        candle_dict = {
                            'timestamp': ts,
                            'open': open_val,
                            'high': high_val,
                            'low': low_val,
                            'close': close_val,
                            'volume': float(candle.volume) if candle.volume else 0
                        }
                        builder.candles.append(candle_dict)
                    
                    recent_timestamps.clear()
                    
                    if partial_restored > 0:
                        logger.info(f"Restored {partial_restored} partial candle(s) to {timeframe} builder current_candle (will continue from live data)")
                    
                    if duplicates_skipped > 0:
                        logger.info(f"Skipped {duplicates_skipped} duplicate candle(s) from {timeframe} during load")
                    
                    if timeframe == 'M1':
                        loaded_m1 = len(builder.candles)
                    elif timeframe == 'M5':
                        loaded_m5 = len(builder.candles)
                    else:
                        loaded_h1 = len(builder.candles)
                
                session.close()
                
                if nan_skipped > 0:
                    logger.warning(f"Skipped {nan_skipped} candles with NaN/Inf values during load")
                
                if loaded_m1 > 0 or loaded_m5 > 0 or loaded_h1 > 0:
                    logger.info(f"✅ Loaded candles from database: M1={loaded_m1}, M5={loaded_m5}, H1={loaded_h1}")
                    logger.info("Bot has candles immediately - no waiting for Deriv API!")
                    
                    if loaded_m1 >= 30 or loaded_m5 >= 30 or loaded_h1 >= 10:
                        self._loaded_from_db = True
                        logger.info("✅ Set _loaded_from_db=True - will skip historical fetch from Deriv")
                    else:
                        logger.warning(f"Loaded candles ({loaded_m1} M1, {loaded_m5} M5, {loaded_h1} H1) below threshold (M1/M5:30, H1:10) - will fetch from Deriv")
                    
                    self._loading_from_db = False
                    logger.debug("Set _loading_from_db=False - WebSocket tick processing enabled")
                    return True
                else:
                    logger.info("No candles in database - will fetch from Deriv API")
                    self._loaded_from_db = False
                    self._loading_from_db = False
                    logger.debug("Set _loading_from_db=False - WebSocket tick processing enabled")
                    return False
                    
            except Exception as e:
                logger.error(f"Error loading candles from database: {e}", exc_info=True)
                logger.warning("Falling back to fetching candles from Deriv API")
                if session is not None:
                    try:
                        session.close()
                    except Exception:
                        pass
                self._loading_from_db = False
                logger.debug("Set _loading_from_db=False - WebSocket tick processing enabled (after error)")
                return False
    
    def _prune_old_candles(self, db_manager, keep_count: int = 150):
        """Prune old candles from database to prevent bloat
        
        Args:
            db_manager: Database manager instance
            keep_count: Number of newest candles to keep per timeframe (default: 150)
                       Must be >= 1 and <= 10000
        
        Returns:
            Number of pruned candles, or 0 on error
        """
        if keep_count is None or not isinstance(keep_count, int):
            logger.warning(f"Invalid keep_count type: {type(keep_count)}. Using default 150")
            keep_count = 150
        elif keep_count < 1:
            logger.warning(f"Invalid keep_count: {keep_count}. Must be >= 1. Using default 150")
            keep_count = 150
        elif keep_count > 10000:
            logger.warning(f"keep_count too large: {keep_count}. Capping at 10000")
            keep_count = 10000
        
        session = None
        try:
            from bot.database import CandleData
            from sqlalchemy import func
            
            session = db_manager.get_session()
            
            pruned_total = 0
            
            for timeframe in ['M1', 'M5']:
                total_count = session.query(func.count(CandleData.id)).filter(
                    CandleData.timeframe == timeframe
                ).scalar()
                
                if total_count is None or total_count <= keep_count:
                    logger.debug(f"{timeframe}: {total_count or 0} candles <= keep_count ({keep_count}), skipping prune")
                    continue
                
                excess_count = total_count - keep_count
                logger.debug(f"{timeframe}: {total_count} candles, need to prune {excess_count}")
                
                candles = session.query(CandleData).filter(
                    CandleData.timeframe == timeframe
                ).order_by(CandleData.timestamp.desc()).limit(keep_count).all()
                
                if candles and len(candles) > 0:
                    oldest_to_keep = candles[-1].timestamp
                    
                    deleted = session.query(CandleData).filter(
                        CandleData.timeframe == timeframe,
                        CandleData.timestamp < oldest_to_keep
                    ).delete(synchronize_session=False)
                    
                    if deleted > 0:
                        logger.debug(f"{timeframe}: Deleted {deleted} candles older than {oldest_to_keep}")
                    pruned_total += deleted
                else:
                    logger.warning(f"{timeframe}: Failed to get candles for pruning reference")
            
            session.commit()
            session.close()
            
            if pruned_total > 0:
                logger.info(f"Pruned {pruned_total} old candles from database (keep_count={keep_count})")
            
            return pruned_total
            
        except Exception as e:
            logger.error(f"Error pruning old candles: {e}", exc_info=True)
            if session is not None:
                try:
                    session.rollback()
                    session.close()
                except Exception:
                    pass
            return 0
    
    async def shutdown(self):
        """Graceful shutdown with proper cleanup of all resources
        
        Shutdown sequence:
        1. Set shutdown flags to disable reconnection
        2. Stop running loops
        3. Shutdown all WebSocket tasks with proper awaiting
        4. Unsubscribe all subscribers
        5. Close WebSocket connection
        6. Set final disconnected state
        """
        logger.info("🛑 Initiating MarketDataClient shutdown...")
        
        self.prepare_for_shutdown()
        
        self.running = False
        self.use_simulator = False
        
        logger.info("🔄 Shutting down WebSocket tasks...")
        task_results = await self._shutdown_websocket_tasks(timeout_per_task=self._ws_task_cancel_timeout)
        
        completed_count = sum(1 for s in task_results.values() if 'cancelled' in s or 'completed' in s)
        logger.info(f"📊 WebSocket tasks shutdown: {completed_count}/{len(task_results)} cleanly terminated")
        
        logger.info("🔄 Unsubscribing all subscribers...")
        await self._unsubscribe_all()
        
        self.connected = False
        if self.ws:
            try:
                await asyncio.wait_for(self.ws.close(), timeout=5.0)
                logger.info("✅ WebSocket closed successfully")
            except asyncio.TimeoutError:
                logger.warning("⚠️ WebSocket close timeout - forcing close")
            except Exception as e:
                logger.debug(f"Error closing WebSocket during shutdown: {e}")
        
        await self._set_connection_state(ConnectionState.DISCONNECTED)
        
        logger.info("✅ MarketDataClient shutdown complete")
    
    async def disconnect_async(self):
        """Async disconnect with proper cleanup - prefer this over sync disconnect()
        
        This method properly awaits all WebSocket tasks before closing.
        """
        logger.info("🔌 Disconnecting MarketDataClient (async)...")
        
        self._is_shutting_down = True
        self.running = False
        self.use_simulator = False
        
        task_results = await self._shutdown_websocket_tasks(timeout_per_task=self._ws_task_cancel_timeout)
        logger.debug(f"Task cleanup results: {task_results}")
        
        self.connected = False
        if self.ws:
            try:
                await asyncio.wait_for(self.ws.close(), timeout=5.0)
                logger.info("✅ WebSocket closed")
            except asyncio.TimeoutError:
                logger.warning("⚠️ WebSocket close timeout")
            except Exception as e:
                logger.debug(f"Error closing WebSocket: {e}")
        
        await self._set_connection_state(ConnectionState.DISCONNECTED)
        logger.info("✅ MarketDataClient disconnected (async)")
    
    def disconnect(self):
        """Synchronous disconnect - schedules async cleanup
        
        Note: For proper cleanup, use disconnect_async() or shutdown() instead.
        This method schedules async cleanup but may not fully await all tasks.
        """
        logger.info("🔌 Disconnecting MarketDataClient (sync)...")
        
        self._is_shutting_down = True
        self.running = False
        self.use_simulator = False
        self.connected = False
        
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._shutdown_websocket_tasks(timeout_per_task=2.0))
            if self.ws:
                loop.create_task(self.ws.close())
        except RuntimeError:
            if self.ws:
                try:
                    asyncio.run(self.ws.close())
                except Exception as e:
                    logger.debug(f"Error closing WebSocket in sync disconnect: {e}")
        
        logger.info("MarketData client disconnect initiated (async cleanup scheduled)")
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected and subscribed to data feed"""
        if self.use_simulator:
            return True
        # Must be both connected AND subscribed for usable data
        return self.connected and self._subscribed
    
    def cleanup_all_tick_data(self, retention_hours: Optional[int] = None) -> Dict[str, int]:
        """Cleanup old tick data from all OHLC builders.
        
        Thread-safe method to clean up old tick data across all timeframes.
        Useful for memory management on constrained environments (like Koyeb free tier).
        
        Args:
            retention_hours: Hours to retain tick data (default: Config.TICK_DATA_RETENTION_HOURS)
            
        Returns:
            Dict with cleaned counts per timeframe
        """
        retention = retention_hours if retention_hours is not None else Config.TICK_DATA_RETENTION_HOURS
        
        results = {
            'M1': self.m1_builder.cleanup_old_tick_data(retention),
            'M5': self.m5_builder.cleanup_old_tick_data(retention),
            'H1': self.h1_builder.cleanup_old_tick_data(retention)
        }
        
        total_cleaned = sum(results.values())
        if total_cleaned > 0:
            logger.info(f"🧹 Cleaned {total_cleaned} old tick data entries (M1:{results['M1']}, M5:{results['M5']}, H1:{results['H1']})")
        
        return results
    
    def run_memory_cleanup(self) -> Dict[str, Any]:
        """Run comprehensive memory cleanup for all market data structures.
        
        This method:
        1. Cleans up old tick data from all builders
        2. Triggers garbage collection on each builder
        3. Forces Python GC
        
        Returns:
            Dict with cleanup statistics
        """
        tick_cleanup = self.cleanup_all_tick_data()
        
        gc_results = {
            'M1': self.m1_builder.run_gc_if_needed(),
            'M5': self.m5_builder.run_gc_if_needed(),
            'H1': self.h1_builder.run_gc_if_needed()
        }
        
        gc.collect()
        
        return {
            'tick_cleanup': tick_cleanup,
            'gc_triggered': gc_results,
            'total_tick_cleaned': sum(tick_cleanup.values())
        }
    
    def get_status(self) -> Dict:
        return {
            'connected': self.connected,
            'simulator_mode': self.use_simulator,
            'connection_state': self._connection_state.value,
            'reconnect_attempts': self.reconnect_attempts,
            'has_data': self.current_bid is not None and self.current_ask is not None,
            'websocket_url': self.ws_url,
            'subscriber_count': len(self.subscribers),
            'circuit_breaker': self.circuit_breaker.get_state(),
            'connection_metrics': self.connection_metrics.get_stats(),
            'm1_builder_stats': self.m1_builder.get_stats(),
            'm5_builder_stats': self.m5_builder.get_stats()
        }
