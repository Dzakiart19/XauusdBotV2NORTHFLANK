import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Union
from numpy.typing import NDArray


logger = logging.getLogger('indicators')


class IndicatorError(Exception):
    """Base exception for indicator calculation errors"""
    pass

def validate_series(series: pd.Series, min_length: int = 1, fill_value: float = 0.0) -> pd.Series:
    """
    Validate and sanitize a pandas Series before mathematical operations.
    
    Args:
        series: The pandas Series to validate
        min_length: Minimum required length
        fill_value: Value to use for filling NaN/None/Inf values
    
    Returns:
        Validated Series with NaN/Inf values filled
    
    Raises:
        ValueError: If series is None or not a pandas Series
    """
    if series is None:
        raise ValueError("Series tidak boleh None")
    
    if not isinstance(series, pd.Series):
        if isinstance(series, (list, np.ndarray)):
            series = pd.Series(series)
        elif isinstance(series, (int, float)):
            series = pd.Series([series])
        else:
            raise ValueError(f"Tipe data tidak valid, expected pandas Series, got {type(series)}")
    
    if len(series) == 0:
        return pd.Series([fill_value] * max(min_length, 1))
    
    if len(series) < min_length:
        padded = pd.Series([fill_value] * min_length)
        padded.iloc[-len(series):] = series.values
        series = padded
    
    result = series.replace([np.inf, -np.inf], np.nan)
    result = result.fillna(fill_value)
    
    return result


def _ensure_series(data: Union[pd.Series, pd.DataFrame, np.ndarray, float, int, None], 
                   index: Optional[pd.Index] = None) -> pd.Series:
    """
    Convert various data types to pandas Series.
    
    Args:
        data: Input data (Series, DataFrame, ndarray, float, int, or None)
        index: Optional index for the resulting Series
    
    Returns:
        pandas Series
    """
    if data is None:
        if index is not None:
            return pd.Series([0.0] * len(index), index=index)
        return pd.Series([0.0])
    
    if isinstance(data, pd.Series):
        return data
    elif isinstance(data, pd.DataFrame):
        if len(data.columns) == 0:
            if index is not None:
                return pd.Series([0.0] * len(index), index=index)
            return pd.Series([0.0])
        if len(data.columns) == 1:
            return pd.Series(data.iloc[:, 0], index=data.index)
        return pd.Series(data.iloc[:, 0], index=data.index)
    elif isinstance(data, np.ndarray):
        if data.size == 0:
            if index is not None:
                return pd.Series([0.0] * len(index), index=index)
            return pd.Series([0.0])
        return pd.Series(data.flatten() if data.ndim > 1 else data, index=index)
    elif isinstance(data, (float, int)):
        if np.isnan(data) or np.isinf(data):
            data = 0.0
        if index is not None:
            return pd.Series([data] * len(index), index=index)
        return pd.Series([data])
    else:
        try:
            return pd.Series([float(data)])
        except (ValueError, TypeError):
            return pd.Series([0.0])


def safe_divide(numerator: Union[pd.Series, pd.DataFrame, np.ndarray, float, None], 
                denominator: Union[pd.Series, pd.DataFrame, np.ndarray, float, None], 
                fill_value: float = 0.0,
                min_denominator: float = 1e-10) -> pd.Series:
    """
    Safely divide two Series, handling division by zero, NaN, and Inf values.
    
    Args:
        numerator: Numerator (Series, DataFrame, ndarray, float, or None)
        denominator: Denominator (Series, DataFrame, ndarray, float, or None)
        fill_value: Value to use when division is undefined
        min_denominator: Minimum absolute value for denominator to prevent division by very small numbers
    
    Returns:
        Result Series with safe division
    """
    num_series = _ensure_series(numerator)
    denom_series = _ensure_series(denominator)
    
    if len(num_series) == 1 and len(denom_series) > 1:
        num_series = pd.Series([num_series.iloc[0]] * len(denom_series), index=denom_series.index)
    elif len(denom_series) == 1 and len(num_series) > 1:
        denom_series = pd.Series([denom_series.iloc[0]] * len(num_series), index=num_series.index)
    elif len(num_series) != len(denom_series):
        if hasattr(denom_series, 'index') and len(denom_series) > len(num_series):
            num_series = pd.Series([num_series.iloc[0] if len(num_series) > 0 else 0.0] * len(denom_series), index=denom_series.index)
        elif hasattr(num_series, 'index'):
            denom_series = pd.Series([denom_series.iloc[0] if len(denom_series) > 0 else 1.0] * len(num_series), index=num_series.index)
    
    num_series = num_series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    denom_series = denom_series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    denom_safe = denom_series.copy()
    zero_denom_mask = denom_safe.abs() < min_denominator
    sign_mask = denom_safe >= 0
    denom_safe = denom_safe.where(~zero_denom_mask, pd.Series(np.where(sign_mask, min_denominator, -min_denominator), index=denom_safe.index))
    
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        result = num_series / denom_safe
        result = result.replace([np.inf, -np.inf], fill_value)
        result = result.fillna(fill_value)
    
    return pd.Series(result, index=denom_series.index if hasattr(denom_series, 'index') else None)


def safe_series_operation(series: pd.Series, operation: str = 'value', 
                          index: int = -1, default: float = 0.0) -> float:
    """
    Safely extract a value from a Series with null and Inf checking.
    
    Args:
        series: The pandas Series
        operation: Type of operation ('value', 'mean', 'sum', 'min', 'max')
        index: Index position for 'value' operation
        default: Default value if operation fails
    
    Returns:
        The extracted value or default
    """
    try:
        if series is None or len(series) == 0:
            return default
        
        def _is_valid(val):
            """Check if value is valid (not NaN, not Inf)"""
            if val is None:
                return False
            try:
                if isinstance(val, (float, np.floating)):
                    return not (np.isnan(val) or np.isinf(val))
                return True
            except (TypeError, ValueError):
                return False
        
        if operation == 'value':
            if abs(index) > len(series):
                return default
            val = series.iloc[index]
            if not _is_valid(val):
                return default
            return float(val)
        elif operation == 'mean':
            clean_series = series.replace([np.inf, -np.inf], np.nan).dropna()
            if len(clean_series) == 0:
                return default
            val = clean_series.mean()
            return default if not _is_valid(val) else float(val)
        elif operation == 'sum':
            clean_series = series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            val = clean_series.sum()
            return default if not _is_valid(val) else float(val)
        elif operation == 'min':
            clean_series = series.replace([np.inf, -np.inf], np.nan).dropna()
            if len(clean_series) == 0:
                return default
            val = clean_series.min()
            return default if not _is_valid(val) else float(val)
        elif operation == 'max':
            clean_series = series.replace([np.inf, -np.inf], np.nan).dropna()
            if len(clean_series) == 0:
                return default
            val = clean_series.max()
            return default if not _is_valid(val) else float(val)
        else:
            return default
    except (IndexError, KeyError, TypeError, ValueError):
        return default


def _safe_clip(series: pd.Series, lower: float, upper: float) -> pd.Series:
    """Safely clip series values with Inf handling."""
    result = series.replace([np.inf, -np.inf], np.nan)
    result = result.fillna((lower + upper) / 2)
    return pd.Series(result.clip(lower, upper))


class VolatilityZone(str):
    """Enum-like class untuk ATR volatility zones"""
    EXTREME_LOW = 'extreme_low'
    LOW = 'low'
    NORMAL = 'normal'
    HIGH = 'high'
    EXTREME_HIGH = 'extreme_high'


class DivergenceType(str):
    """Enum-like class untuk RSI divergence types"""
    BULLISH_REGULAR = 'bullish_regular'
    BEARISH_REGULAR = 'bearish_regular'
    BULLISH_HIDDEN = 'bullish_hidden'
    BEARISH_HIDDEN = 'bearish_hidden'
    NONE = 'none'


class IndicatorEngine:
    def __init__(self, config):
        self.config = config
        self.ema_periods = config.EMA_PERIODS
        self.rsi_period = config.RSI_PERIOD
        self.stoch_k_period = config.STOCH_K_PERIOD
        self.stoch_d_period = config.STOCH_D_PERIOD
        self.stoch_smooth_k = config.STOCH_SMOOTH_K
        self.atr_period = config.ATR_PERIOD
        self.macd_fast = config.MACD_FAST
        self.macd_slow = config.MACD_SLOW
        self.macd_signal = config.MACD_SIGNAL
        self._logger = logging.getLogger('indicators')
    
    def _validate_dataframe(self, df: pd.DataFrame, required_columns: Optional[list] = None) -> bool:
        """
        Validate DataFrame has required columns and sufficient data.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
        
        Returns:
            True if valid, False otherwise
        """
        if df is None or not isinstance(df, pd.DataFrame):
            return False
        
        if len(df) == 0:
            return False
        
        if required_columns is None:
            required_columns = ['open', 'high', 'low', 'close']
        
        for col in required_columns:
            if col not in df.columns:
                return False
        
        return True
    
    def _get_column_series(self, df: pd.DataFrame, column: str, fill_value: float = 0.0) -> pd.Series:
        """
        Safely get a column from DataFrame with null and Inf handling.
        
        Args:
            df: Source DataFrame
            column: Column name
            fill_value: Value to fill NaN/Inf with
        
        Returns:
            Validated Series
        """
        if df is None or len(df) == 0:
            return pd.Series([fill_value])
        
        if column not in df.columns:
            return pd.Series([fill_value] * len(df), index=df.index)
        
        result = df[column].replace([np.inf, -np.inf], np.nan).fillna(fill_value)
        return pd.Series(result)
    
    def _create_default_series(self, df: pd.DataFrame, fill_value: float = 0.0) -> pd.Series:
        """Create a default series matching DataFrame length."""
        if df is None or len(df) == 0:
            return pd.Series([fill_value])
        return pd.Series([fill_value] * len(df), index=df.index)
    
    def calculate_ema(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Exponential Moving Average with null handling."""
        if not self._validate_dataframe(df, ['close']):
            return pd.Series([0.0])
        
        if len(df) == 0:
            return pd.Series([0.0])
        
        close = self._get_column_series(df, 'close')
        if len(close) < period:
            return self._create_default_series(df, close.iloc[-1] if len(close) > 0 else 0.0)
        
        with np.errstate(all='ignore'):
            result = close.ewm(span=period, adjust=False).mean()
            result = result.replace([np.inf, -np.inf], np.nan).fillna(close)
        
        return pd.Series(result)
    
    def calculate_rsi(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate RSI with proper null handling for division operations."""
        if not self._validate_dataframe(df, ['close']):
            return pd.Series([50.0])
        
        if len(df) == 0:
            return pd.Series([50.0])
        
        close = self._get_column_series(df, 'close')
        if len(close) < period + 1:
            return self._create_default_series(df, 50.0)
        
        with np.errstate(all='ignore'):
            delta = close.diff()
            delta = delta.fillna(0.0)
            
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta.where(delta < 0, 0.0))
            
            avg_gain = gain.rolling(window=period, min_periods=1).mean()
            avg_loss = loss.rolling(window=period, min_periods=1).mean()
            
            avg_gain = pd.Series(avg_gain, index=df.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            avg_loss = pd.Series(avg_loss, index=df.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
            rsi = pd.Series(50.0, index=df.index)
            
            both_zero = (avg_gain.abs() < 1e-10) & (avg_loss.abs() < 1e-10)
            only_gain = (avg_gain.abs() >= 1e-10) & (avg_loss.abs() < 1e-10)
            only_loss = (avg_gain.abs() < 1e-10) & (avg_loss.abs() >= 1e-10)
            normal = (avg_gain.abs() >= 1e-10) & (avg_loss.abs() >= 1e-10)
            
            rsi = rsi.where(~both_zero, 50.0)
            rsi = rsi.where(~only_gain, 100.0)
            rsi = rsi.where(~only_loss, 0.0)
            
            if normal.any():
                rs_normal = avg_gain[normal] / avg_loss[normal]
                rsi_normal = 100 - (100 / (1 + rs_normal))
                rsi_normal_series = pd.Series(rsi_normal) if not isinstance(rsi_normal, pd.Series) else rsi_normal
                rsi.loc[normal] = rsi_normal_series.values
            
            rsi = rsi.replace([np.inf, -np.inf], np.nan).fillna(50.0)
            rsi = _safe_clip(rsi, 0, 100)
        
        return pd.Series(rsi, index=df.index)
    
    def calculate_stochastic(self, df: pd.DataFrame, k_period: int, d_period: int, smooth_k: int) -> tuple:
        """Calculate Stochastic oscillator with null handling."""
        if not self._validate_dataframe(df, ['high', 'low', 'close']):
            empty_series = pd.Series([50.0])
            return empty_series, empty_series
        
        if len(df) == 0:
            empty_series = pd.Series([50.0])
            return empty_series, empty_series
        
        high = self._get_column_series(df, 'high')
        low = self._get_column_series(df, 'low')
        close = self._get_column_series(df, 'close')
        
        if len(df) < k_period:
            empty_series = self._create_default_series(df, 50.0)
            return empty_series, empty_series
        
        with np.errstate(all='ignore'):
            low_min = low.rolling(window=k_period, min_periods=1).min()
            high_max = high.rolling(window=k_period, min_periods=1).max()
            
            low_min = low_min.replace([np.inf, -np.inf], np.nan).fillna(low)
            high_max = high_max.replace([np.inf, -np.inf], np.nan).fillna(high)
            
            range_diff = high_max - low_min
            
            numerator = close - low_min
            stoch_k = 100 * safe_divide(numerator, range_diff, fill_value=0.5, min_denominator=1e-10)
            stoch_k = _safe_clip(stoch_k, 0, 100)
            
            stoch_k = pd.Series(stoch_k).rolling(window=smooth_k, min_periods=1).mean()
            stoch_k = pd.Series(stoch_k)
            stoch_d = stoch_k.rolling(window=d_period, min_periods=1).mean()
            stoch_d = pd.Series(stoch_d)
            
            stoch_k = stoch_k.replace([np.inf, -np.inf], np.nan).fillna(50.0)
            stoch_d = stoch_d.replace([np.inf, -np.inf], np.nan).fillna(50.0)
        
        return stoch_k, stoch_d
    
    def calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range with null handling."""
        if not self._validate_dataframe(df, ['high', 'low', 'close']):
            return pd.Series([0.0])
        
        if len(df) == 0:
            return pd.Series([0.0])
        
        high = self._get_column_series(df, 'high')
        low = self._get_column_series(df, 'low')
        close = self._get_column_series(df, 'close')
        
        if len(df) < 2:
            return self._create_default_series(df, 0.0)
        
        with np.errstate(all='ignore'):
            prev_close = close.shift(1)
            prev_close = prev_close.fillna(close)
            
            high_low = (high - low).abs()
            high_close = (high - prev_close).abs()
            low_close = (low - prev_close).abs()
            
            high_low = high_low.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            high_close = high_close.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            low_close = low_close.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            tr = pd.Series(tr).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
            atr = tr.rolling(window=period, min_periods=1).mean()
            atr = pd.Series(atr).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        return pd.Series(atr)
    
    def calculate_volume_average(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate volume average with null handling."""
        if not self._validate_dataframe(df, ['volume']):
            return pd.Series([0.0])
        
        if len(df) == 0:
            return pd.Series([0.0])
        
        volume = self._get_column_series(df, 'volume')
        
        if len(volume) < period:
            window_size = max(1, len(volume))
            result = volume.rolling(window=window_size, min_periods=1).mean()
            return pd.Series(result).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        with np.errstate(all='ignore'):
            result = volume.rolling(window=period, min_periods=1).mean()
        
        return pd.Series(result).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    def calculate_twin_range_filter(self, df: pd.DataFrame, period: int = 27, multiplier: float = 2.0) -> tuple:
        """
        Twin Range Filter - Indikator untuk filter trend menggunakan smooth range
        
        Args:
            df: DataFrame dengan kolom OHLC
            period: Periode untuk smoothing (default 27)
            multiplier: Multiplier untuk range calculation (default 2.0)
        
        Returns:
            tuple: (upper_filter, lower_filter, trend_direction)
                   trend_direction: 1 untuk bullish, -1 untuk bearish, 0 untuk neutral
        """
        if not self._validate_dataframe(df, ['high', 'low', 'close']):
            empty = pd.Series([0.0])
            return empty, empty, pd.Series([0])
        
        if len(df) == 0:
            empty = pd.Series([0.0])
            return empty, empty, pd.Series([0])
        
        close = self._get_column_series(df, 'close')
        high = self._get_column_series(df, 'high')
        low = self._get_column_series(df, 'low')
        
        if len(df) < period:
            zeros = self._create_default_series(df, 0.0)
            trend = pd.Series([0] * len(df), index=df.index)
            return zeros, zeros, trend
        
        with np.errstate(all='ignore'):
            range_val = (high - low).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            smooth_range = range_val.ewm(span=period, adjust=False).mean()
            smooth_range = smooth_range.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
            range_filter = smooth_range * multiplier
            
            upper_filter = close + range_filter
            lower_filter = close - range_filter
            
            upper_filter = upper_filter.ewm(span=period, adjust=False).mean()
            upper_filter = upper_filter.replace([np.inf, -np.inf], np.nan).fillna(close)
            lower_filter = lower_filter.ewm(span=period, adjust=False).mean()
            lower_filter = lower_filter.replace([np.inf, -np.inf], np.nan).fillna(close)
        
        trend = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            try:
                close_val = close.iloc[i] if not pd.isna(close.iloc[i]) else 0.0
                upper_prev = upper_filter.iloc[i-1] if not pd.isna(upper_filter.iloc[i-1]) else close_val
                lower_prev = lower_filter.iloc[i-1] if not pd.isna(lower_filter.iloc[i-1]) else close_val
                
                if close_val > upper_prev:
                    trend.iloc[i] = 1
                elif close_val < lower_prev:
                    trend.iloc[i] = -1
                else:
                    trend.iloc[i] = trend.iloc[i-1] if i > 0 else 0
            except (IndexError, KeyError):
                trend.iloc[i] = 0
        
        return upper_filter, lower_filter, trend
    
    def calculate_market_bias_cerebr(self, df: pd.DataFrame, period: int = 60, smoothing: int = 10) -> tuple:
        """
        Market Bias (CEREBR) - Indikator untuk deteksi bias pasar
        
        Args:
            df: DataFrame dengan kolom OHLC
            period: Periode untuk CEREBR calculation (default 60)
            smoothing: Periode smoothing (default 10)
        
        Returns:
            tuple: (cerebr_value, cerebr_signal, bias_direction)
                   cerebr_value: Nilai CEREBR
                   cerebr_signal: Signal line (smoothed)
                   bias_direction: 1 untuk bullish bias, -1 untuk bearish bias, 0 untuk neutral
        """
        if not self._validate_dataframe(df, ['high', 'low', 'close']):
            empty = pd.Series([50.0])
            return empty, empty, pd.Series([0])
        
        if len(df) == 0:
            empty = pd.Series([50.0])
            return empty, empty, pd.Series([0])
        
        close = self._get_column_series(df, 'close')
        high = self._get_column_series(df, 'high')
        low = self._get_column_series(df, 'low')
        
        if len(df) < period:
            default_val = self._create_default_series(df, 50.0)
            trend = pd.Series([0] * len(df), index=df.index)
            return default_val, default_val, trend
        
        with np.errstate(all='ignore'):
            high_period = high.rolling(window=period, min_periods=1).max()
            low_period = low.rolling(window=period, min_periods=1).min()
            
            high_period = high_period.replace([np.inf, -np.inf], np.nan).fillna(high)
            low_period = low_period.replace([np.inf, -np.inf], np.nan).fillna(low)
            
            range_period = high_period - low_period
            
            numerator = close - low_period
            cerebr_raw = safe_divide(numerator, range_period, fill_value=0.5, min_denominator=1e-10) * 100
            cerebr_raw = cerebr_raw.replace([np.inf, -np.inf], np.nan).fillna(50.0)
            
            cerebr_value = cerebr_raw.ewm(span=smoothing, adjust=False).mean()
            cerebr_signal = cerebr_value.ewm(span=smoothing, adjust=False).mean()
            
            cerebr_value = cerebr_value.replace([np.inf, -np.inf], np.nan).fillna(50.0)
            cerebr_signal = cerebr_signal.replace([np.inf, -np.inf], np.nan).fillna(50.0)
        
        bias_direction = pd.Series(0, index=df.index)
        for i in range(len(df)):
            try:
                val = cerebr_value.iloc[i] if not pd.isna(cerebr_value.iloc[i]) else 50.0
                if np.isinf(val):
                    val = 50.0
                if val > 60:
                    bias_direction.iloc[i] = 1
                elif val < 40:
                    bias_direction.iloc[i] = -1
                else:
                    bias_direction.iloc[i] = 0
            except (IndexError, KeyError):
                bias_direction.iloc[i] = 0
        
        return cerebr_value, cerebr_signal, bias_direction
    
    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD with null handling."""
        if not self._validate_dataframe(df, ['close']):
            empty = pd.Series([0.0])
            return empty, empty, empty
        
        if len(df) == 0:
            empty = pd.Series([0.0])
            return empty, empty, empty
        
        close = self._get_column_series(df, 'close')
        
        if len(close) < slow:
            zeros = self._create_default_series(df, 0.0)
            return zeros, zeros, zeros
        
        with np.errstate(all='ignore'):
            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
            
            ema_fast = ema_fast.replace([np.inf, -np.inf], np.nan).fillna(close)
            ema_slow = ema_slow.replace([np.inf, -np.inf], np.nan).fillna(close)
            
            macd_line = ema_fast - ema_slow
            macd_line = macd_line.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
            macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
            macd_signal = macd_signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
            macd_histogram = macd_line - macd_signal
            macd_histogram = macd_histogram.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        return macd_line, macd_signal, macd_histogram
    
    def get_indicators(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Calculate all indicators with comprehensive null handling.
        
        Args:
            df: DataFrame with OHLC data
        
        Returns:
            Dictionary of indicator values or None if insufficient data
        """
        if not self._validate_dataframe(df, ['open', 'high', 'low', 'close']):
            return None
        
        min_required = max(30, max(self.ema_periods + [self.rsi_period, self.stoch_k_period, self.atr_period]) + 10)
        
        if len(df) < min_required:
            return None
        
        indicators = {}
        failed_indicators = []
        
        for period in self.ema_periods:
            try:
                ema_series = self.calculate_ema(df, period)
                indicators[f'ema_{period}'] = safe_series_operation(ema_series, 'value', -1, 0.0)
            except Exception as e:
                self._logger.warning(f"Gagal menghitung EMA periode {period}: {str(e)}")
                indicators[f'ema_{period}'] = 0.0
                failed_indicators.append(f'ema_{period}')
        
        try:
            rsi_series = self.calculate_rsi(df, self.rsi_period)
            indicators['rsi'] = safe_series_operation(rsi_series, 'value', -1, 50.0)
            indicators['rsi_prev'] = safe_series_operation(rsi_series, 'value', -2, 50.0)
            
            try:
                rsi_tail = rsi_series.tail(20).replace([np.inf, -np.inf], np.nan).fillna(50.0).tolist()
                indicators['rsi_history'] = [float(v) if not (pd.isna(v) or np.isinf(v)) else 50.0 for v in rsi_tail]
            except Exception:
                indicators['rsi_history'] = [50.0] * 20
        except Exception as e:
            self._logger.warning(f"Gagal menghitung RSI: {str(e)}")
            indicators['rsi'] = 50.0
            indicators['rsi_prev'] = 50.0
            indicators['rsi_history'] = [50.0] * 20
            failed_indicators.append('rsi')
        
        try:
            stoch_k, stoch_d = self.calculate_stochastic(
                df, self.stoch_k_period, self.stoch_d_period, self.stoch_smooth_k
            )
            indicators['stoch_k'] = safe_series_operation(stoch_k, 'value', -1, 50.0)
            indicators['stoch_d'] = safe_series_operation(stoch_d, 'value', -1, 50.0)
            indicators['stoch_k_prev'] = safe_series_operation(stoch_k, 'value', -2, 50.0)
            indicators['stoch_d_prev'] = safe_series_operation(stoch_d, 'value', -2, 50.0)
        except Exception as e:
            self._logger.warning(f"Gagal menghitung Stochastic: {str(e)}")
            indicators['stoch_k'] = 50.0
            indicators['stoch_d'] = 50.0
            indicators['stoch_k_prev'] = 50.0
            indicators['stoch_d_prev'] = 50.0
            failed_indicators.append('stochastic')
        
        try:
            atr_series = self.calculate_atr(df, self.atr_period)
            indicators['atr'] = safe_series_operation(atr_series, 'value', -1, 0.0)
        except Exception as e:
            self._logger.warning(f"Gagal menghitung ATR: {str(e)}")
            indicators['atr'] = 0.0
            failed_indicators.append('atr')
        
        try:
            macd_line, macd_signal, macd_histogram = self.calculate_macd(
                df, self.macd_fast, self.macd_slow, self.macd_signal
            )
            indicators['macd'] = safe_series_operation(macd_line, 'value', -1, 0.0)
            indicators['macd_signal'] = safe_series_operation(macd_signal, 'value', -1, 0.0)
            indicators['macd_histogram'] = safe_series_operation(macd_histogram, 'value', -1, 0.0)
            indicators['macd_prev'] = safe_series_operation(macd_line, 'value', -2, 0.0)
            indicators['macd_signal_prev'] = safe_series_operation(macd_signal, 'value', -2, 0.0)
        except Exception as e:
            self._logger.warning(f"Gagal menghitung MACD: {str(e)}")
            indicators['macd'] = 0.0
            indicators['macd_signal'] = 0.0
            indicators['macd_histogram'] = 0.0
            indicators['macd_prev'] = 0.0
            indicators['macd_signal_prev'] = 0.0
            failed_indicators.append('macd')
        
        try:
            volume_series = self._get_column_series(df, 'volume')
            indicators['volume'] = safe_series_operation(volume_series, 'value', -1, 0.0)
            vol_avg_series = self.calculate_volume_average(df)
            indicators['volume_avg'] = safe_series_operation(vol_avg_series, 'value', -1, 0.0)
        except Exception as e:
            self._logger.warning(f"Gagal menghitung Volume: {str(e)}")
            indicators['volume'] = 0.0
            indicators['volume_avg'] = 0.0
            failed_indicators.append('volume')
        
        try:
            trf_upper, trf_lower, trf_trend = self.calculate_twin_range_filter(df, period=27, multiplier=2.0)
            indicators['trf_upper'] = safe_series_operation(trf_upper, 'value', -1, 0.0)
            indicators['trf_lower'] = safe_series_operation(trf_lower, 'value', -1, 0.0)
            indicators['trf_trend'] = safe_series_operation(trf_trend, 'value', -1, 0)
            indicators['trf_trend_prev'] = safe_series_operation(trf_trend, 'value', -2, 0)
        except Exception as e:
            self._logger.warning(f"Gagal menghitung Twin Range Filter: {str(e)}")
            indicators['trf_upper'] = 0.0
            indicators['trf_lower'] = 0.0
            indicators['trf_trend'] = 0
            indicators['trf_trend_prev'] = 0
            failed_indicators.append('trf')
        
        try:
            cerebr_value, cerebr_signal, cerebr_bias = self.calculate_market_bias_cerebr(df, period=60, smoothing=10)
            indicators['cerebr_value'] = safe_series_operation(cerebr_value, 'value', -1, 50.0)
            indicators['cerebr_signal'] = safe_series_operation(cerebr_signal, 'value', -1, 50.0)
            indicators['cerebr_bias'] = safe_series_operation(cerebr_bias, 'value', -1, 0)
            indicators['cerebr_bias_prev'] = safe_series_operation(cerebr_bias, 'value', -2, 0)
        except Exception as e:
            self._logger.warning(f"Gagal menghitung CEREBR: {str(e)}")
            indicators['cerebr_value'] = 50.0
            indicators['cerebr_signal'] = 50.0
            indicators['cerebr_bias'] = 0
            indicators['cerebr_bias_prev'] = 0
            failed_indicators.append('cerebr')
        
        try:
            adx_period = getattr(self.config, 'ADX_PERIOD', 14)
            adx_series, plus_di_series, minus_di_series = self.calculate_adx(df, period=adx_period)
            indicators['adx'] = safe_series_operation(adx_series, 'value', -1, 0.0)
            indicators['adx_prev'] = safe_series_operation(adx_series, 'value', -2, 0.0)
            indicators['plus_di'] = safe_series_operation(plus_di_series, 'value', -1, 0.0)
            indicators['minus_di'] = safe_series_operation(minus_di_series, 'value', -1, 0.0)
        except Exception as e:
            self._logger.warning(f"Gagal menghitung ADX: {str(e)}")
            indicators['adx'] = 0.0
            indicators['adx_prev'] = 0.0
            indicators['plus_di'] = 0.0
            indicators['minus_di'] = 0.0
            failed_indicators.append('adx')
        
        try:
            ema_slope_period = self.ema_periods[1] if len(self.ema_periods) > 1 else 20
            ema_slope = self.calculate_ema_slope(df, period=ema_slope_period, lookback=3)
            indicators['ema_slope'] = safe_series_operation(ema_slope, 'value', -1, 0.0)
            indicators['ema_slope_prev'] = safe_series_operation(ema_slope, 'value', -2, 0.0)
        except Exception as e:
            self._logger.warning(f"Gagal menghitung EMA slope: {str(e)}")
            indicators['ema_slope'] = 0.0
            indicators['ema_slope_prev'] = 0.0
            failed_indicators.append('ema_slope')
        
        try:
            close_series = self._get_column_series(df, 'close')
            high_series = self._get_column_series(df, 'high')
            low_series = self._get_column_series(df, 'low')
            
            indicators['close'] = safe_series_operation(close_series, 'value', -1, 0.0)
            indicators['high'] = safe_series_operation(high_series, 'value', -1, 0.0)
            indicators['low'] = safe_series_operation(low_series, 'value', -1, 0.0)
        except Exception as e:
            self._logger.warning(f"Gagal mendapatkan data OHLC: {str(e)}")
            indicators['close'] = 0.0
            indicators['high'] = 0.0
            indicators['low'] = 0.0
            failed_indicators.append('ohlc')
        
        if failed_indicators:
            self._logger.warning(f"Indikator yang gagal dihitung: {', '.join(failed_indicators)}")
        
        return indicators
    
    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        VWAP = cumsum(typical_price * volume) / cumsum(volume)
        typical_price = (high + low + close) / 3
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            pd.Series: VWAP values
        """
        if not self._validate_dataframe(df, ['high', 'low', 'close', 'volume']):
            return pd.Series([0.0])
        
        if len(df) == 0:
            return pd.Series([0.0])
        
        high = self._get_column_series(df, 'high')
        low = self._get_column_series(df, 'low')
        close = self._get_column_series(df, 'close')
        volume = self._get_column_series(df, 'volume')
        
        if len(df) < 1:
            return pd.Series([0.0])
        
        with np.errstate(all='ignore'):
            typical_price = (high + low + close) / 3
            typical_price = typical_price.replace([np.inf, -np.inf], np.nan).fillna(close)
            
            tp_volume = typical_price * volume
            tp_volume = tp_volume.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        has_date_index = False
        try:
            if isinstance(df.index, pd.DatetimeIndex):
                has_date_index = True
            elif 'time' in df.columns:
                df_time = pd.to_datetime(df['time'], errors='coerce')
                if not df_time.isna().all():
                    has_date_index = True
        except Exception:
            has_date_index = False
        
        if has_date_index:
            try:
                if isinstance(df.index, pd.DatetimeIndex):
                    date_series = pd.Series(df.index).dt.date.values
                else:
                    date_series = pd.to_datetime(df['time']).dt.date.values
                
                vwap = pd.Series(index=df.index, dtype=float)
                
                for date in pd.Series(date_series).unique():
                    mask = date_series == date
                    if isinstance(mask, np.ndarray):
                        mask = pd.Series(mask, index=df.index)
                    
                    cum_tp_vol = tp_volume[mask].cumsum()
                    cum_vol = volume[mask].cumsum()
                    
                    default_price = typical_price[mask].iloc[-1] if len(typical_price[mask]) > 0 else 0.0
                    daily_vwap = safe_divide(cum_tp_vol, cum_vol, fill_value=default_price, min_denominator=1e-10)
                    vwap.loc[mask] = daily_vwap.values
                
                vwap = vwap.replace([np.inf, -np.inf], np.nan).fillna(typical_price)
                return pd.Series(vwap)
            except Exception:
                pass
        
        rolling_period = min(20, len(df))
        if rolling_period < 1:
            rolling_period = 1
        
        with np.errstate(all='ignore'):
            cum_tp_vol = tp_volume.rolling(window=rolling_period, min_periods=1).sum()
            cum_vol = volume.rolling(window=rolling_period, min_periods=1).sum()
            
            vwap = safe_divide(cum_tp_vol, cum_vol, fill_value=0.0, min_denominator=1e-10)
            vwap = vwap.replace([np.inf, -np.inf], np.nan).fillna(typical_price)
        
        return pd.Series(vwap)
    
    def detect_candlestick_patterns(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Detect candlestick patterns from the last candle.
        
        Patterns detected:
        - Pinbar (bullish/bearish): small body (<30% range), long wick (>60% range)
        - Hammer/Inverted Hammer: small body, lower/upper wick > 2x body
        - Engulfing (bullish/bearish): current candle engulfs previous
        
        Args:
            df: DataFrame with OHLC data
        
        Returns:
            Dict with pattern detection results
        """
        default_result = {
            'bullish_pinbar': False,
            'bearish_pinbar': False,
            'hammer': False,
            'inverted_hammer': False,
            'bullish_engulfing': False,
            'bearish_engulfing': False
        }
        
        if not self._validate_dataframe(df, ['open', 'high', 'low', 'close']):
            return default_result
        
        if len(df) < 2:
            return default_result
        
        try:
            def _safe_float(val, default=0.0):
                if val is None or pd.isna(val) or np.isinf(val):
                    return default
                return float(val)
            
            open_curr = _safe_float(df['open'].iloc[-1])
            high_curr = _safe_float(df['high'].iloc[-1])
            low_curr = _safe_float(df['low'].iloc[-1])
            close_curr = _safe_float(df['close'].iloc[-1])
            
            open_prev = _safe_float(df['open'].iloc[-2])
            high_prev = _safe_float(df['high'].iloc[-2])
            low_prev = _safe_float(df['low'].iloc[-2])
            close_prev = _safe_float(df['close'].iloc[-2])
        except (IndexError, KeyError, TypeError):
            return default_result
        
        total_range = high_curr - low_curr
        if total_range <= 1e-10:
            total_range = 1e-10
        
        body = abs(close_curr - open_curr)
        body_pct = body / total_range
        
        is_bullish = close_curr > open_curr
        body_top = max(open_curr, close_curr)
        body_bottom = min(open_curr, close_curr)
        upper_wick = high_curr - body_top
        lower_wick = body_bottom - low_curr
        
        result = default_result.copy()
        
        if body_pct < 0.30:
            upper_wick_pct = upper_wick / total_range
            lower_wick_pct = lower_wick / total_range
            
            pinbar_threshold = 0.6667
            if lower_wick_pct >= pinbar_threshold:
                result['bullish_pinbar'] = True
            if upper_wick_pct >= pinbar_threshold:
                result['bearish_pinbar'] = True
        
        body_size = body if body > 1e-10 else 1e-10
        
        if body_pct < 0.35 and lower_wick > 2 * body_size and upper_wick < body_size:
            result['hammer'] = True
        
        if body_pct < 0.35 and upper_wick > 2 * body_size and lower_wick < body_size:
            result['inverted_hammer'] = True
        
        prev_is_bearish = close_prev < open_prev
        prev_is_bullish = close_prev > open_prev
        
        if is_bullish and prev_is_bearish:
            if close_curr > open_prev and open_curr < close_prev:
                result['bullish_engulfing'] = True
        
        if not is_bullish and prev_is_bullish:
            if close_curr < open_prev and open_curr > close_prev:
                result['bearish_engulfing'] = True
        
        return result
    
    def calculate_micro_support_resistance(self, df: pd.DataFrame, lookback: int = 20) -> Dict:
        """
        Calculate micro support and resistance levels from swing highs/lows.
        
        Args:
            df: DataFrame with OHLC data
            lookback: Number of candles to look back for swing detection
        
        Returns:
            Dict with support/resistance levels:
            - nearest_support: Closest support below current price
            - nearest_resistance: Closest resistance above current price
            - support_levels: List of support levels
            - resistance_levels: List of resistance levels
        """
        default_result = {
            'nearest_support': 0.0,
            'nearest_resistance': 0.0,
            'support_levels': [],
            'resistance_levels': []
        }
        
        if not self._validate_dataframe(df, ['high', 'low', 'close']):
            return default_result
        
        if len(df) < 5:
            return default_result
        
        try:
            high = self._get_column_series(df, 'high')
            low = self._get_column_series(df, 'low')
            close = self._get_column_series(df, 'close')
            
            current_price = safe_series_operation(close, 'value', -1, 0.0)
            if current_price == 0.0:
                return default_result
            
            actual_lookback = min(lookback, len(df))
            df_subset = df.tail(actual_lookback)
            high_subset = high.tail(actual_lookback)
            low_subset = low.tail(actual_lookback)
            
            swing_highs = []
            swing_lows = []
            
            def _safe_float(series, idx, default=0.0):
                try:
                    val = series.iloc[idx]
                    if pd.isna(val) or np.isinf(val):
                        return default
                    return float(val)
                except (IndexError, KeyError):
                    return default
            
            for i in range(2, len(df_subset) - 2):
                high_val = _safe_float(high_subset, i)
                high_prev1 = _safe_float(high_subset, i-1)
                high_prev2 = _safe_float(high_subset, i-2)
                high_next1 = _safe_float(high_subset, i+1)
                high_next2 = _safe_float(high_subset, i+2)
                
                if high_val > high_prev1 and high_val > high_prev2 and high_val > high_next1 and high_val > high_next2:
                    swing_highs.append(high_val)
                
                low_val = _safe_float(low_subset, i)
                low_prev1 = _safe_float(low_subset, i-1)
                low_prev2 = _safe_float(low_subset, i-2)
                low_next1 = _safe_float(low_subset, i+1)
                low_next2 = _safe_float(low_subset, i+2)
                
                if low_val < low_prev1 and low_val < low_prev2 and low_val < low_next1 and low_val < low_next2:
                    swing_lows.append(low_val)
            
            if not swing_lows:
                min_low = safe_series_operation(low_subset, 'min', default=current_price * 0.99)
                swing_lows.append(min_low)
            if not swing_highs:
                max_high = safe_series_operation(high_subset, 'max', default=current_price * 1.01)
                swing_highs.append(max_high)
            
            swing_highs = sorted(list(set([h for h in swing_highs if not (pd.isna(h) or np.isinf(h))])))
            swing_lows = sorted(list(set([l for l in swing_lows if not (pd.isna(l) or np.isinf(l))])))
            
            resistance_levels = [h for h in swing_highs if h > current_price]
            support_levels = [l for l in swing_lows if l < current_price]
            
            nearest_resistance = min(resistance_levels) if resistance_levels else (max(swing_highs) if swing_highs else current_price * 1.01)
            nearest_support = max(support_levels) if support_levels else (min(swing_lows) if swing_lows else current_price * 0.99)
            
            return {
                'nearest_support': float(nearest_support),
                'nearest_resistance': float(nearest_resistance),
                'support_levels': [float(s) for s in sorted(support_levels, reverse=True)[:5]],
                'resistance_levels': [float(r) for r in sorted(resistance_levels)[:5]]
            }
            
        except Exception as e:
            self._logger.warning(f"Gagal menghitung support/resistance: {str(e)}")
            return default_result
    
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> tuple:
        """
        Calculate ADX (Average Directional Index) untuk mengukur kekuatan tren.
        
        ADX mengukur kekuatan tren tanpa memperhatikan arah tren.
        - ADX > 25: Tren kuat
        - ADX > 20: Tren cukup kuat
        - ADX < 20: Sideways/ranging market
        
        Args:
            df: DataFrame with OHLC data
            period: Period for ADX calculation (default: 14)
        
        Returns:
            tuple: (adx_series, plus_di_series, minus_di_series)
        """
        if not self._validate_dataframe(df, ['high', 'low', 'close']):
            empty = pd.Series([0.0])
            return empty, empty, empty
        
        if len(df) < period + 1:
            empty = self._create_default_series(df, 0.0)
            return empty, empty, empty
        
        high = self._get_column_series(df, 'high')
        low = self._get_column_series(df, 'low')
        close = self._get_column_series(df, 'close')
        
        try:
            with np.errstate(all='ignore'):
                plus_dm = high.diff()
                minus_dm = low.diff().abs()
                
                plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
                minus_dm = minus_dm.where((minus_dm > plus_dm.shift(0).abs()) & (low.diff() < 0), 0.0)
                
                plus_dm = plus_dm.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                minus_dm = minus_dm.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                
                tr1 = (high - low).abs()
                tr2 = (high - close.shift(1)).abs()
                tr3 = (low - close.shift(1)).abs()
                
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                tr = tr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                
                smoothed_plus_dm = plus_dm.ewm(span=period, adjust=False).mean()
                smoothed_minus_dm = minus_dm.ewm(span=period, adjust=False).mean()
                smoothed_tr = tr.ewm(span=period, adjust=False).mean()
                
                smoothed_tr_safe = smoothed_tr.replace(0, 1e-10)
                plus_di = 100 * safe_divide(smoothed_plus_dm, smoothed_tr_safe, fill_value=0.0)
                minus_di = 100 * safe_divide(smoothed_minus_dm, smoothed_tr_safe, fill_value=0.0)
                
                di_sum = plus_di + minus_di
                di_sum_safe = di_sum.replace(0, 1e-10)
                di_diff = (plus_di - minus_di).abs()
                dx = 100 * safe_divide(di_diff, di_sum_safe, fill_value=0.0)
                
                adx = dx.ewm(span=period, adjust=False).mean()
                
                adx = _safe_clip(adx.replace([np.inf, -np.inf], np.nan).fillna(0.0), 0, 100)
                plus_di = _safe_clip(plus_di.replace([np.inf, -np.inf], np.nan).fillna(0.0), 0, 100)
                minus_di = _safe_clip(minus_di.replace([np.inf, -np.inf], np.nan).fillna(0.0), 0, 100)
            
            return pd.Series(adx, index=df.index), pd.Series(plus_di, index=df.index), pd.Series(minus_di, index=df.index)
            
        except Exception as e:
            self._logger.warning(f"Gagal menghitung ADX: {str(e)}")
            empty = self._create_default_series(df, 0.0)
            return empty, empty, empty
    
    def calculate_ema_slope(self, df: pd.DataFrame, period: int = 21, lookback: int = 3) -> pd.Series:
        """
        Calculate EMA slope untuk mendeteksi arah kemiringan EMA.
        
        Slope positif = EMA menukik ke atas (bullish)
        Slope negatif = EMA menukik ke bawah (bearish)
        Slope mendekati 0 = EMA flat (sideways)
        
        Args:
            df: DataFrame with OHLC data
            period: EMA period
            lookback: Number of candles to calculate slope over
        
        Returns:
            pd.Series: EMA slope values (normalized as percentage)
        """
        if not self._validate_dataframe(df, ['close']):
            return pd.Series([0.0])
        
        if len(df) < period + lookback:
            return self._create_default_series(df, 0.0)
        
        try:
            ema = self.calculate_ema(df, period)
            
            with np.errstate(all='ignore'):
                ema_diff = ema.diff(lookback)
                
                ema_safe = ema.replace(0, 1e-10)
                slope_pct = (ema_diff / ema_safe) * 100
                
                slope_pct = slope_pct.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
            return pd.Series(slope_pct, index=df.index)
            
        except Exception as e:
            self._logger.warning(f"Gagal menghitung EMA slope: {str(e)}")
            return self._create_default_series(df, 0.0)
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> Dict:
        """
        Calculate Bollinger Bands untuk volatility analysis.
        
        Args:
            df: DataFrame with OHLC data
            period: Period for SMA calculation (default: 20)
            std_dev: Number of standard deviations (default: 2.0)
        
        Returns:
            Dict with:
            - upper: Upper band series
            - middle: Middle band (SMA) series
            - lower: Lower band series
            - width: Band width as percentage of middle
            - squeeze: Boolean indicating squeeze condition
        """
        default_result = {
            'upper': pd.Series([0.0]),
            'middle': pd.Series([0.0]),
            'lower': pd.Series([0.0]),
            'width': pd.Series([0.0]),
            'squeeze': False,
            'width_pct': 0.0
        }
        
        if not self._validate_dataframe(df, ['close']):
            return default_result
        
        if len(df) < period:
            return default_result
        
        try:
            close = self._get_column_series(df, 'close')
            
            with np.errstate(all='ignore'):
                middle = pd.Series(close.rolling(window=period, min_periods=1).mean())
                std = pd.Series(close.rolling(window=period, min_periods=1).std())
                
                middle = middle.replace([np.inf, -np.inf], np.nan).fillna(close)
                std = std.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                
                upper = middle + (std * std_dev)
                lower = middle - (std * std_dev)
                
                width = upper - lower
                middle_safe = middle.replace(0, 1e-10)
                width_pct = (width / middle_safe) * 100
                width_pct = width_pct.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                
                current_width_pct = safe_series_operation(width_pct, 'value', -1, 0.0)
                
                if len(df) >= period * 2:
                    avg_width_pct = width_pct.tail(period * 2).mean()
                    if pd.isna(avg_width_pct) or np.isinf(avg_width_pct):
                        avg_width_pct = current_width_pct
                else:
                    avg_width_pct = current_width_pct
                
                squeeze = current_width_pct < (avg_width_pct * 0.75) if avg_width_pct > 0 else False
            
            return {
                'upper': pd.Series(upper, index=df.index),
                'middle': pd.Series(middle, index=df.index),
                'lower': pd.Series(lower, index=df.index),
                'width': pd.Series(width, index=df.index),
                'squeeze': bool(squeeze),
                'width_pct': float(current_width_pct)
            }
            
        except Exception as e:
            self._logger.warning(f"Gagal menghitung Bollinger Bands: {str(e)}")
            return default_result
    
    def calculate_price_momentum(self, df: pd.DataFrame, period: int = 10) -> Dict:
        """
        Calculate price momentum indicators.
        
        Args:
            df: DataFrame with OHLC data
            period: Period for momentum calculation
        
        Returns:
            Dict with momentum analysis:
            - momentum: Price change over period
            - momentum_pct: Percentage change
            - direction: 'bullish', 'bearish', or 'neutral'
            - strength: 'strong', 'moderate', or 'weak'
        """
        default_result = {
            'momentum': 0.0,
            'momentum_pct': 0.0,
            'direction': 'neutral',
            'strength': 'weak'
        }
        
        if not self._validate_dataframe(df, ['close']):
            return default_result
        
        if len(df) < period + 1:
            return default_result
        
        try:
            close = self._get_column_series(df, 'close')
            
            current_close = safe_series_operation(close, 'value', -1, 0.0)
            period_ago_close = safe_series_operation(close, 'value', -(period + 1), current_close)
            
            if period_ago_close == 0:
                period_ago_close = 1e-10
            
            momentum = current_close - period_ago_close
            momentum_pct = (momentum / period_ago_close) * 100
            
            if np.isnan(momentum_pct) or np.isinf(momentum_pct):
                momentum_pct = 0.0
            
            if momentum_pct > 0.1:
                direction = 'bullish'
            elif momentum_pct < -0.1:
                direction = 'bearish'
            else:
                direction = 'neutral'
            
            abs_momentum = abs(momentum_pct)
            if abs_momentum > 0.5:
                strength = 'strong'
            elif abs_momentum > 0.2:
                strength = 'moderate'
            else:
                strength = 'weak'
            
            return {
                'momentum': float(momentum),
                'momentum_pct': float(momentum_pct),
                'direction': direction,
                'strength': strength
            }
            
        except Exception as e:
            self._logger.warning(f"Gagal menghitung price momentum: {str(e)}")
            return default_result
    
    def calculate_volume_confirmation(self, df: pd.DataFrame, period: int = 10) -> Dict:
        """
        Calculate volume confirmation indicators.
        
        Args:
            df: DataFrame with volume data
            period: Period for volume average calculation (default: 10)
        
        Returns:
            Dict with volume analysis:
            - volume_current: Current volume
            - volume_avg: Average volume over period
            - is_volume_strong: True if current volume > average
            - volume_ratio: Current volume / Average volume
        """
        default_result = {
            'volume_current': 0.0,
            'volume_avg': 0.0,
            'is_volume_strong': False,
            'volume_ratio': 1.0
        }
        
        if not self._validate_dataframe(df, ['volume']):
            return default_result
        
        if len(df) < 1:
            return default_result
        
        try:
            volume = self._get_column_series(df, 'volume')
            
            current_volume = safe_series_operation(volume, 'value', -1, 0.0)
            
            actual_period = max(1, min(period, len(df)))
            
            with np.errstate(all='ignore'):
                volume_avg_series = pd.Series(volume.rolling(window=actual_period, min_periods=1).mean())
                volume_avg = safe_series_operation(volume_avg_series, 'value', -1, 0.0)
            
            if volume_avg > 1e-10:
                volume_ratio = current_volume / volume_avg
            else:
                volume_ratio = 1.0
            
            if np.isnan(volume_ratio) or np.isinf(volume_ratio):
                volume_ratio = 1.0
            
            is_volume_strong = current_volume > volume_avg
            
            return {
                'volume_current': float(current_volume),
                'volume_avg': float(volume_avg),
                'is_volume_strong': bool(is_volume_strong),
                'volume_ratio': float(volume_ratio)
            }
            
        except Exception as e:
            self._logger.warning(f"Gagal menghitung volume confirmation: {str(e)}")
            return default_result
    
    def calculate_smoothed_indicator(self, series: pd.Series, smoothing_type: str = 'ema', 
                                       period: int = 5) -> pd.Series:
        """
        Apply smoothing to any indicator series.
        
        Args:
            series: Input indicator series to smooth
            smoothing_type: Type of smoothing ('ema', 'sma', 'wma', 'hull')
            period: Smoothing period
        
        Returns:
            Smoothed series
        """
        if series is None or len(series) == 0:
            return pd.Series([0.0])
        
        try:
            series = series.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0.0)
            
            if smoothing_type == 'ema':
                smoothed = series.ewm(span=period, adjust=False).mean()
            elif smoothing_type == 'sma':
                smoothed = series.rolling(window=period, min_periods=1).mean()
            elif smoothing_type == 'wma':
                weights = np.arange(1, period + 1)
                smoothed = series.rolling(window=period, min_periods=1).apply(
                    lambda x: np.dot(x, weights[-len(x):]) / weights[-len(x):].sum(), raw=True
                )
            elif smoothing_type == 'hull':
                half_period = max(1, period // 2)
                sqrt_period = max(1, int(np.sqrt(period)))
                series_pd = pd.Series(series) if not isinstance(series, pd.Series) else series
                wma1 = series_pd.rolling(window=half_period, min_periods=1).mean()
                wma2 = series_pd.rolling(window=period, min_periods=1).mean()
                raw_hull = pd.Series(2 * wma1 - wma2)
                smoothed = raw_hull.rolling(window=sqrt_period, min_periods=1).mean()
            else:
                smoothed = series.ewm(span=period, adjust=False).mean()
            
            smoothed_series = pd.Series(smoothed) if not isinstance(smoothed, pd.Series) else smoothed
            return smoothed_series.replace([np.inf, -np.inf], np.nan).fillna(series)
            
        except Exception as e:
            self._logger.warning(f"Gagal smoothing indicator: {str(e)}")
            return series
    
    def detect_rsi_divergence(self, df: pd.DataFrame, rsi_period: int = 14, 
                               lookback: int = 20) -> Dict:
        """
        Detect RSI divergence (bullish and bearish, regular and hidden).
        
        Regular Bullish: Price makes lower low, RSI makes higher low
        Regular Bearish: Price makes higher high, RSI makes lower high
        Hidden Bullish: Price makes higher low, RSI makes lower low (trend continuation)
        Hidden Bearish: Price makes lower high, RSI makes higher high (trend continuation)
        
        Args:
            df: DataFrame with OHLC data
            rsi_period: Period for RSI calculation
            lookback: Number of candles to look back for divergence
        
        Returns:
            Dict with divergence analysis:
            - divergence_type: Type of divergence detected
            - strength: Strength of divergence (weak, moderate, strong)
            - price_swing_low: Price at swing low
            - price_swing_high: Price at swing high
            - rsi_swing_low: RSI at swing low
            - rsi_swing_high: RSI at swing high
            - is_valid: Whether divergence is valid for trading
        """
        default_result = {
            'divergence_type': DivergenceType.NONE,
            'strength': 'none',
            'price_swing_low': 0.0,
            'price_swing_high': 0.0,
            'rsi_swing_low': 50.0,
            'rsi_swing_high': 50.0,
            'is_valid': False,
            'description': 'No divergence detected'
        }
        
        if not self._validate_dataframe(df, ['close', 'high', 'low']):
            return default_result
        
        if len(df) < lookback + rsi_period:
            return default_result
        
        try:
            rsi = self.calculate_rsi(df, rsi_period)
            close = self._get_column_series(df, 'close')
            high = self._get_column_series(df, 'high')
            low = self._get_column_series(df, 'low')
            
            df_lookback = df.tail(lookback)
            rsi_lookback = rsi.tail(lookback)
            close_lookback = close.tail(lookback)
            low_lookback = low.tail(lookback)
            high_lookback = high.tail(lookback)
            
            swing_lows = []
            swing_highs = []
            
            min_pivot_distance = 5
            
            for i in range(2, len(df_lookback) - 2):
                low_val = float(low_lookback.iloc[i])
                rsi_val_low = float(rsi_lookback.iloc[i])
                
                is_swing_low = (low_val < low_lookback.iloc[i-1] and low_val < low_lookback.iloc[i-2] and
                               low_val < low_lookback.iloc[i+1] and low_val < low_lookback.iloc[i+2])
                
                if is_swing_low:
                    if not swing_lows or (i - swing_lows[-1][0]) >= min_pivot_distance:
                        swing_lows.append((i, low_val, rsi_val_low))
                    elif low_val < swing_lows[-1][1]:
                        swing_lows[-1] = (i, low_val, rsi_val_low)
            
            for i in range(2, len(df_lookback) - 2):
                high_val = float(high_lookback.iloc[i])
                rsi_val_high = float(rsi_lookback.iloc[i])
                
                is_swing_high = (high_val > high_lookback.iloc[i-1] and high_val > high_lookback.iloc[i-2] and
                                high_val > high_lookback.iloc[i+1] and high_val > high_lookback.iloc[i+2])
                
                if is_swing_high:
                    if not swing_highs or (i - swing_highs[-1][0]) >= min_pivot_distance:
                        swing_highs.append((i, high_val, rsi_val_high))
                    elif high_val > swing_highs[-1][1]:
                        swing_highs[-1] = (i, high_val, rsi_val_high)
            
            if len(swing_lows) >= 2:
                prev_low_idx, prev_low_price, prev_low_rsi = swing_lows[-2]
                curr_low_idx, curr_low_price, curr_low_rsi = swing_lows[-1]
                
                if curr_low_price < prev_low_price and curr_low_rsi > prev_low_rsi:
                    rsi_diff = curr_low_rsi - prev_low_rsi
                    strength = 'strong' if rsi_diff > 10 else 'moderate' if rsi_diff > 5 else 'weak'
                    return {
                        'divergence_type': DivergenceType.BULLISH_REGULAR,
                        'strength': strength,
                        'price_swing_low': curr_low_price,
                        'price_swing_high': 0.0,
                        'rsi_swing_low': curr_low_rsi,
                        'rsi_swing_high': 0.0,
                        'is_valid': rsi_diff > 3,
                        'description': f'Regular Bullish Divergence: Price lower low, RSI higher low (diff: {rsi_diff:.1f})'
                    }
                
                if curr_low_price > prev_low_price and curr_low_rsi < prev_low_rsi:
                    rsi_diff = prev_low_rsi - curr_low_rsi
                    strength = 'strong' if rsi_diff > 10 else 'moderate' if rsi_diff > 5 else 'weak'
                    return {
                        'divergence_type': DivergenceType.BULLISH_HIDDEN,
                        'strength': strength,
                        'price_swing_low': curr_low_price,
                        'price_swing_high': 0.0,
                        'rsi_swing_low': curr_low_rsi,
                        'rsi_swing_high': 0.0,
                        'is_valid': rsi_diff > 3,
                        'description': f'Hidden Bullish Divergence: Price higher low, RSI lower low (diff: {rsi_diff:.1f})'
                    }
            
            if len(swing_highs) >= 2:
                prev_high_idx, prev_high_price, prev_high_rsi = swing_highs[-2]
                curr_high_idx, curr_high_price, curr_high_rsi = swing_highs[-1]
                
                if curr_high_price > prev_high_price and curr_high_rsi < prev_high_rsi:
                    rsi_diff = prev_high_rsi - curr_high_rsi
                    strength = 'strong' if rsi_diff > 10 else 'moderate' if rsi_diff > 5 else 'weak'
                    return {
                        'divergence_type': DivergenceType.BEARISH_REGULAR,
                        'strength': strength,
                        'price_swing_low': 0.0,
                        'price_swing_high': curr_high_price,
                        'rsi_swing_low': 0.0,
                        'rsi_swing_high': curr_high_rsi,
                        'is_valid': rsi_diff > 3,
                        'description': f'Regular Bearish Divergence: Price higher high, RSI lower high (diff: {rsi_diff:.1f})'
                    }
                
                if curr_high_price < prev_high_price and curr_high_rsi > prev_high_rsi:
                    rsi_diff = curr_high_rsi - prev_high_rsi
                    strength = 'strong' if rsi_diff > 10 else 'moderate' if rsi_diff > 5 else 'weak'
                    return {
                        'divergence_type': DivergenceType.BEARISH_HIDDEN,
                        'strength': strength,
                        'price_swing_low': 0.0,
                        'price_swing_high': curr_high_price,
                        'rsi_swing_low': 0.0,
                        'rsi_swing_high': curr_high_rsi,
                        'is_valid': rsi_diff > 3,
                        'description': f'Hidden Bearish Divergence: Price lower high, RSI higher high (diff: {rsi_diff:.1f})'
                    }
            
            return default_result
            
        except Exception as e:
            self._logger.warning(f"Gagal mendeteksi RSI divergence: {str(e)}")
            return default_result
    
    def calculate_atr_volatility_zones(self, df: pd.DataFrame, atr_period: int = 14, 
                                         zone_lookback: int = 100) -> Dict:
        """
        Calculate ATR-based volatility zones for market condition assessment.
        
        Zones are determined by comparing current ATR to historical ATR:
        - Extreme Low: ATR < 20th percentile
        - Low: ATR < 40th percentile
        - Normal: ATR between 40th-60th percentile
        - High: ATR > 60th percentile
        - Extreme High: ATR > 80th percentile
        
        Args:
            df: DataFrame with OHLC data
            atr_period: Period for ATR calculation
            zone_lookback: Number of candles for percentile calculation
        
        Returns:
            Dict with volatility zone analysis:
            - zone: Current volatility zone
            - atr_current: Current ATR value
            - atr_avg: Average ATR over lookback
            - atr_percentile: Current ATR percentile rank
            - zone_threshold_low: Lower threshold for current zone
            - zone_threshold_high: Upper threshold for current zone
            - volatility_expanding: True if volatility is increasing
            - volatility_contracting: True if volatility is decreasing
            - recommended_sl_multiplier: Suggested SL multiplier based on zone
            - recommended_tp_multiplier: Suggested TP multiplier based on zone
        """
        default_result = {
            'zone': VolatilityZone.NORMAL,
            'atr_current': 0.0,
            'atr_avg': 0.0,
            'atr_percentile': 50.0,
            'zone_threshold_low': 0.0,
            'zone_threshold_high': 0.0,
            'volatility_expanding': False,
            'volatility_contracting': False,
            'recommended_sl_multiplier': 1.0,
            'recommended_tp_multiplier': 1.0,
            'description': 'Normal volatility conditions'
        }
        
        if not self._validate_dataframe(df, ['high', 'low', 'close']):
            return default_result
        
        if len(df) < atr_period + 5:
            return default_result
        
        try:
            atr = self.calculate_atr(df, atr_period)
            
            current_atr = safe_series_operation(atr, 'value', -1, 0.0)
            
            if current_atr == 0.0:
                return default_result
            
            actual_lookback = min(zone_lookback, len(atr))
            atr_history = atr.tail(actual_lookback)
            
            atr_avg = safe_series_operation(atr_history, 'mean', default=current_atr)
            
            atr_sorted = sorted(atr_history.dropna().values)
            if len(atr_sorted) > 0:
                rank = sum(1 for x in atr_sorted if x <= current_atr)
                atr_percentile = (rank / len(atr_sorted)) * 100
            else:
                atr_percentile = 50.0
            
            p20 = np.percentile(atr_sorted, 20) if len(atr_sorted) > 0 else current_atr * 0.5
            p40 = np.percentile(atr_sorted, 40) if len(atr_sorted) > 0 else current_atr * 0.8
            p60 = np.percentile(atr_sorted, 60) if len(atr_sorted) > 0 else current_atr * 1.2
            p80 = np.percentile(atr_sorted, 80) if len(atr_sorted) > 0 else current_atr * 1.5
            
            if current_atr <= p20:
                zone = VolatilityZone.EXTREME_LOW
                zone_low, zone_high = 0, p20
                sl_mult, tp_mult = 0.7, 0.8
                desc = 'Extremely low volatility - tight ranges, potential squeeze'
            elif current_atr <= p40:
                zone = VolatilityZone.LOW
                zone_low, zone_high = p20, p40
                sl_mult, tp_mult = 0.85, 0.9
                desc = 'Low volatility - consolidation phase'
            elif current_atr <= p60:
                zone = VolatilityZone.NORMAL
                zone_low, zone_high = p40, p60
                sl_mult, tp_mult = 1.0, 1.0
                desc = 'Normal volatility - standard conditions'
            elif current_atr <= p80:
                zone = VolatilityZone.HIGH
                zone_low, zone_high = p60, p80
                sl_mult, tp_mult = 1.2, 1.3
                desc = 'High volatility - increased movement, wider stops needed'
            else:
                zone = VolatilityZone.EXTREME_HIGH
                zone_low, zone_high = p80, current_atr * 1.5
                sl_mult, tp_mult = 1.5, 1.5
                desc = 'Extreme volatility - high risk, consider reducing position size'
            
            if len(atr) >= 5:
                recent_atr = atr.tail(5)
                atr_slope = (recent_atr.iloc[-1] - recent_atr.iloc[0]) / 5 if len(recent_atr) >= 5 else 0
                volatility_expanding = atr_slope > 0 and abs(atr_slope) > atr_avg * 0.01
                volatility_contracting = atr_slope < 0 and abs(atr_slope) > atr_avg * 0.01
            else:
                volatility_expanding = False
                volatility_contracting = False
            
            return {
                'zone': zone,
                'atr_current': float(current_atr),
                'atr_avg': float(atr_avg),
                'atr_percentile': float(atr_percentile),
                'zone_threshold_low': float(zone_low),
                'zone_threshold_high': float(zone_high),
                'volatility_expanding': bool(volatility_expanding),
                'volatility_contracting': bool(volatility_contracting),
                'recommended_sl_multiplier': float(sl_mult),
                'recommended_tp_multiplier': float(tp_mult),
                'description': desc
            }
            
        except Exception as e:
            self._logger.warning(f"Gagal menghitung ATR volatility zones: {str(e)}")
            return default_result
    
    def calculate_adaptive_smoothed_rsi(self, df: pd.DataFrame, rsi_period: int = 14, 
                                          smoothing_period: int = 3) -> Dict:
        """
        Calculate RSI with adaptive smoothing based on volatility.
        
        Args:
            df: DataFrame with OHLC data
            rsi_period: Base RSI period
            smoothing_period: Base smoothing period (will be adapted)
        
        Returns:
            Dict with smoothed RSI analysis:
            - rsi_raw: Raw RSI value
            - rsi_smoothed: Smoothed RSI value
            - rsi_signal: Signal line (double smoothed)
            - crossover: 'bullish', 'bearish', or 'none'
            - overbought: True if RSI > 70
            - oversold: True if RSI < 30
        """
        default_result = {
            'rsi_raw': 50.0,
            'rsi_smoothed': 50.0,
            'rsi_signal': 50.0,
            'crossover': 'none',
            'overbought': False,
            'oversold': False
        }
        
        if not self._validate_dataframe(df, ['close']):
            return default_result
        
        try:
            rsi = self.calculate_rsi(df, rsi_period)
            
            vol_zones = self.calculate_atr_volatility_zones(df)
            zone = vol_zones.get('zone', VolatilityZone.NORMAL)
            
            if zone == VolatilityZone.EXTREME_LOW:
                adaptive_period = smoothing_period + 2
            elif zone == VolatilityZone.LOW:
                adaptive_period = smoothing_period + 1
            elif zone == VolatilityZone.HIGH:
                adaptive_period = max(1, smoothing_period - 1)
            elif zone == VolatilityZone.EXTREME_HIGH:
                adaptive_period = max(1, smoothing_period - 2)
            else:
                adaptive_period = smoothing_period
            
            rsi_smoothed = self.calculate_smoothed_indicator(rsi, 'ema', adaptive_period)
            rsi_signal = self.calculate_smoothed_indicator(rsi_smoothed, 'ema', adaptive_period)
            
            rsi_raw_val = safe_series_operation(rsi, 'value', -1, 50.0)
            rsi_smoothed_val = safe_series_operation(rsi_smoothed, 'value', -1, 50.0)
            rsi_signal_val = safe_series_operation(rsi_signal, 'value', -1, 50.0)
            
            crossover = 'none'
            if len(rsi_smoothed) >= 2 and len(rsi_signal) >= 2:
                prev_smoothed = safe_series_operation(rsi_smoothed, 'value', -2, 50.0)
                prev_signal = safe_series_operation(rsi_signal, 'value', -2, 50.0)
                
                if prev_smoothed <= prev_signal and rsi_smoothed_val > rsi_signal_val:
                    crossover = 'bullish'
                elif prev_smoothed >= prev_signal and rsi_smoothed_val < rsi_signal_val:
                    crossover = 'bearish'
            
            return {
                'rsi_raw': float(rsi_raw_val),
                'rsi_smoothed': float(rsi_smoothed_val),
                'rsi_signal': float(rsi_signal_val),
                'crossover': crossover,
                'overbought': rsi_smoothed_val > 70,
                'oversold': rsi_smoothed_val < 30
            }
            
        except Exception as e:
            self._logger.warning(f"Gagal menghitung adaptive smoothed RSI: {str(e)}")
            return default_result
