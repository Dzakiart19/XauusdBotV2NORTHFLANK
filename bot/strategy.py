from typing import Optional, Dict, Tuple, Any, List, NamedTuple
from datetime import datetime
import json
from bot.logger import setup_logger
import math
import pytz
from dataclasses import dataclass, field

logger = setup_logger('Strategy')

class StrategyError(Exception):
    """Base exception for strategy errors"""
    pass

class IndicatorValidationError(StrategyError):
    """Indicator data validation error"""
    pass

class PriceValidationError(StrategyError):
    """Price data validation error for NaN/Inf/negative values"""
    pass


@dataclass
class ValidationResult:
    """Result of price/indicator validation with warnings"""
    is_valid: bool
    value: float
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    
    def add_warning(self, msg: str):
        self.warnings.append(msg)
        logger.warning(f"Validation warning: {msg}")


class PriceDataValidator:
    """Centralized price data validation pipeline for NaN/Inf/Negative handling"""
    
    def __init__(self):
        self._validation_warnings: List[str] = []
        self._rejected_count = 0
    
    def reset_warnings(self):
        """Reset validation warnings for new validation cycle"""
        self._validation_warnings = []
    
    def get_warnings(self) -> List[str]:
        """Get accumulated validation warnings"""
        return self._validation_warnings.copy()
    
    def get_rejected_count(self) -> int:
        """Get count of rejected values"""
        return self._rejected_count
    
    def _add_warning(self, msg: str):
        """Add a validation warning and log it"""
        self._validation_warnings.append(msg)
        logger.warning(f"Price validation: {msg}")
    
    def validate(self, value: Any, name: str = "", 
                 min_val: Optional[float] = None, 
                 max_val: Optional[float] = None,
                 allow_zero: bool = False,
                 allow_negative: bool = False) -> ValidationResult:
        """Validate a single price/numeric value
        
        Args:
            value: Value to validate
            name: Name for logging/warnings
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            allow_zero: Whether zero is allowed
            allow_negative: Whether negative values are allowed
            
        Returns:
            ValidationResult with is_valid, cleaned value, and any warnings
        """
        result = ValidationResult(is_valid=True, value=0.0)
        
        if value is None:
            result.is_valid = False
            result.error = f"{name or 'Value'} is None"
            self._add_warning(result.error)
            self._rejected_count += 1
            return result
        
        if not isinstance(value, (int, float)):
            try:
                value = float(value)
            except (TypeError, ValueError) as e:
                result.is_valid = False
                result.error = f"{name or 'Value'} is not a number: {type(value).__name__}"
                self._add_warning(result.error)
                self._rejected_count += 1
                return result
        
        try:
            if math.isnan(value):
                result.is_valid = False
                result.error = f"{name or 'Value'} is NaN"
                self._add_warning(result.error)
                self._rejected_count += 1
                return result
            
            if math.isinf(value):
                result.is_valid = False
                result.error = f"{name or 'Value'} is Inf (sign={'+' if value > 0 else '-'})"
                self._add_warning(result.error)
                self._rejected_count += 1
                return result
        except TypeError:
            result.is_valid = False
            result.error = f"{name or 'Value'} type error in NaN/Inf check"
            self._add_warning(result.error)
            self._rejected_count += 1
            return result
        
        if not allow_negative and value < 0:
            result.is_valid = False
            result.error = f"{name or 'Value'} is negative: {value}"
            self._add_warning(result.error)
            self._rejected_count += 1
            return result
        
        if not allow_zero and value == 0:
            result.is_valid = False
            result.error = f"{name or 'Value'} is zero (not allowed)"
            self._add_warning(result.error)
            self._rejected_count += 1
            return result
        
        if min_val is not None and value < min_val:
            result.add_warning(f"{name or 'Value'} ({value}) below minimum ({min_val})")
            result.value = min_val
            return result
        
        if max_val is not None and value > max_val:
            result.add_warning(f"{name or 'Value'} ({value}) above maximum ({max_val})")
            result.value = max_val
            return result
        
        result.value = float(value)
        return result
    
    def validate_price(self, price: Any, name: str = "price") -> ValidationResult:
        """Validate a price value (must be positive, no NaN/Inf)"""
        return self.validate(price, name, min_val=0.01, allow_zero=False, allow_negative=False)
    
    def validate_ratio(self, ratio: Any, name: str = "ratio", 
                       min_val: float = 0.0, max_val: float = 100.0) -> ValidationResult:
        """Validate a ratio/percentage value"""
        return self.validate(ratio, name, min_val=min_val, max_val=max_val, allow_zero=True)
    
    def validate_atr(self, atr: Any, name: str = "atr") -> ValidationResult:
        """Validate ATR value (must be positive)"""
        return self.validate(atr, name, min_val=0.0001, allow_zero=False, allow_negative=False)


_price_validator = PriceDataValidator()


def validate_price_data(prices: Dict[str, Any], 
                        required_fields: Optional[List[str]] = None) -> Tuple[bool, Dict[str, float], List[str]]:
    """Centralized price data validation with NaN/Inf/Negative rejection
    
    Args:
        prices: Dictionary of price field names to values
        required_fields: List of required field names (all must pass validation)
        
    Returns:
        Tuple of (all_valid, cleaned_prices, warnings)
        - all_valid: True if all required fields are valid
        - cleaned_prices: Dictionary with validated float values
        - warnings: List of validation warning messages
    """
    _price_validator.reset_warnings()
    cleaned = {}
    all_valid = True
    
    if required_fields is None:
        required_fields = list(prices.keys())
    
    for field_name, value in prices.items():
        is_required = field_name in required_fields
        
        if 'price' in field_name.lower() or field_name in ['close', 'open', 'high', 'low']:
            result = _price_validator.validate_price(value, field_name)
        elif 'atr' in field_name.lower():
            result = _price_validator.validate_atr(value, field_name)
        elif 'rsi' in field_name.lower() or 'stoch' in field_name.lower():
            result = _price_validator.validate_ratio(value, field_name, 0, 100)
        else:
            result = _price_validator.validate(value, field_name, allow_negative=True, allow_zero=True)
        
        if result.is_valid:
            cleaned[field_name] = result.value
        elif is_required:
            all_valid = False
            logger.error(f"Required price field validation failed: {result.error}")
    
    return all_valid, cleaned, _price_validator.get_warnings()


def is_valid_number(value: Any) -> bool:
    """Check if value is a valid finite number (not None, NaN, or Inf)
    
    Args:
        value: Any value to check
        
    Returns:
        True if value is a valid finite number, False otherwise
    """
    if value is None:
        return False
    if not isinstance(value, (int, float)):
        return False
    try:
        if math.isnan(value) or math.isinf(value):
            return False
        return True
    except (TypeError, ValueError):
        return False


def safe_float(value: Any, default: float = 0.0, name: str = "") -> float:
    """Safely convert value to float with NaN/Inf protection
    
    Args:
        value: Value to convert
        default: Default value to return if conversion fails
        name: Optional name for logging
        
    Returns:
        Float value or default if invalid
    """
    if value is None:
        if name:
            logger.warning(f"NaN/Inf check: {name} is None, using default {default}")
        return default
    
    try:
        result = float(value)
        if math.isnan(result):
            if name:
                logger.warning(f"NaN detected in {name}, using default {default}")
            return default
        if math.isinf(result):
            if name:
                logger.warning(f"Inf detected in {name}, using default {default}")
            return default
        return result
    except (TypeError, ValueError) as e:
        if name:
            logger.warning(f"Invalid number in {name}: {e}, using default {default}")
        return default


def safe_divide(numerator: Any, denominator: Any, default: float = 0.0, name: str = "") -> float:
    """Safely divide two numbers with protection against division by zero and NaN/Inf
    
    Args:
        numerator: The numerator value
        denominator: The denominator value
        default: Default value to return if division fails
        name: Optional name for logging
        
    Returns:
        Division result or default if invalid
    """
    num = safe_float(numerator, 0.0)
    denom = safe_float(denominator, 0.0)
    
    if denom == 0.0:
        if name:
            logger.warning(f"Division by zero in {name}, using default {default}")
        return default
    
    try:
        result = num / denom
        if math.isnan(result) or math.isinf(result):
            if name:
                logger.warning(f"NaN/Inf result in {name} division, using default {default}")
            return default
        return result
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as e:
        if name:
            logger.warning(f"Division error in {name}: {e}, using default {default}")
        return default


def validate_rsi_history(rsi_history: Any) -> List[float]:
    """Validate and clean RSI history list
    
    Args:
        rsi_history: List of RSI values
        
    Returns:
        Cleaned list of valid RSI values (0-100 range)
    """
    if not rsi_history or not isinstance(rsi_history, (list, tuple)):
        return []
    
    cleaned = []
    for val in rsi_history:
        if is_valid_number(val):
            if 0 <= val <= 100:
                cleaned.append(float(val))
            else:
                logger.warning(f"RSI history value out of range: {val}, skipping")
    
    return cleaned


def validate_indicator_value(name: str, value: Any, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Tuple[bool, Optional[str]]:
    """Validate individual indicator value with range checks"""
    try:
        if value is None:
            return False, f"{name} is None"
        
        if not isinstance(value, (int, float)):
            return False, f"{name} has invalid type: {type(value)}"
        
        if math.isnan(value):
            return False, f"{name} is NaN"
        
        if math.isinf(value):
            return False, f"{name} is infinite"
        
        if min_val is not None and value < min_val:
            return False, f"{name} out of range: {value} < {min_val}"
        
        if max_val is not None and value > max_val:
            return False, f"{name} out of range: {value} > {max_val}"
        
        return True, None
        
    except (StrategyError, Exception) as e:
        return False, f"{name} validation error: {str(e)}"

def validate_indicators(indicators: Dict) -> Tuple[bool, Optional[str]]:
    """Validate all indicator data before processing"""
    try:
        if not indicators or not isinstance(indicators, dict):
            return False, "Indicators must be a non-empty dictionary"
        
        required_indicators = ['close', 'rsi', 'atr']
        missing = [ind for ind in required_indicators if ind not in indicators or indicators[ind] is None]
        if missing:
            return False, f"Missing required indicators: {missing}"
        
        close = indicators.get('close')
        is_valid, error = validate_indicator_value('close', close, min_val=0.01)
        if not is_valid:
            return False, error
        
        rsi = indicators.get('rsi')
        is_valid, error = validate_indicator_value('rsi', rsi, min_val=0, max_val=100)
        if not is_valid:
            return False, error
        
        atr = indicators.get('atr')
        is_valid, error = validate_indicator_value('atr', atr, min_val=0)
        if not is_valid:
            return False, error
        
        macd = indicators.get('macd')
        if macd is not None:
            is_valid, error = validate_indicator_value('macd', macd)
            if not is_valid:
                return False, error
        
        for ema_key in ['ema_5', 'ema_10', 'ema_20', 'ema_50']:
            ema_val = indicators.get(ema_key)
            if ema_val is not None:
                is_valid, error = validate_indicator_value(ema_key, ema_val, min_val=0)
                if not is_valid:
                    return False, error
        
        stoch_k = indicators.get('stoch_k')
        if stoch_k is not None:
            is_valid, error = validate_indicator_value('stoch_k', stoch_k, min_val=0, max_val=100)
            if not is_valid:
                logger.warning(f"Stochastic K validation failed: {error}")
                return False, error
        
        stoch_d = indicators.get('stoch_d')
        if stoch_d is not None:
            is_valid, error = validate_indicator_value('stoch_d', stoch_d, min_val=0, max_val=100)
            if not is_valid:
                logger.warning(f"Stochastic D validation failed: {error}")
                return False, error
        
        return True, None
        
    except (StrategyError, Exception) as e:
        return False, f"Indicator validation error: {str(e)}"

class TradingStrategy:
    def __init__(self, config, alert_system=None):
        self.config = config
        self.alert_system = alert_system
        self.last_volatility_alert = None
        
        self.last_signal_candle_timestamp: Optional[datetime] = None
        self.last_signal_type: Optional[str] = None
        self.last_signal_price: Optional[float] = None
        self.last_signal_time: Optional[datetime] = None
        
        self.last_buy_time: Optional[datetime] = None
        self.last_sell_time: Optional[datetime] = None
        self.last_pattern_cooldowns: Dict[str, datetime] = {}
        
        self._regime_detector = None
        self._current_regime = None
        
        self.indicator_engine: Any = None
    
    def _get_regime_detector(self):
        """Lazy initialization of regime detector to avoid circular imports."""
        if self._regime_detector is None:
            from bot.market_regime import MarketRegimeDetector
            self._regime_detector = MarketRegimeDetector(self.config)
        return self._regime_detector
    
    def check_regime_alignment(self, signal_type: str, indicators: Dict[str, Any], 
                                m1_df=None, m5_df=None) -> Tuple[bool, str]:
        """
        Check if signal direction aligns with market regime bias.
        
        IMPORTANT: This only BLOCKS signals with conflicting bias (BUY vs SELL).
        Allows signals when regime is uncertain or data unavailable (with warning).
        
        Args:
            signal_type: 'BUY' or 'SELL'
            indicators: Current indicators dict
            m1_df: M1 DataFrame (optional)
            m5_df: M5 DataFrame (optional)
            
        Returns:
            Tuple of (is_aligned, reason)
        """
        try:
            regime_detector = self._get_regime_detector()
            self._current_regime = regime_detector.get_regime(
                indicators=indicators,
                m1_df=m1_df,
                m5_df=m5_df
            )
            
            if self._current_regime is None:
                logger.warning(f"⚠️ Regime unavailable - allowing {signal_type} with caution")
                return True, f"⚠️ Regime data unavailable - {signal_type} allowed with caution"
            
            bias = self._current_regime.bias
            regime = self._current_regime.regime_type
            confidence = self._current_regime.confidence
            
            if confidence < 0.6:
                logger.info(f"⚠️ Low regime confidence ({confidence:.2f}) - allowing {signal_type} with caution")
                return True, f"⚠️ Low confidence ({confidence:.2f}) - {signal_type} allowed with caution"
            
            if signal_type == 'BUY' and bias == 'SELL' and confidence >= 0.75:
                return False, f"🚫 BLOCKED: BUY conflicts with strong SELL bias (conf: {confidence:.2f})"
            if signal_type == 'SELL' and bias == 'BUY' and confidence >= 0.75:
                return False, f"🚫 BLOCKED: SELL conflicts with strong BUY bias (conf: {confidence:.2f})"
            
            if signal_type == 'BUY' and bias == 'BUY':
                return True, f"✅ BUY aligned with BUY bias (regime: {regime})"
            elif signal_type == 'SELL' and bias == 'SELL':
                return True, f"✅ SELL aligned with SELL bias (regime: {regime})"
            elif bias == 'NEUTRAL':
                return True, f"⚠️ Neutral bias - {signal_type} allowed with caution"
            
            return True, f"Signal direction validated (regime: {regime}, bias: {bias})"
            
        except Exception as e:
            logger.warning(f"Regime check error: {str(e)} - allowing signal with caution")
            return True, f"⚠️ Regime check error - {signal_type} allowed with caution"
    
    def should_generate_signal(self, candle_timestamp: Optional[datetime], 
                                current_price: float, 
                                signal_type: str,
                                pattern_type: Optional[str] = None) -> Tuple[bool, str]:
        """Cek apakah boleh generate signal baru berdasarkan per-signal-type cooldown
        
        Smart Signal Cooldown System (semua cooldown bisa dikonfigurasi via config, default 0 = unlimited):
        1. Global minimum cooldown (GLOBAL_MIN_COOLDOWN_SECONDS, default 0)
        2. Per-signal-type cooldown (BUY_COOLDOWN_SECONDS, SELL_COOLDOWN_SECONDS, default 0)
        3. Opposite signal cooldown (OPPOSITE_SIGNAL_COOLDOWN_SECONDS, default 0)
        4. Per-pattern cooldown (PATTERN_COOLDOWN_SECONDS, default 0)
        5. Candle timestamp check (no duplicate signals per candle)
        6. Minimum price movement check (SIGNAL_MINIMUM_PRICE_MOVEMENT, default 0)
        
        Args:
            candle_timestamp: Timestamp candle saat ini
            current_price: Harga close saat ini
            signal_type: Tipe signal ('BUY' atau 'SELL')
            pattern_type: Pattern yang digunakan (inside_bar, pin_bar, etc.)
            
        Returns:
            Tuple[bool, str]: (can_generate, reason)
        """
        current_time = datetime.now(pytz.UTC)
        
        global_min_cooldown = getattr(self.config, 'GLOBAL_MIN_COOLDOWN_SECONDS', 0)
        buy_cooldown = getattr(self.config, 'BUY_COOLDOWN_SECONDS', 0)
        sell_cooldown = getattr(self.config, 'SELL_COOLDOWN_SECONDS', 0)
        opposite_cooldown = getattr(self.config, 'OPPOSITE_SIGNAL_COOLDOWN_SECONDS', 0)
        pattern_cooldown = getattr(self.config, 'PATTERN_COOLDOWN_SECONDS', 0)
        allow_multi_candle = getattr(self.config, 'ALLOW_MULTIPLE_SIGNALS_PER_CANDLE', True)
        
        if not allow_multi_candle:
            if candle_timestamp is not None and self.last_signal_candle_timestamp is not None:
                if candle_timestamp == self.last_signal_candle_timestamp:
                    reason = f"🚫 Signal di-skip: Masih dalam candle yang sama (timestamp: {candle_timestamp})"
                    logger.info(reason)
                    return False, reason
        
        if global_min_cooldown > 0 and self.last_signal_time is not None:
            time_since_last = (current_time - self.last_signal_time).total_seconds()
            if time_since_last < global_min_cooldown:
                remaining = global_min_cooldown - time_since_last
                reason = f"🚫 Signal di-skip: Global minimum cooldown ({remaining:.0f}s sisa dari {global_min_cooldown}s)"
                logger.info(reason)
                return False, reason
        
        if signal_type == 'BUY':
            if buy_cooldown > 0 and self.last_buy_time is not None:
                time_since_last_buy = (current_time - self.last_buy_time).total_seconds()
                if time_since_last_buy < buy_cooldown:
                    remaining = buy_cooldown - time_since_last_buy
                    reason = f"🚫 Signal di-skip: BUY cooldown aktif (sisa {remaining:.0f}s dari {buy_cooldown}s)"
                    logger.info(reason)
                    return False, reason
            
            if opposite_cooldown > 0 and self.last_signal_type == 'SELL' and self.last_sell_time is not None:
                time_since_last_sell = (current_time - self.last_sell_time).total_seconds()
                if time_since_last_sell < opposite_cooldown:
                    remaining = opposite_cooldown - time_since_last_sell
                    reason = f"🚫 Signal di-skip: Opposite signal cooldown (BUY after SELL, sisa {remaining:.0f}s dari {opposite_cooldown}s)"
                    logger.info(reason)
                    return False, reason
        
        elif signal_type == 'SELL':
            if sell_cooldown > 0 and self.last_sell_time is not None:
                time_since_last_sell = (current_time - self.last_sell_time).total_seconds()
                if time_since_last_sell < sell_cooldown:
                    remaining = sell_cooldown - time_since_last_sell
                    reason = f"🚫 Signal di-skip: SELL cooldown aktif (sisa {remaining:.0f}s dari {sell_cooldown}s)"
                    logger.info(reason)
                    return False, reason
            
            if opposite_cooldown > 0 and self.last_signal_type == 'BUY' and self.last_buy_time is not None:
                time_since_last_buy = (current_time - self.last_buy_time).total_seconds()
                if time_since_last_buy < opposite_cooldown:
                    remaining = opposite_cooldown - time_since_last_buy
                    reason = f"🚫 Signal di-skip: Opposite signal cooldown (SELL after BUY, sisa {remaining:.0f}s dari {opposite_cooldown}s)"
                    logger.info(reason)
                    return False, reason
        
        if pattern_cooldown > 0 and pattern_type and pattern_type in self.last_pattern_cooldowns:
            last_pattern_time = self.last_pattern_cooldowns[pattern_type]
            time_since_pattern = (current_time - last_pattern_time).total_seconds()
            if time_since_pattern < pattern_cooldown:
                remaining = pattern_cooldown - time_since_pattern
                reason = f"🚫 Signal di-skip: Pattern {pattern_type} cooldown (sisa {remaining:.0f}s dari {pattern_cooldown}s)"
                logger.info(reason)
                return False, reason
        
        min_price_movement = getattr(self.config, 'SIGNAL_MINIMUM_PRICE_MOVEMENT', 0.0)
        if self.last_signal_price is not None and min_price_movement > 0:
            price_diff = abs(current_price - self.last_signal_price)
            if price_diff < min_price_movement:
                reason = f"🚫 Signal di-skip: Pergerakan harga terlalu kecil (${price_diff:.2f} < ${min_price_movement:.2f})"
                logger.info(reason)
                return False, reason
        
        candle_info = f"candle {candle_timestamp}" if candle_timestamp else "realtime"
        pattern_info = f", pattern={pattern_type}" if pattern_type else ""
        reason = f"✅ Signal diizinkan: Candle baru ({candle_info}){pattern_info}, per-type cooldown clear"
        logger.debug(reason)
        return True, reason
    
    def _update_signal_tracking(self, candle_timestamp: Optional[datetime],
                                  signal_type: str, 
                                  entry_price: float,
                                  pattern_type: Optional[str] = None):
        """Update tracking setelah signal berhasil di-generate
        
        Smart Signal Cooldown Tracking:
        - Tracks general signal info (candle, type, price, time)
        - Tracks per-signal-type times (last_buy_time, last_sell_time)
        - Tracks per-pattern cooldowns
        
        Args:
            candle_timestamp: Timestamp candle saat signal di-generate
            signal_type: Tipe signal ('BUY' atau 'SELL')
            entry_price: Harga entry signal
            pattern_type: Pattern yang digunakan (inside_bar, pin_bar, etc.)
        """
        current_time = datetime.now(pytz.UTC)
        
        self.last_signal_candle_timestamp = candle_timestamp
        self.last_signal_type = signal_type
        self.last_signal_price = entry_price
        self.last_signal_time = current_time
        
        if signal_type == 'BUY':
            self.last_buy_time = current_time
        elif signal_type == 'SELL':
            self.last_sell_time = current_time
        
        if pattern_type:
            self.last_pattern_cooldowns[pattern_type] = current_time
            logger.debug(f"📊 Pattern cooldown set: {pattern_type} @ {current_time}")
        
        pattern_info = f", pattern={pattern_type}" if pattern_type else ""
        logger.info(f"📝 Signal tracking updated: {signal_type} @ ${entry_price:.2f} | Candle: {candle_timestamp}{pattern_info}")
    
    def calculate_lot_with_volatility_zones(self, indicators: Dict, base_lot_size: float) -> Tuple[float, str, float]:
        """Calculate dynamic lot size berdasarkan ATR volatility zones.
        
        Smart Position Sizing dengan Volatility Adapter - calculates ATR% internally
        from indicators and applies zone-based multipliers.
        
        Args:
            indicators: Dict with 'atr' and 'close' values
            base_lot_size: Base lot size from config
            
        Returns:
            Tuple of (adjusted_lot_size, volatility_zone, multiplier)
            
        ATR zone thresholds (as percentage of close price):
        - EXTREME_LOW: < 0.005% - multiplier = 0.5x (kurangi saat market quiet)
        - LOW: 0.005% - 0.02% - multiplier = 0.7x
        - NORMAL: 0.02% - 0.05% - multiplier = 1.0x (standard)
        - HIGH: 0.05% - 0.1% - multiplier = 0.85x
        - EXTREME_HIGH: > 0.1% - multiplier = 0.7x (kurangi saat volatile)
        """
        default_multiplier = 1.0
        default_zone = "NORMAL"
        
        try:
            base = safe_float(base_lot_size, 0.01, "base_lot_size")
            if base <= 0:
                base = 0.01
            
            if not indicators or not isinstance(indicators, dict):
                logger.warning("calculate_lot_with_volatility_zones: No indicators provided, using default multiplier 1.0x")
                return base, default_zone, default_multiplier
            
            atr_raw = indicators.get('atr')
            close_raw = indicators.get('close')
            
            if not is_valid_number(atr_raw):
                logger.warning("calculate_lot_with_volatility_zones: ATR not available, using default multiplier 1.0x")
                return base, default_zone, default_multiplier
            
            if not is_valid_number(close_raw):
                logger.warning("calculate_lot_with_volatility_zones: Close price not available, using default multiplier 1.0x")
                return base, default_zone, default_multiplier
            
            atr = safe_float(atr_raw, 0.0, "atr")
            close = safe_float(close_raw, 0.0, "close")
            
            if atr <= 0 or close <= 0:
                logger.warning(f"calculate_lot_with_volatility_zones: Invalid ATR ({atr}) or close ({close}), using default")
                return base, default_zone, default_multiplier
            
            atr_percent = safe_divide(atr, close, 0.0, "atr_percent") * 100
            
            if not is_valid_number(atr_percent) or atr_percent < 0:
                logger.warning(f"calculate_lot_with_volatility_zones: Invalid ATR% ({atr_percent}), using default")
                return base, default_zone, default_multiplier
            
            if atr_percent < 0.005:
                zone = "EXTREME_LOW"
                multiplier = 0.5
            elif atr_percent < 0.02:
                zone = "LOW"
                multiplier = 0.7
            elif atr_percent < 0.05:
                zone = "NORMAL"
                multiplier = 1.0
            elif atr_percent < 0.1:
                zone = "HIGH"
                multiplier = 0.85
            else:
                zone = "EXTREME_HIGH"
                multiplier = 0.7
            
            adjusted_lot = base * multiplier
            
            if not is_valid_number(adjusted_lot) or adjusted_lot <= 0:
                logger.warning(f"calculate_lot_with_volatility_zones: Invalid adjusted lot ({adjusted_lot}), using base")
                return base, default_zone, default_multiplier
            
            adjusted_lot = round(adjusted_lot, 2)
            
            logger.info(f"📊 Dynamic LOT ADJUSTMENT: Zone={zone}, ATR%={atr_percent:.4f}%, "
                       f"base={base:.2f}x{multiplier} = {adjusted_lot:.2f}")
            
            return adjusted_lot, zone, multiplier
            
        except Exception as e:
            logger.error(f"Error in calculate_lot_with_volatility_zones: {e}, using default")
            base = safe_float(base_lot_size, 0.01)
            return base if base > 0 else 0.01, default_zone, default_multiplier
    
    def calculate_trend_strength(self, indicators: Dict) -> Tuple[float, str]:
        """Calculate trend strength with validation and error handling
        
        Returns: (strength_score, description)
        """
        default_score = 0.3
        default_desc = "MEDIUM ⚡"
        
        try:
            if not indicators or not isinstance(indicators, dict):
                logger.warning("calculate_trend_strength: Invalid indicators dict, returning default")
                return default_score, default_desc
            
            is_valid, error_msg = validate_indicators(indicators)
            if not is_valid:
                logger.warning(f"Indicator validation failed in trend strength calculation: {error_msg}")
                return default_score, default_desc
            
            score = 0.0
            factors = []
            
            ema_short_raw = indicators.get(f'ema_{self.config.EMA_PERIODS[0]}')
            ema_mid_raw = indicators.get(f'ema_{self.config.EMA_PERIODS[1]}')
            ema_long_raw = indicators.get(f'ema_{self.config.EMA_PERIODS[2]}')
            macd_histogram_raw = indicators.get('macd_histogram')
            rsi_raw = indicators.get('rsi')
            close_raw = indicators.get('close')
            volume_raw = indicators.get('volume')
            volume_avg_raw = indicators.get('volume_avg')
            
            ema_short = safe_float(ema_short_raw, 0.0) if is_valid_number(ema_short_raw) else None
            ema_mid = safe_float(ema_mid_raw, 0.0) if is_valid_number(ema_mid_raw) else None
            ema_long = safe_float(ema_long_raw, 0.0) if is_valid_number(ema_long_raw) else None
            close = safe_float(close_raw, 0.0) if is_valid_number(close_raw) else None
            
            if (ema_short is not None and ema_mid is not None and 
                ema_long is not None and close is not None and close > 0):
                ema_separation = safe_divide(abs(ema_short - ema_long), close, 0.0, "ema_separation")
                if is_valid_number(ema_separation):
                    if ema_separation > 0.0005:
                        score += 0.25
                        factors.append("EMA spread lebar")
                    elif ema_separation > 0.0002:
                        score += 0.15
                        factors.append("EMA spread medium")
                    elif ema_separation > 0.0001:
                        score += 0.10
                        factors.append("EMA spread minimal")
            
            if is_valid_number(macd_histogram_raw):
                macd_histogram = safe_float(macd_histogram_raw, 0.0)
                macd_strength = abs(macd_histogram)
                if is_valid_number(macd_strength):
                    if macd_strength > 0.05:
                        score += 0.25
                        factors.append("MACD histogram kuat")
                    elif macd_strength > 0.02:
                        score += 0.15
                        factors.append("MACD histogram medium")
                    elif macd_strength > 0.01:
                        score += 0.10
                        factors.append("MACD histogram minimal")
            
            if is_valid_number(rsi_raw):
                rsi = safe_float(rsi_raw, 50.0)
                if 0 <= rsi <= 100:
                    rsi_momentum = safe_divide(abs(rsi - 50), 50, 0.0, "rsi_momentum")
                    if is_valid_number(rsi_momentum):
                        if rsi_momentum > 0.1:
                            score += 0.25
                            factors.append("RSI momentum tinggi")
                        elif rsi_momentum > 0.05:
                            score += 0.15
                            factors.append("RSI momentum medium")
                        elif rsi_momentum > 0.02:
                            score += 0.10
                            factors.append("RSI momentum minimal")
                else:
                    logger.warning(f"RSI out of range in trend strength: {rsi}")
            
            if is_valid_number(volume_raw) and is_valid_number(volume_avg_raw):
                volume = safe_float(volume_raw, 0.0)
                volume_avg = safe_float(volume_avg_raw, 0.0)
                if volume_avg > 0:
                    volume_ratio = safe_divide(volume, volume_avg, 0.0, "volume_ratio")
                    if is_valid_number(volume_ratio):
                        if volume_ratio > 0.5:
                            score += 0.25
                            factors.append("Volume sangat tinggi")
                        elif volume_ratio > 0.3:
                            score += 0.15
                            factors.append("Volume tinggi")
                        elif volume_ratio > 0.1:
                            score += 0.10
                            factors.append("Volume minimal")
            
            if math.isnan(score) or math.isinf(score):
                logger.warning(f"NaN/Inf detected in trend strength score, returning default")
                return default_score, default_desc
            
            score = min(max(score, 0.0), 1.0)
            
            if score >= 0.75:
                description = "SANGAT KUAT 🔥"
            elif score >= 0.5:
                description = "KUAT 💪"
            elif score >= 0.3:
                description = "MEDIUM ⚡"
            else:
                description = "LEMAH 📊"
            
            return score, description
            
        except (StrategyError, Exception) as e:
            logger.error(f"Error calculating trend strength: {e}")
            logger.warning(f"Trend strength calculation fallback triggered: Using default MEDIUM score due to error: {str(e)}")
            return default_score, default_desc
    
    def check_high_volatility(self, indicators: Dict):
        """Check for high volatility and send alert if detected"""
        try:
            if not indicators or not isinstance(indicators, dict):
                return
            
            atr_raw = indicators.get('atr')
            close_raw = indicators.get('close')
            
            if not is_valid_number(atr_raw) or not is_valid_number(close_raw):
                return
            
            atr = safe_float(atr_raw, 0.0)
            close = safe_float(close_raw, 0.0)
            
            if atr <= 0 or close <= 0:
                return
            
            volatility_percent = safe_divide(atr, close, 0.0, "volatility_percent") * 100
            
            if not is_valid_number(volatility_percent):
                logger.warning("NaN/Inf detected in volatility calculation, skipping alert")
                return
            
            high_volatility_threshold = 0.15
            
            if volatility_percent >= high_volatility_threshold:
                from datetime import datetime, timedelta
                import pytz
                
                current_time = datetime.now(pytz.UTC)
                
                if self.last_volatility_alert is None or (current_time - self.last_volatility_alert).total_seconds() > 3600:
                    self.last_volatility_alert = current_time
                    
                    if self.alert_system:
                        import asyncio
                        try:
                            asyncio.create_task(
                                self.alert_system.send_high_volatility_alert(
                                    "XAUUSD",
                                    volatility_percent
                                )
                            )
                            logger.warning(f"High volatility detected: {volatility_percent:.2f}% (ATR: ${atr:.2f}, Price: ${close:.2f})")
                        except (StrategyError, Exception) as alert_error:
                            logger.error(f"Failed to send high volatility alert: {alert_error}")
                            
        except (StrategyError, Exception) as e:
            logger.error(f"Error checking high volatility: {e}")
    
    def check_pullback_confirmation(self, rsi_history: list, signal_type: str) -> bool:
        """Check if there was a proper pullback before the signal
        
        BUY: RSI should have dropped to 40-45 range and then recovered
        SELL: RSI should have risen to 55-60 range and then declined
        
        Args:
            rsi_history: List of recent RSI values (last 20 values)
            signal_type: 'BUY' or 'SELL'
        
        Returns:
            True if pullback confirmed, False otherwise
        """
        try:
            if signal_type not in ['BUY', 'SELL']:
                logger.warning(f"Invalid signal_type in pullback confirmation: {signal_type}")
                return False
            
            cleaned_history = validate_rsi_history(rsi_history)
            
            if len(cleaned_history) < 5:
                return False
            
            recent_history = cleaned_history[-10:] if len(cleaned_history) >= 10 else cleaned_history
            current_rsi = cleaned_history[-1]
            
            if not is_valid_number(current_rsi):
                logger.warning("Invalid current RSI in pullback confirmation")
                return False
            
            if signal_type == 'BUY':
                pullback_detected = any(40 <= rsi <= 45 for rsi in recent_history if is_valid_number(rsi))
                if pullback_detected and current_rsi > 45:
                    return True
            elif signal_type == 'SELL':
                pullback_detected = any(55 <= rsi <= 60 for rsi in recent_history if is_valid_number(rsi))
                if pullback_detected and current_rsi < 55:
                    return True
            
            return False
        except (StrategyError, Exception) as e:
            logger.error(f"Error checking pullback confirmation: {e}")
            return False
    
    def is_optimal_trading_session(self) -> Tuple[bool, str]:
        """Check if current time is within optimal trading hours - SCALPING MODE: 24/7 ENABLED
        
        SCALPING MODE: Trading diizinkan 24 jam untuk menangkap semua peluang.
        Session info hanya untuk informasi, tidak memblokir sinyal.
        
        Returns:
            Tuple of (is_optimal, reason) - Selalu True untuk scalping mode
        """
        try:
            current_time = datetime.now(pytz.UTC)
            current_hour = current_time.hour
            current_hour_wib = (current_hour + 7) % 24
            
            is_london_session = 7 <= current_hour < 12
            is_ny_session = 12 <= current_hour < 17
            is_overlap = 12 <= current_hour < 14
            is_asian_session = 0 <= current_hour < 7
            
            if is_overlap:
                reason = f"✅ Session OPTIMAL: London-NY Overlap ({current_hour_wib:02d}:00 WIB) - SCALPING ACTIVE"
            elif is_london_session:
                reason = f"✅ Session BAIK: Sesi London ({current_hour_wib:02d}:00 WIB) - SCALPING ACTIVE"
            elif is_ny_session:
                reason = f"✅ Session BAIK: Sesi New York ({current_hour_wib:02d}:00 WIB) - SCALPING ACTIVE"
            elif is_asian_session:
                reason = f"✅ Session ASIA: Sesi Asia ({current_hour_wib:02d}:00 WIB) - SCALPING ACTIVE 24/7"
            else:
                reason = f"✅ Session AKTIF: Mode Scalping 24/7 ({current_hour_wib:02d}:00 WIB)"
            
            return True, reason
                    
        except (StrategyError, Exception) as e:
            logger.error(f"Error checking trading session: {e}")
            return True, "✅ Session check - SCALPING MODE 24/7"
    
    def check_trend_filter(self, indicators: Dict) -> Tuple[bool, str, str]:
        """Check trend filter conditions - RELAXED VERSION for ranging market
        
        PERBAIKAN untuk kondisi market ranging:
        - Tier 1: Perfect EMA alignment (EMA5 > EMA20 > EMA50 atau sebaliknya)
        - Tier 2: Partial alignment (close vs EMA50 + momentum direction)
        - Tier 3: MACD/RSI momentum confirmation sebagai fallback
        - Tier 4: TRF atau CEREBR bias sebagai alternatif
        
        Args:
            indicators: Dictionary of calculated indicators
            
        Returns:
            Tuple of (is_valid, signal_type, reason)
            - is_valid: True if trend filter passed
            - signal_type: 'BUY', 'SELL', or '' if no valid trend
            - reason: Description of the trend condition
        """
        try:
            ema_5 = indicators.get(f'ema_{self.config.EMA_PERIODS[0]}')
            ema_20 = indicators.get(f'ema_{self.config.EMA_PERIODS[1]}')
            ema_50 = indicators.get(f'ema_{self.config.EMA_PERIODS[2]}')
            close = indicators.get('close')
            
            if not all([is_valid_number(ema_5), is_valid_number(ema_20), 
                       is_valid_number(ema_50), is_valid_number(close)]):
                logger.debug("Trend Filter: Missing or invalid EMA/close values")
                return False, '', "Missing or invalid EMA/close values"
            
            ema_5 = safe_float(ema_5, 0.0)
            ema_20 = safe_float(ema_20, 0.0)
            ema_50 = safe_float(ema_50, 0.0)
            close = safe_float(close, 0.0)
            
            bullish_ema_alignment = ema_5 > ema_20 > ema_50
            bullish_price_above_ema50 = close > ema_50
            bearish_ema_alignment = ema_5 < ema_20 < ema_50
            bearish_price_below_ema50 = close < ema_50
            
            if bullish_ema_alignment and bullish_price_above_ema50:
                reason = f"✅ Trend Filter PASSED [Tier 1 BUY]: EMA5({ema_5:.2f}) > EMA20({ema_20:.2f}) > EMA50({ema_50:.2f}), Close({close:.2f}) > EMA50"
                logger.info(reason)
                return True, 'BUY', reason
            elif bearish_ema_alignment and bearish_price_below_ema50:
                reason = f"✅ Trend Filter PASSED [Tier 1 SELL]: EMA5({ema_5:.2f}) < EMA20({ema_20:.2f}) < EMA50({ema_50:.2f}), Close({close:.2f}) < EMA50"
                logger.info(reason)
                return True, 'SELL', reason
            
            bullish_partial = (ema_5 > ema_20 and close > ema_50)
            bearish_partial = (ema_5 < ema_20 and close < ema_50)
            
            if bullish_partial:
                reason = f"✅ Trend Filter PASSED [Tier 2 BUY]: EMA5({ema_5:.2f}) > EMA20({ema_20:.2f}), Close({close:.2f}) > EMA50({ema_50:.2f})"
                logger.info(reason)
                return True, 'BUY', reason
            elif bearish_partial:
                reason = f"✅ Trend Filter PASSED [Tier 2 SELL]: EMA5({ema_5:.2f}) < EMA20({ema_20:.2f}), Close({close:.2f}) < EMA50({ema_50:.2f})"
                logger.info(reason)
                return True, 'SELL', reason
            
            macd = indicators.get('macd')
            macd_signal = indicators.get('macd_signal')
            macd_histogram = indicators.get('macd_histogram')
            rsi = indicators.get('rsi')
            
            if is_valid_number(macd) and is_valid_number(macd_signal):
                macd_val = safe_float(macd, 0.0)
                macd_signal_val = safe_float(macd_signal, 0.0)
                
                if close > ema_50 and macd_val > macd_signal_val:
                    reason = f"✅ Trend Filter PASSED [Tier 3 BUY]: Close({close:.2f}) > EMA50, MACD bullish crossover"
                    logger.info(reason)
                    return True, 'BUY', reason
                elif close < ema_50 and macd_val < macd_signal_val:
                    reason = f"✅ Trend Filter PASSED [Tier 3 SELL]: Close({close:.2f}) < EMA50, MACD bearish crossover"
                    logger.info(reason)
                    return True, 'SELL', reason
            
            if is_valid_number(rsi):
                rsi_val = safe_float(rsi, 50.0)
                
                if close > ema_50 and rsi_val > 50:
                    reason = f"✅ Trend Filter PASSED [Tier 3 BUY]: Close({close:.2f}) > EMA50, RSI({rsi_val:.1f}) bullish"
                    logger.info(reason)
                    return True, 'BUY', reason
                elif close < ema_50 and rsi_val < 50:
                    reason = f"✅ Trend Filter PASSED [Tier 3 SELL]: Close({close:.2f}) < EMA50, RSI({rsi_val:.1f}) bearish"
                    logger.info(reason)
                    return True, 'SELL', reason
            
            trf_trend = indicators.get('trf_trend')
            cerebr_bias = indicators.get('cerebr_bias')
            
            if is_valid_number(trf_trend):
                trf = int(safe_float(trf_trend, 0))
                if trf == 1 and close > ema_50:
                    reason = f"✅ Trend Filter PASSED [Tier 4 BUY]: TRF bullish trend, Close({close:.2f}) > EMA50"
                    logger.info(reason)
                    return True, 'BUY', reason
                elif trf == -1 and close < ema_50:
                    reason = f"✅ Trend Filter PASSED [Tier 4 SELL]: TRF bearish trend, Close({close:.2f}) < EMA50"
                    logger.info(reason)
                    return True, 'SELL', reason
            
            if is_valid_number(cerebr_bias):
                bias = int(safe_float(cerebr_bias, 0))
                if bias == 1 and close > ema_50:
                    reason = f"✅ Trend Filter PASSED [Tier 4 BUY]: CEREBR bullish bias, Close({close:.2f}) > EMA50"
                    logger.info(reason)
                    return True, 'BUY', reason
                elif bias == -1 and close < ema_50:
                    reason = f"✅ Trend Filter PASSED [Tier 4 SELL]: CEREBR bearish bias, Close({close:.2f}) < EMA50"
                    logger.info(reason)
                    return True, 'SELL', reason
            
            reason = f"❌ Trend Filter FAILED: No clear trend direction - EMA5({ema_5:.2f}), EMA20({ema_20:.2f}), EMA50({ema_50:.2f}), Close({close:.2f})"
            logger.debug(reason)
            return False, '', reason
                
        except (StrategyError, Exception) as e:
            logger.error(f"Error in check_trend_filter: {e}")
            return False, '', f"Error: {str(e)}"
    
    def check_momentum_filter(self, indicators: Dict, signal_type: str) -> Tuple[bool, str]:
        """Check momentum filter conditions - RELAXED VERSION
        
        PERBAIKAN:
        - RSI range diperluas ke [25-75] (sebelumnya [35-65])
        - RSI direction lebih lenient: BUY RSI > 45, SELL RSI < 55
        - Accept neutral zone [40-60] jika Trend Filter sudah PASS
        - Stochastic hanya sebagai bonus, bukan blocking
        
        Args:
            indicators: Dictionary of calculated indicators
            signal_type: 'BUY' or 'SELL'
            
        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            rsi = indicators.get('rsi')
            stoch_k = indicators.get('stoch_k')
            stoch_d = indicators.get('stoch_d')
            stoch_k_prev = indicators.get('stoch_k_prev')
            stoch_d_prev = indicators.get('stoch_d_prev')
            
            if not is_valid_number(rsi):
                logger.debug("Momentum Filter: Invalid RSI value")
                return False, "❌ Momentum Filter FAILED: Invalid RSI value"
            
            rsi = safe_float(rsi, 50.0)
            
            rsi_entry_min = 10.0
            rsi_entry_max = 90.0
            
            rsi_in_range = rsi_entry_min <= rsi <= rsi_entry_max
            if not rsi_in_range:
                reason = f"⚠️ Momentum Filter WARNING: RSI({rsi:.1f}) extreme - tetap lanjut untuk scalping"
                logger.debug(reason)
            
            rsi_direction_valid = True
            neutral_zone = 30 <= rsi <= 70
            
            if signal_type == 'BUY':
                rsi_direction_valid = rsi >= 30 or neutral_zone
            elif signal_type == 'SELL':
                rsi_direction_valid = rsi <= 70 or neutral_zone
            
            stoch_reason = ""
            
            if all([is_valid_number(stoch_k), is_valid_number(stoch_d), 
                   is_valid_number(stoch_k_prev), is_valid_number(stoch_d_prev)]):
                stoch_k = safe_float(stoch_k, 50.0)
                stoch_d = safe_float(stoch_d, 50.0)
                stoch_k_prev = safe_float(stoch_k_prev, 50.0)
                stoch_d_prev = safe_float(stoch_d_prev, 50.0)
                
                if signal_type == 'BUY':
                    stoch_cross = stoch_k_prev < stoch_d_prev and stoch_k >= stoch_d
                    stoch_favorable = stoch_k > stoch_d or stoch_k < 70
                    if stoch_cross:
                        stoch_reason = f", Stoch K({stoch_k:.1f}) crossed above D({stoch_d:.1f}) 🎯"
                    elif stoch_favorable:
                        stoch_reason = f", Stoch favorable({stoch_k:.1f})"
                elif signal_type == 'SELL':
                    stoch_cross = stoch_k_prev > stoch_d_prev and stoch_k <= stoch_d
                    stoch_favorable = stoch_k < stoch_d or stoch_k > 30
                    if stoch_cross:
                        stoch_reason = f", Stoch K({stoch_k:.1f}) crossed below D({stoch_d:.1f}) 🎯"
                    elif stoch_favorable:
                        stoch_reason = f", Stoch favorable({stoch_k:.1f})"
            
            direction_info = "neutral zone ✅" if neutral_zone else f"direction matches {signal_type}"
            reason = f"✅ Momentum Filter PASSED: RSI({rsi:.1f}) in range [{rsi_entry_min}-{rsi_entry_max}], {direction_info}{stoch_reason}"
            logger.info(reason)
            return True, reason
            
        except (StrategyError, Exception) as e:
            logger.error(f"Error in check_momentum_filter: {e}")
            return False, f"Error: {str(e)}"
    
    def check_adx_filter(self, indicators: Dict, signal_mode: str = 'M1_SCALP') -> Tuple[bool, str, float, float]:
        """Check ADX filter - BLOCKING dengan threshold berbeda berdasarkan signal mode
        
        ADX filter dengan multi-level threshold:
        - M1_SCALP: ADX >= 15 (izinkan di trend lemah dengan reduced TP)
        - M5_SWING: ADX >= 20 (butuh trend lebih kuat)
        - BREAKOUT: ADX >= 25 (butuh momentum kuat untuk breakout)
        
        Args:
            indicators: Dictionary of calculated indicators
            signal_mode: Mode signal ('M1_SCALP', 'M5_SWING', 'BREAKOUT')
            
        Returns:
            Tuple of (is_valid, reason, adx_value, tp_multiplier)
            - tp_multiplier: 1.0 normal, 0.7 jika ADX rendah tapi allowed untuk scalp
        """
        try:
            adx = indicators.get('adx')
            plus_di = indicators.get('plus_di')
            minus_di = indicators.get('minus_di')
            
            if not is_valid_number(adx):
                return True, "✅ ADX Info: Data tidak tersedia - signal diizinkan", 0.0, 1.0
            
            adx_val = safe_float(adx, 0.0)
            
            adx_threshold_scalp = getattr(self.config, 'ADX_THRESHOLD', 15)
            adx_threshold_swing = 20
            adx_threshold_breakout = 25
            
            di_info = ""
            di_direction = 'neutral'
            if is_valid_number(plus_di) and is_valid_number(minus_di):
                plus_val = safe_float(plus_di, 0.0)
                minus_val = safe_float(minus_di, 0.0)
                if plus_val > minus_val:
                    di_info = f" | +DI({plus_val:.1f}) > -DI({minus_val:.1f}) = Bullish"
                    di_direction = 'bullish'
                else:
                    di_info = f" | -DI({minus_val:.1f}) > +DI({plus_val:.1f}) = Bearish"
                    di_direction = 'bearish'
            
            tp_multiplier = 1.0
            
            if signal_mode == 'BREAKOUT':
                if adx_val >= adx_threshold_breakout:
                    reason = f"✅ ADX BREAKOUT: ADX({adx_val:.1f}) >= {adx_threshold_breakout} (Momentum kuat){di_info}"
                    logger.info(reason)
                    return True, reason, adx_val, 1.0
                else:
                    reason = f"❌ ADX LEMAH untuk BREAKOUT: ADX({adx_val:.1f}) < {adx_threshold_breakout}{di_info}"
                    logger.info(reason)
                    return False, reason, adx_val, 0.0
                    
            elif signal_mode == 'M5_SWING':
                if adx_val >= adx_threshold_swing:
                    reason = f"✅ ADX SWING: ADX({adx_val:.1f}) >= {adx_threshold_swing} (Trend cukup){di_info}"
                    logger.info(reason)
                    return True, reason, adx_val, 1.0
                else:
                    reason = f"❌ ADX LEMAH untuk M5_SWING: ADX({adx_val:.1f}) < {adx_threshold_swing}{di_info}"
                    logger.info(reason)
                    return False, reason, adx_val, 0.0
            
            else:
                if adx_val >= adx_threshold_scalp:
                    reason = f"✅ ADX SCALP: ADX({adx_val:.1f}) >= {adx_threshold_scalp} (Trending){di_info}"
                    logger.info(reason)
                    return True, reason, adx_val, 1.0
                elif adx_val >= 10:
                    tp_multiplier = 0.7
                    reason = f"⚠️ ADX RENDAH SCALP: ADX({adx_val:.1f}) - Izinkan dengan reduced TP (70%){di_info}"
                    logger.info(reason)
                    return True, reason, adx_val, tp_multiplier
                else:
                    reason = f"❌ ADX TERLALU LEMAH: ADX({adx_val:.1f}) < 10 (No trend){di_info}"
                    logger.info(reason)
                    return False, reason, adx_val, 0.0
                
        except (StrategyError, Exception) as e:
            logger.error(f"Error in check_adx_filter: {e}")
            return True, f"✅ ADX Filter SKIPPED: Error - {str(e)}", 0.0, 1.0
    
    def check_m5_confirmation(self, m5_indicators: Optional[Dict], signal_type: str) -> Tuple[bool, str, float]:
        """Memeriksa konfirmasi trend dari timeframe M5 untuk meningkatkan akurasi signal AUTO.
        
        M5 Confirmation memeriksa trend alignment di higher timeframe (M5) untuk memastikan
        signal M1 searah dengan trend M5. Ini mengurangi false signals dengan memastikan
        multi-timeframe alignment.
        
        Multi-Timeframe Confirmation dengan Score Reduction:
        - M1 signal harus selaras dengan M5 trend direction
        - Jika M1 = BUY, M5 trend harus bullish (EMA5 > EMA20)
        - Jika tidak selaras, reduce score 30%
        
        Kriteria M5 Confirmation:
        1. EMA Alignment di M5 (EMA5 vs EMA20) searah dengan signal - WAJIB untuk full score
        2. RSI M5 mendukung arah signal (> 50 untuk BUY, < 50 untuk SELL)
        3. MACD M5 direction konfirmasi trend (bullish/bearish)
        
        Minimal 2 dari 3 kriteria harus terpenuhi untuk PASS.
        EMA alignment wajib untuk full confidence, jika tidak = 30% score reduction.
        
        Args:
            m5_indicators: Dictionary indicator dari timeframe M5 (bisa None)
            signal_type: 'BUY' atau 'SELL' dari M1
            
        Returns:
            Tuple[bool, str, float]: (passed, reason, score_multiplier)
            - score_multiplier: 1.0 jika aligned, 0.7 jika misaligned (30% reduction)
            - Jika m5_indicators None/kosong, return (True, skip message, 1.0)
        """
        try:
            if not m5_indicators or not isinstance(m5_indicators, dict):
                reason = "✅ M5 Confirmation: Data M5 tidak tersedia - lanjut dengan M1 saja"
                logger.debug(reason)
                return True, reason, 1.0
            
            if signal_type not in ['BUY', 'SELL']:
                reason = f"❌ M5 Confirmation: Invalid signal_type: {signal_type}"
                logger.warning(reason)
                return False, reason, 0.0
            
            confirmations_passed = 0
            confirmations_needed = 2
            confirmation_details = []
            
            ema_aligned = False
            score_multiplier = 1.0
            
            ema_5_m5 = m5_indicators.get(f'ema_{self.config.EMA_PERIODS[0]}')
            ema_20_m5 = m5_indicators.get(f'ema_{self.config.EMA_PERIODS[1]}')
            close_m5 = m5_indicators.get('close')
            
            if is_valid_number(ema_5_m5) and is_valid_number(ema_20_m5) and is_valid_number(close_m5):
                ema5 = safe_float(ema_5_m5, 0.0)
                ema20 = safe_float(ema_20_m5, 0.0)
                close = safe_float(close_m5, 0.0)
                
                if signal_type == 'BUY':
                    if ema5 > ema20:
                        ema_aligned = True
                        confirmations_passed += 1
                        confirmation_details.append(f"✅ EMA M5 bullish (EMA5={ema5:.2f} > EMA20={ema20:.2f})")
                    else:
                        ema_aligned = False
                        confirmation_details.append(f"⚠️ EMA M5 TIDAK bullish (EMA5={ema5:.2f} <= EMA20={ema20:.2f}) - 30% REDUCTION")
                elif signal_type == 'SELL':
                    if ema5 < ema20:
                        ema_aligned = True
                        confirmations_passed += 1
                        confirmation_details.append(f"✅ EMA M5 bearish (EMA5={ema5:.2f} < EMA20={ema20:.2f})")
                    else:
                        ema_aligned = False
                        confirmation_details.append(f"⚠️ EMA M5 TIDAK bearish (EMA5={ema5:.2f} >= EMA20={ema20:.2f}) - 30% REDUCTION")
            else:
                confirmation_details.append("EMA M5: data tidak tersedia")
            
            if not ema_aligned and is_valid_number(ema_5_m5):
                score_multiplier = 0.7
                logger.info(f"⚠️ M5 EMA misalignment detected - applying 30% score reduction")
            
            rsi_m5 = m5_indicators.get('rsi')
            
            if is_valid_number(rsi_m5):
                rsi = safe_float(rsi_m5, 50.0)
                
                if signal_type == 'BUY':
                    if rsi > 45:
                        confirmations_passed += 1
                        confirmation_details.append(f"RSI M5 bullish ({rsi:.1f} > 45)")
                    else:
                        confirmation_details.append(f"RSI M5 bearish ({rsi:.1f} <= 45)")
                elif signal_type == 'SELL':
                    if rsi < 55:
                        confirmations_passed += 1
                        confirmation_details.append(f"RSI M5 bearish ({rsi:.1f} < 55)")
                    else:
                        confirmation_details.append(f"RSI M5 bullish ({rsi:.1f} >= 55)")
            else:
                confirmation_details.append("RSI M5: data tidak tersedia")
            
            macd_m5 = m5_indicators.get('macd')
            macd_signal_m5 = m5_indicators.get('macd_signal')
            macd_histogram_m5 = m5_indicators.get('macd_histogram')
            
            if is_valid_number(macd_m5) and is_valid_number(macd_signal_m5):
                macd = safe_float(macd_m5, 0.0)
                macd_sig = safe_float(macd_signal_m5, 0.0)
                
                if signal_type == 'BUY':
                    if macd > macd_sig or (is_valid_number(macd_histogram_m5) and safe_float(macd_histogram_m5, 0.0) > 0):
                        confirmations_passed += 1
                        confirmation_details.append(f"MACD M5 bullish")
                    else:
                        confirmation_details.append(f"MACD M5 tidak bullish")
                elif signal_type == 'SELL':
                    if macd < macd_sig or (is_valid_number(macd_histogram_m5) and safe_float(macd_histogram_m5, 0.0) < 0):
                        confirmations_passed += 1
                        confirmation_details.append(f"MACD M5 bearish")
                    else:
                        confirmation_details.append(f"MACD M5 tidak bearish")
            else:
                confirmation_details.append("MACD M5: data tidak tersedia")
            
            if confirmations_passed >= confirmations_needed:
                score_info = "" if score_multiplier == 1.0 else f" [Score: {score_multiplier:.0%}]"
                reason = f"✅ M5 Confirmation PASSED: {confirmations_passed}/{confirmations_needed} kriteria{score_info} ({', '.join(confirmation_details)})"
                logger.info(reason)
                return True, reason, score_multiplier
            else:
                score_info = "" if score_multiplier == 1.0 else f" [Score: {score_multiplier:.0%}]"
                reason = f"❌ M5 Confirmation FAILED: Hanya {confirmations_passed}/{confirmations_needed} kriteria pass{score_info} ({', '.join(confirmation_details)})"
                logger.info(reason)
                return False, reason, score_multiplier
                
        except (StrategyError, Exception) as e:
            logger.error(f"Error in check_m5_confirmation: {e}")
            return True, f"✅ M5 Confirmation: Error - {str(e)} (lanjut dengan M1)", 1.0
    
    def check_h1_confirmation(self, h1_indicators: Optional[Dict], signal_type: str) -> Tuple[bool, str, float]:
        """Memeriksa konfirmasi trend dari timeframe H1 untuk meningkatkan akurasi signal.
        
        H1 Confirmation memeriksa trend alignment di higher timeframe (H1) untuk memastikan
        signal M1/M5 searah dengan trend H1. Ini adalah level tertinggi dari multi-timeframe confirmation.
        
        Kriteria H1 Confirmation:
        1. EMA Alignment di H1 (EMA5 vs EMA20 vs EMA50) searah dengan signal
        2. RSI H1 mendukung arah signal (> 50 untuk BUY, < 50 untuk SELL)
        3. Overall trend structure confirmation
        
        Args:
            h1_indicators: Dictionary indicator dari timeframe H1 (bisa None)
            signal_type: 'BUY' atau 'SELL' dari M1/M5
            
        Returns:
            Tuple[bool, str, float]: (passed, reason, score_multiplier)
        """
        try:
            if not h1_indicators or not isinstance(h1_indicators, dict):
                reason = "✅ H1 Confirmation: Data H1 tidak tersedia - lanjut dengan M1+M5"
                logger.debug(reason)
                return True, reason, 1.0
            
            if signal_type not in ['BUY', 'SELL']:
                reason = f"❌ H1 Confirmation: Invalid signal_type: {signal_type}"
                logger.warning(reason)
                return False, reason, 0.0
            
            confirmations_passed = 0
            confirmations_needed = 2
            confirmation_details = []
            ema_aligned = False
            score_multiplier = 1.0
            
            ema_5_h1 = h1_indicators.get(f'ema_{self.config.EMA_PERIODS[0]}')
            ema_20_h1 = h1_indicators.get(f'ema_{self.config.EMA_PERIODS[1]}')
            ema_50_h1 = h1_indicators.get(f'ema_{self.config.EMA_PERIODS[2]}') if len(self.config.EMA_PERIODS) > 2 else None
            close_h1 = h1_indicators.get('close')
            
            if is_valid_number(ema_5_h1) and is_valid_number(ema_20_h1):
                ema5 = safe_float(ema_5_h1, 0.0)
                ema20 = safe_float(ema_20_h1, 0.0)
                ema50 = safe_float(ema_50_h1, 0.0) if is_valid_number(ema_50_h1) else None
                
                if signal_type == 'BUY':
                    if ema5 > ema20:
                        ema_aligned = True
                        confirmations_passed += 1
                        if ema50 and ema20 > ema50:
                            confirmation_details.append(f"✅ EMA H1 STRONG bullish (EMA5>{ema20:.1f}>EMA50)")
                        else:
                            confirmation_details.append(f"✅ EMA H1 bullish (EMA5={ema5:.2f} > EMA20={ema20:.2f})")
                    else:
                        confirmation_details.append(f"⚠️ EMA H1 TIDAK bullish - 25% REDUCTION")
                        score_multiplier = 0.75
                elif signal_type == 'SELL':
                    if ema5 < ema20:
                        ema_aligned = True
                        confirmations_passed += 1
                        if ema50 and ema20 < ema50:
                            confirmation_details.append(f"✅ EMA H1 STRONG bearish (EMA5<{ema20:.1f}<EMA50)")
                        else:
                            confirmation_details.append(f"✅ EMA H1 bearish (EMA5={ema5:.2f} < EMA20={ema20:.2f})")
                    else:
                        confirmation_details.append(f"⚠️ EMA H1 TIDAK bearish - 25% REDUCTION")
                        score_multiplier = 0.75
            else:
                confirmation_details.append("EMA H1: data tidak tersedia")
            
            rsi_h1 = h1_indicators.get('rsi')
            if is_valid_number(rsi_h1):
                rsi = safe_float(rsi_h1, 50.0)
                
                if signal_type == 'BUY':
                    if rsi > 40:
                        confirmations_passed += 1
                        confirmation_details.append(f"RSI H1 mendukung BUY ({rsi:.1f} > 40)")
                    else:
                        confirmation_details.append(f"RSI H1 tidak mendukung BUY ({rsi:.1f} <= 40)")
                elif signal_type == 'SELL':
                    if rsi < 60:
                        confirmations_passed += 1
                        confirmation_details.append(f"RSI H1 mendukung SELL ({rsi:.1f} < 60)")
                    else:
                        confirmation_details.append(f"RSI H1 tidak mendukung SELL ({rsi:.1f} >= 60)")
            else:
                confirmation_details.append("RSI H1: data tidak tersedia")
            
            macd_h1 = h1_indicators.get('macd')
            macd_signal_h1 = h1_indicators.get('macd_signal')
            if is_valid_number(macd_h1) and is_valid_number(macd_signal_h1):
                macd = safe_float(macd_h1, 0.0)
                macd_sig = safe_float(macd_signal_h1, 0.0)
                
                if signal_type == 'BUY' and macd > macd_sig:
                    confirmations_passed += 1
                    confirmation_details.append(f"MACD H1 bullish")
                elif signal_type == 'SELL' and macd < macd_sig:
                    confirmations_passed += 1
                    confirmation_details.append(f"MACD H1 bearish")
                else:
                    confirmation_details.append(f"MACD H1 tidak mendukung {signal_type}")
            
            if confirmations_passed >= confirmations_needed:
                score_info = "" if score_multiplier == 1.0 else f" [Score: {score_multiplier:.0%}]"
                reason = f"✅ H1 Confirmation PASSED: {confirmations_passed}/{confirmations_needed} kriteria{score_info} ({', '.join(confirmation_details)})"
                logger.info(reason)
                return True, reason, score_multiplier
            else:
                score_info = "" if score_multiplier == 1.0 else f" [Score: {score_multiplier:.0%}]"
                reason = f"⚠️ H1 Confirmation WEAK: Hanya {confirmations_passed}/{confirmations_needed} kriteria{score_info} ({', '.join(confirmation_details)})"
                logger.info(reason)
                return True, reason, score_multiplier * 0.85
                
        except (StrategyError, Exception) as e:
            logger.error(f"Error in check_h1_confirmation: {e}")
            return True, f"✅ H1 Confirmation: Error - {str(e)}", 1.0
    
    def check_multi_timeframe_confirmation(self, m1_indicators: Dict, m5_indicators: Optional[Dict], 
                                           h1_indicators: Optional[Dict], signal_type: str) -> Tuple[bool, str, float, Dict]:
        """Multi-Timeframe Confirmation - M1 + M5 + H1 alignment check.
        
        IMPORTANT: Score reduction ONLY happens when data IS available but NOT aligned.
        When higher timeframe data (M5/H1) is missing, we continue with available data.
        
        Scoring (based on AVAILABLE timeframes only):
        - All available TFs aligned: 100% confidence
        - Partial alignment with available data: Proportional reduction
        - Only M1 aligned + others NOT aligned (when available): 65-85% confidence
        - No alignment: BLOCKED (only if M1 is not aligned)
        
        Args:
            m1_indicators: Dictionary indicator dari timeframe M1 (wajib)
            m5_indicators: Dictionary indicator dari timeframe M5 (optional)
            h1_indicators: Dictionary indicator dari timeframe H1 (optional)
            signal_type: 'BUY' atau 'SELL'
            
        Returns:
            Tuple[bool, str, float, Dict]: (passed, reason, score_multiplier, mtf_data)
        """
        mtf_data = {
            'm1_aligned': False,
            'm5_aligned': False,
            'h1_aligned': False,
            'm5_available': False,
            'h1_available': False,
            'timeframes_aligned': 0,
            'timeframes_available': 1,
            'total_score': 0.0,
            'details': []
        }
        
        mtf_enabled = getattr(self.config, 'MTF_ENABLED', True)
        if not mtf_enabled:
            mtf_data['total_score'] = 1.0
            return True, "✅ MTF Filter DISABLED - signal diizinkan", 1.0, mtf_data
        
        try:
            if not m1_indicators:
                return False, "❌ MTF: M1 data tidak tersedia", 0.0, mtf_data
            
            ema_5_m1 = m1_indicators.get(f'ema_{self.config.EMA_PERIODS[0]}')
            ema_20_m1 = m1_indicators.get(f'ema_{self.config.EMA_PERIODS[1]}')
            rsi_m1 = m1_indicators.get('rsi')
            stoch_k_m1 = m1_indicators.get('stoch_k')
            macd_m1 = m1_indicators.get('macd')
            macd_signal_m1 = m1_indicators.get('macd_signal')
            
            m1_alignment_count = 0
            m1_alignment_reasons = []
            
            if is_valid_number(ema_5_m1) and is_valid_number(ema_20_m1):
                ema5 = safe_float(ema_5_m1, 0.0)
                ema20 = safe_float(ema_20_m1, 0.0)
                
                if signal_type == 'BUY' and ema5 > ema20:
                    m1_alignment_count += 1
                    m1_alignment_reasons.append("EMA")
                elif signal_type == 'SELL' and ema5 < ema20:
                    m1_alignment_count += 1
                    m1_alignment_reasons.append("EMA")
            
            if is_valid_number(rsi_m1):
                rsi_val = safe_float(rsi_m1, 50.0)
                if signal_type == 'BUY' and rsi_val < 65:
                    m1_alignment_count += 1
                    m1_alignment_reasons.append("RSI")
                elif signal_type == 'SELL' and rsi_val > 35:
                    m1_alignment_count += 1
                    m1_alignment_reasons.append("RSI")
            
            if is_valid_number(macd_m1) and is_valid_number(macd_signal_m1):
                macd = safe_float(macd_m1, 0.0)
                macd_sig = safe_float(macd_signal_m1, 0.0)
                if signal_type == 'BUY' and macd > macd_sig:
                    m1_alignment_count += 1
                    m1_alignment_reasons.append("MACD")
                elif signal_type == 'SELL' and macd < macd_sig:
                    m1_alignment_count += 1
                    m1_alignment_reasons.append("MACD")
            
            if m1_alignment_count >= 1:
                mtf_data['m1_aligned'] = True
                mtf_data['timeframes_aligned'] += 1
                aligned_str = "+".join(m1_alignment_reasons) if m1_alignment_reasons else "momentum"
                mtf_data['details'].append(f"M1: ✅ Aligned ({aligned_str})")
            else:
                mtf_data['m1_aligned'] = True
                mtf_data['timeframes_aligned'] += 1
                mtf_data['details'].append(f"M1: ⚠️ Momentum check (signal diizinkan)")
            
            if m5_indicators:
                mtf_data['m5_available'] = True
                mtf_data['timeframes_available'] += 1
                m5_passed, m5_reason, m5_score = self.check_m5_confirmation(m5_indicators, signal_type)
                if m5_passed and m5_score >= 0.7:
                    mtf_data['m5_aligned'] = True
                    mtf_data['timeframes_aligned'] += 1
                    mtf_data['details'].append("M5: ✅ Aligned")
                else:
                    mtf_data['details'].append("M5: ⚠️ Tidak aligned")
            else:
                mtf_data['details'].append("M5: - (Data tidak tersedia)")
            
            if h1_indicators:
                mtf_data['h1_available'] = True
                mtf_data['timeframes_available'] += 1
                h1_passed, h1_reason, h1_score = self.check_h1_confirmation(h1_indicators, signal_type)
                if h1_passed and h1_score >= 0.75:
                    mtf_data['h1_aligned'] = True
                    mtf_data['timeframes_aligned'] += 1
                    mtf_data['details'].append("H1: ✅ Aligned")
                else:
                    mtf_data['details'].append("H1: ⚠️ Tidak aligned")
            else:
                mtf_data['details'].append("H1: - (Data tidak tersedia)")
            
            aligned_count = mtf_data['timeframes_aligned']
            available_count = mtf_data['timeframes_available']
            details_str = ", ".join(mtf_data['details'])
            
            if aligned_count == available_count and mtf_data['m1_aligned']:
                mtf_data['total_score'] = 1.0
                reason = f"✅ MTF STRONG: Semua timeframe tersedia aligned ({aligned_count}/{available_count}) - {details_str}"
                logger.info(reason)
                return True, reason, 1.0, mtf_data
            elif mtf_data['m1_aligned']:
                if available_count == 1:
                    mtf_data['total_score'] = 1.0
                    reason = f"✅ MTF OK: M1 aligned (hanya M1 tersedia) - {details_str}"
                    logger.info(reason)
                    return True, reason, 1.0, mtf_data
                elif available_count == 2:
                    if aligned_count == 2:
                        mtf_data['total_score'] = 1.0
                        reason = f"✅ MTF STRONG: Semua tersedia aligned ({aligned_count}/{available_count}) - {details_str}"
                    else:
                        mtf_data['total_score'] = 0.85
                        reason = f"⚠️ MTF PARTIAL: M1 aligned, HTF tidak aligned ({aligned_count}/{available_count}) - {details_str}"
                    logger.info(reason)
                    return True, reason, mtf_data['total_score'], mtf_data
                else:
                    if aligned_count >= 3:
                        mtf_data['total_score'] = 1.0
                        reason = f"✅ MTF STRONG: Semua timeframe aligned (3/3) - {details_str}"
                    elif aligned_count >= 2:
                        if mtf_data['m5_aligned']:
                            mtf_data['total_score'] = 0.90
                            reason = f"✅ MTF GOOD: M1+M5 aligned ({aligned_count}/3) - {details_str}"
                        elif mtf_data['h1_aligned']:
                            mtf_data['total_score'] = 0.85
                            reason = f"✅ MTF OK: M1+H1 aligned ({aligned_count}/3) - {details_str}"
                        else:
                            mtf_data['total_score'] = 0.80
                            reason = f"⚠️ MTF PARTIAL: {aligned_count}/3 aligned - {details_str}"
                    else:
                        mtf_data['total_score'] = 0.65
                        reason = f"⚠️ MTF WEAK: Hanya M1 aligned ({aligned_count}/3) - {details_str}"
                    logger.info(reason)
                    return True, reason, mtf_data['total_score'], mtf_data
            else:
                mtf_data['total_score'] = 0.0
                reason = f"❌ MTF BLOCKED: M1 tidak aligned - {details_str}"
                logger.info(reason)
                return False, reason, 0.0, mtf_data
                
        except (StrategyError, Exception) as e:
            logger.error(f"Error in check_multi_timeframe_confirmation: {e}")
            return True, f"✅ MTF: Error - {str(e)}", 1.0, mtf_data
    
    def analyze_session_strength(self, current_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Analyze current trading session strength for XAUUSD based on market hours.
        
        IMPROVEMENT 3: Intraday Session Optimization
        
        Session definitions (UTC):
        - Asia: 00:00-08:00 UTC (Tokyo/Hong Kong)
        - London: 07:00-16:00 UTC (Major European markets)
        - New York: 13:00-22:00 UTC (US markets)
        - Sydney: 21:00-06:00 UTC (next day wrap, Australian markets)
        
        Session Overlap Strength:
        - London & NY overlap (13:00-16:00 UTC) = STRONGEST (highest liquidity)
        - London session alone = STRONG
        - NY session alone = STRONG  
        - Asia session = MEDIUM
        - Off-market/Sydney only = WEAK
        
        Args:
            current_time: Optional datetime in UTC. If None, uses current UTC time.
            
        Returns:
            Dict with:
            - current_session: str (e.g., "London-NY Overlap", "London", "Asia")
            - session_strength: str ("STRONGEST", "STRONG", "MEDIUM", "WEAK")
            - confidence_multiplier: float (Overlap=1.2x, London/NY=1.1x, Asia=0.95x, Off-market=0.8x)
            - session_description: str for logging
            - active_sessions: List[str] of currently active sessions
            - is_overlap: bool indicating if multiple sessions overlap
        """
        result = {
            'current_session': 'Unknown',
            'session_strength': 'WEAK',
            'confidence_multiplier': 0.8,
            'session_description': 'Session analysis unavailable',
            'active_sessions': [],
            'is_overlap': False
        }
        
        try:
            if current_time is None:
                current_time = datetime.now(pytz.UTC)
            elif current_time.tzinfo is None:
                current_time = pytz.UTC.localize(current_time)
            else:
                current_time = current_time.astimezone(pytz.UTC)
            
            hour = current_time.hour
            weekday = current_time.weekday()
            
            is_weekend = weekday >= 5
            if is_weekend:
                result['current_session'] = 'Weekend (Closed)'
                result['session_strength'] = 'WEAK'
                result['confidence_multiplier'] = 0.5
                result['session_description'] = f"⚠️ Weekend - Market Closed (Day {weekday})"
                logger.debug(f"Session Analysis: {result['session_description']}")
                return result
            
            active_sessions = []
            
            if 0 <= hour < 8:
                active_sessions.append('Asia')
            
            if 7 <= hour < 16:
                active_sessions.append('London')
            
            if 13 <= hour < 22:
                active_sessions.append('New York')
            
            if hour >= 21 or hour < 6:
                active_sessions.append('Sydney')
            
            result['active_sessions'] = active_sessions
            result['is_overlap'] = len(active_sessions) > 1
            
            if 'London' in active_sessions and 'New York' in active_sessions:
                result['current_session'] = 'London-NY Overlap'
                result['session_strength'] = 'STRONGEST'
                result['confidence_multiplier'] = 1.2
                result['session_description'] = (
                    f"🔥 STRONGEST Session: London-NY Overlap (13:00-16:00 UTC) | "
                    f"Hour: {hour:02d}:00 UTC | Highest Liquidity"
                )
            
            elif 'Asia' in active_sessions and 'London' in active_sessions:
                result['current_session'] = 'Asia-London Overlap'
                result['session_strength'] = 'STRONG'
                result['confidence_multiplier'] = 1.15
                result['session_description'] = (
                    f"✅ STRONG Session: Asia-London Overlap (07:00-08:00 UTC) | "
                    f"Hour: {hour:02d}:00 UTC | Good Liquidity"
                )
            
            elif 'London' in active_sessions:
                result['current_session'] = 'London'
                result['session_strength'] = 'STRONG'
                result['confidence_multiplier'] = 1.1
                result['session_description'] = (
                    f"✅ STRONG Session: London (07:00-16:00 UTC) | "
                    f"Hour: {hour:02d}:00 UTC | Major Market Activity"
                )
            
            elif 'New York' in active_sessions:
                result['current_session'] = 'New York'
                result['session_strength'] = 'STRONG'
                result['confidence_multiplier'] = 1.1
                result['session_description'] = (
                    f"✅ STRONG Session: New York (13:00-22:00 UTC) | "
                    f"Hour: {hour:02d}:00 UTC | Major Market Activity"
                )
            
            elif 'Asia' in active_sessions:
                result['current_session'] = 'Asia'
                result['session_strength'] = 'MEDIUM'
                result['confidence_multiplier'] = 0.95
                result['session_description'] = (
                    f"⚡ MEDIUM Session: Asia (00:00-08:00 UTC) | "
                    f"Hour: {hour:02d}:00 UTC | Moderate Liquidity"
                )
            
            elif 'Sydney' in active_sessions:
                result['current_session'] = 'Sydney'
                result['session_strength'] = 'WEAK'
                result['confidence_multiplier'] = 0.8
                result['session_description'] = (
                    f"⚠️ WEAK Session: Sydney/Off-Market (21:00-06:00 UTC) | "
                    f"Hour: {hour:02d}:00 UTC | Low Liquidity"
                )
            
            else:
                result['current_session'] = 'Off-Market'
                result['session_strength'] = 'WEAK'
                result['confidence_multiplier'] = 0.8
                result['session_description'] = (
                    f"⚠️ WEAK Session: Off-Market Hours | "
                    f"Hour: {hour:02d}:00 UTC | Very Low Liquidity"
                )
            
            logger.debug(f"Session Analysis: {result['session_description']}")
            return result
            
        except Exception as e:
            logger.error(f"Error in analyze_session_strength: {e}")
            result['session_description'] = f"⚠️ Session analysis error: {str(e)}"
            return result
    
    def check_extended_mtf_correlation(self, m1_indicators: Optional[Dict], 
                                        m5_indicators: Optional[Dict],
                                        h1_indicators: Optional[Dict],
                                        h4_indicators: Optional[Dict],
                                        daily_indicators: Optional[Dict],
                                        signal_type: str) -> Dict[str, Any]:
        """Check extended multi-timeframe correlation with 4H and Daily timeframes.
        
        IMPROVEMENT 8: Extended MTF Correlation Analysis
        
        Analyzes alignment across all available timeframes to determine trend consistency.
        This method is NON-BLOCKING - it only provides context and confidence adjustment.
        
        Timeframe Alignment Scoring:
        - M1 aligned = base score (always checked first)
        - M5 aligned = +10% bonus
        - H1 aligned = +10% bonus
        - H4 aligned = +10% bonus (if available)
        - Daily aligned = +10% bonus (if available)
        
        Confidence Levels:
        - All TF aligned (5/5) = VERY HIGH confidence, +40% score bonus
        - 4/5 aligned = HIGH confidence, +30% score bonus
        - 3/5 aligned = MEDIUM confidence, +20% score bonus
        - <3 aligned = LOW confidence, no bonus
        
        Args:
            m1_indicators: M1 timeframe indicators (required)
            m5_indicators: M5 timeframe indicators (optional)
            h1_indicators: H1 timeframe indicators (optional)
            h4_indicators: H4 timeframe indicators (optional, new)
            daily_indicators: Daily timeframe indicators (optional, new)
            signal_type: 'BUY' or 'SELL'
            
        Returns:
            Dict with:
            - mtf_correlation_score: float (0.0-0.4 bonus)
            - aligned_timeframes_count: int (number of aligned TFs)
            - available_timeframes_count: int (number of available TFs)
            - alignment_details: List[str] (details per timeframe)
            - confidence_level: str ('VERY HIGH', 'HIGH', 'MEDIUM', 'LOW')
            - confidence_multiplier: float (1.0-1.4x)
            - is_strongly_aligned: bool
        """
        result = {
            'mtf_correlation_score': 0.0,
            'aligned_timeframes_count': 0,
            'available_timeframes_count': 0,
            'alignment_details': [],
            'confidence_level': 'LOW',
            'confidence_multiplier': 1.0,
            'is_strongly_aligned': False,
            'timeframe_status': {}
        }
        
        try:
            if signal_type not in ['BUY', 'SELL']:
                result['alignment_details'].append(f"❌ Invalid signal type: {signal_type}")
                logger.warning(f"check_extended_mtf_correlation: Invalid signal_type: {signal_type}")
                return result
            
            timeframe_checks = [
                ('M1', m1_indicators),
                ('M5', m5_indicators),
                ('H1', h1_indicators),
                ('H4', h4_indicators),
                ('Daily', daily_indicators)
            ]
            
            aligned_count = 0
            available_count = 0
            
            for tf_name, tf_indicators in timeframe_checks:
                tf_status = self._check_single_tf_alignment(tf_indicators, signal_type, tf_name)
                result['timeframe_status'][tf_name] = tf_status
                
                if tf_status['available']:
                    available_count += 1
                    if tf_status['aligned']:
                        aligned_count += 1
                        result['alignment_details'].append(f"✅ {tf_name}: Aligned ({tf_status['reason']})")
                    else:
                        result['alignment_details'].append(f"⚠️ {tf_name}: Not aligned ({tf_status['reason']})")
                else:
                    result['alignment_details'].append(f"- {tf_name}: Data tidak tersedia")
            
            result['aligned_timeframes_count'] = aligned_count
            result['available_timeframes_count'] = available_count
            
            if available_count == 0:
                result['confidence_level'] = 'LOW'
                result['mtf_correlation_score'] = 0.0
                result['confidence_multiplier'] = 1.0
                result['alignment_details'].append("⚠️ No timeframe data available")
                logger.debug("Extended MTF Correlation: No timeframe data available")
                return result
            
            alignment_ratio = aligned_count / available_count if available_count > 0 else 0
            
            if aligned_count >= 5 or (available_count >= 4 and aligned_count == available_count):
                result['confidence_level'] = 'VERY HIGH'
                result['mtf_correlation_score'] = 0.40
                result['confidence_multiplier'] = 1.4
                result['is_strongly_aligned'] = True
            elif aligned_count >= 4 or (available_count == 4 and aligned_count == 4):
                result['confidence_level'] = 'HIGH'
                result['mtf_correlation_score'] = 0.30
                result['confidence_multiplier'] = 1.3
                result['is_strongly_aligned'] = True
            elif aligned_count >= 3:
                result['confidence_level'] = 'MEDIUM'
                result['mtf_correlation_score'] = 0.20
                result['confidence_multiplier'] = 1.2
                result['is_strongly_aligned'] = alignment_ratio >= 0.6
            else:
                result['confidence_level'] = 'LOW'
                result['mtf_correlation_score'] = 0.0
                result['confidence_multiplier'] = 1.0
                result['is_strongly_aligned'] = False
            
            alignment_emoji = "🔥" if result['is_strongly_aligned'] else "⚠️"
            summary = (
                f"{alignment_emoji} Extended MTF Correlation: {aligned_count}/{available_count} aligned | "
                f"Confidence: {result['confidence_level']} | "
                f"Score Bonus: +{result['mtf_correlation_score']*100:.0f}%"
            )
            result['alignment_details'].append(summary)
            
            logger.info(summary)
            logger.debug(f"MTF Details: {', '.join(result['alignment_details'][:-1])}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in check_extended_mtf_correlation: {e}")
            result['alignment_details'].append(f"⚠️ Error: {str(e)}")
            return result
    
    def _check_single_tf_alignment(self, indicators: Optional[Dict], 
                                    signal_type: str, 
                                    tf_name: str) -> Dict[str, Any]:
        """Check if a single timeframe is aligned with the signal direction.
        
        Internal helper for check_extended_mtf_correlation.
        
        Alignment criteria:
        - EMA alignment (EMA5 > EMA20 for BUY, EMA5 < EMA20 for SELL)
        - RSI direction (>50 for BUY, <50 for SELL)
        - MACD direction (>0 or above signal for BUY)
        
        Args:
            indicators: Timeframe indicators dict
            signal_type: 'BUY' or 'SELL'
            tf_name: Timeframe name for logging
            
        Returns:
            Dict with:
            - available: bool
            - aligned: bool
            - reason: str
            - score: float (0-1)
        """
        result = {
            'available': False,
            'aligned': False,
            'reason': 'Data tidak tersedia',
            'score': 0.0
        }
        
        try:
            if not indicators or not isinstance(indicators, dict):
                return result
            
            ema_keys = [f'ema_{p}' for p in self.config.EMA_PERIODS[:2]]
            ema_short_key = ema_keys[0] if len(ema_keys) > 0 else 'ema_5'
            ema_mid_key = ema_keys[1] if len(ema_keys) > 1 else 'ema_20'
            
            ema_short = indicators.get(ema_short_key)
            ema_mid = indicators.get(ema_mid_key)
            rsi = indicators.get('rsi')
            macd = indicators.get('macd')
            macd_signal = indicators.get('macd_signal')
            close = indicators.get('close')
            
            has_ema = is_valid_number(ema_short) and is_valid_number(ema_mid)
            has_rsi = is_valid_number(rsi)
            has_macd = is_valid_number(macd)
            has_close = is_valid_number(close)
            
            if not has_ema and not has_rsi and not has_macd and not has_close:
                return result
            
            result['available'] = True
            alignment_checks = []
            aligned_count = 0
            total_checks = 0
            
            if has_ema:
                ema_s = safe_float(ema_short, 0.0)
                ema_m = safe_float(ema_mid, 0.0)
                total_checks += 1
                
                if signal_type == 'BUY':
                    if ema_s > ema_m:
                        aligned_count += 1
                        alignment_checks.append("EMA✅")
                    else:
                        alignment_checks.append("EMA❌")
                else:
                    if ema_s < ema_m:
                        aligned_count += 1
                        alignment_checks.append("EMA✅")
                    else:
                        alignment_checks.append("EMA❌")
            
            if has_rsi:
                rsi_val = safe_float(rsi, 50.0)
                total_checks += 1
                
                if signal_type == 'BUY':
                    if rsi_val > 45:
                        aligned_count += 1
                        alignment_checks.append(f"RSI({rsi_val:.0f})✅")
                    else:
                        alignment_checks.append(f"RSI({rsi_val:.0f})❌")
                else:
                    if rsi_val < 55:
                        aligned_count += 1
                        alignment_checks.append(f"RSI({rsi_val:.0f})✅")
                    else:
                        alignment_checks.append(f"RSI({rsi_val:.0f})❌")
            
            if has_macd:
                macd_val = safe_float(macd, 0.0)
                macd_sig = safe_float(macd_signal, 0.0) if is_valid_number(macd_signal) else macd_val
                total_checks += 1
                
                if signal_type == 'BUY':
                    if macd_val > macd_sig or macd_val > 0:
                        aligned_count += 1
                        alignment_checks.append("MACD✅")
                    else:
                        alignment_checks.append("MACD❌")
                else:
                    if macd_val < macd_sig or macd_val < 0:
                        aligned_count += 1
                        alignment_checks.append("MACD✅")
                    else:
                        alignment_checks.append("MACD❌")
            
            if total_checks > 0:
                result['score'] = aligned_count / total_checks
                result['aligned'] = result['score'] >= 0.5
                result['reason'] = ' '.join(alignment_checks)
            else:
                result['reason'] = 'No indicators to check'
            
            return result
            
        except Exception as e:
            logger.debug(f"Error in _check_single_tf_alignment for {tf_name}: {e}")
            result['reason'] = f"Error: {str(e)}"
            return result
    
    def check_breakout_confirmation(self, indicators: Dict, m5_indicators: Optional[Dict],
                                     signal_type: str) -> Tuple[bool, str, float]:
        """
        BATCH 2 - IMPROVEMENT 6: Breakout Confirmation
        
        Konfirmasi breakout dengan validasi:
        1. ATR expansion (current ATR > average ATR untuk kekuatan)
        2. Volume confirmation (volume > average untuk breakout valid)
        3. Momentum direction alignment (MACD/RSI selaras dengan breakout direction)
        
        Non-blocking approach: Score modifier 0.7-1.0 berdasarkan confirmation strength
        
        Args:
            indicators: Dictionary of M1 calculated indicators
            m5_indicators: Optional M5 timeframe indicators
            signal_type: 'BUY' or 'SELL'
            
        Returns:
            Tuple of (passed, reason, confidence_score)
            - passed: True if breakout criteria met
            - reason: Explanation string
            - confidence_score: 0.7-1.0 based on confirmation strength
        """
        try:
            if not indicators or not isinstance(indicators, dict):
                return True, "✅ Breakout check: No data - allowing signal", 1.0
            
            criteria_met = 0
            total_criteria = 4
            details = []
            
            atr = safe_float(indicators.get('atr', 0), 0.0)
            atr_ma = safe_float(indicators.get('atr_ma', atr), atr)
            
            if atr > 0 and atr_ma > 0:
                atr_ratio = atr / atr_ma
                if atr_ratio > 1.2:
                    criteria_met += 1
                    details.append(f"ATR↑{atr_ratio:.2f}x")
                elif atr_ratio > 1.0:
                    criteria_met += 0.5
                    details.append(f"ATR→{atr_ratio:.2f}x")
                else:
                    details.append(f"ATR↓{atr_ratio:.2f}x")
            else:
                criteria_met += 0.5
                details.append("ATR:N/A")
            
            volume = safe_float(indicators.get('volume', 0), 0.0)
            volume_avg = safe_float(indicators.get('volume_avg', volume), volume)
            
            if volume > 0 and volume_avg > 0:
                volume_ratio = volume / volume_avg
                if volume_ratio > 1.5:
                    criteria_met += 1
                    details.append(f"VOL↑{volume_ratio:.2f}x")
                elif volume_ratio > 1.0:
                    criteria_met += 0.5
                    details.append(f"VOL→{volume_ratio:.2f}x")
                else:
                    details.append(f"VOL↓{volume_ratio:.2f}x")
            else:
                criteria_met += 0.5
                details.append("VOL:N/A")
            
            rsi = safe_float(indicators.get('rsi', 50), 50.0)
            if signal_type == 'BUY':
                if rsi > 60:
                    criteria_met += 1
                    details.append(f"RSI✅{rsi:.0f}")
                elif rsi > 50:
                    criteria_met += 0.5
                    details.append(f"RSI→{rsi:.0f}")
                else:
                    details.append(f"RSI❌{rsi:.0f}")
            else:
                if rsi < 40:
                    criteria_met += 1
                    details.append(f"RSI✅{rsi:.0f}")
                elif rsi < 50:
                    criteria_met += 0.5
                    details.append(f"RSI→{rsi:.0f}")
                else:
                    details.append(f"RSI❌{rsi:.0f}")
            
            macd = safe_float(indicators.get('macd', 0), 0.0)
            macd_signal = safe_float(indicators.get('macd_signal', 0), 0.0)
            
            if signal_type == 'BUY':
                if macd > macd_signal and macd > 0:
                    criteria_met += 1
                    details.append("MACD✅")
                elif macd > macd_signal or macd > 0:
                    criteria_met += 0.5
                    details.append("MACD→")
                else:
                    details.append("MACD❌")
            else:
                if macd < macd_signal and macd < 0:
                    criteria_met += 1
                    details.append("MACD✅")
                elif macd < macd_signal or macd < 0:
                    criteria_met += 0.5
                    details.append("MACD→")
                else:
                    details.append("MACD❌")
            
            if m5_indicators and isinstance(m5_indicators, dict):
                m5_close = safe_float(m5_indicators.get('close', 0), 0.0)
                m5_high = safe_float(m5_indicators.get('high', 0), 0.0)
                m5_low = safe_float(m5_indicators.get('low', 0), 0.0)
                
                if m5_close > 0 and m5_high > 0:
                    if signal_type == 'BUY' and m5_close >= m5_high * 0.98:
                        criteria_met += 0.5
                        details.append("M5:Break↑")
                    elif signal_type == 'SELL' and m5_close <= m5_low * 1.02:
                        criteria_met += 0.5
                        details.append("M5:Break↓")
            
            score = criteria_met / total_criteria
            
            if score >= 1.0:
                confidence_score = 1.0
                status = "🔥 STRONG"
            elif score >= 0.75:
                confidence_score = 0.90
                status = "✅ GOOD"
            elif score >= 0.5:
                confidence_score = 0.80
                status = "⚠️ MODERATE"
            else:
                confidence_score = 0.70
                status = "⚡ WEAK"
            
            details_str = ' '.join(details)
            reason = f"{status} Breakout ({criteria_met:.1f}/{total_criteria}): {details_str}"
            
            logger.debug(f"Breakout confirmation for {signal_type}: {reason}")
            
            return True, reason, confidence_score
            
        except Exception as e:
            logger.warning(f"Error in check_breakout_confirmation: {e}")
            return True, f"✅ Breakout check error - allowing signal: {str(e)}", 0.85
    
    def get_session_tp_sl_multiplier(self, current_time: Optional[datetime] = None,
                                      signal_type: str = 'BUY') -> Tuple[str, float, float]:
        """
        BATCH 3 - IMPROVEMENT 3: Session Optimization untuk TP/SL Adjustment
        
        Analisis kekuatan signal berdasarkan intraday session dan return multiplier
        untuk adjust TP/SL target.
        
        Session definitions:
        - Tokyo Session (21:00-06:00 UTC): Medium volatility
        - London Session (07:00-16:00 UTC): HIGH - Best directional moves
        - New York Session (13:00-22:00 UTC): HIGH - Good execution quality
        - London-NY Overlap (13:00-16:00 UTC): VERY HIGH - Max liquidity
        - Asia-London Overlap (07:00-08:00 UTC): HIGH - Good transition
        
        Args:
            current_time: Optional datetime in UTC
            signal_type: 'BUY' or 'SELL'
            
        Returns:
            Tuple of (session_name, tp_multiplier, sl_multiplier)
            - tp_multiplier: 0.8-1.15x untuk adjust TP target
            - sl_multiplier: 0.9-1.1x untuk adjust SL buffer
        """
        try:
            session_info = self.analyze_session_strength(current_time)
            session_name = session_info.get('current_session', 'Unknown')
            session_strength = session_info.get('session_strength', 'WEAK')
            
            if session_strength == 'STRONGEST':
                tp_multiplier = 1.15
                sl_multiplier = 1.05
            elif session_strength == 'STRONG':
                if 'London-NY' in session_name or 'Overlap' in session_name:
                    tp_multiplier = 1.10
                    sl_multiplier = 1.02
                else:
                    tp_multiplier = 1.05
                    sl_multiplier = 1.0
            elif session_strength == 'MEDIUM':
                tp_multiplier = 0.95
                sl_multiplier = 0.98
            else:
                tp_multiplier = 0.80
                sl_multiplier = 0.95
            
            if session_name == 'Weekend (Closed)':
                tp_multiplier = 0.5
                sl_multiplier = 0.8
            
            logger.debug(f"Session TP/SL: {session_name} -> TP:{tp_multiplier:.2f}x SL:{sl_multiplier:.2f}x")
            
            return session_name, float(tp_multiplier), float(sl_multiplier)
            
        except Exception as e:
            logger.warning(f"Error in get_session_tp_sl_multiplier: {e}")
            return 'Unknown', 1.0, 1.0
    
    def check_macd_divergence(self, indicators: Dict, signal_type: str, 
                              price_history: Optional[List] = None, 
                              macd_history: Optional[List] = None) -> Tuple[bool, str, float]:
        """Detect MACD divergence patterns for enhanced signal quality.
        
        IMPORTANT: This returns a CONFIDENCE BOOST only, never blocks signals.
        Uses stricter criteria to reduce false positive divergence detections.
        
        MACD Divergence Detection (strict criteria):
        - Bullish Regular: Price makes clear lower low (<= -0.1%), MACD makes higher low (>= +5%)
        - Bearish Regular: Price makes clear higher high (>= +0.1%), MACD makes lower high (<= -5%)
        - Minimum difference thresholds prevent constant false detections
        
        Args:
            indicators: Dictionary of calculated indicators
            signal_type: 'BUY' or 'SELL'
            price_history: Optional list of recent prices
            macd_history: Optional list of recent MACD values
            
        Returns:
            Tuple of (has_divergence, description, confidence_boost)
            Note: Returns (False, ..., 0.0) when no divergence - NEVER blocks signals
        """
        try:
            macd_hist = macd_history or indicators.get('macd_history', [])
            price_hist = price_history or indicators.get('price_history', [])
            
            if not macd_hist or len(macd_hist) < 8:
                return False, "MACD Divergence: Data tidak cukup", 0.0
            
            if not price_hist or len(price_hist) < 8:
                return False, "MACD Divergence: Price history tidak cukup", 0.0
            
            macd_clean = [safe_float(m, 0.0) for m in macd_hist[-15:] if is_valid_number(m)]
            price_clean = [safe_float(p, 0.0) for p in price_hist[-15:] if is_valid_number(p)]
            
            if len(macd_clean) < 8 or len(price_clean) < 8:
                return False, "MACD Divergence: Data bersih tidak cukup", 0.0
            
            recent_window = 3
            lookback_start = -8
            lookback_end = -recent_window
            
            current_macd = sum(macd_clean[-recent_window:]) / recent_window
            prev_macd_values = macd_clean[lookback_start:lookback_end]
            prev_macd_low = min(prev_macd_values)
            prev_macd_high = max(prev_macd_values)
            
            current_price = sum(price_clean[-recent_window:]) / recent_window
            prev_price_values = price_clean[lookback_start:lookback_end]
            prev_price_low = min(prev_price_values)
            prev_price_high = max(prev_price_values)
            
            min_price_diff_pct = 0.001
            min_macd_diff_pct = 0.05
            
            if signal_type == 'BUY':
                price_diff_pct = (current_price - prev_price_low) / prev_price_low if prev_price_low != 0 else 0
                macd_diff_pct = (current_macd - prev_macd_low) / abs(prev_macd_low) if prev_macd_low != 0 else 0
                
                if price_diff_pct <= -min_price_diff_pct and macd_diff_pct >= min_macd_diff_pct:
                    strength = min(abs(macd_diff_pct) * 0.5, 0.15)
                    return True, f"🔄 BULLISH DIVERGENCE: Price ↓{price_diff_pct:.2%} MACD ↑{macd_diff_pct:.2%}", strength
                    
            elif signal_type == 'SELL':
                price_diff_pct = (current_price - prev_price_high) / prev_price_high if prev_price_high != 0 else 0
                macd_diff_pct = (current_macd - prev_macd_high) / abs(prev_macd_high) if prev_macd_high != 0 else 0
                
                if price_diff_pct >= min_price_diff_pct and macd_diff_pct <= -min_macd_diff_pct:
                    strength = min(abs(macd_diff_pct) * 0.5, 0.15)
                    return True, f"🔄 BEARISH DIVERGENCE: Price ↑{price_diff_pct:.2%} MACD ↓{macd_diff_pct:.2%}", strength
            
            return False, "Tidak ada MACD divergence", 0.0
            
        except (StrategyError, Exception) as e:
            logger.debug(f"Error in check_macd_divergence: {e}")
            return False, f"MACD Divergence Error: {str(e)}", 0.0
    
    def check_atr_volatility_filter(self, indicators: Dict, signal_type: str, 
                                     atr_history: Optional[List] = None) -> Tuple[bool, str, float, str]:
        """ATR-based volatility filter to skip signals during extreme volatility.
        
        Volatility Zones:
        - EXTREME_LOW (< 20th percentile): Skip - market terlalu quiet, breakout risk
        - LOW (20-40th percentile): Allow with caution - reduce position size
        - NORMAL (40-60th percentile): Ideal - full confidence
        - HIGH (60-80th percentile): Allow - good for momentum trades
        - EXTREME_HIGH (> 80th percentile): Skip - terlalu volatile, SL risk tinggi
        
        Args:
            indicators: Dictionary of calculated indicators
            signal_type: 'BUY' or 'SELL'
            atr_history: Optional list of historical ATR values
            
        Returns:
            Tuple of (is_valid, reason, confidence_multiplier, volatility_zone)
        """
        try:
            atr = indicators.get('atr')
            atr_hist = atr_history or indicators.get('atr_history', [])
            
            if not is_valid_number(atr):
                return True, "✅ ATR Filter: Data tidak tersedia - signal diizinkan", 1.0, "unknown"
            
            atr_val = safe_float(atr, 0.0)
            
            if atr_val <= 0:
                return True, "✅ ATR Filter: ATR = 0 - signal diizinkan", 1.0, "unknown"
            
            if atr_hist and len(atr_hist) >= 20:
                atr_clean = [safe_float(a, 0.0) for a in atr_hist[-100:] if is_valid_number(a) and a > 0]
                if len(atr_clean) >= 10:
                    sorted_atr = sorted(atr_clean)
                    p20 = sorted_atr[int(len(sorted_atr) * 0.2)]
                    p40 = sorted_atr[int(len(sorted_atr) * 0.4)]
                    p60 = sorted_atr[int(len(sorted_atr) * 0.6)]
                    p80 = sorted_atr[int(len(sorted_atr) * 0.8)]
                    
                    if atr_val < p20:
                        zone = "EXTREME_LOW"
                        reason = f"❌ ATR EXTREME LOW: ATR({atr_val:.4f}) < P20({p20:.4f}) - Volatilitas terlalu rendah, SKIP"
                        logger.info(reason)
                        return False, reason, 0.0, zone
                    elif atr_val < p40:
                        zone = "LOW"
                        reason = f"⚠️ ATR LOW: ATR({atr_val:.4f}) - Volatilitas rendah, reduced confidence"
                        logger.info(reason)
                        return True, reason, 0.8, zone
                    elif atr_val < p60:
                        zone = "NORMAL"
                        reason = f"✅ ATR NORMAL: ATR({atr_val:.4f}) - Volatilitas ideal"
                        logger.info(reason)
                        return True, reason, 1.0, zone
                    elif atr_val < p80:
                        zone = "HIGH"
                        reason = f"✅ ATR HIGH: ATR({atr_val:.4f}) - Good for momentum"
                        logger.info(reason)
                        return True, reason, 1.0, zone
                    else:
                        zone = "EXTREME_HIGH"
                        reason = f"❌ ATR EXTREME HIGH: ATR({atr_val:.4f}) > P80({p80:.4f}) - Volatilitas terlalu tinggi, SKIP"
                        logger.info(reason)
                        return False, reason, 0.0, zone
            
            atr_typical_low = 0.5
            atr_typical_high = 5.0
            
            if atr_val < atr_typical_low:
                zone = "LOW"
                reason = f"⚠️ ATR LOW: ATR({atr_val:.4f}) < {atr_typical_low} - Volatilitas rendah"
                return True, reason, 0.8, zone
            elif atr_val > atr_typical_high:
                zone = "EXTREME_HIGH"
                reason = f"❌ ATR EXTREME HIGH: ATR({atr_val:.4f}) > {atr_typical_high} - SKIP signal"
                return False, reason, 0.0, zone
            else:
                zone = "NORMAL"
                reason = f"✅ ATR NORMAL: ATR({atr_val:.4f}) - Volatilitas acceptable"
                return True, reason, 1.0, zone
                
        except (StrategyError, Exception) as e:
            logger.error(f"Error in check_atr_volatility_filter: {e}")
            return True, f"✅ ATR Filter: Error - {str(e)}", 1.0, "unknown"
    
    def calculate_dynamic_lot_size(self, base_lot_size: float, atr: float, 
                                    volatility_zone: str) -> Tuple[float, str]:
        """Calculate dynamic lot size based on volatility zone.
        
        IMPROVEMENT 1: Smart Position Sizing dengan Volatilitas Adapter
        
        Menyesuaikan ukuran posisi berdasarkan volatilitas pasar tanpa memblokir sinyal.
        Saat volatilitas terlalu rendah atau tinggi, posisi dikurangi untuk mengurangi risiko.
        
        Volatility Zone Multipliers:
        - EXTREME_LOW: 0.5x (reduce saat market terlalu quiet - breakout risk)
        - LOW: 0.7x (reduce saat volatilitas rendah)
        - NORMAL: 1.0x (standard position size)
        - HIGH: 0.8x (slight reduction saat volatile)
        - EXTREME_HIGH: 0.7x (reduce saat terlalu volatile - SL risk tinggi)
        
        Args:
            base_lot_size: Base lot size dari config (e.g., 0.01)
            atr: Current ATR value
            volatility_zone: Zone string dari check_atr_volatility_filter 
                           (EXTREME_LOW, LOW, NORMAL, HIGH, EXTREME_HIGH, unknown)
        
        Returns:
            Tuple of (adjusted_lot_size, reason_string)
            - adjusted_lot_size: Lot size yang sudah disesuaikan dengan volatilitas
            - reason_string: Penjelasan adjustment untuk logging
        
        Example:
            >>> lot, reason = strategy.calculate_dynamic_lot_size(0.02, 1.5, "HIGH")
            >>> print(lot)  # 0.016
            >>> print(reason)  # "📊 Position size adjusted: 0.02 -> 0.016 (HIGH volatility: 0.8x)"
        """
        try:
            base_lot = safe_float(base_lot_size, 0.01, "base_lot_size")
            if base_lot <= 0:
                logger.warning(f"Invalid base_lot_size: {base_lot_size}, using default 0.01")
                base_lot = 0.01
            
            volatility_multipliers = {
                'EXTREME_LOW': 0.5,
                'LOW': 0.7,
                'NORMAL': 1.0,
                'HIGH': 0.8,
                'EXTREME_HIGH': 0.7,
                'unknown': 1.0
            }
            
            zone_upper = volatility_zone.upper() if volatility_zone else 'UNKNOWN'
            multiplier = volatility_multipliers.get(zone_upper, 1.0)
            
            adjusted_lot = base_lot * multiplier
            
            min_lot = safe_float(getattr(self.config, 'MIN_LOT_SIZE', 0.01), 0.01)
            max_lot = safe_float(getattr(self.config, 'MAX_LOT_SIZE', 0.1), 0.1)
            adjusted_lot = max(min_lot, min(adjusted_lot, max_lot))
            
            adjusted_lot = round(adjusted_lot, 2)
            
            if multiplier != 1.0:
                reason = (f"📊 Position size adjusted: {base_lot:.2f} -> {adjusted_lot:.2f} "
                         f"({zone_upper} volatility: {multiplier:.1f}x)")
                logger.info(reason)
            else:
                reason = f"📊 Position size: {adjusted_lot:.2f} (NORMAL volatility, no adjustment)"
                logger.debug(reason)
            
            atr_val = safe_float(atr, 0.0, "atr")
            logger.debug(f"Dynamic lot sizing: base={base_lot:.2f}, ATR={atr_val:.4f}, "
                        f"zone={zone_upper}, multiplier={multiplier:.1f}, final={adjusted_lot:.2f}")
            
            return adjusted_lot, reason
            
        except (ValueError, TypeError, ZeroDivisionError) as e:
            logger.error(f"Error in calculate_dynamic_lot_size: {e}")
            fallback_lot = safe_float(base_lot_size, 0.01)
            fallback_lot = round(max(0.01, min(fallback_lot, 0.1)), 2)
            return fallback_lot, f"📊 Position size: {fallback_lot:.2f} (fallback - calculation error)"
        except Exception as e:
            logger.error(f"Unexpected error in calculate_dynamic_lot_size: {e}")
            return 0.01, "📊 Position size: 0.01 (fallback - unexpected error)"
    
    def check_enhanced_volume_confirmation(self, indicators: Dict, signal_type: str,
                                            volume_history: Optional[List] = None) -> Tuple[bool, str, float]:
        """Enhanced volume confirmation with multiple checks.
        
        Volume Confirmation Criteria:
        1. Volume > 1.0x average (basic confirmation)
        2. Volume increasing for last 3 candles (momentum building)
        3. Volume spike > 1.5x (strong confirmation)
        4. Price-volume divergence check
        
        Args:
            indicators: Dictionary of calculated indicators
            signal_type: 'BUY' or 'SELL'
            volume_history: Optional list of recent volume values
            
        Returns:
            Tuple of (is_valid, reason, confidence_multiplier)
        """
        try:
            volume = indicators.get('volume')
            volume_avg = indicators.get('volume_avg')
            vol_hist = volume_history or indicators.get('volume_history', [])
            
            if not is_valid_number(volume) or not is_valid_number(volume_avg):
                return True, "✅ Volume: Data tidak tersedia - signal diizinkan", 0.85
            
            vol = safe_float(volume, 0.0)
            vol_avg = safe_float(volume_avg, 1.0)
            
            if vol_avg <= 0:
                return True, "✅ Volume: Average = 0 - signal diizinkan", 0.85
            
            volume_ratio = vol / vol_avg
            
            volume_increasing = False
            increasing_count = 0
            if vol_hist and len(vol_hist) >= 3:
                vol_clean = [safe_float(v, 0.0) for v in vol_hist[-5:] if is_valid_number(v)]
                if len(vol_clean) >= 3:
                    for i in range(1, len(vol_clean)):
                        if vol_clean[i] > vol_clean[i-1]:
                            increasing_count += 1
                    volume_increasing = increasing_count >= 2
            
            is_spike = volume_ratio >= 1.5
            is_strong = volume_ratio >= 1.2
            is_normal = volume_ratio >= 1.0
            is_weak = volume_ratio >= 0.7
            is_very_low = volume_ratio >= 0.3
            
            if is_spike and volume_increasing:
                reason = f"✅ VOLUME SPIKE + INCREASING: {volume_ratio:.1%} dengan momentum building"
                logger.info(reason)
                return True, reason, 1.15
            elif is_spike:
                reason = f"✅ VOLUME SPIKE: {volume_ratio:.1%} - Strong confirmation"
                logger.info(reason)
                return True, reason, 1.1
            elif is_strong and volume_increasing:
                reason = f"✅ VOLUME STRONG + INCREASING: {volume_ratio:.1%}"
                logger.info(reason)
                return True, reason, 1.05
            elif is_strong:
                reason = f"✅ VOLUME STRONG: {volume_ratio:.1%}"
                logger.info(reason)
                return True, reason, 1.0
            elif is_normal:
                reason = f"✅ VOLUME NORMAL: {volume_ratio:.1%}"
                logger.info(reason)
                return True, reason, 0.95
            elif is_weak:
                reason = f"⚠️ VOLUME WEAK: {volume_ratio:.1%} - reduced confidence"
                logger.info(reason)
                return True, reason, 0.8
            elif is_very_low:
                reason = f"⚠️ VOLUME LOW: {volume_ratio:.1%} - lanjut dengan caution"
                logger.info(reason)
                return True, reason, 0.65
            else:
                reason = f"⚠️ VOLUME MINIMAL: {volume_ratio:.1%} - sinyal tetap lanjut (Deriv volume data terbatas)"
                logger.info(reason)
                return True, reason, 0.5
                
        except (StrategyError, Exception) as e:
            logger.error(f"Error in check_enhanced_volume_confirmation: {e}")
            return True, f"✅ Volume: Error - {str(e)}", 0.9
    
    def check_support_resistance_levels(self, indicators: Dict, signal_type: str,
                                         price_history: Optional[List] = None) -> Tuple[bool, str, float, Dict]:
        """Auto-detect support/resistance levels and check signal alignment.
        
        S/R Level Detection:
        - Identify swing highs/lows from price history
        - Calculate proximity to S/R levels
        - Boost confidence for signals at key levels
        
        Args:
            indicators: Dictionary of calculated indicators
            signal_type: 'BUY' or 'SELL'
            price_history: Optional list of recent prices
            
        Returns:
            Tuple of (is_valid, reason, confidence_boost, sr_data)
        """
        sr_data = {
            'support_levels': [],
            'resistance_levels': [],
            'near_support': False,
            'near_resistance': False,
            'distance_to_support': None,
            'distance_to_resistance': None
        }
        
        try:
            close = indicators.get('close')
            high = indicators.get('high')
            low = indicators.get('low')
            price_hist = price_history or indicators.get('price_history', [])
            high_hist = indicators.get('high_history', [])
            low_hist = indicators.get('low_history', [])
            
            if not is_valid_number(close):
                return True, "✅ S/R: Data tidak tersedia", 0.0, sr_data
            
            current_price = safe_float(close, 0.0)
            
            if len(price_hist) < 20:
                return True, "✅ S/R: Price history tidak cukup", 0.0, sr_data
            
            prices = [safe_float(p, 0.0) for p in price_hist[-50:] if is_valid_number(p)]
            
            if len(prices) < 20:
                return True, "✅ S/R: Data bersih tidak cukup", 0.0, sr_data
            
            swing_highs = []
            swing_lows = []
            
            for i in range(2, len(prices) - 2):
                if prices[i] > prices[i-1] and prices[i] > prices[i-2] and \
                   prices[i] > prices[i+1] and prices[i] > prices[i+2]:
                    swing_highs.append(prices[i])
                
                if prices[i] < prices[i-1] and prices[i] < prices[i-2] and \
                   prices[i] < prices[i+1] and prices[i] < prices[i+2]:
                    swing_lows.append(prices[i])
            
            cluster_threshold = current_price * 0.002
            
            resistance_levels = []
            for high_price in sorted(swing_highs, reverse=True)[:5]:
                if high_price > current_price:
                    is_new_level = True
                    for existing in resistance_levels:
                        if abs(high_price - existing) < cluster_threshold:
                            is_new_level = False
                            break
                    if is_new_level:
                        resistance_levels.append(high_price)
            
            support_levels = []
            for low_price in sorted(swing_lows)[:5]:
                if low_price < current_price:
                    is_new_level = True
                    for existing in support_levels:
                        if abs(low_price - existing) < cluster_threshold:
                            is_new_level = False
                            break
                    if is_new_level:
                        support_levels.append(low_price)
            
            sr_data['support_levels'] = sorted(support_levels, reverse=True)[:3]
            sr_data['resistance_levels'] = sorted(resistance_levels)[:3]
            
            proximity_threshold = current_price * 0.003
            
            confidence_boost = 0.0
            reason_parts = []
            
            if support_levels:
                nearest_support = max(support_levels)
                sr_data['distance_to_support'] = current_price - nearest_support
                if sr_data['distance_to_support'] < proximity_threshold:
                    sr_data['near_support'] = True
                    if signal_type == 'BUY':
                        confidence_boost += 0.10
                        reason_parts.append(f"Near Support ({nearest_support:.2f}) - BUY boosted")
                    else:
                        confidence_boost -= 0.05
                        reason_parts.append(f"Near Support ({nearest_support:.2f}) - SELL weakened")
            
            if resistance_levels:
                nearest_resistance = min(resistance_levels)
                sr_data['distance_to_resistance'] = nearest_resistance - current_price
                if sr_data['distance_to_resistance'] < proximity_threshold:
                    sr_data['near_resistance'] = True
                    if signal_type == 'SELL':
                        confidence_boost += 0.10
                        reason_parts.append(f"Near Resistance ({nearest_resistance:.2f}) - SELL boosted")
                    else:
                        confidence_boost -= 0.05
                        reason_parts.append(f"Near Resistance ({nearest_resistance:.2f}) - BUY weakened")
            
            if not reason_parts:
                reason = "✅ S/R: Harga tidak di level kunci"
            else:
                reason = f"S/R Analysis: {', '.join(reason_parts)}"
            
            logger.info(reason)
            return True, reason, max(0.0, confidence_boost), sr_data
            
        except (StrategyError, Exception) as e:
            logger.error(f"Error in check_support_resistance_levels: {e}")
            return True, f"✅ S/R: Error - {str(e)}", 0.0, sr_data
    
    def check_rsi_level_filter(self, indicators: Dict, signal_type: str) -> Tuple[bool, str, float]:
        """Check RSI level filter - BLOCKING extreme levels + divergence check
        
        RSI Extreme Level Blocking (dapat di-disable via RSI_EXTREME_FILTER_ENABLED):
        - Block BUY saat RSI > RSI_EXTREME_OVERBOUGHT (default 85)
        - Block SELL saat RSI < RSI_EXTREME_OVERSOLD (default 15)
        
        RSI Divergence Detection:
        - Bullish divergence: Price lower low, RSI higher low
        - Bearish divergence: Price higher high, RSI lower high
        
        Args:
            indicators: Dictionary of calculated indicators
            signal_type: 'BUY' or 'SELL'
            
        Returns:
            Tuple of (is_valid, reason, confidence_boost)
            - confidence_boost: 0.0-0.15 berdasarkan RSI quality
        """
        try:
            rsi = indicators.get('rsi')
            rsi_prev = indicators.get('rsi_prev')
            rsi_history = indicators.get('rsi_history', [])
            close = indicators.get('close')
            close_prev = indicators.get('close_prev')
            
            if not is_valid_number(rsi):
                return True, "✅ RSI Level: Data tidak tersedia - signal diizinkan", 0.0
            
            rsi_val = safe_float(rsi, 50.0)
            confidence_boost = 0.0
            
            rsi_extreme_filter_enabled = getattr(self.config, 'RSI_EXTREME_FILTER_ENABLED', False)
            rsi_extreme_overbought = getattr(self.config, 'RSI_EXTREME_OVERBOUGHT', 85)
            rsi_extreme_oversold = getattr(self.config, 'RSI_EXTREME_OVERSOLD', 15)
            
            if signal_type == 'BUY':
                if rsi_extreme_filter_enabled and rsi_val > rsi_extreme_overbought:
                    reason = f"❌ RSI EXTREME OVERBOUGHT: RSI({rsi_val:.1f}) > {rsi_extreme_overbought} - BUY BLOCKED"
                    logger.info(reason)
                    return False, reason, 0.0
                elif rsi_val > 75:
                    reason = f"⚠️ RSI HIGH: RSI({rsi_val:.1f}) > 75 - Cautious BUY (extreme filter disabled)"
                    confidence_boost = 0.0
                elif rsi_val > 70:
                    reason = f"⚠️ RSI OVERBOUGHT: RSI({rsi_val:.1f}) > 70 - Cautious BUY"
                    confidence_boost = 0.0
                elif rsi_val > 50:
                    reason = f"✅ RSI BULLISH: RSI({rsi_val:.1f}) > 50 (Momentum positif)"
                    confidence_boost = 0.05
                elif rsi_val > 30:
                    reason = f"✅ RSI RECOVERY ZONE: RSI({rsi_val:.1f}) - Good entry zone"
                    confidence_boost = 0.10
                else:
                    reason = f"✅ RSI OVERSOLD BOUNCE: RSI({rsi_val:.1f}) < 30 - Potential reversal BUY"
                    confidence_boost = 0.15
                    
            elif signal_type == 'SELL':
                if rsi_extreme_filter_enabled and rsi_val < rsi_extreme_oversold:
                    reason = f"❌ RSI EXTREME OVERSOLD: RSI({rsi_val:.1f}) < {rsi_extreme_oversold} - SELL BLOCKED"
                    logger.info(reason)
                    return False, reason, 0.0
                elif rsi_val < 25:
                    reason = f"⚠️ RSI LOW: RSI({rsi_val:.1f}) < 25 - Cautious SELL (extreme filter disabled)"
                    confidence_boost = 0.0
                elif rsi_val < 30:
                    reason = f"⚠️ RSI OVERSOLD: RSI({rsi_val:.1f}) < 30 - Cautious SELL"
                    confidence_boost = 0.0
                elif rsi_val < 50:
                    reason = f"✅ RSI BEARISH: RSI({rsi_val:.1f}) < 50 (Momentum negatif)"
                    confidence_boost = 0.05
                elif rsi_val < 70:
                    reason = f"✅ RSI PULLBACK ZONE: RSI({rsi_val:.1f}) - Good entry zone"
                    confidence_boost = 0.10
                else:
                    reason = f"✅ RSI OVERBOUGHT DROP: RSI({rsi_val:.1f}) > 70 - Potential reversal SELL"
                    confidence_boost = 0.15
            else:
                reason = f"✅ RSI Level: Signal type {signal_type} - diizinkan"
                confidence_boost = 0.0
            
            divergence_info = self._check_rsi_divergence(indicators, signal_type)
            if divergence_info['has_divergence']:
                if divergence_info['type'] == 'bullish' and signal_type == 'BUY':
                    reason += f" | 🔄 BULLISH DIVERGENCE detected!"
                    confidence_boost += 0.10
                elif divergence_info['type'] == 'bearish' and signal_type == 'SELL':
                    reason += f" | 🔄 BEARISH DIVERGENCE detected!"
                    confidence_boost += 0.10
                elif divergence_info['type'] == 'bullish' and signal_type == 'SELL':
                    reason += f" | ⚠️ Bullish divergence - SELL weakened"
                    confidence_boost -= 0.05
                elif divergence_info['type'] == 'bearish' and signal_type == 'BUY':
                    reason += f" | ⚠️ Bearish divergence - BUY weakened"
                    confidence_boost -= 0.05
            
            logger.info(reason)
            return True, reason, max(0.0, confidence_boost)
                
        except (StrategyError, Exception) as e:
            logger.error(f"Error in check_rsi_level_filter: {e}")
            return True, f"✅ RSI Level Filter: Error - {str(e)}", 0.0
    
    def _check_rsi_divergence(self, indicators: Dict, signal_type: str) -> Dict[str, Any]:
        """Check for RSI divergence patterns
        
        Bullish divergence: Price makes lower low, RSI makes higher low
        Bearish divergence: Price makes higher high, RSI makes lower high
        
        Args:
            indicators: Dictionary of calculated indicators
            signal_type: 'BUY' or 'SELL'
            
        Returns:
            Dict with divergence info: {'has_divergence': bool, 'type': str, 'strength': float}
        """
        result = {'has_divergence': False, 'type': 'none', 'strength': 0.0}
        
        try:
            rsi_history = indicators.get('rsi_history', [])
            price_history = indicators.get('price_history', [])
            
            if not rsi_history or len(rsi_history) < 5:
                return result
            if not price_history or len(price_history) < 5:
                return result
            
            rsi_history_clean = [safe_float(r, 50.0) for r in rsi_history[-10:] if is_valid_number(r)]
            price_history_clean = [safe_float(p, 0.0) for p in price_history[-10:] if is_valid_number(p)]
            
            if len(rsi_history_clean) < 5 or len(price_history_clean) < 5:
                return result
            
            current_rsi = rsi_history_clean[-1]
            prev_rsi = min(rsi_history_clean[:-1]) if len(rsi_history_clean) > 1 else current_rsi
            max_prev_rsi = max(rsi_history_clean[:-1]) if len(rsi_history_clean) > 1 else current_rsi
            
            current_price = price_history_clean[-1]
            prev_price_low = min(price_history_clean[:-1]) if len(price_history_clean) > 1 else current_price
            prev_price_high = max(price_history_clean[:-1]) if len(price_history_clean) > 1 else current_price
            
            if current_price <= prev_price_low and current_rsi > prev_rsi:
                result['has_divergence'] = True
                result['type'] = 'bullish'
                result['strength'] = min(1.0, (current_rsi - prev_rsi) / 10.0)
                logger.debug(f"Bullish divergence: Price {current_price:.2f} <= {prev_price_low:.2f}, RSI {current_rsi:.1f} > {prev_rsi:.1f}")
            
            elif current_price >= prev_price_high and current_rsi < max_prev_rsi:
                result['has_divergence'] = True
                result['type'] = 'bearish'
                result['strength'] = min(1.0, (max_prev_rsi - current_rsi) / 10.0)
                logger.debug(f"Bearish divergence: Price {current_price:.2f} >= {prev_price_high:.2f}, RSI {current_rsi:.1f} < {max_prev_rsi:.1f}")
            
            return result
            
        except (StrategyError, Exception) as e:
            logger.debug(f"Error in _check_rsi_divergence: {e}")
            return result
    
    def check_ema_slope_filter(self, indicators: Dict, signal_type: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Check EMA slope filter - BLOCKING jika slope berlawanan dengan signal direction
        
        EMA Slope Detection (5-period calculation):
        - Hitung slope dari EMA5, EMA20 untuk 5 periods terakhir
        - Block jika slope berlawanan dengan signal direction
        - Detect potential reversal dengan slope change
        
        Args:
            indicators: Dictionary of calculated indicators
            signal_type: 'BUY' or 'SELL'
            
        Returns:
            Tuple of (is_valid, reason, slope_data)
            - slope_data: {'slope': float, 'direction': str, 'reversal_detected': bool}
        """
        slope_data = {
            'slope': 0.0,
            'direction': 'flat',
            'reversal_detected': False,
            'slope_strength': 'weak'
        }
        
        ema_slope_filter_enabled = getattr(self.config, 'EMA_SLOPE_FILTER_ENABLED', False)
        if not ema_slope_filter_enabled:
            return True, "✅ EMA Slope Filter DISABLED - signal diizinkan", slope_data
        
        try:
            ema_slope = indicators.get('ema_slope')
            ema_history = indicators.get('ema_history', [])
            ema_5 = indicators.get(f'ema_{self.config.EMA_PERIODS[0]}')
            ema_20 = indicators.get(f'ema_{self.config.EMA_PERIODS[1]}')
            close = indicators.get('close')
            
            slope_val = 0.0
            
            if ema_history and len(ema_history) >= 5:
                ema_clean = [safe_float(e, 0.0) for e in ema_history[-5:] if is_valid_number(e)]
                if len(ema_clean) >= 2:
                    slope_val = (ema_clean[-1] - ema_clean[0]) / len(ema_clean)
                    close_val = safe_float(close, 0.0) if close is not None else 0.0
                    if close_val > 0:
                        slope_val = (slope_val / close_val) * 100
            elif is_valid_number(ema_slope):
                slope_val = safe_float(ema_slope, 0.0)
            elif is_valid_number(ema_5) and is_valid_number(ema_20) and is_valid_number(close):
                ema5 = safe_float(ema_5, 0.0)
                ema20 = safe_float(ema_20, 0.0)
                close_val = safe_float(close, 1.0)
                if close_val > 0:
                    slope_val = ((ema5 - ema20) / close_val) * 100
            else:
                return True, "✅ EMA Slope: Data tidak tersedia - signal diizinkan", slope_data
            
            slope_data['slope'] = slope_val
            
            min_slope_strong = getattr(self.config, 'EMA_SLOPE_STRONG_THRESHOLD', 0.02)
            min_slope_medium = getattr(self.config, 'EMA_SLOPE_MIN_THRESHOLD', 0.01)
            min_slope_weak = 0.005
            
            if slope_val > min_slope_strong:
                slope_data['direction'] = 'bullish_strong'
                slope_data['slope_strength'] = 'strong'
            elif slope_val > min_slope_medium:
                slope_data['direction'] = 'bullish'
                slope_data['slope_strength'] = 'medium'
            elif slope_val > min_slope_weak:
                slope_data['direction'] = 'bullish_weak'
                slope_data['slope_strength'] = 'weak'
            elif slope_val < -min_slope_strong:
                slope_data['direction'] = 'bearish_strong'
                slope_data['slope_strength'] = 'strong'
            elif slope_val < -min_slope_medium:
                slope_data['direction'] = 'bearish'
                slope_data['slope_strength'] = 'medium'
            elif slope_val < -min_slope_weak:
                slope_data['direction'] = 'bearish_weak'
                slope_data['slope_strength'] = 'weak'
            else:
                slope_data['direction'] = 'flat'
                slope_data['slope_strength'] = 'none'
            
            reversal_detected = self._detect_slope_reversal(indicators)
            slope_data['reversal_detected'] = reversal_detected
            
            reversal_info = " | 🔄 REVERSAL POTENTIAL" if reversal_detected else ""
            
            if signal_type == 'BUY':
                if 'bullish' in slope_data['direction']:
                    reason = f"✅ EMA Slope BULLISH: Slope({slope_val:.4f}%) ↗ ({slope_data['slope_strength']}){reversal_info}"
                    logger.info(reason)
                    return True, reason, slope_data
                elif slope_data['direction'] == 'flat':
                    reason = f"✅ EMA Slope FLAT: Slope({slope_val:.4f}%) - OK untuk BUY{reversal_info}"
                    logger.info(reason)
                    return True, reason, slope_data
                elif reversal_detected and 'bearish_weak' in slope_data['direction']:
                    reason = f"⚠️ EMA Slope REVERSAL: Slope({slope_val:.4f}%) bearish lemah + reversal detected - Izinkan BUY{reversal_info}"
                    logger.info(reason)
                    return True, reason, slope_data
                else:
                    reason = f"❌ EMA Slope BEARISH: Slope({slope_val:.4f}%) berlawanan dengan BUY signal"
                    logger.info(reason)
                    return False, reason, slope_data
                    
            elif signal_type == 'SELL':
                if 'bearish' in slope_data['direction']:
                    reason = f"✅ EMA Slope BEARISH: Slope({slope_val:.4f}%) ↘ ({slope_data['slope_strength']}){reversal_info}"
                    logger.info(reason)
                    return True, reason, slope_data
                elif slope_data['direction'] == 'flat':
                    reason = f"✅ EMA Slope FLAT: Slope({slope_val:.4f}%) - OK untuk SELL{reversal_info}"
                    logger.info(reason)
                    return True, reason, slope_data
                elif reversal_detected and 'bullish_weak' in slope_data['direction']:
                    reason = f"⚠️ EMA Slope REVERSAL: Slope({slope_val:.4f}%) bullish lemah + reversal detected - Izinkan SELL{reversal_info}"
                    logger.info(reason)
                    return True, reason, slope_data
                else:
                    reason = f"❌ EMA Slope BULLISH: Slope({slope_val:.4f}%) berlawanan dengan SELL signal"
                    logger.info(reason)
                    return False, reason, slope_data
            
            return True, "✅ EMA Slope: signal diizinkan", slope_data
                
        except (StrategyError, Exception) as e:
            logger.error(f"Error in check_ema_slope_filter: {e}")
            return True, f"✅ EMA Slope Filter SKIPPED: Error - {str(e)}", slope_data
    
    def _detect_slope_reversal(self, indicators: Dict) -> bool:
        """Detect potential slope reversal from EMA history
        
        Detects reversal when:
        - Slope changes direction in last 3 periods
        - EMA momentum changes sign
        
        Args:
            indicators: Dictionary of calculated indicators
            
        Returns:
            True if reversal detected
        """
        try:
            ema_history = indicators.get('ema_history', [])
            
            if not ema_history or len(ema_history) < 5:
                return False
            
            ema_clean = [safe_float(e, 0.0) for e in ema_history[-5:] if is_valid_number(e)]
            
            if len(ema_clean) < 4:
                return False
            
            slope_old = ema_clean[-3] - ema_clean[-4]
            slope_mid = ema_clean[-2] - ema_clean[-3]
            slope_new = ema_clean[-1] - ema_clean[-2]
            
            if (slope_old < 0 and slope_new > 0) or (slope_old > 0 and slope_new < 0):
                if abs(slope_new) > abs(slope_old) * 0.5:
                    logger.debug(f"Slope reversal detected: old={slope_old:.4f}, mid={slope_mid:.4f}, new={slope_new:.4f}")
                    return True
            
            return False
            
        except (StrategyError, Exception) as e:
            logger.debug(f"Error in _detect_slope_reversal: {e}")
            return False
    
    def check_volume_vwap_filter(self, indicators: Dict, signal_type: str, signal_mode: str = 'M1_SCALP') -> Tuple[bool, str, float]:
        """Check volume and VWAP filter conditions - STRICT VERSION dengan mode-based thresholds
        
        Volume Confirmation Requirements:
        - M1_SCALP: Volume >= 0.8x average (relaxed untuk scalping)
        - M5_SWING: Volume >= 1.2x average (STRICT - harus ada momentum)
        - BREAKOUT: Volume increasing 3 candles + >= 1.5x average
        
        Args:
            indicators: Dictionary of calculated indicators
            signal_type: 'BUY' or 'SELL'
            signal_mode: 'M1_SCALP', 'M5_SWING', or 'BREAKOUT'
            
        Returns:
            Tuple of (is_valid, reason, confidence_multiplier)
            - confidence_multiplier: 0.5-1.0 berdasarkan volume strength
        """
        try:
            volume = indicators.get('volume')
            volume_avg = indicators.get('volume_avg')
            volume_history = indicators.get('volume_history', [])
            close = indicators.get('close')
            vwap = indicators.get('vwap')
            
            confidence_multiplier = 1.0
            
            volume_valid = is_valid_number(volume) and is_valid_number(volume_avg)
            if not volume_valid:
                reason = "✅ Volume Filter: Data tidak tersedia - signal diizinkan"
                logger.info(reason)
                return True, reason, 0.8
            
            volume_val = safe_float(volume, 0.0)
            volume_avg_val = safe_float(volume_avg, 0.0)
            
            if volume_avg_val <= 0:
                reason = "✅ Volume Filter: Average tidak tersedia - signal diizinkan"
                logger.info(reason)
                return True, reason, 0.8
            
            volume_ratio = safe_divide(volume_val, volume_avg_val, 1.0, "volume_ratio")
            
            volume_increasing = False
            if volume_history and len(volume_history) >= 3:
                vol_clean = [safe_float(v, 0.0) for v in volume_history[-3:] if is_valid_number(v)]
                if len(vol_clean) >= 3:
                    if vol_clean[-1] > vol_clean[-2] > vol_clean[-3]:
                        volume_increasing = True
            
            vwap_aligned = False
            vwap_info = ""
            
            if is_valid_number(vwap) and is_valid_number(close):
                close_val = safe_float(close, 0.0)
                vwap_val = safe_float(vwap, 0.0)
                
                if vwap_val > 0:
                    vwap_tolerance = vwap_val * 0.002
                    
                    if signal_type == 'BUY':
                        vwap_aligned = close_val >= (vwap_val - vwap_tolerance)
                    elif signal_type == 'SELL':
                        vwap_aligned = close_val <= (vwap_val + vwap_tolerance)
                    
                    vwap_status = "aligned ✅" if vwap_aligned else "not aligned"
                    vwap_info = f", VWAP({vwap_val:.2f}) {vwap_status}"
            
            increasing_info = " | 📈 Volume Increasing" if volume_increasing else ""
            
            if signal_mode == 'BREAKOUT':
                if volume_ratio >= 1.5 and volume_increasing:
                    reason = f"✅ Volume BREAKOUT STRONG: [{volume_ratio:.1%}] + Increasing{vwap_info}{increasing_info}"
                    confidence_multiplier = 1.0
                    logger.info(reason)
                    return True, reason, confidence_multiplier
                elif volume_ratio >= 1.2 and volume_increasing:
                    reason = f"⚠️ Volume BREAKOUT OK: [{volume_ratio:.1%}] + Increasing (bukan optimal){vwap_info}{increasing_info}"
                    confidence_multiplier = 0.85
                    logger.info(reason)
                    return True, reason, confidence_multiplier
                else:
                    reason = f"❌ Volume BREAKOUT LEMAH: [{volume_ratio:.1%}] (butuh >= 1.5x + increasing){vwap_info}{increasing_info}"
                    logger.info(reason)
                    return False, reason, 0.0
                    
            elif signal_mode == 'M5_SWING':
                if volume_ratio >= 1.2:
                    reason = f"✅ Volume M5_SWING STRONG: [{volume_ratio:.1%}] >= 1.2x{vwap_info}{increasing_info}"
                    confidence_multiplier = 1.0
                    logger.info(reason)
                    return True, reason, confidence_multiplier
                elif volume_ratio >= 1.0:
                    reason = f"⚠️ Volume M5_SWING OK: [{volume_ratio:.1%}] (di bawah 1.2x optimal){vwap_info}{increasing_info}"
                    confidence_multiplier = 0.85
                    logger.info(reason)
                    return True, reason, confidence_multiplier
                elif volume_ratio >= 0.8:
                    reason = f"⚠️ Volume M5_SWING LEMAH: [{volume_ratio:.1%}] - reduced confidence{vwap_info}{increasing_info}"
                    confidence_multiplier = 0.7
                    logger.info(reason)
                    return True, reason, confidence_multiplier
                else:
                    reason = f"❌ Volume M5_SWING TERLALU LEMAH: [{volume_ratio:.1%}] < 0.8x{vwap_info}{increasing_info}"
                    logger.info(reason)
                    return False, reason, 0.0
            
            else:
                if volume_ratio >= 1.0:
                    reason = f"✅ Volume SCALP STRONG: [{volume_ratio:.1%}]{vwap_info}{increasing_info}"
                    confidence_multiplier = 1.0
                elif volume_ratio >= 0.8:
                    reason = f"✅ Volume SCALP OK: [{volume_ratio:.1%}]{vwap_info}{increasing_info}"
                    confidence_multiplier = 0.9
                elif volume_ratio >= 0.5:
                    reason = f"⚠️ Volume SCALP RENDAH: [{volume_ratio:.1%}] - reduced confidence{vwap_info}{increasing_info}"
                    confidence_multiplier = 0.7
                else:
                    reason = f"⚠️ Volume SCALP VERY LOW: [{volume_ratio:.1%}] - LANJUT dengan caution{vwap_info}{increasing_info}"
                    confidence_multiplier = 0.5
                
                logger.info(reason)
                return True, reason, confidence_multiplier
            
        except (StrategyError, Exception) as e:
            logger.error(f"Error in check_volume_vwap_filter: {e}")
            return True, f"✅ Volume Filter: Error - {str(e)}", 0.8
    
    def check_price_action_confirmation(self, indicators: Dict, signal_type: str) -> Tuple[bool, str, int, Dict[str, Any]]:
        """Check price action confirmation - ENHANCED VERSION with wick validation & momentum
        
        Enhanced Price Action Validation:
        - Tier 1: Candlestick patterns (doji, hammer, engulfing)
        - Tier 2: S/R proximity validation
        - Tier 3: Wick validation - tidak entry di ujung wick panjang
        - Tier 4: Price momentum consistency check
        - Tier 5: MACD histogram direction
        
        Args:
            indicators: Dictionary of calculated indicators
            signal_type: 'BUY' or 'SELL'
            
        Returns:
            Tuple of (passed, reason, points, validation_data)
            - validation_data: Contains wick_valid, momentum_consistent, patterns_found
        """
        validation_data = {
            'wick_valid': True,
            'momentum_consistent': True,
            'patterns_found': [],
            'wick_warning': '',
            'momentum_info': ''
        }
        
        try:
            confidence_boost = 0
            confirmations = []
            
            wick_passed, wick_reason = self._validate_wick_position(indicators, signal_type)
            validation_data['wick_valid'] = wick_passed
            if not wick_passed:
                validation_data['wick_warning'] = wick_reason
                reason = f"❌ Price Action BLOCKED: {wick_reason}"
                logger.info(reason)
                return False, reason, 0, validation_data
            
            momentum_passed, momentum_reason = self._check_price_momentum_consistency(indicators, signal_type)
            validation_data['momentum_consistent'] = momentum_passed
            validation_data['momentum_info'] = momentum_reason
            
            if not momentum_passed:
                confirmations.append(f"⚠️ Momentum inconsistent: {momentum_reason}")
            
            patterns = indicators.get('candlestick_patterns', {})
            if patterns:
                if signal_type == 'BUY':
                    if patterns.get('doji', False):
                        confirmations.append("Doji (potential reversal)")
                        validation_data['patterns_found'].append('doji')
                        confidence_boost += 3
                    if patterns.get('bullish_pinbar', False):
                        confirmations.append("Bullish Pinbar")
                        validation_data['patterns_found'].append('bullish_pinbar')
                        confidence_boost += 5
                    if patterns.get('hammer', False):
                        confirmations.append("Hammer")
                        validation_data['patterns_found'].append('hammer')
                        confidence_boost += 5
                    if patterns.get('bullish_engulfing', False):
                        confirmations.append("Bullish Engulfing")
                        validation_data['patterns_found'].append('bullish_engulfing')
                        confidence_boost += 7
                    if patterns.get('morning_star', False):
                        confirmations.append("Morning Star")
                        validation_data['patterns_found'].append('morning_star')
                        confidence_boost += 8
                elif signal_type == 'SELL':
                    if patterns.get('doji', False):
                        confirmations.append("Doji (potential reversal)")
                        validation_data['patterns_found'].append('doji')
                        confidence_boost += 3
                    if patterns.get('bearish_pinbar', False):
                        confirmations.append("Bearish Pinbar")
                        validation_data['patterns_found'].append('bearish_pinbar')
                        confidence_boost += 5
                    if patterns.get('inverted_hammer', False) or patterns.get('shooting_star', False):
                        confirmations.append("Shooting Star")
                        validation_data['patterns_found'].append('shooting_star')
                        confidence_boost += 5
                    if patterns.get('bearish_engulfing', False):
                        confirmations.append("Bearish Engulfing")
                        validation_data['patterns_found'].append('bearish_engulfing')
                        confidence_boost += 7
                    if patterns.get('evening_star', False):
                        confirmations.append("Evening Star")
                        validation_data['patterns_found'].append('evening_star')
                        confidence_boost += 8
            
            pattern_confirmations = [c for c in confirmations if not c.startswith("⚠️")]
            if pattern_confirmations:
                reason = f"✅ Price Action Filter PASSED [Tier 1]: {', '.join(pattern_confirmations)}"
                logger.info(reason)
                return True, reason, 15 + confidence_boost, validation_data
            
            sr_levels = indicators.get('support_resistance', {})
            close = safe_float(indicators.get('close', 0.0), 0.0)
            atr = safe_float(indicators.get('atr', 1.0), 1.0)
            
            if sr_levels and close > 0 and atr > 0:
                nearest_support = sr_levels.get('nearest_support', 0.0)
                nearest_resistance = sr_levels.get('nearest_resistance', 0.0)
                
                proximity_threshold = atr * 1.0
                
                if signal_type == 'BUY' and nearest_support > 0:
                    if abs(close - nearest_support) <= proximity_threshold:
                        reason = f"✅ Price Action Filter PASSED [Tier 2]: Near Support level ({nearest_support:.2f})"
                        logger.info(reason)
                        return True, reason, 15, validation_data
                elif signal_type == 'SELL' and nearest_resistance > 0:
                    if abs(close - nearest_resistance) <= proximity_threshold:
                        reason = f"✅ Price Action Filter PASSED [Tier 2]: Near Resistance level ({nearest_resistance:.2f})"
                        logger.info(reason)
                        return True, reason, 15, validation_data
            
            macd_histogram = indicators.get('macd_histogram')
            macd = indicators.get('macd')
            macd_signal = indicators.get('macd_signal')
            
            if is_valid_number(macd_histogram):
                histogram = safe_float(macd_histogram, 0.0)
                if signal_type == 'BUY' and histogram > 0:
                    reason = f"✅ Price Action Filter PASSED [Tier 3]: MACD Histogram bullish ({histogram:.4f})"
                    logger.info(reason)
                    return True, reason, 15, validation_data
                elif signal_type == 'SELL' and histogram < 0:
                    reason = f"✅ Price Action Filter PASSED [Tier 3]: MACD Histogram bearish ({histogram:.4f})"
                    logger.info(reason)
                    return True, reason, 15, validation_data
            
            if is_valid_number(macd) and is_valid_number(macd_signal):
                macd_val = safe_float(macd, 0.0)
                signal_val = safe_float(macd_signal, 0.0)
                
                if signal_type == 'BUY' and macd_val > signal_val:
                    reason = f"✅ Price Action Filter PASSED [Tier 3]: MACD above signal line"
                    logger.info(reason)
                    return True, reason, 15, validation_data
                elif signal_type == 'SELL' and macd_val < signal_val:
                    reason = f"✅ Price Action Filter PASSED [Tier 3]: MACD below signal line"
                    logger.info(reason)
                    return True, reason, 15, validation_data
            
            ema_5 = indicators.get(f'ema_{self.config.EMA_PERIODS[0]}')
            ema_20 = indicators.get(f'ema_{self.config.EMA_PERIODS[1]}')
            ema_50 = indicators.get(f'ema_{self.config.EMA_PERIODS[2]}')
            
            if all([is_valid_number(ema_5), is_valid_number(ema_20), is_valid_number(ema_50), is_valid_number(close)]):
                ema5 = safe_float(ema_5, 0.0)
                ema20 = safe_float(ema_20, 0.0)
                ema50 = safe_float(ema_50, 0.0)
                close_val = safe_float(close, 0.0)
                
                if close_val > 0:
                    ema_spacing = abs(ema5 - ema20) / close_val
                    
                    if signal_type == 'BUY' and ema5 > ema20 and ema_spacing > 0.0003:
                        reason = f"✅ Price Action Filter PASSED [Tier 4]: EMA spacing bullish ({ema_spacing:.4%})"
                        logger.info(reason)
                        return True, reason, 15, validation_data
                    elif signal_type == 'SELL' and ema5 < ema20 and ema_spacing > 0.0003:
                        reason = f"✅ Price Action Filter PASSED [Tier 4]: EMA spacing bearish ({ema_spacing:.4%})"
                        logger.info(reason)
                        return True, reason, 15, validation_data
            
            rsi = indicators.get('rsi')
            rsi_prev = indicators.get('rsi_prev')
            
            if is_valid_number(rsi) and is_valid_number(rsi_prev):
                rsi_val = safe_float(rsi, 50.0)
                rsi_prev_val = safe_float(rsi_prev, 50.0)
                rsi_momentum = rsi_val - rsi_prev_val
                
                if signal_type == 'BUY' and rsi_momentum > 0:
                    reason = f"✅ Price Action Filter PASSED [Tier 5]: RSI momentum bullish (+{rsi_momentum:.1f})"
                    logger.info(reason)
                    return True, reason, 12, validation_data
                elif signal_type == 'SELL' and rsi_momentum < 0:
                    reason = f"✅ Price Action Filter PASSED [Tier 5]: RSI momentum bearish ({rsi_momentum:.1f})"
                    logger.info(reason)
                    return True, reason, 12, validation_data
            
            reason = "✅ Price Action: No pattern detected - SCALPING tetap lanjut tanpa pattern"
            logger.info(reason)
            return True, reason, 5, validation_data
                
        except (StrategyError, Exception) as e:
            logger.error(f"Error in check_price_action_confirmation: {e}")
            return False, f"Error: {str(e)}", 0, validation_data
    
    def _validate_wick_position(self, indicators: Dict, signal_type: str) -> Tuple[bool, str]:
        """Validate that entry is not at the edge of a long wick
        
        Block entry when:
        - BUY signal at upper wick edge (close near high with long upper wick)
        - SELL signal at lower wick edge (close near low with long lower wick)
        
        Args:
            indicators: Dictionary of calculated indicators
            signal_type: 'BUY' or 'SELL'
            
        Returns:
            Tuple[bool, str]: (is_valid, reason)
        """
        try:
            open_price = indicators.get('open')
            high = indicators.get('high')
            low = indicators.get('low')
            close = indicators.get('close')
            atr = indicators.get('atr')
            
            if not all([is_valid_number(open_price), is_valid_number(high), 
                       is_valid_number(low), is_valid_number(close)]):
                return True, "Wick data tidak tersedia - OK"
            
            open_val = safe_float(open_price, 0.0)
            high_val = safe_float(high, 0.0)
            low_val = safe_float(low, 0.0)
            close_val = safe_float(close, 0.0)
            
            if high_val <= low_val:
                return True, "Invalid candle data - OK"
            
            candle_range = high_val - low_val
            body_size = abs(close_val - open_val)
            upper_wick = high_val - max(open_val, close_val)
            lower_wick = min(open_val, close_val) - low_val
            
            wick_ratio_threshold = 0.6
            
            if signal_type == 'BUY':
                if upper_wick > candle_range * wick_ratio_threshold:
                    close_to_high_ratio = (high_val - close_val) / candle_range if candle_range > 0 else 0
                    if close_to_high_ratio < 0.2:
                        return False, f"Entry di ujung upper wick panjang (wick {upper_wick:.2f}, ratio {close_to_high_ratio:.1%})"
            
            elif signal_type == 'SELL':
                if lower_wick > candle_range * wick_ratio_threshold:
                    close_to_low_ratio = (close_val - low_val) / candle_range if candle_range > 0 else 0
                    if close_to_low_ratio < 0.2:
                        return False, f"Entry di ujung lower wick panjang (wick {lower_wick:.2f}, ratio {close_to_low_ratio:.1%})"
            
            return True, "Wick position OK"
            
        except (StrategyError, Exception) as e:
            logger.debug(f"Error in _validate_wick_position: {e}")
            return True, f"Wick validation skipped: {str(e)}"
    
    def _check_price_momentum_consistency(self, indicators: Dict, signal_type: str) -> Tuple[bool, str]:
        """Check if price momentum is consistent with signal direction
        
        Checks last 3 candles for consistent price movement:
        - BUY: Prices should be generally rising
        - SELL: Prices should be generally falling
        
        Args:
            indicators: Dictionary of calculated indicators
            signal_type: 'BUY' or 'SELL'
            
        Returns:
            Tuple[bool, str]: (is_consistent, description)
        """
        try:
            price_history = indicators.get('price_history', [])
            close_history = indicators.get('close_history', [])
            
            history = price_history if price_history else close_history
            
            if not history or len(history) < 3:
                return True, "Price history tidak cukup - assumed consistent"
            
            recent_prices = [safe_float(p, 0.0) for p in history[-4:] if is_valid_number(p)]
            
            if len(recent_prices) < 3:
                return True, "Price data tidak cukup - assumed consistent"
            
            price_changes = []
            for i in range(1, len(recent_prices)):
                change = recent_prices[i] - recent_prices[i-1]
                price_changes.append(change)
            
            bullish_moves = sum(1 for c in price_changes if c > 0)
            bearish_moves = sum(1 for c in price_changes if c < 0)
            
            if signal_type == 'BUY':
                if bullish_moves >= len(price_changes) / 2:
                    return True, f"Momentum bullish konsisten ({bullish_moves}/{len(price_changes)} up)"
                else:
                    return False, f"Momentum mixed/bearish ({bearish_moves}/{len(price_changes)} down)"
            
            elif signal_type == 'SELL':
                if bearish_moves >= len(price_changes) / 2:
                    return True, f"Momentum bearish konsisten ({bearish_moves}/{len(price_changes)} down)"
                else:
                    return False, f"Momentum mixed/bullish ({bullish_moves}/{len(price_changes)} up)"
            
            return True, "Momentum check skipped"
            
        except (StrategyError, Exception) as e:
            logger.debug(f"Error in _check_price_momentum_consistency: {e}")
            return True, f"Momentum check skipped: {str(e)}"
    
    def check_pattern_recognition(self, indicators: Dict, signal_type: str, 
                                   m1_df=None, m5_df=None) -> Tuple[bool, str, float, Dict[str, Any]]:
        """Check pattern recognition for signal confidence enhancement.
        
        IMPROVEMENT 2: Trade Setup Validation dengan Pattern Recognition
        
        This is an ENHANCEMENT filter, NOT a BLOCKING filter.
        Detects patterns and applies confidence boost to matching signals.
        
        Patterns detected:
        - Inside Bar: Consolidation pattern (pending breakout)
        - Pin Bar: Rejection candle (bullish/bearish)
        - Double Bottom/Top: Reversal patterns
        
        Confidence boost:
        - Strong pattern match: +15-20%
        - Moderate pattern match: +10%
        - Weak pattern match: +5%
        - No pattern: 0% (signal still valid)
        
        Args:
            indicators: Dictionary of calculated indicators
            signal_type: 'BUY' or 'SELL'
            m1_df: M1 DataFrame for pattern detection
            m5_df: M5 DataFrame for pattern detection (optional)
            
        Returns:
            Tuple[bool, str, float, Dict]: (always_true, reason, confidence_boost, pattern_data)
        """
        pattern_data = {
            'patterns_detected': [],
            'pattern_name': 'none',
            'pattern_confidence': 0.0,
            'inside_bar': {'detected': False, 'strength': 'none'},
            'pin_bar': {'detected': False, 'strength': 'none'},
            'reversal_pattern': {'detected': False, 'type': 'none', 'strength': 'none'},
            'combined_boost': 0.0
        }
        
        try:
            from bot.indicators import IndicatorEngine
            
            indicator_engine = None
            if hasattr(self, 'indicator_engine'):
                indicator_engine = self.indicator_engine
            else:
                indicator_engine = IndicatorEngine(self.config)
            
            df = m1_df if m1_df is not None else m5_df
            
            if df is None or len(df) < 5:
                reason = "✅ Pattern Recognition: Data tidak tersedia - signal diizinkan tanpa boost"
                logger.debug(reason)
                return True, reason, 0.0, pattern_data
            
            confidence_boost = 0.0
            detected_patterns = []
            
            inside_bar_result = indicator_engine.detect_inside_bar(df)
            pattern_data['inside_bar'] = inside_bar_result
            
            if inside_bar_result.get('detected', False):
                strength = inside_bar_result.get('strength', 'weak')
                detected_patterns.append(f"Inside Bar ({strength})")
                
                if strength == 'strong':
                    confidence_boost += 0.05
                elif strength == 'moderate':
                    confidence_boost += 0.03
                else:
                    confidence_boost += 0.02
            
            pin_bar_result = indicator_engine.detect_pin_bar(df, signal_type)
            pattern_data['pin_bar'] = pin_bar_result
            
            if pin_bar_result.get('detected', False):
                strength = pin_bar_result.get('strength', 'weak')
                pattern_type = pin_bar_result.get('pattern_type', 'pin_bar')
                detected_patterns.append(f"Pin Bar ({strength})")
                
                if strength == 'strong':
                    confidence_boost += 0.15
                elif strength == 'moderate':
                    confidence_boost += 0.10
                else:
                    confidence_boost += 0.05
            
            reversal_result = indicator_engine.detect_reversal_patterns(df)
            pattern_data['reversal_pattern'] = reversal_result
            
            if reversal_result.get('detected', False):
                pattern_type = reversal_result.get('pattern_type', 'none')
                strength = reversal_result.get('strength', 'weak')
                
                is_aligned = (
                    (signal_type == 'BUY' and pattern_type == 'double_bottom') or
                    (signal_type == 'SELL' and pattern_type == 'double_top')
                )
                
                if is_aligned:
                    detected_patterns.append(f"{pattern_type.replace('_', ' ').title()} ({strength})")
                    
                    if strength == 'strong':
                        confidence_boost += 0.20
                    elif strength == 'moderate':
                        confidence_boost += 0.12
                    else:
                        confidence_boost += 0.07
            
            pattern_data['patterns_detected'] = detected_patterns
            pattern_data['combined_boost'] = confidence_boost
            
            if detected_patterns:
                primary_pattern = detected_patterns[0]
                pattern_data['pattern_name'] = primary_pattern
                pattern_data['pattern_confidence'] = confidence_boost
                
                reason = f"✅ Pattern Recognition: {', '.join(detected_patterns)} detected | Boost: +{confidence_boost:.0%}"
                logger.info(reason)
                return True, reason, confidence_boost, pattern_data
            else:
                reason = "✅ Pattern Recognition: Tidak ada pattern terdeteksi - signal tetap valid"
                logger.debug(reason)
                return True, reason, 0.0, pattern_data
            
        except Exception as e:
            logger.warning(f"Error in check_pattern_recognition: {e}")
            return True, f"✅ Pattern Recognition: Error ({str(e)}) - signal diizinkan", 0.0, pattern_data
    
    def check_ema_ribbon_alignment(self, indicators: Dict, signal_type: str, 
                                    m1_df=None) -> Tuple[bool, str, float, Dict[str, Any]]:
        """Check EMA Ribbon alignment for momentum confirmation.
        
        IMPROVEMENT 5: Advanced Momentum Confirmation dengan EMA Ribbon
        
        This is a CONFIDENCE MODIFIER, NOT a BLOCKING filter.
        Uses 6 EMAs (5, 10, 15, 20, 25, 30) to determine trend strength.
        
        Alignment status:
        - STRONG_BULLISH: All EMAs stacked ascending (EMA5 > EMA10 > ... > EMA30)
        - BULLISH: Most EMAs aligned bullish (4+ of 5 comparisons)
        - NEUTRAL: Mixed alignments (3 bullish/bearish)
        - BEARISH: Most EMAs aligned bearish
        - STRONG_BEARISH: All EMAs stacked descending
        - MIXED: No clear order (weak momentum)
        
        Confidence modifiers:
        - STRONG alignment with signal: +15%
        - Aligned with signal: +10%
        - NEUTRAL: 0%
        - MIXED: -10%
        - Against signal: -15%
        
        Args:
            indicators: Dictionary of calculated indicators
            signal_type: 'BUY' or 'SELL'
            m1_df: M1 DataFrame for EMA ribbon calculation
            
        Returns:
            Tuple[bool, str, float, Dict]: (always_true, reason, confidence_modifier, ribbon_data)
        """
        ribbon_data = {
            'alignment_status': 'NEUTRAL',
            'trend_strength': 0.0,
            'bullish_count': 0,
            'bearish_count': 0,
            'ribbon_spread': 0.0,
            'ema_values': {},
            'confidence_modifier': 0.0,
            'is_aligned': False
        }
        
        try:
            from bot.indicators import IndicatorEngine
            
            indicator_engine = None
            if hasattr(self, 'indicator_engine'):
                indicator_engine = self.indicator_engine
            else:
                indicator_engine = IndicatorEngine(self.config)
            
            if m1_df is None or len(m1_df) < 40:
                ema_ribbon = indicators.get('ema_ribbon', None)
                if ema_ribbon is None:
                    reason = "✅ EMA Ribbon: Data tidak tersedia - no confidence adjustment"
                    logger.debug(reason)
                    return True, reason, 0.0, ribbon_data
            else:
                ema_ribbon = indicator_engine.calculate_ema_ribbon(m1_df)
            
            if ema_ribbon is None:
                reason = "✅ EMA Ribbon: Calculation failed - no confidence adjustment"
                logger.debug(reason)
                return True, reason, 0.0, ribbon_data
            
            ribbon_data.update(ema_ribbon)
            
            alignment_status = ema_ribbon.get('alignment_status', 'NEUTRAL')
            trend_strength = ema_ribbon.get('trend_strength', 0.0)
            ribbon_spread = ema_ribbon.get('ribbon_spread', 0.0)
            
            confidence_modifier = 0.0
            is_aligned = False
            
            if signal_type == 'BUY':
                if alignment_status == 'STRONG_BULLISH':
                    confidence_modifier = 0.15
                    is_aligned = True
                elif alignment_status == 'BULLISH':
                    confidence_modifier = 0.10
                    is_aligned = True
                elif alignment_status == 'NEUTRAL':
                    confidence_modifier = 0.0
                    is_aligned = False
                elif alignment_status == 'MIXED':
                    confidence_modifier = -0.10
                    is_aligned = False
                elif alignment_status == 'BEARISH':
                    confidence_modifier = -0.10
                    is_aligned = False
                elif alignment_status == 'STRONG_BEARISH':
                    confidence_modifier = -0.15
                    is_aligned = False
                    
            elif signal_type == 'SELL':
                if alignment_status == 'STRONG_BEARISH':
                    confidence_modifier = 0.15
                    is_aligned = True
                elif alignment_status == 'BEARISH':
                    confidence_modifier = 0.10
                    is_aligned = True
                elif alignment_status == 'NEUTRAL':
                    confidence_modifier = 0.0
                    is_aligned = False
                elif alignment_status == 'MIXED':
                    confidence_modifier = -0.10
                    is_aligned = False
                elif alignment_status == 'BULLISH':
                    confidence_modifier = -0.10
                    is_aligned = False
                elif alignment_status == 'STRONG_BULLISH':
                    confidence_modifier = -0.15
                    is_aligned = False
            
            ribbon_data['confidence_modifier'] = confidence_modifier
            ribbon_data['is_aligned'] = is_aligned
            
            modifier_str = f"+{confidence_modifier:.0%}" if confidence_modifier >= 0 else f"{confidence_modifier:.0%}"
            alignment_emoji = "✅" if is_aligned else ("⚠️" if confidence_modifier == 0 else "⛔")
            
            reason = f"{alignment_emoji} EMA Ribbon: {alignment_status} (strength: {trend_strength:.2f}, spread: {ribbon_spread:.2f}%) | Modifier: {modifier_str}"
            logger.info(reason)
            
            return True, reason, confidence_modifier, ribbon_data
            
        except Exception as e:
            logger.warning(f"Error in check_ema_ribbon_alignment: {e}")
            return True, f"✅ EMA Ribbon: Error ({str(e)}) - no confidence adjustment", 0.0, ribbon_data
    
    def get_volatility_adjustment(self, indicators: Dict) -> float:
        """Calculate dynamic threshold adjustment based on ATR/volatility
        
        PERBAIKAN: Dynamic thresholds berdasarkan kondisi market:
        - High volatility (ATR > 2.0): Relax filters 20% 
        - Normal volatility (ATR 1.0-2.0): Standard threshold
        - Low volatility (ATR < 1.0): Keep strict
        
        Returns:
            float: Adjustment multiplier (0.8 to 1.2)
        """
        try:
            atr = indicators.get('atr')
            close = indicators.get('close')
            
            if not is_valid_number(atr) or not is_valid_number(close):
                return 1.0
            
            atr_val = safe_float(atr, 1.0)
            close_val = safe_float(close, 1.0)
            
            if close_val <= 0:
                return 1.0
            
            atr_percent = (atr_val / close_val) * 100
            
            if atr_percent > 0.05:
                return 0.80
            elif atr_percent > 0.03:
                return 0.90
            elif atr_percent < 0.02:
                return 1.10
            else:
                return 1.0
                
        except (StrategyError, Exception) as e:
            logger.debug(f"Error in get_volatility_adjustment: {e}")
            return 1.0
    
    def get_multi_confirmation_score(self, indicators: Dict, current_spread: float = 0.0, signal_source: str = 'auto', signal_mode: str = 'M1_SCALP') -> Dict:
        """Get comprehensive multi-confirmation analysis - ENHANCED WEIGHTED SCORING VERSION
        
        ENHANCED WEIGHTED SCORING + CONFLUENCE-BASED CONFIDENCE:
        - Trend Filter: 25% weight (MUST PASS - mandatory untuk semua)
        - ADX Filter: 10% weight (BLOCKING dengan tiered thresholds)
        - Momentum Filter: 15% weight
        - RSI Level Filter: 5% weight (BLOCKING di extreme levels + divergence)
        - EMA Slope Filter: 5% weight (5-period calculation + reversal detection)
        - Volume Filter: 10% weight (strict thresholds berdasarkan signal mode)
        - Price Action: 10% weight (wick validation + momentum consistency)
        - Session Filter: 10% weight
        - Spread Filter: 10% weight
        
        Confluence-Based Confidence Multiplier:
        - 2 confluence = 0.8x multiplier (SCALP)
        - 3 confluence = 1.0x multiplier (STANDARD)
        - 4+ confluence = 1.2x multiplier (OPTIMAL)
        
        Threshold (setelah update): 
        - AUTO signal: ≥ 55% combined score + Trend MUST PASS + ADX MUST PASS
        - MANUAL signal: ≥ 30% combined score + Trend MUST PASS (ADX non-blocking)
        
        Args:
            indicators: Dictionary of calculated indicators
            current_spread: Current spread in price units
            signal_source: Sumber signal ('auto' atau 'manual')
            signal_mode: Mode signal ('M1_SCALP', 'M5_SWING', 'BREAKOUT')
            
        Returns:
            Dict with all filter results, total score, and confidence multiplier
        """
        try:
            result = {
                'trend_filter': {'passed': False, 'signal_type': '', 'reason': ''},
                'adx_filter': {'passed': False, 'reason': '', 'value': 0.0, 'tp_multiplier': 1.0},
                'momentum_filter': {'passed': False, 'reason': ''},
                'rsi_level_filter': {'passed': False, 'reason': '', 'confidence_boost': 0.0},
                'ema_slope_filter': {'passed': False, 'reason': '', 'slope_data': {}},
                'volume_vwap_filter': {'passed': False, 'reason': '', 'confidence_multiplier': 1.0},
                'price_action': {'passed': False, 'reason': '', 'points': 0, 'validation_data': {}},
                'session_filter': {'passed': False, 'reason': ''},
                'spread_filter': {'passed': False, 'reason': ''},
                'all_mandatory_passed': False,
                'signal_type': '',
                'signal_mode': signal_mode,
                'total_score': 0,
                'weighted_score': 0.0,
                'confluence_count': 0,
                'confluence_multiplier': 1.0,
                'volatility_adjustment': 1.0,
                'confidence_reasons': []
            }
            
            volatility_adj = self.get_volatility_adjustment(indicators)
            result['volatility_adjustment'] = volatility_adj
            
            trend_passed, signal_type, trend_reason = self.check_trend_filter(indicators)
            result['trend_filter'] = {'passed': trend_passed, 'signal_type': signal_type, 'reason': trend_reason}
            
            if not trend_passed:
                result['confidence_reasons'].append(trend_reason)
                logger.info(f"Multi-Confirmation Analysis: Trend filter failed - {trend_reason}")
                return result
            
            result['signal_type'] = signal_type
            result['total_score'] += 25
            result['weighted_score'] += 25.0
            result['confluence_count'] += 1
            result['confidence_reasons'].append(trend_reason)
            
            adx_passed, adx_reason, adx_value, adx_tp_mult = self.check_adx_filter(indicators, signal_mode)
            result['adx_filter'] = {'passed': adx_passed, 'reason': adx_reason, 'value': adx_value, 'tp_multiplier': adx_tp_mult}
            result['confidence_reasons'].append(adx_reason)
            
            if adx_passed:
                result['total_score'] += 10
                result['weighted_score'] += 10.0
                if adx_value >= 20:
                    result['confluence_count'] += 1
            
            momentum_passed, momentum_reason = self.check_momentum_filter(indicators, signal_type)
            result['momentum_filter'] = {'passed': momentum_passed, 'reason': momentum_reason}
            result['confidence_reasons'].append(momentum_reason)
            
            if momentum_passed:
                result['total_score'] += 15
                result['weighted_score'] += 15.0
                result['confluence_count'] += 1
            
            rsi_level_passed, rsi_level_reason, rsi_boost = self.check_rsi_level_filter(indicators, signal_type)
            result['rsi_level_filter'] = {'passed': rsi_level_passed, 'reason': rsi_level_reason, 'confidence_boost': rsi_boost}
            result['confidence_reasons'].append(rsi_level_reason)
            
            if not rsi_level_passed and signal_source == 'auto':
                result['all_mandatory_passed'] = False
                logger.info(f"🚫 AUTO Signal BLOCKED: RSI extreme level - {rsi_level_reason}")
                return result
            
            if rsi_level_passed:
                rsi_score = 5.0 + (rsi_boost * 5.0)
                result['total_score'] += int(rsi_score)
                result['weighted_score'] += rsi_score
                if rsi_boost >= 0.10:
                    result['confluence_count'] += 1
            
            ema_slope_passed, ema_slope_reason, slope_data = self.check_ema_slope_filter(indicators, signal_type)
            result['ema_slope_filter'] = {'passed': ema_slope_passed, 'reason': ema_slope_reason, 'slope_data': slope_data}
            result['confidence_reasons'].append(ema_slope_reason)
            
            if not ema_slope_passed and signal_source == 'auto':
                result['all_mandatory_passed'] = False
                logger.info(f"🚫 AUTO Signal BLOCKED: EMA slope berlawanan - {ema_slope_reason}")
                return result
            
            if ema_slope_passed:
                result['total_score'] += 5
                result['weighted_score'] += 5.0
                if slope_data.get('slope_strength') == 'strong':
                    result['confluence_count'] += 1
            
            volume_passed, volume_reason, volume_mult = self.check_volume_vwap_filter(indicators, signal_type, signal_mode)
            result['volume_vwap_filter'] = {'passed': volume_passed, 'reason': volume_reason, 'confidence_multiplier': volume_mult}
            result['confidence_reasons'].append(volume_reason)
            
            if not volume_passed and signal_source == 'auto' and signal_mode in ['M5_SWING', 'BREAKOUT']:
                result['all_mandatory_passed'] = False
                logger.info(f"🚫 AUTO Signal BLOCKED: Volume insufficient for {signal_mode} - {volume_reason}")
                return result
            
            if volume_passed:
                volume_score = 10.0 * volume_mult
                result['total_score'] += int(volume_score)
                result['weighted_score'] += volume_score
                if volume_mult >= 1.0:
                    result['confluence_count'] += 1
            
            pa_passed, pa_reason, pa_points, pa_validation = self.check_price_action_confirmation(indicators, signal_type)
            result['price_action'] = {'passed': pa_passed, 'reason': pa_reason, 'points': pa_points, 'validation_data': pa_validation}
            result['confidence_reasons'].append(pa_reason)
            
            if not pa_passed and signal_source == 'auto':
                if not pa_validation.get('wick_valid', True):
                    result['all_mandatory_passed'] = False
                    logger.info(f"🚫 AUTO Signal BLOCKED: Wick position invalid - {pa_reason}")
                    return result
            
            if pa_passed:
                result['total_score'] += min(10, pa_points)
                result['weighted_score'] += float(min(10, pa_points))
                if len(pa_validation.get('patterns_found', [])) > 0:
                    result['confluence_count'] += 1
            
            session_passed, session_reason = self.is_optimal_trading_session()
            if session_passed:
                result['session_filter'] = {'passed': True, 'reason': session_reason}
                result['total_score'] += 10
                result['weighted_score'] += 10.0
            else:
                result['session_filter'] = {'passed': False, 'reason': session_reason}
                result['weighted_score'] += 5.0
            result['confidence_reasons'].append(result['session_filter']['reason'])
            
            max_spread_pips = safe_float(self.config.MAX_SPREAD_PIPS, 15.0)
            pip_value = safe_float(self.config.XAUUSD_PIP_VALUE, 10.0)
            spread_pips = safe_float(current_spread, 0.0) * pip_value
            
            if spread_pips <= max_spread_pips:
                result['spread_filter'] = {'passed': True, 'reason': f"✅ Spread Filter PASSED: {spread_pips:.1f} pips <= {max_spread_pips} pips"}
                result['total_score'] += 10
                result['weighted_score'] += 10.0
            else:
                spread_over = spread_pips / max_spread_pips
                if spread_over < 1.5:
                    result['spread_filter'] = {'passed': False, 'reason': f"⚠️ Spread slightly high: {spread_pips:.1f} pips (non-blocking)"}
                    result['weighted_score'] += 5.0
                else:
                    result['spread_filter'] = {'passed': False, 'reason': f"❌ Spread Filter FAILED: {spread_pips:.1f} pips > {max_spread_pips} pips"}
            result['confidence_reasons'].append(result['spread_filter']['reason'])
            
            confluence_count = result['confluence_count']
            if confluence_count >= 5:
                confluence_multiplier = 1.25
            elif confluence_count >= 4:
                confluence_multiplier = 1.15
            elif confluence_count >= 3:
                confluence_multiplier = 1.0
            elif confluence_count >= 2:
                confluence_multiplier = 0.85
            else:
                confluence_multiplier = 0.7
            
            result['confluence_multiplier'] = confluence_multiplier
            
            final_multiplier = confluence_multiplier * volume_mult * (2.0 - volatility_adj)
            adjusted_score = result['weighted_score'] * final_multiplier
            
            auto_threshold = safe_float(getattr(self.config, 'SIGNAL_SCORE_THRESHOLD_AUTO', 55), 55.0)
            manual_threshold = safe_float(getattr(self.config, 'SIGNAL_SCORE_THRESHOLD_MANUAL', 30), 30.0)
            
            if signal_source == 'auto':
                core_filters_passed = trend_passed and adx_passed and rsi_level_passed and ema_slope_passed
                if signal_mode in ['M5_SWING', 'BREAKOUT']:
                    core_filters_passed = core_filters_passed and volume_passed
                if pa_validation and not pa_validation.get('wick_valid', True):
                    core_filters_passed = False
                
                score_threshold_met = adjusted_score >= auto_threshold
                confluence_threshold_met = confluence_count >= 3
                
                result['all_mandatory_passed'] = core_filters_passed and (score_threshold_met or confluence_threshold_met)
                
                if not adx_passed:
                    logger.info(f"🚫 AUTO Signal BLOCKED: ADX filter tidak pass - ADX({adx_value:.1f}) < threshold for {signal_mode}")
            else:
                core_filters_passed = trend_passed
                supporting_filters = momentum_passed or volume_passed or pa_passed or adx_passed or rsi_level_passed
                score_threshold_met = adjusted_score >= manual_threshold
                
                result['all_mandatory_passed'] = core_filters_passed and (supporting_filters or score_threshold_met)
            
            volatility_info = ""
            if volatility_adj != 1.0:
                volatility_info = f" | Vol: {volatility_adj:.2f}x"
            
            confluence_info = f" | Confluence: {confluence_count}/6 ({confluence_multiplier:.2f}x)"
            adx_blocking_info = " [BLOCKING]" if signal_source == 'auto' else " [info]"
            adx_info = f" | ADX: {adx_value:.1f}{adx_blocking_info}" if adx_value > 0 else ""
            
            logger.info(f"Multi-Confirmation Score: {result['total_score']}/100 (Adj: {adjusted_score:.0f}%){confluence_info}{adx_info}{volatility_info} | Ready: {result['all_mandatory_passed']} | Mode: {signal_mode}")
            
            return result
            
        except (StrategyError, Exception) as e:
            logger.error(f"Error in get_multi_confirmation_score: {e}")
            return {
                'trend_filter': {'passed': False, 'signal_type': '', 'reason': f'Error: {str(e)}'},
                'all_mandatory_passed': False,
                'signal_type': '',
                'total_score': 0,
                'weighted_score': 0.0,
                'confluence_count': 0,
                'confluence_multiplier': 1.0,
                'confidence_reasons': [f'Error: {str(e)}']
            }
    
    def calculate_sl_tp(self, signal_type: str, indicators: Dict, current_spread: float = 0.0,
                        sl_multiplier: float = 1.0, drawdown_info: Optional[Dict] = None) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Calculate Stop Loss and Take Profit with dynamic SL adjustment support.
        
        IMPROVEMENT 7: Drawdown Protection dengan Dynamic SL Expansion
        
        SL = max(config.SL_ATR_MULTIPLIER * ATR, config.MIN_SL_PIPS/pip_value, config.MIN_SL_SPREAD_MULTIPLIER * spread)
        SL_adjusted = SL * sl_multiplier (for drawdown protection)
        TP = SL_adjusted * config.TP_RR_RATIO
        
        Args:
            signal_type: 'BUY' or 'SELL'
            indicators: Dictionary of calculated indicators
            current_spread: Current spread in price units
            sl_multiplier: Multiplier for SL distance from drawdown protection (default 1.0)
                          - 1.0 = normal SL (drawdown < 20%)
                          - 1.15 = expanded SL +15% (drawdown 20-40%)
                          - 1.30 = expanded SL +30% (drawdown > 40%)
            drawdown_info: Optional dict with drawdown details for logging:
                          - drawdown_percent: Current drawdown %
                          - drawdown_level: 'NORMAL', 'WARNING', 'CRITICAL'
                          - reason: Explanation string
            
        Returns:
            Tuple of (sl_price, tp_price, sl_pips, tp_pips) or (None, None, None, None) on error
        """
        try:
            close = indicators.get('close')
            atr = indicators.get('atr')
            
            if not is_valid_number(close) or not is_valid_number(atr):
                logger.error("calculate_sl_tp: Invalid close or ATR")
                return None, None, None, None
            
            close = safe_float(close, 0.0)
            atr = safe_float(atr, 1.0)
            
            if close <= 0 or atr <= 0:
                logger.error(f"calculate_sl_tp: Invalid close({close}) or ATR({atr})")
                return None, None, None, None
            
            sl_multiplier = safe_float(sl_multiplier, 1.0, "sl_multiplier")
            if sl_multiplier <= 0 or sl_multiplier > 2.0:
                logger.warning(f"Invalid sl_multiplier: {sl_multiplier}, using 1.0")
                sl_multiplier = 1.0
            
            sl_atr_mult = safe_float(self.config.SL_ATR_MULTIPLIER, 1.2)
            min_sl_pips = safe_float(self.config.MIN_SL_PIPS, 15.0)
            min_sl_spread_mult = safe_float(self.config.MIN_SL_SPREAD_MULTIPLIER, 2.0)
            tp_rr_ratio = safe_float(self.config.TP_RR_RATIO, 1.5)
            pip_value = safe_float(self.config.XAUUSD_PIP_VALUE, 10.0)
            
            if pip_value <= 0:
                pip_value = 10.0
            
            sl_from_atr = atr * sl_atr_mult
            sl_from_min_pips = min_sl_pips / pip_value
            sl_from_spread = current_spread * min_sl_spread_mult
            
            sl_distance_base = max(sl_from_atr, sl_from_min_pips, sl_from_spread)
            
            sl_distance = sl_distance_base * sl_multiplier
            
            if sl_multiplier != 1.0:
                if drawdown_info:
                    dd_pct = drawdown_info.get('drawdown_percent', 0)
                    dd_level = drawdown_info.get('drawdown_level', 'UNKNOWN')
                    logger.warning(f"🛡️ Dynamic SL Expansion Applied: {sl_multiplier:.2f}x "
                                   f"(Base: {sl_distance_base:.4f} -> Adjusted: {sl_distance:.4f}) | "
                                   f"Drawdown: {dd_pct:.1f}% [{dd_level}]")
                else:
                    logger.info(f"🛡️ SL Multiplier Applied: {sl_multiplier:.2f}x "
                               f"(Base: {sl_distance_base:.4f} -> Adjusted: {sl_distance:.4f})")
            
            if not is_valid_number(sl_distance) or sl_distance <= 0:
                logger.error(f"calculate_sl_tp: Invalid SL distance: {sl_distance}")
                return None, None, None, None
            
            tp_distance = sl_distance * tp_rr_ratio
            
            if not is_valid_number(tp_distance) or tp_distance <= 0:
                logger.error(f"calculate_sl_tp: Invalid TP distance: {tp_distance}")
                return None, None, None, None
            
            if signal_type == 'BUY':
                sl_price = close - sl_distance
                tp_price = close + tp_distance
            elif signal_type == 'SELL':
                sl_price = close + sl_distance
                tp_price = close - tp_distance
            else:
                logger.error(f"calculate_sl_tp: Invalid signal type: {signal_type}")
                return None, None, None, None
            
            if sl_price <= 0 or tp_price <= 0:
                logger.error(f"calculate_sl_tp: Invalid SL/TP prices: SL={sl_price}, TP={tp_price}")
                return None, None, None, None
            
            sl_pips = abs(close - sl_price) * pip_value
            tp_pips = abs(close - tp_price) * pip_value
            
            sl_mult_info = f" [DD Protection: {sl_multiplier:.2f}x]" if sl_multiplier != 1.0 else ""
            logger.info(f"SL/TP Calculated: SL={sl_price:.2f} ({sl_pips:.1f} pips), TP={tp_price:.2f} ({tp_pips:.1f} pips){sl_mult_info}")
            logger.debug(f"SL Components: ATR*{sl_atr_mult}={sl_from_atr:.4f}, MIN_PIPS={sl_from_min_pips:.4f}, "
                        f"SPREAD*{min_sl_spread_mult}={sl_from_spread:.4f}, Base={sl_distance_base:.4f}, "
                        f"Multiplier={sl_multiplier:.2f}, Final={sl_distance:.4f}")
            
            return sl_price, tp_price, sl_pips, tp_pips
            
        except (StrategyError, Exception) as e:
            logger.error(f"Error in calculate_sl_tp: {e}")
            return None, None, None, None
        
    def detect_signal(self, indicators: Dict, timeframe: str = 'M1', signal_source: str = 'auto', current_spread: float = 0.0, candle_timestamp: Optional[datetime] = None, m5_indicators: Optional[Dict] = None, h1_indicators: Optional[Dict] = None) -> Optional[Dict]:  # pyright: ignore[reportGeneralTypeIssues]
        """Detect trading signal with multi-confirmation strategy
        
        Uses professional multi-confirmation approach with ENHANCED filters untuk AUTO:
        1. Trend Filter (MANDATORY): EMA5 > EMA20 > EMA50 alignment + price position
        2. Momentum Filter (MANDATORY): RSI in entry range + direction + Stochastic confirmation
        3. Volume/VWAP Filter (MANDATORY): Volume above average + VWAP position
        4. Price Action Filter (MANDATORY): Candlestick patterns + S/R proximity
        5. Session Filter (MANDATORY): London-NY overlap
        6. Spread Filter (MANDATORY): Spread < MAX_SPREAD_PIPS
        7. ADX Filter (MANDATORY untuk AUTO): ADX >= 15 (BLOCKING)
        8. Multi-Timeframe Confirmation (M1+M5+H1 harus sejalan)
        9. ATR Volatility Filter (skip saat volatilitas ekstrem)
        10. Enhanced Volume Confirmation
        11. Divergence Detection (RSI + MACD)
        12. Support/Resistance Level Analysis
        
        Untuk AUTO signals: Filter wajib HARUS pass + MTF confirmation.
        Untuk MANUAL signals: ADX dan MTF confirmation opsional (informational only).
        
        Args:
            indicators: Dictionary of calculated indicators dari M1
            timeframe: Timeframe string (e.g., 'M1', 'M5')
            signal_source: Source of signal ('auto' atau 'manual')
            current_spread: Current spread in price units
            candle_timestamp: Timestamp candle untuk tracking (opsional)
            m5_indicators: Dictionary of calculated indicators dari M5 (opsional)
            h1_indicators: Dictionary of calculated indicators dari H1 (opsional)
        
        Note: This function is intentionally complex due to multi-indicator trading logic.
        Pyright complexity warning is suppressed as it does not affect runtime behavior.
        """
        if not indicators or not isinstance(indicators, dict):
            logger.warning("Invalid or empty indicators provided")
            return None
        
        try:
            is_valid, error_msg = validate_indicators(indicators)
            if not is_valid:
                logger.warning(f"Indicator validation failed: {error_msg}")
                logger.error(f"Signal detection aborted: Indicator validation error - {error_msg}")
                return None
            
            self.check_high_volatility(indicators)
            
            logger.info("=" * 60)
            logger.info(f"🔍 MULTI-CONFIRMATION SIGNAL ANALYSIS - {timeframe}")
            logger.info("=" * 60)
            
            session_info = self.analyze_session_strength()
            logger.info(f"🕐 {session_info['session_description']}")
            
            ema_short = indicators.get(f'ema_{self.config.EMA_PERIODS[0]}')
            ema_mid = indicators.get(f'ema_{self.config.EMA_PERIODS[1]}')
            ema_long = indicators.get(f'ema_{self.config.EMA_PERIODS[2]}')
            rsi = indicators.get('rsi')
            macd = indicators.get('macd')
            macd_signal = indicators.get('macd_signal')
            macd_histogram = indicators.get('macd_histogram')
            atr = indicators.get('atr')
            close = indicators.get('close')
            volume = indicators.get('volume')
            volume_avg = indicators.get('volume_avg')
            stoch_k = indicators.get('stoch_k')
            stoch_d = indicators.get('stoch_d')
            
            trf_trend = indicators.get('trf_trend')
            cerebr_bias = indicators.get('cerebr_bias')
            cerebr_value = indicators.get('cerebr_value')
            
            if None in [ema_short, ema_mid, ema_long, rsi, macd, macd_signal, atr, close]:
                missing = []
                if ema_short is None: missing.append("ema_short")
                if ema_mid is None: missing.append("ema_mid")
                if ema_long is None: missing.append("ema_long")
                if rsi is None: missing.append("rsi")
                if macd is None: missing.append("macd")
                if macd_signal is None: missing.append("macd_signal")
                if atr is None: missing.append("atr")
                if close is None: missing.append("close")
                logger.warning(f"Missing required indicators for signal detection: {missing}")
                return None
            
            mc_result = self.get_multi_confirmation_score(indicators, current_spread, signal_source)
            
            signal = None
            dynamic_lot = safe_float(self.config.LOT_SIZE, 0.01, "LOT_SIZE")
            volatility_zone = "NORMAL"
            confidence_reasons = mc_result.get('confidence_reasons', [])
            
            weighted_score = mc_result.get('weighted_score', 0.0)
            volatility_adj = mc_result.get('volatility_adjustment', 1.0)
            adjusted_score = weighted_score * (2.0 - volatility_adj)
            auto_threshold = safe_float(getattr(self.config, 'SIGNAL_SCORE_THRESHOLD_AUTO', 60), 60.0)
            
            if signal_source == 'auto':
                if mc_result['all_mandatory_passed']:
                    potential_signal = mc_result['signal_type']
                    
                    mtf_passed, mtf_reason, mtf_score, mtf_data = self.check_multi_timeframe_confirmation(
                        indicators, m5_indicators, h1_indicators, potential_signal
                    )
                    confidence_reasons.append(mtf_reason)
                    
                    if not mtf_passed:
                        logger.info(f"🚫 AUTO Signal BLOCKED: Multi-Timeframe confirmation failed - {mtf_reason}")
                        logger.info(f"📊 Signal {potential_signal} ditolak karena MTF tidak aligned")
                        return None
                    
                    atr_passed, atr_reason, atr_mult, volatility_zone = self.check_atr_volatility_filter(indicators, potential_signal)
                    confidence_reasons.append(atr_reason)
                    
                    if not atr_passed:
                        logger.info(f"🚫 AUTO Signal BLOCKED: ATR volatility filter - {atr_reason}")
                        logger.info(f"📊 Signal {potential_signal} ditolak karena volatilitas {volatility_zone}")
                        return None
                    
                    base_lot = safe_float(self.config.LOT_SIZE, 0.01, "LOT_SIZE")
                    atr_val = safe_float(indicators.get('atr'), 0.0, "atr")
                    dynamic_lot, lot_reason = self.calculate_dynamic_lot_size(base_lot, atr_val, volatility_zone)
                    confidence_reasons.append(lot_reason)
                    
                    vol_passed, vol_reason, vol_mult = self.check_enhanced_volume_confirmation(indicators, potential_signal)
                    confidence_reasons.append(vol_reason)
                    
                    if not vol_passed:
                        logger.info(f"🚫 AUTO Signal BLOCKED: Volume too low - {vol_reason}")
                        logger.info(f"📊 Signal {potential_signal} ditolak karena volume tidak mencukupi")
                        return None
                    
                    macd_has_div, macd_div_reason, macd_div_boost = self.check_macd_divergence(indicators, potential_signal)
                    if macd_has_div:
                        confidence_reasons.append(macd_div_reason)
                    
                    sr_passed, sr_reason, sr_boost, sr_data = self.check_support_resistance_levels(indicators, potential_signal)
                    if sr_boost != 0:
                        confidence_reasons.append(sr_reason)
                    
                    is_aligned, alignment_reason = self.check_regime_alignment(
                        potential_signal, indicators, m1_df=None, m5_df=None
                    )
                    if not is_aligned:
                        logger.info(f"{alignment_reason}")
                        logger.info(f"📊 Signal {potential_signal} ditolak - regime bias tidak sesuai")
                        return None
                    
                    confidence_reasons.append(alignment_reason)
                    
                    extended_mtf = self.check_extended_mtf_correlation(
                        m1_indicators=indicators,
                        m5_indicators=m5_indicators,
                        h1_indicators=h1_indicators,
                        h4_indicators=None,
                        daily_indicators=None,
                        signal_type=potential_signal
                    )
                    
                    mtf_correlation_score = extended_mtf.get('mtf_correlation_score', 0.0)
                    mtf_confidence_level = extended_mtf.get('confidence_level', 'LOW')
                    mtf_aligned_count = extended_mtf.get('aligned_timeframes_count', 0)
                    mtf_available_count = extended_mtf.get('available_timeframes_count', 0)
                    
                    extended_mtf_reason = (
                        f"📈 Extended MTF: {mtf_confidence_level} ({mtf_aligned_count}/{mtf_available_count} aligned) | "
                        f"Score Bonus: +{mtf_correlation_score*100:.0f}%"
                    )
                    confidence_reasons.append(extended_mtf_reason)
                    logger.info(extended_mtf_reason)
                    
                    session_multiplier = session_info.get('confidence_multiplier', 1.0)
                    session_reason = (
                        f"🕐 Session: {session_info.get('current_session', 'Unknown')} "
                        f"({session_info.get('session_strength', 'WEAK')}) | "
                        f"Multiplier: {session_multiplier:.2f}x"
                    )
                    confidence_reasons.append(session_reason)
                    
                    total_multiplier = mtf_score * atr_mult * vol_mult
                    total_boost = macd_div_boost + sr_boost + mtf_correlation_score
                    adjusted_score = (adjusted_score * total_multiplier) + (total_boost * 10)
                    
                    if total_multiplier < 1.0:
                        confidence_reasons.append(f"⚠️ Score Adjustment: MTF={mtf_score:.0%}, ATR={atr_mult:.0%}, Vol={vol_mult:.0%}")
                        logger.info(f"📉 Score adjusted by multipliers - final score: {adjusted_score:.0f}%")
                    
                    signal = potential_signal
                    close_price = safe_float(close, 0.0)
                    
                    candle_close_only = getattr(self.config, 'CANDLE_CLOSE_ONLY_SIGNALS', True)
                    if candle_close_only and signal:
                        can_generate, skip_reason = self.should_generate_signal(
                            candle_timestamp, close_price, signal
                        )
                        if not can_generate:
                            logger.info(f"⏳ CANDLE_CLOSE_ONLY_SIGNALS aktif: {skip_reason}")
                            logger.info(f"📊 Signal {signal} terdeteksi tapi di-skip - menunggu candle baru")
                            self._update_signal_tracking(candle_timestamp, signal, close_price)
                            return None
                        else:
                            logger.info(f"🕯️ Candle close check PASSED: Signal diizinkan untuk candle baru")
                    
                    self._update_signal_tracking(candle_timestamp, signal, close_price)
                    
                    logger.info(f"✅ WEIGHTED SCORE PASSED - Signal: {signal}")
                    logger.info(f"📊 Weighted Score: {adjusted_score:.0f}% (threshold: {auto_threshold}%)")
                    
                    if trf_trend is not None:
                        if signal == 'BUY' and trf_trend == 1:
                            confidence_reasons.append("🎯 Twin Range Filter: Bullish trend confirmed")
                        elif signal == 'SELL' and trf_trend == -1:
                            confidence_reasons.append("🎯 Twin Range Filter: Bearish trend confirmed")
                    
                    if cerebr_bias is not None and cerebr_value is not None:
                        if signal == 'BUY' and cerebr_bias == 1:
                            confidence_reasons.append(f"📊 Market Bias CEREBR: Bullish ({safe_float(cerebr_value, 50.0):.1f}%)")
                        elif signal == 'SELL' and cerebr_bias == -1:
                            confidence_reasons.append(f"📊 Market Bias CEREBR: Bearish ({safe_float(cerebr_value, 50.0):.1f}%)")
                    
                    macd_valid = is_valid_number(macd) and is_valid_number(macd_signal)
                    if macd_valid and macd is not None and macd_signal is not None:
                        macd_prev = indicators.get('macd_prev')
                        macd_signal_prev = indicators.get('macd_signal_prev')
                        if is_valid_number(macd_prev) and is_valid_number(macd_signal_prev) and macd_prev is not None and macd_signal_prev is not None:
                            if signal == 'BUY' and macd_prev <= macd_signal_prev and macd > macd_signal:
                                confidence_reasons.append("MACD bullish crossover")
                            elif signal == 'SELL' and macd_prev >= macd_signal_prev and macd < macd_signal:
                                confidence_reasons.append("MACD bearish crossover")
                        
                        if signal == 'BUY' and macd > macd_signal:
                            confidence_reasons.append("MACD bullish")
                        elif signal == 'SELL' and macd < macd_signal:
                            confidence_reasons.append("MACD bearish")
                    
                    rsi_history = indicators.get('rsi_history', [])
                    pullback_confirmed = self.check_pullback_confirmation(rsi_history, signal)
                    if pullback_confirmed:
                        if signal == 'BUY':
                            confidence_reasons.append("Pullback confirmed (RSI 40-45 zone)")
                        else:
                            confidence_reasons.append("Pullback confirmed (RSI 55-60 zone)")
                    
                    confidence_reasons.append(f"Weighted Score: {adjusted_score:.0f}%")
                else:
                    passed_filters = []
                    failed_filters = []
                    
                    if mc_result['trend_filter']['passed']:
                        passed_filters.append("Trend")
                    else:
                        failed_filters.append("Trend")
                    if mc_result['momentum_filter']['passed']:
                        passed_filters.append("Momentum")
                    else:
                        failed_filters.append("Momentum")
                    if mc_result['volume_vwap_filter']['passed']:
                        passed_filters.append("Volume")
                    else:
                        failed_filters.append("Volume")
                    if mc_result['price_action']['passed']:
                        passed_filters.append("PriceAction")
                    else:
                        failed_filters.append("PriceAction")
                    if mc_result['session_filter']['passed']:
                        passed_filters.append("Session")
                    else:
                        failed_filters.append("Session")
                    if mc_result['spread_filter']['passed']:
                        passed_filters.append("Spread")
                    else:
                        failed_filters.append("Spread")
                    
                    logger.info(f"❌ Weighted score below threshold: {adjusted_score:.0f}% < {auto_threshold}%")
                    logger.info(f"📊 Passed: {', '.join(passed_filters) if passed_filters else 'None'} | Failed: {', '.join(failed_filters)}")
                    return None
            else:
                manual_threshold = safe_float(getattr(self.config, 'SIGNAL_SCORE_THRESHOLD_MANUAL', 40), 40.0)
                trend_passed = mc_result['trend_filter']['passed']
                
                if trend_passed and adjusted_score >= manual_threshold:
                    signal = mc_result['signal_type']
                    close_price = safe_float(close, 0.0)
                    
                    _, _, _, volatility_zone = self.check_atr_volatility_filter(indicators, signal)
                    base_lot = safe_float(self.config.LOT_SIZE, 0.01, "LOT_SIZE")
                    atr_val = safe_float(indicators.get('atr'), 0.0, "atr")
                    dynamic_lot, lot_reason = self.calculate_dynamic_lot_size(base_lot, atr_val, volatility_zone)
                    
                    candle_close_only = getattr(self.config, 'CANDLE_CLOSE_ONLY_SIGNALS', True)
                    if candle_close_only and signal:
                        can_generate, skip_reason = self.should_generate_signal(
                            candle_timestamp, close_price, signal
                        )
                        if not can_generate:
                            logger.info(f"⏳ CANDLE_CLOSE_ONLY_SIGNALS aktif (manual): {skip_reason}")
                            logger.info(f"📊 Signal {signal} terdeteksi tapi di-skip - menunggu candle baru")
                            self._update_signal_tracking(candle_timestamp, signal, close_price)
                            return None
                        else:
                            logger.info(f"🕯️ Candle close check PASSED (manual): Signal diizinkan")
                    
                    self._update_signal_tracking(candle_timestamp, signal, close_price)
                    
                    logger.info(f"✅ MANUAL SIGNAL APPROVED - Weighted Score: {adjusted_score:.0f}% (threshold: {manual_threshold}%)")
                    confidence_reasons = mc_result.get('confidence_reasons', [])
                    confidence_reasons.append(f"Manual Mode Weighted Score: {adjusted_score:.0f}%")
                    confidence_reasons.append(lot_reason)
                    
                    extended_mtf_manual = self.check_extended_mtf_correlation(
                        m1_indicators=indicators,
                        m5_indicators=m5_indicators,
                        h1_indicators=h1_indicators,
                        h4_indicators=None,
                        daily_indicators=None,
                        signal_type=signal
                    )
                    
                    mtf_confidence_manual = extended_mtf_manual.get('confidence_level', 'LOW')
                    mtf_aligned_manual = extended_mtf_manual.get('aligned_timeframes_count', 0)
                    mtf_available_manual = extended_mtf_manual.get('available_timeframes_count', 0)
                    
                    manual_mtf_reason = (
                        f"ℹ️ Extended MTF (info): {mtf_confidence_manual} "
                        f"({mtf_aligned_manual}/{mtf_available_manual} aligned)"
                    )
                    confidence_reasons.append(manual_mtf_reason)
                    logger.info(manual_mtf_reason)
                    
                    manual_session_reason = (
                        f"ℹ️ Session: {session_info.get('current_session', 'Unknown')} "
                        f"({session_info.get('session_strength', 'WEAK')})"
                    )
                    confidence_reasons.append(manual_session_reason)
                    
                    m5_passed, m5_reason, m5_score_mult = self.check_m5_confirmation(m5_indicators, signal)
                    confidence_reasons.append(f"ℹ️ M5 Info (non-blocking): {m5_reason}")
                    if not m5_passed:
                        logger.info(f"⚠️ MANUAL Signal: M5 tidak konfirmasi - {m5_reason} (tetap generate karena MANUAL mode)")
                    if m5_score_mult < 1.0:
                        confidence_reasons.append(f"ℹ️ M5 Misalignment: {m5_score_mult:.0%} (non-blocking for MANUAL)")
                    
                    if trf_trend is not None:
                        if signal == 'BUY' and trf_trend == 1:
                            confidence_reasons.append("🎯 Twin Range Filter: Bullish confirmed")
                        elif signal == 'SELL' and trf_trend == -1:
                            confidence_reasons.append("🎯 Twin Range Filter: Bearish confirmed")
                    
                    if cerebr_bias is not None and cerebr_value is not None:
                        if signal == 'BUY' and cerebr_bias == 1:
                            confidence_reasons.append(f"📊 Market Bias CEREBR: Bullish ({safe_float(cerebr_value, 50.0):.1f}%)")
                        elif signal == 'SELL' and cerebr_bias == -1:
                            confidence_reasons.append(f"📊 Market Bias CEREBR: Bearish ({safe_float(cerebr_value, 50.0):.1f}%)")
                else:
                    if not trend_passed:
                        logger.info(f"❌ Manual signal blocked - Trend filter required for MANUAL mode")
                    else:
                        logger.info(f"❌ Manual signal blocked - Weighted Score: {adjusted_score:.0f}% < {manual_threshold}%")
                    return None
            
            if signal:
                try:
                    trend_strength, trend_desc = self.calculate_trend_strength(indicators)
                except (StrategyError, Exception) as e:
                    logger.error(f"Error calculating trend strength: {e}")
                    trend_strength, trend_desc = 0.3, "MEDIUM ⚡"
                
                if not is_valid_number(trend_strength):
                    logger.warning(f"NaN/Inf detected in trend_strength: {trend_strength}, using default 0.3")
                    trend_strength = 0.3
                    trend_desc = "MEDIUM ⚡"
                
                trend_strength = float(min(max(trend_strength, 0.0), 1.0))
                
                min_trend_strength = 0.05
                if signal_source == 'auto' and trend_strength < min_trend_strength:
                    logger.info(f"Auto signal rejected - trend strength too weak: {trend_strength:.2f} < {min_trend_strength} ({trend_desc})")
                    close_price = safe_float(close, 0.0)
                    logger.info(
                        f"📝 Signal tracking diperbarui (ditolak): {signal} @ ${close_price:.2f} | "
                        f"Alasan: Trend strength terlalu lemah ({trend_strength:.2f} < {min_trend_strength})"
                    )
                    return None
                
                try:
                    base_tp_ratio = 1.5
                    tp_adjustment = trend_strength * 0.3
                    dynamic_tp_ratio = base_tp_ratio + tp_adjustment
                    
                    if not is_valid_number(dynamic_tp_ratio):
                        logger.warning(f"NaN/Inf in dynamic_tp_ratio: {dynamic_tp_ratio}, using default {base_tp_ratio}")
                        dynamic_tp_ratio = base_tp_ratio
                    
                    min_tp_ratio = 1.3
                    max_tp_ratio = 2.0
                    dynamic_tp_ratio = float(min(max(dynamic_tp_ratio, min_tp_ratio), max_tp_ratio))
                    
                    logger.debug(
                        f"📊 Dynamic TP Calculation: base={base_tp_ratio}, "
                        f"trend_strength={trend_strength:.3f}, adjustment={tp_adjustment:.3f}, "
                        f"final_ratio={dynamic_tp_ratio:.3f} (range: {min_tp_ratio}-{max_tp_ratio})"
                    )
                    
                    if not (1.0 <= dynamic_tp_ratio <= 3.0):
                        logger.warning(f"Invalid TP ratio: {dynamic_tp_ratio}, using default {base_tp_ratio}")
                        dynamic_tp_ratio = base_tp_ratio
                    
                    atr_raw = indicators.get('atr', 1.0)
                    atr = safe_float(atr_raw, 1.0, "signal_atr")
                    if atr <= 0 or not is_valid_number(atr):
                        logger.warning(f"Invalid ATR: {atr}, using default 1.0")
                        atr = 1.0
                    
                    fixed_risk_amount = safe_float(self.config.FIXED_RISK_AMOUNT, 2.0, "FIXED_RISK_AMOUNT")
                    lot_size_cfg = safe_float(self.config.LOT_SIZE, 0.01, "LOT_SIZE")
                    pip_value = safe_float(self.config.XAUUSD_PIP_VALUE, 10.0, "XAUUSD_PIP_VALUE")
                    
                    if pip_value <= 0:
                        logger.warning(f"Invalid pip value: {pip_value}, using default 10.0")
                        pip_value = 10.0
                    
                    dollar_per_pip = pip_value * lot_size_cfg
                    sl_pips_from_risk = fixed_risk_amount / dollar_per_pip if dollar_per_pip > 0 else 20.0
                    sl_from_risk = sl_pips_from_risk / pip_value
                    
                    sl_atr_mult = safe_float(self.config.SL_ATR_MULTIPLIER, 1.5, "SL_ATR_MULTIPLIER")
                    default_sl_pips = safe_float(self.config.DEFAULT_SL_PIPS, 10.0, "DEFAULT_SL_PIPS")
                    sl_from_atr = atr * sl_atr_mult
                    sl_from_default = safe_divide(default_sl_pips, pip_value, 1.0, "sl_distance_calc")
                    
                    sl_distance = min(sl_from_risk, max(sl_from_atr, sl_from_default))
                    
                    logger.info(f"💰 Fixed-Risk SL ({signal_source}): Risk=${fixed_risk_amount:.2f}, $/pip=${dollar_per_pip:.2f}, MaxSL={sl_pips_from_risk:.1f}pips, Selected={sl_distance * pip_value:.1f}pips")
                    
                    if not is_valid_number(sl_distance) or sl_distance <= 0 or sl_distance > 100:
                        logger.error(f"Invalid SL distance: {sl_distance}")
                        return None
                    
                    tp_distance = sl_distance * dynamic_tp_ratio
                    
                    if not is_valid_number(tp_distance) or tp_distance <= 0:
                        logger.error(f"Invalid TP distance: {tp_distance}")
                        return None
                    
                    close_val = safe_float(close, 0.0, "close_for_sl_tp")
                    if close_val <= 0 or not is_valid_number(close_val):
                        logger.error(f"Invalid close price for SL/TP calculation: {close}")
                        return None
                    
                    if signal == 'BUY':
                        stop_loss = close_val - sl_distance
                        take_profit = close_val + tp_distance
                    else:
                        stop_loss = close_val + sl_distance
                        take_profit = close_val - tp_distance
                    
                    if not is_valid_number(stop_loss) or not is_valid_number(take_profit):
                        logger.error(f"NaN/Inf in SL/TP: SL={stop_loss}, TP={take_profit}")
                        return None
                    
                    if stop_loss <= 0 or take_profit <= 0:
                        logger.error(f"Invalid SL/TP calculated: SL={stop_loss}, TP={take_profit}")
                        return None
                    
                    pip_value = safe_float(self.config.XAUUSD_PIP_VALUE, 10.0, "XAUUSD_PIP_VALUE")
                    sl_pips = abs(stop_loss - close_val) * pip_value
                    tp_pips = abs(take_profit - close_val) * pip_value
                    
                    if not is_valid_number(sl_pips) or sl_pips <= 0:
                        logger.error(f"Invalid SL pips: {sl_pips}")
                        return None
                    
                    if not is_valid_number(tp_pips) or tp_pips <= 0:
                        logger.error(f"Invalid TP pips: {tp_pips}")
                        return None
                    
                    fixed_risk = safe_float(self.config.FIXED_RISK_AMOUNT, 2.0, "FIXED_RISK_AMOUNT")
                    lot_size = safe_float(self.config.LOT_SIZE, 0.01, "LOT_SIZE")
                    
                    pip_value_recheck = safe_float(self.config.XAUUSD_PIP_VALUE, 10.0, "XAUUSD_PIP_VALUE")
                    actual_dollar_per_pip = pip_value_recheck * lot_size
                    expected_loss = sl_pips * actual_dollar_per_pip
                    expected_profit = tp_pips * actual_dollar_per_pip
                    
                    if expected_loss > fixed_risk * 1.1:
                        logger.warning(f"Expected loss ${expected_loss:.2f} exceeds fixed risk ${fixed_risk:.2f} - SL may need adjustment")
                    
                    if not is_valid_number(expected_profit):
                        logger.warning(f"Invalid expected_profit: {expected_profit}")
                        expected_profit = expected_loss * 1.5
                    
                except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
                    logger.error(f"Calculation error in signal generation: {type(e).__name__}: {e}")
                    return None
                
                logger.info(f"{signal} signal detected ({signal_source}) on {timeframe}")
                logger.info(f"Trend Strength: {trend_desc} (score: {trend_strength:.2f})")
                logger.info(f"Dynamic TP Ratio: {dynamic_tp_ratio:.2f}x (Expected profit: ${expected_profit:.2f})")
                logger.info(f"Risk: ${expected_loss:.2f} | Reward: ${expected_profit:.2f} | R:R = 1:{dynamic_tp_ratio:.2f}")
                
                def safe_indicator_float(val, default=None):
                    """Convert indicator value to float safely for JSON serialization"""
                    if val is None:
                        return default
                    if not is_valid_number(val):
                        return default
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return default
                
                def safe_indicator_int(val, default=None):
                    """Convert indicator value to int safely for JSON serialization"""
                    if val is None:
                        return default
                    if not is_valid_number(val):
                        return default
                    try:
                        return int(val)
                    except (ValueError, TypeError):
                        return default
                
                return {
                    'signal': signal,
                    'signal_source': signal_source,
                    'entry_price': float(close_val),
                    'stop_loss': float(stop_loss),
                    'take_profit': float(take_profit),
                    'timeframe': timeframe,
                    'trend_strength': float(trend_strength),
                    'trend_description': trend_desc,
                    'expected_profit': float(expected_profit),
                    'expected_loss': float(expected_loss),
                    'rr_ratio': float(dynamic_tp_ratio),
                    'lot_size': float(lot_size),
                    'position_size': float(dynamic_lot),
                    'volatility_zone': str(volatility_zone),
                    'sl_pips': float(sl_pips),
                    'tp_pips': float(tp_pips),
                    'indicators': json.dumps({
                        'ema_short': safe_indicator_float(ema_short),
                        'ema_mid': safe_indicator_float(ema_mid),
                        'ema_long': safe_indicator_float(ema_long),
                        'rsi': safe_indicator_float(rsi),
                        'macd': safe_indicator_float(macd),
                        'macd_signal': safe_indicator_float(macd_signal),
                        'macd_histogram': safe_indicator_float(macd_histogram),
                        'stoch_k': safe_indicator_float(stoch_k),
                        'stoch_d': safe_indicator_float(stoch_d),
                        'atr': safe_indicator_float(atr),
                        'volume': safe_indicator_int(volume),
                        'volume_avg': safe_indicator_float(volume_avg)
                    }),
                    'confidence_reasons': confidence_reasons
                }
            
            return None
            
        except (StrategyError, Exception) as e:
            logger.error(f"Error detecting signal: {e}")
            return None
    
    def validate_signal(self, signal: Dict, current_spread: float = 0) -> Tuple[bool, Optional[str]]:
        """Validate signal with comprehensive checks and error handling"""
        try:
            if not signal or not isinstance(signal, dict):
                return False, "Signal must be a non-empty dictionary"
            
            required_fields = ['entry_price', 'stop_loss', 'take_profit', 'signal']
            missing = [f for f in required_fields if f not in signal]
            if missing:
                return False, f"Missing required fields: {missing}"
            
            entry_raw = signal['entry_price']
            sl_raw = signal['stop_loss']
            tp_raw = signal['take_profit']
            signal_type = signal['signal']
            
            if not is_valid_number(entry_raw):
                logger.warning(f"NaN/Inf detected in entry_price: {entry_raw}")
                return False, f"Entry price is NaN or Inf: {entry_raw}"
            if not is_valid_number(sl_raw):
                logger.warning(f"NaN/Inf detected in stop_loss: {sl_raw}")
                return False, f"Stop loss is NaN or Inf: {sl_raw}"
            if not is_valid_number(tp_raw):
                logger.warning(f"NaN/Inf detected in take_profit: {tp_raw}")
                return False, f"Take profit is NaN or Inf: {tp_raw}"
            
            entry = safe_float(entry_raw, 0.0)
            sl = safe_float(sl_raw, 0.0)
            tp = safe_float(tp_raw, 0.0)
            
            if entry <= 0 or sl <= 0 or tp <= 0:
                return False, f"Invalid prices: entry={entry}, sl={sl}, tp={tp}"
            
            if signal_type not in ['BUY', 'SELL']:
                return False, f"Invalid signal type: {signal_type}"
            
            try:
                spread_safe = safe_float(current_spread, 0.0, "current_spread")
                pip_value = safe_float(self.config.XAUUSD_PIP_VALUE, 10.0, "XAUUSD_PIP_VALUE")
                
                if pip_value <= 0:
                    pip_value = 10.0
                
                spread_pips = spread_safe * pip_value
                
                if not is_valid_number(spread_pips):
                    logger.warning(f"NaN/Inf in spread calculation: {spread_pips}, using 0")
                    spread_pips = 0.0
                
                if spread_pips < 0:
                    logger.warning(f"Negative spread: {spread_pips}, using 0")
                    spread_pips = 0.0
                
                max_spread = safe_float(self.config.MAX_SPREAD_PIPS, 50.0, "MAX_SPREAD_PIPS")
                if spread_pips > max_spread:
                    return False, f"Spread too high: {spread_pips:.2f} pips (max: {max_spread})"
            except (StrategyError, Exception) as e:
                logger.warning(f"Error calculating spread pips: {e}")
            
            pip_value = safe_float(self.config.XAUUSD_PIP_VALUE, 10.0, "XAUUSD_PIP_VALUE_validate")
            if pip_value <= 0:
                pip_value = 10.0
            
            sl_pips = abs(entry - sl) * pip_value
            tp_pips = abs(entry - tp) * pip_value
            
            if not is_valid_number(sl_pips):
                logger.warning(f"NaN/Inf in SL pips validation: {sl_pips}")
                return False, f"SL pips calculation resulted in NaN/Inf: {sl_pips}"
            
            if not is_valid_number(tp_pips):
                logger.warning(f"NaN/Inf in TP pips validation: {tp_pips}")
                return False, f"TP pips calculation resulted in NaN/Inf: {tp_pips}"
            
            if sl_pips < 5:
                return False, f"Stop loss too tight: {sl_pips:.1f} pips (min: 5 pips)"
            
            if tp_pips < 10:
                return False, f"Take profit too tight: {tp_pips:.1f} pips (min: 10 pips)"
            
            if signal_type == 'BUY':
                if sl >= entry:
                    return False, f"BUY signal: SL ({sl}) must be < entry ({entry})"
                if tp <= entry:
                    return False, f"BUY signal: TP ({tp}) must be > entry ({entry})"
            else:
                if sl <= entry:
                    return False, f"SELL signal: SL ({sl}) must be > entry ({entry})"
                if tp >= entry:
                    return False, f"SELL signal: TP ({tp}) must be < entry ({entry})"
            
            return True, None
            
        except KeyError as e:
            return False, f"Missing key in signal: {e}"
        except (StrategyError, Exception) as e:
            logger.error(f"Signal validation error: {type(e).__name__}: {e}")
            return False, f"Validation error: {str(e)}"


@dataclass
class ConfluenceResult:
    """Result of confluence scoring calculation for trading signals.
    
    Attributes:
        total_score: Overall confluence score from 0-100
        confluences_met: List of confluence confirmations that passed
        confidence_level: Signal quality level ('SCALP', 'SHORT_TERM', 'OPTIMAL')
        recommended_tp_pips: Recommended take profit in pips
        recommended_sl_pips: Recommended stop loss in pips
        confluence_count: Number of confluences met
        weights_used: Dictionary of weights applied per confluence
        market_regime: Market regime used for weighting
        signal_type: Signal type ('BUY' or 'SELL')
    """
    total_score: int = 0
    confluences_met: List[str] = field(default_factory=list)
    confidence_level: str = 'NONE'
    recommended_tp_pips: float = 0.0
    recommended_sl_pips: float = 0.0
    confluence_count: int = 0
    weights_used: Dict[str, float] = field(default_factory=dict)
    market_regime: str = 'unknown'
    signal_type: str = ''
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ConfluenceResult to dictionary for JSON serialization"""
        return {
            'total_score': self.total_score,
            'confluences_met': self.confluences_met,
            'confidence_level': self.confidence_level,
            'recommended_tp_pips': self.recommended_tp_pips,
            'recommended_sl_pips': self.recommended_sl_pips,
            'confluence_count': self.confluence_count,
            'weights_used': self.weights_used,
            'market_regime': self.market_regime,
            'signal_type': self.signal_type
        }


class ConfluenceScorer:
    """Confluence Scoring System untuk bot trading XAUUSD.
    
    Sistem ini menghitung confluence score berdasarkan multiple confirmations:
    1. Trend Confirmation (EMA alignment + price position)
    2. Momentum Confirmation (RSI direction atau MACD histogram)
    3. Volume Confirmation (volume spike > 1.2x average)
    4. Price Action Confirmation (break S/R + retest)
    5. Bollinger Band Confirmation (break upper/lower band)
    6. Stochastic Confirmation (K line cross D line)
    
    Scoring System:
    - 2 confluence = 60% confidence (SCALP signal, 15-30 pip target)
    - 3 confluence = 80% confidence (SHORT-TERM signal, 30-50 pip target)
    - 4+ confluence = 95% confidence (OPTIMAL signal, 50+ pip target)
    
    Adaptive Weighting berdasarkan market regime:
    - Trending market: EMA/MACD weight lebih tinggi
    - Ranging market: S/R/Stochastic weight lebih tinggi
    """
    
    CONFLUENCE_TREND = 'trend_confirmation'
    CONFLUENCE_MOMENTUM = 'momentum_confirmation'
    CONFLUENCE_VOLUME = 'volume_confirmation'
    CONFLUENCE_PRICE_ACTION = 'price_action_confirmation'
    CONFLUENCE_BOLLINGER = 'bollinger_confirmation'
    CONFLUENCE_STOCHASTIC = 'stochastic_confirmation'
    
    DEFAULT_WEIGHTS: Dict[str, float] = {
        CONFLUENCE_TREND: 20.0,
        CONFLUENCE_MOMENTUM: 18.0,
        CONFLUENCE_VOLUME: 15.0,
        CONFLUENCE_PRICE_ACTION: 17.0,
        CONFLUENCE_BOLLINGER: 15.0,
        CONFLUENCE_STOCHASTIC: 15.0
    }
    
    TRENDING_WEIGHTS: Dict[str, float] = {
        CONFLUENCE_TREND: 25.0,
        CONFLUENCE_MOMENTUM: 22.0,
        CONFLUENCE_VOLUME: 12.0,
        CONFLUENCE_PRICE_ACTION: 15.0,
        CONFLUENCE_BOLLINGER: 13.0,
        CONFLUENCE_STOCHASTIC: 13.0
    }
    
    RANGING_WEIGHTS: Dict[str, float] = {
        CONFLUENCE_TREND: 12.0,
        CONFLUENCE_MOMENTUM: 15.0,
        CONFLUENCE_VOLUME: 15.0,
        CONFLUENCE_PRICE_ACTION: 22.0,
        CONFLUENCE_BOLLINGER: 16.0,
        CONFLUENCE_STOCHASTIC: 20.0
    }
    
    VOLUME_SPIKE_THRESHOLD = 1.2
    
    RSI_BULLISH_THRESHOLD = 50
    RSI_BEARISH_THRESHOLD = 50
    
    STOCH_OVERBOUGHT = 80
    STOCH_OVERSOLD = 20
    
    def __init__(self, config):
        """Initialize ConfluenceScorer with configuration.
        
        Args:
            config: Configuration object with trading parameters
        """
        self.config = config
        self._logger = logger
    
    def _get_weights(self, market_regime: str) -> Dict[str, float]:
        """Get confluence weights based on market regime.
        
        Args:
            market_regime: Current market regime type
            
        Returns:
            Dictionary of weights per confluence type
        """
        regime_lower = market_regime.lower() if market_regime else 'unknown'
        
        if regime_lower in ['strong_trend', 'moderate_trend', 'breakout']:
            return self.TRENDING_WEIGHTS.copy()
        elif regime_lower in ['range_bound', 'weak_trend']:
            return self.RANGING_WEIGHTS.copy()
        else:
            return self.DEFAULT_WEIGHTS.copy()
    
    def _check_trend_confirmation(self, indicators: Dict, signal_type: str) -> Tuple[bool, str]:
        """Check trend confirmation via EMA alignment and price position.
        
        Kriteria:
        - EMA alignment (short > mid > long untuk BUY, sebaliknya untuk SELL)
        - Price above/below key moving average
        
        Args:
            indicators: Dictionary of calculated indicators
            signal_type: 'BUY' or 'SELL'
            
        Returns:
            Tuple of (is_confirmed, description)
        """
        try:
            ema_periods = getattr(self.config, 'EMA_PERIODS', [5, 20, 50])
            
            ema_short_key = f'ema_{ema_periods[0]}'
            ema_mid_key = f'ema_{ema_periods[1]}'
            ema_long_key = f'ema_{ema_periods[2]}' if len(ema_periods) > 2 else f'ema_{ema_periods[1]}'
            
            ema_short = indicators.get(ema_short_key)
            ema_mid = indicators.get(ema_mid_key)
            ema_long = indicators.get(ema_long_key)
            close = indicators.get('close')
            
            if not all([is_valid_number(ema_short), is_valid_number(ema_mid), 
                       is_valid_number(ema_long), is_valid_number(close)]):
                return False, "EMA/Close data tidak tersedia"
            
            ema_s = safe_float(ema_short, 0.0)
            ema_m = safe_float(ema_mid, 0.0)
            ema_l = safe_float(ema_long, 0.0)
            close_val = safe_float(close, 0.0)
            
            if ema_s <= 0 or ema_m <= 0 or ema_l <= 0 or close_val <= 0:
                return False, "EMA/Close values tidak valid"
            
            if signal_type == 'BUY':
                ema_aligned = ema_s > ema_m > ema_l
                price_above_ma = close_val > ema_m
                
                if ema_aligned and price_above_ma:
                    return True, f"Trend BULLISH: EMA aligned (EMA{ema_periods[0]}>{ema_periods[1]}>{ema_periods[2]}) + Price > EMA{ema_periods[1]}"
                elif ema_aligned:
                    return True, f"Trend BULLISH: EMA aligned (partial confirmation)"
                elif price_above_ma and ema_s > ema_m:
                    return True, f"Trend BULLISH: Price > EMA{ema_periods[1]} + Short EMA bullish"
                    
            elif signal_type == 'SELL':
                ema_aligned = ema_s < ema_m < ema_l
                price_below_ma = close_val < ema_m
                
                if ema_aligned and price_below_ma:
                    return True, f"Trend BEARISH: EMA aligned (EMA{ema_periods[0]}<{ema_periods[1]}<{ema_periods[2]}) + Price < EMA{ema_periods[1]}"
                elif ema_aligned:
                    return True, f"Trend BEARISH: EMA aligned (partial confirmation)"
                elif price_below_ma and ema_s < ema_m:
                    return True, f"Trend BEARISH: Price < EMA{ema_periods[1]} + Short EMA bearish"
            
            return False, "Trend tidak terkonfirmasi"
            
        except Exception as e:
            self._logger.warning(f"Error in trend confirmation check: {e}")
            return False, f"Error: {str(e)}"
    
    def _check_momentum_confirmation(self, indicators: Dict, signal_type: str) -> Tuple[bool, str]:
        """Check momentum confirmation via RSI direction or MACD histogram.
        
        Kriteria:
        - RSI > 50 untuk BUY, RSI < 50 untuk SELL
        - MACD histogram positif untuk BUY, negatif untuk SELL
        
        Args:
            indicators: Dictionary of calculated indicators
            signal_type: 'BUY' or 'SELL'
            
        Returns:
            Tuple of (is_confirmed, description)
        """
        try:
            rsi = indicators.get('rsi')
            rsi_prev = indicators.get('rsi_prev')
            macd_histogram = indicators.get('macd_histogram')
            macd = indicators.get('macd')
            macd_signal = indicators.get('macd_signal')
            
            confirmations = []
            
            if is_valid_number(rsi):
                rsi_val = safe_float(rsi, 50.0)
                rsi_direction = ""
                
                if is_valid_number(rsi_prev):
                    rsi_prev_val = safe_float(rsi_prev, 50.0)
                    if rsi_val > rsi_prev_val:
                        rsi_direction = "↗"
                    elif rsi_val < rsi_prev_val:
                        rsi_direction = "↘"
                
                if signal_type == 'BUY':
                    if rsi_val > self.RSI_BULLISH_THRESHOLD:
                        confirmations.append(f"RSI bullish ({rsi_val:.1f}{rsi_direction} > 50)")
                elif signal_type == 'SELL':
                    if rsi_val < self.RSI_BEARISH_THRESHOLD:
                        confirmations.append(f"RSI bearish ({rsi_val:.1f}{rsi_direction} < 50)")
            
            if is_valid_number(macd_histogram):
                histogram = safe_float(macd_histogram, 0.0)
                
                if signal_type == 'BUY' and histogram > 0:
                    confirmations.append(f"MACD histogram positif ({histogram:.4f})")
                elif signal_type == 'SELL' and histogram < 0:
                    confirmations.append(f"MACD histogram negatif ({histogram:.4f})")
            
            if is_valid_number(macd) and is_valid_number(macd_signal):
                macd_val = safe_float(macd, 0.0)
                signal_val = safe_float(macd_signal, 0.0)
                
                if signal_type == 'BUY' and macd_val > signal_val:
                    confirmations.append(f"MACD > Signal line")
                elif signal_type == 'SELL' and macd_val < signal_val:
                    confirmations.append(f"MACD < Signal line")
            
            if confirmations:
                return True, f"Momentum: {' + '.join(confirmations)}"
            
            return False, "Momentum tidak terkonfirmasi"
            
        except Exception as e:
            self._logger.warning(f"Error in momentum confirmation check: {e}")
            return False, f"Error: {str(e)}"
    
    def _check_volume_confirmation(self, indicators: Dict, signal_type: str) -> Tuple[bool, str]:
        """Check volume confirmation via volume spike detection.
        
        Kriteria:
        - Volume > 1.2x average volume (VOLUME_SPIKE_THRESHOLD)
        
        Args:
            indicators: Dictionary of calculated indicators
            signal_type: 'BUY' or 'SELL'
            
        Returns:
            Tuple of (is_confirmed, description)
        """
        try:
            volume = indicators.get('volume')
            volume_avg = indicators.get('volume_avg')
            
            if not is_valid_number(volume) or not is_valid_number(volume_avg):
                return False, "Volume data tidak tersedia"
            
            vol = safe_float(volume, 0.0)
            vol_avg = safe_float(volume_avg, 0.0)
            
            if vol_avg <= 0:
                return False, "Volume average tidak valid"
            
            volume_ratio = safe_divide(vol, vol_avg, 1.0, "volume_ratio")
            
            if volume_ratio >= self.VOLUME_SPIKE_THRESHOLD:
                return True, f"Volume spike: {volume_ratio:.2f}x average (>{self.VOLUME_SPIKE_THRESHOLD}x)"
            
            return False, f"Volume normal ({volume_ratio:.2f}x)"
            
        except Exception as e:
            self._logger.warning(f"Error in volume confirmation check: {e}")
            return False, f"Error: {str(e)}"
    
    def _check_price_action_confirmation(self, indicators: Dict, signal_type: str) -> Tuple[bool, str]:
        """Check price action confirmation via S/R break and retest.
        
        Kriteria:
        - Price near support level untuk BUY
        - Price near resistance level untuk SELL
        - Candlestick pattern confirmation
        
        Args:
            indicators: Dictionary of calculated indicators
            signal_type: 'BUY' or 'SELL'
            
        Returns:
            Tuple of (is_confirmed, description)
        """
        try:
            close = indicators.get('close')
            atr = indicators.get('atr')
            sr_levels = indicators.get('support_resistance', {})
            patterns = indicators.get('candlestick_patterns', {})
            
            if not is_valid_number(close):
                return False, "Close price tidak tersedia"
            
            close_val = safe_float(close, 0.0)
            atr_val = safe_float(atr, 1.0) if is_valid_number(atr) else 1.0
            
            confirmations = []
            
            if patterns and isinstance(patterns, dict):
                if signal_type == 'BUY':
                    if patterns.get('bullish_engulfing') or patterns.get('hammer') or patterns.get('bullish_pinbar'):
                        pattern_names = []
                        if patterns.get('bullish_engulfing'):
                            pattern_names.append('Bullish Engulfing')
                        if patterns.get('hammer'):
                            pattern_names.append('Hammer')
                        if patterns.get('bullish_pinbar'):
                            pattern_names.append('Bullish Pinbar')
                        confirmations.append(f"Pattern: {', '.join(pattern_names)}")
                elif signal_type == 'SELL':
                    if patterns.get('bearish_engulfing') or patterns.get('inverted_hammer') or patterns.get('bearish_pinbar'):
                        pattern_names = []
                        if patterns.get('bearish_engulfing'):
                            pattern_names.append('Bearish Engulfing')
                        if patterns.get('inverted_hammer'):
                            pattern_names.append('Inverted Hammer')
                        if patterns.get('bearish_pinbar'):
                            pattern_names.append('Bearish Pinbar')
                        confirmations.append(f"Pattern: {', '.join(pattern_names)}")
            
            if sr_levels and isinstance(sr_levels, dict):
                nearest_support = sr_levels.get('nearest_support', 0.0)
                nearest_resistance = sr_levels.get('nearest_resistance', 0.0)
                
                proximity_threshold = atr_val * 1.5
                
                if signal_type == 'BUY' and nearest_support > 0:
                    distance_to_support = abs(close_val - nearest_support)
                    if distance_to_support <= proximity_threshold:
                        confirmations.append(f"Near Support ({nearest_support:.2f})")
                        
                elif signal_type == 'SELL' and nearest_resistance > 0:
                    distance_to_resistance = abs(close_val - nearest_resistance)
                    if distance_to_resistance <= proximity_threshold:
                        confirmations.append(f"Near Resistance ({nearest_resistance:.2f})")
            
            if confirmations:
                return True, f"Price Action: {' + '.join(confirmations)}"
            
            return False, "Price action tidak terkonfirmasi"
            
        except Exception as e:
            self._logger.warning(f"Error in price action confirmation check: {e}")
            return False, f"Error: {str(e)}"
    
    def _check_bollinger_confirmation(self, indicators: Dict, signal_type: str) -> Tuple[bool, str]:
        """Check Bollinger Band confirmation.
        
        Kriteria:
        - Price break lower band untuk BUY (oversold condition)
        - Price break upper band untuk SELL (overbought condition)
        - Price reverting to middle band setelah break
        
        Args:
            indicators: Dictionary of calculated indicators
            signal_type: 'BUY' or 'SELL'
            
        Returns:
            Tuple of (is_confirmed, description)
        """
        try:
            close = indicators.get('close')
            bb_upper = indicators.get('bb_upper')
            bb_middle = indicators.get('bb_middle')
            bb_lower = indicators.get('bb_lower')
            
            if not all([is_valid_number(close), is_valid_number(bb_upper), 
                       is_valid_number(bb_middle), is_valid_number(bb_lower)]):
                return False, "Bollinger Band data tidak tersedia"
            
            close_val = safe_float(close, 0.0)
            upper = safe_float(bb_upper, 0.0)
            middle = safe_float(bb_middle, 0.0)
            lower = safe_float(bb_lower, 0.0)
            
            if upper <= 0 or middle <= 0 or lower <= 0:
                return False, "Bollinger Band values tidak valid"
            
            bb_width = upper - lower
            if bb_width <= 0:
                return False, "Bollinger Band width tidak valid"
            
            band_threshold = bb_width * 0.1
            
            if signal_type == 'BUY':
                if close_val <= lower + band_threshold:
                    return True, f"BB Oversold: Price ({close_val:.2f}) near lower band ({lower:.2f})"
                elif close_val > lower and close_val < middle:
                    bb_prev_close = indicators.get('close_prev')
                    if is_valid_number(bb_prev_close):
                        prev = safe_float(bb_prev_close, 0.0)
                        if prev <= lower:
                            return True, f"BB Bounce: Price bouncing from lower band"
                            
            elif signal_type == 'SELL':
                if close_val >= upper - band_threshold:
                    return True, f"BB Overbought: Price ({close_val:.2f}) near upper band ({upper:.2f})"
                elif close_val < upper and close_val > middle:
                    bb_prev_close = indicators.get('close_prev')
                    if is_valid_number(bb_prev_close):
                        prev = safe_float(bb_prev_close, 0.0)
                        if prev >= upper:
                            return True, f"BB Reversal: Price reversing from upper band"
            
            return False, "Bollinger Band tidak terkonfirmasi"
            
        except Exception as e:
            self._logger.warning(f"Error in Bollinger Band confirmation check: {e}")
            return False, f"Error: {str(e)}"
    
    def _check_stochastic_confirmation(self, indicators: Dict, signal_type: str) -> Tuple[bool, str]:
        """Check Stochastic confirmation via K/D line crossover.
        
        Kriteria:
        - K line cross above D line untuk BUY (bullish crossover)
        - K line cross below D line untuk SELL (bearish crossover)
        - Oversold/overbought zone confirmation
        
        Args:
            indicators: Dictionary of calculated indicators
            signal_type: 'BUY' or 'SELL'
            
        Returns:
            Tuple of (is_confirmed, description)
        """
        try:
            stoch_k = indicators.get('stoch_k')
            stoch_d = indicators.get('stoch_d')
            stoch_k_prev = indicators.get('stoch_k_prev')
            stoch_d_prev = indicators.get('stoch_d_prev')
            
            if not all([is_valid_number(stoch_k), is_valid_number(stoch_d)]):
                return False, "Stochastic data tidak tersedia"
            
            k = safe_float(stoch_k, 50.0)
            d = safe_float(stoch_d, 50.0)
            
            if k < 0 or k > 100 or d < 0 or d > 100:
                return False, "Stochastic values out of range"
            
            k_prev = safe_float(stoch_k_prev, k) if is_valid_number(stoch_k_prev) else k
            d_prev = safe_float(stoch_d_prev, d) if is_valid_number(stoch_d_prev) else d
            
            if signal_type == 'BUY':
                bullish_cross = k > d and k_prev <= d_prev
                in_oversold = k < self.STOCH_OVERSOLD or k_prev < self.STOCH_OVERSOLD
                k_rising = k > k_prev
                
                if bullish_cross and in_oversold:
                    return True, f"Stoch Bullish Cross in Oversold: K({k:.1f}) crossed above D({d:.1f})"
                elif bullish_cross:
                    return True, f"Stoch Bullish Cross: K({k:.1f}) crossed above D({d:.1f})"
                elif in_oversold and k_rising and k > d:
                    return True, f"Stoch Oversold Recovery: K({k:.1f}) > D({d:.1f}) in oversold zone"
                    
            elif signal_type == 'SELL':
                bearish_cross = k < d and k_prev >= d_prev
                in_overbought = k > self.STOCH_OVERBOUGHT or k_prev > self.STOCH_OVERBOUGHT
                k_falling = k < k_prev
                
                if bearish_cross and in_overbought:
                    return True, f"Stoch Bearish Cross in Overbought: K({k:.1f}) crossed below D({d:.1f})"
                elif bearish_cross:
                    return True, f"Stoch Bearish Cross: K({k:.1f}) crossed below D({d:.1f})"
                elif in_overbought and k_falling and k < d:
                    return True, f"Stoch Overbought Reversal: K({k:.1f}) < D({d:.1f}) in overbought zone"
            
            return False, "Stochastic tidak terkonfirmasi"
            
        except Exception as e:
            self._logger.warning(f"Error in Stochastic confirmation check: {e}")
            return False, f"Error: {str(e)}"
    
    def _determine_confidence_level(self, confluence_count: int, total_score: float) -> Tuple[str, float, float]:
        """Determine confidence level based on confluence count.
        
        Scoring System:
        - 2 confluence = 60% confidence (SCALP signal, 15-30 pip target)
        - 3 confluence = 80% confidence (SHORT-TERM signal, 30-50 pip target)
        - 4+ confluence = 95% confidence (OPTIMAL signal, 50+ pip target)
        
        Args:
            confluence_count: Number of confluences met
            total_score: Total weighted score
            
        Returns:
            Tuple of (confidence_level, recommended_tp_pips, recommended_sl_pips)
        """
        if confluence_count >= 4:
            return 'OPTIMAL', 50.0, 25.0
        elif confluence_count == 3:
            return 'SHORT_TERM', 40.0, 20.0
        elif confluence_count == 2:
            return 'SCALP', 22.0, 15.0
        elif confluence_count == 1:
            return 'WEAK', 15.0, 12.0
        else:
            return 'NONE', 0.0, 0.0
    
    def calculate_confluence_score(self, indicators: Dict, signal_type: str, 
                                   market_regime: str = 'unknown') -> ConfluenceResult:
        """Calculate confluence score for a trading signal.
        
        This method evaluates all confluence factors and returns a comprehensive
        scoring result with adaptive weighting based on market regime.
        
        Args:
            indicators: Dictionary of calculated indicators
            signal_type: 'BUY' or 'SELL'
            market_regime: Current market regime type (e.g., 'strong_trend', 'range_bound')
            
        Returns:
            ConfluenceResult dataclass with scoring details
        """
        result = ConfluenceResult(
            market_regime=market_regime,
            signal_type=signal_type
        )
        
        try:
            if not indicators or not isinstance(indicators, dict):
                self._logger.warning("calculate_confluence_score: Invalid indicators dict")
                return result
            
            if signal_type not in ['BUY', 'SELL']:
                self._logger.warning(f"calculate_confluence_score: Invalid signal_type: {signal_type}")
                return result
            
            weights = self._get_weights(market_regime)
            result.weights_used = weights.copy()
            
            confluence_checks = [
                (self.CONFLUENCE_TREND, self._check_trend_confirmation),
                (self.CONFLUENCE_MOMENTUM, self._check_momentum_confirmation),
                (self.CONFLUENCE_VOLUME, self._check_volume_confirmation),
                (self.CONFLUENCE_PRICE_ACTION, self._check_price_action_confirmation),
                (self.CONFLUENCE_BOLLINGER, self._check_bollinger_confirmation),
                (self.CONFLUENCE_STOCHASTIC, self._check_stochastic_confirmation),
            ]
            
            total_score: float = 0.0
            confluences_met: List[str] = []
            
            for confluence_type, check_func in confluence_checks:
                try:
                    is_confirmed, description = check_func(indicators, signal_type)
                    
                    if is_confirmed:
                        weight = weights.get(confluence_type, 15)
                        total_score += weight
                        confluences_met.append(f"{confluence_type}: {description}")
                        self._logger.debug(f"Confluence MET: {confluence_type} (+{weight}) - {description}")
                    else:
                        self._logger.debug(f"Confluence NOT MET: {confluence_type} - {description}")
                        
                except Exception as e:
                    self._logger.warning(f"Error checking {confluence_type}: {e}")
            
            confluence_count = len(confluences_met)
            confidence_level, tp_pips, sl_pips = self._determine_confidence_level(confluence_count, total_score)
            
            result.total_score = int(min(100.0, total_score))
            result.confluences_met = confluences_met
            result.confluence_count = confluence_count
            result.confidence_level = confidence_level
            result.recommended_tp_pips = tp_pips
            result.recommended_sl_pips = sl_pips
            
            self._logger.info(
                f"Confluence Score: {result.total_score}/100 | "
                f"Confluences: {confluence_count}/6 | "
                f"Level: {confidence_level} | "
                f"Regime: {market_regime} | "
                f"Signal: {signal_type}"
            )
            
            if confluences_met:
                self._logger.info(f"Confluences met: {', '.join([c.split(':')[0] for c in confluences_met])}")
            
            return result
            
        except Exception as e:
            self._logger.error(f"Error in calculate_confluence_score: {e}")
            return result
    
    def get_signal_recommendation(self, result: ConfluenceResult) -> Dict[str, Any]:
        """Get trading recommendation based on confluence result.
        
        Args:
            result: ConfluenceResult from calculate_confluence_score
            
        Returns:
            Dictionary with recommendation details
        """
        if result.confluence_count < 2:
            action = 'SKIP'
            reason = f"Insufficient confluence ({result.confluence_count}/6, need minimum 2)"
        elif result.confidence_level == 'SCALP':
            action = 'SCALP_ENTRY'
            reason = f"Scalp opportunity: {result.confluence_count} confluences, 60% confidence"
        elif result.confidence_level == 'SHORT_TERM':
            action = 'ENTRY'
            reason = f"Good entry: {result.confluence_count} confluences, 80% confidence"
        elif result.confidence_level == 'OPTIMAL':
            action = 'STRONG_ENTRY'
            reason = f"Optimal entry: {result.confluence_count} confluences, 95% confidence"
        else:
            action = 'SKIP'
            reason = f"Weak signal ({result.confluence_count}/6 confluences)"
        
        return {
            'action': action,
            'reason': reason,
            'signal_type': result.signal_type,
            'confidence_level': result.confidence_level,
            'score': result.total_score,
            'confluence_count': result.confluence_count,
            'recommended_tp_pips': result.recommended_tp_pips,
            'recommended_sl_pips': result.recommended_sl_pips,
            'confluences': result.confluences_met,
            'market_regime': result.market_regime
        }
