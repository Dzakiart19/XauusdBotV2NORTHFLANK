import os
import re
from typing import List, Any, Optional
from dotenv import load_dotenv

load_dotenv()


class ConfigError(Exception):
    """
    Raised when configuration validation fails.
    
    Attributes:
        errors: List of validation error messages
        warnings: List of validation warning messages
    """
    
    def __init__(self, message: str, errors: Optional[List[str]] = None, warnings: Optional[List[str]] = None):
        self.errors = errors if errors is not None else []
        self.warnings = warnings if warnings is not None else []
        self.message = message
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format error message with all details."""
        parts = [self.message]
        
        if self.errors:
            parts.append("\nErrors:")
            for i, err in enumerate(self.errors, 1):
                parts.append(f"  {i}. {err}")
        
        if self.warnings:
            parts.append("\nWarnings:")
            for i, warn in enumerate(self.warnings, 1):
                parts.append(f"  {i}. {warn}")
        
        return "\n".join(parts)
    
    def get_error_count(self) -> int:
        """Return total number of errors."""
        return len(self.errors)
    
    def get_warning_count(self) -> int:
        """Return total number of warnings."""
        return len(self.warnings)


ConfigValidationError = ConfigError


def _validate_type(value: Any, expected_type: type, field_name: str) -> Optional[str]:
    """
    Validate that a value is of the expected type.
    
    Returns:
        Error message if validation fails, None otherwise
    """
    if not isinstance(value, expected_type):
        return f"{field_name} must be {expected_type.__name__}, got {type(value).__name__}"
    return None


def _validate_range(value: float, min_val: float, max_val: float, 
                    field_name: str, inclusive: bool = True) -> Optional[str]:
    """
    Validate that a numeric value is within range.
    
    Returns:
        Error message if validation fails, None otherwise
    """
    if inclusive:
        if value < min_val or value > max_val:
            return f"{field_name} must be between {min_val} and {max_val}, got {value}"
    else:
        if value <= min_val or value >= max_val:
            return f"{field_name} must be between {min_val} and {max_val} (exclusive), got {value}"
    return None


def _validate_positive(value: float, field_name: str, allow_zero: bool = False) -> Optional[str]:
    """
    Validate that a value is positive.
    
    Returns:
        Error message if validation fails, None otherwise
    """
    if allow_zero:
        if value < 0:
            return f"{field_name} must be non-negative, got {value}"
    else:
        if value <= 0:
            return f"{field_name} must be positive, got {value}"
    return None


def _validate_non_empty_string(value: str, field_name: str) -> Optional[str]:
    """
    Validate that a string is non-empty.
    
    Returns:
        Error message if validation fails, None otherwise
    """
    if not value or not isinstance(value, str) or not value.strip():
        return f"{field_name} is required but not set or empty"
    return None


def _validate_list_not_empty(value: list, field_name: str) -> Optional[str]:
    """
    Validate that a list is non-empty.
    
    Returns:
        Error message if validation fails, None otherwise
    """
    if not value or not isinstance(value, list) or len(value) == 0:
        return f"{field_name} must contain at least one item"
    return None


def _validate_list_items_type(value: list, item_type: type, field_name: str) -> Optional[str]:
    """
    Validate that all items in a list are of expected type.
    
    Returns:
        Error message if validation fails, None otherwise
    """
    if not isinstance(value, list):
        return f"{field_name} must be a list"
    
    for i, item in enumerate(value):
        if not isinstance(item, item_type):
            return f"{field_name}[{i}] must be {item_type.__name__}, got {type(item).__name__}"
    return None


def _validate_url_format(value: str, field_name: str) -> Optional[str]:
    """
    Validate URL format.
    
    Returns:
        Error message if validation fails, None otherwise
    """
    if not value:
        return None
    
    value = value.strip()
    if not (value.startswith('http://') or value.startswith('https://')):
        return f"{field_name} must start with http:// or https://, got: {value[:50]}..."
    return None


def _get_float_env(key: str, default: str) -> float:
    """Safely get float environment variable with validation"""
    value = os.getenv(key, default)
    
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        value = value.strip()
        if not value:
            value = default
    
    try:
        return float(value)
    except (ValueError, TypeError):
        return float(default)


def _get_int_env(key: str, default: str) -> int:
    """Safely get integer environment variable with validation"""
    value = os.getenv(key, default)
    
    if isinstance(value, int):
        return value
    
    if isinstance(value, str):
        value = value.strip()
        if not value:
            value = default
    
    try:
        return int(value)
    except (ValueError, TypeError):
        return int(default)


def _parse_user_ids(env_value: str) -> list:
    """Parse comma-separated user IDs, returning empty list on error"""
    if not env_value:
        return []
    try:
        result = [int(uid.strip()) for uid in env_value.split(',') if uid.strip()]
        for uid in result:
            if uid <= 0:
                return []
        return result
    except (ValueError, AttributeError):
        return []


def _parse_int_list(env_value: str, default_list: list) -> list:
    """Parse comma-separated integers, returning default on error"""
    if not env_value:
        return default_list
    try:
        result = [int(p.strip()) for p in env_value.split(',') if p.strip()]
        if not result:
            return default_list
        for val in result:
            if val <= 0:
                return default_list
        return result
    except (ValueError, AttributeError):
        return default_list


class Config:
    # Critical secrets - di-set sebagai class attributes
    # Akan di-refresh oleh _refresh_secrets() untuk Koyeb compatibility
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    AUTHORIZED_USER_IDS = _parse_user_ids(os.getenv('AUTHORIZED_USER_IDS', ''))
    ID_USER_PUBLIC = _parse_user_ids(os.getenv('ID_USER_PUBLIC', ''))
    
    @classmethod
    def _refresh_secrets(cls):
        """
        Refresh secrets dari environment variables.
        Dipanggil saat bot startup untuk memastikan secrets terbaca dengan benar.
        Penting untuk deployment Koyeb dimana secrets mungkin di-inject setelah module import.
        """
        # Re-read dari environment
        new_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        new_users = _parse_user_ids(os.getenv('AUTHORIZED_USER_IDS', ''))
        new_public = _parse_user_ids(os.getenv('ID_USER_PUBLIC', ''))
        
        # Update jika ada nilai baru
        if new_token:
            cls.TELEGRAM_BOT_TOKEN = new_token
        if new_users:
            cls.AUTHORIZED_USER_IDS = new_users
        if new_public:
            cls.ID_USER_PUBLIC = new_public
        
        return {
            'token_set': bool(cls.TELEGRAM_BOT_TOKEN),
            'users_count': len(cls.AUTHORIZED_USER_IDS),
            'public_count': len(cls.ID_USER_PUBLIC)
        }
    
    # Koyeb deployment detection - auto-enable webhook mode
    KOYEB_PUBLIC_DOMAIN = os.getenv('KOYEB_PUBLIC_DOMAIN', '')
    KOYEB_REGION = os.getenv('KOYEB_REGION', '')
    KOYEB_SERVICE_NAME = os.getenv('KOYEB_SERVICE_NAME', '')
    IS_KOYEB = bool(KOYEB_PUBLIC_DOMAIN or KOYEB_REGION or KOYEB_SERVICE_NAME or os.getenv('KOYEB_APP_NAME', ''))
    
    # Auto-enable webhook mode for Koyeb deployment
    _telegram_webhook_mode_env = os.getenv('TELEGRAM_WEBHOOK_MODE', '')
    if _telegram_webhook_mode_env:
        TELEGRAM_WEBHOOK_MODE = _telegram_webhook_mode_env.lower() == 'true'
    else:
        # Auto-detect: enable webhook mode if running on Koyeb
        TELEGRAM_WEBHOOK_MODE = IS_KOYEB
    
    # Auto-generate webhook URL for Koyeb if not explicitly set
    _webhook_url_env = os.getenv('WEBHOOK_URL', '')
    if _webhook_url_env:
        WEBHOOK_URL = _webhook_url_env
    elif KOYEB_PUBLIC_DOMAIN:
        WEBHOOK_URL = f"https://{KOYEB_PUBLIC_DOMAIN}/webhook"
    else:
        WEBHOOK_URL = ''
    FREE_TIER_MODE = os.getenv('FREE_TIER_MODE', 'true').lower() == 'true'
    TICK_LOG_SAMPLE_RATE = _get_int_env('TICK_LOG_SAMPLE_RATE', '30')
    EMA_PERIODS = _parse_int_list(os.getenv('EMA_PERIODS', '5,20,50'), [5, 20, 50])
    
    MEMORY_WARNING_THRESHOLD_MB = _get_int_env('MEMORY_WARNING_THRESHOLD_MB', '400')
    MEMORY_CRITICAL_THRESHOLD_MB = _get_int_env('MEMORY_CRITICAL_THRESHOLD_MB', '450')
    OOM_GRACEFUL_DEGRADATION = os.getenv('OOM_GRACEFUL_DEGRADATION', 'true').lower() == 'true'
    MAX_CANDLE_HISTORY = _get_int_env('MAX_CANDLE_HISTORY', '100') if FREE_TIER_MODE else _get_int_env('MAX_CANDLE_HISTORY', '500')
    CHART_CLEANUP_AGE_MINUTES = _get_int_env('CHART_CLEANUP_AGE_MINUTES', '60')
    MEMORY_MONITOR_INTERVAL_SECONDS = _get_int_env('MEMORY_MONITOR_INTERVAL_SECONDS', '60')
    
    TASK_AUTO_CANCEL_THRESHOLD = _get_int_env('TASK_AUTO_CANCEL_THRESHOLD', '600')
    TASK_ROTATION_INTERVAL = _get_int_env('TASK_ROTATION_INTERVAL', '600')
    TASK_RESTART_BACKOFF_BASE = _get_float_env('TASK_RESTART_BACKOFF_BASE', '1.0')
    TASK_RESTART_BACKOFF_MAX = _get_float_env('TASK_RESTART_BACKOFF_MAX', '8.0')
    COMPLETED_TASKS_MAX_SIZE = _get_int_env('COMPLETED_TASKS_MAX_SIZE', '1000')
    COMPLETED_TASKS_CLEANUP_INTERVAL = _get_int_env('COMPLETED_TASKS_CLEANUP_INTERVAL', '300')
    
    GC_INTERVAL_SECONDS = _get_int_env('GC_INTERVAL_SECONDS', '180')
    TICK_DATA_RETENTION_HOURS = _get_int_env('TICK_DATA_RETENTION_HOURS', '1')
    CHART_LAZY_LOAD = os.getenv('CHART_LAZY_LOAD', 'true').lower() == 'true'
    
    TELEGRAM_CIRCUIT_BREAKER_THRESHOLD = _get_int_env('TELEGRAM_CIRCUIT_BREAKER_THRESHOLD', '5')
    TELEGRAM_CIRCUIT_BREAKER_TIMEOUT = _get_float_env('TELEGRAM_CIRCUIT_BREAKER_TIMEOUT', '60.0')
    TELEGRAM_RATE_LIMIT_CALLS = _get_int_env('TELEGRAM_RATE_LIMIT_CALLS', '30')
    TELEGRAM_RATE_LIMIT_WINDOW = _get_float_env('TELEGRAM_RATE_LIMIT_WINDOW', '60.0')
    TELEGRAM_MESSAGE_QUEUE_RATE = _get_int_env('TELEGRAM_MESSAGE_QUEUE_RATE', '20')
    WEBHOOK_TIMEOUT_THRESHOLD = _get_int_env('WEBHOOK_TIMEOUT_THRESHOLD', '3')
    
    SIGNAL_QUEUE_BUFFER_SIZE = _get_int_env('SIGNAL_QUEUE_BUFFER_SIZE', '100')
    SIGNAL_LOW_CONFIDENCE_THRESHOLD = _get_int_env('SIGNAL_LOW_CONFIDENCE_THRESHOLD', '40')
    SIGNAL_HIGH_CONFIDENCE_THRESHOLD = _get_int_env('SIGNAL_HIGH_CONFIDENCE_THRESHOLD', '60')
    MTF_RELAXATION_WEIGHT = _get_float_env('MTF_RELAXATION_WEIGHT', '0.3')
    VOLUME_BYPASS_THRESHOLD = _get_float_env('VOLUME_BYPASS_THRESHOLD', '0.5')
    
    PERFORMANCE_SAMPLE_RATE = _get_int_env('PERFORMANCE_SAMPLE_RATE', '100')
    ALERT_BATCH_WINDOW_SECONDS = _get_int_env('ALERT_BATCH_WINDOW_SECONDS', '60')
    DASHBOARD_UPDATE_INTERVAL = _get_int_env('DASHBOARD_UPDATE_INTERVAL', '10')
    
    WS_HEARTBEAT_INTERVAL = _get_int_env('WS_HEARTBEAT_INTERVAL', '25')
    WS_RECONNECT_JITTER_MAX = _get_float_env('WS_RECONNECT_JITTER_MAX', '5.0')
    HEALTH_CACHE_SECONDS = _get_int_env('HEALTH_CACHE_SECONDS', '30')
    STATIC_CACHE_HOURS = _get_int_env('STATIC_CACHE_HOURS', '1')
    
    DB_MAX_CONNECTIONS = _get_int_env('DB_MAX_CONNECTIONS', '3')
    DB_TRADE_PAGINATION_SIZE = _get_int_env('DB_TRADE_PAGINATION_SIZE', '50')
    DB_ARCHIVE_DAYS = _get_int_env('DB_ARCHIVE_DAYS', '30')
    DB_VACUUM_INTERVAL_HOURS = _get_int_env('DB_VACUUM_INTERVAL_HOURS', '6')
    
    LOG_MAX_SIZE_MB = _get_int_env('LOG_MAX_SIZE_MB', '2')
    LOG_BACKUP_COUNT = _get_int_env('LOG_BACKUP_COUNT', '2')
    
    @classmethod
    def get_masked_token(cls) -> str:
        from bot.logger import mask_token
        return mask_token(cls.TELEGRAM_BOT_TOKEN)
    
    @classmethod
    def validate(cls, strict: bool = True) -> bool:
        """
        Validate all configuration settings with comprehensive checks.
        
        Args:
            strict: If True, raises exception on errors. If False, returns False.
        
        Raises:
            ConfigError: If strict=True and critical configuration is missing or invalid
        
        Returns:
            True if validation passes, False otherwise (when strict=False)
        """
        errors = []
        warnings = []
        
        err = _validate_non_empty_string(cls.TELEGRAM_BOT_TOKEN, "TELEGRAM_BOT_TOKEN")
        if err:
            errors.append(err)
        else:
            token_pattern = r'^\d+:[A-Za-z0-9_-]{35,}$'
            if not re.match(token_pattern, cls.TELEGRAM_BOT_TOKEN.strip()):
                warnings.append("TELEGRAM_BOT_TOKEN format may be invalid (expected format: numbers:alphanumeric)")
        
        err = _validate_list_not_empty(cls.AUTHORIZED_USER_IDS, "AUTHORIZED_USER_IDS")
        if err:
            errors.append(err)
        else:
            err = _validate_list_items_type(cls.AUTHORIZED_USER_IDS, int, "AUTHORIZED_USER_IDS")
            if err:
                errors.append(err)
        
        err = _validate_list_items_type(cls.ID_USER_PUBLIC, int, "ID_USER_PUBLIC")
        if err:
            errors.append(err)
        
        err = _validate_list_not_empty(cls.EMA_PERIODS, "EMA_PERIODS")
        if err:
            errors.append(err)
        else:
            err = _validate_list_items_type(cls.EMA_PERIODS, int, "EMA_PERIODS")
            if err:
                errors.append(err)
        
        if cls.TELEGRAM_WEBHOOK_MODE:
            if not cls.WEBHOOK_URL or cls.WEBHOOK_URL.strip() == '':
                warnings.append("TELEGRAM_WEBHOOK_MODE is enabled but WEBHOOK_URL is not set - will attempt auto-detection")
            else:
                err = _validate_url_format(cls.WEBHOOK_URL, "WEBHOOK_URL")
                if err:
                    errors.append(err)
                elif '/bot' not in cls.WEBHOOK_URL:
                    warnings.append("WEBHOOK_URL should typically contain '/bot<token>' endpoint for Telegram webhooks")
        
        err = _validate_range(cls.RISK_PER_TRADE_PERCENT, 0.01, 100, "RISK_PER_TRADE_PERCENT")
        if err:
            errors.append(err)
        
        err = _validate_range(cls.DAILY_LOSS_PERCENT, 0.0, 100, "DAILY_LOSS_PERCENT")
        if err:
            errors.append(err)
        
        err = _validate_positive(cls.FIXED_RISK_AMOUNT, "FIXED_RISK_AMOUNT")
        if err:
            errors.append(err)
        
        err = _validate_positive(cls.TP_RR_RATIO, "TP_RR_RATIO")
        if err:
            errors.append(err)
        
        err = _validate_positive(cls.SL_ATR_MULTIPLIER, "SL_ATR_MULTIPLIER")
        if err:
            errors.append(err)
        
        err = _validate_positive(cls.MIN_SL_PIPS, "MIN_SL_PIPS")
        if err:
            errors.append(err)
        
        err = _validate_positive(cls.MIN_SL_SPREAD_MULTIPLIER, "MIN_SL_SPREAD_MULTIPLIER")
        if err:
            errors.append(err)
        
        err = _validate_positive(cls.DEFAULT_SL_PIPS, "DEFAULT_SL_PIPS")
        if err:
            errors.append(err)
        
        err = _validate_positive(cls.DEFAULT_TP_PIPS, "DEFAULT_TP_PIPS")
        if err:
            errors.append(err)
        
        err = _validate_positive(cls.RSI_PERIOD, "RSI_PERIOD")
        if err:
            errors.append(err)
        
        err = _validate_range(cls.RSI_OVERSOLD_LEVEL, 0, 100, "RSI_OVERSOLD_LEVEL")
        if err:
            errors.append(err)
        
        err = _validate_range(cls.RSI_OVERBOUGHT_LEVEL, 0, 100, "RSI_OVERBOUGHT_LEVEL")
        if err:
            errors.append(err)
        
        if cls.RSI_OVERSOLD_LEVEL >= cls.RSI_OVERBOUGHT_LEVEL:
            errors.append(f"RSI_OVERSOLD_LEVEL ({cls.RSI_OVERSOLD_LEVEL}) must be less than RSI_OVERBOUGHT_LEVEL ({cls.RSI_OVERBOUGHT_LEVEL})")
        
        err = _validate_range(cls.RSI_ENTRY_MIN, 0, 100, "RSI_ENTRY_MIN")
        if err:
            errors.append(err)
        
        err = _validate_range(cls.RSI_ENTRY_MAX, 0, 100, "RSI_ENTRY_MAX")
        if err:
            errors.append(err)
        
        if cls.RSI_ENTRY_MIN >= cls.RSI_ENTRY_MAX:
            errors.append(f"RSI_ENTRY_MIN ({cls.RSI_ENTRY_MIN}) must be less than RSI_ENTRY_MAX ({cls.RSI_ENTRY_MAX})")
        
        err = _validate_positive(cls.ATR_PERIOD, "ATR_PERIOD")
        if err:
            errors.append(err)
        
        err = _validate_positive(cls.MACD_FAST, "MACD_FAST")
        if err:
            errors.append(err)
        
        err = _validate_positive(cls.MACD_SLOW, "MACD_SLOW")
        if err:
            errors.append(err)
        
        if cls.MACD_FAST >= cls.MACD_SLOW:
            errors.append(f"MACD_FAST ({cls.MACD_FAST}) must be less than MACD_SLOW ({cls.MACD_SLOW})")
        
        err = _validate_positive(cls.SIGNAL_COOLDOWN_SECONDS, "SIGNAL_COOLDOWN_SECONDS", allow_zero=True)
        if err:
            errors.append(err)
        
        err = _validate_positive(cls.MAX_TRADES_PER_DAY, "MAX_TRADES_PER_DAY", allow_zero=True)
        if err:
            errors.append(err)
        
        err = _validate_positive(cls.MEMORY_WARNING_THRESHOLD_MB, "MEMORY_WARNING_THRESHOLD_MB")
        if err:
            errors.append(err)
        
        err = _validate_positive(cls.MEMORY_CRITICAL_THRESHOLD_MB, "MEMORY_CRITICAL_THRESHOLD_MB")
        if err:
            errors.append(err)
        
        if cls.MEMORY_WARNING_THRESHOLD_MB >= cls.MEMORY_CRITICAL_THRESHOLD_MB:
            warnings.append(f"MEMORY_WARNING_THRESHOLD_MB ({cls.MEMORY_WARNING_THRESHOLD_MB}) should be less than MEMORY_CRITICAL_THRESHOLD_MB ({cls.MEMORY_CRITICAL_THRESHOLD_MB})")
        
        err = _validate_positive(cls.VOLUME_AVG_PERIOD, "VOLUME_AVG_PERIOD")
        if err:
            errors.append(err)
        
        err = _validate_positive(cls.VOLUME_THRESHOLD_MULTIPLIER, "VOLUME_THRESHOLD_MULTIPLIER")
        if err:
            errors.append(err)
        
        err = _validate_range(cls.TRADING_HOURS_START, 0, 23, "TRADING_HOURS_START")
        if err:
            errors.append(err)
        
        err = _validate_range(cls.TRADING_HOURS_END, 0, 23, "TRADING_HOURS_END")
        if err:
            errors.append(err)
        
        if cls.TRADING_HOURS_START >= cls.TRADING_HOURS_END:
            warnings.append(f"TRADING_HOURS_START ({cls.TRADING_HOURS_START}) should be less than TRADING_HOURS_END ({cls.TRADING_HOURS_END})")
        
        err = _validate_range(cls.FRIDAY_CUTOFF_HOUR, 0, 23, "FRIDAY_CUTOFF_HOUR")
        if err:
            errors.append(err)
        
        err = _validate_positive(cls.NEWS_AVOID_MINUTES_BEFORE, "NEWS_AVOID_MINUTES_BEFORE", allow_zero=True)
        if err:
            errors.append(err)
        
        err = _validate_positive(cls.NEWS_AVOID_MINUTES_AFTER, "NEWS_AVOID_MINUTES_AFTER", allow_zero=True)
        if err:
            errors.append(err)
        
        err = _validate_positive(cls.ACCOUNT_BALANCE, "ACCOUNT_BALANCE")
        if err:
            errors.append(err)
        
        valid_timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN']
        if cls.MTF_CONFIRMATION_TIMEFRAME not in valid_timeframes:
            errors.append(f"MTF_CONFIRMATION_TIMEFRAME must be one of {valid_timeframes}, got {cls.MTF_CONFIRMATION_TIMEFRAME}")
        
        if warnings:
            try:
                from bot.logger import setup_logger
                logger = setup_logger('Config')
                for warning in warnings:
                    logger.warning(f"Configuration warning: {warning}")
            except (ImportError, AttributeError):
                pass
        
        if errors:
            if strict:
                raise ConfigError(
                    f"Configuration validation failed with {len(errors)} error(s)",
                    errors=errors,
                    warnings=warnings
                )
            return False
        
        return True
    
    @classmethod
    def validate_runtime(cls) -> dict:
        """
        Perform runtime validation and return status.
        
        Returns:
            dict with 'valid' boolean and 'issues' list
        """
        issues = []
        
        if not cls.TELEGRAM_BOT_TOKEN:
            issues.append("TELEGRAM_BOT_TOKEN not configured")
        
        if not cls.AUTHORIZED_USER_IDS:
            issues.append("AUTHORIZED_USER_IDS not configured")
        
        if cls.RISK_PER_TRADE_PERCENT > 5:
            issues.append(f"RISK_PER_TRADE_PERCENT ({cls.RISK_PER_TRADE_PERCENT}%) is unusually high")
        
        if cls.DAILY_LOSS_PERCENT > 10:
            issues.append(f"DAILY_LOSS_PERCENT ({cls.DAILY_LOSS_PERCENT}%) is unusually high")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
    
    RSI_PERIOD = _get_int_env('RSI_PERIOD', '14')
    RSI_OVERSOLD_LEVEL = _get_int_env('RSI_OVERSOLD_LEVEL', '20')
    RSI_OVERBOUGHT_LEVEL = _get_int_env('RSI_OVERBOUGHT_LEVEL', '80')
    RSI_ENTRY_MIN = _get_int_env('RSI_ENTRY_MIN', '15')
    RSI_ENTRY_MAX = _get_int_env('RSI_ENTRY_MAX', '85')
    STOCH_K_PERIOD = _get_int_env('STOCH_K_PERIOD', '8')
    STOCH_D_PERIOD = _get_int_env('STOCH_D_PERIOD', '3')
    STOCH_SMOOTH_K = _get_int_env('STOCH_SMOOTH_K', '3')
    STOCH_OVERSOLD_LEVEL = _get_int_env('STOCH_OVERSOLD_LEVEL', '10')
    STOCH_OVERBOUGHT_LEVEL = _get_int_env('STOCH_OVERBOUGHT_LEVEL', '90')
    ATR_PERIOD = _get_int_env('ATR_PERIOD', '14')
    MACD_FAST = _get_int_env('MACD_FAST', '12')
    MACD_SLOW = _get_int_env('MACD_SLOW', '26')
    MACD_SIGNAL = _get_int_env('MACD_SIGNAL', '9')
    VOLUME_AVG_PERIOD = _get_int_env('VOLUME_AVG_PERIOD', '10')
    VOLUME_THRESHOLD_MULTIPLIER = _get_float_env('VOLUME_THRESHOLD_MULTIPLIER', '0.3')
    MAX_SPREAD_PIPS = _get_float_env('MAX_SPREAD_PIPS', '50.0')
    
    ADX_PERIOD = _get_int_env('ADX_PERIOD', '14')
    ADX_THRESHOLD = _get_int_env('ADX_THRESHOLD', '15')
    ADX_FILTER_ENABLED = os.getenv('ADX_FILTER_ENABLED', 'false').lower() == 'true'
    
    RSI_BUY_MAX_LEVEL = _get_int_env('RSI_BUY_MAX_LEVEL', '80')
    RSI_SELL_MIN_LEVEL = _get_int_env('RSI_SELL_MIN_LEVEL', '20')
    RSI_LEVEL_FILTER_ENABLED = os.getenv('RSI_LEVEL_FILTER_ENABLED', 'false').lower() == 'true'
    
    RSI_EXTREME_FILTER_ENABLED = os.getenv('RSI_EXTREME_FILTER_ENABLED', 'false').lower() == 'true'
    RSI_EXTREME_OVERBOUGHT = _get_int_env('RSI_EXTREME_OVERBOUGHT', '85')
    RSI_EXTREME_OVERSOLD = _get_int_env('RSI_EXTREME_OVERSOLD', '15')
    
    EMA_SLOPE_FILTER_ENABLED = os.getenv('EMA_SLOPE_FILTER_ENABLED', 'false').lower() == 'true'
    EMA_SLOPE_MIN_THRESHOLD = _get_float_env('EMA_SLOPE_MIN_THRESHOLD', '0.0001')
    
    SESSION_FILTER_STRICT = os.getenv('SESSION_FILTER_STRICT', 'false').lower() == 'true'
    LONDON_SESSION_START_UTC = _get_int_env('LONDON_SESSION_START_UTC', '0')
    NY_SESSION_END_UTC = _get_int_env('NY_SESSION_END_UTC', '23')
    
    SIGNAL_SCORE_THRESHOLD_AUTO = _get_int_env('SIGNAL_SCORE_THRESHOLD_AUTO', '60')
    SIGNAL_SCORE_THRESHOLD_MANUAL = _get_int_env('SIGNAL_SCORE_THRESHOLD_MANUAL', '40')
    
    SL_ATR_MULTIPLIER = _get_float_env('SL_ATR_MULTIPLIER', '1.2')
    MIN_SL_PIPS = _get_float_env('MIN_SL_PIPS', '10.0')
    MIN_SL_SPREAD_MULTIPLIER = _get_float_env('MIN_SL_SPREAD_MULTIPLIER', '2.0')
    DEFAULT_SL_PIPS = _get_float_env('DEFAULT_SL_PIPS', '20.0')
    TP_RR_RATIO = _get_float_env('TP_RR_RATIO', '1.5')
    TP_RR_RATIO_MAX = _get_float_env('TP_RR_RATIO_MAX', '2.5')
    DEFAULT_TP_PIPS = _get_float_env('DEFAULT_TP_PIPS', '40.0')
    
    SIGNAL_COOLDOWN_SECONDS = _get_int_env('SIGNAL_COOLDOWN_SECONDS', '0')
    MAX_TRADES_PER_DAY = _get_int_env('MAX_TRADES_PER_DAY', '0')
    
    SIGNAL_MINIMUM_PRICE_MOVEMENT = _get_float_env('SIGNAL_MINIMUM_PRICE_MOVEMENT', '0.0')
    TICK_COOLDOWN_FOR_SAME_SIGNAL = _get_int_env('TICK_COOLDOWN_FOR_SAME_SIGNAL', '0')
    
    BUY_COOLDOWN_SECONDS = _get_int_env('BUY_COOLDOWN_SECONDS', '0')
    SELL_COOLDOWN_SECONDS = _get_int_env('SELL_COOLDOWN_SECONDS', '0')
    PATTERN_COOLDOWN_SECONDS = _get_int_env('PATTERN_COOLDOWN_SECONDS', '0')
    GLOBAL_MIN_COOLDOWN_SECONDS = _get_int_env('GLOBAL_MIN_COOLDOWN_SECONDS', '0')
    OPPOSITE_SIGNAL_COOLDOWN_SECONDS = _get_int_env('OPPOSITE_SIGNAL_COOLDOWN_SECONDS', '0')
    
    ALLOW_MULTIPLE_SIGNALS_PER_CANDLE = os.getenv('ALLOW_MULTIPLE_SIGNALS_PER_CANDLE', 'true').lower() == 'true'
    AUTO_SIGNAL_REPLACEMENT_ALLOWED = os.getenv('AUTO_SIGNAL_REPLACEMENT_ALLOWED', 'true').lower() == 'true'
    CANDLE_CLOSE_ONLY_SIGNALS = os.getenv('CANDLE_CLOSE_ONLY_SIGNALS', 'false').lower() == 'true'
    DAILY_LOSS_PERCENT = _get_float_env('DAILY_LOSS_PERCENT', '0.0')
    RISK_PER_TRADE_PERCENT = _get_float_env('RISK_PER_TRADE_PERCENT', '20.0')
    FIXED_RISK_AMOUNT = _get_float_env('FIXED_RISK_AMOUNT', '2.0')
    
    DYNAMIC_SL_LOSS_THRESHOLD = _get_float_env('DYNAMIC_SL_LOSS_THRESHOLD', '3.0')
    DYNAMIC_SL_TIGHTENING_MULTIPLIER = _get_float_env('DYNAMIC_SL_TIGHTENING_MULTIPLIER', '0.4')
    BREAKEVEN_PROFIT_THRESHOLD = _get_float_env('BREAKEVEN_PROFIT_THRESHOLD', '2.0')
    TRAILING_STOP_PROFIT_THRESHOLD = _get_float_env('TRAILING_STOP_PROFIT_THRESHOLD', '2.5')
    TRAILING_STOP_DISTANCE_PIPS = _get_float_env('TRAILING_STOP_DISTANCE_PIPS', '5.0')
    
    CHART_AUTO_DELETE = os.getenv('CHART_AUTO_DELETE', 'true').lower() == 'true'
    CHART_EXPIRY_MINUTES = _get_int_env('CHART_EXPIRY_MINUTES', '60')
    
    WS_DISCONNECT_ALERT_SECONDS = _get_int_env('WS_DISCONNECT_ALERT_SECONDS', '30')
    
    DATABASE_URL = os.getenv('DATABASE_URL', '')
    DATABASE_PATH = os.getenv('DATABASE_PATH', 'data/bot.db')
    RESET_HISTORICAL_DATA_ON_START = os.getenv('RESET_HISTORICAL_DATA_ON_START', 'false').lower() == 'true'
    
    DRY_RUN = os.getenv('DRY_RUN', 'false').lower() == 'true'
    
    HEALTH_CHECK_PORT = _get_int_env('PORT', '8080')
    HEALTH_CHECK_INTERVAL = _get_int_env('HEALTH_CHECK_INTERVAL', '120')
    HEALTH_CHECK_TIMEOUT = _get_int_env('HEALTH_CHECK_TIMEOUT', '30')
    HEALTH_CHECK_LONG_TIMEOUT = _get_int_env('HEALTH_CHECK_LONG_TIMEOUT', '35')
    
    SELF_PING_ENABLED = os.getenv('SELF_PING_ENABLED', 'true').lower() == 'true'
    SELF_PING_INTERVAL = _get_int_env('SELF_PING_INTERVAL', '240')
    SELF_PING_TIMEOUT = _get_int_env('SELF_PING_TIMEOUT', '5')
    
    TRADING_HOURS_START = _get_int_env('TRADING_HOURS_START', '0')
    TRADING_HOURS_END = _get_int_env('TRADING_HOURS_END', '23')
    FRIDAY_CUTOFF_HOUR = _get_int_env('FRIDAY_CUTOFF_HOUR', '23')
    UNLIMITED_TRADING_HOURS = os.getenv('UNLIMITED_TRADING_HOURS', 'true').lower() == 'true'
    NEWS_AVOID_MINUTES_BEFORE = _get_int_env('NEWS_AVOID_MINUTES_BEFORE', '5')
    NEWS_AVOID_MINUTES_AFTER = _get_int_env('NEWS_AVOID_MINUTES_AFTER', '10')
    
    MTF_ENABLED = os.getenv('MTF_ENABLED', 'true').lower() == 'true'
    MTF_CONFIRMATION_TIMEFRAME = os.getenv('MTF_CONFIRMATION_TIMEFRAME', 'M15')
    
    ACCOUNT_BALANCE = _get_float_env('ACCOUNT_BALANCE', '10.0')
    ACCOUNT_CURRENCY = os.getenv('ACCOUNT_CURRENCY', 'USD')
    
    XAUUSD_PIP_VALUE = 10.0
    LOT_SIZE = 0.01
    
    TRAILING_STOP_NOTIFY_COOLDOWN = _get_float_env('TRAILING_STOP_NOTIFY_COOLDOWN', '30.0')
    
    @classmethod
    def check_memory_status(cls) -> dict:
        """Check current memory usage and return status
        
        Returns:
            dict: Memory status with level ('normal', 'warning', 'critical')
        """
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            mem_mb = mem_info.rss / (1024 * 1024)
            
            status = 'normal'
            if mem_mb >= cls.MEMORY_CRITICAL_THRESHOLD_MB:
                status = 'critical'
            elif mem_mb >= cls.MEMORY_WARNING_THRESHOLD_MB:
                status = 'warning'
            
            return {
                'memory_mb': round(mem_mb, 2),
                'warning_threshold': cls.MEMORY_WARNING_THRESHOLD_MB,
                'critical_threshold': cls.MEMORY_CRITICAL_THRESHOLD_MB,
                'status': status,
                'graceful_degradation_enabled': cls.OOM_GRACEFUL_DEGRADATION
            }
        except ImportError:
            try:
                import resource
                mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                mem_mb = mem_usage / 1024
                
                status = 'normal'
                if mem_mb >= cls.MEMORY_CRITICAL_THRESHOLD_MB:
                    status = 'critical'
                elif mem_mb >= cls.MEMORY_WARNING_THRESHOLD_MB:
                    status = 'warning'
                
                return {
                    'memory_mb': round(mem_mb, 2),
                    'warning_threshold': cls.MEMORY_WARNING_THRESHOLD_MB,
                    'critical_threshold': cls.MEMORY_CRITICAL_THRESHOLD_MB,
                    'status': status,
                    'source': 'resource',
                    'graceful_degradation_enabled': cls.OOM_GRACEFUL_DEGRADATION
                }
            except Exception:
                return {'status': 'unknown', 'error': 'psutil/resource not available'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    @classmethod
    def should_degrade_gracefully(cls) -> bool:
        """Check if bot should enter graceful degradation mode
        
        Returns:
            bool: True if memory is critical and degradation is enabled
        """
        if not cls.OOM_GRACEFUL_DEGRADATION:
            return False
        
        status = cls.check_memory_status()
        return status.get('status') == 'critical'
    
    @classmethod
    def get_adjusted_settings(cls) -> dict:
        """Get adjusted settings based on memory status
        
        Returns settings that should be reduced when memory is critical
        """
        if not cls.should_degrade_gracefully():
            return {}
        
        return {
            'max_candle_history': 100,
            'chart_generation': False,
            'signal_cache_expiry': 60,
            'position_update_interval': 5
        }
