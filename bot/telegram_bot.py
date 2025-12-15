import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.request import HTTPXRequest
from telegram.error import (
    TelegramError, NetworkError, TimedOut, RetryAfter, BadRequest,
    Forbidden, Conflict, ChatMigrated, InvalidToken
)
from telegram import error as telegram_error
from datetime import datetime, timedelta
import pytz
import pandas as pd
from typing import Optional, List, Callable, Any, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
import time
from bot.logger import setup_logger, mask_user_id, mask_token, sanitize_log_message
from bot.database import Trade, Position, Performance
from sqlalchemy.exc import SQLAlchemyError
from bot.signal_session_manager import SignalSessionManager
from bot.message_templates import MessageFormatter
from bot.resilience import RateLimiter
from bot.signal_event_store import SignalEventStore
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bot.market_regime import MarketRegimeDetector
    from bot.signal_rules import AggressiveSignalRules
    from bot.signal_quality_tracker import SignalQualityTracker
    from bot.auto_optimizer import AutoOptimizer

logger = setup_logger('TelegramBot')


class MonitoringAction(Enum):
    """Actions returned by monitoring helper coroutines"""
    CONTINUE = auto()
    SKIP_TICK = auto()
    PROCESS_SIGNAL = auto()
    BREAK_LOOP = auto()
    RESUBSCRIBE = auto()


@dataclass
class MonitoringContext:
    """Mutable state container for monitoring loop - reduces parameter passing"""
    chat_id: int
    last_signal_check: datetime = field(default_factory=lambda: datetime.now() - timedelta(seconds=60))
    last_tick_process_time: datetime = field(default_factory=lambda: datetime.now() - timedelta(seconds=1))
    last_sent_signal: Optional[str] = None
    last_sent_signal_price: Optional[float] = None
    last_sent_signal_time: datetime = field(default_factory=lambda: datetime.now() - timedelta(seconds=5))
    retry_delay: float = 1.0
    max_retry_delay: float = 30.0
    last_candle_timestamp: Any = None
    consecutive_timeouts: int = 0
    max_consecutive_timeouts: int = 5
    daily_summary_skip_count: int = 0
    last_daily_summary_log_time: datetime = field(default_factory=lambda: datetime.now() - timedelta(seconds=60))
    
    def reset_retry_delay(self):
        """Reset retry delay setelah sukses"""
        self.retry_delay = 1.0
    
    def increase_retry_delay(self):
        """Increase retry delay dengan exponential backoff"""
        self.retry_delay = min(self.retry_delay * 2, self.max_retry_delay)
    
    def record_timeout(self) -> bool:
        """Record timeout dan return True jika perlu resubscribe"""
        self.consecutive_timeouts += 1
        return self.consecutive_timeouts >= self.max_consecutive_timeouts
    
    def reset_timeouts(self):
        """Reset timeout counter"""
        self.consecutive_timeouts = 0
    
    def update_last_signal(self, direction: Optional[str], price: Optional[float], timestamp: datetime):
        """Update tracking sinyal terakhir"""
        if direction is not None:
            self.last_sent_signal = direction
        if price is not None:
            self.last_sent_signal_price = price
        self.last_sent_signal_time = timestamp
    
    def copy(self) -> 'MonitoringContext':
        """Create defensive copy untuk per-chat isolation"""
        return MonitoringContext(
            chat_id=self.chat_id,
            last_signal_check=self.last_signal_check,
            last_tick_process_time=self.last_tick_process_time,
            last_sent_signal=self.last_sent_signal,
            last_sent_signal_price=self.last_sent_signal_price,
            last_sent_signal_time=self.last_sent_signal_time,
            retry_delay=self.retry_delay,
            max_retry_delay=self.max_retry_delay,
            last_candle_timestamp=self.last_candle_timestamp,
            consecutive_timeouts=self.consecutive_timeouts,
            max_consecutive_timeouts=self.max_consecutive_timeouts,
            daily_summary_skip_count=self.daily_summary_skip_count,
            last_daily_summary_log_time=self.last_daily_summary_log_time
        )
    
    def validate(self, expected_chat_id: int) -> bool:
        """Validate bahwa context adalah untuk chat yang benar"""
        return self.chat_id == expected_chat_id

class TelegramBotError(Exception):
    """Base exception for Telegram bot errors"""
    pass

class RateLimitError(TelegramBotError):
    """Rate limit exceeded"""
    pass

class ValidationError(TelegramBotError):
    """Input validation error"""
    pass

def validate_user_id(user_id_str: str) -> tuple[bool, Optional[int], Optional[str]]:
    """Validate and sanitize user ID input
    
    Returns:
        tuple: (is_valid, sanitized_user_id, error_message)
    """
    try:
        if not user_id_str or not isinstance(user_id_str, str):
            return False, None, "User ID tidak boleh kosong"
        
        user_id_str = user_id_str.strip()
        
        if not user_id_str.isdigit():
            return False, None, "User ID harus berupa angka"
        
        user_id = int(user_id_str)
        
        if user_id <= 0:
            return False, None, "User ID harus positif"
        
        if user_id > 9999999999:
            return False, None, "User ID tidak valid (terlalu besar)"
        
        return True, user_id, None
        
    except ValueError:
        return False, None, "Format user ID tidak valid"
    except (TypeError, AttributeError) as e:
        return False, None, f"Error validasi user ID: {str(e)}"

def validate_duration_days(duration_str: str) -> tuple[bool, Optional[int], Optional[str]]:
    """Validate and sanitize duration days input
    
    Returns:
        tuple: (is_valid, sanitized_duration, error_message)
    """
    try:
        if not duration_str or not isinstance(duration_str, str):
            return False, None, "Durasi tidak boleh kosong"
        
        duration_str = duration_str.strip()
        
        if not duration_str.isdigit():
            return False, None, "Durasi harus berupa angka"
        
        duration = int(duration_str)
        
        if duration <= 0:
            return False, None, "Durasi harus lebih dari 0 hari"
        
        if duration > 365:
            return False, None, "Durasi maksimal 365 hari"
        
        return True, duration, None
        
    except ValueError:
        return False, None, "Format durasi tidak valid"
    except (TypeError, AttributeError) as e:
        return False, None, f"Error validasi durasi: {str(e)}"

def sanitize_command_argument(arg: str, max_length: int = 100) -> str:
    """Sanitize command argument to prevent injection
    
    Args:
        arg: Argument string to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized argument string
    """
    if not arg or not isinstance(arg, str):
        return ""
    
    sanitized = arg.strip()
    
    sanitized = ''.join(c for c in sanitized if c.isprintable())
    
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    dangerous_patterns = ['<script>', 'javascript:', 'onerror=', 'onclick=', '${', '`']
    for pattern in dangerous_patterns:
        if pattern.lower() in sanitized.lower():
            logger.warning(f"Dangerous pattern detected in input: {pattern}")
            sanitized = sanitized.replace(pattern, '')
    
    return sanitized

def retry_on_telegram_error(max_retries: int = 3, initial_delay: float = 1.0):
    """Decorator to retry Telegram API calls with exponential backoff.
    
    Error handling strategy:
    - Transient errors (NetworkError, TimedOut): Retry with exponential backoff
    - Rate limit (RetryAfter): Wait for specified duration then retry
    - Permanent errors (BadRequest, Forbidden): No retry, log and raise
    - Auth errors (InvalidToken, Conflict): Critical alert, no retry
    - Migration (ChatMigrated): Extract new chat ID and raise with info
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            delay = initial_delay
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                    
                except RetryAfter as e:
                    retry_after = e.retry_after if hasattr(e, 'retry_after') else 5
                    logger.warning(f"Rate limit hit in {func.__name__}, retrying after {retry_after}s")
                    await asyncio.sleep(retry_after + 1)
                    last_exception = e
                    
                except TimedOut as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Timeout in {func.__name__} (attempt {attempt + 1}/{max_retries}), retrying in {delay}s")
                        await asyncio.sleep(delay)
                        delay *= 2
                        last_exception = e
                    else:
                        logger.error(f"Max retries reached for {func.__name__} due to timeout")
                        raise
                        
                except BadRequest as e:
                    logger.error(f"BadRequest in {func.__name__}: {e} - Invalid request, tidak akan retry")
                    raise
                
                except NetworkError as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Network error in {func.__name__} (attempt {attempt + 1}/{max_retries}): {e}")
                        await asyncio.sleep(delay)
                        delay *= 2
                        last_exception = e
                    else:
                        logger.error(f"Max retries reached for {func.__name__} due to network error")
                        raise
                
                except Forbidden as e:
                    logger.warning(f"Forbidden in {func.__name__}: {e} - User mungkin memblokir bot atau chat tidak dapat diakses")
                    raise
                
                except ChatMigrated as e:
                    new_chat_id = e.new_chat_id if hasattr(e, 'new_chat_id') else None
                    logger.warning(f"ChatMigrated in {func.__name__}: Chat migrated to new ID: {new_chat_id}")
                    raise
                
                except Conflict as e:
                    logger.critical(f"🔴 CONFLICT in {func.__name__}: {e} - Multiple bot instances detected!")
                    raise
                
                except InvalidToken as e:
                    logger.critical(f"🔴 UNAUTHORIZED in {func.__name__}: {e} - Token tidak valid atau bot di-revoke!")
                    raise
                        
                except TelegramError as e:
                    logger.error(f"Telegram API error in {func.__name__}: {e}")
                    raise
                    
                except asyncio.CancelledError:
                    logger.info(f"Task cancelled in {func.__name__}")
                    raise
                except (ValueError, TypeError, AttributeError, KeyError) as e:
                    logger.error(f"Data error in {func.__name__}: {type(e).__name__}: {e}")
                    raise
            
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator

def validate_chat_id(chat_id: Any) -> bool:
    """Validate chat ID"""
    try:
        if chat_id is None:
            return False
        if isinstance(chat_id, int):
            return chat_id != 0
        if isinstance(chat_id, str):
            return chat_id.lstrip('-').isdigit()
        return False
    except (ValueError, TypeError, AttributeError):
        return False

class TradingBot:
    MAX_CACHE_SIZE = 1000
    MAX_DASHBOARDS = 100
    MAX_MONITORING_CHATS = 50
    
    def __init__(self, config, db_manager, strategy, risk_manager, 
                 market_data, position_tracker, chart_generator,
                 alert_system=None, error_handler=None, user_manager=None, signal_session_manager=None, task_scheduler=None,
                 market_regime_detector: Optional['MarketRegimeDetector'] = None,
                 signal_rules: Optional['AggressiveSignalRules'] = None,
                 signal_quality_tracker: Optional['SignalQualityTracker'] = None,
                 auto_optimizer: Optional['AutoOptimizer'] = None):
        self.config = config
        self.db = db_manager
        self.db_manager = db_manager
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.market_data = market_data
        self.position_tracker = position_tracker
        self.chart_generator = chart_generator
        self.alert_system = alert_system
        self.error_handler = error_handler
        self.user_manager = user_manager
        self.signal_session_manager = signal_session_manager
        self.task_scheduler = task_scheduler
        self.market_regime_detector = market_regime_detector
        self.signal_rules = signal_rules
        self.signal_quality_tracker = signal_quality_tracker
        self.auto_optimizer = auto_optimizer
        
        from bot.analytics import TradingAnalytics
        self.analytics = TradingAnalytics(db_manager, config)
        
        self.app = None
        self.monitoring = False
        self.monitoring_chats = []
        self.signal_lock = asyncio.Lock()
        self.monitoring_tasks: Dict[int, asyncio.Task] = {}
        self.active_dashboards: Dict[int, Dict] = {}
        self._is_shutting_down: bool = False
        self._shutdown_drain_timeout: float = 30.0
        self._active_monitoring: Dict[int, Dict[str, Any]] = {}
        self._monitoring_drain_complete: Dict[int, asyncio.Event] = {}
        
        self._bot_healthy: bool = False
        self._last_health_check: Optional[datetime] = None
        self._health_check_interval: float = 30.0
        
        self.dashboard_messages: Dict[int, int] = {}
        self.dashboard_tasks: Dict[int, asyncio.Task] = {}
        self.dashboard_enabled: Dict[int, bool] = {}
        self._dashboard_last_hash: Dict[int, str] = {}
        self._realtime_dashboard_lock = asyncio.Lock()
        self.DASHBOARD_UPDATE_INTERVAL = 5  # Update setiap 5 detik untuk real-time
        logger.info("✅ Real-time dashboard system initialized (interval: 5s)")
        
        self.rate_limiter = RateLimiter(
            max_calls=30,
            time_window=60.0,
            name="TelegramAPI"
        )
        logger.info("✅ Rate limiter global diinisialisasi untuk Telegram API")
        
        self._user_rate_limiters: Dict[int, RateLimiter] = {}
        self._user_rate_limiter_lock = asyncio.Lock()
        self.MAX_USER_RATE_LIMITERS = 1000
        self.USER_RATE_LIMIT_MAX_CALLS = 10
        self.USER_RATE_LIMIT_TIME_WINDOW = 60.0
        logger.info(f"✅ Per-user rate limiter diinisialisasi: max {self.USER_RATE_LIMIT_MAX_CALLS} calls/{self.USER_RATE_LIMIT_TIME_WINDOW}s per user")
        
        self.global_last_signal_time = datetime.now() - timedelta(seconds=60)
        self.signal_detection_interval = 0  # INSTANT - 0 delay, check on every tick
        self.global_signal_cooldown = 0.0  # UNLIMITED - tidak ada cooldown global
        self.tick_throttle_seconds = 0.0  # UNLIMITED - tidak ada throttle
        logger.info(f"✅ UNLIMITED signal detection: global_cooldown={self.global_signal_cooldown}s, tick_throttle={self.tick_throttle_seconds}s")
        
        self.sent_signals_cache: Dict[str, Dict[str, Any]] = {}
        self.signal_cache_expiry_seconds = 300  # 5 menit cache expiry
        self.signal_price_tolerance_pips = 10.0
        self.last_signal_per_type: Dict[str, Dict] = {}  # Tracking terakhir per signal type (BUY/SELL)
        logger.info(f"✅ Anti-duplicate cache: expiry={self.signal_cache_expiry_seconds}s, dengan tracking per signal type")
        
        self.signal_event_store = SignalEventStore(
            ttl_seconds=3600,  # 1 jam TTL
            max_signals_per_user=100
        )
        logger.info("✅ SignalEventStore diinisialisasi untuk sinkronisasi sinyal dengan WebApp Dashboard")
        self._cache_lock = asyncio.Lock()
        self._dashboard_lock = asyncio.Lock()
        self._chart_cleanup_lock = asyncio.Lock()
        self._cache_cleanup_task: Optional[asyncio.Task] = None
        self._dashboard_cleanup_task: Optional[asyncio.Task] = None
        self._auto_optimization_task: Optional[asyncio.Task] = None
        self._chart_cleanup_task: Optional[asyncio.Task] = None
        self._aggressive_chart_cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_tasks_running: bool = False
        
        self._chart_cleanup_interval_minutes = getattr(self.config, 'CHART_CLEANUP_INTERVAL_MINUTES', 5)
        logger.info(f"✅ Aggressive chart cleanup interval: {self._chart_cleanup_interval_minutes} minutes")
        
        self._cache_telemetry = {
            'hits': 0,
            'misses': 0,
            'pending_set': 0,
            'confirmed': 0,
            'rollbacks': 0,
            'expired_cleanups': 0,
            'size_enforcements': 0,
            'last_cleanup_time': None,
            'last_cleanup_count': 0,
        }
        self._pending_charts: Dict[int, Dict[str, Any]] = {}
        self._chart_eviction_callbacks: List[Callable] = []
        logger.info("✅ Two-phase anti-duplicate signal cache initialized (pending→confirmed) with telemetry")
        
        import os
        self.instance_lock_file = os.path.join('data', '.bot_instance.lock')
        os.makedirs('data', exist_ok=True)
        
    def is_authorized(self, user_id: int) -> bool:
        if self.user_manager:
            return self.user_manager.has_access(user_id)
        
        if not self.config.AUTHORIZED_USER_IDS:
            return True
        return user_id in self.config.AUTHORIZED_USER_IDS
    
    def is_admin(self, user_id: int) -> bool:
        # Check AUTHORIZED_USER_IDS dulu (dari secrets)
        if user_id in self.config.AUTHORIZED_USER_IDS:
            return True
        # Kalau tidak, check database kalau user_manager ada
        if self.user_manager:
            user = self.user_manager.get_user(user_id)
            return user.is_admin if user else False
        return False
    
    async def prepare_for_shutdown(self, drain_timeout: Optional[float] = None) -> Dict[str, Any]:
        """Prepare bot for graceful shutdown. Call this before TaskScheduler.stop().
        
        This method sets the shutdown flag and waits for monitoring loops to drain
        their current iterations gracefully without interrupting mid-signal-processing.
        
        Args:
            drain_timeout: Maximum time to wait for monitoring tasks to drain (default: 30s)
            
        Returns:
            Dict containing shutdown statistics
        """
        if drain_timeout is not None:
            self._shutdown_drain_timeout = drain_timeout
        
        logger.info("🛑 Preparing TelegramBot for shutdown...")
        
        self._is_shutting_down = True
        logger.info("✅ Shutdown flag set - monitoring loops will exit after current iteration")
        
        monitoring_count = len(self.monitoring_tasks)
        dashboard_count = len(self.active_dashboards)
        
        for chat_id in list(self.monitoring_tasks.keys()):
            if chat_id not in self._monitoring_drain_complete:
                self._monitoring_drain_complete[chat_id] = asyncio.Event()
        
        drain_results = await self._wait_for_monitoring_drain(self._shutdown_drain_timeout)
        
        shutdown_stats = {
            'monitoring_tasks_drained': drain_results.get('drained_count', 0),
            'monitoring_tasks_timeout': drain_results.get('timeout_count', 0),
            'dashboard_tasks_cancelled': dashboard_count,
            'shutdown_initiated_at': datetime.now().isoformat(),
            'drain_timeout_used': self._shutdown_drain_timeout
        }
        
        logger.info(f"✅ TelegramBot prepared for shutdown: {shutdown_stats}")
        return shutdown_stats
    
    async def _wait_for_monitoring_drain(self, timeout: float) -> Dict[str, int]:
        """Wait for all monitoring tasks to drain naturally.
        
        Args:
            timeout: Maximum time to wait
            
        Returns:
            Dict with drain statistics
        """
        if not self.monitoring_tasks:
            logger.debug("No monitoring tasks to drain")
            return {'drained_count': 0, 'timeout_count': 0}
        
        drained_count = 0
        timeout_count = 0
        
        logger.info(f"⏳ Waiting for {len(self.monitoring_tasks)} monitoring task(s) to drain (timeout: {timeout}s)...")
        
        tasks_to_wait = []
        chat_ids = list(self.monitoring_tasks.keys())
        
        for chat_id in chat_ids:
            event = self._monitoring_drain_complete.get(chat_id)
            if event:
                tasks_to_wait.append((chat_id, event))
        
        if tasks_to_wait:
            try:
                wait_coros = [event.wait() for _, event in tasks_to_wait]
                
                done, pending = await asyncio.wait(
                    [asyncio.create_task(coro) for coro in wait_coros],
                    timeout=timeout
                )
                
                drained_count = len(done)
                timeout_count = len(pending)
                
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                
                if timeout_count > 0:
                    logger.warning(f"⚠️ {timeout_count} monitoring task(s) did not drain within {timeout}s, will be force-cancelled")
                
            except asyncio.CancelledError:
                logger.info("Drain wait was cancelled")
                raise
        
        logger.info(f"✅ Monitoring drain complete: {drained_count} drained, {timeout_count} timed out")
        return {'drained_count': drained_count, 'timeout_count': timeout_count}
    
    async def _drain_user_monitoring(self, chat_id: int, reason: str = "shutdown") -> bool:
        """Drain monitoring for a specific user with proper cleanup.
        
        This method handles per-chat cleanup:
        - Clear signal cache for the user
        - Cancel dashboard update tasks
        - Cleanup active_monitoring tracking
        - Log drain completion
        
        Args:
            chat_id: The chat ID to drain
            reason: Reason for drain (for logging)
            
        Returns:
            bool: True if successfully drained
        """
        try:
            logger.info(f"🔄 Draining monitoring for user {mask_user_id(chat_id)} (reason: {reason})...")
            
            await self._clear_signal_cache(chat_id)
            logger.debug(f"✅ Signal cache cleared for user {mask_user_id(chat_id)}")
            
            type_keys_to_remove = [k for k in self.last_signal_per_type.keys() if k.startswith(f"{chat_id}_")]
            async with self._cache_lock:
                for key in type_keys_to_remove:
                    self.last_signal_per_type.pop(key, None)
            if type_keys_to_remove:
                logger.debug(f"✅ Signal type tracking cleared for user {mask_user_id(chat_id)} ({len(type_keys_to_remove)} entries)")
            
            if chat_id in self.active_dashboards:
                dashboard_info = self.active_dashboards.get(chat_id)
                if dashboard_info:
                    task = dashboard_info.get('task')
                    if task and not task.done():
                        task.cancel()
                        try:
                            await asyncio.wait_for(task, timeout=2.0)
                        except (asyncio.CancelledError, asyncio.TimeoutError):
                            pass
                        logger.debug(f"✅ Dashboard task cancelled for user {mask_user_id(chat_id)}")
                self.active_dashboards.pop(chat_id, None)
            
            self._active_monitoring.pop(chat_id, None)
            
            if chat_id in self._monitoring_drain_complete:
                self._monitoring_drain_complete[chat_id].set()
            
            logger.info(f"✅ Monitoring drained for user {mask_user_id(chat_id)}")
            return True
            
        except (asyncio.CancelledError, asyncio.TimeoutError, KeyError, ValueError, RuntimeError) as e:
            logger.error(f"Error draining monitoring for user {mask_user_id(chat_id)}: {type(e).__name__}: {e}")
            if chat_id in self._monitoring_drain_complete:
                self._monitoring_drain_complete[chat_id].set()
            return False
    
    async def _get_user_rate_limiter(self, user_id: int) -> RateLimiter:
        """Mendapatkan atau membuat rate limiter untuk user tertentu.
        
        Thread-safe dengan lock untuk mencegah race condition saat concurrent requests.
        """
        async with self._user_rate_limiter_lock:
            if user_id not in self._user_rate_limiters:
                if len(self._user_rate_limiters) >= self.MAX_USER_RATE_LIMITERS:
                    oldest_user = min(
                        self._user_rate_limiters.items(),
                        key=lambda x: x[1].last_call_time if hasattr(x[1], 'last_call_time') else 0
                    )[0]
                    del self._user_rate_limiters[oldest_user]
                    logger.debug(f"Per-user rate limiter evicted untuk user {mask_user_id(oldest_user)}")
                
                self._user_rate_limiters[user_id] = RateLimiter(
                    max_calls=self.USER_RATE_LIMIT_MAX_CALLS,
                    time_window=self.USER_RATE_LIMIT_TIME_WINDOW,
                    name=f"User_{mask_user_id(user_id)}"
                )
                logger.debug(f"Per-user rate limiter dibuat untuk user {mask_user_id(user_id)}")
            
            return self._user_rate_limiters[user_id]
    
    async def _check_user_rate_limit(self, user_id: int) -> bool:
        """Cek apakah user masih dalam batas rate limit.
        
        Returns:
            True jika request diizinkan, False jika rate limited
        """
        try:
            user_limiter = await self._get_user_rate_limiter(user_id)
            can_proceed = await user_limiter.acquire_async(wait=False)
            
            if not can_proceed:
                logger.warning(f"⚠️ Per-user rate limit tercapai untuk user {mask_user_id(user_id)}")
                return False
            
            return True
        except (asyncio.CancelledError, RuntimeError) as e:
            logger.error(f"Error saat cek user rate limit: {e}")
            return True
    
    def _generate_signal_hash(self, user_id: int, signal_type: str, entry_price: float) -> str:
        """Generate hash unik untuk signal berdasarkan user, type, price bucket, dan timestamp bucket per menit."""
        price_bucket = round(entry_price / (self.signal_price_tolerance_pips / self.config.XAUUSD_PIP_VALUE))
        # Tambahkan timestamp bucket per menit untuk memastikan signal baru setelah 1 menit dianggap berbeda
        now = datetime.now()
        time_bucket = now.strftime('%Y%m%d%H%M')  # Format: YYYYMMDDHHMM (per menit)
        return f"{user_id}_{signal_type}_{price_bucket}_{time_bucket}"
    
    async def _check_and_set_pending(self, user_id: int, signal_type: str, entry_price: float) -> bool:
        """Check for duplicate and set pending status atomically. Returns True if signal can proceed.
        
        Enhanced anti-duplicate dengan:
        - TTL-backed cache dengan time decay untuk automatic expiry
        - Cooldown per signal type (minimal TICK_COOLDOWN_FOR_SAME_SIGNAL detik antara signal sama)
        - Minimum price movement check (SIGNAL_MINIMUM_PRICE_MOVEMENT)
        - Telemetry tracking untuk cache hits/misses
        """
        async with self._cache_lock:
            now = datetime.now()
            cache_copy = dict(self.sent_signals_cache)
            
            # Cleanup expired entries
            expired_keys = [
                k for k, v in cache_copy.items() 
                if (now - v['timestamp']).total_seconds() > self.signal_cache_expiry_seconds
            ]
            if expired_keys:
                for k in expired_keys:
                    self.sent_signals_cache.pop(k, None)
                self._cache_telemetry['expired_cleanups'] += len(expired_keys)
            
            # Cache size enforcement
            if len(self.sent_signals_cache) >= self.MAX_CACHE_SIZE:
                sorted_entries = sorted(
                    self.sent_signals_cache.items(),
                    key=lambda x: x[1]['timestamp']
                )
                entries_to_remove = len(self.sent_signals_cache) - self.MAX_CACHE_SIZE + 1
                for i in range(min(entries_to_remove, len(sorted_entries))):
                    key_to_remove = sorted_entries[i][0]
                    self.sent_signals_cache.pop(key_to_remove, None)
                self._cache_telemetry['size_enforcements'] += 1
                logger.debug(f"Cache limit enforcement: dihapus {entries_to_remove} entry terlama")
            
            # === CEK COOLDOWN PER SIGNAL TYPE ===
            # Cek apakah signal type yang sama sudah dikirim dalam periode cooldown
            type_key = f"{user_id}_{signal_type}"
            last_same_type = self.last_signal_per_type.get(type_key)
            if last_same_type:
                time_since_last = (now - last_same_type['timestamp']).total_seconds()
                cooldown = getattr(self.config, 'TICK_COOLDOWN_FOR_SAME_SIGNAL', 60)
                
                logger.debug(
                    f"📊 [COOLDOWN_CHECK] User:{mask_user_id(user_id)} Type:{signal_type} | "
                    f"TimeSinceLast:{time_since_last:.1f}s | Cooldown:{cooldown}s | "
                    f"GlobalCooldown:{self.global_signal_cooldown}s | TickThrottle:{self.tick_throttle_seconds}s"
                )
                
                if time_since_last < cooldown:
                    logger.info(
                        f"🚫 [BLOCKED:COOLDOWN] Signal {signal_type} diblokir: cooldown per type belum habis "
                        f"({time_since_last:.1f}s < {cooldown}s) | User:{mask_user_id(user_id)}"
                    )
                    self._cache_telemetry['hits'] += 1
                    return False
                
                # === CEK MINIMUM PRICE MOVEMENT ===
                last_price = last_same_type.get('entry_price', 0)
                min_movement = getattr(self.config, 'SIGNAL_MINIMUM_PRICE_MOVEMENT', 0.50)
                price_diff = abs(entry_price - last_price)
                
                if price_diff < min_movement:
                    logger.info(
                        f"🚫 [BLOCKED:PRICE_MOVEMENT] Signal {signal_type} diblokir: pergerakan harga terlalu kecil "
                        f"(${price_diff:.2f} < ${min_movement:.2f}) | Last:${last_price:.2f} → New:${entry_price:.2f} | "
                        f"User:{mask_user_id(user_id)}"
                    )
                    self._cache_telemetry['hits'] += 1
                    return False
            
            # === CEK HASH CACHE (duplicate exact signal) ===
            signal_hash = self._generate_signal_hash(user_id, signal_type, entry_price)
            
            cached = self.sent_signals_cache.get(signal_hash)
            if cached:
                status = cached.get('status', 'confirmed')
                time_since = (now - cached['timestamp']).total_seconds()
                self._cache_telemetry['hits'] += 1
                logger.info(
                    f"🚫 [BLOCKED:DUPLICATE_CACHE] Signal duplikat diblokir | Hash:{signal_hash} | "
                    f"Status:{status} | TimeSince:{time_since:.1f}s | User:{mask_user_id(user_id)}"
                )
                return False
            
            # === SIGNAL DIIZINKAN - Set pending dan update tracking ===
            self._cache_telemetry['misses'] += 1
            self._cache_telemetry['pending_set'] += 1
            
            # Update cache dengan signal baru
            self.sent_signals_cache[signal_hash] = {
                'status': 'pending',
                'timestamp': now,
                'user_id': user_id,
                'signal_type': signal_type,
                'entry_price': entry_price
            }
            
            # Update tracking per signal type
            self.last_signal_per_type[type_key] = {
                'timestamp': now,
                'entry_price': entry_price,
                'signal_type': signal_type
            }
            
            logger.debug(f"Cache MISS - Signal ditandai pending: {signal_hash}")
            return True
    
    async def _confirm_signal_sent(self, user_id: int, signal_type: str, entry_price: float):
        """Confirm signal was sent successfully - upgrade from pending to confirmed.
        
        Layered cache transition: pending → confirmed
        """
        async with self._cache_lock:
            signal_hash = self._generate_signal_hash(user_id, signal_type, entry_price)
            if signal_hash in self.sent_signals_cache:
                self.sent_signals_cache[signal_hash]['status'] = 'confirmed'
                self.sent_signals_cache[signal_hash]['timestamp'] = datetime.now()
                self.sent_signals_cache[signal_hash]['confirmed_at'] = datetime.now()
                self._cache_telemetry['confirmed'] += 1
                logger.debug(f"Signal confirmed (pending→confirmed): {signal_hash}")
    
    async def _rollback_signal_cache(self, user_id: int, signal_type: str, entry_price: float):
        """Remove pending signal entry on failure.
        
        Cleans up pending entries when signal send fails.
        """
        async with self._cache_lock:
            signal_hash = self._generate_signal_hash(user_id, signal_type, entry_price)
            removed = self.sent_signals_cache.pop(signal_hash, None)
            if removed:
                self._cache_telemetry['rollbacks'] += 1
                logger.debug(f"Signal cache rolled back: {signal_hash}")
    
    async def _clear_signal_cache(self, user_id: Optional[int] = None):
        """Clear signal cache for user or all users."""
        async with self._cache_lock:
            if user_id is not None:
                cache_keys = list(self.sent_signals_cache.keys())
                for k in cache_keys:
                    if k.startswith(f"{user_id}_"):
                        self.sent_signals_cache.pop(k, None)
                logger.debug(f"Cleared signal cache for user {user_id}")
            else:
                self.sent_signals_cache.clear()
                logger.debug("Cleared all signal cache")
    
    async def _handle_telegram_error(self, update, context):
        """Global error handler untuk Telegram updates - log and swallow"""
        error = context.error
        if error is None:
            return
        
        if isinstance(error, asyncio.TimeoutError):
            logger.warning("Telegram update timeout - continuing")
        elif isinstance(error, TimedOut):
            logger.warning("Telegram API timeout - continuing")
        elif isinstance(error, NetworkError):
            logger.warning(f"Telegram network error: {error} - continuing")
        elif isinstance(error, TelegramError):
            logger.error(f"Telegram API error: {error}")
        else:
            logger.error(f"Unexpected error in update handler: {type(error).__name__}: {error}")
    
    async def _handle_forbidden_error(self, chat_id: int, error: Forbidden):
        """Handle Forbidden error - user blocked bot or chat inaccessible.
        
        Actions:
        - Remove chat from monitoring list
        - Cancel monitoring task for this chat
        - Clear any pending signals for this user
        - Log the event for analytics
        
        Args:
            chat_id: The chat ID that generated the error
            error: The Forbidden exception
        """
        logger.warning(f"🚫 Forbidden error for chat {mask_user_id(chat_id)}: {error}")
        
        if chat_id in self.monitoring_chats:
            self.monitoring_chats.remove(chat_id)
            logger.info(f"Removed chat {mask_user_id(chat_id)} from monitoring list (user blocked bot)")
        
        task = self.monitoring_tasks.pop(chat_id, None)
        if task and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            logger.info(f"Cancelled monitoring task for blocked chat {mask_user_id(chat_id)}")
        
        await self._clear_signal_cache(chat_id)
        
        if self.signal_session_manager:
            try:
                await self.signal_session_manager.end_session(
                    chat_id, 
                    reason='user_blocked'
                )
            except (TelegramError, asyncio.TimeoutError, ValueError, AttributeError, TypeError) as e:
                logger.debug(f"Could not end session for blocked user: {e}")
        
        if self.alert_system:
            try:
                await self.alert_system.send_alert(
                    level='warning',
                    title='User Blocked Bot',
                    message=f"User {mask_user_id(chat_id)} telah memblokir bot atau chat tidak dapat diakses",
                    context={'chat_id': chat_id, 'error': str(error)}
                )
            except (TelegramError, asyncio.TimeoutError, ConnectionError) as e:
                logger.debug(f"Could not send alert for blocked user: {e}")
    
    async def _handle_chat_migrated(self, old_chat_id: int, error: ChatMigrated) -> Optional[int]:
        """Handle ChatMigrated error - group migrated to supergroup.
        
        Actions:
        - Extract new chat ID from error
        - Update monitoring list with new chat ID
        - Update any active sessions/positions with new chat ID
        - Return new chat ID for caller to retry operation
        
        Args:
            old_chat_id: The old chat ID that triggered migration
            error: The ChatMigrated exception containing new_chat_id
            
        Returns:
            int or None: The new chat ID if available
        """
        new_chat_id = getattr(error, 'new_chat_id', None)
        logger.warning(f"📤 Chat migrated: {mask_user_id(old_chat_id)} -> {mask_user_id(new_chat_id) if new_chat_id else 'unknown'}")
        
        if not new_chat_id:
            logger.error(f"ChatMigrated error without new_chat_id for {mask_user_id(old_chat_id)}")
            return None
        
        if old_chat_id in self.monitoring_chats:
            idx = self.monitoring_chats.index(old_chat_id)
            self.monitoring_chats[idx] = new_chat_id
            logger.info(f"Updated monitoring list: {mask_user_id(old_chat_id)} -> {mask_user_id(new_chat_id)}")
        
        old_task = self.monitoring_tasks.pop(old_chat_id, None)
        if old_task:
            if not old_task.done():
                old_task.cancel()
                try:
                    await asyncio.wait_for(old_task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            new_task = asyncio.create_task(self._monitoring_loop(new_chat_id))
            self.monitoring_tasks[new_chat_id] = new_task
            logger.info(f"Created new monitoring task for migrated chat {mask_user_id(new_chat_id)}")
        
        if self.signal_session_manager:
            try:
                await self.signal_session_manager.migrate_user_sessions(old_chat_id, new_chat_id)
            except (TelegramError, asyncio.TimeoutError, ValueError, KeyError) as e:
                logger.warning(f"Could not migrate sessions for chat migration: {e}")
        
        if self.alert_system:
            try:
                await self.alert_system.send_alert(
                    level='info',
                    title='Chat Migrated',
                    message=f"Chat {mask_user_id(old_chat_id)} migrated to {mask_user_id(new_chat_id)}",
                    context={'old_chat_id': old_chat_id, 'new_chat_id': new_chat_id}
                )
            except (TelegramError, asyncio.TimeoutError, ConnectionError) as e:
                logger.debug(f"Could not send migration alert: {e}")
        
        return new_chat_id
    
    async def _handle_unauthorized_error(self, error: InvalidToken):
        """Handle InvalidToken error - invalid or revoked bot token.
        
        This is a CRITICAL error that requires immediate attention.
        
        Actions:
        - Log critical error
        - Send critical alert (if possible through alternative channels)
        - Stop all monitoring and cleanup
        - Set shutdown flag
        
        Args:
            error: The InvalidToken exception
        """
        logger.critical(f"🔴 CRITICAL: InvalidToken error - Bot token invalid or revoked: {error}")
        
        self._is_shutting_down = True
        
        if self.alert_system:
            try:
                await self.alert_system.send_critical_alert(
                    title='🔴 BOT TOKEN INVALID',
                    message=f"Bot token tidak valid atau telah di-revoke. Bot akan berhenti.\nError: {error}",
                    context={'error': str(error), 'timestamp': datetime.now().isoformat()}
                )
            except (TelegramError, asyncio.TimeoutError, ConnectionError, OSError) as e:
                logger.error(f"Could not send critical alert for unauthorized error: {e}")
        
        if self.error_handler:
            try:
                await self.error_handler.handle_critical_error(
                    error_type='UNAUTHORIZED',
                    error=error,
                    context={'action': 'bot_shutdown_required'}
                )
            except (TelegramError, asyncio.TimeoutError, ValueError, AttributeError) as e:
                logger.error(f"Error handler failed for unauthorized error: {e}")
        
        logger.critical("🛑 Initiating emergency shutdown due to unauthorized error...")
        await self.stop_background_cleanup_tasks()
    
    async def _handle_conflict_error(self, error: Conflict):
        """Handle Conflict error - multiple bot instances detected.
        
        This is a CRITICAL error indicating duplicate bot instances.
        
        Actions:
        - Log critical error with instance info
        - Send alert to admins
        - Gracefully stop this instance to avoid conflicts
        
        Args:
            error: The Conflict exception
        """
        logger.critical(f"🔴 CRITICAL: Conflict error - Multiple bot instances detected: {error}")
        
        if self.alert_system:
            try:
                await self.alert_system.send_critical_alert(
                    title='🔴 BOT INSTANCE CONFLICT',
                    message=f"Multiple bot instances detected! Ini dapat menyebabkan behavior yang tidak konsisten.\nError: {error}",
                    context={
                        'error': str(error),
                        'timestamp': datetime.now().isoformat(),
                        'instance_lock_file': self.instance_lock_file
                    }
                )
            except (TelegramError, asyncio.TimeoutError, ConnectionError, OSError) as e:
                logger.error(f"Could not send critical alert for conflict error: {e}")
        
        if self.error_handler:
            try:
                await self.error_handler.handle_critical_error(
                    error_type='CONFLICT',
                    error=error,
                    context={'action': 'check_other_instances'}
                )
            except (TelegramError, asyncio.TimeoutError, ValueError, AttributeError) as e:
                logger.error(f"Error handler failed for conflict error: {e}")
        
        logger.critical("🛑 This instance will stop to avoid conflicts...")
        self._is_shutting_down = True
        await self.stop_background_cleanup_tasks()
    
    async def _handle_bad_request(self, chat_id: int, error: BadRequest, context: str = ""):
        """Handle BadRequest error - invalid request parameters.
        
        This error should NOT be retried as it indicates a programming error
        or invalid input that won't change on retry.
        
        Actions:
        - Log detailed error with context
        - Track for debugging
        - Optionally notify developers
        
        Args:
            chat_id: The chat ID involved (if any)
            error: The BadRequest exception
            context: Additional context about what operation failed
        """
        error_message = str(error)
        logger.error(f"❌ BadRequest for chat {mask_user_id(chat_id)}: {error_message} (context: {context})")
        
        known_bad_requests = {
            'message is not modified': 'info',
            'message to edit not found': 'warning',
            'chat not found': 'warning',
            'user not found': 'warning',
            'message text is empty': 'error',
            'can\'t parse entities': 'error',
            'wrong file identifier': 'error',
        }
        
        log_level = 'error'
        for pattern, level in known_bad_requests.items():
            if pattern.lower() in error_message.lower():
                log_level = level
                break
        
        if log_level == 'info':
            logger.info(f"BadRequest (expected): {error_message}")
        elif log_level == 'warning':
            logger.warning(f"BadRequest (expected): {error_message}")
        else:
            if self.error_handler:
                try:
                    await self.error_handler.track_error(
                        error_type='BAD_REQUEST',
                        error=error,
                        context={'chat_id': chat_id, 'operation': context}
                    )
                except (TelegramError, asyncio.TimeoutError, ValueError, AttributeError) as e:
                    logger.debug(f"Could not track bad request error: {e}")
    
    async def start_background_cleanup_tasks(self):
        """Mulai background tasks untuk cleanup cache, dashboards, pending charts, dan auto-optimization"""
        self._cleanup_tasks_running = True
        
        if self._cache_cleanup_task is None or self._cache_cleanup_task.done():
            self._cache_cleanup_task = asyncio.create_task(self._signal_cache_cleanup_loop())
            logger.info("✅ Signal cache cleanup background task started")
        
        if self._dashboard_cleanup_task is None or self._dashboard_cleanup_task.done():
            self._dashboard_cleanup_task = asyncio.create_task(self._dashboard_cleanup_loop())
            logger.info("✅ Dashboard cleanup background task started")
        
        if self._chart_cleanup_task is None or self._chart_cleanup_task.done():
            self._chart_cleanup_task = asyncio.create_task(self._pending_chart_cleanup_loop())
            logger.info("✅ Pending chart cleanup background task started")
        
        if self._aggressive_chart_cleanup_task is None or self._aggressive_chart_cleanup_task.done():
            self._aggressive_chart_cleanup_task = asyncio.create_task(self._aggressive_chart_cleanup_loop())
            logger.info(f"✅ Aggressive chart cleanup background task started (interval: {self._chart_cleanup_interval_minutes}min)")
        
        if self.auto_optimizer and (self._auto_optimization_task is None or self._auto_optimization_task.done()):
            self._auto_optimization_task = asyncio.create_task(self._auto_optimization_loop())
            logger.info("✅ Auto-optimization background task started")
    
    async def stop_background_cleanup_tasks(self):
        """Stop background cleanup tasks and all monitoring tasks"""
        self._cleanup_tasks_running = False
        self._is_shutting_down = True
        
        if self._cache_cleanup_task and not self._cache_cleanup_task.done():
            self._cache_cleanup_task.cancel()
            try:
                await asyncio.wait_for(self._cache_cleanup_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._cache_cleanup_task = None
            logger.info("Signal cache cleanup task stopped")
        
        if self._dashboard_cleanup_task and not self._dashboard_cleanup_task.done():
            self._dashboard_cleanup_task.cancel()
            try:
                await asyncio.wait_for(self._dashboard_cleanup_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._dashboard_cleanup_task = None
            logger.info("Dashboard cleanup task stopped")
        
        if self._chart_cleanup_task and not self._chart_cleanup_task.done():
            self._chart_cleanup_task.cancel()
            try:
                await asyncio.wait_for(self._chart_cleanup_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._chart_cleanup_task = None
            logger.info("Pending chart cleanup task stopped")
        
        if self._aggressive_chart_cleanup_task and not self._aggressive_chart_cleanup_task.done():
            self._aggressive_chart_cleanup_task.cancel()
            try:
                await asyncio.wait_for(self._aggressive_chart_cleanup_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._aggressive_chart_cleanup_task = None
            logger.info("Aggressive chart cleanup task stopped")
        
        if self._auto_optimization_task and not self._auto_optimization_task.done():
            self._auto_optimization_task.cancel()
            try:
                await asyncio.wait_for(self._auto_optimization_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._auto_optimization_task = None
            logger.info("Auto-optimization task stopped")
        
        await self._cleanup_all_pending_charts()
        
        await self._cancel_all_monitoring_tasks()
        
        await self._cancel_all_dashboard_tasks()
        
        logger.info("All background and monitoring tasks stopped")
    
    async def _cancel_all_monitoring_tasks(self, timeout: float = 10.0, graceful: bool = True):
        """Cancel all monitoring tasks with proper cleanup using graceful drain pattern.
        
        Graceful cancellation flow:
        1. Set shutdown flag (_is_shutting_down = True) 
        2. Wait for monitoring loops to finish current iteration naturally
        3. Only force-cancel if timeout exceeded
        4. Run per-user cleanup (signal cache, dashboards, etc)
        
        Args:
            timeout: Maximum time to wait for graceful drain
            graceful: If True, wait for loops to drain; if False, immediate cancel
        """
        if not self.monitoring_tasks:
            logger.debug("No monitoring tasks to cancel")
            return
        
        self.monitoring = False
        task_count = len(self.monitoring_tasks)
        logger.info(f"🛑 Cancelling {task_count} monitoring task(s) (graceful={graceful})...")
        
        if graceful and not self._is_shutting_down:
            self._is_shutting_down = True
            logger.info("✅ Shutdown flag set - monitoring loops will exit after current iteration")
        
        chat_ids = list(self.monitoring_tasks.keys())
        for chat_id in chat_ids:
            if chat_id not in self._monitoring_drain_complete:
                self._monitoring_drain_complete[chat_id] = asyncio.Event()
        
        if graceful:
            drained_count = 0
            force_cancelled_count = 0
            
            try:
                events_to_wait: List[asyncio.Event] = []
                for cid in chat_ids:
                    event = self._monitoring_drain_complete.get(cid)
                    if event is not None:
                        events_to_wait.append(event)
                
                if events_to_wait:
                    wait_tasks = [asyncio.create_task(ev.wait()) for ev in events_to_wait]
                    done, pending = await asyncio.wait(wait_tasks, timeout=timeout)
                    drained_count = len(done)
                    
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                    
                    if pending:
                        force_cancelled_count = len(pending)
                        logger.warning(f"⚠️ {force_cancelled_count} task(s) did not drain in time, force-cancelling...")
                        
                        for chat_id, task in list(self.monitoring_tasks.items()):
                            if task and not task.done():
                                task.cancel()
                                logger.debug(f"Force-cancelled monitoring task for chat {mask_user_id(chat_id)}")
                        
                        tasks_to_cancel = [t for t in self.monitoring_tasks.values() if t and not t.done()]
                        if tasks_to_cancel:
                            try:
                                await asyncio.wait_for(
                                    asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                                    timeout=5.0
                                )
                            except (asyncio.TimeoutError, asyncio.CancelledError):
                                pass
                
                logger.info(f"✅ Monitoring tasks handled: {drained_count} drained gracefully, {force_cancelled_count} force-cancelled")
                
            except asyncio.CancelledError:
                logger.info("Monitoring task cancellation was itself cancelled")
                raise
        else:
            tasks_to_cancel = []
            for chat_id, task in list(self.monitoring_tasks.items()):
                if task and not task.done():
                    task.cancel()
                    tasks_to_cancel.append(task)
                    logger.debug(f"Immediately cancelled monitoring task for chat {mask_user_id(chat_id)}")
            
            if tasks_to_cancel:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                        timeout=timeout
                    )
                    logger.info(f"✅ All {len(tasks_to_cancel)} monitoring tasks cancelled immediately")
                except asyncio.TimeoutError:
                    logger.warning(f"Some monitoring tasks did not complete within {timeout}s timeout")
                except asyncio.CancelledError:
                    logger.debug("Monitoring task cancellation was itself cancelled")
        
        self.monitoring_tasks.clear()
        self.monitoring_chats.clear()
        self._active_monitoring.clear()
        self._monitoring_drain_complete.clear()
        
        logger.info("✅ Monitoring tasks cleanup completed")
    
    async def _cancel_all_dashboard_tasks(self, timeout: float = 5.0):
        """Cancel all dashboard update tasks"""
        if not self.active_dashboards:
            logger.debug("No dashboard tasks to cancel")
            return
        
        dashboard_count = len(self.active_dashboards)
        logger.info(f"Cancelling {dashboard_count} dashboard tasks...")
        
        tasks_to_cancel = []
        for user_id, dash_info in list(self.active_dashboards.items()):
            task = dash_info.get('task')
            if task and not task.done():
                task.cancel()
                tasks_to_cancel.append(task)
        
        if tasks_to_cancel:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                    timeout=timeout
                )
                logger.info(f"✅ All {len(tasks_to_cancel)} dashboard tasks cancelled successfully")
            except asyncio.TimeoutError:
                logger.warning(f"Some dashboard tasks did not complete within {timeout}s timeout")
            except asyncio.CancelledError:
                pass
        
        self.active_dashboards.clear()
    
    async def _auto_optimization_loop(self):
        """Background task untuk auto-optimization parameter strategy"""
        optimization_interval = 3600
        
        try:
            while self._cleanup_tasks_running:
                await asyncio.sleep(optimization_interval)
                
                if not self._cleanup_tasks_running:
                    break
                
                if not self.auto_optimizer:
                    continue
                
                try:
                    should_optimize, optimize_reason = self.auto_optimizer.should_run_optimization()
                    
                    if should_optimize:
                        logger.info(f"🔧 Starting auto-optimization check: {optimize_reason}")
                        
                        optimization_result = self.auto_optimizer.run_optimization()
                        
                        if optimization_result:
                            adjustments = optimization_result.adjustments if hasattr(optimization_result, 'adjustments') else []
                            if adjustments:
                                logger.info(f"✅ Auto-optimization completed: {len(adjustments)} parameter(s) updated")
                                logger.debug(f"Optimization status: {optimization_result.status}")
                            else:
                                logger.debug("Auto-optimization: no parameter changes needed")
                        else:
                            logger.debug("Auto-optimization: no optimization result")
                    else:
                        logger.debug(f"Auto-optimization: {optimize_reason}")
                        
                except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
                    logger.error(f"Error in auto-optimization: {type(e).__name__}: {e}")
                    
        except asyncio.CancelledError:
            logger.info("Auto-optimization loop cancelled")
    
    async def _signal_cache_cleanup_loop(self):
        """Background task untuk cleanup expired signal cache entries"""
        cleanup_interval = 60
        
        try:
            while self._cleanup_tasks_running:
                await asyncio.sleep(cleanup_interval)
                
                if not self._cleanup_tasks_running:
                    break
                
                try:
                    cleaned = await self._cleanup_expired_cache_entries()
                    if cleaned > 0:
                        logger.debug(f"Signal cache cleanup: removed {cleaned} expired entries")
                    
                    restarted = await self._check_and_restart_dead_monitoring_tasks()
                    if restarted > 0:
                        logger.info(f"🔄 Monitoring watchdog: restarted {restarted} dead monitoring task(s)")
                except (asyncio.TimeoutError, KeyError, ValueError, RuntimeError) as e:
                    logger.error(f"Error in signal cache cleanup: {e}")
                    
        except asyncio.CancelledError:
            logger.info("Signal cache cleanup loop cancelled")
    
    async def _check_and_restart_dead_monitoring_tasks(self) -> int:
        """Watchdog: check and restart any dead monitoring tasks.
        
        This ensures monitoring continues even if a task unexpectedly dies.
        Returns the number of tasks restarted.
        """
        if self._is_shutting_down:
            return 0
        
        restarted_count = 0
        
        for chat_id in list(self.monitoring_chats):
            task = self.monitoring_tasks.get(chat_id)
            
            if task is None or task.done():
                if self._is_shutting_down:
                    break
                    
                logger.warning(f"🔄 [WATCHDOG] Dead monitoring task detected for chat {mask_user_id(chat_id)} - restarting...")
                
                try:
                    new_task = asyncio.create_task(self._monitoring_loop(chat_id))
                    new_task.add_done_callback(
                        lambda t, cid=chat_id: self._on_monitoring_task_done(cid, t)
                    )
                    self.monitoring_tasks[chat_id] = new_task
                    restarted_count += 1
                    logger.info(f"✅ [WATCHDOG] Monitoring task restarted for chat {mask_user_id(chat_id)}")
                except Exception as e:
                    logger.error(f"❌ [WATCHDOG] Failed to restart monitoring for chat {mask_user_id(chat_id)}: {e}")
        
        return restarted_count
    
    async def _cleanup_expired_cache_entries(self) -> int:
        """Cleanup expired entries dari signal cache dengan time decay.
        
        Implements TTL-backed cache with time decay:
        - Pending entries expire faster (60s) to prevent stuck entries
        - Confirmed entries use full TTL (120s) for proper duplicate prevention
        
        Returns:
            int: Number of entries cleaned up
        """
        async with self._cache_lock:
            now = datetime.now()
            pending_ttl = 60
            confirmed_ttl = self.signal_cache_expiry_seconds
            
            expired_keys = []
            for k, v in self.sent_signals_cache.items():
                age_seconds = (now - v['timestamp']).total_seconds()
                status = v.get('status', 'confirmed')
                
                if status == 'pending' and age_seconds > pending_ttl:
                    expired_keys.append(k)
                    logger.debug(f"Time decay: pending entry {k} expired after {age_seconds:.1f}s")
                elif status == 'confirmed' and age_seconds > confirmed_ttl:
                    expired_keys.append(k)
            
            for k in expired_keys:
                self.sent_signals_cache.pop(k, None)
            
            if expired_keys:
                self._cache_telemetry['expired_cleanups'] += len(expired_keys)
                self._cache_telemetry['last_cleanup_time'] = now
                self._cache_telemetry['last_cleanup_count'] = len(expired_keys)
            
            return len(expired_keys)
    
    async def _dashboard_cleanup_loop(self):
        """Background task untuk cleanup dead/stale dashboards"""
        cleanup_interval = 30
        max_dashboard_age_seconds = 3600
        
        try:
            while self._cleanup_tasks_running:
                await asyncio.sleep(cleanup_interval)
                
                if not self._cleanup_tasks_running:
                    break
                
                try:
                    cleaned = await self._cleanup_stale_dashboards(max_dashboard_age_seconds)
                    if cleaned > 0:
                        logger.info(f"Dashboard cleanup: removed {cleaned} stale entries")
                except (asyncio.TimeoutError, KeyError, ValueError, RuntimeError) as e:
                    logger.error(f"Error in dashboard cleanup: {e}")
                    
        except asyncio.CancelledError:
            logger.info("Dashboard cleanup loop cancelled")
    
    async def _cleanup_stale_dashboards(self, max_age_seconds: int = 3600) -> int:
        """Cleanup stale dashboard entries"""
        cleaned = 0
        stale_users = []
        
        async with self._dashboard_lock:
            now = datetime.now()
            
            for user_id, dashboard_info in list(self.active_dashboards.items()):
                try:
                    task = dashboard_info.get('task')
                    created_at = dashboard_info.get('created_at', now)
                    
                    age_seconds = (now - created_at).total_seconds() if isinstance(created_at, datetime) else 0
                    
                    is_stale = False
                    if task is None:
                        is_stale = True
                    elif task.done():
                        is_stale = True
                    elif age_seconds > max_age_seconds:
                        is_stale = True
                        if not task.done():
                            task.cancel()
                    
                    if is_stale:
                        stale_users.append(user_id)
                        
                except (asyncio.CancelledError, asyncio.InvalidStateError, KeyError, TypeError) as e:
                    logger.error(f"Error checking dashboard for user {user_id}: {e}")
                    stale_users.append(user_id)
            
            for user_id in stale_users:
                self.active_dashboards.pop(user_id, None)
                cleaned += 1
        
        return cleaned
    
    def get_cache_stats(self) -> Dict:
        """Dapatkan statistik cache untuk monitoring dengan telemetry data.
        
        Returns comprehensive cache statistics including:
        - Cache size and usage
        - Hit/miss ratios
        - Pending vs confirmed entries breakdown
        - Dashboard and monitoring stats
        - Pending charts info
        """
        try:
            pending_count = sum(1 for v in self.sent_signals_cache.values() if v.get('status') == 'pending')
            confirmed_count = sum(1 for v in self.sent_signals_cache.values() if v.get('status') == 'confirmed')
            
            total_lookups = self._cache_telemetry['hits'] + self._cache_telemetry['misses']
            hit_rate = (self._cache_telemetry['hits'] / total_lookups * 100) if total_lookups > 0 else 0.0
            
            return {
                'signal_cache_size': len(self.sent_signals_cache),
                'signal_cache_max': self.MAX_CACHE_SIZE,
                'signal_cache_usage_pct': (len(self.sent_signals_cache) / self.MAX_CACHE_SIZE * 100) if self.MAX_CACHE_SIZE > 0 else 0,
                'pending_entries': pending_count,
                'confirmed_entries': confirmed_count,
                'active_dashboards': len(self.active_dashboards),
                'dashboards_max': self.MAX_DASHBOARDS,
                'dashboards_usage_pct': (len(self.active_dashboards) / self.MAX_DASHBOARDS * 100) if self.MAX_DASHBOARDS > 0 else 0,
                'monitoring_chats': len(self.monitoring_chats),
                'monitoring_chats_max': self.MAX_MONITORING_CHATS,
                'monitoring_tasks': len(self.monitoring_tasks),
                'pending_charts': len(self._pending_charts),
                'cache_expiry_seconds': self.signal_cache_expiry_seconds,
                'is_shutting_down': self._is_shutting_down,
                'cleanup_tasks_running': self._cleanup_tasks_running,
                'telemetry': {
                    'cache_hits': self._cache_telemetry['hits'],
                    'cache_misses': self._cache_telemetry['misses'],
                    'hit_rate_pct': round(hit_rate, 2),
                    'total_lookups': total_lookups,
                    'pending_set': self._cache_telemetry['pending_set'],
                    'confirmed': self._cache_telemetry['confirmed'],
                    'rollbacks': self._cache_telemetry['rollbacks'],
                    'expired_cleanups': self._cache_telemetry['expired_cleanups'],
                    'size_enforcements': self._cache_telemetry['size_enforcements'],
                    'last_cleanup_time': self._cache_telemetry['last_cleanup_time'].isoformat() if self._cache_telemetry['last_cleanup_time'] else None,
                    'last_cleanup_count': self._cache_telemetry['last_cleanup_count'],
                }
            }
        except AttributeError as e:
            logger.warning(f"Attribute error getting cache stats (bot may not be fully initialized): {e}")
            return {'error': str(e), 'initialized': False}
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
            logger.error(f"Unexpected error getting cache stats: {type(e).__name__}: {e}")
            return {'error': str(e)}
    
    def _get_cache_stats(self) -> Dict:
        """Alias untuk get_cache_stats() untuk backward compatibility."""
        return self.get_cache_stats()
    
    async def register_pending_chart(self, user_id: int, chart_path: str, signal_type: Optional[str] = None):
        """Register a pending chart for cleanup tracking.
        
        Args:
            user_id: User ID associated with the chart
            chart_path: Path to the chart file
            signal_type: Type of signal (BUY/SELL)
        """
        async with self._chart_cleanup_lock:
            self._pending_charts[user_id] = {
                'chart_path': chart_path,
                'signal_type': signal_type if signal_type is not None else '',
                'created_at': datetime.now(),
                'status': 'pending'
            }
            logger.debug(f"Registered pending chart for user {mask_user_id(user_id)}: {chart_path}")
    
    async def confirm_chart_sent(self, user_id: int):
        """Confirm chart was sent successfully, update status.
        
        Args:
            user_id: User ID whose chart was sent
        """
        async with self._chart_cleanup_lock:
            if user_id in self._pending_charts:
                self._pending_charts[user_id]['status'] = 'sent'
                self._pending_charts[user_id]['sent_at'] = datetime.now()
                logger.debug(f"Chart confirmed sent for user {mask_user_id(user_id)}")
    
    async def evict_pending_chart(self, user_id: int, reason: str = "manual"):
        """Evict and cleanup a pending chart for a user.
        
        Args:
            user_id: User ID whose chart should be evicted
            reason: Reason for eviction
            
        Returns:
            bool: True if chart was evicted, False otherwise
        """
        import os
        
        chart_info = None
        async with self._chart_cleanup_lock:
            chart_info = self._pending_charts.pop(user_id, None)
        
        if chart_info:
            chart_path = chart_info.get('chart_path')
            if chart_path and os.path.exists(chart_path):
                try:
                    os.remove(chart_path)
                    logger.info(f"🗑️ Evicted pending chart for user {mask_user_id(user_id)}: {chart_path} (reason: {reason})")
                    
                    for callback in self._chart_eviction_callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(user_id, chart_path, reason)
                            else:
                                callback(user_id, chart_path, reason)
                        except (TelegramError, asyncio.TimeoutError, ValueError, TypeError, RuntimeError) as e:
                            logger.error(f"Error in chart eviction callback: {e}")
                    
                    return True
                except FileNotFoundError:
                    logger.debug(f"Chart already deleted: {chart_path}")
                except (PermissionError, OSError, IOError) as e:
                    logger.warning(f"Failed to evict chart {chart_path}: {e}")
            return True
        return False
    
    def register_chart_eviction_callback(self, callback: Callable):
        """Register a callback to be called when a chart is evicted.
        
        Args:
            callback: Callback function (can be async) with signature (user_id, chart_path, reason)
        """
        self._chart_eviction_callbacks.append(callback)
        logger.debug(f"Registered chart eviction callback: {callback.__name__}")
    
    async def _pending_chart_cleanup_loop(self):
        """Background task for cleaning up stale pending charts.
        
        Runs periodically to cleanup charts that were never sent or confirmed.
        Uses time decay: charts older than TTL are automatically evicted.
        """
        cleanup_interval = 30
        pending_chart_ttl_seconds = 120
        
        try:
            while self._cleanup_tasks_running:
                await asyncio.sleep(cleanup_interval)
                
                if not self._cleanup_tasks_running:
                    break
                
                try:
                    cleaned = await self._cleanup_stale_pending_charts(pending_chart_ttl_seconds)
                    if cleaned > 0:
                        logger.info(f"Pending chart cleanup: evicted {cleaned} stale charts")
                except (asyncio.TimeoutError, OSError, IOError, KeyError, ValueError) as e:
                    logger.error(f"Error in pending chart cleanup: {e}")
                    
        except asyncio.CancelledError:
            logger.info("Pending chart cleanup loop cancelled")
    
    async def _aggressive_chart_cleanup_loop(self):
        """Background task for aggressive chart cleanup using ChartGenerator.
        
        Runs every CHART_CLEANUP_INTERVAL_MINUTES (default 5 min) to:
        - Clean up old charts from filesystem
        - Evict expired charts from LRU cache
        - Free up disk space and memory
        
        This complements the pending chart cleanup with more thorough 
        filesystem scanning and cleanup.
        """
        cleanup_interval_seconds = self._chart_cleanup_interval_minutes * 60
        expiry_minutes = getattr(self.config, 'CHART_EXPIRY_MINUTES', 15)
        
        logger.info(f"🧹 Aggressive chart cleanup loop started (interval: {self._chart_cleanup_interval_minutes}min, expiry: {expiry_minutes}min)")
        
        try:
            while self._cleanup_tasks_running:
                await asyncio.sleep(cleanup_interval_seconds)
                
                if not self._cleanup_tasks_running:
                    break
                
                try:
                    if self.chart_generator and hasattr(self.chart_generator, 'cleanup_charts_aggressive_async'):
                        cleaned = await self.chart_generator.cleanup_charts_aggressive_async(max_age_minutes=expiry_minutes)
                        
                        if cleaned > 0:
                            logger.info(f"🧹 Aggressive cleanup: deleted {cleaned} expired chart(s)")
                        
                        stats = self.chart_generator.get_stats()
                        lru_stats = stats.get('lru_cache', {})
                        current_memory = stats.get('current_memory_mb', 0)
                        
                        logger.debug(
                            f"Chart cleanup stats - tracked: {stats.get('tracked_charts', 0)}, "
                            f"LRU size: {lru_stats.get('size', 0)}/{lru_stats.get('max_size', 0)}, "
                            f"memory: {current_memory:.1f}MB, "
                            f"total_deleted: {stats.get('cleanup_stats', {}).get('total_deleted', 0)}"
                        )
                    else:
                        logger.debug("ChartGenerator does not have aggressive cleanup method")
                        
                except (asyncio.TimeoutError, OSError, IOError, AttributeError, KeyError, ValueError) as e:
                    logger.error(f"Error in aggressive chart cleanup: {e}")
                    
        except asyncio.CancelledError:
            logger.info("Aggressive chart cleanup loop cancelled")
    
    async def _cleanup_stale_pending_charts(self, max_age_seconds: int = 120) -> int:
        """Cleanup stale pending charts with time decay.
        
        Args:
            max_age_seconds: Maximum age in seconds before a chart is considered stale
            
        Returns:
            int: Number of charts cleaned up
        """
        import os
        
        stale_users = []
        charts_to_cleanup = []
        
        async with self._chart_cleanup_lock:
            now = datetime.now()
            
            for user_id, chart_info in list(self._pending_charts.items()):
                created_at = chart_info.get('created_at', now)
                age_seconds = (now - created_at).total_seconds()
                status = chart_info.get('status', 'pending')
                
                is_stale = False
                if status == 'pending' and age_seconds > max_age_seconds:
                    is_stale = True
                    logger.debug(f"Stale pending chart for user {mask_user_id(user_id)}: {age_seconds:.1f}s old")
                elif status == 'sent' and age_seconds > (max_age_seconds * 2):
                    is_stale = True
                
                if is_stale:
                    stale_users.append(user_id)
                    charts_to_cleanup.append((user_id, chart_info.get('chart_path')))
            
            for user_id in stale_users:
                self._pending_charts.pop(user_id, None)
        
        cleaned = 0
        for user_id, chart_path in charts_to_cleanup:
            if chart_path and os.path.exists(chart_path):
                try:
                    os.remove(chart_path)
                    logger.info(f"🗑️ Cleaned stale pending chart: {chart_path}")
                    cleaned += 1
                except FileNotFoundError:
                    pass
                except (PermissionError, OSError, IOError) as e:
                    logger.warning(f"Failed to cleanup stale chart {chart_path}: {e}")
        
        return cleaned
    
    async def _cleanup_all_pending_charts(self):
        """Cleanup all pending charts - called during shutdown.
        
        Evicts all registered pending charts for a clean shutdown.
        """
        import os
        
        async with self._chart_cleanup_lock:
            chart_count = len(self._pending_charts)
            
            if chart_count == 0:
                logger.debug("No pending charts to cleanup")
                return
            
            logger.info(f"Cleaning up {chart_count} pending charts during shutdown...")
            
            for user_id, chart_info in list(self._pending_charts.items()):
                chart_path = chart_info.get('chart_path')
                if chart_path and os.path.exists(chart_path):
                    try:
                        os.remove(chart_path)
                        logger.debug(f"Cleaned up pending chart: {chart_path}")
                    except (FileNotFoundError, PermissionError, OSError, IOError) as e:
                        logger.warning(f"Failed to cleanup chart {chart_path}: {e}")
            
            self._pending_charts.clear()
            logger.info(f"✅ All {chart_count} pending charts cleaned up")
    
    async def _integrate_chart_with_session_manager(self):
        """Integrate chart cleanup with SignalSessionManager.
        
        Registers event handlers for session events to cleanup charts
        when sessions end.
        """
        if self.signal_session_manager:
            async def on_session_end_chart_cleanup(session):
                """Handler untuk cleanup chart saat session berakhir."""
                try:
                    user_id = session.user_id
                    chart_path = session.chart_path
                    
                    if user_id in self._pending_charts:
                        await self.evict_pending_chart(user_id, reason="session_end")
                    elif chart_path:
                        import os
                        if os.path.exists(chart_path):
                            try:
                                os.remove(chart_path)
                                logger.info(f"🗑️ Cleaned session chart on end: {chart_path}")
                            except (FileNotFoundError, PermissionError, OSError, IOError) as e:
                                logger.warning(f"Failed to cleanup session chart: {e}")
                except (TelegramError, asyncio.TimeoutError, OSError, KeyError, AttributeError) as e:
                    logger.error(f"Error in session end chart cleanup: {e}")
            
            self.signal_session_manager.register_event_handler('on_session_end', on_session_end_chart_cleanup)
            logger.info("✅ Chart cleanup integrated with SignalSessionManager")
    
    def _escape_markdown(self, text: str) -> str:
        """Escape karakter khusus Markdown untuk Telegram."""
        if not text:
            return ""
        escape_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        result = str(text)
        for char in escape_chars:
            result = result.replace(char, f'\\{char}')
        return result
    
    async def _render_dashboard_message(self, chat_id: int) -> str:
        """Render pesan dashboard real-time dengan format Markdown.
        
        Returns:
            str: Pesan dashboard dalam format Markdown
        """
        import hashlib
        wib = pytz.timezone('Asia/Jakarta')
        now_wib = datetime.now(wib)
        
        lines = []
        lines.append("📊 *DASHBOARD REAL-TIME XAUUSD*")
        lines.append(f"⏰ Update: {now_wib.strftime('%H:%M:%S WIB')}")
        lines.append("")
        
        lines.append("💰 *HARGA XAUUSD (REAL-TIME)*")
        try:
            if self.market_data:
                bid = 0.0
                ask = 0.0
                spread = 0.0
                high_24h = 0.0
                low_24h = 0.0
                change_pct = 0.0
                mid_price = 0.0
                data_age = "N/A"
                
                # Ambil harga bid/ask real-time langsung dari WebSocket
                if hasattr(self.market_data, 'get_bid_ask'):
                    bid_ask = await self.market_data.get_bid_ask()
                    if bid_ask:
                        bid, ask = bid_ask
                        mid_price = (bid + ask) / 2
                        spread = (ask - bid) * 10  # Convert to pips
                
                # Jika tidak ada dari get_bid_ask, coba dari current_bid/ask langsung
                if bid == 0 and hasattr(self.market_data, 'current_bid') and self.market_data.current_bid:
                    bid = self.market_data.current_bid
                    ask = self.market_data.current_ask if self.market_data.current_ask else bid
                    mid_price = (bid + ask) / 2
                    spread = (ask - bid) * 10
                
                # Cek umur data (seberapa fresh)
                if hasattr(self.market_data, 'last_data_received') and self.market_data.last_data_received:
                    age_seconds = (datetime.now() - self.market_data.last_data_received).total_seconds()
                    if age_seconds < 5:
                        data_age = "🟢 LIVE"
                    elif age_seconds < 30:
                        data_age = f"🟡 {int(age_seconds)}s ago"
                    else:
                        data_age = f"🔴 {int(age_seconds)}s ago"
                
                # Ambil 24h high/low dan change dari M5 candles
                if hasattr(self.market_data, 'm5_builder') and self.market_data.m5_builder:
                    df = self.market_data.m5_builder.get_dataframe(288)
                    if df is not None and len(df) > 0:
                        high_24h = df['high'].max() if 'high' in df.columns else 0
                        low_24h = df['low'].min() if 'low' in df.columns else 0
                        first_close = df['close'].iloc[0] if 'close' in df.columns else 0
                        last_close = df['close'].iloc[-1] if 'close' in df.columns else 0
                        if first_close > 0:
                            change_pct = ((last_close - first_close) / first_close) * 100
                
                change_emoji = "📈" if change_pct >= 0 else "📉"
                
                lines.append(f"• Status: {data_age}")
                lines.append(f"• Mid: *${mid_price:.2f}*")
                lines.append(f"• Bid: ${bid:.2f} | Ask: ${ask:.2f}")
                lines.append(f"• Spread: {spread:.1f} pips")
                if high_24h > 0 and low_24h > 0:
                    lines.append(f"• 24h: ${low_24h:.2f} - ${high_24h:.2f}")
                    lines.append(f"• Change: {change_emoji} {change_pct:+.2f}%")
            else:
                lines.append("• Data tidak tersedia")
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            lines.append(f"• Error: Data tidak tersedia")
            logger.debug(f"Dashboard price error: {e}")
        
        lines.append("")
        
        lines.append("📊 *MARKET REGIME*")
        try:
            if self.market_regime_detector and self.market_data:
                df_m5 = None
                if hasattr(self.market_data, 'm5_builder') and self.market_data.m5_builder:
                    df_m5 = self.market_data.m5_builder.get_dataframe(100)
                
                if df_m5 is not None and len(df_m5) >= 50:
                    from bot.indicators import IndicatorEngine
                    indicator_engine = IndicatorEngine(self.config)
                    indicators = indicator_engine.get_indicators(df_m5)
                    
                    regime_result = self.market_regime_detector.get_regime(indicators or {}, None, df_m5)
                    
                    if regime_result:
                        regime_type = regime_result.regime_type if hasattr(regime_result, 'regime_type') else 'unknown'
                        volatility_info = regime_result.volatility_analysis if hasattr(regime_result, 'volatility_analysis') else None
                        volatility = volatility_info.volatility_level if volatility_info else 'normal'
                        bias = regime_result.bias if hasattr(regime_result, 'bias') else 'NEUTRAL'
                        confidence = regime_result.confidence * 100 if hasattr(regime_result, 'confidence') else 0
                        
                        regime_emoji = {'strong_trend': '📈', 'moderate_trend': '📈', 'weak_trend': '📉', 
                                       'range_bound': '↔️', 'high_volatility': '⚡', 'breakout': '🚀'}.get(regime_type, '❓')
                        vol_emoji = {'high': '🔴', 'normal': '🟡', 'low': '🟢'}.get(volatility, '⚪')
                        bias_emoji = {'BUY': '🟢', 'SELL': '🔴', 'NEUTRAL': '⚪'}.get(bias, '⚪')
                        
                        lines.append(f"• Tren: {regime_emoji} {regime_type.upper().replace('_', ' ')}")
                        lines.append(f"• Volatilitas: {vol_emoji} {volatility.upper()}")
                        lines.append(f"• Bias: {bias_emoji} {bias}")
                        lines.append(f"• Confidence: {confidence:.0f}%")
                    else:
                        lines.append("• Analisis tidak tersedia")
                else:
                    lines.append("• Data tidak cukup")
            else:
                lines.append("• Detector tidak tersedia")
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            lines.append("• Error analisis regime")
            logger.debug(f"Dashboard regime error: {e}")
        
        lines.append("")
        
        lines.append("📡 *SINYAL TERAKHIR*")
        try:
            has_signal = False
            user_prefix = f"{chat_id}_"
            for type_key, signal_info in list(self.last_signal_per_type.items()):
                if not type_key.startswith(user_prefix):
                    continue
                if signal_info:
                    signal_type = signal_info.get('signal_type', 'N/A')
                    entry_price = signal_info.get('entry_price', 0)
                    timestamp = signal_info.get('timestamp')
                    
                    if timestamp:
                        time_ago = (datetime.now() - timestamp).total_seconds()
                        if time_ago < 3600:
                            time_str = f"{int(time_ago/60)}m lalu"
                        else:
                            time_str = f"{int(time_ago/3600)}h lalu"
                        
                        signal_emoji = "🟢" if signal_type == 'BUY' else "🔴"
                        lines.append(f"• {signal_emoji} {signal_type} @${entry_price:.2f} ({time_str})")
                        has_signal = True
            
            if not has_signal:
                lines.append("• Belum ada sinyal hari ini")
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            lines.append("• Error data sinyal")
            logger.debug(f"Dashboard signal error: {e}")
        
        lines.append("")
        
        lines.append("📈 *POSISI AKTIF*")
        try:
            if self.position_tracker:
                positions = await self.position_tracker.get_active_positions_async(chat_id)
                if positions:
                    for pos in list(positions.values())[:3]:
                        signal_type = pos.get('signal_type', 'N/A')
                        entry = pos.get('entry_price', 0)
                        sl = pos.get('stop_loss', 0)
                        tp = pos.get('take_profit', 0)
                        current_price = pos.get('current_price', entry)
                        
                        if signal_type == 'BUY':
                            pnl = (current_price - entry) if current_price > 0 else 0
                        else:
                            pnl = (entry - current_price) if current_price > 0 else 0
                        
                        pnl_emoji = "🟢" if pnl >= 0 else "🔴"
                        type_emoji = "📗" if signal_type == 'BUY' else "📕"
                        
                        lines.append(f"• {type_emoji} {signal_type}")
                        lines.append(f"  Entry: ${entry:.2f}")
                        lines.append(f"  SL: ${sl:.2f} | TP: ${tp:.2f}")
                        lines.append(f"  P/L: {pnl_emoji} ${pnl:+.2f}")
                else:
                    lines.append("• Tidak ada posisi aktif")
            else:
                lines.append("• Tracker tidak tersedia")
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            lines.append("• Error data posisi")
            logger.debug(f"Dashboard position error: {e}")
        
        lines.append("")
        
        lines.append("📉 *STATISTIK*")
        try:
            if self.signal_quality_tracker:
                stats = self.signal_quality_tracker.get_overall_stats(days=1)
                if stats:
                    total_signals = stats.get('total_signals', 0)
                    accuracy = stats.get('overall_accuracy', 0) * 100
                    wins = stats.get('total_wins', 0)
                    losses = stats.get('total_losses', 0)
                    
                    lines.append(f"• Sinyal hari ini: {total_signals}")
                    lines.append(f"• Win Rate: {accuracy:.1f}%")
                    lines.append(f"• W/L: {wins}/{losses}")
                else:
                    lines.append("• Belum ada data hari ini")
            else:
                session = self.db.get_session()
                try:
                    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                    from bot.database import Trade
                    today_trades = session.query(Trade).filter(Trade.created_at >= today).all()
                    
                    total = len(today_trades)
                    wins = sum(1 for t in today_trades if t.status == 'WIN')
                    losses = sum(1 for t in today_trades if t.status == 'LOSS')
                    win_rate = (wins / total * 100) if total > 0 else 0
                    
                    lines.append(f"• Sinyal hari ini: {total}")
                    lines.append(f"• Win Rate: {win_rate:.1f}%")
                    lines.append(f"• W/L: {wins}/{losses}")
                finally:
                    session.close()
        except (ValueError, TypeError, KeyError, AttributeError, SQLAlchemyError) as e:
            lines.append("• Error data statistik")
            logger.debug(f"Dashboard stats error: {e}")
        
        lines.append("")
        lines.append("_Update otomatis setiap 5 detik_")
        lines.append("_Gunakan /stopdashboard untuk menghentikan_")
        
        return "\n".join(lines)
    
    async def _realtime_dashboard_update_loop(self, chat_id: int):
        """Loop untuk update dashboard real-time setiap 5 detik.
        
        Args:
            chat_id: ID chat Telegram untuk update
        """
        import hashlib
        
        logger.info(f"🖥️ Starting realtime dashboard update loop for chat {mask_user_id(chat_id)}")
        
        try:
            while self.dashboard_enabled.get(chat_id, False) and not self._is_shutting_down:
                try:
                    message_id = self.dashboard_messages.get(chat_id)
                    if not message_id:
                        logger.warning(f"Dashboard message_id not found for chat {mask_user_id(chat_id)}, stopping loop")
                        break
                    
                    new_content = await self._render_dashboard_message(chat_id)
                    
                    content_hash = hashlib.md5(new_content.encode()).hexdigest()
                    last_hash = self._dashboard_last_hash.get(chat_id)
                    
                    if content_hash != last_hash:
                        try:
                            await self.rate_limiter.acquire_async(wait=True)
                            
                            if self.app is None:
                                logger.error("Application not initialized")
                                break
                            
                            await self.app.bot.edit_message_text(
                                chat_id=chat_id,
                                message_id=message_id,
                                text=new_content,
                                parse_mode='Markdown'
                            )
                            
                            self._dashboard_last_hash[chat_id] = content_hash
                            logger.debug(f"Dashboard updated for chat {mask_user_id(chat_id)}")
                            
                        except BadRequest as e:
                            error_msg = str(e).lower()
                            if 'message is not modified' in error_msg:
                                pass
                            elif 'message to edit not found' in error_msg or 'message can\'t be edited' in error_msg:
                                logger.info(f"Dashboard message deleted for chat {mask_user_id(chat_id)}, sending new one")
                                try:
                                    if self.app is None:
                                        logger.error("Application not initialized")
                                        break
                                    new_msg = await self.app.bot.send_message(
                                        chat_id=chat_id,
                                        text=new_content,
                                        parse_mode='Markdown'
                                    )
                                    async with self._realtime_dashboard_lock:
                                        self.dashboard_messages[chat_id] = new_msg.message_id
                                        self._dashboard_last_hash[chat_id] = content_hash
                                except (TelegramError, asyncio.TimeoutError) as send_err:
                                    logger.error(f"Failed to send new dashboard: {send_err}")
                                    break
                            else:
                                logger.warning(f"Dashboard edit error: {e}")
                        except Forbidden as e:
                            logger.warning(f"User blocked bot, stopping dashboard for {mask_user_id(chat_id)}")
                            break
                        except RetryAfter as e:
                            retry_after = e.retry_after if hasattr(e, 'retry_after') else 30
                            logger.warning(f"Rate limited, waiting {retry_after}s")
                            await asyncio.sleep(retry_after)
                        except (TimedOut, NetworkError) as e:
                            logger.warning(f"Network error updating dashboard: {e}")
                            await asyncio.sleep(5)
                        except TelegramError as e:
                            logger.error(f"Telegram error updating dashboard: {e}")
                            await asyncio.sleep(5)
                    
                    await asyncio.sleep(self.DASHBOARD_UPDATE_INTERVAL)
                    
                except asyncio.CancelledError:
                    logger.info(f"Dashboard loop cancelled for chat {mask_user_id(chat_id)}")
                    break
                except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
                    logger.error(f"Error in dashboard loop: {type(e).__name__}: {e}")
                    await asyncio.sleep(5)
                    
        finally:
            async with self._realtime_dashboard_lock:
                self.dashboard_enabled.pop(chat_id, None)
                self.dashboard_messages.pop(chat_id, None)
                self.dashboard_tasks.pop(chat_id, None)
                self._dashboard_last_hash.pop(chat_id, None)
            logger.info(f"🖥️ Dashboard loop stopped for chat {mask_user_id(chat_id)}")

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        message = update.effective_message
        chat = update.effective_chat
        if user is None or message is None or chat is None:
            return
        
        if not await self._check_user_rate_limit(user.id):
            try:
                await message.reply_text("⚠️ Anda mengirim terlalu banyak request. Silakan tunggu sebentar.")
            except (TelegramError, asyncio.CancelledError):
                pass
            return
        
        try:
            # PENTING: Pastikan config secrets sudah di-load sebelum check authorization
            # Ini fix bug dimana AUTHORIZED_USER_IDS bisa kosong saat startup
            self.config.ensure_secrets_loaded()
            
            if self.user_manager:
                self.user_manager.create_user(
                    telegram_id=user.id,
                    username=user.username,
                    first_name=user.first_name,
                    last_name=user.last_name
                )
                self.user_manager.update_user_activity(user.id)
            
            # Gunakan method dari Config class untuk consistency
            is_owner = self.config.is_owner(user.id)
            is_public_user = self.config.is_public_user(user.id)
            has_full_access = self.config.has_full_access(user.id)
            is_admin_user = self.is_admin(user.id)
            
            # Debug log untuk troubleshooting authorization issues
            logger.debug(f"User {mask_user_id(user.id)} access check: owner={is_owner}, public={is_public_user}, full_access={has_full_access}, admin={is_admin_user}")
            logger.debug(f"Config AUTHORIZED_USER_IDS count: {len(self.config.AUTHORIZED_USER_IDS)}")
            
            trial_info = None
            trial_msg = ""
            
            # HANYA process trial untuk user yang TIDAK punya full access
            if not has_full_access and self.user_manager:
                trial_status = self.user_manager.check_trial_status(user.id)
                
                if trial_status is None:
                    trial_info = self.user_manager.start_trial(user.id)
                    if trial_info and not trial_info.get('already_exists', False):
                        trial_end = trial_info.get('trial_end')
                        if trial_end:
                            jakarta_tz = pytz.timezone('Asia/Jakarta')
                            trial_end_local = trial_end.replace(tzinfo=pytz.UTC).astimezone(jakarta_tz)
                            end_date = trial_end_local.strftime('%d %B %Y, %H:%M WIB')
                        else:
                            end_date = "3 hari dari sekarang"
                        trial_msg = (
                            f"\n🎁 *Selamat! Anda mendapat Trial 3 Hari GRATIS!*\n"
                            f"📅 Berakhir: {end_date}\n"
                            f"💡 Nikmati semua fitur bot selama masa trial!\n"
                        )
                        logger.info(f"New trial started for user {mask_user_id(user.id)}")
                elif trial_status.get('is_expired', False):
                    expired_msg = (
                        "⚠️ *Masa Trial Berakhir*\n\n"
                        "Masa trial 3 hari Anda telah berakhir.\n\n"
                        "Untuk melanjutkan menggunakan bot ini, silakan berlangganan.\n\n"
                        "📞 Hubungi admin untuk informasi berlangganan.\n\n"
                        "Gunakan /buyaccess untuk info berlangganan."
                    )
                    await message.reply_text(expired_msg, parse_mode='Markdown')
                    return
                else:
                    remaining_days = trial_status.get('remaining_days', 0)
                    remaining_hours = trial_status.get('remaining_hours', 0)
                    if remaining_days > 0:
                        time_left = f"{remaining_days} hari"
                    elif remaining_hours > 0:
                        time_left = f"{remaining_hours} jam"
                    else:
                        time_left = "kurang dari 1 jam"
                    trial_msg = f"\n🎁 *Trial Aktif* - Sisa: {time_left}\n"
            
            if not self.is_authorized(user.id):
                access_denied_msg = (
                    "⛔ *Akses Ditolak*\n\n"
                    "Maaf, terjadi kesalahan saat memulai trial Anda.\n\n"
                    "🔒 *Bot ini bersifat privat*\n"
                    "Silakan hubungi pemilik bot untuk bantuan."
                )
                await message.reply_text(access_denied_msg, parse_mode='Markdown')
                return
            
            # Tentukan status user dengan jelas dan konsisten
            if is_owner:
                user_status = "👑 Owner/Admin"
            elif is_admin_user:
                user_status = "👑 Admin"
            elif is_public_user:
                user_status = "✅ User Premium"
            else:
                user_status = "🎁 Trial User"
            
            mode = "LIVE" if not self.config.DRY_RUN else "DRY RUN"
            
            welcome_msg = (
                "🤖 *XAUUSD Trading Bot*\n\n"
                "Bot sinyal trading XAUUSD - ringan dan efisien.\n\n"
                f"*Status:* {user_status}\n"
                f"{trial_msg}\n"
                "*Commands Utama:*\n"
                "/help - Bantuan\n"
                "/monitor - Mulai monitoring sinyal\n"
                "/stopmonitor - Stop monitoring\n"
                "/getsignal - Dapatkan sinyal manual\n"
                "/riwayat - Lihat riwayat trading\n"
                "/performa - Statistik performa\n\n"
                "*Access:*\n"
                "/trialstatus - Cek status trial\n"
                "/buyaccess - Info berlangganan\n"
            )
            
            if is_admin_user:
                welcome_msg += (
                    "\n*Admin:*\n"
                    "/riset - Reset database\n"
                )
            
            welcome_msg += (
                f"\nChart & Analytics: Tersedia di webapp\n"
            )
            
            await message.reply_text(welcome_msg, parse_mode='Markdown')
            
            logger.info(f"Start command executed successfully for user {mask_user_id(user.id)}")
            
            chat_id = chat.id
            if chat_id not in self.monitoring_chats and chat_id not in self.monitoring_tasks:
                if len(self.monitoring_chats) >= self.MAX_MONITORING_CHATS:
                    logger.warning(f"Cannot auto-start monitoring for user {mask_user_id(user.id)} - limit reached ({self.MAX_MONITORING_CHATS})")
                else:
                    await self.auto_start_monitoring([chat_id])
                    await message.reply_text("✅ Auto-monitoring diaktifkan! Bot akan mendeteksi sinyal secara real-time.")
                    logger.info(f"✅ Auto-monitoring started for user {mask_user_id(user.id)} on /start")
            else:
                logger.debug(f"Monitoring already active for user {mask_user_id(user.id)}")
            
        except asyncio.CancelledError:
            logger.info(f"Start command dibatalkan untuk user {mask_user_id(user.id)}")
            raise
        except RetryAfter as e:
            logger.warning(f"Rate limit pada start command: retry setelah {e.retry_after}s")
            await asyncio.sleep(e.retry_after)
        except BadRequest as e:
            await self._handle_bad_request(chat.id, e, context="start_command")
        except Forbidden as e:
            await self._handle_forbidden_error(chat.id, e)
        except ChatMigrated as e:
            new_chat_id = await self._handle_chat_migrated(chat.id, e)
            if new_chat_id:
                logger.info(f"Chat bermigrasi di start command, ID baru: {new_chat_id}")
        except (TimedOut, NetworkError) as e:
            logger.warning(f"Network/timeout error pada start command: {e}")
            try:
                await message.reply_text("⏳ Koneksi timeout, silakan coba lagi.")
            except (TelegramError, asyncio.CancelledError):
                pass
        except Conflict as e:
            await self._handle_conflict_error(e)
        except InvalidToken as e:
            await self._handle_unauthorized_error(e)
        except TelegramError as e:
            logger.error(f"Telegram error pada start command: {e}")
            try:
                await message.reply_text("❌ Terjadi error Telegram. Silakan coba lagi.")
            except (TelegramError, asyncio.CancelledError):
                pass
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error(f"Data error pada start command: {type(e).__name__}: {e}", exc_info=True)
            try:
                await message.reply_text("❌ Terjadi error saat memproses command. Silakan coba lagi.")
            except (TelegramError, asyncio.CancelledError):
                pass
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_user is None or update.message is None or update.effective_chat is None:
            return
        
        user = update.effective_user
        message = update.message
        chat = update.effective_chat
        
        if not await self._check_user_rate_limit(user.id):
            try:
                await message.reply_text("⚠️ Anda mengirim terlalu banyak request. Silakan tunggu sebentar.")
            except (TelegramError, asyncio.CancelledError):
                pass
            return
        
        try:
            if not self.is_authorized(user.id):
                await message.reply_text("⛔ Anda tidak memiliki akses. Gunakan /start untuk mendaftar trial.")
                return
            
            is_admin_user = self.is_admin(user.id)
            is_authorized_user = user.id in self.config.AUTHORIZED_USER_IDS
            is_public_user = hasattr(self.config, 'ID_USER_PUBLIC') and user.id in self.config.ID_USER_PUBLIC
            
            if is_admin_user:
                user_status = "👑 Admin"
            elif is_authorized_user or is_public_user:
                user_status = "✅ User Terdaftar"
            else:
                user_status = "🎁 Trial User"
            
            help_msg = (
                "🤖 *XAUUSD Trading Bot*\n\n"
                "Bot sinyal trading XAUUSD - ringan dan efisien.\n\n"
                f"*Status:* {user_status}\n\n"
                "*📋 Commands Utama:*\n"
                "/start - Mulai bot\n"
                "/help - Bantuan\n"
                "/monitor - Mulai monitoring sinyal\n"
                "/stopmonitor - Stop monitoring\n"
                "/getsignal - Dapatkan sinyal manual\n"
                "/status - Lihat posisi aktif\n"
                "/riwayat - Lihat riwayat trading\n"
                "/performa - Statistik performa\n\n"
                "*📊 Dashboard:*\n"
                "/dashboard - Mulai real-time dashboard\n"
                "/stopdashboard - Stop dashboard\n\n"
                "*🔧 System:*\n"
                "/optimize - Status auto-optimizer\n\n"
                "*🔑 Access:*\n"
                "/trialstatus - Status trial\n"
                "/buyaccess - Info berlangganan\n\n"
            )
            
            if is_admin_user:
                help_msg += (
                    "*👨‍💼 Admin:*\n"
                    "/riset - Reset database\n\n"
                )
            
            help_msg += (
                "*⚙️ System Info:*\n"
                f"Risk: ${self.config.FIXED_RISK_AMOUNT:.2f}/trade\n"
                f"Chart & Analytics: Tersedia di webapp\n"
            )
            
            await message.reply_text(help_msg, parse_mode='Markdown')
            logger.info(f"Help command dijalankan untuk user {mask_user_id(user.id)}")
            
        except asyncio.CancelledError:
            logger.info(f"Help command dibatalkan untuk user {mask_user_id(user.id)}")
            raise
        except RetryAfter as e:
            logger.warning(f"Rate limit pada help command: retry setelah {e.retry_after}s")
        except BadRequest as e:
            await self._handle_bad_request(chat.id, e, context="help_command")
        except Forbidden as e:
            await self._handle_forbidden_error(chat.id, e)
        except ChatMigrated as e:
            await self._handle_chat_migrated(chat.id, e)
        except (TimedOut, NetworkError) as e:
            logger.warning(f"Network/timeout error pada help command: {e}")
        except Conflict as e:
            await self._handle_conflict_error(e)
        except InvalidToken as e:
            await self._handle_unauthorized_error(e)
        except TelegramError as e:
            logger.error(f"Telegram error pada help command: {e}")
            try:
                await message.reply_text("❌ Error menampilkan bantuan.")
            except (TelegramError, asyncio.CancelledError):
                pass
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error(f"Data error pada help command: {type(e).__name__}: {e}", exc_info=True)
            try:
                await message.reply_text("❌ Error menampilkan bantuan.")
            except (TelegramError, asyncio.CancelledError):
                pass
    
    async def monitor_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_user is None or update.message is None or update.effective_chat is None:
            return
        
        user = update.effective_user
        message = update.message
        chat = update.effective_chat
        
        if not await self._check_user_rate_limit(user.id):
            try:
                await message.reply_text("⚠️ Anda mengirim terlalu banyak request. Silakan tunggu sebentar.")
            except (TelegramError, asyncio.CancelledError):
                pass
            return
        
        try:
            if not self.is_authorized(user.id):
                return
            
            chat_id = chat.id
            
            if self.monitoring and chat_id in self.monitoring_chats:
                await message.reply_text("⚠️ Monitoring sudah berjalan untuk Anda!")
                return
            
            if len(self.monitoring_chats) >= self.MAX_MONITORING_CHATS:
                await message.reply_text("⚠️ Batas maksimum monitoring tercapai. Silakan coba lagi nanti.")
                logger.warning(f"Limit monitoring tercapai ({self.MAX_MONITORING_CHATS})")
                return
            
            if not self.monitoring:
                self.monitoring = True
            
            if chat_id not in self.monitoring_chats:
                self.monitoring_chats.append(chat_id)
                await message.reply_text("✅ Monitoring dimulai! Bot akan mendeteksi sinyal secara real-time...")
                task = asyncio.create_task(self._monitoring_loop(chat_id))
                self.monitoring_tasks[chat_id] = task
                logger.info(f"✅ Monitoring task dibuat untuk chat {mask_user_id(chat_id)}")
                
        except asyncio.CancelledError:
            logger.info(f"Monitor command dibatalkan untuk user {mask_user_id(user.id)}")
            raise
        except RetryAfter as e:
            logger.warning(f"Rate limit pada monitor command: retry setelah {e.retry_after}s")
        except BadRequest as e:
            await self._handle_bad_request(chat.id, e, context="monitor_command")
        except Forbidden as e:
            await self._handle_forbidden_error(chat.id, e)
        except ChatMigrated as e:
            new_chat_id = await self._handle_chat_migrated(chat.id, e)
            if new_chat_id:
                logger.info(f"Chat bermigrasi di monitor command, chat baru: {new_chat_id}")
        except (TimedOut, NetworkError) as e:
            logger.warning(f"Network/timeout error pada monitor command: {e}")
            try:
                await message.reply_text("⏳ Koneksi timeout, silakan coba lagi.")
            except (TelegramError, asyncio.CancelledError):
                pass
        except Conflict as e:
            await self._handle_conflict_error(e)
        except InvalidToken as e:
            await self._handle_unauthorized_error(e)
        except TelegramError as e:
            logger.error(f"Telegram error pada monitor command: {e}")
            try:
                await message.reply_text("❌ Error memulai monitoring. Silakan coba lagi.")
            except (TelegramError, asyncio.CancelledError):
                pass
        except (ValueError, TypeError, AttributeError, KeyError, RuntimeError) as e:
            logger.error(f"Error tidak terduga pada monitor command: {type(e).__name__}: {e}", exc_info=True)
            try:
                await message.reply_text("❌ Error memulai monitoring. Silakan coba lagi.")
            except (TelegramError, asyncio.CancelledError):
                pass
    
    async def auto_start_monitoring(self, chat_ids: List[int]):
        if not self.monitoring:
            self.monitoring = True
        
        for chat_id in chat_ids:
            if chat_id not in self.monitoring_chats:
                self.monitoring_chats.append(chat_id)
                logger.info(f"Auto-starting monitoring for chat {mask_user_id(chat_id)}")
                task = asyncio.create_task(self._monitoring_loop(chat_id))
                task.add_done_callback(
                    lambda t, cid=chat_id: self._on_monitoring_task_done(cid, t)
                )
                self.monitoring_tasks[chat_id] = task
                logger.info(f"✅ Monitoring task created for chat {mask_user_id(chat_id)} with completion callback")
    
    def _on_monitoring_task_done(self, chat_id: int, task: asyncio.Task):
        """Callback when monitoring task completes - handles cleanup.
        
        This callback is registered via add_done_callback() to handle:
        1. Normal completion (shutdown signal received)
        2. Cancelled tasks (explicit cancellation)
        3. Failed tasks (exceptions)
        
        PENTING: Callback ini HANYA dipanggil saat monitoring task benar-benar SELESAI.
        Session end (posisi ditutup) TIDAK menyebabkan monitoring task selesai.
        Monitoring task selesai karena:
        - User menjalankan /stopmonitor (removed from monitoring_chats)
        - Sistem shutdown (_is_shutting_down = True)
        - Error fatal (forbidden, invalid token, dll)
        
        Args:
            chat_id: The chat ID whose monitoring task completed
            task: The completed asyncio.Task
        """
        try:
            task_status = "unknown"
            task_error = None
            if task.cancelled():
                task_status = "cancelled"
            elif task.done():
                try:
                    exc = task.exception()
                    if exc:
                        task_status = f"failed: {type(exc).__name__}"
                        task_error = exc
                        logger.error(f"❌ Monitoring task for chat {mask_user_id(chat_id)} failed: {exc}")
                    else:
                        task_status = "completed"
                except (asyncio.CancelledError, asyncio.InvalidStateError):
                    task_status = "cancelled_during_check"
            
            # Log kondisi saat task selesai untuk debugging
            in_monitoring_chats = chat_id in self.monitoring_chats
            is_shutting_down = getattr(self, '_is_shutting_down', False)
            monitoring_flag = getattr(self, 'monitoring', False)
            
            logger.info(
                f"📋 Monitoring task DONE for chat {mask_user_id(chat_id)} | "
                f"Status: {task_status} | "
                f"In monitoring_chats: {in_monitoring_chats} | "
                f"Shutdown: {is_shutting_down} | "
                f"Monitoring flag: {monitoring_flag}"
            )
            
            if chat_id in self.monitoring_tasks:
                del self.monitoring_tasks[chat_id]
            
            # Hanya hapus dari monitoring_chats jika memang sudah di-register
            # Ini mencegah error jika sudah dihapus sebelumnya
            if chat_id in self.monitoring_chats:
                self.monitoring_chats.remove(chat_id)
                logger.debug(f"Removed chat {mask_user_id(chat_id)} from monitoring_chats after task completion")
            
            if chat_id in self._monitoring_drain_complete:
                self._monitoring_drain_complete[chat_id].set()
            
            # Cleanup resources setelah monitoring task benar-benar selesai
            asyncio.create_task(self._cleanup_user_monitoring_resources(chat_id))
            
        except Exception as e:
            logger.error(f"Error in monitoring task done callback for chat {mask_user_id(chat_id)}: {e}")
    
    async def _cleanup_user_monitoring_resources(self, chat_id: int):
        """Clean up all user-specific monitoring resources after task completion.
        
        Cleans up:
        - Dashboard streaming tasks
        - Signal cache entries for this user
        - Active monitoring tracking
        - Any pending charts for this user
        
        Args:
            chat_id: The chat ID to clean up resources for
        """
        logger.debug(f"🧹 Cleaning up monitoring resources for chat {mask_user_id(chat_id)}")
        cleanup_count = 0
        
        try:
            if chat_id in self.active_dashboards:
                dash_info = self.active_dashboards.pop(chat_id, None)
                if dash_info:
                    task = dash_info.get('task')
                    if task and not task.done():
                        task.cancel()
                        try:
                            await asyncio.wait_for(task, timeout=2.0)
                        except (asyncio.CancelledError, asyncio.TimeoutError):
                            pass
                    cleanup_count += 1
                    logger.debug(f"  - Dashboard task cancelled for chat {mask_user_id(chat_id)}")
            
            if chat_id in self.dashboard_tasks:
                task = self.dashboard_tasks.pop(chat_id, None)
                if task and not task.done():
                    task.cancel()
                    cleanup_count += 1
            
            if chat_id in self._active_monitoring:
                del self._active_monitoring[chat_id]
                cleanup_count += 1
            
            if chat_id in self._pending_charts:
                del self._pending_charts[chat_id]
                cleanup_count += 1
            
            if chat_id in self.dashboard_enabled:
                del self.dashboard_enabled[chat_id]
            if chat_id in self.dashboard_messages:
                del self.dashboard_messages[chat_id]
            if chat_id in self._dashboard_last_hash:
                del self._dashboard_last_hash[chat_id]
            
            if cleanup_count > 0:
                logger.info(f"✅ Cleaned up {cleanup_count} resources for chat {mask_user_id(chat_id)}")
            
        except Exception as e:
            logger.error(f"Error cleaning up resources for chat {mask_user_id(chat_id)}: {e}")
    
    async def stopmonitor_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_user is None or update.message is None or update.effective_chat is None:
            return
        
        user = update.effective_user
        message = update.message
        chat = update.effective_chat
        
        if not await self._check_user_rate_limit(user.id):
            try:
                await message.reply_text("⚠️ Anda mengirim terlalu banyak request. Silakan tunggu sebentar.")
            except (TelegramError, asyncio.CancelledError):
                pass
            return
        
        try:
            if not self.is_authorized(user.id):
                return
            
            chat_id = chat.id
            
            if chat_id in self.monitoring_chats:
                self.monitoring_chats.remove(chat_id)
                
                task = self.monitoring_tasks.pop(chat_id, None)
                if task:
                    if not task.done():
                        task.cancel()
                        try:
                            await asyncio.wait_for(task, timeout=3.0)
                        except (asyncio.CancelledError, asyncio.TimeoutError):
                            pass
                    logger.info(f"✅ Monitoring task dibatalkan untuk chat {mask_user_id(chat_id)}")
                
                await message.reply_text("🛑 Monitoring dihentikan untuk Anda.")
                
                if len(self.monitoring_chats) == 0:
                    self.monitoring = False
                    logger.info("Semua monitoring dihentikan")
            else:
                await message.reply_text("⚠️ Monitoring tidak sedang berjalan untuk Anda.")
                
            logger.info(f"Stop monitor command dijalankan untuk user {mask_user_id(user.id)}")
            
        except asyncio.CancelledError:
            logger.info(f"Stopmonitor command dibatalkan untuk user {mask_user_id(user.id)}")
            raise
        except RetryAfter as e:
            logger.warning(f"Rate limit pada stopmonitor command: retry setelah {e.retry_after}s")
        except BadRequest as e:
            await self._handle_bad_request(chat.id, e, context="stopmonitor_command")
        except Forbidden as e:
            await self._handle_forbidden_error(chat.id, e)
        except ChatMigrated as e:
            await self._handle_chat_migrated(chat.id, e)
        except (TimedOut, NetworkError) as e:
            logger.warning(f"Network/timeout error pada stopmonitor command: {e}")
            try:
                await message.reply_text("⏳ Koneksi timeout, silakan coba lagi.")
            except (TelegramError, asyncio.CancelledError):
                pass
        except Conflict as e:
            await self._handle_conflict_error(e)
        except InvalidToken as e:
            await self._handle_unauthorized_error(e)
        except TelegramError as e:
            logger.error(f"Telegram error pada stopmonitor command: {e}")
            try:
                await message.reply_text("❌ Error menghentikan monitoring. Silakan coba lagi.")
            except (TelegramError, asyncio.CancelledError):
                pass
        except (ValueError, TypeError, AttributeError, KeyError, RuntimeError) as e:
            logger.error(f"Error tidak terduga pada stopmonitor command: {type(e).__name__}: {e}", exc_info=True)
            try:
                await message.reply_text("❌ Error menghentikan monitoring. Silakan coba lagi.")
            except (TelegramError, asyncio.CancelledError):
                pass
    
    async def dashboard_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Command handler untuk /dashboard - Memulai real-time dashboard"""
        if update.effective_user is None or update.message is None or update.effective_chat is None:
            return
        
        user = update.effective_user
        message = update.message
        chat = update.effective_chat
        
        if not await self._check_user_rate_limit(user.id):
            try:
                await message.reply_text("⚠️ Anda mengirim terlalu banyak request. Silakan tunggu sebentar.")
            except (TelegramError, asyncio.CancelledError):
                pass
            return
        
        try:
            if not self.is_authorized(user.id):
                return
            
            chat_id = chat.id
            
            if self.dashboard_enabled.get(chat_id, False):
                await message.reply_text("⚠️ Dashboard sudah berjalan. Gunakan /stopdashboard untuk menghentikan.")
                return
            
            dashboard_content = await self._render_dashboard_message(chat_id)
            
            sent_message = await message.reply_text(dashboard_content, parse_mode='Markdown')
            
            async with self._realtime_dashboard_lock:
                self.dashboard_enabled[chat_id] = True
                self.dashboard_messages[chat_id] = sent_message.message_id
                self._dashboard_last_hash[chat_id] = ""
            
            dashboard_task = asyncio.create_task(self._realtime_dashboard_update_loop(chat_id))
            self.dashboard_tasks[chat_id] = dashboard_task
            
            logger.info(f"📊 Dashboard command started for user {mask_user_id(user.id)}")
            
        except asyncio.CancelledError:
            logger.info(f"Dashboard command cancelled for user {mask_user_id(user.id)}")
            raise
        except RetryAfter as e:
            logger.warning(f"Rate limit pada dashboard command: retry setelah {e.retry_after}s")
        except BadRequest as e:
            await self._handle_bad_request(chat.id, e, context="dashboard_command")
        except Forbidden as e:
            await self._handle_forbidden_error(chat.id, e)
        except ChatMigrated as e:
            await self._handle_chat_migrated(chat.id, e)
        except (TimedOut, NetworkError) as e:
            logger.warning(f"Network/timeout error pada dashboard command: {e}")
            try:
                await message.reply_text("⏳ Koneksi timeout, silakan coba lagi.")
            except (TelegramError, asyncio.CancelledError):
                pass
        except Conflict as e:
            await self._handle_conflict_error(e)
        except InvalidToken as e:
            await self._handle_unauthorized_error(e)
        except TelegramError as e:
            logger.error(f"Telegram error pada dashboard command: {e}")
            try:
                await message.reply_text("❌ Error memulai dashboard. Silakan coba lagi.")
            except (TelegramError, asyncio.CancelledError):
                pass
        except (ValueError, TypeError, AttributeError, KeyError, RuntimeError) as e:
            logger.error(f"Error tidak terduga pada dashboard command: {type(e).__name__}: {e}", exc_info=True)
            try:
                await message.reply_text("❌ Error memulai dashboard. Silakan coba lagi.")
            except (TelegramError, asyncio.CancelledError):
                pass
    
    async def stopdashboard_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Command handler untuk /stopdashboard - Menghentikan real-time dashboard"""
        if update.effective_user is None or update.message is None or update.effective_chat is None:
            return
        
        user = update.effective_user
        message = update.message
        chat = update.effective_chat
        
        if not await self._check_user_rate_limit(user.id):
            try:
                await message.reply_text("⚠️ Anda mengirim terlalu banyak request. Silakan tunggu sebentar.")
            except (TelegramError, asyncio.CancelledError):
                pass
            return
        
        try:
            if not self.is_authorized(user.id):
                return
            
            chat_id = chat.id
            
            if not self.dashboard_enabled.get(chat_id, False):
                await message.reply_text("⚠️ Tidak ada dashboard yang berjalan.")
                return
            
            async with self._realtime_dashboard_lock:
                self.dashboard_enabled[chat_id] = False
            
            task = self.dashboard_tasks.pop(chat_id, None)
            if task and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            
            async with self._realtime_dashboard_lock:
                self.dashboard_messages.pop(chat_id, None)
                self._dashboard_last_hash.pop(chat_id, None)
            
            await message.reply_text("🛑 Dashboard dihentikan.")
            logger.info(f"📊 Dashboard stopped for user {mask_user_id(user.id)}")
            
        except asyncio.CancelledError:
            logger.info(f"Stopdashboard command cancelled for user {mask_user_id(user.id)}")
            raise
        except RetryAfter as e:
            logger.warning(f"Rate limit pada stopdashboard command: retry setelah {e.retry_after}s")
        except BadRequest as e:
            await self._handle_bad_request(chat.id, e, context="stopdashboard_command")
        except Forbidden as e:
            await self._handle_forbidden_error(chat.id, e)
        except ChatMigrated as e:
            await self._handle_chat_migrated(chat.id, e)
        except (TimedOut, NetworkError) as e:
            logger.warning(f"Network/timeout error pada stopdashboard command: {e}")
            try:
                await message.reply_text("⏳ Koneksi timeout, silakan coba lagi.")
            except (TelegramError, asyncio.CancelledError):
                pass
        except Conflict as e:
            await self._handle_conflict_error(e)
        except InvalidToken as e:
            await self._handle_unauthorized_error(e)
        except TelegramError as e:
            logger.error(f"Telegram error pada stopdashboard command: {e}")
            try:
                await message.reply_text("❌ Error menghentikan dashboard. Silakan coba lagi.")
            except (TelegramError, asyncio.CancelledError):
                pass
        except (ValueError, TypeError, AttributeError, KeyError, RuntimeError) as e:
            logger.error(f"Error tidak terduga pada stopdashboard command: {type(e).__name__}: {e}", exc_info=True)
            try:
                await message.reply_text("❌ Error menghentikan dashboard. Silakan coba lagi.")
            except (TelegramError, asyncio.CancelledError):
                pass

    async def _check_daily_summary_pause(self, ctx: MonitoringContext) -> bool:
        """Cek apakah perlu pause untuk daily summary. Returns True jika harus skip."""
        if not (self.alert_system and self.alert_system.is_sending_daily_summary()):
            if ctx.daily_summary_skip_count > 0:
                logger.info(f"📊 [MONITORING] Daily summary selesai - signal detection resume (total skip: {ctx.daily_summary_skip_count}) | User:{mask_user_id(ctx.chat_id)}")
                ctx.daily_summary_skip_count = 0
            return False
        
        ctx.daily_summary_skip_count += 1
        now_check = datetime.now()
        
        if (now_check - ctx.last_daily_summary_log_time).total_seconds() >= 10:
            logger.debug(f"📊 [MONITORING] Skip signal detection - daily summary sedang dikirim (skip count: {ctx.daily_summary_skip_count}) | User:{mask_user_id(ctx.chat_id)}")
            ctx.last_daily_summary_log_time = now_check
        
        await asyncio.sleep(0.5)
        return True
    
    async def _check_position_eligibility(self, ctx: MonitoringContext) -> Tuple[bool, Optional[str]]:
        """Cek apakah user eligible untuk sinyal baru. Returns (can_proceed, block_reason)."""
        # CRITICAL: Early check position_tracker FIRST sebelum SignalSessionManager
        # Ini mencegah session dibuat sia-sia jika user sudah punya active position
        if await self.position_tracker.has_active_position_async(ctx.chat_id):
            logger.debug(f"⏸️ Signal blocked (EARLY) - active position exists | User:{mask_user_id(ctx.chat_id)}")
            return False, "active_position_early_check"
        
        if self.signal_session_manager:
            can_create, block_reason = await self.signal_session_manager.can_create_signal(
                ctx.chat_id, 'auto', position_tracker=self.position_tracker
            )
            if not can_create:
                logger.debug(f"⏸️ Signal blocked - {block_reason} | User:{mask_user_id(ctx.chat_id)} | Will recheck in 0.5s")
                await asyncio.sleep(0.5)
                return False, block_reason
        
        logger.debug(f"✅ No active position - ready for new signal | User:{mask_user_id(ctx.chat_id)}")
        return True, None
    
    def _check_candle_filter(self, ctx: MonitoringContext, df_m1: pd.DataFrame) -> bool:
        """Cek candle close filter. Returns True jika boleh lanjut."""
        candle_close_only = getattr(self.config, 'CANDLE_CLOSE_ONLY_SIGNALS', False)
        if not candle_close_only or len(df_m1) == 0:
            return True
        
        current_candle_timestamp = df_m1.index[-1] if hasattr(df_m1.index[-1], 'timestamp') else df_m1.index[-1]
        
        if ctx.last_candle_timestamp is not None:
            if current_candle_timestamp == ctx.last_candle_timestamp:
                return False
            else:
                logger.debug(f"🕯️ Candle baru terdeteksi: {current_candle_timestamp}")
        
        ctx.last_candle_timestamp = current_candle_timestamp
        return True
    
    def _is_duplicate_signal(self, ctx: MonitoringContext, signal_direction: str, 
                             signal_price: float, now: datetime) -> bool:
        """Cek apakah sinyal adalah duplikat."""
        if not signal_direction or not ctx.last_sent_signal:
            return False
        
        same_direction = (signal_direction == ctx.last_sent_signal)
        time_too_soon = (now - ctx.last_sent_signal_time).total_seconds() < 5
        
        same_price = False
        price_diff_pips = 0.0
        if signal_price is not None and ctx.last_sent_signal_price is not None:
            price_diff_pips = abs(signal_price - ctx.last_sent_signal_price) * self.config.XAUUSD_PIP_VALUE
            same_price = price_diff_pips < 5.0
        
        is_duplicate = same_direction and time_too_soon and same_price
        
        if is_duplicate and signal_price is not None and ctx.last_sent_signal_price is not None:
            logger.debug(f"Duplicate signal detected: {signal_direction} @{signal_price:.2f} (last: {ctx.last_sent_signal_price:.2f}, diff: {price_diff_pips:.1f} pips)")
        
        return is_duplicate
    
    async def _fetch_m5_indicators(self, indicator_engine) -> Optional[Dict]:
        """Fetch M5 indicators untuk multi-timeframe confirmation.
        
        Note: Fetches 100 candles to ensure enough data for EMA50 calculation
        (min_required = 60 for indicators with EMA50)
        """
        try:
            df_m5 = await self.market_data.get_historical_data('M5', 100)
            if df_m5 is not None and len(df_m5) >= 60:
                m5_indicators = indicator_engine.get_indicators(df_m5)
                if m5_indicators:
                    logger.debug(f"✅ M5 MTF: {len(df_m5)} candles -> indicators calculated")
                    return m5_indicators
                else:
                    logger.debug(f"⚠️ M5 MTF: {len(df_m5)} candles but get_indicators returned None")
                    return None
            else:
                candle_count = len(df_m5) if df_m5 is not None else 0
                logger.debug(f"⚠️ M5 MTF: Hanya {candle_count}/60 candles tersedia - lanjut tanpa M5")
                return None
        except Exception as m5_error:
            logger.warning(f"⚠️ M5 MTF Error: {m5_error} - lanjut tanpa M5")
            return None
    
    async def _fetch_h1_indicators(self, indicator_engine) -> Optional[Dict]:
        """Fetch H1 indicators untuk multi-timeframe confirmation.
        
        Note: Fetches 100 candles to ensure enough data for EMA50 calculation
        (min_required = 60 for indicators with EMA50)
        Returns None on failure - signal will continue without H1 data (no blocking).
        """
        try:
            df_h1 = await self.market_data.get_historical_data('H1', 100)
            if df_h1 is not None and len(df_h1) >= 60:
                h1_indicators = indicator_engine.get_indicators(df_h1)
                if h1_indicators:
                    logger.debug(f"✅ H1 MTF: {len(df_h1)} candles -> indicators calculated")
                    return h1_indicators
                else:
                    logger.debug(f"⚠️ H1 MTF: {len(df_h1)} candles but get_indicators returned None")
                    return None
            else:
                candle_count = len(df_h1) if df_h1 is not None else 0
                logger.debug(f"⚠️ H1 MTF: Hanya {candle_count}/60 candles tersedia - lanjut tanpa H1")
                return None
        except Exception as h1_error:
            logger.warning(f"⚠️ H1 MTF Error: {h1_error} - lanjut tanpa H1")
            return None
    
    async def _dispatch_signal(self, ctx: MonitoringContext, signal: Dict, 
                               df_m1: pd.DataFrame, now: datetime) -> bool:
        """Dispatch signal ke user. Returns True jika berhasil.
        
        FALLBACK DEFENSE: Quality check dilakukan di sini sebagai last line of defense
        untuk memastikan tidak ada signal yang bypass quality gate dari _process_signal_detection().
        """
        signal_direction = signal.get('signal')
        signal_price = signal.get('entry_price')
        
        if hasattr(self, 'signal_quality_tracker') and self.signal_quality_tracker:
            try:
                signal_params = {
                    'rule_name': signal.get('rule_type', 'UNKNOWN'),
                    'signal_type': signal.get('signal', 'UNKNOWN'),
                    'confluence_level': signal.get('confluence_score', 0) // 25 if signal.get('confluence_score') else 2,
                    'confidence': signal.get('confidence', 0.5),
                    'market_regime': signal.get('market_regime', 'unknown')
                }
                
                should_block, blocking_reason = self.signal_quality_tracker.should_block_signal(signal_params)
                
                if should_block:
                    logger.info(
                        f"🛡️ [DISPATCH_FALLBACK_BLOCK] User:{mask_user_id(ctx.chat_id)} | "
                        f"Rule:{signal_params['rule_name']} | Type:{signal_params['signal_type']} | "
                        f"Reason: {blocking_reason} (fallback defense)"
                    )
                    self.signal_quality_tracker.track_blocked_signal(
                        user_id=ctx.chat_id,
                        signal_data=signal_params,
                        blocking_reason=f"DISPATCH_FALLBACK: {blocking_reason}"
                    )
                    return False
                
                logger.debug(
                    f"✅ [DISPATCH_QUALITY_PASSED] User:{mask_user_id(ctx.chat_id)} | "
                    f"Rule:{signal_params['rule_name']} | Passed fallback quality check"
                )
            except Exception as quality_check_error:
                logger.warning(
                    f"⚠️ [DISPATCH_QUALITY_CHECK] Error in fallback check: {quality_check_error} - "
                    f"proceeding with signal (fail-open)"
                )
        
        async with self.signal_lock:
            global_time_since_signal = (datetime.now() - self.global_last_signal_time).total_seconds()
            
            if global_time_since_signal < self.global_signal_cooldown:
                wait_time = self.global_signal_cooldown - global_time_since_signal
                logger.info(f"Global cooldown aktif, menunda sinyal {wait_time:.1f}s untuk user {mask_user_id(ctx.chat_id)}")
                await asyncio.sleep(wait_time)
            
            if self.signal_session_manager:
                can_create, block_reason = await self.signal_session_manager.can_create_signal(
                    ctx.chat_id, 'auto', position_tracker=self.position_tracker
                )
                if not can_create:
                    logger.info(f"Signal creation blocked for user {mask_user_id(ctx.chat_id)}: {block_reason}")
                    return False
                
                await self.signal_session_manager.create_session(
                    ctx.chat_id,
                    f"auto_{int(time.time())}",
                    'auto',
                    signal['signal'],
                    signal['entry_price'],
                    signal['stop_loss'],
                    signal['take_profit']
                )
            elif await self.position_tracker.has_active_position_async(ctx.chat_id):
                logger.debug(f"Skipping - user has active position (race condition check)")
                return False
            
            await self._send_signal(ctx.chat_id, ctx.chat_id, signal, df_m1)
            
            ctx.update_last_signal(signal_direction, signal_price, now)
            self.global_last_signal_time = now
        
        self.risk_manager.record_signal(ctx.chat_id)
        ctx.last_signal_check = now
        
        if self.user_manager:
            self.user_manager.update_user_activity(ctx.chat_id)
        
        ctx.reset_retry_delay()
        return True
    
    async def _process_signal_detection(self, ctx: MonitoringContext, df_m1: pd.DataFrame, 
                                        spread: float, now: datetime) -> bool:
        """Process signal detection dan dispatch. Returns True jika signal terkirim."""
        from bot.indicators import IndicatorEngine
        indicator_engine = IndicatorEngine(self.config)
        indicators = indicator_engine.get_indicators(df_m1)
        
        if not indicators:
            return False
        
        m5_indicators = await self._fetch_m5_indicators(indicator_engine)
        h1_indicators = await self._fetch_h1_indicators(indicator_engine)
        
        market_regime = None
        if self.market_regime_detector:
            try:
                df_m5 = await self.market_data.get_historical_data('M5', 100)
                if df_m5 is not None and len(df_m5) >= 50:
                    regime_result = self.market_regime_detector.get_regime(indicators, df_m1, df_m5)
                    if regime_result:
                        market_regime = regime_result.to_dict()
                        logger.debug(f"Market regime: {market_regime.get('regime_type', 'UNKNOWN')} (confidence: {market_regime.get('confidence', 0):.2f})")
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                logger.debug(f"Market regime detection error: {e}")
        
        signal = None
        signal_rule_type = None
        
        if self.signal_rules:
            try:
                df_m5_for_rules = await self.market_data.get_historical_data('M5', 100)
                df_h1 = await self.market_data.get_historical_data('H1', 50)
                
                best_signal_result = self.signal_rules.get_best_signal(
                    df_m1=df_m1,
                    df_m5=df_m5_for_rules,
                    df_h1=df_h1
                )
                
                if best_signal_result and best_signal_result.is_valid():
                    signal_dict = best_signal_result.to_dict()
                    signal = {
                        'signal': signal_dict.get('signal_type'),
                        'entry_price': signal_dict.get('entry_price', 0),
                        'stop_loss': signal_dict.get('entry_price', 0) - (signal_dict.get('sl_pips', 10) * 0.1) if signal_dict.get('signal_type') == 'BUY' else signal_dict.get('entry_price', 0) + (signal_dict.get('sl_pips', 10) * 0.1),
                        'take_profit': signal_dict.get('entry_price', 0) + (signal_dict.get('tp_pips', 20) * 0.1) if signal_dict.get('signal_type') == 'BUY' else signal_dict.get('entry_price', 0) - (signal_dict.get('tp_pips', 20) * 0.1),
                        'confidence': signal_dict.get('confidence', 0),
                        'confluence_score': signal_dict.get('confluence_count', 0) * 25,
                        'reason': signal_dict.get('reason', ''),
                        'rule_type': signal_dict.get('rule_name', 'UNKNOWN'),
                        'timeframe': 'M1'
                    }
                    signal_rule_type = signal_dict.get('rule_name', 'UNKNOWN')
                    logger.debug(f"📡 Signal dari AggressiveSignalRules: {signal_rule_type} - {signal.get('signal')} (confidence: {signal_dict.get('confidence', 0):.2f})")
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                logger.debug(f"AggressiveSignalRules evaluation error: {e}")
        
        if not signal:
            signal = self.strategy.detect_signal(
                indicators, 'M1', 
                signal_source='auto',
                m5_indicators=m5_indicators,
                h1_indicators=h1_indicators
            )
            signal_rule_type = 'STRATEGY'
        
        if not signal:
            return False
        
        if market_regime:
            signal['market_regime'] = market_regime.get('regime', 'UNKNOWN')
            signal['regime_confidence'] = market_regime.get('confidence', 0)
        if signal_rule_type:
            signal['rule_type'] = signal_rule_type
        
        signal_direction = signal.get('signal', '')
        signal_price = signal.get('entry_price', 0.0)
        
        if not signal_direction or not signal_price:
            return False
        
        if self._is_duplicate_signal(ctx, str(signal_direction), float(signal_price), now):
            return False
        
        time_since_last_check = (now - ctx.last_signal_check).total_seconds()
        if time_since_last_check < self.config.SIGNAL_COOLDOWN_SECONDS:
            logger.debug(f"Per-user cooldown aktif, tunggu {self.config.SIGNAL_COOLDOWN_SECONDS - time_since_last_check:.1f}s lagi")
            return False
        
        can_trade, rejection_reason = self.risk_manager.can_trade(ctx.chat_id, signal['signal'])
        if not can_trade:
            return False
        
        trading_allowed = self.risk_manager.is_trading_allowed(ctx.chat_id)
        if not trading_allowed:
            drawdown_status = self.risk_manager.get_drawdown_status(ctx.chat_id)
            drawdown_level = drawdown_status.get('level', 'UNKNOWN')
            drawdown_percent = drawdown_status.get('drawdown_percent', 0)
            
            logger.warning(f"[DRAWDOWN] 🛑 Signal DIBLOKIR untuk user {mask_user_id(ctx.chat_id)}: "
                         f"Drawdown {drawdown_percent:.1f}% (Level: {drawdown_level}) - Emergency brake aktif")
            
            try:
                await self.risk_manager.check_and_alert_drawdown(ctx.chat_id)
            except Exception as alert_error:
                logger.error(f"[DRAWDOWN] Error sending alert: {alert_error}")
            
            return False
        
        drawdown_level = self.risk_manager.check_drawdown_level(ctx.chat_id)
        if drawdown_level in ('WARNING', 'CRITICAL'):
            try:
                await self.risk_manager.check_and_alert_drawdown(ctx.chat_id)
            except Exception as alert_error:
                logger.debug(f"[DRAWDOWN] Alert error (non-blocking): {alert_error}")
        
        is_valid, validation_msg = self.strategy.validate_signal(signal, spread)
        if not is_valid:
            return False
        
        if hasattr(self, 'signal_quality_tracker') and self.signal_quality_tracker:
            try:
                signal_params = {
                    'rule_name': signal.get('rule_type', 'UNKNOWN'),
                    'signal_type': signal.get('signal', 'UNKNOWN'),
                    'confluence_level': signal.get('confluence_score', 0) // 25 if signal.get('confluence_score') else 2,
                    'confidence': signal.get('confidence', 0.5),
                    'market_regime': signal.get('market_regime', 'unknown')
                }
                
                should_block, blocking_reason = self.signal_quality_tracker.should_block_signal(signal_params)
                
                if should_block:
                    logger.info(
                        f"🚫 [SIGNAL_BLOCKED] User:{mask_user_id(ctx.chat_id)} | "
                        f"Rule:{signal_params['rule_name']} | Type:{signal_params['signal_type']} | "
                        f"Reason: {blocking_reason}"
                    )
                    self.signal_quality_tracker.track_blocked_signal(
                        user_id=ctx.chat_id,
                        signal_data=signal_params,
                        blocking_reason=blocking_reason
                    )
                    return False
                
                logger.debug(
                    f"✅ [SIGNAL_PASSED_QUALITY] User:{mask_user_id(ctx.chat_id)} | "
                    f"Grade:{signal_params.get('grade', 'N/A')} | Rule:{signal_params['rule_name']}"
                )
            except Exception as quality_check_error:
                logger.warning(f"⚠️ [SIGNAL_QUALITY_CHECK] Error: {quality_check_error} - proceeding with signal")
        
        return await self._dispatch_signal(ctx, signal, df_m1, now)
    
    async def _handle_tick_timeout(self, ctx: MonitoringContext, tick_queue) -> Tuple[bool, Any]:
        """Handle tick timeout. Returns (should_continue, new_tick_queue)."""
        needs_resubscribe = ctx.record_timeout()
        logger.debug(f"Tick queue timeout untuk user {mask_user_id(ctx.chat_id)} ({ctx.consecutive_timeouts}/{ctx.max_consecutive_timeouts}), mencoba lagi...")
        
        if needs_resubscribe:
            try:
                logger.info(f"🔄 Re-subscribing setelah {ctx.max_consecutive_timeouts} consecutive timeouts untuk user {mask_user_id(ctx.chat_id)}")
                await self.market_data.unsubscribe_ticks(f'telegram_bot_{ctx.chat_id}')
                new_queue = await self.market_data.subscribe_ticks(f'telegram_bot_{ctx.chat_id}')
                ctx.reset_timeouts()
                logger.info(f"✅ Re-subscribed berhasil untuk user {mask_user_id(ctx.chat_id)}")
                return True, new_queue
            except Exception as resubscribe_error:
                logger.error(f"❌ Error saat re-subscribe untuk user {mask_user_id(ctx.chat_id)}: {resubscribe_error}")
                await asyncio.sleep(5.0)
        
        return True, tick_queue
    
    async def _monitoring_loop(self, chat_id: int):
        """Main monitoring loop - orchestrates signal detection dengan helper coroutines.
        
        Implements graceful shutdown pattern:
        - Checks _is_shutting_down flag at each iteration
        - Finishes current signal processing before exiting (drain pattern)
        - Calls _drain_user_monitoring() for proper per-chat cleanup
        - Signals drain completion for shutdown coordination
        - Heartbeat logging setiap 30 detik untuk monitoring health
        """
        tick_queue = await self.market_data.subscribe_ticks(f'telegram_bot_{chat_id}')
        logger.info(f"🟢 Monitoring STARTED untuk user {mask_user_id(chat_id)}")
        
        ctx = MonitoringContext(chat_id=chat_id)
        logger.debug(f"Created new MonitoringContext for chat {mask_user_id(chat_id)}")
        ctx.last_signal_check = datetime.now() - timedelta(seconds=self.config.SIGNAL_COOLDOWN_SECONDS)
        
        self._active_monitoring[chat_id] = {
            'started_at': datetime.now(),
            'context': ctx,
            'status': 'running',
            'last_heartbeat': datetime.now(),
            'iteration_count': 0
        }
        
        exit_reason = "normal"
        last_heartbeat_log = datetime.now()
        iteration_count = 0
        heartbeat_interval = 30.0
        
        try:
            while self.monitoring and chat_id in self.monitoring_chats and not self._is_shutting_down:
                try:
                    iteration_count += 1
                    now = datetime.now()
                    
                    if (now - last_heartbeat_log).total_seconds() >= heartbeat_interval:
                        last_heartbeat_log = now
                        if chat_id in self._active_monitoring:
                            self._active_monitoring[chat_id]['last_heartbeat'] = now
                            self._active_monitoring[chat_id]['iteration_count'] = iteration_count
                        logger.info(f"💓 [HEARTBEAT] Monitoring aktif user {mask_user_id(chat_id)} | Iterasi: {iteration_count} | Status: running")
                    
                    if await self._check_daily_summary_pause(ctx):
                        continue
                    
                    tick = await asyncio.wait_for(tick_queue.get(), timeout=30.0)
                    ctx.reset_timeouts()
                    
                    if self._is_shutting_down:
                        logger.info(f"🛑 Shutdown detected mid-iteration for user {mask_user_id(chat_id)}, finishing current tick...")
                    
                    now = datetime.now()
                    time_since_last_tick = (now - ctx.last_tick_process_time).total_seconds()
                    if time_since_last_tick < self.tick_throttle_seconds:
                        continue
                    ctx.last_tick_process_time = now
                    
                    df_m1 = await self.market_data.get_historical_data('M1', 100)
                    if df_m1 is None or len(df_m1) < 30:
                        continue
                    
                    if not self._check_candle_filter(ctx, df_m1):
                        continue
                    
                    can_proceed, _ = await self._check_position_eligibility(ctx)
                    if not can_proceed:
                        continue
                    
                    spread_value = await self.market_data.get_spread()
                    spread = spread_value if spread_value else 0.5
                    if spread > self.config.MAX_SPREAD_PIPS:
                        logger.debug(f"Spread terlalu lebar ({spread:.2f} pips), skip signal detection")
                        continue
                    
                    await self._process_signal_detection(ctx, df_m1, spread, now)
                    
                except asyncio.TimeoutError:
                    if self._is_shutting_down:
                        logger.info(f"🛑 Timeout during shutdown for user {mask_user_id(chat_id)}, exiting gracefully...")
                        exit_reason = "shutdown_timeout"
                        break
                    should_continue, tick_queue = await self._handle_tick_timeout(ctx, tick_queue)
                    if should_continue:
                        continue
                except asyncio.CancelledError:
                    logger.info(f"Monitoring loop cancelled for user {mask_user_id(chat_id)}")
                    exit_reason = "cancelled"
                    break
                except ConnectionError as e:
                    logger.warning(f"Connection error dalam monitoring loop: {e}, retry in {ctx.retry_delay}s")
                    if self._is_shutting_down:
                        exit_reason = "shutdown_connection_error"
                        break
                    await asyncio.sleep(ctx.retry_delay)
                    ctx.increase_retry_delay()
                except Forbidden as e:
                    logger.warning(f"Forbidden error in monitoring loop for {mask_user_id(chat_id)}: {e}")
                    await self._handle_forbidden_error(chat_id, e)
                    exit_reason = "forbidden"
                    break
                except ChatMigrated as e:
                    new_chat_id = await self._handle_chat_migrated(chat_id, e)
                    if new_chat_id:
                        logger.info(f"Monitoring loop: chat migrated {mask_user_id(chat_id)} -> {mask_user_id(new_chat_id)}")
                    exit_reason = "chat_migrated"
                    break
                except Conflict as e:
                    await self._handle_conflict_error(e)
                    exit_reason = "conflict"
                    break
                except InvalidToken as e:
                    await self._handle_unauthorized_error(e)
                    exit_reason = "invalid_token"
                    break
                except BadRequest as e:
                    await self._handle_bad_request(chat_id, e, context="monitoring_loop")
                    if self._is_shutting_down:
                        exit_reason = "shutdown_bad_request"
                        break
                    await asyncio.sleep(ctx.retry_delay)
                    ctx.increase_retry_delay()
                except (TimedOut, NetworkError) as e:
                    logger.warning(f"Network/Timeout error in monitoring loop for {mask_user_id(chat_id)}: {e}")
                    if self._is_shutting_down:
                        exit_reason = "shutdown_network_error"
                        break
                    await asyncio.sleep(ctx.retry_delay)
                    ctx.increase_retry_delay()
                except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
                    logger.error(f"Error processing tick dalam monitoring loop: {type(e).__name__}: {e}")
                    if self._is_shutting_down:
                        exit_reason = "shutdown_processing_error"
                        break
                    await asyncio.sleep(ctx.retry_delay)
                    ctx.increase_retry_delay()
                except Exception as e:
                    logger.error(f"⚠️ [UNEXPECTED ERROR] in monitoring loop for {mask_user_id(chat_id)}: {type(e).__name__}: {e}")
                    if self._is_shutting_down:
                        exit_reason = "shutdown_unexpected_error"
                        break
                    await asyncio.sleep(max(ctx.retry_delay, 5.0))
                    ctx.increase_retry_delay()
            
            if self._is_shutting_down and exit_reason == "normal":
                exit_reason = "shutdown_graceful"
            elif exit_reason == "normal":
                # Log detail kondisi yang menyebabkan loop keluar
                in_monitoring_chats = chat_id in self.monitoring_chats
                logger.warning(
                    f"⚠️ Monitoring loop exited - Kondisi yang berubah untuk user {mask_user_id(chat_id)}:\n"
                    f"   - self.monitoring: {self.monitoring}\n"
                    f"   - chat_id in monitoring_chats: {in_monitoring_chats}\n"
                    f"   - _is_shutting_down: {self._is_shutting_down}\n"
                    f"   - Total iterasi: {iteration_count}\n"
                    f"   CATATAN: Session end seharusnya TIDAK menyebabkan monitoring berhenti!"
                )
                exit_reason = "condition_changed"
                    
        except Exception as outer_error:
            logger.error(f"🔴 [CRITICAL] Outer exception in monitoring loop for {mask_user_id(chat_id)}: {type(outer_error).__name__}: {outer_error}")
            exit_reason = "critical_error"
                    
        finally:
            logger.info(f"🔴 Monitoring STOPPING untuk user {mask_user_id(chat_id)} | Iterasi total: {iteration_count} | Reason: {exit_reason}")
            
            if chat_id in self._active_monitoring:
                self._active_monitoring[chat_id]['status'] = 'draining'
                self._active_monitoring[chat_id]['exit_reason'] = exit_reason
                self._active_monitoring[chat_id]['stopped_at'] = datetime.now()
            
            try:
                await self.market_data.unsubscribe_ticks(f'telegram_bot_{chat_id}')
            except Exception as unsub_error:
                logger.warning(f"Error unsubscribing ticks for {mask_user_id(chat_id)}: {unsub_error}")
            
            if self.monitoring_tasks.pop(chat_id, None):
                logger.debug(f"Monitoring task removed from tracking for chat {mask_user_id(chat_id)}")
            
            if self._is_shutting_down:
                await self._drain_user_monitoring(chat_id, reason=exit_reason)
            else:
                self._active_monitoring.pop(chat_id, None)
            
            if chat_id in self._monitoring_drain_complete:
                self._monitoring_drain_complete[chat_id].set()
            
            logger.info(f"✅ Monitoring STOPPED untuk user {mask_user_id(chat_id)} (reason: {exit_reason}, iterations: {iteration_count})")
    
    @retry_on_telegram_error(max_retries=3, initial_delay=1.0)
    async def _send_telegram_message(self, chat_id: int, text: str, parse_mode: str = 'Markdown', timeout: float = 30.0):
        """Send Telegram message with retry logic and validation"""
        if not validate_chat_id(chat_id):
            raise ValidationError(f"Invalid chat_id: {chat_id}")
        
        if not text or not text.strip():
            raise ValidationError("Empty message text")
        
        if len(text) > 4096:
            logger.warning(f"Message too long ({len(text)} chars), truncating to 4096")
            text = text[:4090] + "..."
        
        await self.rate_limiter.acquire_async(wait=True)
        
        if not self.app or not self.app.bot:
            raise ValidationError("Bot not initialized")
        
        try:
            return await asyncio.wait_for(
                self.app.bot.send_message(chat_id=chat_id, text=text, parse_mode=parse_mode),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Markdown message timeout, trying plain text fallback")
            try:
                plain_text = text.replace('*', '').replace('_', '').replace('`', '')
                return await asyncio.wait_for(
                    self.app.bot.send_message(chat_id=chat_id, text=plain_text, parse_mode=None),
                    timeout=15.0
                )
            except asyncio.TimeoutError:
                logger.error(f"Plain text fallback also timeout for chat {mask_user_id(chat_id)}")
                raise TimedOut("Message send timeout (fallback failed)")
            except (TelegramError, ValueError, TypeError) as fallback_error:
                logger.error(f"Fallback error: {fallback_error}")
                raise TimedOut("Message send timeout")
    
    @retry_on_telegram_error(max_retries=3, initial_delay=1.0)
    async def _send_telegram_photo(self, chat_id: int, photo_path: str, caption: Optional[str] = None, timeout: float = 90.0):
        """Send Telegram photo with retry logic and validation"""
        if not validate_chat_id(chat_id):
            raise ValidationError(f"Invalid chat_id: {chat_id}")
        
        if not photo_path or not photo_path.strip():
            raise ValidationError("Empty photo path")
        
        import os
        if not os.path.exists(photo_path):
            raise ValidationError(f"Photo file not found: {photo_path}")
        
        if caption is not None and len(caption) > 1024:
            logger.warning(f"Caption too long ({len(caption)} chars), truncating")
            caption = caption[:1020] + "..."
        
        await self.rate_limiter.acquire_async(wait=True)
        
        if not self.app or not self.app.bot:
            raise ValidationError("Bot not initialized")
        
        try:
            with open(photo_path, 'rb') as photo:
                return await asyncio.wait_for(
                    self.app.bot.send_photo(chat_id=chat_id, photo=photo, caption=caption),
                    timeout=timeout
                )
        except asyncio.TimeoutError:
            logger.warning(f"Photo with caption timeout, trying without caption")
            try:
                with open(photo_path, 'rb') as photo:
                    return await asyncio.wait_for(
                        self.app.bot.send_photo(chat_id=chat_id, photo=photo, caption=None),
                        timeout=45.0
                    )
            except asyncio.TimeoutError:
                logger.error(f"Photo send timeout (fallback also failed) for chat {mask_user_id(chat_id)}")
                raise TimedOut("Photo send timeout (fallback failed)")
            except (TelegramError, IOError, OSError, ValueError) as fallback_error:
                logger.error(f"Photo fallback error: {fallback_error}")
                raise TimedOut("Photo send timeout")
    
    async def _send_signal(self, user_id: int, chat_id: int, signal: dict, df: Optional[pd.DataFrame] = None):
        """Send trading signal with enhanced error handling and validation"""
        try:
            if not validate_chat_id(user_id):
                logger.error(f"Invalid user_id: {user_id}")
                return
            
            if not validate_chat_id(chat_id):
                logger.error(f"Invalid chat_id: {chat_id}")
                return
            
            if not signal or not isinstance(signal, dict):
                logger.error(f"Invalid signal data: {type(signal)}")
                return
            
            required_fields = ['signal', 'entry_price', 'stop_loss', 'take_profit', 'timeframe']
            missing_fields = [f for f in required_fields if f not in signal]
            if missing_fields:
                logger.error(f"Signal missing required fields: {missing_fields}")
                return
            
            can_proceed = await self._check_and_set_pending(user_id, signal['signal'], signal['entry_price'])
            if not can_proceed:
                logger.warning(f"🚫 Duplicate signal blocked for user {mask_user_id(user_id)}: {signal['signal']} @${signal['entry_price']:.2f}")
                
                signal_data_for_tracking = {
                    'signal_type': signal.get('signal', 'UNKNOWN'),
                    'confidence': signal.get('confidence', 0),
                    'grade': signal.get('grade', 'N/A'),
                    'rule_name': signal.get('rule_type', 'STRATEGY'),
                    'entry_price': signal.get('entry_price', 0)
                }
                
                if hasattr(self, 'signal_quality_tracker') and self.signal_quality_tracker:
                    self.signal_quality_tracker.track_blocked_signal(
                        user_id=user_id,
                        signal_data=signal_data_for_tracking,
                        blocking_reason='DUPLICATE_SIGNAL'
                    )
                
                return

            if await self.position_tracker.has_active_position_async(user_id):
                logger.warning(f"🚫 Signal blocked - user {mask_user_id(user_id)} already has active position (position_tracker)")
                await self._rollback_signal_cache(user_id, signal['signal'], signal['entry_price'])
                
                signal_data_for_tracking = {
                    'signal_type': signal.get('signal', 'UNKNOWN'),
                    'confidence': signal.get('confidence', 0),
                    'grade': signal.get('grade', 'N/A'),
                    'rule_name': signal.get('rule_type', 'STRATEGY'),
                    'entry_price': signal.get('entry_price', 0)
                }
                
                if hasattr(self, 'signal_quality_tracker') and self.signal_quality_tracker:
                    self.signal_quality_tracker.track_blocked_signal(
                        user_id=user_id,
                        signal_data=signal_data_for_tracking,
                        blocking_reason='ACTIVE_POSITION'
                    )
                
                return
            
            signal_sent_successfully = False
            signal_type = signal['signal']
            entry_price = signal['entry_price']
            
            session = self.db.get_session()
            trade_id = None
            position_id = None
            
            try:
                signal_source = signal.get('signal_source', 'auto')
                
                trade = Trade(
                    user_id=user_id,
                    ticker='XAUUSD',
                    signal_type=signal_type,
                    signal_source=signal_source,
                    entry_price=entry_price,
                    stop_loss=signal['stop_loss'],
                    take_profit=signal['take_profit'],
                    timeframe=signal['timeframe'],
                    status='OPEN'
                )
                session.add(trade)
                session.flush()
                trade_id = trade.id
                
                logger.debug(f"Trade created in DB with ID {trade_id}, preparing to add position...")
                
                signal_quality_id = None
                if self.signal_quality_tracker and trade_id is not None:
                    try:
                        rule_name = signal.get('rule_type', 'STRATEGY')
                        confluence_score = signal.get('confluence_score', 0)
                        market_regime = signal.get('market_regime', 'unknown')
                        sl_pips = abs(entry_price - signal['stop_loss']) / 0.1 if 'stop_loss' in signal else 10.0
                        tp_pips = abs(signal['take_profit'] - entry_price) / 0.1 if 'take_profit' in signal else 20.0
                        
                        signal_data = {
                            'user_id': user_id,
                            'signal_type': signal_type,
                            'rule_name': rule_name,
                            'confluence_level': confluence_score if isinstance(confluence_score, int) else int(confluence_score / 25) + 1,
                            'market_regime': market_regime,
                            'entry_price': entry_price,
                            'sl_pips': sl_pips,
                            'tp_pips': tp_pips,
                            'confidence': signal.get('confidence', 0.5),
                            'reason': signal.get('reason', '')
                        }
                        
                        signal_quality_id = self.signal_quality_tracker.record_signal(signal_data)
                        if signal_quality_id:
                            logger.debug(f"📝 Signal recorded to quality tracker - ID:{signal_quality_id} Trade:{trade_id} Rule:{rule_name}")
                            trade.signal_quality_id = signal_quality_id
                    except (ValueError, TypeError, KeyError, AttributeError) as sqt_error:
                        logger.warning(f"Failed to record signal to quality tracker: {sqt_error}")
                
                signal_grade = signal.get('grade', 'B')
                if signal_grade not in ['A', 'B', 'C']:
                    confidence = signal.get('confidence', 0.5)
                    if confidence >= 0.85:
                        signal_grade = 'A'
                    elif confidence >= 0.65:
                        signal_grade = 'B'
                    else:
                        signal_grade = 'C'
                
                confidence_score = signal.get('confidence', 0.5)
                
                position_id = await self.position_tracker.add_position(
                    user_id,
                    trade_id,
                    signal_type,
                    entry_price,
                    signal['stop_loss'],
                    signal['take_profit'],
                    signal_quality_id,
                    signal_grade=signal_grade,
                    confidence_score=confidence_score
                )
                
                if not position_id:
                    raise ValueError("Failed to create position in position tracker")
                
                session.commit()
                logger.debug(f"✅ Database committed - Trade:{trade_id} Position:{position_id}")
                
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
                logger.error(f"DB/Position error: {type(e).__name__}: {e}")
                session.rollback()
                raise
            finally:
                session.close()
            
            try:
                sl_pips = signal.get('sl_pips', abs(signal['entry_price'] - signal['stop_loss']) * self.config.XAUUSD_PIP_VALUE)
                tp_pips = signal.get('tp_pips', abs(signal['entry_price'] - signal['take_profit']) * self.config.XAUUSD_PIP_VALUE)
                lot_size = signal.get('lot_size', self.config.LOT_SIZE)
                
                source_icon = "🤖" if signal_source == 'auto' else "👤"
                source_text = "OTOMATIS" if signal_source == 'auto' else "MANUAL"
                
                msg = MessageFormatter.signal_alert(signal, signal_source)
                
                signal_message = None
                if self.app and self.app.bot:
                    try:
                        signal_message = await self._send_telegram_message(chat_id, msg, parse_mode='Markdown', timeout=30.0)
                    except Forbidden as e:
                        logger.warning(f"User blocked bot, cannot send signal: {e}")
                        await self._handle_forbidden_error(chat_id, e)
                        return
                    except ChatMigrated as e:
                        new_chat_id = await self._handle_chat_migrated(chat_id, e)
                        if new_chat_id:
                            try:
                                signal_message = await self._send_telegram_message(new_chat_id, msg, parse_mode='Markdown', timeout=30.0)
                                chat_id = new_chat_id
                            except (TelegramError, asyncio.TimeoutError, ValueError) as retry_error:
                                logger.error(f"Failed to send signal to migrated chat: {retry_error}")
                    except BadRequest as e:
                        await self._handle_bad_request(chat_id, e, context="send_signal_message")
                        try:
                            fallback_msg = f"🚨 SINYAL {signal['signal']} @${signal['entry_price']:.2f} | SL: ${signal['stop_loss']:.2f} | TP: ${signal['take_profit']:.2f}"
                            signal_message = await self._send_telegram_message(chat_id, fallback_msg, parse_mode=None, timeout=15.0)
                        except (TelegramError, asyncio.TimeoutError, ValueError) as fallback_error:
                            logger.error(f"Fallback message also failed: {fallback_error}")
                    except Conflict as e:
                        await self._handle_conflict_error(e)
                        return
                    except InvalidToken as e:
                        await self._handle_unauthorized_error(e)
                        return
                    except (TimedOut, NetworkError, TelegramError) as e:
                        logger.error(f"Failed to send signal message: {e}")
                        try:
                            fallback_msg = f"🚨 SINYAL {signal['signal']} @${signal['entry_price']:.2f} | SL: ${signal['stop_loss']:.2f} | TP: ${signal['take_profit']:.2f}"
                            await self._send_telegram_message(chat_id, fallback_msg, parse_mode=None, timeout=15.0)
                        except (TelegramError, asyncio.TimeoutError, ValueError) as fallback_error:
                            logger.error(f"Fallback message also failed: {fallback_error}")
                    
                    if df is not None and len(df) >= 30:
                        # Check if photo already sent for this session (prevent duplicates)
                        photo_already_sent = False
                        if self.signal_session_manager:
                            session_data = self.signal_session_manager.get_active_session(user_id)
                            if session_data and session_data.photo_sent:
                                photo_already_sent = True
                                logger.debug(f"Photo already sent for user {mask_user_id(user_id)}, skipping duplicate")
                        
                        if not photo_already_sent:
                            try:
                                chart_path = await asyncio.wait_for(
                                    self.chart_generator.generate_chart_async(df, signal, signal['timeframe']),
                                    timeout=45.0
                                )
                                
                                if chart_path:
                                    try:
                                        await self._send_telegram_photo(chat_id, chart_path, timeout=60.0)
                                        if self.signal_session_manager:
                                            await self.signal_session_manager.update_session(
                                                user_id, 
                                                photo_sent=True,
                                                chart_path=chart_path
                                            )
                                        logger.info(f"📸 Chart sent successfully for user {mask_user_id(user_id)}")
                                    except Forbidden as e:
                                        logger.warning(f"User blocked bot, cannot send chart: {e}")
                                        if self.signal_session_manager:
                                            await self.signal_session_manager.update_session(user_id, photo_sent=True)
                                    except ChatMigrated as e:
                                        new_chat_id = await self._handle_chat_migrated(chat_id, e)
                                        if new_chat_id:
                                            try:
                                                await self._send_telegram_photo(new_chat_id, chart_path, timeout=60.0)
                                                if self.signal_session_manager:
                                                    await self.signal_session_manager.update_session(user_id, photo_sent=True, chart_path=chart_path)
                                            except (TelegramError, asyncio.TimeoutError):
                                                pass
                                    except BadRequest as e:
                                        await self._handle_bad_request(chat_id, e, context="send_chart_photo")
                                        if self.signal_session_manager:
                                            await self.signal_session_manager.update_session(user_id, photo_sent=True)
                                    except (TimedOut, NetworkError, TelegramError) as e:
                                        logger.warning(f"Failed to send chart: {e}. Signal sent successfully.")
                                        if self.signal_session_manager:
                                            await self.signal_session_manager.update_session(user_id, photo_sent=True)
                                    finally:
                                        if chart_path:
                                            try:
                                                await self.chart_generator.immediate_delete_chart_async(chart_path)
                                                logger.debug(f"🗑️ Immediate delete chart after send: {chart_path}")
                                            except (OSError, IOError, AttributeError) as cleanup_error:
                                                logger.warning(f"Failed to immediately delete chart: {cleanup_error}")
                                else:
                                    logger.warning(f"Chart generation returned None for {signal['signal']} signal")
                            except asyncio.TimeoutError:
                                logger.warning("Chart generation timeout - signal sent without chart")
                            except (TelegramError, ValueError, TypeError, IOError, OSError, RuntimeError) as e:
                                logger.warning(f"Chart generation/send failed: {e}. Signal sent successfully.")
                    else:
                        logger.debug(f"Skipping chart - insufficient candles ({len(df) if df is not None else 0}/30)")
                
                signal_sent_successfully = True
                await self._confirm_signal_sent(user_id, signal_type, entry_price)
                
                type_key = f"{user_id}_{signal_type}"
                signal_record_data = {
                    'timestamp': datetime.now(),
                    'entry_price': entry_price,
                    'signal_type': signal_type,
                    'stop_loss': signal.get('stop_loss'),
                    'take_profit': signal.get('take_profit'),
                    'trade_id': trade_id,
                    'position_id': position_id,
                    'confidence': signal.get('confidence', 0),
                    'grade': signal.get('grade', 'B'),
                    'timeframe': signal.get('timeframe', 'M1')
                }
                
                async with self._cache_lock:
                    self.last_signal_per_type[type_key] = signal_record_data.copy()
                
                if hasattr(self, 'signal_event_store') and self.signal_event_store:
                    try:
                        await self.signal_event_store.record_signal(user_id, signal_record_data)
                        logger.debug(f"📝 Sinyal direkam ke SignalEventStore untuk user {mask_user_id(user_id)}")
                    except Exception as store_error:
                        logger.warning(f"Gagal merekam sinyal ke SignalEventStore: {store_error}")
                
                logger.info(f"✅ Signal sent - Trade:{trade_id} Position:{position_id} User:{mask_user_id(user_id)} {signal_type} @${entry_price:.2f}")
                
                if signal_message and signal_message.message_id:
                    await self.start_dashboard(user_id, chat_id, position_id, signal_message.message_id)
                    
            except (ValidationError, ValueError) as e:
                logger.error(f"Validation error in signal processing: {e}")
            except (TelegramError, asyncio.TimeoutError, KeyError, TypeError, AttributeError, RuntimeError) as e:
                logger.error(f"Error in signal processing: {type(e).__name__}: {e}", exc_info=True)
            finally:
                if not signal_sent_successfully:
                    await self._rollback_signal_cache(user_id, signal_type, entry_price)
                    logger.debug(f"Signal cache rolled back after failure for user {mask_user_id(user_id)}")
            
        except (TelegramError, asyncio.TimeoutError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, OSError) as e:
            logger.error(f"Critical error sending signal: {type(e).__name__}: {e}", exc_info=True)
            await self._rollback_signal_cache(user_id, signal['signal'], signal['entry_price'])
            if self.error_handler:
                self.error_handler.log_exception(e, "send_signal")
            if self.alert_system:
                try:
                    await asyncio.wait_for(
                        self.alert_system.send_system_error(f"Error sending signal: {str(e)}"),
                        timeout=10.0
                    )
                except (TelegramError, asyncio.TimeoutError, ConnectionError) as alert_error:
                    logger.error(f"Failed to send error alert: {alert_error}")
    
    async def _on_session_end_handler(self, session):
        """Handler untuk event on_session_end dari SignalSessionManager.
        
        PENTING: Handler ini HANYA membersihkan resources terkait SESI SINYAL yang berakhir:
        - Dashboard untuk posisi tersebut (karena posisi sudah ditutup)
        - Signal cache untuk user (agar bisa menerima sinyal baru)
        
        Handler ini TIDAK dan TIDAK BOLEH menghentikan monitoring loop!
        Monitoring harus tetap berjalan agar user bisa menerima sinyal baru setelah posisi ditutup.
        
        Alur yang benar:
        1. Posisi ditutup (hit TP/SL)
        2. Session di-end (sinyal siklus ini selesai)
        3. Dashboard dihentikan (karena tidak ada posisi aktif)
        4. Signal cache dibersihkan (reset untuk sinyal baru)
        5. Monitoring TETAP BERJALAN (user siap terima sinyal baru)
        """
        try:
            user_id = session.user_id
            
            # CRITICAL: Cek apakah monitoring masih aktif - JANGAN diubah di sini!
            monitoring_active = user_id in self.monitoring_chats
            
            logger.info(
                f"📋 Session ended for user {mask_user_id(user_id)} | "
                f"Signal type: {session.signal_type} | "
                f"Monitoring active: {monitoring_active} | "
                f"Cleaning up session resources (dashboard + cache) - MONITORING CONTINUES"
            )
            
            # Stop dashboard untuk posisi yang sudah ditutup
            await self.stop_dashboard(user_id)
            
            # Clear signal cache agar user bisa terima sinyal baru
            await self._clear_signal_cache(user_id)
            
            logger.info(
                f"✅ Session cleanup complete for user {mask_user_id(user_id)} | "
                f"Dashboard stopped, cache cleared | "
                f"Monitoring still active: {user_id in self.monitoring_chats}"
            )
            
        except (TelegramError, asyncio.TimeoutError, KeyError, ValueError, AttributeError) as e:
            logger.error(f"Error in session end handler: {e}")
    
    async def start_dashboard(self, user_id: int, chat_id: int, position_id: int, message_id: int):
        """Start real-time dashboard monitoring untuk posisi aktif"""
        try:
            if user_id in self.active_dashboards:
                logger.debug(f"Dashboard already running for user {mask_user_id(user_id)}, stopping old one first")
                await self.stop_dashboard(user_id)
            
            async with self._dashboard_lock:
                if len(self.active_dashboards) >= self.MAX_DASHBOARDS:
                    now = datetime.now(pytz.UTC)
                    stale_users = []
                    
                    for uid, dash_info in list(self.active_dashboards.items()):
                        task = dash_info.get('task')
                        if task is None or task.done():
                            stale_users.append(uid)
                    
                    for uid in stale_users:
                        self.active_dashboards.pop(uid, None)
                    
                    if len(self.active_dashboards) >= self.MAX_DASHBOARDS:
                        sorted_dashboards = sorted(
                            self.active_dashboards.items(),
                            key=lambda x: x[1].get('started_at', now)
                        )
                        oldest_user_id = sorted_dashboards[0][0]
                        oldest_task = sorted_dashboards[0][1].get('task')
                        if oldest_task and not oldest_task.done():
                            oldest_task.cancel()
                        self.active_dashboards.pop(oldest_user_id, None)
                        logger.warning(f"Dashboard limit reached, removed oldest dashboard for user {mask_user_id(oldest_user_id)}")
            
            dashboard_task = asyncio.create_task(
                self._dashboard_update_loop(user_id, chat_id, position_id, message_id)
            )
            
            self.active_dashboards[user_id] = {
                'task': dashboard_task,
                'chat_id': chat_id,
                'position_id': position_id,
                'message_id': message_id,
                'started_at': datetime.now(pytz.UTC),
                'created_at': datetime.now()
            }
            
            logger.info(f"📊 Dashboard started - User:{mask_user_id(user_id)} Position:{position_id} Message:{message_id}")
            
        except asyncio.CancelledError:
            logger.info(f"Dashboard start cancelled for user {mask_user_id(user_id)}")
            raise
        except Forbidden as e:
            logger.warning(f"User blocked bot, cannot start dashboard: {e}")
            await self._handle_forbidden_error(chat_id, e)
        except ChatMigrated as e:
            await self._handle_chat_migrated(chat_id, e)
        except BadRequest as e:
            await self._handle_bad_request(chat_id, e, context="start_dashboard")
        except Conflict as e:
            await self._handle_conflict_error(e)
        except InvalidToken as e:
            await self._handle_unauthorized_error(e)
        except (TelegramError, NetworkError, TimedOut) as e:
            logger.error(f"Telegram error starting dashboard for user {mask_user_id(user_id)}: {e}")
        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
            logger.error(f"Data error starting dashboard for user {mask_user_id(user_id)}: {type(e).__name__}: {e}")
    
    async def stop_dashboard(self, user_id: int):
        """Stop dashboard monitoring dan cleanup task"""
        try:
            dashboard = self.active_dashboards.pop(user_id, None)
            if dashboard is None:
                logger.debug(f"No active dashboard for user {mask_user_id(user_id)}")
                return
            
            task = dashboard.get('task')
            
            if task and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            
            duration = (datetime.now(pytz.UTC) - dashboard['started_at']).total_seconds()
            logger.info(f"🛑 Dashboard stopped - User:{mask_user_id(user_id)} Duration:{duration:.1f}s")
            
        except (asyncio.CancelledError, asyncio.TimeoutError, ValueError, RuntimeError) as e:
            logger.error(f"Error stopping dashboard for user {mask_user_id(user_id)}: {type(e).__name__}: {e}")
    
    async def _dashboard_update_loop(self, user_id: int, chat_id: int, position_id: int, message_id: int):
        """Loop update dashboard INSTANT dengan progress TP/SL real-time.
        
        Menggunakan asyncio.wait_for dengan timeout yang proper untuk mencegah stuck.
        """
        update_count = 0
        last_message_text = None
        dashboard_update_interval = 5
        dashboard_operation_timeout = 15.0
        price_fetch_timeout = 10.0
        consecutive_timeout_count = 0
        max_consecutive_timeouts = 5
        
        try:
            while True:
                try:
                    await asyncio.wait_for(
                        asyncio.sleep(dashboard_update_interval),
                        timeout=dashboard_update_interval + 5.0
                    )
                    
                    if user_id not in self.active_dashboards:
                        logger.debug(f"Dashboard dihapus untuk user {mask_user_id(user_id)}, menghentikan loop")
                        break
                    
                    if not self.position_tracker:
                        logger.warning("Position tracker tidak tersedia, menghentikan dashboard")
                        break
                    
                    session = self.db.get_session()
                    try:
                        position_db = session.query(Position).filter(
                            Position.id == position_id,
                            Position.user_id == user_id
                        ).first()
                        
                        if not position_db:
                            logger.info(f"Position {position_id} tidak ditemukan di DB, menghentikan dashboard")
                            break
                        
                        # Ekstrak data position SEGERA untuk mencegah DetachedInstanceError
                        # (sebelum await apapun yang bisa menyebabkan session expire)
                        pos_status = position_db.status
                        pos_signal_type = position_db.signal_type
                        pos_entry_price = position_db.entry_price
                        pos_stop_loss = position_db.stop_loss
                        pos_take_profit = position_db.take_profit
                        pos_sl_adjustment_count = getattr(position_db, 'sl_adjustment_count', 0) or 0
                        pos_max_profit_reached = getattr(position_db, 'max_profit_reached', 0) or 0
                        
                        if pos_status != 'ACTIVE':
                            logger.info(f"Position {position_id} status {pos_status}, mengirim pesan EXPIRED")
                            
                            try:
                                expired_msg = (
                                    f"⏱️ *DASHBOARD EXPIRED*\n"
                                    f"{'━' * 32}\n\n"
                                    f"✅ Posisi sudah ditutup\n"
                                    f"📊 Status: {pos_status}\n\n"
                                    f"💡 Cek hasil:\n"
                                    f"  • /riwayat - Riwayat trading\n"
                                    f"  • /performa - Statistik lengkap\n"
                                    f"  • /stats - Statistik harian\n\n"
                                    f"⏰ {datetime.now(pytz.timezone('Asia/Jakarta')).strftime('%H:%M:%S WIB')}"
                                )
                                
                                if self.app and self.app.bot:
                                    await asyncio.wait_for(
                                        self.app.bot.edit_message_text(
                                            chat_id=chat_id,
                                            message_id=message_id,
                                            text=expired_msg,
                                            parse_mode='Markdown'
                                        ),
                                        timeout=dashboard_operation_timeout
                                    )
                                    logger.info(f"✅ Pesan EXPIRED terkirim ke user {mask_user_id(user_id)}")
                            except asyncio.TimeoutError:
                                logger.warning(f"Timeout saat mengirim pesan EXPIRED untuk user {mask_user_id(user_id)}")
                            except (TelegramError, ValueError) as e:
                                logger.error(f"Error mengirim pesan EXPIRED: {e}")
                            
                            break
                        
                        try:
                            current_price = await asyncio.wait_for(
                                self.market_data.get_current_price(),
                                timeout=price_fetch_timeout
                            )
                            consecutive_timeout_count = 0
                        except asyncio.TimeoutError:
                            consecutive_timeout_count += 1
                            logger.warning(f"Timeout mengambil harga saat ini (count: {consecutive_timeout_count})")
                            if consecutive_timeout_count >= max_consecutive_timeouts:
                                logger.error(f"Terlalu banyak timeout berturut-turut, menghentikan dashboard")
                                break
                            continue
                        
                        if current_price is None:
                            logger.warning("Gagal mendapatkan harga saat ini, melewatkan update")
                            continue
                        
                        # Gunakan variabel lokal yang sudah diekstrak
                        signal_type = pos_signal_type
                        entry_price = pos_entry_price
                        stop_loss = pos_stop_loss
                        take_profit = pos_take_profit
                        sl_adjustment_count = pos_sl_adjustment_count
                        max_profit_reached = pos_max_profit_reached
                        
                        unrealized_pl = self.risk_manager.calculate_pl(entry_price, current_price, signal_type)
                        
                        position_data = {
                            'signal_type': signal_type,
                            'entry_price': entry_price,
                            'current_price': current_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'unrealized_pl': unrealized_pl,
                            'sl_adjustment_count': sl_adjustment_count,
                            'max_profit_reached': max_profit_reached
                        }
                        
                        message_text = MessageFormatter.position_update(position_data)
                        
                        if message_text == last_message_text:
                            continue
                        
                        if not self.app or not self.app.bot:
                            logger.warning("Bot tidak diinisialisasi, tidak bisa update dashboard")
                            break
                        
                        try:
                            await asyncio.wait_for(
                                self.app.bot.edit_message_text(
                                    chat_id=chat_id,
                                    message_id=message_id,
                                    text=message_text,
                                    parse_mode='Markdown'
                                ),
                                timeout=dashboard_operation_timeout
                            )
                            
                            update_count += 1
                            last_message_text = message_text
                            logger.debug(f"Dashboard diupdate #{update_count} untuk user {mask_user_id(user_id)}")
                            
                        except asyncio.TimeoutError:
                            logger.warning(f"Timeout saat mengupdate dashboard untuk user {mask_user_id(user_id)}")
                            continue
                        except BadRequest as e:
                            if "message is not modified" in str(e).lower():
                                logger.debug("Konten pesan tidak berubah, melewatkan edit")
                                continue
                            elif "message to edit not found" in str(e).lower() or "message can't be edited" in str(e).lower():
                                logger.warning(f"Pesan {message_id} terlalu lama atau dihapus, menghentikan dashboard")
                                break
                            else:
                                logger.error(f"BadRequest saat edit pesan: {e}")
                                continue
                        
                    finally:
                        session.close()
                    
                except asyncio.CancelledError:
                    logger.info(f"Dashboard update loop cancelled for user {mask_user_id(user_id)}")
                    break
                except Forbidden as e:
                    logger.warning(f"User blocked bot in dashboard loop: {e}")
                    await self._handle_forbidden_error(chat_id, e)
                    break
                except ChatMigrated as e:
                    new_chat_id = await self._handle_chat_migrated(chat_id, e)
                    if new_chat_id:
                        chat_id = new_chat_id
                        logger.info(f"Dashboard: Updated chat ID to {mask_user_id(new_chat_id)}")
                    else:
                        break
                except Conflict as e:
                    await self._handle_conflict_error(e)
                    break
                except InvalidToken as e:
                    await self._handle_unauthorized_error(e)
                    break
                except (TimedOut, NetworkError) as e:
                    logger.warning(f"Network/Timeout error in dashboard update: {e}")
                    await asyncio.sleep(5)
                except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
                    logger.error(f"Error in dashboard update loop: {type(e).__name__}: {e}")
                    await asyncio.sleep(5)
                    
        except (TelegramError, asyncio.TimeoutError, ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
            logger.error(f"Critical error in dashboard loop: {type(e).__name__}: {e}")
        
        finally:
            if user_id in self.active_dashboards:
                await self.stop_dashboard(user_id)

    async def getsignal_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_user is None or update.message is None or update.effective_chat is None:
            return
        
        if not self.is_authorized(update.effective_user.id):
            return
        
        user_id = update.effective_user.id
        
        try:
            if self.signal_session_manager:
                can_create, block_reason = await self.signal_session_manager.can_create_signal(
                    user_id, 'manual', position_tracker=self.position_tracker
                )
                if not can_create:
                    await update.message.reply_text(
                        block_reason if block_reason else MessageFormatter.session_blocked('auto', 'manual'),
                        parse_mode='Markdown'
                    )
                    return
            elif self.position_tracker and await self.position_tracker.has_active_position_async(user_id):
                await update.message.reply_text(
                    "⏳ *Tidak Dapat Membuat Sinyal Baru*\n\n"
                    "Saat ini Anda memiliki posisi aktif yang sedang berjalan.\n"
                    "Bot akan tracking hingga TP/SL tercapai.\n\n"
                    "Tunggu hasil posisi Anda saat ini sebelum request sinyal baru.",
                    parse_mode='Markdown'
                )
                return
            
            can_trade, rejection_reason = self.risk_manager.can_trade(user_id, 'ANY')
            
            if not can_trade:
                await update.message.reply_text(
                    f"⛔ *Tidak Bisa Trading*\n\n{rejection_reason}",
                    parse_mode='Markdown'
                )
                return
            
            candle_counts = self.market_data.get_candle_counts()
            total_candles = candle_counts['M1'] + candle_counts['M5']
            
            if total_candles < 30:
                # Cek status koneksi dan subscription secara detail
                ws_connected = self.market_data.connected
                ws_subscribed = self.market_data._subscribed
                market_closed = self.market_data._market_closed
                
                # Logika lebih cerdas: market tidak tersedia jika salah satu kondisi terpenuhi
                is_market_unavailable = (
                    market_closed or 
                    not ws_connected or 
                    not ws_subscribed or 
                    total_candles == 0
                )
                
                if is_market_unavailable:
                    # Tentukan penyebab spesifik ketidaktersediaan data
                    penyebab_list = []
                    if market_closed:
                        penyebab_list.append("🔒 Market sedang tutup (Sabtu-Minggu atau hari libur)")
                    if not ws_connected:
                        penyebab_list.append("❌ WebSocket tidak terhubung ke server")
                    elif not ws_subscribed:
                        penyebab_list.append("⚠️ Subscription data belum berhasil")
                    if total_candles == 0:
                        penyebab_list.append("📭 Tidak ada data candle yang tersedia")
                    
                    penyebab_text = "\n".join([f"  {p}" for p in penyebab_list]) if penyebab_list else "  • Penyebab tidak diketahui"
                    
                    message = (
                        "🔒 *Market Tidak Tersedia*\n\n"
                        "XAUUSD tidak dapat diakses saat ini.\n\n"
                        "📋 *Status Koneksi:*\n"
                        f"• WebSocket: {'🟢 Terhubung' if ws_connected else '🔴 Terputus'}\n"
                        f"• Subscription: {'🟢 Aktif' if ws_subscribed else '🔴 Tidak Aktif'}\n"
                        f"• Market: {'🔒 Tutup' if market_closed else '🟢 Buka'}\n\n"
                        "📍 *Penyebab Tidak Tersedia:*\n"
                        f"{penyebab_text}\n\n"
                        f"📊 *Data Tersedia Saat Ini:*\n"
                        f"• M1: {candle_counts['M1']} candles\n"
                        f"• M5: {candle_counts['M5']} candles\n"
                        f"• H1: {candle_counts['H1']} candles\n\n"
                        "💡 *Langkah Selanjutnya:*\n"
                        "• Jika market tutup: Tunggu hingga buka (Senin ~05:00 WIB)\n"
                        "• Jika koneksi terputus: Bot akan reconnect otomatis\n"
                        "• Coba lagi dalam beberapa saat"
                    )
                else:
                    # Has some candles but not enough yet
                    market_status = self.market_data.market_status
                    message = (
                        "⚠️ *Data Belum Lengkap*\n\n"
                        f"Candles tersedia: {total_candles}/30\n"
                        f"Status: {market_status}\n\n"
                        "Tunggu beberapa saat dan coba lagi."
                    )
                
                await update.message.reply_text(message, parse_mode='Markdown')
                return
            
            df_m1 = await self.market_data.get_historical_data('M1', 100)
            
            if df_m1 is None or len(df_m1) < 30:
                await update.message.reply_text(
                    "⚠️ *Data Tidak Cukup*\n\n"
                    "Belum cukup data candle untuk analisis.\n"
                    f"Candles: {len(df_m1) if df_m1 is not None else 0}/30\n\n"
                    "Tunggu beberapa saat dan coba lagi.",
                    parse_mode='Markdown'
                )
                return
            
            from bot.indicators import IndicatorEngine
            indicator_engine = IndicatorEngine(self.config)
            indicators = indicator_engine.get_indicators(df_m1)
            
            if not indicators:
                await update.message.reply_text(
                    "⚠️ *Analisis Gagal*\n\n"
                    "Tidak dapat menghitung indikator.\n"
                    "Coba lagi nanti.",
                    parse_mode='Markdown'
                )
                return
            
            signal = self.strategy.detect_signal(indicators, 'M1', signal_source='manual')
            
            if not signal:
                trend_strength = indicators.get('trend_strength', 'UNKNOWN')
                current_price = await self.market_data.get_current_price()
                
                msg = (
                    "⚠️ *Tidak Ada Sinyal*\n\n"
                    "Kondisi market saat ini tidak memenuhi kriteria trading.\n\n"
                    f"*Market Info:*\n"
                    f"Price: ${current_price:.2f}\n"
                    f"Trend: {trend_strength}\n\n"
                    "Gunakan /monitor untuk auto-detect sinyal."
                )
                await update.message.reply_text(msg, parse_mode='Markdown')
                return
            
            current_price = await self.market_data.get_current_price()
            spread_value = await self.market_data.get_spread()
            spread = spread_value if spread_value else 0.5
            
            is_valid, validation_msg = self.strategy.validate_signal(signal, spread)
            
            if not is_valid:
                await update.message.reply_text(
                    f"⚠️ *Sinyal Tidak Valid*\n\n{validation_msg}",
                    parse_mode='Markdown'
                )
                return
            
            await self._send_signal(user_id, update.effective_chat.id, signal, df_m1)
            self.risk_manager.record_signal(user_id)
            
            if self.user_manager:
                self.user_manager.update_user_activity(user_id)
            
        except (TelegramError, asyncio.TimeoutError, ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
            logger.error(f"Error generating manual signal: {e}")
            await update.message.reply_text("❌ Error membuat sinyal. Coba lagi nanti.")

    async def riset_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_user is None or update.message is None:
            return
        
        if not self.is_admin(update.effective_user.id):
            await update.message.reply_text("⛔ Perintah ini hanya untuk admin.")
            return
        
        try:
            logger.info("=" * 60)
            logger.info("STARTING COMPLETE SYSTEM RESET")
            logger.info("=" * 60)
            
            monitoring_count = len(self.monitoring_chats)
            active_tasks = len(self.monitoring_tasks)
            
            logger.info("Stopping all monitoring...")
            self.monitoring = False
            self.monitoring_chats.clear()
            
            logger.info("Stopping all active dashboards...")
            dashboard_count = len(self.active_dashboards)
            dashboard_users = list(self.active_dashboards.keys())
            for user_id in dashboard_users:
                await self.stop_dashboard(user_id)
            logger.info(f"Stopped {dashboard_count} dashboards")
            
            logger.info(f"Cancelling {active_tasks} monitoring tasks...")
            for chat_id, task in list(self.monitoring_tasks.items()):
                if not task.done():
                    task.cancel()
                    logger.debug(f"Cancelled monitoring task for chat {mask_user_id(chat_id)}")
            
            if self.monitoring_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*self.monitoring_tasks.values(), return_exceptions=True),
                        timeout=5
                    )
                    logger.info("All monitoring tasks cancelled")
                except asyncio.TimeoutError:
                    logger.warning("Some monitoring tasks did not complete within timeout")
                except (asyncio.CancelledError, ValueError, RuntimeError) as e:
                    logger.error(f"Error during task cleanup: {e}")
            
            self.monitoring_tasks.clear()
            
            if self.position_tracker:
                logger.info("Clearing active positions from memory...")
                active_pos_count = sum(len(positions) for positions in self.position_tracker.active_positions.values())
                self.position_tracker.active_positions.clear()
                self.position_tracker.stop_monitoring()
                logger.info(f"Cleared {active_pos_count} positions from tracker")
            else:
                active_pos_count = 0
            
            if self.signal_session_manager:
                logger.info("Clearing all active signal sessions...")
                cleared_sessions = await self.signal_session_manager.clear_all_sessions(reason="system_reset")
                logger.info(f"Cleared {cleared_sessions} signal sessions")
            else:
                cleared_sessions = 0
            
            logger.info("Cleaning up pending charts...")
            pending_charts_count = len(self._pending_charts)
            await self._cleanup_all_pending_charts()
            logger.info(f"Cleaned up {pending_charts_count} pending charts")
            
            logger.info("Clearing signal cache...")
            signal_cache_count = len(self.sent_signals_cache)
            await self._clear_signal_cache()
            logger.info(f"Cleared {signal_cache_count} signal cache entries")
            
            logger.info("Clearing database records...")
            session = self.db.get_session()
            
            deleted_trades = session.query(Trade).delete()
            deleted_positions = session.query(Position).delete()
            deleted_performance = session.query(Performance).delete()
            
            session.commit()
            session.close()
            
            logger.info("=" * 60)
            logger.info("SYSTEM RESET COMPLETE")
            logger.info("=" * 60)
            
            msg = (
                "✅ *Reset Sistem Berhasil - Semua Dibersihkan!*\n\n"
                "*Database:*\n"
                f"• Trades dihapus: {deleted_trades}\n"
                f"• Positions dihapus: {deleted_positions}\n"
                f"• Performance dihapus: {deleted_performance}\n\n"
                "*Monitoring & Sinyal:*\n"
                f"• Monitoring dihentikan: {monitoring_count} chat\n"
                f"• Task dibatalkan: {active_tasks}\n"
                f"• Posisi aktif dihapus: {active_pos_count}\n"
                f"• Sesi sinyal dibersihkan: {cleared_sessions}\n\n"
                "*Cache & Charts:*\n"
                f"• Pending charts dibersihkan: {pending_charts_count}\n"
                f"• Signal cache dibersihkan: {signal_cache_count}\n\n"
                "✨ *Sistem sekarang bersih dan siap digunakan lagi!*\n"
                "Gunakan /monitor untuk mulai monitoring baru."
            )
            
            await update.message.reply_text(msg, parse_mode='Markdown')
            logger.info(f"Complete system reset by admin {mask_user_id(update.effective_user.id)}")
            
        except (TelegramError, asyncio.TimeoutError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, OSError) as e:
            logger.error(f"Error resetting system: {e}")
            await update.message.reply_text("❌ Error reset sistem. Cek logs untuk detail.")
    
    async def riwayat_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /riwayat command - Show last 10 completed trades"""
        if self._is_shutting_down:
            return
        
        message = update.message
        if not message:
            return
        
        chat = message.chat
        if not chat:
            return
        
        user_id = chat.id
        
        try:
            if not self.analytics:
                await message.reply_text("Modul analytics tidak tersedia.")
                return
            
            trades = self.analytics.get_recent_trades(user_id=user_id, limit=10)
            
            from bot.message_templates import MessageFormatter
            response = MessageFormatter.trade_history_format(trades)
            
            await message.reply_text(response, parse_mode='Markdown')
            logger.info(f"Sent riwayat to user {user_id}: {len(trades)} trades")
            
        except Forbidden as e:
            await self._handle_forbidden_error(chat.id, e)
        except ChatMigrated as e:
            await self._handle_chat_migrated(chat.id, e)
        except (TimedOut, NetworkError) as e:
            logger.warning(f"Network/timeout error pada riwayat command: {e}")
            try:
                await message.reply_text("Koneksi timeout, silakan coba lagi.")
            except (TelegramError, asyncio.CancelledError):
                pass
        except Conflict as e:
            await self._handle_conflict_error(e)
        except InvalidToken as e:
            await self._handle_unauthorized_error(e)
        except TelegramError as e:
            logger.error(f"Telegram error pada riwayat command: {e}")
            try:
                await message.reply_text("Error mengambil riwayat.")
            except (TelegramError, asyncio.CancelledError):
                pass
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error(f"Data error pada riwayat command: {type(e).__name__}: {e}", exc_info=True)
            try:
                await message.reply_text("Error mengambil riwayat.")
            except (TelegramError, asyncio.CancelledError):
                pass

    async def performa_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /performa command - Show performance stats (7d, 30d, all-time)"""
        if self._is_shutting_down:
            return
        
        message = update.message
        if not message:
            return
        
        chat = message.chat
        if not chat:
            return
        
        user_id = chat.id
        
        try:
            if not self.analytics:
                await message.reply_text("Modul analytics tidak tersedia.")
                return
            
            perf_7d = self.analytics.get_trading_performance(user_id=user_id, days=7)
            perf_30d = self.analytics.get_trading_performance(user_id=user_id, days=30)
            perf_all = self.analytics.get_trading_performance(user_id=user_id, days=3650)
            
            from bot.message_templates import MessageFormatter
            response = MessageFormatter.performance_summary_format(perf_7d, perf_30d, perf_all)
            
            await message.reply_text(response, parse_mode='Markdown')
            logger.info(f"Sent performa to user {user_id}")
            
        except Forbidden as e:
            await self._handle_forbidden_error(chat.id, e)
        except ChatMigrated as e:
            await self._handle_chat_migrated(chat.id, e)
        except (TimedOut, NetworkError) as e:
            logger.warning(f"Network/timeout error pada performa command: {e}")
            try:
                await message.reply_text("Koneksi timeout, silakan coba lagi.")
            except (TelegramError, asyncio.CancelledError):
                pass
        except Conflict as e:
            await self._handle_conflict_error(e)
        except InvalidToken as e:
            await self._handle_unauthorized_error(e)
        except TelegramError as e:
            logger.error(f"Telegram error pada performa command: {e}")
            try:
                await message.reply_text("Error mengambil performa.")
            except (TelegramError, asyncio.CancelledError):
                pass
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error(f"Data error pada performa command: {type(e).__name__}: {e}", exc_info=True)
            try:
                await message.reply_text("Error mengambil performa.")
            except (TelegramError, asyncio.CancelledError):
                pass

    async def trialstatus_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Command untuk melihat status trial user."""
        if update.effective_user is None or update.message is None or update.effective_chat is None:
            return
        
        user = update.effective_user
        message = update.message
        chat = update.effective_chat
        
        if not await self._check_user_rate_limit(user.id):
            try:
                await message.reply_text("⚠️ Anda mengirim terlalu banyak request. Silakan tunggu sebentar.")
            except (TelegramError, asyncio.CancelledError):
                pass
            return
        
        try:
            if self.user_manager:
                self.user_manager.update_user_activity(user.id)
            
            if user.id in self.config.AUTHORIZED_USER_IDS:
                premium_msg = (
                    "👑 *Status Akun Premium*\n\n"
                    "✅ *Akun Premium Aktif*\n\n"
                    "Anda adalah pengguna premium dengan akses penuh ke semua fitur bot.\n\n"
                    "🎯 Fitur yang tersedia:\n"
                    "• Sinyal trading real-time tanpa batas\n"
                    "• Dashboard monitoring 24/7\n"
                    "• Analisis market regime\n"
                    "• Auto-optimization\n"
                    "• Dan semua fitur premium lainnya!"
                )
                await message.reply_text(premium_msg, parse_mode='Markdown')
                logger.info(f"Trialstatus command: user {mask_user_id(user.id)} is premium (AUTHORIZED_USER_IDS)")
                return
            
            if hasattr(self.config, 'ID_USER_PUBLIC') and user.id in self.config.ID_USER_PUBLIC:
                premium_msg = (
                    "👑 *Status Akun Premium*\n\n"
                    "✅ *Akun Premium Aktif*\n\n"
                    "Anda adalah pengguna premium dengan akses penuh ke semua fitur bot.\n\n"
                    "🎯 Fitur yang tersedia:\n"
                    "• Sinyal trading real-time tanpa batas\n"
                    "• Dashboard monitoring 24/7\n"
                    "• Analisis market regime\n"
                    "• Auto-optimization\n"
                    "• Dan semua fitur premium lainnya!"
                )
                await message.reply_text(premium_msg, parse_mode='Markdown')
                logger.info(f"Trialstatus command: user {mask_user_id(user.id)} is premium (ID_USER_PUBLIC)")
                return
            
            if self.user_manager:
                trial_info = self.user_manager.get_trial_info_message(user.id)
                
                if trial_info:
                    await message.reply_text(trial_info, parse_mode='Markdown')
                    logger.info(f"Trialstatus command: showed trial info for user {mask_user_id(user.id)}")
                else:
                    no_access_msg = (
                        "⛔ *Tidak Ada Akses*\n\n"
                        "Anda belum memiliki akses ke bot ini.\n\n"
                        "🛒 *Ingin berlangganan?*\n"
                        "Gunakan /buyaccess untuk informasi berlangganan.\n\n"
                        "💡 Dapatkan akses penuh ke:\n"
                        "• Sinyal trading real-time\n"
                        "• Dashboard monitoring\n"
                        "• Analisis market regime\n"
                        "• Dan fitur premium lainnya!"
                    )
                    await message.reply_text(no_access_msg, parse_mode='Markdown')
                    logger.info(f"Trialstatus command: user {mask_user_id(user.id)} has no access")
            else:
                error_msg = (
                    "⚠️ *System Error*\n\n"
                    "Tidak dapat mengecek status trial. Silakan coba lagi nanti."
                )
                await message.reply_text(error_msg, parse_mode='Markdown')
                logger.warning(f"Trialstatus command: user_manager not available for user {mask_user_id(user.id)}")
                
        except asyncio.CancelledError:
            logger.info(f"Trialstatus command cancelled for user {mask_user_id(user.id)}")
            raise
        except RetryAfter as e:
            logger.warning(f"Rate limit pada trialstatus command: retry setelah {e.retry_after}s")
        except BadRequest as e:
            await self._handle_bad_request(chat.id, e, context="trialstatus_command")
        except Forbidden as e:
            await self._handle_forbidden_error(chat.id, e)
        except ChatMigrated as e:
            await self._handle_chat_migrated(chat.id, e)
        except (TimedOut, NetworkError) as e:
            logger.warning(f"Network/timeout error pada trialstatus command: {e}")
            try:
                await message.reply_text("⏳ Koneksi timeout, silakan coba lagi.")
            except (TelegramError, asyncio.CancelledError):
                pass
        except Conflict as e:
            await self._handle_conflict_error(e)
        except InvalidToken as e:
            await self._handle_unauthorized_error(e)
        except TelegramError as e:
            logger.error(f"Telegram error pada trialstatus command: {e}")
            try:
                await message.reply_text("❌ Error mengecek status trial.")
            except (TelegramError, asyncio.CancelledError):
                pass
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error(f"Data error pada trialstatus command: {type(e).__name__}: {e}", exc_info=True)
            try:
                await message.reply_text("❌ Error mengecek status trial.")
            except (TelegramError, asyncio.CancelledError):
                pass

    async def buyaccess_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Command untuk informasi berlangganan."""
        if update.effective_user is None or update.message is None or update.effective_chat is None:
            return
        
        user = update.effective_user
        message = update.message
        chat = update.effective_chat
        
        if not await self._check_user_rate_limit(user.id):
            try:
                await message.reply_text("⚠️ Anda mengirim terlalu banyak request. Silakan tunggu sebentar.")
            except (TelegramError, asyncio.CancelledError):
                pass
            return
        
        try:
            if self.user_manager:
                self.user_manager.update_user_activity(user.id)
            
            buyaccess_msg = (
                "💎 *Informasi Berlangganan*\n\n"
                "Dapatkan akses penuh ke XAUUSD Trading Bot Pro!\n\n"
                "📋 *Paket Berlangganan:*\n"
                "├─ 1 Bulan: Rp 150.000\n"
                "├─ 3 Bulan: Rp 400.000 _(hemat Rp 50.000)_\n"
                "└─ 6 Bulan: Rp 750.000 _(hemat Rp 150.000)_\n\n"
                "🎯 *Fitur Premium:*\n"
                "• Sinyal trading real-time unlimited\n"
                "• Dashboard monitoring 24/7\n"
                "• Analisis market regime\n"
                "• Auto-optimization strategy\n"
                "• Support prioritas\n\n"
                "💳 *Metode Pembayaran:*\n"
                "• Transfer Bank (BCA, Mandiri, BNI, BRI)\n"
                "• E-Wallet (GoPay, OVO, DANA, ShopeePay)\n"
                "• QRIS\n\n"
                "📱 *Cara Berlangganan:*\n"
                "Hubungi admin untuk melakukan pembayaran dan aktivasi akun.\n\n"
                "📞 *Kontak Admin:*\n"
                "Telegram: @dzeckyete\n\n"
                "⏰ Aktivasi akun dilakukan maksimal 1x24 jam setelah pembayaran dikonfirmasi."
            )
            
            await message.reply_text(buyaccess_msg, parse_mode='Markdown')
            logger.info(f"Buyaccess command executed for user {mask_user_id(user.id)}")
                
        except asyncio.CancelledError:
            logger.info(f"Buyaccess command cancelled for user {mask_user_id(user.id)}")
            raise
        except RetryAfter as e:
            logger.warning(f"Rate limit pada buyaccess command: retry setelah {e.retry_after}s")
        except BadRequest as e:
            await self._handle_bad_request(chat.id, e, context="buyaccess_command")
        except Forbidden as e:
            await self._handle_forbidden_error(chat.id, e)
        except ChatMigrated as e:
            await self._handle_chat_migrated(chat.id, e)
        except (TimedOut, NetworkError) as e:
            logger.warning(f"Network/timeout error pada buyaccess command: {e}")
            try:
                await message.reply_text("⏳ Koneksi timeout, silakan coba lagi.")
            except (TelegramError, asyncio.CancelledError):
                pass
        except Conflict as e:
            await self._handle_conflict_error(e)
        except InvalidToken as e:
            await self._handle_unauthorized_error(e)
        except TelegramError as e:
            logger.error(f"Telegram error pada buyaccess command: {e}")
            try:
                await message.reply_text("❌ Error menampilkan info berlangganan.")
            except (TelegramError, asyncio.CancelledError):
                pass
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error(f"Data error pada buyaccess command: {type(e).__name__}: {e}", exc_info=True)
            try:
                await message.reply_text("❌ Error menampilkan info berlangganan.")
            except (TelegramError, asyncio.CancelledError):
                pass

    async def optimize_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Command untuk melihat status dan menjalankan optimasi parameter trading."""
        if update.effective_user is None or update.message is None or update.effective_chat is None:
            return
        
        user = update.effective_user
        message = update.message
        chat = update.effective_chat
        
        if not await self._check_user_rate_limit(user.id):
            try:
                await message.reply_text("⚠️ Anda mengirim terlalu banyak request. Silakan tunggu sebentar.")
            except (TelegramError, asyncio.CancelledError):
                pass
            return
        
        try:
            if self.user_manager:
                self.user_manager.update_user_activity(user.id)
            
            if not self.auto_optimizer:
                await message.reply_text(
                    "⚠️ *Modul Auto-Optimizer Tidak Tersedia*\n\n"
                    "Auto-Optimizer sedang dalam proses inisialisasi. "
                    "Silakan coba lagi dalam beberapa saat.",
                    parse_mode='Markdown'
                )
                return
            
            status = self.auto_optimizer.get_status()
            report = self.auto_optimizer.get_status_report()
            
            await message.reply_text(report, parse_mode='Markdown')
            logger.info(f"Optimize command executed for user {mask_user_id(user.id)}")
                
        except asyncio.CancelledError:
            logger.info(f"Optimize command cancelled for user {mask_user_id(user.id)}")
            raise
        except RetryAfter as e:
            logger.warning(f"Rate limit pada optimize command: retry setelah {e.retry_after}s")
        except BadRequest as e:
            await self._handle_bad_request(chat.id, e, context="optimize_command")
        except Forbidden as e:
            await self._handle_forbidden_error(chat.id, e)
        except ChatMigrated as e:
            await self._handle_chat_migrated(chat.id, e)
        except (TimedOut, NetworkError) as e:
            logger.warning(f"Network/timeout error pada optimize command: {e}")
            try:
                await message.reply_text("⏳ Koneksi timeout, silakan coba lagi.")
            except (TelegramError, asyncio.CancelledError):
                pass
        except Conflict as e:
            await self._handle_conflict_error(e)
        except InvalidToken as e:
            await self._handle_unauthorized_error(e)
        except TelegramError as e:
            logger.error(f"Telegram error pada optimize command: {e}")
            try:
                await message.reply_text("❌ Error menampilkan status optimizer.")
            except (TelegramError, asyncio.CancelledError):
                pass
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error(f"Data error pada optimize command: {type(e).__name__}: {e}", exc_info=True)
            try:
                await message.reply_text("❌ Error menampilkan status optimizer.")
            except (TelegramError, asyncio.CancelledError):
                pass

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Command untuk melihat status trading dan posisi aktif user."""
        if update.effective_user is None or update.message is None or update.effective_chat is None:
            return
        
        user = update.effective_user
        message = update.message
        chat = update.effective_chat
        
        if not await self._check_user_rate_limit(user.id):
            try:
                await message.reply_text("⚠️ Anda mengirim terlalu banyak request. Silakan tunggu sebentar.")
            except (TelegramError, asyncio.CancelledError):
                pass
            return
        
        try:
            if self.user_manager:
                self.user_manager.update_user_activity(user.id)
            
            response_lines = ["📊 *Status Trading Anda*\n"]
            
            connection_status = "🔴 Terputus"
            if self.market_data:
                try:
                    conn_state = self.market_data.get_connection_state()
                    if conn_state.value == "CONNECTED":
                        connection_status = "🟢 Terhubung"
                    elif conn_state.value == "CONNECTING":
                        connection_status = "🟡 Menghubungkan..."
                    elif conn_state.value == "RECONNECTING":
                        connection_status = "🟡 Menghubungkan ulang..."
                    else:
                        connection_status = "🔴 Terputus"
                except (AttributeError, TypeError):
                    connection_status = "⚪ Tidak diketahui"
            
            response_lines.append(f"*📡 Koneksi:* {connection_status}")
            
            is_monitoring = chat.id in self.monitoring_chats if self.monitoring_chats else False
            monitoring_status = "✅ Aktif" if is_monitoring else "❌ Tidak Aktif"
            response_lines.append(f"*📈 Monitoring:* {monitoring_status}")
            
            last_update_text = "Tidak ada data"
            if self.market_data and hasattr(self.market_data, 'last_data_received') and self.market_data.last_data_received:
                try:
                    now = datetime.now()
                    last_data = self.market_data.last_data_received
                    if last_data.tzinfo:
                        now = datetime.now(pytz.UTC)
                    seconds_ago = (now - last_data).total_seconds()
                    
                    if seconds_ago < 60:
                        last_update_text = f"{int(seconds_ago)} detik lalu"
                    elif seconds_ago < 3600:
                        last_update_text = f"{int(seconds_ago // 60)} menit lalu"
                    else:
                        last_update_text = f"{int(seconds_ago // 3600)} jam lalu"
                except (AttributeError, TypeError):
                    pass
            
            response_lines.append(f"*🕐 Update Terakhir:* {last_update_text}")
            response_lines.append("")
            
            user_positions = {}
            if self.position_tracker and hasattr(self.position_tracker, 'active_positions'):
                user_positions = self.position_tracker.active_positions.get(user.id, {})
            
            if user_positions:
                current_price = None
                if self.market_data:
                    try:
                        current_price = await self.market_data.get_current_price()
                    except (AttributeError, TypeError, asyncio.CancelledError):
                        pass
                
                position_count = len(user_positions)
                response_lines.append(f"*💼 Posisi Aktif:* {position_count} posisi\n")
                
                for pos_id, pos in user_positions.items():
                    try:
                        signal_type = pos.get('signal_type', 'UNKNOWN')
                        entry_price = pos.get('entry_price', 0.0)
                        stop_loss = pos.get('stop_loss', 0.0)
                        take_profit = pos.get('take_profit', 0.0)
                        opened_at = pos.get('opened_at')
                        
                        response_lines.append(f"├─ {signal_type} @${entry_price:.2f}")
                        response_lines.append(f"├─ SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}")
                        
                        if current_price and entry_price > 0:
                            if signal_type == 'BUY':
                                pl_value = current_price - entry_price
                            else:
                                pl_value = entry_price - current_price
                            
                            pips = abs(pl_value) * 100
                            pl_sign = "+" if pl_value >= 0 else ""
                            pl_emoji = "🟢" if pl_value >= 0 else "🔴"
                            response_lines.append(f"├─ {pl_emoji} P/L: {pl_sign}${pl_value:.2f} ({pl_sign}{pips:.0f} pips)")
                        
                        if opened_at:
                            try:
                                now_utc = datetime.now(pytz.UTC)
                                if isinstance(opened_at, str):
                                    opened_at = datetime.fromisoformat(opened_at.replace('Z', '+00:00'))
                                if opened_at.tzinfo is None:
                                    opened_at = pytz.UTC.localize(opened_at)
                                
                                duration_seconds = (now_utc - opened_at).total_seconds()
                                if duration_seconds < 60:
                                    duration_text = f"{int(duration_seconds)} detik lalu"
                                elif duration_seconds < 3600:
                                    duration_text = f"{int(duration_seconds // 60)} menit lalu"
                                else:
                                    duration_text = f"{int(duration_seconds // 3600)} jam lalu"
                                response_lines.append(f"└─ Sejak: {duration_text}")
                            except (ValueError, TypeError, AttributeError):
                                response_lines.append("└─ Sejak: -")
                        else:
                            response_lines.append("└─ Sejak: -")
                        
                        response_lines.append("")
                    except (KeyError, TypeError, AttributeError) as pos_err:
                        logger.debug(f"Error parsing position {pos_id}: {pos_err}")
                        continue
            else:
                response_lines.append("*💼 Posisi Aktif:* Tidak ada")
            
            response_lines.append("\n_Gunakan /getsignal untuk request sinyal baru_")
            
            response_text = "\n".join(response_lines)
            await message.reply_text(response_text, parse_mode='Markdown')
            logger.info(f"Status command executed for user {mask_user_id(user.id)}")
                
        except asyncio.CancelledError:
            logger.info(f"Status command cancelled for user {mask_user_id(user.id)}")
            raise
        except RetryAfter as e:
            logger.warning(f"Rate limit pada status command: retry setelah {e.retry_after}s")
        except BadRequest as e:
            await self._handle_bad_request(chat.id, e, context="status_command")
        except Forbidden as e:
            await self._handle_forbidden_error(chat.id, e)
        except ChatMigrated as e:
            await self._handle_chat_migrated(chat.id, e)
        except (TimedOut, NetworkError) as e:
            logger.warning(f"Network/timeout error pada status command: {e}")
            try:
                await message.reply_text("⏳ Koneksi timeout, silakan coba lagi.")
            except (TelegramError, asyncio.CancelledError):
                pass
        except Conflict as e:
            await self._handle_conflict_error(e)
        except InvalidToken as e:
            await self._handle_unauthorized_error(e)
        except TelegramError as e:
            logger.error(f"Telegram error pada status command: {e}")
            try:
                await message.reply_text("❌ Error menampilkan status trading.")
            except (TelegramError, asyncio.CancelledError):
                pass
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error(f"Data error pada status command: {type(e).__name__}: {e}", exc_info=True)
            try:
                await message.reply_text("❌ Error menampilkan status trading.")
            except (TelegramError, asyncio.CancelledError):
                pass

    async def initialize(self):
        self._is_shutting_down = False
        logger.info("🔄 Reset shutdown flag for fresh start")
        
        if not self.config.TELEGRAM_BOT_TOKEN:
            logger.error("Telegram bot token not configured!")
            return False
        
        request = HTTPXRequest(
            connection_pool_size=8,
            read_timeout=30.0,
            write_timeout=30.0,
            connect_timeout=30.0,
            pool_timeout=30.0
        )
        
        self.app = (
            Application.builder()
            .token(self.config.TELEGRAM_BOT_TOKEN)
            .request(request)
            .get_updates_request(request)
            .build()
        )
        logger.info("✅ Application built with properly initialized HTTPXRequest")
        
        if self.signal_session_manager:
            self.signal_session_manager.register_event_handler('on_session_end', self._on_session_end_handler)
            logger.info("Registered dashboard cleanup handler for session end events")
        
        await self._integrate_chart_with_session_manager()
        
        if self.chart_generator:
            def on_chart_eviction_notify(user_id: int, chart_path: str, reason: str):
                """Notify chart generator when chart is evicted."""
                try:
                    if hasattr(self.chart_generator, '_pending_charts'):
                        self.chart_generator._pending_charts.discard(chart_path)
                except (AttributeError, KeyError, TypeError) as e:
                    logger.debug(f"Chart generator notification skipped: {e}")
            
            self.register_chart_eviction_callback(on_chart_eviction_notify)
            logger.info("Registered chart eviction callback for ChartGenerator integration")
        
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(CommandHandler("help", self.help_command))
        self.app.add_handler(CommandHandler("monitor", self.monitor_command))
        self.app.add_handler(CommandHandler("stopmonitor", self.stopmonitor_command))
        self.app.add_handler(CommandHandler("getsignal", self.getsignal_command))
        self.app.add_handler(CommandHandler("trialstatus", self.trialstatus_command))
        self.app.add_handler(CommandHandler("buyaccess", self.buyaccess_command))
        self.app.add_handler(CommandHandler("riset", self.riset_command))
        self.app.add_handler(CommandHandler("riwayat", self.riwayat_command))
        self.app.add_handler(CommandHandler("performa", self.performa_command))
        self.app.add_handler(CommandHandler("optimize", self.optimize_command))
        self.app.add_handler(CommandHandler("status", self.status_command))
        self.app.add_handler(CommandHandler("dashboard", self.dashboard_command))
        self.app.add_handler(CommandHandler("stopdashboard", self.stopdashboard_command))
        logger.info("✅ Dashboard command handlers registered (/dashboard, /stopdashboard)")
        
        self.app.add_error_handler(self._handle_telegram_error)
        logger.info("✅ Global error handler registered for Telegram updates")
        
        logger.info("Initializing Telegram bot...")
        await self.app.initialize()
        await self.app.start()
        logger.info("Telegram bot initialized and ready!")
        return True
    
    async def setup_webhook(self, webhook_url: str, max_retries: int = 3):
        if not self.app:
            logger.error("Bot not initialized! Call initialize() first.")
            return False
        
        if not webhook_url or not webhook_url.strip():
            logger.error("Invalid webhook URL provided - empty or None")
            return False
        
        webhook_url = webhook_url.strip()
        
        if not (webhook_url.startswith('http://') or webhook_url.startswith('https://')):
            logger.error(f"Invalid webhook URL format: {webhook_url[:50]}... (must start with http:// or https://)")
            return False
        
        is_https = webhook_url.startswith('https://')
        if not is_https:
            logger.warning("⚠️ Webhook URL uses HTTP instead of HTTPS - this may cause issues with Telegram")
        
        retry_delay = 2.0
        max_delay = 30.0
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Setting up webhook (attempt {attempt}/{max_retries}): {webhook_url}")
                
                await self.app.bot.set_webhook(
                    url=webhook_url,
                    allowed_updates=['message', 'callback_query', 'edited_message'],
                    drop_pending_updates=True
                )
                
                webhook_info = await self.app.bot.get_webhook_info()
                
                if webhook_info.url == webhook_url:
                    logger.info(f"✅ Webhook configured successfully!")
                    logger.info(f"Webhook URL: {webhook_info.url}")
                    logger.info(f"Pending updates: {webhook_info.pending_update_count}")
                    if webhook_info.last_error_message:
                        logger.warning(f"Previous webhook error: {webhook_info.last_error_message}")
                    return True
                else:
                    logger.warning(f"Webhook URL mismatch - Expected: {webhook_url}, Got: {webhook_info.url}")
                    if attempt < max_retries:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, max_delay)
                        continue
                    return False
            
            except asyncio.TimeoutError as e:
                logger.error(f"Timeout setting webhook (attempt {attempt}/{max_retries})")
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, max_delay)
                else:
                    logger.error("❌ Webhook setup timed out. Check network connectivity.")
                    
            except ConnectionError as e:
                logger.error(f"Connection error setting webhook (attempt {attempt}/{max_retries}): {e}")
                logger.error("Possible causes:")
                logger.error("  - Server not reachable from internet")
                logger.error("  - Firewall blocking incoming connections")
                logger.error("  - DNS resolution failed")
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, max_delay)
                    
            except TelegramError as e:
                error_msg = str(e).lower()
                logger.error(f"Telegram API error (attempt {attempt}/{max_retries}): {e}")
                
                if 'ssl' in error_msg or 'certificate' in error_msg:
                    logger.error("❌ SSL Certificate Error!")
                    logger.error("Possible solutions:")
                    logger.error("  - Ensure HTTPS URL has valid SSL certificate")
                    logger.error("  - Check if certificate is from trusted CA")
                    logger.error("  - Verify certificate chain is complete")
                    logger.error("  - Use a service like Let's Encrypt for free SSL")
                elif 'not found' in error_msg or 'resolve' in error_msg:
                    logger.error("❌ DNS Resolution Error!")
                    logger.error("Possible solutions:")
                    logger.error("  - Check if domain name is correct")
                    logger.error("  - Verify DNS records are propagated")
                    logger.error("  - Wait 5-10 minutes for DNS propagation")
                elif 'refused' in error_msg or 'connection' in error_msg:
                    logger.error("❌ Connection Refused!")
                    logger.error("Possible solutions:")
                    logger.error("  - Check if server is running and accessible")
                    logger.error("  - Verify firewall allows incoming HTTPS (port 443)")
                    logger.error("  - Ensure webhook endpoint is listening")
                
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, max_delay)
                    
            except (ValueError, TypeError, OSError, IOError) as e:
                error_type = type(e).__name__
                logger.error(f"Failed to setup webhook (attempt {attempt}/{max_retries}): [{error_type}] {e}")
                
                if self.error_handler:
                    self.error_handler.log_exception(e, f"setup_webhook_attempt_{attempt}")
                
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, max_delay)
                else:
                    logger.error("❌ All webhook setup attempts failed!")
                    logger.error("General troubleshooting:")
                    logger.error("  1. Webhook URL is publicly accessible")
                    logger.error("  2. SSL certificate is valid (for HTTPS)")
                    logger.error("  3. Telegram Bot API can reach your server")
                    logger.error("  4. No firewall blocking incoming connections")
                    logger.error("  5. Webhook endpoint is properly configured")
                    return False
        
        return False
    
    async def run_webhook(self):
        if not self.app:
            logger.error("Bot not initialized! Call initialize() first.")
            return
        
        logger.info("=" * 60)
        logger.info("🚀 STARTING TELEGRAM BOT IN WEBHOOK MODE")
        logger.info("=" * 60)
        
        import os
        from config import Config
        Config._refresh_secrets()
        webhook_url = self.config.WEBHOOK_URL
        koyeb_domain = os.getenv('KOYEB_PUBLIC_DOMAIN', '')
        
        logger.info(f"WEBHOOK_URL env: {os.getenv('WEBHOOK_URL', 'NOT SET')}")
        logger.info(f"KOYEB_PUBLIC_DOMAIN env: {koyeb_domain or 'NOT SET'}")
        logger.info(f"Config.WEBHOOK_URL: {webhook_url or 'NOT SET'}")
        logger.info(f"Config.IS_KOYEB: {self.config.IS_KOYEB}")
        
        if not webhook_url and koyeb_domain:
            webhook_url = f"https://{koyeb_domain}/webhook"
            self.config.WEBHOOK_URL = webhook_url
            logger.info(f"Auto-generated WEBHOOK_URL from domain: {webhook_url}")
        
        if webhook_url:
            logger.info("=" * 60)
            logger.info("📡 REGISTERING WEBHOOK TO TELEGRAM API")
            logger.info("=" * 60)
            logger.info(f"Webhook URL: {webhook_url}")
            
            await asyncio.sleep(2.0)
            
            try:
                success = await self.setup_webhook(webhook_url, max_retries=5)
                if success:
                    logger.info("=" * 60)
                    logger.info("✅ WEBHOOK REGISTERED SUCCESSFULLY!")
                    logger.info("=" * 60)
                    logger.info("Bot will now receive commands from Telegram.")
                    logger.info("Test by sending /start to your bot.")
                else:
                    logger.error("=" * 60)
                    logger.error("❌ WEBHOOK REGISTRATION FAILED!")
                    logger.error("=" * 60)
                    logger.error("Bot will NOT receive any commands from Telegram!")
                    logger.error("")
                    logger.error("Troubleshooting:")
                    logger.error(f"1. Verify URL is accessible: {webhook_url}")
                    logger.error("2. Check SSL certificate is valid")
                    logger.error("3. Run manually: python fix_webhook.py --status")
                    logger.error("4. Redeploy with correct WEBHOOK_URL")
            except Exception as e:
                logger.error(f"❌ Exception during webhook setup: {e}")
                if self.error_handler:
                    self.error_handler.log_exception(e, "run_webhook_setup")
        else:
            logger.error("=" * 60)
            logger.error("❌ NO WEBHOOK URL AVAILABLE!")
            logger.error("=" * 60)
            logger.error("Bot CANNOT receive commands without webhook URL!")
            logger.error("")
            logger.error("SOLUTION - Set ONE of these in Koyeb Dashboard:")
            logger.error("  Option 1: WEBHOOK_URL=https://your-app.koyeb.app/webhook")
            logger.error("  Option 2: KOYEB_PUBLIC_DOMAIN=your-app.koyeb.app")
            logger.error("")
            logger.error("Then REDEPLOY the service!")
            logger.error("=" * 60)
        
        logger.info("Telegram bot running in webhook mode...")
        logger.info("Waiting for webhook updates via HTTP endpoint /webhook")
        logger.info("=" * 60)
    
    async def process_update(self, update_data):
        if not self.app:
            logger.error("❌ Bot not initialized! Cannot process update.")
            logger.error("This usually means bot is running in limited mode")
            logger.error("Set TELEGRAM_BOT_TOKEN and AUTHORIZED_USER_IDS and restart")
            return
        
        if not update_data:
            logger.error("❌ Received empty update data")
            return
        
        try:
            from telegram import Update
            import json
            
            parsed_data: Any = None
            
            if isinstance(update_data, Update):
                update = update_data
                logger.info(f"📥 Received native telegram.Update object: {update.update_id}")
            else:
                parsed_data = update_data
                
                if isinstance(update_data, str):
                    try:
                        parsed_data = json.loads(update_data)
                        logger.debug("Parsed webhook update from JSON string")
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Failed to parse JSON string update: {e}")
                        return
                elif hasattr(update_data, 'to_dict') and callable(update_data.to_dict):
                    try:
                        parsed_data = update_data.to_dict()
                        logger.debug(f"Converted update data via to_dict(): {type(update_data)}")
                    except (AttributeError, TypeError, ValueError) as e:
                        logger.warning(f"Failed to convert via to_dict: {e}")
                elif not hasattr(update_data, '__getitem__'):
                    logger.warning(f"Update data is not dict-like: {type(update_data)}")
                    logger.debug(f"Attempting to use as-is: {str(update_data)[:200]}")
                
                update = Update.de_json(parsed_data, self.app.bot)
            
            if update:
                update_id = update.update_id
                
                message_info = ""
                if update.message:
                    from_user = update.message.from_user
                    if from_user:
                        message_info = f" dari user {from_user.id}"
                    if update.message.text:
                        message_info += f": '{update.message.text}'"
                
                logger.info(f"🔄 Memproses webhook update {update_id}{message_info}")
                
                await self.app.process_update(update)
                
                logger.info(f"✅ Successfully processed update {update_id}")
            else:
                logger.warning("⚠️ Received invalid or malformed update data")
                from collections.abc import Mapping
                if isinstance(parsed_data, Mapping):
                    logger.debug(f"Update data keys: {list(parsed_data.keys())}")
                
        except ValueError as e:
            logger.error(f"ValueError parsing update data: {e}")
            logger.debug(f"Problematic update data: {str(update_data)[:200]}...")
        except AttributeError as e:
            logger.error(f"AttributeError processing update: {e}")
            if self.error_handler:
                self.error_handler.log_exception(e, "process_webhook_update_attribute")
        except (TelegramError, TypeError, KeyError, RuntimeError) as e:
            error_type = type(e).__name__
            logger.error(f"Unexpected error processing webhook update: [{error_type}] {e}")
            if self.error_handler:
                self.error_handler.log_exception(e, "process_webhook_update")
            
            if hasattr(e, '__traceback__'):
                import traceback
                tb_str = ''.join(traceback.format_tb(e.__traceback__)[:3])
                logger.debug(f"Traceback: {tb_str}")
    
    async def run(self):
        """
        Main run method dengan keep-alive loop untuk webhook mode.
        Dilengkapi exception handling robust untuk mencegah task mati tanpa log.
        """
        if not self.app:
            logger.error("Bot not initialized! Call initialize() first.")
            return
        
        try:
            if self.config.TELEGRAM_WEBHOOK_MODE:
                if not self.config.WEBHOOK_URL:
                    logger.error("=" * 60)
                    logger.error("❌ KOYEB WEBHOOK ERROR: WEBHOOK_URL tidak ter-set!")
                    logger.error("=" * 60)
                    logger.error("Bot tidak bisa menerima command di Koyeb tanpa webhook URL.")
                    logger.error("")
                    logger.error("SOLUSI - Set salah satu environment variable di Koyeb:")
                    logger.error("  1. WEBHOOK_URL=https://nama-app.koyeb.app/webhook")
                    logger.error("  2. KOYEB_PUBLIC_DOMAIN=nama-app.koyeb.app")
                    logger.error("")
                    logger.error("Contoh nilai KOYEB_PUBLIC_DOMAIN:")
                    logger.error("  - trading-bot-xyz123.koyeb.app")
                    logger.error("=" * 60)
                    return
                
                logger.info("=" * 60)
                logger.info("🔗 WEBHOOK MODE AKTIF")
                logger.info("=" * 60)
                logger.info(f"Webhook URL: {self.config.WEBHOOK_URL[:50]}...")
                logger.info(f"Port: {self.config.HEALTH_CHECK_PORT}")
                logger.info(f"Is Koyeb: {self.config.IS_KOYEB}")
                logger.info("=" * 60)
                
                WEBHOOK_RETRY_DELAY = 30
                MAX_WEBHOOK_RETRIES = 10
                webhook_retry_count = 0
                webhook_setup_success = False
                
                while not self._is_shutting_down and not webhook_setup_success:
                    try:
                        webhook_retry_count += 1
                        logger.info(f"🔄 Webhook setup attempt {webhook_retry_count}/{MAX_WEBHOOK_RETRIES}...")
                        
                        await self.run_webhook()
                        webhook_setup_success = True
                        logger.info("✅ Webhook setup berhasil!")
                        
                    except asyncio.CancelledError:
                        logger.info("🛑 Webhook setup cancelled")
                        raise
                    except Exception as e:
                        logger.error(f"❌ Webhook setup gagal: {type(e).__name__}: {e}")
                        
                        if webhook_retry_count >= MAX_WEBHOOK_RETRIES:
                            logger.error(f"❌ Webhook setup gagal setelah {MAX_WEBHOOK_RETRIES} percobaan!")
                            raise
                        
                        logger.info(f"⏳ Retry dalam {WEBHOOK_RETRY_DELAY}s...")
                        await asyncio.sleep(WEBHOOK_RETRY_DELAY)
                
                if not webhook_setup_success:
                    logger.error("❌ Webhook setup tidak berhasil - keluar dari run()")
                    self._bot_healthy = False
                    return
                
                self._bot_healthy = True
                self._last_health_check = datetime.now()
                logger.info("✅ Bot health flag set to TRUE (webhook mode)")
                
                logger.info("🔄 Memulai keep-alive loop untuk webhook mode...")
                keep_alive_counter = 0
                consecutive_errors = 0
                MAX_CONSECUTIVE_ERRORS = 5
                
                while not self._is_shutting_down:
                    try:
                        await asyncio.sleep(30)
                        keep_alive_counter += 1
                        
                        consecutive_errors = 0
                        self._bot_healthy = True
                        self._last_health_check = datetime.now()
                        
                        if keep_alive_counter % 60 == 0:
                            active_chats = len(self.monitoring_chats)
                            active_dashboards = len(self.active_dashboards)
                            logger.info(
                                f"🤖 Telegram bot keep-alive #{keep_alive_counter} | "
                                f"Monitoring: {active_chats} chats | "
                                f"Dashboards: {active_dashboards} | "
                                f"Status: HEALTHY ✅"
                            )
                        
                    except asyncio.CancelledError:
                        logger.info("🛑 Telegram bot keep-alive loop cancelled")
                        self._bot_healthy = False
                        raise
                    except Exception as e:
                        consecutive_errors += 1
                        logger.error(f"❌ Error dalam keep-alive loop ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}): {type(e).__name__}: {e}")
                        
                        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                            logger.error(f"❌ Terlalu banyak error berturut-turut - keluar dari keep-alive loop")
                            self._bot_healthy = False
                            raise RuntimeError(f"Keep-alive loop failed after {MAX_CONSECUTIVE_ERRORS} consecutive errors")
                        
                        await asyncio.sleep(5)
                
                self._bot_healthy = False
                logger.info("🛑 Telegram bot webhook mode dihentikan (shutdown flag set)")
            else:
                import os
                import fcntl
                
                if self.config.IS_KOYEB:
                    logger.error("=" * 60)
                    logger.error("❌ KOYEB DETECTED BUT WEBHOOK MODE IS DISABLED!")
                    logger.error("=" * 60)
                    logger.error("Polling TIDAK BISA digunakan di Koyeb!")
                    logger.error("Bot akan tetap jalan tapi TIDAK BISA menerima command!")
                    logger.error("")
                    logger.error("SOLUSI: Set environment variable di Koyeb:")
                    logger.error("  TELEGRAM_WEBHOOK_MODE=true")
                    logger.error("  KOYEB_PUBLIC_DOMAIN=nama-app.koyeb.app")
                    logger.error("=" * 60)
                    return
                
                if os.path.exists(self.instance_lock_file):
                    try:
                        with open(self.instance_lock_file, 'r') as f:
                            pid_str = f.read().strip()
                            if pid_str.isdigit():
                                old_pid = int(pid_str)
                                
                                try:
                                    os.kill(old_pid, 0)
                                    logger.error(f"🔴 CRITICAL: Another bot instance is RUNNING (PID: {old_pid})!")
                                    logger.error("Multiple bot instances will cause 'Conflict: terminated by other getUpdates' errors!")
                                    logger.error(f"Solutions:")
                                    logger.error(f"  1. Kill the other instance: kill {old_pid}")
                                    logger.error(f"  2. Use webhook mode instead: TELEGRAM_WEBHOOK_MODE=true")
                                    logger.error(f"  3. Delete lock file if you're sure: rm {self.instance_lock_file}")
                                    logger.error("Bot will continue but may not work properly!")
                                except OSError:
                                    logger.warning(f"⚠️ Stale lock file detected (PID {old_pid} not running)")
                                    logger.info(f"Removing stale lock file: {self.instance_lock_file}")
                                    try:
                                        os.remove(self.instance_lock_file)
                                        logger.info("✅ Stale lock file removed successfully")
                                    except (PermissionError, OSError, IOError) as remove_error:
                                        logger.error(f"Failed to remove stale lock: {remove_error}")
                            else:
                                logger.warning(f"Invalid PID in lock file: {pid_str}")
                                logger.info("Removing invalid lock file")
                                try:
                                    os.remove(self.instance_lock_file)
                                except (PermissionError, OSError, IOError):
                                    pass
                    except (FileNotFoundError, PermissionError, OSError, IOError, ValueError) as e:
                        logger.error(f"Error reading lock file: {e}")
                        logger.info("Attempting to remove potentially corrupted lock file")
                        try:
                            os.remove(self.instance_lock_file)
                        except (PermissionError, OSError, IOError):
                            pass
                
                try:
                    with open(self.instance_lock_file, 'w') as f:
                        f.write(str(os.getpid()))
                    logger.info(f"✅ Bot instance lock created: PID {os.getpid()}")
                except (PermissionError, OSError, IOError) as e:
                    logger.warning(f"Could not create instance lock: {e}")
                
                logger.info("Starting Telegram bot polling with optimized timeouts...")
                if self.app and self.app.updater:
                    await self.app.updater.start_polling(
                        timeout=120,
                        read_timeout=60,
                        write_timeout=60,
                        connect_timeout=60,
                        pool_timeout=60,
                        allowed_updates=['message', 'callback_query'],
                        drop_pending_updates=False
                    )
                    self._bot_healthy = True
                    self._last_health_check = datetime.now()
                    logger.info("✅ Telegram bot polling started with optimized parameters!")
                    logger.info("✅ Bot health flag set to TRUE (polling mode)")
                    
                    logger.info("🔄 Memulai keep-alive loop untuk polling mode...")
                    keep_alive_counter = 0
                    consecutive_errors = 0
                    HEALTH_CHECK_INTERVAL = 30
                    
                    while not self._is_shutting_down:
                        try:
                            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
                            keep_alive_counter += 1
                            
                            if self.app and self.app.updater and self.app.updater.running:
                                consecutive_errors = 0
                                self._bot_healthy = True
                                self._last_health_check = datetime.now()
                                
                                if keep_alive_counter % 60 == 0:
                                    active_chats = len(self.monitoring_chats)
                                    active_dashboards = len(self.active_dashboards)
                                    logger.info(
                                        f"🤖 Telegram bot keep-alive #{keep_alive_counter} | "
                                        f"Monitoring: {active_chats} chats | "
                                        f"Dashboards: {active_dashboards} | "
                                        f"Mode: Polling | Status: HEALTHY ✅"
                                    )
                            else:
                                consecutive_errors += 1
                                self._bot_healthy = False
                                logger.warning(f"⚠️ Polling not running ({consecutive_errors})")
                            
                        except asyncio.CancelledError:
                            logger.info("🛑 Telegram bot polling keep-alive loop cancelled")
                            self._bot_healthy = False
                            raise
                        except Exception as e:
                            consecutive_errors += 1
                            self._bot_healthy = False
                            logger.error(f"❌ Error dalam polling keep-alive: {type(e).__name__}: {e}")
                            import random
                            jitter = random.uniform(0.5, 2.0)
                            await asyncio.sleep(jitter)
                    
                    self._bot_healthy = False
                    logger.info("🛑 Telegram bot polling mode dihentikan (shutdown flag set)")
                else:
                    self._bot_healthy = False
                    logger.error("Bot or updater not initialized, cannot start polling")
        
        except asyncio.CancelledError:
            logger.info("🛑 Telegram bot run() task cancelled")
            self._bot_healthy = False
            raise
        except Exception as e:
            logger.error(f"❌ FATAL ERROR dalam telegram_bot.run(): {type(e).__name__}: {e}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise
    
    async def stop(self):
        logger.info("=" * 50)
        logger.info("STOPPING TELEGRAM BOT")
        logger.info("=" * 50)
        
        self._is_shutting_down = True
        
        import os
        if os.path.exists(self.instance_lock_file):
            try:
                os.remove(self.instance_lock_file)
                logger.info("✅ Bot instance lock removed")
            except OSError as e:
                logger.warning(f"Could not remove instance lock: {type(e).__name__}: {e}")
        
        if not self.app:
            logger.warning("Bot app not initialized, nothing to stop")
            return
        
        await self.stop_background_cleanup_tasks()
        logger.info("✅ All background tasks and monitoring stopped")
        
        if self.config.TELEGRAM_WEBHOOK_MODE:
            logger.info("Deleting Telegram webhook...")
            try:
                await asyncio.wait_for(
                    self.app.bot.delete_webhook(drop_pending_updates=True),
                    timeout=5
                )
                logger.info("✅ Webhook deleted successfully")
            except asyncio.TimeoutError:
                logger.warning("Webhook deletion timed out after 5s")
            except (TelegramError, ConnectionError) as e:
                logger.error(f"Error deleting webhook: {e}")
        else:
            logger.info("Stopping Telegram bot polling...")
            try:
                if self.app.updater and self.app.updater.running:
                    await asyncio.wait_for(
                        self.app.updater.stop(),
                        timeout=5
                    )
                    logger.info("✅ Telegram bot polling stopped")
            except asyncio.TimeoutError:
                logger.warning("Updater stop timed out after 5s")
            except (TelegramError, RuntimeError) as e:
                logger.error(f"Error stopping updater: {e}")
        
        logger.info("Stopping Telegram application...")
        try:
            await asyncio.wait_for(self.app.stop(), timeout=5)
            logger.info("✅ Telegram application stopped")
        except asyncio.TimeoutError:
            logger.warning("App stop timed out after 5s")
        except (TelegramError, RuntimeError) as e:
            logger.error(f"Error stopping app: {e}")
        
        logger.info("Shutting down Telegram application...")
        try:
            await asyncio.wait_for(self.app.shutdown(), timeout=5)
            logger.info("✅ Telegram application shutdown complete")
        except asyncio.TimeoutError:
            logger.warning("App shutdown timed out after 5s")
        except (TelegramError, RuntimeError) as e:
            logger.error(f"Error shutting down app: {e}")
        
        logger.info("=" * 50)
        logger.info("TELEGRAM BOT STOPPED")
        logger.info("=" * 50)
