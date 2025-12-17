import asyncio
import threading
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Tuple, Set, Callable, Any, List
import pytz
from telegram.error import TimedOut, TelegramError, NetworkError, BadRequest, Forbidden, RetryAfter
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError
from bot.logger import setup_logger
from bot.database import Position, Trade
from bot.signal_session_manager import SignalSessionManager
from bot.message_templates import MessageFormatter

logger = setup_logger('PositionTracker')

DEFAULT_OPERATION_TIMEOUT = 30.0
NOTIFICATION_TIMEOUT = 10.0
FALLBACK_NOTIFICATION_TIMEOUT = 5.0

MAX_NOTIFICATION_RETRIES = 3
NOTIFICATION_RETRY_BASE_DELAY = 1.0
NOTIFICATION_RETRY_MAX_DELAY = 8.0


class NotificationError(Exception):
    """Base exception for notification errors"""
    pass


class RetryableNotificationError(NotificationError):
    """Error yang bisa di-retry (network, timeout)"""
    pass


class NonRetryableNotificationError(NotificationError):
    """Error yang tidak bisa di-retry (bad request, forbidden)"""
    pass


class NotificationService:
    """Service untuk mengirim notifikasi Telegram dengan robust error handling.
    
    Features:
    - Retry mechanism dengan exponential backoff
    - Initialization check sebelum kirim
    - Fallback ke simple text jika Markdown fail
    - Timeout handling
    - Error categorization (retryable vs non-retryable)
    """
    
    RETRYABLE_ERRORS = (NetworkError, TimedOut, RetryAfter, asyncio.TimeoutError)
    NON_RETRYABLE_ERRORS = (BadRequest, Forbidden)
    
    def __init__(self, telegram_app=None, max_retries: int = MAX_NOTIFICATION_RETRIES,
                 base_delay: float = NOTIFICATION_RETRY_BASE_DELAY,
                 max_delay: float = NOTIFICATION_RETRY_MAX_DELAY,
                 timeout: float = NOTIFICATION_TIMEOUT):
        self.telegram_app = telegram_app
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self._stats = {
            'sent': 0,
            'failed': 0,
            'retried': 0,
            'fallback_used': 0
        }
    
    def is_initialized(self) -> bool:
        """Check apakah telegram_app dan bot sudah siap."""
        if self.telegram_app is None:
            return False
        if not hasattr(self.telegram_app, 'bot') or self.telegram_app.bot is None:
            return False
        return True
    
    def _categorize_error(self, error: Exception) -> str:
        """Kategorikan error sebagai retryable atau non-retryable."""
        if isinstance(error, self.RETRYABLE_ERRORS):
            return 'retryable'
        elif isinstance(error, self.NON_RETRYABLE_ERRORS):
            return 'non-retryable'
        elif isinstance(error, TelegramError):
            error_msg = str(error).lower()
            if 'timed out' in error_msg or 'network' in error_msg:
                return 'retryable'
            return 'non-retryable'
        elif isinstance(error, RuntimeError):
            error_msg = str(error).lower()
            if 'not initialized' in error_msg or 'httpxrequest' in error_msg:
                return 'retryable'
            return 'non-retryable'
        return 'unknown'
    
    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        delay = self.base_delay * (2 ** attempt)
        return min(delay, self.max_delay)
    
    async def send_notification(self, user_id: int, message: str, 
                                 parse_mode: str = 'Markdown') -> bool:
        """Kirim notifikasi dengan retry dan fallback.
        
        Args:
            user_id: Telegram user ID
            message: Pesan yang akan dikirim
            parse_mode: Format parsing ('Markdown', 'HTML', atau None)
            
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        if not self.is_initialized():
            logger.warning(f"📵 NotificationService tidak terinisialisasi - notifikasi ke user {user_id} diabaikan")
            return False
        
        if self.telegram_app is None or not hasattr(self.telegram_app, 'bot') or self.telegram_app.bot is None:
            logger.warning(f"📵 telegram_app.bot tidak tersedia - notifikasi ke user {user_id} diabaikan")
            return False
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"📤 Mengirim notifikasi ke user {user_id} (attempt {attempt + 1}/{self.max_retries})...")
                await asyncio.wait_for(
                    self.telegram_app.bot.send_message(
                        chat_id=user_id,
                        text=message,
                        parse_mode=parse_mode
                    ),
                    timeout=self.timeout
                )
                self._stats['sent'] += 1
                logger.info(f"✅ Notifikasi BERHASIL terkirim ke user {user_id} (attempt {attempt + 1})")
                return True
                
            except self.NON_RETRYABLE_ERRORS as e:
                logger.error(f"❌ Non-retryable error mengirim notifikasi ke user {user_id}: {type(e).__name__}: {e}")
                self._stats['failed'] += 1
                
                if parse_mode and attempt == 0:
                    logger.info(f"🔄 Mencoba fallback ke plain text untuk user {user_id}")
                    try:
                        if self.telegram_app is not None and hasattr(self.telegram_app, 'bot') and self.telegram_app.bot is not None:
                            plain_message = message.replace('*', '').replace('_', '').replace('`', '')
                            await asyncio.wait_for(
                                self.telegram_app.bot.send_message(
                                    chat_id=user_id,
                                    text=plain_message
                                ),
                                timeout=FALLBACK_NOTIFICATION_TIMEOUT
                            )
                            self._stats['fallback_used'] += 1
                            logger.info(f"✅ Fallback notification berhasil ke user {user_id}")
                            return True
                    except Exception as fallback_err:
                        logger.error(f"❌ Fallback juga gagal: {fallback_err}")
                return False
                
            except self.RETRYABLE_ERRORS as e:
                last_error = e
                self._stats['retried'] += 1
                
                if attempt < self.max_retries - 1:
                    backoff = self._calculate_backoff(attempt)
                    logger.warning(
                        f"⚠️ Retryable error (attempt {attempt + 1}/{self.max_retries}) "
                        f"mengirim notifikasi ke user {user_id}: {type(e).__name__}. "
                        f"Retry dalam {backoff:.1f}s..."
                    )
                    await asyncio.sleep(backoff)
                else:
                    logger.error(
                        f"❌ Gagal mengirim notifikasi ke user {user_id} setelah {self.max_retries} percobaan: "
                        f"{type(e).__name__}: {e}"
                    )
                    
            except RuntimeError as e:
                error_msg = str(e).lower()
                if 'not initialized' in error_msg or 'httpxrequest' in error_msg:
                    last_error = e
                    self._stats['retried'] += 1
                    
                    if attempt < self.max_retries - 1:
                        backoff = self._calculate_backoff(attempt)
                        logger.warning(
                            f"⚠️ HTTPXRequest not initialized (attempt {attempt + 1}/{self.max_retries}) "
                            f"untuk user {user_id}. Retry dalam {backoff:.1f}s..."
                        )
                        await asyncio.sleep(backoff)
                    else:
                        logger.error(f"❌ HTTPXRequest masih belum siap setelah {self.max_retries} percobaan")
                else:
                    logger.error(f"❌ RuntimeError mengirim notifikasi: {e}")
                    self._stats['failed'] += 1
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Unexpected error mengirim notifikasi ke user {user_id}: {type(e).__name__}: {e}")
                self._stats['failed'] += 1
                return False
        
        self._stats['failed'] += 1
        return False
    
    def get_stats(self) -> Dict[str, int]:
        """Get notification statistics."""
        return self._stats.copy()
    
    def reset_stats(self) -> None:
        """Reset notification statistics."""
        self._stats = {'sent': 0, 'failed': 0, 'retried': 0, 'fallback_used': 0}


TICK_QUEUE_TIMEOUT = 5.0
TASK_CLEANUP_TIMEOUT = 10.0
LONG_RUNNING_TASK_THRESHOLD = 60.0
TASK_AUTO_CANCEL_THRESHOLD = 300.0
SLOW_TASK_ALERT_COUNT = 3
TASK_MONITOR_INTERVAL = 30.0
MAX_EXCEPTION_HISTORY = 100
STALE_POSITION_TIMEOUT = 600  # 10 minutes
STALE_CLEANUP_INTERVAL = 60   # Check every 60 seconds


@dataclass
class TaskExceptionRecord:
    """Record of a task exception for debugging"""
    task_name: str
    exception_type: str
    exception_message: str
    traceback_str: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(pytz.UTC))


@dataclass
class TaskCompletionResult:
    """Result of wait_for_completion with detailed information"""
    completed: bool
    total_tasks: int
    completed_tasks: int
    pending_tasks: List[str]
    timed_out_tasks: List[str]
    cancelled_tasks: List[str]
    error_tasks: List[str]
    elapsed_time: float = 0.0

class PositionError(Exception):
    """Base exception for position tracking errors"""
    pass

class ValidationError(PositionError):
    """Position data validation error"""
    pass

def validate_position_data(user_id: int, trade_id: int, signal_type: str,
                          entry_price: float, stop_loss: float, take_profit: float) -> Tuple[bool, Optional[str]]:
    """Validate position data before processing"""
    try:
        if user_id is None or user_id <= 0:
            return False, f"Invalid user_id: {user_id}"
        
        if trade_id is None or trade_id <= 0:
            return False, f"Invalid trade_id: {trade_id}"
        
        if signal_type not in ['BUY', 'SELL']:
            return False, f"Invalid signal_type: {signal_type}. Must be 'BUY' or 'SELL'"
        
        if entry_price is None or entry_price <= 0:
            return False, f"Invalid entry_price: {entry_price}"
        
        if stop_loss is None or stop_loss <= 0:
            return False, f"Invalid stop_loss: {stop_loss}"
        
        if take_profit is None or take_profit <= 0:
            return False, f"Invalid take_profit: {take_profit}"
        
        if signal_type == 'BUY':
            if stop_loss >= entry_price:
                return False, f"BUY: stop_loss ({stop_loss}) must be < entry_price ({entry_price})"
            if take_profit <= entry_price:
                return False, f"BUY: take_profit ({take_profit}) must be > entry_price ({entry_price})"
        else:
            if stop_loss <= entry_price:
                return False, f"SELL: stop_loss ({stop_loss}) must be > entry_price ({entry_price})"
            if take_profit >= entry_price:
                return False, f"SELL: take_profit ({take_profit}) must be < entry_price ({entry_price})"
        
        sl_distance = abs(entry_price - stop_loss)
        tp_distance = abs(entry_price - take_profit)
        
        if sl_distance < 0.10:
            return False, f"SL distance too small: ${sl_distance:.2f}"
        
        if tp_distance < 0.10:
            return False, f"TP distance too small: ${tp_distance:.2f}"
        
        return True, None
        
    except (ValueError, TypeError, AttributeError) as e:
        return False, f"Validation error: {str(e)}"

class PositionTracker:
    """Position tracker with async task lifecycle management.
    
    Task Lifecycle Tracking:
    - All spawned tasks are registered in _pending_tasks set
    - Done callbacks automatically remove completed tasks and drain exceptions
    - wait_for_completion() allows waiting for all tasks to finish
    - SignalSessionManager states are resolved on task completion
    """
    MAX_SLIPPAGE_PIPS = 5.0
    
    def __init__(self, config, db_manager, risk_manager, alert_system=None, user_manager=None, 
                 chart_generator=None, market_data=None, telegram_app=None, signal_session_manager=None,
                 signal_quality_tracker=None):
        self.config = config
        self.db = db_manager
        self.risk_manager = risk_manager
        self.alert_system = alert_system
        self.user_manager = user_manager
        self.chart_generator = chart_generator
        self.market_data = market_data
        self.telegram_app = telegram_app
        self.signal_session_manager = signal_session_manager
        self.signal_quality_tracker = signal_quality_tracker
        self.active_positions = {}
        self.monitoring = False
        
        self._position_lock = asyncio.Lock()
        self._pending_tasks: Set[asyncio.Task] = set()
        self._pending_tasks_lock = threading.RLock()
        self._shutdown_event = asyncio.Event()
        self._completion_event = asyncio.Event()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._task_monitor_task: Optional[asyncio.Task] = None
        self._task_callbacks: Dict[str, Callable] = {}
        self._task_results: Dict[str, Any] = {}
        
        self._task_start_times: Dict[str, datetime] = {}
        self._exception_history: List[TaskExceptionRecord] = []
        self._slow_task_counts: Dict[str, int] = defaultdict(int)
        self._cancelled_task_names: Set[str] = set()
        
        self._task_state_lock = threading.Lock()
        self._history_lock = threading.Lock()
        self._trailing_stop_lock = threading.Lock()
        
        self.long_running_threshold = LONG_RUNNING_TASK_THRESHOLD
        self.auto_cancel_threshold = TASK_AUTO_CANCEL_THRESHOLD
        self.slow_task_alert_count = SLOW_TASK_ALERT_COUNT
        
        self._trailing_stop_last_notify: Dict[int, Dict[int, datetime]] = {}
        self._trailing_stop_notify_cooldown = getattr(config, 'TRAILING_STOP_NOTIFY_COOLDOWN', 30.0)
        
        self._notification_service = NotificationService(telegram_app=telegram_app)
    
    def _on_task_done(self, task: asyncio.Task, task_name: Optional[str] = None) -> None:
        """Callback invoked when a tracked task completes.
        
        Handles:
        - Exception draining and logging with full traceback
        - Task removal from pending set
        - Exception history tracking for debugging
        - InvalidStateError graceful handling
        - SignalSessionManager state resolution if applicable
        - Custom callback execution
        """
        effective_name = task_name or task.get_name()
        
        try:
            with self._pending_tasks_lock:
                self._pending_tasks.discard(task)
            
            with self._task_state_lock:
                if effective_name in self._task_start_times:
                    start_time = self._task_start_times.pop(effective_name)
                    elapsed = (datetime.now(pytz.UTC) - start_time).total_seconds()
                    if elapsed > self.long_running_threshold:
                        logger.warning(f"⏱️ Task {effective_name} took {elapsed:.2f}s to complete (threshold: {self.long_running_threshold}s)")
                        with self._history_lock:
                            self._slow_task_counts[effective_name] += 1
        except (KeyError, AttributeError, TypeError) as cleanup_err:
            logger.debug(f"Error during task cleanup: {cleanup_err}")
        
        try:
            if task.cancelled():
                with self._history_lock:
                    self._cancelled_task_names.add(effective_name)
                logger.debug(f"Task {effective_name} was cancelled")
                self._cleanup_task_state(effective_name)
                return
        except asyncio.InvalidStateError as e:
            logger.debug(f"InvalidStateError checking cancelled state for {effective_name}: {e}")
            self._cleanup_task_state(effective_name)
            return
        
        exception = None
        try:
            exception = task.exception()
        except asyncio.CancelledError:
            with self._history_lock:
                self._cancelled_task_names.add(effective_name)
            logger.debug(f"Task {effective_name} was cancelled (detected via CancelledError)")
            self._cleanup_task_state(effective_name)
            return
        except asyncio.InvalidStateError as e:
            logger.debug(f"InvalidStateError getting exception for {effective_name}: {e}")
            self._cleanup_task_state(effective_name)
            return
        
        if exception:
            self._record_exception(effective_name, exception)
            
            tb_str = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
            logger.error(
                f"Task {effective_name} raised exception: {type(exception).__name__}: {exception}\n"
                f"Traceback:\n{tb_str}"
            )
            
            if self.alert_system:
                try:
                    asyncio.create_task(
                        self.alert_system.send_system_error(
                            f"⚠️ Task Error: {effective_name}\n"
                            f"Exception: {type(exception).__name__}: {str(exception)}\n"
                            f"Check logs for full traceback"
                        )
                    )
                except (RuntimeError, TypeError, ValueError) as alert_err:
                    logger.error(f"Failed to send task error alert: {alert_err}")
        else:
            try:
                result = task.result()
                if task_name:
                    with self._task_state_lock:
                        self._task_results[task_name] = result
                logger.debug(f"Task {effective_name} completed successfully")
            except asyncio.CancelledError:
                with self._history_lock:
                    self._cancelled_task_names.add(effective_name)
                logger.debug(f"Task {effective_name} was cancelled (detected via result())")
            except asyncio.InvalidStateError as e:
                logger.debug(f"InvalidStateError getting result for {effective_name}: {e}")
            except (ValueError, TypeError, RuntimeError) as result_err:
                logger.error(f"Error getting result for {effective_name}: {result_err}")
        
        callback = None
        with self._task_state_lock:
            if task_name and task_name in self._task_callbacks:
                try:
                    callback = self._task_callbacks.pop(task_name)
                except (KeyError, ValueError):
                    callback = None
        if callback:
            try:
                callback(task)
            except (ValueError, TypeError, RuntimeError, AttributeError) as cb_err:
                logger.error(f"Task callback error for {task_name}: {cb_err}")
        
        self._cleanup_task_state(effective_name)
        
        with self._pending_tasks_lock:
            if len(self._pending_tasks) == 0:
                self._completion_event.set()
    
    def _record_exception(self, task_name: str, exception: BaseException) -> None:
        """Record exception in history for debugging"""
        try:
            tb_str = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
            record = TaskExceptionRecord(
                task_name=task_name,
                exception_type=type(exception).__name__,
                exception_message=str(exception),
                traceback_str=tb_str
            )
            with self._history_lock:
                self._exception_history.append(record)
                
                if len(self._exception_history) > MAX_EXCEPTION_HISTORY:
                    self._exception_history = self._exception_history[-MAX_EXCEPTION_HISTORY:]
        except (ValueError, TypeError, AttributeError) as e:
            logger.debug(f"Error recording exception: {e}")
    
    def _cleanup_task_state(self, task_name: str) -> None:
        """Clean up task-related state"""
        with self._task_state_lock:
            self._task_start_times.pop(task_name, None)
            self._task_callbacks.pop(task_name, None)
    
    def get_exception_history(self, limit: int = 20) -> List[TaskExceptionRecord]:
        """Get recent exception history for debugging"""
        with self._history_lock:
            return self._exception_history[-limit:]
    
    def get_slow_task_stats(self) -> Dict[str, int]:
        """Get statistics on slow tasks"""
        with self._history_lock:
            return dict(self._slow_task_counts)
    
    async def _safe_send_notification(self, user_id: int, message: str, 
                                       parse_mode: str = 'Markdown') -> bool:
        """Helper method untuk mengirim notifikasi Telegram dengan aman.
        
        Method ini menggunakan NotificationService untuk:
        - Check initialization sebelum kirim
        - Retry mechanism dengan exponential backoff
        - Fallback ke plain text jika Markdown fail
        - Timeout handling
        
        Args:
            user_id: Telegram user ID
            message: Pesan yang akan dikirim
            parse_mode: Format parsing ('Markdown', 'HTML', atau None)
            
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        if self.telegram_app is None:
            logger.warning(f"📵 telegram_app is None - notifikasi ke user {user_id} diabaikan")
            return False
        
        if not hasattr(self.telegram_app, 'bot') or self.telegram_app.bot is None:
            logger.warning(f"📵 telegram_app.bot is None - notifikasi ke user {user_id} diabaikan")
            return False
        
        self._notification_service.telegram_app = self.telegram_app
        
        try:
            result = await self._notification_service.send_notification(
                user_id=user_id,
                message=message,
                parse_mode=parse_mode
            )
            if result:
                logger.debug(f"✅ Notifikasi berhasil dikirim ke user {user_id}")
            else:
                logger.warning(f"⚠️ Notifikasi gagal dikirim ke user {user_id}")
            return result
        except Exception as e:
            logger.error(f"❌ Exception saat mengirim notifikasi ke user {user_id}: {type(e).__name__}: {e}")
            return False
    
    def _check_trailing_stop_notify_cooldown(self, user_id: int, position_id: int) -> bool:
        """Check apakah sudah waktunya untuk kirim trailing stop notification.
        
        Args:
            user_id: User ID
            position_id: Position ID
            
        Returns:
            bool: True jika boleh kirim notifikasi, False jika masih dalam cooldown
        """
        now = datetime.now(pytz.UTC)
        
        with self._trailing_stop_lock:
            if user_id not in self._trailing_stop_last_notify:
                self._trailing_stop_last_notify[user_id] = {}
                return True
            
            user_notify_times = self._trailing_stop_last_notify[user_id]
            last_notify = user_notify_times.get(position_id)
            
            if last_notify is None:
                return True
            
            elapsed = (now - last_notify).total_seconds()
            return elapsed >= self._trailing_stop_notify_cooldown
    
    def _update_trailing_stop_notify_time(self, user_id: int, position_id: int) -> None:
        """Update waktu terakhir kirim trailing stop notification.
        
        Args:
            user_id: User ID
            position_id: Position ID
        """
        now = datetime.now(pytz.UTC)
        
        with self._trailing_stop_lock:
            if user_id not in self._trailing_stop_last_notify:
                self._trailing_stop_last_notify[user_id] = {}
            
            self._trailing_stop_last_notify[user_id][position_id] = now
    
    def _cleanup_trailing_stop_notify(self, user_id: int, position_id: int) -> None:
        """Cleanup trailing stop notification tracking untuk posisi yang ditutup.
        
        Args:
            user_id: User ID
            position_id: Position ID
        """
        with self._trailing_stop_lock:
            if user_id in self._trailing_stop_last_notify:
                self._trailing_stop_last_notify[user_id].pop(position_id, None)
                if not self._trailing_stop_last_notify[user_id]:
                    del self._trailing_stop_last_notify[user_id]
    
    def _create_tracked_task(self, coro, name: Optional[str] = None, 
                             on_complete: Optional[Callable] = None,
                             resolve_session_user_id: Optional[int] = None,
                             timeout: Optional[float] = None) -> asyncio.Task:
        """Create a task and track it for proper cleanup on shutdown.
        
        Args:
            coro: Coroutine to execute
            name: Optional task name for identification
            on_complete: Optional callback when task completes
            resolve_session_user_id: If set, resolve SignalSessionManager for this user on completion
            timeout: Optional task timeout in seconds (auto-cancels if exceeded)
            
        Returns:
            asyncio.Task: The created and tracked task
        """
        effective_name = name or f"task_{id(coro)}"
        
        if timeout:
            async def wrapped_coro():
                try:
                    return await asyncio.wait_for(coro, timeout=timeout)
                except asyncio.TimeoutError:
                    logger.warning(f"⏱️ Task {effective_name} timed out after {timeout}s")
                    raise
            task = asyncio.create_task(wrapped_coro(), name=effective_name)
        else:
            task = asyncio.create_task(coro, name=effective_name)
        
        with self._pending_tasks_lock:
            self._pending_tasks.add(task)
            self._completion_event.clear()
        
        with self._task_state_lock:
            self._task_start_times[effective_name] = datetime.now(pytz.UTC)
            
            if on_complete:
                self._task_callbacks[effective_name] = on_complete
        
        def done_callback(t: asyncio.Task):
            self._on_task_done(t, effective_name)
            
            if resolve_session_user_id and self.signal_session_manager:
                try:
                    if not t.cancelled():
                        exc = None
                        try:
                            exc = t.exception()
                        except (asyncio.CancelledError, asyncio.InvalidStateError):
                            pass
                        if exc is None:
                            asyncio.create_task(
                                self._resolve_session_state(resolve_session_user_id)
                            )
                except (asyncio.CancelledError, asyncio.TimeoutError, ValueError, TypeError, AttributeError) as e:
                    logger.error(f"Error resolving session state: {e}")
        
        task.add_done_callback(done_callback)
        return task
    
    async def _resolve_session_state(self, user_id: int) -> None:
        """Resolve SignalSessionManager state after task completion.
        
        PENTING: Method ini hanya membersihkan orphaned sessions, BUKAN menghentikan monitoring.
        Session cleanup dan monitoring adalah dua hal yang terpisah:
        - Session = satu siklus sinyal (dari signal dikirim sampai posisi ditutup)
        - Monitoring = loop yang terus berjalan untuk mendeteksi sinyal baru
        
        Setelah posisi ditutup, session di-end agar user bisa menerima sinyal baru,
        tapi monitoring harus TETAP BERJALAN selama user tidak menjalankan /stopmonitor.
        """
        if not self.signal_session_manager:
            return
        
        try:
            async with self._position_lock:
                has_active = user_id in self.active_positions and len(self.active_positions[user_id]) > 0
            
            if not has_active:
                session = self.signal_session_manager.get_active_session(user_id)
                if session:
                    # Hanya end session jika masih ada session aktif yang orphaned
                    # CATATAN: Ini TIDAK menghentikan monitoring, hanya membersihkan state session
                    logger.info(f"🧹 Resolving orphaned session state for user {user_id} (session cleanup only, monitoring continues)")
                    await self.signal_session_manager.end_session(user_id, 'RESOLVED')
                else:
                    logger.debug(f"No orphaned session to resolve for user {user_id} - session already cleaned up")
        except (asyncio.CancelledError, asyncio.TimeoutError, KeyError, ValueError, AttributeError) as e:
            logger.error(f"Error resolving session state for user {user_id}: {e}")
    
    async def wait_for_completion(self, timeout: Optional[float] = None, 
                                   progress_interval: float = 5.0) -> TaskCompletionResult:
        """Wait for all pending tasks to complete with detailed progress tracking.
        
        Args:
            timeout: Maximum time to wait in seconds. None means wait indefinitely.
            progress_interval: Interval for logging progress updates.
            
        Returns:
            TaskCompletionResult: Detailed result with completion status and task information
        """
        start_time = datetime.now(pytz.UTC)
        with self._pending_tasks_lock:
            initial_count = len(self._pending_tasks)
        
            if not self._pending_tasks:
                return TaskCompletionResult(
                    completed=True,
                    total_tasks=0,
                    completed_tasks=0,
                    pending_tasks=[],
                    timed_out_tasks=[],
                    cancelled_tasks=[],
                    error_tasks=[],
                    elapsed_time=0.0
                )
        
        logger.info(f"📊 Waiting for {initial_count} pending tasks to complete...")
        pending_names_initial = self.get_pending_task_names()
        
        async def progress_reporter():
            """Report progress at intervals"""
            last_count = initial_count
            while True:
                await asyncio.sleep(progress_interval)
                with self._pending_tasks_lock:
                    current_count = len(self._pending_tasks)
                if current_count != last_count:
                    completed = initial_count - current_count
                    logger.info(f"📊 Progress: {completed}/{initial_count} tasks completed, {current_count} remaining")
                    if current_count > 0:
                        remaining = self.get_pending_task_names()[:5]
                        with self._pending_tasks_lock:
                            logger.debug(f"   Remaining tasks: {remaining}{'...' if len(self._pending_tasks) > 5 else ''}")
                    last_count = current_count
        
        progress_task = asyncio.create_task(progress_reporter())
        
        completed_ok = False
        timed_out_tasks = []
        error_tasks = []
        
        try:
            if timeout:
                try:
                    await asyncio.wait_for(self._completion_event.wait(), timeout=timeout)
                    completed_ok = True
                except asyncio.TimeoutError:
                    timed_out_tasks = self.get_pending_task_names()
                    with self._pending_tasks_lock:
                        pending_count = len(self._pending_tasks)
                    logger.warning(
                        f"⏰ Timeout after {timeout}s waiting for task completion. "
                        f"{pending_count} tasks still pending: {timed_out_tasks[:10]}{'...' if len(timed_out_tasks) > 10 else ''}"
                    )
            else:
                await self._completion_event.wait()
                completed_ok = True
        finally:
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass
        
        elapsed = (datetime.now(pytz.UTC) - start_time).total_seconds()
        pending_final = self.get_pending_task_names()
        with self._history_lock:
            cancelled = list(self._cancelled_task_names)
            error_tasks = [record.task_name for record in self._exception_history 
                          if record.timestamp >= start_time]
        
        completed_count = initial_count - len(pending_final)
        
        result = TaskCompletionResult(
            completed=completed_ok and len(pending_final) == 0,
            total_tasks=initial_count,
            completed_tasks=completed_count,
            pending_tasks=pending_final,
            timed_out_tasks=timed_out_tasks,
            cancelled_tasks=cancelled,
            error_tasks=error_tasks,
            elapsed_time=elapsed
        )
        
        if result.completed:
            logger.info(f"✅ All {initial_count} tasks completed in {elapsed:.2f}s")
        else:
            logger.warning(
                f"⚠️ Partial completion: {completed_count}/{initial_count} tasks completed in {elapsed:.2f}s. "
                f"Pending: {len(pending_final)}, Errors: {len(error_tasks)}"
            )
        
        return result
    
    def get_pending_task_count(self) -> int:
        """Get the number of pending tasks."""
        with self._pending_tasks_lock:
            return len(self._pending_tasks)
    
    def get_pending_task_names(self) -> list:
        """Get names of all pending tasks."""
        with self._pending_tasks_lock:
            return [t.get_name() for t in self._pending_tasks]
    
    def get_task_ages(self) -> Dict[str, float]:
        """Get age in seconds for each running task"""
        now = datetime.now(pytz.UTC)
        ages = {}
        with self._task_state_lock:
            for task_name, start_time in self._task_start_times.items():
                ages[task_name] = (now - start_time).total_seconds()
        return ages
    
    async def _monitor_task_timeouts(self) -> None:
        """Background task to monitor for long-running and stuck tasks.
        
        - Logs warnings for tasks exceeding long_running_threshold
        - Auto-cancels tasks exceeding auto_cancel_threshold
        - Sends alerts for recurring slow tasks
        """
        logger.info(f"⏱️ Task timeout monitor started (warning: {self.long_running_threshold}s, auto-cancel: {self.auto_cancel_threshold}s)")
        
        try:
            while not self._shutdown_event.is_set():
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=TASK_MONITOR_INTERVAL
                    )
                    break
                except asyncio.TimeoutError:
                    pass
                
                with self._pending_tasks_lock:
                    if not self._pending_tasks:
                        continue
                
                now = datetime.now(pytz.UTC)
                tasks_to_cancel = []
                
                with self._pending_tasks_lock:
                    pending_tasks_copy = list(self._pending_tasks)
                for task in pending_tasks_copy:
                    if task.done():
                        continue
                    
                    task_name = task.get_name()
                    with self._task_state_lock:
                        start_time = self._task_start_times.get(task_name)
                    
                    if not start_time:
                        continue
                    
                    age = (now - start_time).total_seconds()
                    
                    if age >= self.auto_cancel_threshold:
                        logger.warning(
                            f"🚨 Task {task_name} exceeded auto-cancel threshold "
                            f"({age:.1f}s > {self.auto_cancel_threshold}s) - scheduling cancellation"
                        )
                        tasks_to_cancel.append((task, task_name, age))
                        
                    elif age >= self.long_running_threshold:
                        with self._history_lock:
                            slow_count = self._slow_task_counts.get(task_name, 0)
                        
                        if slow_count == 0:
                            logger.warning(
                                f"⏱️ Task {task_name} is running long ({age:.1f}s > {self.long_running_threshold}s)"
                            )
                        
                        if slow_count >= self.slow_task_alert_count:
                            if slow_count == self.slow_task_alert_count:
                                logger.error(
                                    f"🔥 Recurring slow task detected: {task_name} "
                                    f"(slow {slow_count} times)"
                                )
                                if self.alert_system:
                                    try:
                                        asyncio.create_task(
                                            self.alert_system.send_system_error(
                                                f"🔥 Recurring Slow Task Alert\n\n"
                                                f"Task: {task_name}\n"
                                                f"Slow count: {slow_count} times\n"
                                                f"Current age: {age:.1f}s\n"
                                                f"Threshold: {self.long_running_threshold}s\n\n"
                                                f"Consider investigating this task for performance issues"
                                            )
                                        )
                                    except (RuntimeError, TypeError, ValueError) as e:
                                        logger.error(f"Failed to send slow task alert: {e}")
                
                for task, task_name, age in tasks_to_cancel:
                    try:
                        logger.warning(f"🛑 Auto-cancelling stuck task: {task_name} (age: {age:.1f}s)")
                        task.cancel()
                        
                        try:
                            await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
                        except asyncio.TimeoutError:
                            logger.error(f"❌ Task {task_name} did not respond to auto-cancel")
                        except asyncio.CancelledError:
                            logger.info(f"✓ Task {task_name} auto-cancelled successfully")
                            
                    except (asyncio.CancelledError, asyncio.TimeoutError, RuntimeError) as e:
                        logger.error(f"Error auto-cancelling task {task_name}: {e}")
                        
        except asyncio.CancelledError:
            logger.info("Task timeout monitor cancelled")
        except (RuntimeError, KeyError, ValueError, AttributeError) as e:
            logger.error(f"Error in task timeout monitor: {e}")
        finally:
            logger.info("Task timeout monitor stopped")
    
    def start_task_monitor(self) -> asyncio.Task:
        """Start the background task timeout monitor.
        
        Returns:
            asyncio.Task: The monitor task
        """
        if self._task_monitor_task and not self._task_monitor_task.done():
            logger.warning("Task monitor already running")
            return self._task_monitor_task
        
        self._task_monitor_task = asyncio.create_task(
            self._monitor_task_timeouts(),
            name="task_timeout_monitor"
        )
        return self._task_monitor_task
    
    async def stop_task_monitor(self) -> None:
        """Stop the task timeout monitor"""
        if self._task_monitor_task and not self._task_monitor_task.done():
            self._task_monitor_task.cancel()
            try:
                await self._task_monitor_task
            except asyncio.CancelledError:
                pass
            self._task_monitor_task = None
            logger.info("Task timeout monitor stopped")
    
    def configure_timeouts(self, long_running_threshold: Optional[float] = None,
                           auto_cancel_threshold: Optional[float] = None,
                           slow_task_alert_count: Optional[int] = None) -> None:
        """Configure task timeout thresholds.
        
        Args:
            long_running_threshold: Seconds before logging warning for slow tasks
            auto_cancel_threshold: Seconds before auto-cancelling stuck tasks
            slow_task_alert_count: Number of slow occurrences before alerting
        """
        if long_running_threshold is not None:
            self.long_running_threshold = long_running_threshold
            logger.info(f"Long running threshold set to {long_running_threshold}s")
        
        if auto_cancel_threshold is not None:
            self.auto_cancel_threshold = auto_cancel_threshold
            logger.info(f"Auto-cancel threshold set to {auto_cancel_threshold}s")
        
        if slow_task_alert_count is not None:
            self.slow_task_alert_count = slow_task_alert_count
            logger.info(f"Slow task alert count set to {slow_task_alert_count}")
    
    def get_task_monitoring_stats(self) -> Dict[str, Any]:
        """Get current task monitoring statistics.
        
        Returns:
            Dict with current monitoring statistics
        """
        ages = self.get_task_ages()
        
        with self._pending_tasks_lock:
            pending_count = len(self._pending_tasks)
        
        with self._history_lock:
            slow_task_counts = dict(self._slow_task_counts)
            exception_count = len(self._exception_history)
            recent_exceptions = [
                {"task": r.task_name, "type": r.exception_type, "time": r.timestamp.isoformat()}
                for r in self._exception_history[-5:]
            ]
            cancelled_tasks = list(self._cancelled_task_names)
        
        return {
            "pending_count": pending_count,
            "pending_tasks": self.get_pending_task_names(),
            "task_ages": ages,
            "slow_task_counts": slow_task_counts,
            "exception_count": exception_count,
            "recent_exceptions": recent_exceptions,
            "cancelled_tasks": cancelled_tasks,
            "thresholds": {
                "long_running": self.long_running_threshold,
                "auto_cancel": self.auto_cancel_threshold,
                "slow_alert_count": self.slow_task_alert_count
            },
            "monitor_running": self._task_monitor_task is not None and not self._task_monitor_task.done()
        }
    
    async def _cancel_pending_tasks(self, timeout: float = TASK_CLEANUP_TIMEOUT,
                                     per_task_timeout: float = 2.0,
                                     force_cancel_on_timeout: bool = True) -> Dict[str, str]:
        """Cancel all pending tasks with proper cleanup and detailed logging.
        
        Args:
            timeout: Maximum total time to wait for all task cancellations
            per_task_timeout: Timeout for individual task cancellation attempts
            force_cancel_on_timeout: Whether to force cancel tasks that don't respond
            
        Returns:
            Dict mapping task names to their final status (cancelled, timeout, error, etc.)
        """
        cancellation_results: Dict[str, str] = {}
        
        with self._pending_tasks_lock:
            if not self._pending_tasks:
                logger.info("No pending tasks to cancel")
                return cancellation_results
        
            task_count = len(self._pending_tasks)
            logger.info(f"🛑 Initiating cancellation of {task_count} pending tasks (timeout: {timeout}s)...")
        
            tasks_to_cancel = list(self._pending_tasks)
        task_names = {task: task.get_name() for task in tasks_to_cancel}
        
        uncancellable_tasks: List[str] = []
        
        for task in tasks_to_cancel:
            task_name = task_names[task]
            if not task.done():
                try:
                    task.cancel()
                    logger.debug(f"  Cancellation requested for: {task_name}")
                except (asyncio.InvalidStateError, RuntimeError, AttributeError) as e:
                    logger.warning(f"  ⚠️ Failed to request cancellation for {task_name}: {e}")
                    uncancellable_tasks.append(task_name)
                    cancellation_results[task_name] = f"cancel_request_failed: {e}"
        
        if uncancellable_tasks:
            logger.warning(f"⚠️ {len(uncancellable_tasks)} tasks could not be cancelled: {uncancellable_tasks}")
        
        if tasks_to_cancel:
            try:
                done, pending = await asyncio.wait(
                    tasks_to_cancel,
                    timeout=timeout,
                    return_when=asyncio.ALL_COMPLETED
                )
                
                for task in done:
                    task_name = task_names[task]
                    try:
                        if task.cancelled():
                            cancellation_results[task_name] = "cancelled"
                            logger.debug(f"  ✓ {task_name}: cancelled successfully")
                        else:
                            try:
                                exc = task.exception()
                                if exc:
                                    cancellation_results[task_name] = f"error: {type(exc).__name__}"
                                    logger.warning(f"  ⚠️ {task_name}: raised {type(exc).__name__} during cancel")
                                else:
                                    cancellation_results[task_name] = "completed"
                                    logger.debug(f"  ✓ {task_name}: completed normally")
                            except asyncio.CancelledError:
                                cancellation_results[task_name] = "cancelled"
                            except asyncio.InvalidStateError:
                                cancellation_results[task_name] = "invalid_state"
                                logger.debug(f"  ? {task_name}: invalid state")
                    except (asyncio.CancelledError, asyncio.InvalidStateError, RuntimeError) as e:
                        cancellation_results[task_name] = f"check_failed: {e}"
                        logger.error(f"  ❌ Error checking {task_name} status: {e}")
                
                if pending:
                    logger.warning(f"⏰ {len(pending)} tasks did not complete within {timeout}s timeout")
                    
                    for task in pending:
                        task_name = task_names[task]
                        cancellation_results[task_name] = "timeout"
                        
                        age = "unknown"
                        with self._task_state_lock:
                            if task_name in self._task_start_times:
                                age_seconds = (datetime.now(pytz.UTC) - self._task_start_times[task_name]).total_seconds()
                                age = f"{age_seconds:.1f}s"
                        
                        logger.warning(f"  ⏰ {task_name}: did not respond to cancel (age: {age})")
                        
                        if force_cancel_on_timeout and not task.done():
                            logger.info(f"  🔨 Force cancelling {task_name}...")
                            task.cancel()
                            try:
                                await asyncio.wait_for(asyncio.shield(task), timeout=per_task_timeout)
                            except asyncio.TimeoutError:
                                cancellation_results[task_name] = "force_timeout"
                                logger.error(f"  ❌ {task_name}: STUCK - did not respond to force cancel")
                            except asyncio.CancelledError:
                                cancellation_results[task_name] = "force_cancelled"
                                logger.info(f"  ✓ {task_name}: force cancelled")
                            except (RuntimeError, AttributeError) as e:
                                cancellation_results[task_name] = f"force_error: {e}"
                                logger.error(f"  ❌ {task_name}: error during force cancel: {e}")
                else:
                    logger.info(f"✅ All {len(done)} tasks completed cancellation gracefully")
                    
            except (asyncio.CancelledError, asyncio.TimeoutError, RuntimeError) as e:
                logger.error(f"❌ Error during task cancellation phase: {e}")
        
        with self._pending_tasks_lock:
            self._pending_tasks.clear()
        with self._task_state_lock:
            self._task_callbacks.clear()
            self._task_results.clear()
            self._task_start_times.clear()
        self._completion_event.set()
        
        cancelled_count = sum(1 for s in cancellation_results.values() if 'cancelled' in s)
        timeout_count = sum(1 for s in cancellation_results.values() if 'timeout' in s)
        error_count = sum(1 for s in cancellation_results.values() if 'error' in s)
        
        logger.info(
            f"🧹 Task cleanup complete: {cancelled_count} cancelled, "
            f"{timeout_count} timed out, {error_count} errors"
        )
        
        return cancellation_results
    
    def set_signal_session_manager(self, signal_session_manager: SignalSessionManager):
        """Set signal session manager for dependency injection"""
        self.signal_session_manager = signal_session_manager
        logger.info("Signal session manager injected to position tracker")
    
    def set_signal_quality_tracker(self, signal_quality_tracker):
        """Set signal quality tracker for result updates"""
        self.signal_quality_tracker = signal_quality_tracker
        logger.info("Signal quality tracker injected to position tracker")
    
    def check_slippage(self, expected_price: float, actual_price: float, operation: str, user_id: int, position_id: Optional[int] = None):
        """Check for excessive slippage and send alert if threshold is exceeded
        
        Args:
            expected_price: Expected price (signal price or TP/SL price)
            actual_price: Actual execution price
            operation: Operation type ('OPEN' or 'CLOSE')
            user_id: User ID
            position_id: Position ID (optional for OPEN)
        """
        try:
            slippage_dollars = abs(actual_price - expected_price)
            slippage_pips = slippage_dollars * self.config.XAUUSD_PIP_VALUE
            
            if slippage_pips > self.MAX_SLIPPAGE_PIPS:
                logger.warning(f"⚠️ High slippage detected on {operation} - Expected: ${expected_price:.2f}, Actual: ${actual_price:.2f}, Slippage: {slippage_pips:.1f} pips")
                
                if self.alert_system:
                    try:
                        self._create_tracked_task(
                            self.alert_system.send_system_error(
                                f"⚠️ High Slippage Alert\n\n"
                                f"Operation: {operation}\n"
                                f"Position ID: {position_id if position_id else 'N/A'}\n"
                                f"User: {user_id}\n"
                                f"Expected Price: ${expected_price:.2f}\n"
                                f"Actual Price: ${actual_price:.2f}\n"
                                f"Slippage: {slippage_pips:.1f} pips (>${self.MAX_SLIPPAGE_PIPS} pips threshold)\n\n"
                                f"⚠️ Check market conditions and broker execution quality"
                            ),
                            name=f"slippage_alert_{user_id}_{position_id}"
                        )
                    except (asyncio.TimeoutError, RuntimeError, ValueError) as e:
                        logger.error(f"Failed to send slippage alert: {e}")
            else:
                logger.debug(f"Slippage within threshold on {operation}: {slippage_pips:.1f} pips")
                
        except (ValueError, TypeError, AttributeError, ZeroDivisionError) as e:
            logger.error(f"Error checking slippage: {e}")
    
    def _normalize_position_dict(self, pos: Dict) -> Dict:
        """Ensure all required keys exist in position dict with safe defaults"""
        if 'original_sl' not in pos or pos['original_sl'] is None:
            pos['original_sl'] = pos.get('stop_loss', 0.0)
        if 'sl_adjustment_count' not in pos or pos['sl_adjustment_count'] is None:
            pos['sl_adjustment_count'] = 0
        if 'max_profit_reached' not in pos or pos['max_profit_reached'] is None:
            pos['max_profit_reached'] = 0.0
        if 'last_price_update' not in pos:
            pos['last_price_update'] = datetime.now(pytz.UTC)
        return pos
        
    async def add_position(self, user_id: int, trade_id: int, signal_type: str, entry_price: float,
                          stop_loss: float, take_profit: float, signal_quality_id: Optional[int] = None,
                          signal_grade: Optional[str] = 'B', confidence_score: Optional[float] = 0.5):
        """Add position with comprehensive validation and error handling
        
        Thread-safe with asyncio.Lock for active_positions access
        
        Args:
            signal_grade: Signal quality grade (A/B/C) from strategy analysis
            confidence_score: Confidence score 0.0-1.0 from signal detection
        """
        is_valid, error_msg = validate_position_data(user_id, trade_id, signal_type, entry_price, stop_loss, take_profit)
        if not is_valid:
            logger.error(f"Position validation failed: {error_msg}")
            logger.debug(f"Invalid data: user_id={user_id}, trade_id={trade_id}, signal={signal_type}, entry={entry_price}, sl={stop_loss}, tp={take_profit}")
            return None
        
        valid_grades = ['A', 'B', 'C']
        if signal_grade not in valid_grades:
            signal_grade = 'B'
        if confidence_score is None or confidence_score < 0.0 or confidence_score > 1.0:
            confidence_score = 0.5
        
        session = None
        position_id = None
        try:
            session = self.db.get_session()
            try:
                session.rollback()
            except Exception:
                pass
            
            opened_at = datetime.now(pytz.UTC)
            position = Position(
                user_id=user_id,
                trade_id=trade_id,
                ticker='XAUUSD',
                signal_type=signal_type,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                current_price=entry_price,
                unrealized_pl=0.0,
                status='ACTIVE',
                original_sl=stop_loss,
                sl_adjustment_count=0,
                max_profit_reached=0.0,
                last_price_update=datetime.now(pytz.UTC),
                signal_quality_id=signal_quality_id,
                signal_grade=signal_grade,
                confidence_score=confidence_score,
                opened_at=opened_at
            )
            session.add(position)
            session.flush()
            position_id = getattr(position, 'id', None)
            session.commit()
            
            logger.info(f"📊 Position created with Grade:{signal_grade} Confidence:{confidence_score:.2f}")
            
            async with self._position_lock:
                if user_id not in self.active_positions:
                    self.active_positions[user_id] = {}
                
                self.active_positions[user_id][position_id] = {
                    'trade_id': trade_id,
                    'signal_type': signal_type,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'original_sl': stop_loss,
                    'sl_adjustment_count': 0,
                    'max_profit_reached': 0.0,
                    'signal_quality_id': signal_quality_id,
                    'signal_grade': signal_grade,
                    'confidence_score': confidence_score,
                    'opened_at': opened_at
                }
            
            logger.info(f"✅ Position added - User:{user_id} ID:{position_id} {signal_type} @${entry_price:.2f} SL:${stop_loss:.2f} TP:${take_profit:.2f}")
            
            if self.signal_session_manager:
                try:
                    pos_id_int: Optional[int] = int(position_id) if position_id is not None else None
                    trade_id_str: Optional[str] = str(trade_id) if trade_id is not None else None
                    await asyncio.wait_for(
                        self.signal_session_manager.update_session(
                            user_id,
                            position_id=pos_id_int,
                            trade_id=trade_id_str
                        ),
                        timeout=DEFAULT_OPERATION_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout updating signal session for position {position_id}")
            
            return position_id
            
        except (IntegrityError, OperationalError, SQLAlchemyError, ValueError, TypeError) as e:
            logger.error(f"Database error adding position for User:{user_id} Trade:{trade_id}: {type(e).__name__}: {e}", exc_info=True)
            if session:
                try:
                    session.rollback()
                except (OperationalError, SQLAlchemyError) as rollback_error:
                    logger.error(f"Error during rollback: {rollback_error}")
            return None
        finally:
            if session:
                try:
                    session.close()
                except (OperationalError, SQLAlchemyError) as close_error:
                    logger.error(f"Error closing session: {close_error}")
    
    async def _apply_dynamic_sl_internal(self, user_id: int, position_id: int, pos: Dict, 
                                          current_price: float, unrealized_pl: float) -> tuple[bool, Optional[float], Dict]:
        """Internal method: Apply dynamic SL tightening when loss >= threshold
        
        Args:
            user_id: User ID
            position_id: Position ID
            pos: Position dict (will be modified in place)
            current_price: Current market price
            unrealized_pl: Unrealized P/L
            
        Returns:
            tuple[bool, Optional[float], Dict]: (sl_adjusted, new_stop_loss, updated_pos)
        """
        pos = self._normalize_position_dict(pos)
        signal_type = pos['signal_type']
        entry_price = pos['entry_price']
        stop_loss = pos['stop_loss']
        original_sl = pos.get('original_sl')
        
        if original_sl is None:
            original_sl = stop_loss
            pos['original_sl'] = stop_loss
            logger.warning(f"original_sl was None for position {position_id}, using current stop_loss")
        
        if unrealized_pl >= 0 or abs(unrealized_pl) < self.config.DYNAMIC_SL_LOSS_THRESHOLD:
            return False, None, pos
        
        original_sl_distance = abs(entry_price - original_sl)
        new_sl_distance = original_sl_distance * self.config.DYNAMIC_SL_TIGHTENING_MULTIPLIER
        
        new_stop_loss = None
        sl_adjusted = False
        
        if signal_type == 'BUY':
            new_stop_loss = entry_price - new_sl_distance
            if new_stop_loss > stop_loss:
                pos['stop_loss'] = new_stop_loss
                pos['sl_adjustment_count'] = pos.get('sl_adjustment_count', 0) + 1
                sl_adjusted = True
                logger.info(f"🛡️ Dynamic SL activated! Loss ${abs(unrealized_pl):.2f} >= ${self.config.DYNAMIC_SL_LOSS_THRESHOLD}. SL tightened from ${stop_loss:.2f} → ${new_stop_loss:.2f} (protect capital)")
                
                msg = (
                    f"🛡️ *Dynamic SL Activated*\n\n"
                    f"Position ID: {position_id}\n"
                    f"Type: {signal_type}\n"
                    f"Current Loss: ${abs(unrealized_pl):.2f}\n"
                    f"SL Updated: ${stop_loss:.2f} → ${new_stop_loss:.2f}\n"
                    f"Protection: Capital preservation mode"
                )
                await self._safe_send_notification(user_id, msg, parse_mode='Markdown')
        else:
            new_stop_loss = entry_price + new_sl_distance
            if new_stop_loss < stop_loss:
                pos['stop_loss'] = new_stop_loss
                pos['sl_adjustment_count'] = pos.get('sl_adjustment_count', 0) + 1
                sl_adjusted = True
                logger.info(f"🛡️ Dynamic SL activated! Loss ${abs(unrealized_pl):.2f} >= ${self.config.DYNAMIC_SL_LOSS_THRESHOLD}. SL tightened from ${stop_loss:.2f} → ${new_stop_loss:.2f} (protect capital)")
                
                msg = (
                    f"🛡️ *Dynamic SL Activated*\n\n"
                    f"Position ID: {position_id}\n"
                    f"Type: {signal_type}\n"
                    f"Current Loss: ${abs(unrealized_pl):.2f}\n"
                    f"SL Updated: ${stop_loss:.2f} → ${new_stop_loss:.2f}\n"
                    f"Protection: Capital preservation mode"
                )
                await self._safe_send_notification(user_id, msg, parse_mode='Markdown')
        
        return sl_adjusted, new_stop_loss if sl_adjusted else None, pos
    
    async def apply_dynamic_sl(self, user_id: int, position_id: int, current_price: float, unrealized_pl: float) -> tuple[bool, Optional[float]]:
        """Apply dynamic SL tightening when loss >= threshold
        
        Thread-safe wrapper for _apply_dynamic_sl_internal
        
        Returns:
            tuple[bool, Optional[float]]: (sl_adjusted, new_stop_loss)
        """
        async with self._position_lock:
            if user_id not in self.active_positions or position_id not in self.active_positions[user_id]:
                return False, None
            
            pos = self.active_positions[user_id][position_id].copy()
        
        sl_adjusted, new_sl, updated_pos = await self._apply_dynamic_sl_internal(
            user_id, position_id, pos, current_price, unrealized_pl
        )
        
        if sl_adjusted:
            async with self._position_lock:
                if user_id in self.active_positions and position_id in self.active_positions[user_id]:
                    self.active_positions[user_id][position_id].update(updated_pos)
        
        return sl_adjusted, new_sl
    
    async def _apply_trailing_stop_internal(self, user_id: int, position_id: int, pos: Dict,
                                             current_price: float, unrealized_pl: float) -> tuple[bool, Optional[float], Dict]:
        """Internal method: Apply trailing stop when profit >= threshold
        
        Args:
            user_id: User ID
            position_id: Position ID
            pos: Position dict (will be modified)
            current_price: Current market price
            unrealized_pl: Unrealized P/L
            
        Returns:
            tuple[bool, Optional[float], Dict]: (sl_adjusted, new_stop_loss, updated_pos)
        """
        pos = self._normalize_position_dict(pos)
        signal_type = pos['signal_type']
        stop_loss = pos['stop_loss']
        
        if unrealized_pl <= 0 or unrealized_pl < self.config.TRAILING_STOP_PROFIT_THRESHOLD:
            return False, None, pos
        
        max_profit = pos.get('max_profit_reached', 0.0)
        if max_profit is None:
            max_profit = 0.0
        
        if unrealized_pl > max_profit:
            pos['max_profit_reached'] = unrealized_pl
        
        trailing_distance = self.config.TRAILING_STOP_DISTANCE_PIPS / self.config.XAUUSD_PIP_VALUE
        
        new_trailing_sl = None
        sl_adjusted = False
        
        if signal_type == 'BUY':
            new_trailing_sl = current_price - trailing_distance
            if new_trailing_sl > stop_loss:
                pos['stop_loss'] = new_trailing_sl
                pos['sl_adjustment_count'] = pos.get('sl_adjustment_count', 0) + 1
                sl_adjusted = True
                logger.info(f"💎 Trailing stop activated! Profit ${unrealized_pl:.2f}. SL moved to ${new_trailing_sl:.2f} (lock-in profit)")
                
                should_notify = self._check_trailing_stop_notify_cooldown(user_id, position_id)
                
                if should_notify:
                    msg = (
                        f"💎 *Trailing Stop Active*\n\n"
                        f"Position ID: {position_id}\n"
                        f"Type: {signal_type}\n"
                        f"Current Profit: ${unrealized_pl:.2f}\n"
                        f"Max Profit: ${pos['max_profit_reached']:.2f}\n"
                        f"SL Updated: ${stop_loss:.2f} → ${new_trailing_sl:.2f}\n"
                        f"Status: Profit locked-in!"
                    )
                    if await self._safe_send_notification(user_id, msg, parse_mode='Markdown'):
                        self._update_trailing_stop_notify_time(user_id, position_id)
        else:
            new_trailing_sl = current_price + trailing_distance
            if new_trailing_sl < stop_loss:
                pos['stop_loss'] = new_trailing_sl
                pos['sl_adjustment_count'] = pos.get('sl_adjustment_count', 0) + 1
                sl_adjusted = True
                logger.info(f"💎 Trailing stop activated! Profit ${unrealized_pl:.2f}. SL moved to ${new_trailing_sl:.2f} (lock-in profit)")
                
                should_notify = self._check_trailing_stop_notify_cooldown(user_id, position_id)
                
                if should_notify:
                    msg = (
                        f"💎 *Trailing Stop Active*\n\n"
                        f"Position ID: {position_id}\n"
                        f"Type: {signal_type}\n"
                        f"Current Profit: ${unrealized_pl:.2f}\n"
                        f"Max Profit: ${pos['max_profit_reached']:.2f}\n"
                        f"SL Updated: ${stop_loss:.2f} → ${new_trailing_sl:.2f}\n"
                        f"Status: Profit locked-in!"
                    )
                    if await self._safe_send_notification(user_id, msg, parse_mode='Markdown'):
                        self._update_trailing_stop_notify_time(user_id, position_id)
        
        return sl_adjusted, new_trailing_sl if sl_adjusted else None, pos
    
    async def apply_trailing_stop(self, user_id: int, position_id: int, current_price: float, unrealized_pl: float) -> tuple[bool, Optional[float]]:
        """Apply trailing stop when profit >= threshold
        
        Thread-safe wrapper for _apply_trailing_stop_internal
        
        Returns:
            tuple[bool, Optional[float]]: (sl_adjusted, new_stop_loss)
        """
        async with self._position_lock:
            if user_id not in self.active_positions or position_id not in self.active_positions[user_id]:
                return False, None
            
            pos = self.active_positions[user_id][position_id].copy()
        
        sl_adjusted, new_sl, updated_pos = await self._apply_trailing_stop_internal(
            user_id, position_id, pos, current_price, unrealized_pl
        )
        
        if sl_adjusted:
            async with self._position_lock:
                if user_id in self.active_positions and position_id in self.active_positions[user_id]:
                    self.active_positions[user_id][position_id].update(updated_pos)
        
        return sl_adjusted, new_sl
    
    async def auto_update_trailing_stop(self, user_id: int, position_id: int, 
                                         current_price: float, unrealized_pl: float) -> tuple[bool, Optional[float]]:
        """Auto-update trailing stop dengan database persistence dan audit logging.
        
        Method ini dipanggil dari update_position() setiap tick.
        - Cek profit >= TRAILING_STOP_MOVE_PIPS (default 5 pips)
        - Move SL dengan distance TRAILING_STOP_TRAIL_DISTANCE_PIPS (default 3 pips) dari current price
        - Update Position.stop_loss di database
        - Log semua trailing stop updates untuk audit trail
        - Respect DRY_RUN_TRAILING_STOP flag
        
        Args:
            user_id: User ID
            position_id: Position ID
            current_price: Current market price
            unrealized_pl: Unrealized P/L in dollars
            
        Returns:
            tuple[bool, Optional[float]]: (sl_adjusted, new_stop_loss)
        """
        async with self._position_lock:
            if user_id not in self.active_positions or position_id not in self.active_positions[user_id]:
                return False, None
            
            pos = self.active_positions[user_id][position_id].copy()
        
        pos = self._normalize_position_dict(pos)
        signal_type = pos['signal_type']
        entry_price = pos['entry_price']
        stop_loss = pos['stop_loss']
        
        profit_pips = unrealized_pl * self.config.XAUUSD_PIP_VALUE
        
        move_threshold_pips = getattr(self.config, 'TRAILING_STOP_MOVE_PIPS', 5.0)
        trail_distance_pips = getattr(self.config, 'TRAILING_STOP_TRAIL_DISTANCE_PIPS', 3.0)
        
        if profit_pips < move_threshold_pips:
            return False, None
        
        trail_distance = trail_distance_pips / self.config.XAUUSD_PIP_VALUE
        
        new_trailing_sl = None
        sl_adjusted = False
        old_stop_loss = stop_loss
        
        if signal_type == 'BUY':
            new_trailing_sl = current_price - trail_distance
            if new_trailing_sl > stop_loss:
                sl_adjusted = True
        else:
            new_trailing_sl = current_price + trail_distance
            if new_trailing_sl < stop_loss:
                sl_adjusted = True
        
        if not sl_adjusted:
            return False, None
        
        dry_run = getattr(self.config, 'DRY_RUN_TRAILING_STOP', False)
        
        logger.info(
            f"📊 [TRAILING_STOP_AUDIT] Position:{position_id} User:{user_id} | "
            f"Mode:{'DRY_RUN' if dry_run else 'LIVE'} | "
            f"Type:{signal_type} | Entry:${entry_price:.2f} | "
            f"Price:${current_price:.2f} | Profit:{profit_pips:.1f}pips (${unrealized_pl:.2f}) | "
            f"SL:${old_stop_loss:.2f}→${new_trailing_sl:.2f}"
        )
        
        if dry_run:
            logger.info(f"🧪 [DRY_RUN] Trailing stop TIDAK di-update karena DRY_RUN_TRAILING_STOP=true")
            return False, None
        
        async with self._position_lock:
            if user_id in self.active_positions and position_id in self.active_positions[user_id]:
                self.active_positions[user_id][position_id]['stop_loss'] = new_trailing_sl
                self.active_positions[user_id][position_id]['sl_adjustment_count'] = \
                    self.active_positions[user_id][position_id].get('sl_adjustment_count', 0) + 1
                
                max_profit = self.active_positions[user_id][position_id].get('max_profit_reached', 0.0)
                if max_profit is None:
                    max_profit = 0.0
                if unrealized_pl > max_profit:
                    self.active_positions[user_id][position_id]['max_profit_reached'] = unrealized_pl
        
        db_session = None
        db_commit_success = False
        max_retry_attempts = 3
        
        try:
            db_session = self.db.get_session()
            if db_session is None:
                logger.error(f"❌ [TRAILING_STOP_DB] Failed to get database session for position {position_id}")
            else:
                position = db_session.query(Position).filter(
                    Position.id == position_id, 
                    Position.user_id == user_id
                ).first()
                
                if position:
                    old_db_sl = position.stop_loss
                    position.stop_loss = new_trailing_sl
                    position.sl_adjustment_count = (position.sl_adjustment_count or 0) + 1
                    position.last_price_update = datetime.now(pytz.UTC)
                    
                    current_max = position.max_profit_reached if position.max_profit_reached is not None else 0.0
                    if unrealized_pl > current_max:
                        position.max_profit_reached = unrealized_pl
                    
                    last_commit_error = None
                    for attempt in range(1, max_retry_attempts + 1):
                        try:
                            logger.info(
                                f"🔄 [TRAILING_STOP_DB_RETRY] Position:{position_id} | "
                                f"Attempt {attempt}/{max_retry_attempts} | Committing SL update..."
                            )
                            db_session.commit()
                            db_commit_success = True
                            
                            logger.info(
                                f"✅ [TRAILING_STOP_DB_COMMIT] Position:{position_id} | "
                                f"SL: ${old_db_sl:.2f} → ${new_trailing_sl:.2f} | "
                                f"Adjustment count: {position.sl_adjustment_count} | "
                                f"Commit: SUCCESS (attempt {attempt}/{max_retry_attempts})"
                            )
                            break
                            
                        except (IntegrityError, OperationalError, SQLAlchemyError) as commit_err:
                            last_commit_error = commit_err
                            logger.warning(
                                f"⚠️ [TRAILING_STOP_DB_RETRY] Position:{position_id} | "
                                f"Attempt {attempt}/{max_retry_attempts} FAILED: {type(commit_err).__name__}: {commit_err}"
                            )
                            
                            try:
                                db_session.rollback()
                                logger.info(
                                    f"🔄 [TRAILING_STOP_DB_RETRY] Position:{position_id} | "
                                    f"Rollback executed after attempt {attempt}"
                                )
                            except Exception as rb_err:
                                logger.error(
                                    f"❌ [TRAILING_STOP_DB_RETRY] Position:{position_id} | "
                                    f"Rollback failed after attempt {attempt}: {rb_err}"
                                )
                            
                            if attempt < max_retry_attempts:
                                retry_delay = 0.5 * attempt
                                logger.info(
                                    f"⏳ [TRAILING_STOP_DB_RETRY] Position:{position_id} | "
                                    f"Waiting {retry_delay}s before retry {attempt + 1}..."
                                )
                                await asyncio.sleep(retry_delay)
                                
                                position = db_session.query(Position).filter(
                                    Position.id == position_id, 
                                    Position.user_id == user_id
                                ).first()
                                if position:
                                    position.stop_loss = new_trailing_sl
                                    position.sl_adjustment_count = (position.sl_adjustment_count or 0) + 1
                                    position.last_price_update = datetime.now(pytz.UTC)
                                    if unrealized_pl > (position.max_profit_reached or 0.0):
                                        position.max_profit_reached = unrealized_pl
                    
                    if not db_commit_success and last_commit_error:
                        logger.error(
                            f"❌ [TRAILING_STOP_DB_RETRY] Position:{position_id} | "
                            f"All {max_retry_attempts} attempts FAILED | "
                            f"Last error: {type(last_commit_error).__name__}: {last_commit_error}"
                        )
                else:
                    logger.warning(f"⚠️ Position {position_id} tidak ditemukan di database untuk trailing stop update")
                
        except (IntegrityError, OperationalError, SQLAlchemyError) as e:
            logger.error(f"❌ [TRAILING_STOP_DB_ERROR] Database error updating position {position_id}: {e}")
            if db_session:
                try:
                    db_session.rollback()
                    logger.info(f"🔄 [TRAILING_STOP_DB] Rollback executed for position {position_id}")
                except Exception as rb_err:
                    logger.error(f"❌ [TRAILING_STOP_DB] Rollback failed: {rb_err}")
        except Exception as e:
            logger.error(f"❌ [TRAILING_STOP_DB_ERROR] Unexpected error for position {position_id}: {e}")
            if db_session:
                try:
                    db_session.rollback()
                except Exception:
                    pass
        finally:
            if db_session:
                try:
                    db_session.close()
                    logger.debug(f"🔒 [TRAILING_STOP_DB] Session closed for position {position_id}")
                except Exception as close_err:
                    logger.error(f"❌ [TRAILING_STOP_DB] Session close error: {close_err}")
        
        if not db_commit_success:
            logger.warning(f"⚠️ [TRAILING_STOP_DB] Database commit was NOT successful for position {position_id}")
        
        should_notify = self._check_trailing_stop_notify_cooldown(user_id, position_id)
        
        if should_notify:
            msg = (
                f"💎 *Trailing Stop Auto-Update*\n\n"
                f"Position ID: {position_id}\n"
                f"Type: {signal_type}\n"
                f"Profit Saat Ini: +{profit_pips:.1f} pips (${unrealized_pl:.2f})\n"
                f"SL Diperbarui: ${old_stop_loss:.2f} → ${new_trailing_sl:.2f}\n"
                f"Jarak Trail: {trail_distance_pips} pips\n"
                f"Status: Profit terkunci! ✅"
            )
            if await self._safe_send_notification(user_id, msg, parse_mode='Markdown'):
                self._update_trailing_stop_notify_time(user_id, position_id)
        
        return True, new_trailing_sl
    
    async def check_and_execute_partial_exit(self, user_id: int, position_id: int,
                                              current_price: float, unrealized_pl: float) -> Optional[str]:
        """Check dan execute partial exit saat TP1 (1.0x risk) dan TP2 (1.5x risk).
        
        Strategy:
        - TP1 (1.0x risk): Close 40% posisi, move SL ke breakeven
        - TP2 (1.5x risk): Close 35% posisi
        - Remaining 25%: Trailing stop aktif sampai TP full atau SL hit
        
        Args:
            user_id: User ID
            position_id: Position ID
            current_price: Current market price
            unrealized_pl: Unrealized P/L in dollars
            
        Returns:
            Optional[str]: Partial exit reason if triggered, None otherwise
        """
        if not getattr(self.config, 'PARTIAL_EXIT_ENABLED', False):
            return None
        
        async with self._position_lock:
            if user_id not in self.active_positions or position_id not in self.active_positions[user_id]:
                return None
            
            pos = self.active_positions[user_id][position_id].copy()
        
        pos = self._normalize_position_dict(pos)
        signal_type = pos['signal_type']
        entry_price = pos['entry_price']
        stop_loss = pos['stop_loss']
        original_sl = pos.get('original_sl', stop_loss)
        
        tp1_triggered = pos.get('tp1_triggered', False)
        tp2_triggered = pos.get('tp2_triggered', False)
        
        if tp1_triggered and tp2_triggered:
            return None
        
        risk_distance = abs(entry_price - original_sl)
        profit_distance = abs(current_price - entry_price)
        
        if risk_distance <= 0:
            return None
        
        risk_reward_ratio = profit_distance / risk_distance
        
        tp1_rr = getattr(self.config, 'PARTIAL_EXIT_TP1_RR', 1.0)
        tp2_rr = getattr(self.config, 'PARTIAL_EXIT_TP2_RR', 1.5)
        tp1_percent = getattr(self.config, 'PARTIAL_EXIT_TP1_PERCENT', 40.0)
        tp2_percent = getattr(self.config, 'PARTIAL_EXIT_TP2_PERCENT', 35.0)
        
        if signal_type == 'BUY':
            is_in_profit = current_price > entry_price
        else:
            is_in_profit = current_price < entry_price
        
        if not is_in_profit:
            return None
        
        partial_exit_reason = None
        
        if not tp1_triggered and risk_reward_ratio >= tp1_rr:
            partial_exit_reason = 'PARTIAL_TP1'
            
            logger.info(
                f"🎯 [PARTIAL_EXIT_TP1] Position:{position_id} User:{user_id} | "
                f"RR={risk_reward_ratio:.2f}x >= TP1={tp1_rr}x | "
                f"Exit {tp1_percent}% posisi | Move SL ke breakeven"
            )
            
            async with self._position_lock:
                if user_id in self.active_positions and position_id in self.active_positions[user_id]:
                    self.active_positions[user_id][position_id]['tp1_triggered'] = True
                    
                    old_sl = self.active_positions[user_id][position_id]['stop_loss']
                    if signal_type == 'BUY' and entry_price > old_sl:
                        self.active_positions[user_id][position_id]['stop_loss'] = entry_price
                        logger.info(f"🛡️ SL moved to breakeven: ${old_sl:.2f} → ${entry_price:.2f}")
                    elif signal_type == 'SELL' and entry_price < old_sl:
                        self.active_positions[user_id][position_id]['stop_loss'] = entry_price
                        logger.info(f"🛡️ SL moved to breakeven: ${old_sl:.2f} → ${entry_price:.2f}")
            
            partial_pl = unrealized_pl * (tp1_percent / 100)
            
            db_session = None
            try:
                db_session = self.db.get_session()
                if db_session is None:
                    logger.error(f"❌ [PARTIAL_TP1] Failed to get database session")
                else:
                    position = db_session.query(Position).filter(
                        Position.id == position_id,
                        Position.user_id == user_id
                    ).first()
                    
                    if position:
                        position.stop_loss = entry_price
                        position.sl_adjustment_count = (position.sl_adjustment_count or 0) + 1
                        position.last_price_update = datetime.now(pytz.UTC)
                        
                        partial_trade = Trade(
                            user_id=user_id,
                            ticker='XAUUSD',
                            signal_type=signal_type,
                            signal_source='partial_tp1',
                            entry_price=entry_price,
                            stop_loss=original_sl,
                            take_profit=pos.get('take_profit', 0),
                            exit_price=current_price,
                            actual_pl=partial_pl,
                            status='CLOSED',
                            signal_time=pos.get('opened_at', datetime.now(pytz.UTC)),
                            close_time=datetime.now(pytz.UTC),
                            result='WIN' if partial_pl > 0 else 'LOSS',
                            timeframe='M1'
                        )
                        db_session.add(partial_trade)
                        db_session.commit()
                        
                        logger.info(
                            f"✅ [PARTIAL_TP1_DB_COMMIT] Position:{position_id} | "
                            f"Trade recorded: ${partial_pl:.2f} ({tp1_percent}%) | "
                            f"SL moved to breakeven=${entry_price:.2f}"
                        )
            except Exception as e:
                logger.error(f"❌ [PARTIAL_TP1] Database error: {e}")
                if db_session:
                    try:
                        db_session.rollback()
                    except Exception:
                        pass
            finally:
                if db_session:
                    try:
                        db_session.close()
                    except Exception:
                        pass
            
            msg = (
                f"🎯 *Partial Take Profit - TP1*\n\n"
                f"Position ID: {position_id}\n"
                f"Type: {signal_type}\n"
                f"Risk/Reward: {risk_reward_ratio:.2f}x (target: {tp1_rr}x)\n"
                f"Profit Partial: ${partial_pl:.2f} ({tp1_percent}%)\n"
                f"Total Unrealized: ${unrealized_pl:.2f}\n\n"
                f"✅ Exit {tp1_percent}% posisi\n"
                f"🛡️ SL dipindah ke breakeven: ${entry_price:.2f}\n"
                f"📊 Sisa {100 - tp1_percent}% dengan trailing stop"
            )
            notification_sent = await self._safe_send_notification(user_id, msg, parse_mode='Markdown')
            logger.info(f"📤 [PARTIAL_TP1] Notification sent: {notification_sent}")
            
        elif tp1_triggered and not tp2_triggered and risk_reward_ratio >= tp2_rr:
            partial_exit_reason = 'PARTIAL_TP2'
            
            logger.info(
                f"🎯 [PARTIAL_EXIT_TP2] Position:{position_id} User:{user_id} | "
                f"RR={risk_reward_ratio:.2f}x >= TP2={tp2_rr}x | "
                f"Exit {tp2_percent}% posisi"
            )
            
            async with self._position_lock:
                if user_id in self.active_positions and position_id in self.active_positions[user_id]:
                    self.active_positions[user_id][position_id]['tp2_triggered'] = True
            
            remaining_percent = 100 - tp1_percent - tp2_percent
            partial_pl = unrealized_pl * (tp2_percent / 100)
            
            db_session = None
            try:
                db_session = self.db.get_session()
                if db_session is None:
                    logger.error(f"❌ [PARTIAL_TP2] Failed to get database session")
                else:
                    position = db_session.query(Position).filter(
                        Position.id == position_id,
                        Position.user_id == user_id
                    ).first()
                    
                    if position:
                        position.last_price_update = datetime.now(pytz.UTC)
                        
                        partial_trade = Trade(
                            user_id=user_id,
                            ticker='XAUUSD',
                            signal_type=signal_type,
                            signal_source='partial_tp2',
                            entry_price=entry_price,
                            stop_loss=original_sl,
                            take_profit=pos.get('take_profit', 0),
                            exit_price=current_price,
                            actual_pl=partial_pl,
                            status='CLOSED',
                            signal_time=pos.get('opened_at', datetime.now(pytz.UTC)),
                            close_time=datetime.now(pytz.UTC),
                            result='WIN' if partial_pl > 0 else 'LOSS',
                            timeframe='M1'
                        )
                        db_session.add(partial_trade)
                        db_session.commit()
                        
                        logger.info(
                            f"✅ [PARTIAL_TP2_DB_COMMIT] Position:{position_id} | "
                            f"Trade recorded: ${partial_pl:.2f} ({tp2_percent}%) | "
                            f"Remaining: {remaining_percent}%"
                        )
            except Exception as e:
                logger.error(f"❌ [PARTIAL_TP2] Database error: {e}")
                if db_session:
                    try:
                        db_session.rollback()
                    except Exception:
                        pass
            finally:
                if db_session:
                    try:
                        db_session.close()
                    except Exception:
                        pass
            
            msg = (
                f"🎯 *Partial Take Profit - TP2*\n\n"
                f"Position ID: {position_id}\n"
                f"Type: {signal_type}\n"
                f"Risk/Reward: {risk_reward_ratio:.2f}x (target: {tp2_rr}x)\n"
                f"Profit Partial: ${partial_pl:.2f} ({tp2_percent}%)\n"
                f"Total Unrealized: ${unrealized_pl:.2f}\n\n"
                f"✅ Exit tambahan {tp2_percent}% posisi\n"
                f"📊 Sisa {remaining_percent}% dengan trailing stop aktif"
            )
            notification_sent = await self._safe_send_notification(user_id, msg, parse_mode='Markdown')
            logger.info(f"📤 [PARTIAL_TP2] Notification sent: {notification_sent}")
        
        return partial_exit_reason
    
    async def _check_grade_based_auto_close(self, user_id: int, position_id: int, 
                                            pos: Dict, unrealized_pl: float) -> Optional[str]:
        """Check if position should be auto-closed based on signal grade and duration.
        
        Auto-close rules:
        - Grade C: auto-close if >20 min without profit
        - Grade B: auto-close if >45 min with minimal profit (<25% of TP)
        - Grade A: hold until TP/SL or >60 min timeout
        
        Args:
            user_id: User ID
            position_id: Position ID
            pos: Position dict
            unrealized_pl: Current unrealized P/L
            
        Returns:
            Auto-close reason string if should close, None otherwise
        """
        try:
            signal_grade = pos.get('signal_grade', 'B')
            opened_at = pos.get('opened_at')
            take_profit = pos.get('take_profit', 0)
            entry_price = pos.get('entry_price', 0)
            
            if not opened_at:
                return None
            
            if isinstance(opened_at, str):
                try:
                    opened_at = datetime.fromisoformat(opened_at.replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    return None
            
            now = datetime.now(pytz.UTC)
            if opened_at.tzinfo is None:
                opened_at = pytz.UTC.localize(opened_at)
            
            duration_minutes = (now - opened_at).total_seconds() / 60
            
            tp_distance = abs(take_profit - entry_price) if take_profit and entry_price else 0
            tp_progress = unrealized_pl / (tp_distance * 10) if tp_distance > 0 else 0
            
            if signal_grade == 'C':
                if duration_minutes > 20 and unrealized_pl <= 0:
                    logger.info(
                        f"🔴 [GRADE_AUTO_CLOSE] Grade C position {position_id} closing: "
                        f"Duration={duration_minutes:.1f}min (>20min), P/L=${unrealized_pl:.2f} (<=0)"
                    )
                    msg = (
                        f"🔴 *Auto-Close: Grade C Signal*\n\n"
                        f"Position ID: {position_id}\n"
                        f"Grade: C (Low confidence)\n"
                        f"Duration: {duration_minutes:.1f} menit\n"
                        f"P/L: ${unrealized_pl:.2f}\n"
                        f"Reason: No profit after 20 minutes"
                    )
                    await self._safe_send_notification(user_id, msg, parse_mode='Markdown')
                    return 'GRADE_C_TIMEOUT'
            
            elif signal_grade == 'B':
                if duration_minutes > 45 and tp_progress < 0.25:
                    logger.info(
                        f"🟡 [GRADE_AUTO_CLOSE] Grade B position {position_id} closing: "
                        f"Duration={duration_minutes:.1f}min (>45min), TP Progress={tp_progress*100:.1f}% (<25%)"
                    )
                    msg = (
                        f"🟡 *Auto-Close: Grade B Signal*\n\n"
                        f"Position ID: {position_id}\n"
                        f"Grade: B (Medium confidence)\n"
                        f"Duration: {duration_minutes:.1f} menit\n"
                        f"P/L: ${unrealized_pl:.2f}\n"
                        f"TP Progress: {tp_progress*100:.1f}%\n"
                        f"Reason: Minimal profit after 45 minutes"
                    )
                    await self._safe_send_notification(user_id, msg, parse_mode='Markdown')
                    return 'GRADE_B_TIMEOUT'
            
            elif signal_grade == 'A':
                if duration_minutes > 60:
                    logger.info(
                        f"🟢 [GRADE_AUTO_CLOSE] Grade A position {position_id} closing: "
                        f"Duration={duration_minutes:.1f}min (>60min max hold), P/L=${unrealized_pl:.2f}"
                    )
                    msg = (
                        f"🟢 *Auto-Close: Grade A Max Duration*\n\n"
                        f"Position ID: {position_id}\n"
                        f"Grade: A (High confidence)\n"
                        f"Duration: {duration_minutes:.1f} menit\n"
                        f"P/L: ${unrealized_pl:.2f}\n"
                        f"Reason: Maximum hold time reached (60 min)"
                    )
                    await self._safe_send_notification(user_id, msg, parse_mode='Markdown')
                    return 'GRADE_A_MAX_DURATION'
            
            return None
            
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error(f"Error in grade-based auto-close check: {e}")
            return None
    
    async def update_position(self, user_id: int, position_id: int, current_price: float) -> Optional[str]:
        """Update position with current price and apply dynamic SL/TP logic with validation
        
        Thread-safe position update with lock protection
        """
        try:
            if current_price is None or current_price <= 0:
                logger.warning(f"Invalid current_price for position {position_id}: {current_price}")
                return None
            
            async with self._position_lock:
                if user_id not in self.active_positions or position_id not in self.active_positions[user_id]:
                    logger.debug(f"Position {position_id} for User:{user_id} not found in active positions")
                    return None
                
                pos = self.active_positions[user_id][position_id].copy()
            
            try:
                signal_type = pos['signal_type']
                entry_price = pos['entry_price']
                stop_loss = pos['stop_loss']
                take_profit = pos['take_profit']
                
                if None in [signal_type, entry_price, stop_loss, take_profit]:
                    logger.error(f"Position {position_id} has None values: signal={signal_type}, entry={entry_price}, sl={stop_loss}, tp={take_profit}")
                    return None
                    
            except KeyError as e:
                logger.error(f"Missing required key in position {position_id}: {e}")
                return None
            
            unrealized_pl = self.risk_manager.calculate_pl(entry_price, current_price, signal_type)
        
            sl_adjusted = False
            dynamic_sl_applied = False
            
            try:
                dynamic_sl_applied, new_sl = await self.apply_dynamic_sl(user_id, position_id, current_price, unrealized_pl)
                if dynamic_sl_applied:
                    sl_adjusted = True
                    stop_loss = new_sl
            except (asyncio.TimeoutError, asyncio.CancelledError, KeyError, ValueError, AttributeError) as e:
                logger.error(f"Error applying dynamic SL for position {position_id}: {e}")
            
            if not dynamic_sl_applied:
                try:
                    auto_trailing_applied, new_sl = await self.auto_update_trailing_stop(
                        user_id, position_id, current_price, unrealized_pl
                    )
                    if auto_trailing_applied:
                        sl_adjusted = True
                        stop_loss = new_sl
                except (asyncio.TimeoutError, asyncio.CancelledError, KeyError, ValueError, AttributeError) as e:
                    logger.error(f"Error applying auto trailing stop for position {position_id}: {e}")
            
            try:
                partial_exit_reason = await self.check_and_execute_partial_exit(
                    user_id, position_id, current_price, unrealized_pl
                )
                if partial_exit_reason:
                    logger.info(f"📊 Partial exit triggered: {partial_exit_reason} for position {position_id}")
                    async with self._position_lock:
                        if user_id in self.active_positions and position_id in self.active_positions[user_id]:
                            stop_loss = self.active_positions[user_id][position_id]['stop_loss']
                            sl_adjusted = True
            except (asyncio.TimeoutError, asyncio.CancelledError, KeyError, ValueError, AttributeError) as e:
                logger.error(f"Error checking partial exit for position {position_id}: {e}")
            
            async with self._position_lock:
                if user_id in self.active_positions and position_id in self.active_positions[user_id]:
                    stop_loss = self.active_positions[user_id][position_id]['stop_loss']
                    sl_adjustment_count = self.active_positions[user_id][position_id].get('sl_adjustment_count', 0)
                else:
                    return None
            
            session = None
            try:
                session = self.db.get_session()
                try:
                    session.rollback()
                except Exception:
                    pass
                
                position = session.query(Position).filter(Position.id == position_id, Position.user_id == user_id).first()
                if not position:
                    logger.warning(f"Position {position_id} not found in database for User:{user_id}")
                    return None
                
                position.current_price = current_price
                position.unrealized_pl = unrealized_pl
                position.last_price_update = datetime.now(pytz.UTC)
                
                current_max_profit = position.max_profit_reached if position.max_profit_reached is not None else 0.0
                if unrealized_pl > 0 and unrealized_pl > current_max_profit:
                    position.max_profit_reached = unrealized_pl
                
                if sl_adjusted:
                    position.stop_loss = stop_loss
                    position.sl_adjustment_count = sl_adjustment_count
                
                session.commit()
            except (IntegrityError, OperationalError, SQLAlchemyError, ValueError, TypeError) as e:
                logger.error(f"Database error updating position {position_id}: {type(e).__name__}: {e}", exc_info=True)
                if session:
                    try:
                        session.rollback()
                    except (OperationalError, SQLAlchemyError) as rollback_error:
                        logger.error(f"Error during rollback: {rollback_error}")
            finally:
                if session:
                    try:
                        session.close()
                    except (OperationalError, SQLAlchemyError) as close_error:
                        logger.error(f"Error closing session: {close_error}")
            
            auto_close_reason = await self._check_grade_based_auto_close(
                user_id, position_id, pos, unrealized_pl
            )
            if auto_close_reason:
                await self.close_position(user_id, position_id, current_price, auto_close_reason)
                return auto_close_reason
            
            hit_tp = False
            hit_sl = False
            
            try:
                if signal_type == 'BUY':
                    hit_tp = current_price >= take_profit
                    hit_sl = current_price <= stop_loss
                else:
                    hit_tp = current_price <= take_profit
                    hit_sl = current_price >= stop_loss
                
                if hit_tp or hit_sl:
                    logger.info(f"🎯 Position {position_id} TP/SL trigger: Price=${current_price:.2f}, TP=${take_profit:.2f}, SL=${stop_loss:.2f}, hit_tp={hit_tp}, hit_sl={hit_sl}")
            except (ValueError, TypeError, AttributeError) as e:
                logger.error(f"Error checking TP/SL conditions: {e}")
                return None
            
            if hit_tp:
                logger.info(f"✅ TP HIT - Position {position_id}: Price=${current_price:.2f} >= TP=${take_profit:.2f}")
                await self.close_position(user_id, position_id, current_price, 'TP_HIT')
                return 'TP_HIT'
            elif hit_sl:
                reason = 'DYNAMIC_SL_HIT' if sl_adjusted else 'SL_HIT'
                logger.info(f"❌ SL HIT - Position {position_id}: Price=${current_price:.2f} <= SL=${stop_loss:.2f}")
                await self.close_position(user_id, position_id, current_price, reason)
                return reason
            
            return None
            
        except (asyncio.CancelledError, asyncio.TimeoutError, KeyError, ValueError, TypeError, RuntimeError) as e:
            logger.error(f"Critical error updating position {position_id} for User:{user_id}: {type(e).__name__}: {e}", exc_info=True)
            return None
    
    async def close_position(self, user_id: int, position_id: int, exit_price: float, reason: str):
        """Close a position with thread-safe access and proper cleanup
        
        Thread-safe position closure with lock protection
        """
        async with self._position_lock:
            if user_id not in self.active_positions or position_id not in self.active_positions[user_id]:
                return
            
            pos = self.active_positions[user_id][position_id].copy()
        
        trade_id = pos['trade_id']
        signal_type = pos['signal_type']
        entry_price = pos['entry_price']
        
        actual_pl = self.risk_manager.calculate_pl(entry_price, exit_price, signal_type)
        
        trade_result = 'WIN' if actual_pl >= 0 else 'LOSS'
        
        session = None
        try:
            session = self.db.get_session()
            try:
                session.rollback()
            except Exception:
                pass
            
            position = session.query(Position).filter(Position.id == position_id, Position.user_id == user_id).first()
            if position:
                position.status = 'CLOSED'
                position.current_price = exit_price
                position.unrealized_pl = actual_pl
                position.closed_at = datetime.now(pytz.UTC)
            else:
                logger.warning(f"⚠️ Position {position_id} not found in database for User:{user_id} - may cause data inconsistency")
                
            trade = session.query(Trade).filter(Trade.id == trade_id, Trade.user_id == user_id).first()
            if trade:
                trade.status = 'CLOSED'
                trade.exit_price = exit_price
                trade.actual_pl = actual_pl
                trade.close_time = datetime.now(pytz.UTC)
                trade.result = trade_result
            else:
                logger.warning(f"⚠️ Trade {trade_id} not found in database for User:{user_id} - trade history may be incomplete")
                
            session.commit()
            
            async with self._position_lock:
                if user_id in self.active_positions and position_id in self.active_positions[user_id]:
                    del self.active_positions[user_id][position_id]
                    if not self.active_positions[user_id]:
                        del self.active_positions[user_id]
            
            with self._trailing_stop_lock:
                if user_id in self._trailing_stop_last_notify:
                    self._trailing_stop_last_notify[user_id].pop(position_id, None)
                    if not self._trailing_stop_last_notify[user_id]:
                        del self._trailing_stop_last_notify[user_id]
            
            result_emoji = "✅" if trade_result == 'WIN' else "❌"
            logger.info(f"{'='*50}")
            logger.info(f"{result_emoji} TRADE CLOSED: {trade_result}")
            logger.info(f"   User: {user_id} | Position: {position_id}")
            logger.info(f"   Type: {signal_type} | Reason: {reason}")
            logger.info(f"   Entry: ${entry_price:.2f} | Exit: ${exit_price:.2f}")
            logger.info(f"   P/L: ${actual_pl:+.2f}")
            logger.info(f"{'='*50}")
            
            # Update signal quality tracker with trade result
            if hasattr(self, 'signal_quality_tracker') and self.signal_quality_tracker:
                try:
                    # Calculate pips and duration
                    opened_at = pos.get('opened_at')
                    duration_minutes = 0
                    if opened_at:
                        try:
                            if isinstance(opened_at, datetime):
                                if opened_at.tzinfo is None:
                                    opened_at = pytz.UTC.localize(opened_at)
                                duration_minutes = int((datetime.now(pytz.UTC) - opened_at).total_seconds() / 60)
                            elif isinstance(opened_at, str):
                                parsed_dt = datetime.fromisoformat(opened_at.replace('Z', '+00:00'))
                                if parsed_dt.tzinfo is None:
                                    parsed_dt = pytz.UTC.localize(parsed_dt)
                                duration_minutes = int((datetime.now(pytz.UTC) - parsed_dt).total_seconds() / 60)
                        except Exception:
                            duration_minutes = 0
                    
                    # Calculate actual pips
                    if signal_type == 'BUY':
                        actual_pips = (exit_price - entry_price) * 10  # 10 pips per $1
                    else:
                        actual_pips = (entry_price - exit_price) * 10
                    
                    result_data = {
                        'exit_price': exit_price,
                        'actual_pips': actual_pips,
                        'result': trade_result,
                        'duration_minutes': duration_minutes
                    }
                    
                    # Get signal_quality_id from position if available
                    signal_quality_id = pos.get('signal_quality_id')
                    if not signal_quality_id:
                        # Try to get from database (Trade or Position record)
                        try:
                            db_session = self.db.get_session()
                            try:
                                trade_record = db_session.query(Trade).filter(Trade.id == trade_id).first()
                                if trade_record and hasattr(trade_record, 'signal_quality_id'):
                                    signal_quality_id = trade_record.signal_quality_id
                                if not signal_quality_id:
                                    position_record = db_session.query(Position).filter(Position.id == position_id).first()
                                    if position_record and hasattr(position_record, 'signal_quality_id'):
                                        signal_quality_id = position_record.signal_quality_id
                            finally:
                                db_session.close()
                        except Exception as e:
                            logger.warning(f"Could not retrieve signal_quality_id from database: {e}")
                    
                    if signal_quality_id:
                        self.signal_quality_tracker.update_result(signal_quality_id, result_data)
                        logger.info(f"📊 Signal quality updated: ID={signal_quality_id}, Result={trade_result}")
                    else:
                        logger.warning(f"⚠️ No signal_quality_id found for position {position_id}, cannot update signal quality")
                except Exception as e:
                    logger.error(f"❌ Error updating signal quality for position {position_id}: {e}")
            
            # End signal session for THIS SPECIFIC position to allow new signals
            # CRITICAL: Pass position_id so only this session is ended, not all user sessions
            if self.signal_session_manager:
                try:
                    await self.signal_session_manager.end_session(user_id, reason, position_id=position_id)
                    logger.info(f"✅ Signal session ended for user {user_id} position {position_id} - reason: {reason}")
                except Exception as e:
                    logger.error(f"❌ Error ending signal session for user {user_id} position {position_id}: {e}")
            
            if self.telegram_app and self.chart_generator and self.market_data:
                try:
                    df_m1 = await self.market_data.get_historical_data('M1', 100)
                    
                    if df_m1 is not None and len(df_m1) >= 30:
                        exit_signal = {
                            'signal': signal_type,
                            'entry_price': entry_price,
                            'stop_loss': pos['stop_loss'],
                            'take_profit': pos['take_profit'],
                            'timeframe': 'M1'
                        }
                        
                        chart_path = await self.chart_generator.generate_chart_async(df_m1, exit_signal, 'M1')
                        
                        # Note: Session sudah diakhiri di atas, tidak perlu update chart_path ke session
                        # Chart hanya digunakan untuk exit notification di bawah ini
                        
                        opened_at = pos.get('opened_at')
                        duration_seconds = 0
                        if opened_at:
                            if isinstance(opened_at, datetime):
                                if opened_at.tzinfo is None:
                                    opened_at = pytz.UTC.localize(opened_at)
                                duration_seconds = (datetime.now(pytz.UTC) - opened_at).total_seconds()
                            elif isinstance(opened_at, str):
                                try:
                                    parsed_dt = datetime.fromisoformat(opened_at.replace('Z', '+00:00'))
                                    if parsed_dt.tzinfo is None:
                                        parsed_dt = pytz.UTC.localize(parsed_dt)
                                    duration_seconds = (datetime.now(pytz.UTC) - parsed_dt).total_seconds()
                                except (ValueError, TypeError, AttributeError):
                                    duration_seconds = 0
                        
                        exit_data = {
                            'result': trade_result,
                            'signal_type': signal_type,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'actual_pl': actual_pl,
                            'reason': reason,
                            'duration': duration_seconds
                        }
                        exit_msg = MessageFormatter.trade_exit(exit_data, pip_value=self.config.XAUUSD_PIP_VALUE)
                        
                        logger.info(f"📤 Mengirim notifikasi hasil trade {trade_result} ke user {user_id}...")
                        notification_sent = await self._safe_send_notification(user_id, exit_msg, parse_mode='Markdown')
                        
                        if notification_sent:
                            logger.info(f"✅ Notifikasi hasil trade {trade_result} BERHASIL terkirim ke user {user_id}")
                            
                            if chart_path and self.config.CHART_AUTO_DELETE:
                                await asyncio.sleep(1)
                                self.chart_generator.delete_chart(chart_path)
                                logger.info(f"Auto-deleted unused exit chart: {chart_path}")
                        else:
                            logger.warning(f"⚠️ Notifikasi hasil trade {trade_result} GAGAL - mencoba kirim via alert_system...")
                            
                            if self.alert_system:
                                try:
                                    await self.alert_system.send_trade_exit_alert({
                                        'signal_type': signal_type,
                                        'entry_price': entry_price,
                                        'exit_price': exit_price,
                                        'actual_pl': actual_pl
                                    }, trade_result)
                                    logger.info(f"✅ Notifikasi dikirim via alert_system sebagai fallback")
                                except Exception as alert_err:
                                    logger.error(f"❌ Fallback alert_system juga gagal: {alert_err}")
                        
                        self._cleanup_trailing_stop_notify(user_id, position_id)
                    else:
                        logger.warning(f"Not enough candles for exit chart: {len(df_m1) if df_m1 else 0}")
                        
                        if self.alert_system and trade_result:
                            await self.alert_system.send_trade_exit_alert({
                                'signal_type': signal_type,
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'actual_pl': actual_pl
                            }, trade_result)
                except (asyncio.TimeoutError, asyncio.CancelledError, RetryAfter, TimedOut, BadRequest, Forbidden, NetworkError, TelegramError, FileNotFoundError, IOError, OSError, ValueError, TypeError) as e:
                    logger.error(f"Error sending exit chart: {e}")
                    
                    if self.alert_system and trade_result:
                        await self.alert_system.send_trade_exit_alert({
                            'signal_type': signal_type,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'actual_pl': actual_pl
                        }, trade_result)
            elif self.alert_system and trade_result:
                await self.alert_system.send_trade_exit_alert({
                    'signal_type': signal_type,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'actual_pl': actual_pl
                }, trade_result)
            
            if self.user_manager:
                self.user_manager.update_user_stats(user_id, actual_pl)
            
            # REMOVED: Duplicate end_session call - sudah dipanggil di line 2289
            # end_session hanya perlu dipanggil SEKALI setelah posisi ditutup
            # Duplicate call ini menyebabkan event handler dipanggil 2x yang bisa
            # memicu cleanup prematur dan mengganggu monitoring loop
            
        except (IntegrityError, OperationalError, SQLAlchemyError, asyncio.CancelledError, asyncio.TimeoutError, ValueError, TypeError, KeyError) as e:
            logger.error(f"Error closing position {position_id}: {e}", exc_info=True)
            if session:
                try:
                    session.rollback()
                except (OperationalError, SQLAlchemyError) as rollback_error:
                    logger.error(f"Error during rollback: {rollback_error}")
        finally:
            if session:
                try:
                    session.close()
                except (OperationalError, SQLAlchemyError) as close_error:
                    logger.error(f"Error closing session: {close_error}")
    
    async def monitor_active_positions(self):
        """Monitor all active positions and apply dynamic SL/TP
        
        This method is called by the scheduler every 10 seconds.
        Returns a list of updated positions.
        Thread-safe with lock protection for reading active_positions
        """
        if self._shutdown_event.is_set():
            return []
        
        async with self._position_lock:
            if not self.active_positions:
                return []
            positions_snapshot = {
                user_id: list(positions.keys()) 
                for user_id, positions in self.active_positions.items()
            }
        
        if not self.market_data:
            logger.warning("Market data not available for position monitoring")
            return []
        
        updated_positions = []
        
        try:
            current_price = None
            try:
                current_price = await asyncio.wait_for(
                    self.market_data.get_current_price(),
                    timeout=3.0
                )
            except (asyncio.TimeoutError, ValueError, TypeError, ConnectionError):
                logger.debug("get_current_price timeout, trying fallback...")
                try:
                    last_candle = await asyncio.wait_for(
                        self.market_data.get_last_candle('M1'),
                        timeout=2.0
                    )
                    if last_candle is not None and len(last_candle) > 0:
                        current_price = float(last_candle.iloc[-1]['close']) if 'close' in last_candle.columns else None
                except (asyncio.TimeoutError, asyncio.CancelledError, ValueError, TypeError, KeyError, IndexError, AttributeError):
                    pass
            
            if not current_price or current_price <= 0:
                # CRITICAL: No price data available - this is dangerous for open positions!
                # Count consecutive failures and force close at SL if too many
                if not hasattr(self, '_no_price_count'):
                    self._no_price_count = 0
                self._no_price_count += 1
                
                total_active = sum(len(positions) for positions in positions_snapshot.values())
                logger.warning(f"No price data available (count={self._no_price_count}), {total_active} positions at risk!")
                
                # After 6 consecutive failures (~30-60 seconds with 10s interval), force close all positions at their SL
                if self._no_price_count >= 6 and total_active > 0:
                    logger.error(f"🚨 CRITICAL: No price data for {self._no_price_count} cycles, FORCE CLOSING {total_active} positions at SL!")
                    
                    # Track failed closes for escalation
                    if not hasattr(self, '_force_close_failures'):
                        self._force_close_failures = {}
                    
                    # Collect position IDs first, then close each with fresh SL under per-position lock
                    position_ids_to_close = []
                    async with self._position_lock:
                        for user_id in list(self.active_positions.keys()):
                            for position_id in list(self.active_positions.get(user_id, {}).keys()):
                                position_ids_to_close.append((user_id, position_id))
                    
                    # Close each position with fresh SL obtained right before close
                    for user_id, position_id in position_ids_to_close:
                        try:
                            # Re-acquire lock to get the LATEST SL value (may have been updated by trailing stop)
                            async with self._position_lock:
                                pos_data = self.active_positions.get(user_id, {}).get(position_id)
                                if not pos_data:
                                    logger.debug(f"Position {position_id} already closed, skipping")
                                    continue
                                sl_price = pos_data.get('stop_loss', pos_data.get('entry_price'))
                            
                            logger.warning(f"Force closing position {position_id} at SL=${sl_price:.2f} due to no price data")
                            await self.close_position(user_id, position_id, sl_price, 'NO_PRICE_DATA_SL')
                            updated_positions.append({
                                'user_id': user_id,
                                'position_id': position_id,
                                'result': 'NO_PRICE_DATA_SL',
                                'price': sl_price
                            })
                            # Clear failure count on success
                            self._force_close_failures.pop(position_id, None)
                        except Exception as e:
                            # Track failures for escalation
                            fail_count = self._force_close_failures.get(position_id, 0) + 1
                            self._force_close_failures[position_id] = fail_count
                            logger.error(f"Failed to force close position {position_id} (attempt {fail_count}): {e}")
                            
                            # After 3 failed attempts, remove from active tracking to prevent endless retries
                            if fail_count >= 3:
                                logger.critical(f"🚨 MANUAL INTERVENTION REQUIRED: Position {position_id} failed to close {fail_count} times!")
                                # Remove from active_positions to stop endless retry loop
                                async with self._position_lock:
                                    if user_id in self.active_positions and position_id in self.active_positions[user_id]:
                                        del self.active_positions[user_id][position_id]
                                        if not self.active_positions[user_id]:
                                            del self.active_positions[user_id]
                                        logger.warning(f"Removed position {position_id} from active tracking due to repeated close failures")
                                self._force_close_failures.pop(position_id, None)
                    
                    self._no_price_count = 0  # Reset counter after force close
                    return updated_positions
                
                return []
            
            # Reset counter when price is available
            self._no_price_count = 0
            
            logger.debug(f"Position monitoring: checking {len(positions_snapshot)} user(s) with price=${current_price:.2f}")
            
            for user_id, position_ids in positions_snapshot.items():
                for position_id in position_ids:
                    if self._shutdown_event.is_set():
                        logger.info("Shutdown requested, stopping position monitoring")
                        return updated_positions
                    
                    try:
                        result = await asyncio.wait_for(
                            self.update_position(user_id, position_id, current_price),
                            timeout=5.0
                        )
                        if result:
                            updated_positions.append({
                                'user_id': user_id,
                                'position_id': position_id,
                                'result': result,
                                'price': current_price
                            })
                            logger.info(f"Position {position_id} User:{user_id} closed: {result} at ${current_price:.2f}")
                    except asyncio.TimeoutError:
                        logger.debug(f"Timeout updating position {position_id} for user {user_id}")
                    except (asyncio.CancelledError, ValueError, TypeError, KeyError, AttributeError) as e:
                        logger.debug(f"Error monitoring position {position_id} for user {user_id}: {e}")
            
            return updated_positions
            
        except (asyncio.CancelledError, ConnectionError, ValueError, TypeError, KeyError, AttributeError) as e:
            logger.debug(f"Error in monitor_active_positions: {e}")
            return []
    
    async def check_stale_positions(self) -> List[Dict]:
        """Check dan auto-close posisi yang open >10 menit tanpa update.
        
        Method ini dipanggil oleh scheduler setiap 60 detik.
        Returns list posisi yang di-close.
        """
        closed_positions = []
        now = datetime.now(pytz.UTC)
        
        async with self._position_lock:
            users_to_check = list(self.active_positions.keys())
        
        for user_id in users_to_check:
            async with self._position_lock:
                if user_id not in self.active_positions:
                    continue
                positions_to_check = list(self.active_positions[user_id].items())
            
            for position_id, pos in positions_to_check:
                opened_at = pos.get('opened_at')
                last_update = pos.get('last_price_update')
                
                check_time = last_update or opened_at
                if check_time:
                    if isinstance(check_time, str):
                        try:
                            check_time = datetime.fromisoformat(check_time.replace('Z', '+00:00'))
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Failed to parse check_time for position {position_id}: {e}")
                            continue
                    if check_time.tzinfo is None:
                        check_time = pytz.UTC.localize(check_time)
                    
                    elapsed = (now - check_time).total_seconds()
                    
                    if elapsed >= STALE_POSITION_TIMEOUT:
                        logger.warning(
                            f"⏰ STALE POSITION detected - User:{user_id} Position:{position_id} "
                            f"No update for {elapsed:.0f}s (>{STALE_POSITION_TIMEOUT}s). Auto-closing..."
                        )
                        
                        current_price = pos.get('current_price') or pos.get('entry_price')
                        
                        try:
                            await self.close_position(user_id, position_id, current_price, 'STALE_TIMEOUT')
                        except (asyncio.CancelledError, asyncio.TimeoutError, ValueError, TypeError, KeyError) as e:
                            logger.error(f"Failed to close stale position {position_id}: {e}")
                            continue
                        
                        closed_positions.append({
                            'user_id': user_id,
                            'position_id': position_id,
                            'elapsed_seconds': elapsed,
                            'reason': 'STALE_TIMEOUT'
                        })
                        
                        msg = (
                            f"⏰ *Posisi Auto-Close: STALE_TIMEOUT*\n\n"
                            f"Position ID: {position_id}\n"
                            f"Type: {pos.get('signal_type', 'UNKNOWN')}\n"
                            f"Alasan: Tidak ada update selama {elapsed/60:.1f} menit\n"
                            f"Entry: ${pos.get('entry_price', 0):.2f}\n"
                            f"Exit: ${current_price:.2f}\n\n"
                            f"💡 Posisi ditutup otomatis karena tidak aktif."
                        )
                        await self._safe_send_notification(user_id, msg, parse_mode='Markdown')
                        
                        self._cleanup_trailing_stop_notify(user_id, position_id)
                        
                        if self.signal_session_manager:
                            try:
                                await self.signal_session_manager.end_session(user_id, 'STALE_TIMEOUT')
                            except (asyncio.CancelledError, asyncio.TimeoutError, ValueError, TypeError, AttributeError) as e:
                                logger.error(f"Failed to end session for stale position: {e}")
        
        if closed_positions:
            logger.info(f"🧹 Stale position cleanup: {len(closed_positions)} positions closed")
        
        return closed_positions
    
    async def monitor_positions(self, market_data_client):
        """Monitor positions with tick data stream
        
        Uses timeout on tick_queue.get() to allow graceful shutdown checking.
        This task is tracked for proper lifecycle management.
        """
        tick_queue = await market_data_client.subscribe_ticks('position_tracker')
        logger.info("Position tracker monitoring started")
        
        self.monitoring = True
        self._shutdown_event.clear()
        
        try:
            while self.monitoring and not self._shutdown_event.is_set():
                try:
                    try:
                        tick = await asyncio.wait_for(
                            tick_queue.get(),
                            timeout=TICK_QUEUE_TIMEOUT
                        )
                    except asyncio.TimeoutError:
                        continue
                    
                    async with self._position_lock:
                        if not self.active_positions:
                            continue
                        positions_snapshot = {
                            user_id: list(positions.keys()) 
                            for user_id, positions in self.active_positions.items()
                        }
                    
                    mid_price = tick['quote']
                    
                    for user_id, position_ids in positions_snapshot.items():
                        if not self.monitoring or self._shutdown_event.is_set():
                            break
                        for position_id in position_ids:
                            if not self.monitoring or self._shutdown_event.is_set():
                                break
                            try:
                                result = await asyncio.wait_for(
                                    self.update_position(user_id, position_id, mid_price),
                                    timeout=DEFAULT_OPERATION_TIMEOUT
                                )
                                if result:
                                    logger.info(f"Position {position_id} User:{user_id} closed: {result}")
                            except asyncio.TimeoutError:
                                logger.error(f"Timeout updating position {position_id}")
                            except (asyncio.CancelledError, ValueError, TypeError, KeyError, AttributeError) as e:
                                logger.error(f"Error updating position {position_id}: {e}")
                    
                except asyncio.CancelledError:
                    logger.info("Position monitoring cancelled")
                    break
                except (ConnectionError, ValueError, TypeError, KeyError, AttributeError) as e:
                    logger.error(f"Error processing tick dalam position monitoring: {e}")
                    if self.monitoring and not self._shutdown_event.is_set():
                        await asyncio.sleep(1)
                    
        finally:
            try:
                await market_data_client.unsubscribe_ticks('position_tracker')
            except (asyncio.CancelledError, asyncio.TimeoutError, ConnectionError, ValueError, TypeError) as e:
                logger.error(f"Error unsubscribing from ticks: {e}")
            logger.info("Position tracker monitoring stopped")
    
    def start_monitoring_task(self, market_data_client) -> asyncio.Task:
        """Start the monitoring task and track it for lifecycle management.
        
        Returns:
            asyncio.Task: The monitoring task
        """
        self._monitoring_task = self._create_tracked_task(
            self.monitor_positions(market_data_client),
            name="position_monitoring"
        )
        return self._monitoring_task
    
    def stop_monitoring(self):
        """Stop the monitoring loop"""
        self.monitoring = False
        self._shutdown_event.set()
        logger.info("Position monitoring stop requested")
    
    async def shutdown(self, timeout: float = TASK_CLEANUP_TIMEOUT):
        """Graceful shutdown with proper resource cleanup
        
        Cancels all pending tasks, stops monitoring loop and task monitor.
        
        Args:
            timeout: Maximum time to wait for task cleanup
        """
        logger.info("PositionTracker shutdown initiated...")
        
        self.monitoring = False
        self._shutdown_event.set()
        
        await self.stop_task_monitor()
        
        if self._monitoring_task and not self._monitoring_task.done():
            logger.info("Cancelling monitoring task...")
            self._monitoring_task.cancel()
            try:
                await asyncio.wait_for(self._monitoring_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        
        cancellation_results = await self._cancel_pending_tasks(timeout=timeout)
        
        if cancellation_results:
            timeout_count = sum(1 for s in cancellation_results.values() if 'timeout' in s)
            if timeout_count > 0:
                logger.warning(f"Shutdown completed with {timeout_count} tasks that did not cancel cleanly")
        
        with self._history_lock:
            self._slow_task_counts.clear()
            self._cancelled_task_names.clear()
        
        async with self._position_lock:
            active_count = sum(len(positions) for positions in self.active_positions.values())
            if active_count > 0:
                logger.warning(f"Shutting down with {active_count} active positions")
        
        logger.info("PositionTracker shutdown complete")
    
    async def get_active_positions_async(self, user_id: Optional[int] = None) -> Dict:
        """Thread-safe getter for active positions (async version)"""
        async with self._position_lock:
            if user_id is not None:
                return self.active_positions.get(user_id, {}).copy()
            return {uid: pos.copy() for uid, pos in self.active_positions.items()}
    
    def get_active_positions(self, user_id: Optional[int] = None) -> Dict:
        """Get active positions (sync version, returns snapshot)
        
        Note: For thread-safe access in async context, use get_active_positions_async
        WARNING: This is NOT fully thread-safe. Use get_active_positions_async for critical operations.
        """
        if user_id is not None:
            return self.active_positions.get(user_id, {}).copy()
        return {uid: pos.copy() for uid, pos in self.active_positions.items()}
    
    async def get_active_position_count_async(self, user_id: int) -> int:
        """Thread-safe count of active positions for a user.
        
        Returns:
            int: Number of active positions (in-memory + DB fallback)
        """
        count = 0
        
        async with self._position_lock:
            if user_id in self.active_positions:
                count = len(self.active_positions[user_id])
        
        if count == 0:
            try:
                session = self.db.get_session()
                try:
                    session.rollback()
                except Exception:
                    pass
                
                try:
                    db_count = session.query(Position).filter(
                        Position.user_id == user_id,
                        Position.status == 'ACTIVE'
                    ).count()
                    count = db_count
                finally:
                    session.close()
            except Exception as e:
                logger.error(f"Error counting active positions in DB: {e}")
        
        return count
    
    async def has_active_position_async(self, user_id: int) -> bool:
        """Thread-safe check for active position with multi-source verification including DB fallback
        
        NOTE: Does NOT check SignalSessionManager to avoid circular dependency
        Session != Position. Session tracks pending signal, Position tracks actual trade.
        """
        async with self._position_lock:
            if user_id in self.active_positions and len(self.active_positions[user_id]) > 0:
                return True
        
        db_has_active = await self.verify_active_position_in_db(user_id)
        if db_has_active:
            logger.debug(f"Active position found in DB for user {user_id} (not in cache/session)")
            return True
        
        return False
    
    def has_active_position(self, user_id: int) -> bool:
        """Check if user has active position with multi-source verification including DB fallback
        
        ⚠️ DEPRECATED: Gunakan has_active_position_async() untuk operasi kritis dalam context async.
        
        CATATAN THREAD-SAFETY:
        Method sinkron ini mengakses active_positions tanpa asyncio.Lock.
        Dalam context async dengan concurrent tasks, gunakan has_active_position_async()
        untuk thread-safety yang lebih baik.
        
        Method ini aman digunakan untuk:
        - Quick read-only checks di synchronous code
        - Non-critical logging/debugging
        - Fallback ketika async context tidak tersedia
        
        JANGAN gunakan untuk:
        - Signal creation decisions (gunakan has_active_position_async)
        - Race-critical position checks (gunakan has_active_position_async)
        
        Checks multiple sources for redundancy:
        1. In-memory active_positions cache (non-locked, read-only)
        2. Database fallback (critical for restart scenarios)
        
        NOTE: Does NOT check SignalSessionManager to avoid circular dependency
        Session != Position. Session tracks pending signal, Position tracks actual trade.
        """
        if user_id in self.active_positions and len(self.active_positions[user_id]) > 0:
            return True
        
        try:
            session = self.db.get_session()
            try:
                try:
                    session.rollback()
                except Exception:
                    pass
                
                active_count = session.query(Position).filter(
                    Position.user_id == user_id,
                    Position.status == 'ACTIVE'
                ).count()
                if active_count > 0:
                    logger.debug(f"Active position found in DB for user {user_id} (not in cache/session)")
                    return True
            finally:
                session.close()
        except Exception as e:
            logger.error(f"Error checking DB for active position: {e}")
        
        return False
    
    async def verify_active_position_in_db(self, user_id: int) -> bool:
        """Verify active position exists in database (for critical operations)"""
        session = None
        try:
            session = self.db.get_session()
            try:
                session.rollback()
            except Exception:
                pass
            
            active_count = session.query(Position).filter(
                Position.user_id == user_id,
                Position.status == 'ACTIVE'
            ).count()
            return active_count > 0
        except Exception as e:
            logger.error(f"Error verifying position in DB: {e}")
            return False
        finally:
            if session:
                session.close()
    
    async def has_active_position_verified(self, user_id: int) -> bool:
        """
        Check if user has active position by verifying ONLY database and memory cache.
        This method does NOT check SignalSessionManager to avoid circular dependency.
        Used by SignalSessionManager to validate if stale sessions should be cleaned.
        """
        async with self._position_lock:
            if user_id in self.active_positions and len(self.active_positions[user_id]) > 0:
                return True
        
        return await self.verify_active_position_in_db(user_id)
    
    async def reload_active_positions(self):
        """Reload active positions from the database
        
        This method is used to restore position tracking after a restart.
        Thread-safe with lock protection.
        """
        session = None
        try:
            session = self.db.get_session()
            try:
                session.rollback()
            except Exception:
                pass
            
            active_positions = session.query(Position).filter(Position.status == 'ACTIVE').all()
            
            async with self._position_lock:
                self.active_positions.clear()
                
                for pos in active_positions:
                    if pos.user_id not in self.active_positions:
                        self.active_positions[pos.user_id] = {}
                    
                    self.active_positions[pos.user_id][pos.id] = {
                        'trade_id': pos.trade_id,
                        'signal_type': pos.signal_type,
                        'entry_price': pos.entry_price,
                        'stop_loss': pos.stop_loss,
                        'take_profit': pos.take_profit,
                        'original_sl': pos.original_sl or pos.stop_loss,
                        'sl_adjustment_count': pos.sl_adjustment_count or 0,
                        'max_profit_reached': pos.max_profit_reached or 0.0,
                        'signal_quality_id': getattr(pos, 'signal_quality_id', None),
                        'signal_grade': getattr(pos, 'signal_grade', 'B'),
                        'confidence_score': getattr(pos, 'confidence_score', 0.5),
                        'opened_at': pos.opened_at
                    }
            
            total_positions = sum(len(positions) for positions in self.active_positions.values())
            logger.info(f"Reloaded {total_positions} active positions from database")
            
        except (IntegrityError, OperationalError, SQLAlchemyError, ValueError, TypeError) as e:
            logger.error(f"Error reloading active positions: {e}", exc_info=True)
            if session:
                try:
                    session.rollback()
                except (OperationalError, SQLAlchemyError) as rollback_error:
                    logger.error(f"Error during rollback: {rollback_error}")
        finally:
            if session:
                try:
                    session.close()
                except (OperationalError, SQLAlchemyError) as close_error:
                    logger.error(f"Error closing session: {close_error}")
