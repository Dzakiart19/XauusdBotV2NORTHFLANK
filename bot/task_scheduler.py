import asyncio
import gc
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Callable, Optional, Dict, List, Set, Any, Tuple
import pytz
from bot.logger import setup_logger
from config import Config

logger = setup_logger('TaskScheduler')


class TaskSchedulerError(Exception):
    """Exception untuk error pada task scheduler"""
    pass


class LifecycleState(Enum):
    """
    State enum untuk AsyncLifecycleController pattern.
    
    State Machine Transitions:
    - IDLE -> STARTING (via start())
    - STARTING -> RUNNING (setelah start berhasil)
    - RUNNING -> STOPPING (via stop())
    - STOPPING -> IDLE (setelah stop berhasil)
    - STOPPING -> FAILED_STOP (jika cleanup gagal)
    - FAILED_STOP -> IDLE (via reset_failed_stop())
    """
    IDLE = "idle"               # Belum pernah start atau sudah stop sukses
    STARTING = "starting"       # Sedang dalam proses start
    RUNNING = "running"         # Berjalan normal
    STOPPING = "stopping"       # Sedang dalam proses stop
    FAILED_STOP = "failed_stop" # Stop gagal, perlu reset manual


@dataclass
class CleanupResult:
    """
    Structured result dari cleanup operation.
    
    Digunakan untuk tracking cleanup status dan error details.
    """
    success: bool
    operation: str
    items_cleaned: int = 0
    error_message: Optional[str] = None
    duration_ms: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'operation': self.operation,
            'items_cleaned': self.items_cleaned,
            'error_message': self.error_message,
            'duration_ms': round(self.duration_ms, 2)
        }


# Timeout constants (dalam detik)
TASK_EXECUTION_TIMEOUT = 60
TASK_CANCEL_TIMEOUT = 5
SCHEDULER_STOP_TIMEOUT = 10
FLUSH_PENDING_TIMEOUT = 15
STALE_TASK_THRESHOLD_MINUTES = 30
MAX_CONSECUTIVE_FAILURES_AUTO_DISABLE = 10
EXCEPTION_HISTORY_LIMIT = 20
CLEANUP_INTERVAL_SECONDS = Config.COMPLETED_TASKS_CLEANUP_INTERVAL

# Task history cleanup constants
MAX_COMPLETED_TASK_HISTORY = 100
COMPLETED_TASK_MAX_AGE_MINUTES = 30
HISTORY_CLEANUP_INTERVAL_SECONDS = 300  # 5 minutes
DEFAULT_DRAIN_TIMEOUT = 30  # Default timeout for draining tasks during shutdown


class TaskPriority(Enum):
    """
    Priority levels for task execution.
    
    - CRITICAL: Tasks run immediately, bypass all delays
    - HIGH: Tasks skip regular interval delays
    - NORMAL: Tasks follow regular schedule
    - LOW: Tasks run only when scheduler is not busy
    """
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4

# Task lifecycle optimization constants (dari Config)
TASK_AUTO_CANCEL_THRESHOLD = Config.TASK_AUTO_CANCEL_THRESHOLD
TASK_ROTATION_INTERVAL = Config.TASK_ROTATION_INTERVAL
TASK_RESTART_BACKOFF_BASE = Config.TASK_RESTART_BACKOFF_BASE
TASK_RESTART_BACKOFF_MAX = Config.TASK_RESTART_BACKOFF_MAX
COMPLETED_TASKS_MAX_SIZE = Config.COMPLETED_TASKS_MAX_SIZE

# Timeout tambahan untuk cleanup operations yang robust
CLEANUP_TASK_TIMEOUT = 10  # Timeout untuk setiap cleanup task individual
CLEANUP_FALLBACK_TIMEOUT = 5  # Timeout untuk fallback cleanup mechanism
LOCK_ACQUIRE_TIMEOUT = 3  # Timeout untuk acquire lock
DST_AMBIGUOUS_RESOLUTION = 'earlier'  # Resolusi untuk waktu DST ambigu (earlier/later)


def _safe_localize(tz, dt: datetime, is_dst: Optional[bool] = None) -> datetime:
    """
    Localize datetime dengan handling DST yang aman.
    
    Args:
        tz: Timezone pytz
        dt: Datetime naive yang akan di-localize
        is_dst: Flag untuk menentukan DST saat ambigu (None untuk auto-detect)
    
    Returns:
        Datetime yang sudah di-localize dengan benar
    """
    try:
        if dt.tzinfo is not None:
            return dt.astimezone(tz)
        return tz.localize(dt, is_dst=is_dst)
    except pytz.exceptions.AmbiguousTimeError:
        # Waktu ambigu saat DST berakhir (jam mundur)
        is_dst_fallback = DST_AMBIGUOUS_RESOLUTION == 'later'
        logger.warning(f"Waktu ambigu terdeteksi: {dt}, menggunakan is_dst={is_dst_fallback}")
        return tz.localize(dt, is_dst=is_dst_fallback)
    except pytz.exceptions.NonExistentTimeError:
        # Waktu tidak ada saat DST dimulai (jam melompat)
        # Geser waktu ke jam berikutnya yang valid
        logger.warning(f"Waktu tidak ada (DST gap) terdeteksi: {dt}, menggeser ke waktu valid berikutnya")
        normalized = tz.normalize(tz.localize(dt, is_dst=True))
        return normalized
    except Exception as e:
        logger.error(f"Error saat localize datetime {dt}: {e}")
        # Fallback: gunakan UTC dan convert
        return datetime.now(tz)


def _calculate_next_schedule_time_dst_safe(schedule_time: time, tz, now: datetime) -> datetime:
    """
    Hitung next_run untuk scheduled task dengan handling DST yang aman.
    
    Args:
        schedule_time: Waktu schedule (time object)
        tz: Timezone pytz
        now: Waktu sekarang (timezone-aware)
    
    Returns:
        Datetime next_run yang sudah di-normalize dengan benar
    """
    try:
        # Buat datetime naive untuk target waktu hari ini
        next_run_naive = now.replace(
            hour=schedule_time.hour,
            minute=schedule_time.minute,
            second=schedule_time.second,
            microsecond=0,
            tzinfo=None
        )
        
        # Localize dengan handling DST
        next_run = _safe_localize(tz, next_run_naive)
        
        # Jika sudah lewat, geser ke hari berikutnya
        if next_run <= now:
            next_run_naive += timedelta(days=1)
            next_run = _safe_localize(tz, next_run_naive.replace(tzinfo=None))
            
            # Validasi bahwa next_run benar-benar di masa depan setelah normalisasi DST
            if next_run <= now:
                # Edge case: DST menyebabkan next_run masih di masa lalu
                logger.warning(f"DST edge case: next_run {next_run} masih <= now {now}, menambah 1 hari lagi")
                next_run_naive += timedelta(days=1)
                next_run = _safe_localize(tz, next_run_naive.replace(tzinfo=None))
        
        # Normalize untuk memastikan offset DST yang benar
        next_run = tz.normalize(next_run)
        
        return next_run
        
    except Exception as e:
        logger.error(f"Error saat menghitung next schedule time: {e}")
        # Fallback: gunakan pendekatan sederhana
        next_run = now.replace(
            hour=schedule_time.hour,
            minute=schedule_time.minute,
            second=schedule_time.second,
            microsecond=0
        )
        if next_run <= now:
            next_run += timedelta(days=1)
        return next_run


async def _safe_cleanup_with_timeout(
    cleanup_func,
    timeout: float,
    fallback_func=None,
    operation_name: str = "cleanup"
) -> bool:
    """
    Jalankan cleanup function dengan timeout dan fallback mechanism.
    
    Args:
        cleanup_func: Async function untuk cleanup
        timeout: Timeout dalam detik
        fallback_func: Optional fallback function jika primary gagal
        operation_name: Nama operasi untuk logging
    
    Returns:
        bool: True jika berhasil, False jika gagal
    """
    primary_success = False
    
    try:
        await asyncio.wait_for(cleanup_func(), timeout=timeout)
        primary_success = True
        return True
    except asyncio.TimeoutError:
        logger.warning(f"Timeout saat {operation_name} setelah {timeout}s")
    except asyncio.CancelledError:
        logger.info(f"{operation_name} dibatalkan")
        raise
    except Exception as e:
        logger.error(f"Error saat {operation_name}: {e}")
    
    # Jalankan fallback jika primary gagal
    if not primary_success and fallback_func is not None:
        try:
            logger.info(f"Menjalankan fallback untuk {operation_name}...")
            await asyncio.wait_for(fallback_func(), timeout=CLEANUP_FALLBACK_TIMEOUT)
            logger.info(f"Fallback {operation_name} berhasil")
            return True
        except asyncio.TimeoutError:
            logger.error(f"Fallback {operation_name} juga timeout")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error saat fallback {operation_name}: {e}")
    
    return False


async def _acquire_lock_with_timeout(lock: asyncio.Lock, timeout: float = LOCK_ACQUIRE_TIMEOUT) -> bool:
    """
    Acquire lock dengan timeout untuk mencegah deadlock.
    
    Args:
        lock: asyncio.Lock yang akan di-acquire
        timeout: Timeout dalam detik
    
    Returns:
        bool: True jika berhasil acquire, False jika timeout
    """
    try:
        await asyncio.wait_for(lock.acquire(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        logger.warning(f"Timeout saat acquire lock setelah {timeout}s")
        return False
    except Exception as e:
        logger.error(f"Error saat acquire lock: {e}")
        return False


@dataclass
class ExceptionRecord:
    """Record of an exception with timestamp."""
    exception: Exception
    timestamp: datetime
    error_type: str = ""
    
    def __post_init__(self):
        self.error_type = type(self.exception).__name__
    
    def to_dict(self) -> Dict:
        return {
            'error_type': self.error_type,
            'message': str(self.exception),
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class TaskHealthMetrics:
    """Health metrics for a scheduled task."""
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    total_execution_time_ms: float = 0.0
    last_successful_run: Optional[datetime] = None
    last_execution_time_ms: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_runs == 0:
            return 0.0
        return (self.successful_runs / self.total_runs) * 100.0
    
    @property
    def average_execution_time_ms(self) -> float:
        """Calculate average execution time in milliseconds."""
        if self.successful_runs == 0:
            return 0.0
        return self.total_execution_time_ms / self.successful_runs
    
    def time_since_last_success(self, timezone) -> Optional[timedelta]:
        """Get time elapsed since last successful run."""
        if self.last_successful_run is None:
            return None
        now = datetime.now(timezone)
        if self.last_successful_run.tzinfo is None:
            last_success = timezone.localize(self.last_successful_run)
        else:
            last_success = self.last_successful_run
        return now - last_success
    
    def to_dict(self) -> Dict:
        return {
            'total_runs': self.total_runs,
            'successful_runs': self.successful_runs,
            'failed_runs': self.failed_runs,
            'success_rate': round(self.success_rate, 2),
            'average_execution_time_ms': round(self.average_execution_time_ms, 2),
            'last_execution_time_ms': round(self.last_execution_time_ms, 2),
            'last_successful_run': self.last_successful_run.isoformat() if self.last_successful_run else None
        }


class ScheduledTask:
    """Represents a scheduled task with execution tracking and error handling.
    
    Task Lifecycle:
    - Tasks are created with interval or schedule_time
    - should_run() determines if task is ready to execute
    - execute() runs the task with timeout protection
    - Done callbacks drain exceptions and log completion
    
    Health Monitoring:
    - Tracks success rate, execution times, and exception history
    - Auto-disables tasks with excessive consecutive failures
    - Detects stale tasks that haven't run for extended periods
    
    Priority System:
    - CRITICAL: Run immediately, bypass all delays
    - HIGH: Skip interval delays
    - NORMAL: Follow regular schedule
    - LOW: Run only when scheduler is not busy
    """
    def __init__(self, name: str, func: Callable, interval: Optional[int] = None,
                 schedule_time: Optional[time] = None, timezone: str = 'Asia/Jakarta',
                 auto_disable_threshold: int = MAX_CONSECUTIVE_FAILURES_AUTO_DISABLE,
                 priority: TaskPriority = TaskPriority.NORMAL):
        self.name = name
        self.func = func
        self.interval = interval
        self.schedule_time = schedule_time
        self.timezone = pytz.timezone(timezone)
        self.priority = priority
        self.last_run = None
        self.next_run = None
        self.enabled = True
        self.run_count = 0
        self.error_count = 0
        self.consecutive_failures = 0
        self.current_execution_task = None
        self._is_executing = False
        self._last_exception: Optional[Exception] = None
        
        self._exception_history: deque = deque(maxlen=EXCEPTION_HISTORY_LIMIT)
        self._health_metrics = TaskHealthMetrics()
        self._auto_disable_threshold = auto_disable_threshold
        self._auto_disabled = False
        self._created_at = datetime.now(self.timezone)
        self._execution_start_time: Optional[datetime] = None
        self._completion_time: Optional[datetime] = None
        
        self._calculate_next_run()
    
    def _calculate_next_run(self):
        """
        Hitung waktu next_run dengan handling DST yang robust.
        
        Menangani edge cases:
        - DST transition saat jam melompat (non-existent time)
        - DST transition saat jam mundur (ambiguous time)
        - Timezone validation
        """
        try:
            now = datetime.now(self.timezone)
            
            if self.schedule_time:
                # Gunakan helper function yang DST-safe
                self.next_run = _calculate_next_schedule_time_dst_safe(
                    self.schedule_time,
                    self.timezone,
                    now
                )
                logger.debug(f"Task {self.name}: next_run dihitung = {self.next_run}")
            
            elif self.interval:
                if self.last_run:
                    # Pastikan last_run memiliki timezone info
                    if self.last_run.tzinfo is None:
                        last_run_aware = _safe_localize(self.timezone, self.last_run)
                    else:
                        last_run_aware = self.last_run.astimezone(self.timezone)
                    
                    self.next_run = last_run_aware + timedelta(seconds=self.interval)
                    
                    # Normalize untuk DST transition
                    self.next_run = self.timezone.normalize(self.next_run)
                else:
                    self.next_run = now
                    
        except pytz.exceptions.UnknownTimeZoneError as e:
            logger.error(f"Timezone tidak dikenal untuk task {self.name}: {e}")
            # Fallback ke UTC
            self.next_run = datetime.now(pytz.UTC)
        except Exception as e:
            logger.error(f"Error menghitung next_run untuk task {self.name}: {e}")
            # Fallback sederhana
            self.next_run = datetime.now(self.timezone) + timedelta(seconds=60)
    
    def should_run(self) -> bool:
        if not self.enabled:
            return False
        
        if self.next_run is None:
            return False
        
        if self._is_executing:
            return False
        
        now = datetime.now(self.timezone)
        
        # CRITICAL priority tasks always run if not currently executing
        if self.priority == TaskPriority.CRITICAL:
            return True
        
        # HIGH priority tasks skip delay check
        if self.priority == TaskPriority.HIGH:
            return True
        
        return now >= self.next_run
    
    def should_run_priority(self, active_task_count: int = 0, max_concurrent: int = 10) -> bool:
        """
        Check if task should run considering priority and system load.
        
        Args:
            active_task_count: Number of currently active tasks
            max_concurrent: Maximum concurrent tasks allowed
            
        Returns:
            bool: True if task should run
        """
        if not self.enabled or self._is_executing:
            return False
        
        # CRITICAL always runs regardless of load
        if self.priority == TaskPriority.CRITICAL:
            return True
        
        # HIGH priority runs unless at max capacity
        if self.priority == TaskPriority.HIGH:
            return active_task_count < max_concurrent
        
        # NORMAL priority follows regular schedule
        if self.priority == TaskPriority.NORMAL:
            if self.next_run is None:
                return False
            now = datetime.now(self.timezone)
            return now >= self.next_run and active_task_count < max_concurrent
        
        # LOW priority only runs when scheduler is not busy (< 50% capacity)
        if self.priority == TaskPriority.LOW:
            if self.next_run is None:
                return False
            now = datetime.now(self.timezone)
            return now >= self.next_run and active_task_count < (max_concurrent // 2)
        
        return False
    
    def set_priority(self, priority: TaskPriority) -> None:
        """Set task priority."""
        self.priority = priority
        logger.info(f"Task {self.name} priority set to {priority.name}")
    
    def mark_completed(self) -> None:
        """Mark task as completed with timestamp."""
        self._completion_time = datetime.now(self.timezone)
    
    def get_completion_time(self) -> Optional[datetime]:
        """Get task completion timestamp."""
        return self._completion_time
    
    def is_stale(self, threshold_minutes: int = STALE_TASK_THRESHOLD_MINUTES) -> bool:
        """Check if task is stale (enabled but hasn't run for extended period).
        
        Args:
            threshold_minutes: Minutes without running to be considered stale
            
        Returns:
            bool: True if task is stale
        """
        if not self.enabled:
            return False
        
        if self._is_executing:
            return False
        
        now = datetime.now(self.timezone)
        
        if self.last_run is None:
            time_since_creation = now - self._created_at
            return time_since_creation > timedelta(minutes=threshold_minutes)
        
        if self.last_run.tzinfo is None:
            last_run = self.timezone.localize(self.last_run)
        else:
            last_run = self.last_run
        
        time_since_last_run = now - last_run
        return time_since_last_run > timedelta(minutes=threshold_minutes)
    
    def has_high_failure_rate(self, threshold: int = 5) -> bool:
        """Check if task has high consecutive failures.
        
        Args:
            threshold: Number of consecutive failures to be considered high
            
        Returns:
            bool: True if consecutive failures exceed threshold
        """
        return self.consecutive_failures >= threshold
    
    def should_auto_disable(self) -> bool:
        """Check if task should be auto-disabled due to excessive failures."""
        return self.consecutive_failures >= self._auto_disable_threshold
    
    def _record_exception(self, exception: Exception) -> None:
        """Record exception with timestamp in history."""
        record = ExceptionRecord(
            exception=exception,
            timestamp=datetime.now(self.timezone)
        )
        self._exception_history.append(record)
    
    def get_exception_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Get exception history for this task.
        
        Args:
            limit: Maximum number of recent exceptions to return
            
        Returns:
            List of exception records as dictionaries
        """
        history = list(self._exception_history)
        if limit is not None:
            history = history[-limit:]
        return [record.to_dict() for record in history]
    
    def get_health_metrics(self) -> Dict:
        """Get health metrics for this task."""
        metrics = self._health_metrics.to_dict()
        time_since_success = self._health_metrics.time_since_last_success(self.timezone)
        if time_since_success:
            metrics['time_since_last_success_seconds'] = time_since_success.total_seconds()
        else:
            metrics['time_since_last_success_seconds'] = None
        metrics['is_stale'] = self.is_stale()
        metrics['has_high_failure_rate'] = self.has_high_failure_rate()
        metrics['auto_disabled'] = self._auto_disabled
        return metrics
    
    def clear_exception_history(self) -> None:
        """Clear all stored exception history."""
        self._exception_history.clear()
    
    def reset_health_metrics(self) -> None:
        """Reset health metrics to initial state."""
        self._health_metrics = TaskHealthMetrics()
        self.consecutive_failures = 0
        self._auto_disabled = False
    
    async def execute(self, alert_system=None, shutdown_flag: Optional[asyncio.Event] = None,
                      on_complete: Optional[Callable[[asyncio.Task, str], None]] = None):
        """Execute the scheduled task with proper lifecycle management.
        
        Args:
            alert_system: Optional alert system for error notifications
            shutdown_flag: Event that signals shutdown request
            on_complete: Optional callback invoked when task completes (receives task and task name)
        """
        if self._is_executing:
            logger.warning(f"Task {self.name} is already executing, skipping")
            return
        
        self._is_executing = True
        self._last_exception = None
        self._execution_start_time = datetime.now(self.timezone)
        
        try:
            logger.info(f"Executing scheduled task: {self.name}")
            
            if shutdown_flag and shutdown_flag.is_set():
                logger.info(f"Shutdown requested, skipping task: {self.name}")
                return
            
            if asyncio.iscoroutinefunction(self.func):
                self.current_execution_task = asyncio.create_task(self.func())
                self.current_execution_task.set_name(f"exec_{self.name}")
                
                if on_complete:
                    def _done_callback(t: asyncio.Task):
                        try:
                            on_complete(t, self.name)
                        except (ValueError, TypeError) as cb_err:
                            logger.error(f"Completion callback validation error for {self.name}: {cb_err}")
                        except RuntimeError as cb_err:
                            logger.error(f"Completion callback runtime error for {self.name}: {cb_err}")
                        except Exception as cb_err:
                            logger.error(f"Completion callback error for {self.name}: {cb_err}")
                    self.current_execution_task.add_done_callback(_done_callback)
                
                try:
                    await asyncio.wait_for(self.current_execution_task, timeout=TASK_EXECUTION_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.warning(f"Task {self.name} timed out after {TASK_EXECUTION_TIMEOUT}s")
                    if self.current_execution_task and not self.current_execution_task.done():
                        self.current_execution_task.cancel()
                        try:
                            await asyncio.wait_for(self.current_execution_task, timeout=TASK_CANCEL_TIMEOUT)
                        except (asyncio.CancelledError, asyncio.TimeoutError):
                            pass
                    raise
            else:
                self.func()
            
            execution_end_time = datetime.now(self.timezone)
            execution_time_ms = (execution_end_time - self._execution_start_time).total_seconds() * 1000
            
            self.last_run = execution_end_time
            self.run_count += 1
            self.consecutive_failures = 0
            
            self._health_metrics.total_runs += 1
            self._health_metrics.successful_runs += 1
            self._health_metrics.total_execution_time_ms += execution_time_ms
            self._health_metrics.last_execution_time_ms = execution_time_ms
            self._health_metrics.last_successful_run = execution_end_time
            
            self._calculate_next_run()
            
            logger.info(f"Task completed: {self.name} (Total runs: {self.run_count}, Execution time: {execution_time_ms:.2f}ms)")
            
        except asyncio.CancelledError:
            logger.info(f"Task cancelled: {self.name}")
            raise
        except asyncio.TimeoutError:
            self.error_count += 1
            self.consecutive_failures += 1
            timeout_exception = asyncio.TimeoutError(f"Task {self.name} timed out")
            self._last_exception = timeout_exception
            self._record_exception(timeout_exception)
            self._health_metrics.total_runs += 1
            self._health_metrics.failed_runs += 1
            self._calculate_next_run()
            logger.error(f"Task {self.name} timed out (Consecutive failures: {self.consecutive_failures})")
            
            self._check_auto_disable(alert_system)
            
        except (ValueError, TypeError) as e:
            self.error_count += 1
            self.consecutive_failures += 1
            self._last_exception = e
            self._record_exception(e)
            self._health_metrics.total_runs += 1
            self._health_metrics.failed_runs += 1
            self._calculate_next_run()
            logger.error(f"Validation error executing task {self.name}: {e} (Consecutive failures: {self.consecutive_failures})")
            self._check_auto_disable(alert_system)
            
        except RuntimeError as e:
            self.error_count += 1
            self.consecutive_failures += 1
            self._last_exception = e
            self._record_exception(e)
            self._health_metrics.total_runs += 1
            self._health_metrics.failed_runs += 1
            self._calculate_next_run()
            logger.error(f"Runtime error executing task {self.name}: {e} (Consecutive failures: {self.consecutive_failures})")
            self._check_auto_disable(alert_system)
            
        except Exception as e:
            self.error_count += 1
            self.consecutive_failures += 1
            self._last_exception = e
            self._record_exception(e)
            self._health_metrics.total_runs += 1
            self._health_metrics.failed_runs += 1
            self._calculate_next_run()
            logger.error(f"Error executing task {self.name}: {e} (Consecutive failures: {self.consecutive_failures})")
            
            if self.consecutive_failures > 3 and alert_system:
                try:
                    alert_task = asyncio.create_task(
                        alert_system.send_system_error(
                            f"Task '{self.name}' failed {self.consecutive_failures}x consecutively\n"
                            f"Last error: {str(e)}\n"
                            f"Total errors: {self.error_count}/{self.run_count} runs\n"
                            f"Success rate: {self._health_metrics.success_rate:.1f}%"
                        )
                    )
                    try:
                        await asyncio.wait_for(alert_task, timeout=10)
                    except asyncio.TimeoutError:
                        logger.warning(f"Alert task for {self.name} timed out")
                    except asyncio.CancelledError:
                        raise
                    logger.warning(f"Alert sent: Task {self.name} has {self.consecutive_failures} consecutive failures")
                except asyncio.CancelledError:
                    raise
                except asyncio.TimeoutError as alert_error:
                    logger.error(f"Timeout sending task failure alert: {alert_error}")
                except Exception as alert_error:
                    logger.error(f"Failed to send task failure alert: {alert_error}")
            
            self._check_auto_disable(alert_system)
            
        finally:
            self.current_execution_task = None
            self._is_executing = False
            self._execution_start_time = None
    
    def _check_auto_disable(self, alert_system=None) -> None:
        """Check and apply auto-disable if threshold is reached."""
        if self.should_auto_disable() and not self._auto_disabled:
            self._auto_disabled = True
            self.enabled = False
            logger.warning(f"Task {self.name} auto-disabled after {self.consecutive_failures} consecutive failures")
            
            if alert_system:
                try:
                    import asyncio
                    asyncio.create_task(
                        alert_system.send_system_error(
                            f"⚠️ Task '{self.name}' has been AUTO-DISABLED\n"
                            f"Reason: {self.consecutive_failures} consecutive failures\n"
                            f"Total errors: {self.error_count}\n"
                            f"Success rate: {self._health_metrics.success_rate:.1f}%\n"
                            f"Last error: {self._last_exception}"
                        )
                    )
                except RuntimeError as e:
                    logger.error(f"Runtime error sending auto-disable alert: {e}")
                except Exception as e:
                    logger.error(f"Failed to send auto-disable alert: {e}")
    
    def get_last_exception(self) -> Optional[Exception]:
        """Get the last exception raised by this task."""
        return self._last_exception
    
    async def cancel_execution(self, timeout: float = TASK_CANCEL_TIMEOUT) -> bool:
        if self.current_execution_task and not self.current_execution_task.done():
            logger.info(f"Cancelling task execution: {self.name}")
            self.current_execution_task.cancel()
            try:
                await asyncio.wait_for(self.current_execution_task, timeout=timeout)
                logger.info(f"Task {self.name} cancelled successfully")
                return True
            except asyncio.TimeoutError:
                logger.warning(f"Task {self.name} cancellation timed out after {timeout}s")
                return False
            except asyncio.CancelledError:
                logger.info(f"Task {self.name} cancelled")
                return True
        return True
    
    def enable(self, reset_failures: bool = False):
        """Enable the task.
        
        Args:
            reset_failures: If True, reset consecutive failures and auto_disabled flag
        """
        self.enabled = True
        if reset_failures:
            self.consecutive_failures = 0
            self._auto_disabled = False
        self._calculate_next_run()
        logger.info(f"Task enabled: {self.name}" + (" (failures reset)" if reset_failures else ""))
    
    def disable(self):
        self.enabled = False
        logger.info(f"Task disabled: {self.name}")
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'interval': self.interval,
            'schedule_time': self.schedule_time.isoformat() if self.schedule_time else None,
            'priority': self.priority.name,
            'priority_value': self.priority.value,
            'enabled': self.enabled,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'next_run': self.next_run.isoformat() if self.next_run else None,
            'completion_time': self._completion_time.isoformat() if self._completion_time else None,
            'run_count': self.run_count,
            'error_count': self.error_count,
            'consecutive_failures': self.consecutive_failures,
            'is_executing': self._is_executing,
            'auto_disabled': self._auto_disabled,
            'is_stale': self.is_stale(),
            'has_high_failure_rate': self.has_high_failure_rate(),
            'last_exception': str(self._last_exception) if self._last_exception else None,
            'health_metrics': self._health_metrics.to_dict(),
            'exception_history_count': len(self._exception_history)
        }

@dataclass
class CompletedTaskRecord:
    """Record of a completed task execution."""
    task_name: str
    completion_time: datetime
    success: bool
    execution_time_ms: float = 0.0
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'task_name': self.task_name,
            'completion_time': self.completion_time.isoformat(),
            'success': self.success,
            'execution_time_ms': round(self.execution_time_ms, 2),
            'error_message': self.error_message
        }
    
    def age_minutes(self, now: datetime) -> float:
        """Get age of this record in minutes."""
        if self.completion_time.tzinfo is None:
            return (now.replace(tzinfo=None) - self.completion_time).total_seconds() / 60
        return (now - self.completion_time).total_seconds() / 60


class TaskScheduler:
    """Task scheduler with background task callbacks and graceful shutdown.
    
    Background Task Callbacks:
    - All scheduled task futures are tracked in _active_task_executions
    - Completion callbacks drain exceptions and log task completion
    - flush_pending_tasks() allows waiting for all tasks with timeout
    - Graceful cancellation with bounded wait timeout
    
    Aggressive Cleanup:
    - Periodic cleanup of completed tasks from _active_task_executions
    - Cleanup of orphaned tasks (tasks without scheduler)
    - Detection of stale tasks that haven't run for extended periods
    
    Priority System:
    - CRITICAL: Tasks run immediately, bypass all delays
    - HIGH: Tasks skip interval delays  
    - NORMAL: Tasks follow regular schedule
    - LOW: Tasks run only when scheduler is not busy
    """
    def __init__(self, config, alert_system=None, cleanup_interval: int = CLEANUP_INTERVAL_SECONDS):
        self.config = config
        self.alert_system = alert_system
        self.tasks: Dict[str, ScheduledTask] = {}
        self.running = False
        self.scheduler_task = None
        self._cleanup_task = None
        self._history_cleanup_task = None
        self._shutdown_flag = asyncio.Event()
        self._active_task_executions: Set[asyncio.Task] = set()
        self._all_created_tasks: Set[asyncio.Task] = set()
        self._lock = asyncio.Lock()
        self._task_exceptions: Dict[str, BaseException] = {}
        self._exception_history: deque = deque(maxlen=100)
        self._completion_event = asyncio.Event()
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = datetime.now(pytz.timezone('Asia/Jakarta'))
        
        # Completed task history tracking
        self._completed_tasks_history: deque = deque(maxlen=MAX_COMPLETED_TASK_HISTORY * 2)
        self._last_history_cleanup = datetime.now(pytz.timezone('Asia/Jakarta'))
        
        self._cleanup_stats = {
            'completed_tasks_cleaned': 0,
            'orphaned_tasks_cleaned': 0,
            'stale_tasks_detected': 0,
            'auto_disabled_tasks': 0,
            'last_cleanup_time': None,
            'auto_cancelled_stuck_tasks': 0,
            'entries_cleaned_by_size': 0,
            'task_rotations': 0,
            'history_entries_cleaned': 0,
            'history_cleanups_run': 0
        }
        self._last_rotation = datetime.now(pytz.timezone('Asia/Jakarta'))
        self._state: LifecycleState = LifecycleState.IDLE
        self._lifecycle_lock = asyncio.Lock()
        self._cleanup_results: List[CleanupResult] = []
        self._state_transition_history: deque = deque(maxlen=50)
        self._log_state_transition(None, LifecycleState.IDLE, "Scheduler initialized")
        logger.info(f"Task scheduler initialized with alert system (cleanup interval: {cleanup_interval}s, state: {self._state.value})")
    
    def _on_task_done(self, task: asyncio.Task, task_name: str) -> None:
        """Callback invoked when a tracked task completes.
        
        Handles exception draining and logging for background tasks.
        """
        try:
            if task.cancelled():
                logger.debug(f"Scheduled task {task_name} was cancelled")
                return
            
            try:
                exception = task.exception()
                if exception:
                    self._task_exceptions[task_name] = exception
                    logger.error(f"Scheduled task {task_name} raised exception: {exception}", exc_info=exception)
            except asyncio.InvalidStateError:
                pass
            except asyncio.CancelledError:
                pass
                
        except (ValueError, TypeError) as e:
            logger.error(f"Validation error in task done callback for {task_name}: {e}")
        except RuntimeError as e:
            logger.error(f"Runtime error in task done callback for {task_name}: {e}")
        except Exception as e:
            logger.error(f"Error in task done callback for {task_name}: {e}")
    
    def _log_state_transition(
        self,
        from_state: Optional[LifecycleState],
        to_state: LifecycleState,
        reason: str = ""
    ) -> None:
        """
        Log state transition untuk telemetry dan debugging.
        
        Args:
            from_state: State sebelumnya (None untuk initial)
            to_state: State tujuan
            reason: Alasan transisi
        """
        timestamp = datetime.now(pytz.timezone('Asia/Jakarta'))
        transition_record = {
            'timestamp': timestamp.isoformat(),
            'from_state': from_state.value if from_state else None,
            'to_state': to_state.value,
            'reason': reason
        }
        self._state_transition_history.append(transition_record)
        
        from_str = from_state.value if from_state else "INIT"
        logger.info(f"[LIFECYCLE] State transition: {from_str} -> {to_state.value} ({reason})")
    
    def _transition_state(self, to_state: LifecycleState, reason: str = "") -> None:
        """
        Transisi ke state baru dengan logging.
        
        Args:
            to_state: State tujuan
            reason: Alasan transisi
        """
        from_state = self._state
        self._state = to_state
        self._log_state_transition(from_state, to_state, reason)
    
    def get_lifecycle_state(self) -> LifecycleState:
        """Get current lifecycle state."""
        return self._state
    
    def get_state_transition_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get state transition history untuk telemetry.
        
        Args:
            limit: Maksimum jumlah record (None untuk semua)
            
        Returns:
            List of transition records
        """
        history = list(self._state_transition_history)
        if limit is not None:
            history = history[-limit:]
        return history
    
    def get_cleanup_results(self) -> List[Dict]:
        """Get structured cleanup results dari stop terakhir."""
        return [result.to_dict() for result in self._cleanup_results]
    
    async def reset_failed_stop(self) -> bool:
        """
        Reset dari FAILED_STOP state ke IDLE untuk recovery manual.
        
        Hanya bisa dipanggil jika current state adalah FAILED_STOP.
        Method ini akan membersihkan semua resources yang tersisa
        dan melakukan emergency cleanup.
        
        Returns:
            bool: True jika berhasil reset, False jika state bukan FAILED_STOP
        """
        async with self._lifecycle_lock:
            if self._state != LifecycleState.FAILED_STOP:
                logger.warning(
                    f"reset_failed_stop() dipanggil tapi state bukan FAILED_STOP "
                    f"(current: {self._state.value})"
                )
                return False
            
            logger.info("=" * 50)
            logger.info("RESET FAILED STOP - Manual Recovery")
            logger.info("=" * 50)
            
            try:
                self.running = False
                self._shutdown_flag.set()
                
                await self._emergency_stop_cleanup()
                
                self._shutdown_flag.clear()
                self._completion_event.clear()
                self._cleanup_results.clear()
                
                gc.collect()
                
                self._transition_state(LifecycleState.IDLE, "Manual reset from FAILED_STOP")
                
                logger.info("✅ Reset berhasil, scheduler siap untuk start")
                return True
                
            except Exception as e:
                logger.error(f"Error saat reset_failed_stop: {e}")
                return False
    
    def can_start(self) -> Tuple[bool, str]:
        """
        Check apakah scheduler bisa di-start.
        
        Returns:
            Tuple (can_start, reason)
        """
        if self._state == LifecycleState.IDLE:
            return True, "Ready to start"
        elif self._state == LifecycleState.RUNNING:
            return False, "Scheduler sudah berjalan"
        elif self._state == LifecycleState.STARTING:
            return False, "Scheduler sedang dalam proses start"
        elif self._state == LifecycleState.STOPPING:
            return False, "Scheduler sedang dalam proses stop"
        elif self._state == LifecycleState.FAILED_STOP:
            return False, "Stop sebelumnya gagal, panggil reset_failed_stop() terlebih dahulu"
        else:
            return False, f"Unknown state: {self._state.value}"
    
    def can_stop(self) -> Tuple[bool, str]:
        """
        Check apakah scheduler bisa di-stop.
        
        Returns:
            Tuple (can_stop, reason)
        """
        if self._state == LifecycleState.RUNNING:
            return True, "Ready to stop"
        elif self._state == LifecycleState.IDLE:
            return False, "Scheduler belum berjalan"
        elif self._state == LifecycleState.STARTING:
            return False, "Scheduler sedang dalam proses start"
        elif self._state == LifecycleState.STOPPING:
            return False, "Scheduler sudah dalam proses stop"
        elif self._state == LifecycleState.FAILED_STOP:
            return False, "Scheduler dalam state FAILED_STOP"
        else:
            return False, f"Unknown state: {self._state.value}"
    
    def add_task(self, name: str, func: Callable, interval: Optional[int] = None,
                schedule_time: Optional[time] = None, timezone: str = 'Asia/Jakarta',
                priority: TaskPriority = TaskPriority.NORMAL):
        if name in self.tasks:
            logger.warning(f"Task {name} already exists, replacing")
        
        task = ScheduledTask(name, func, interval, schedule_time, timezone, priority=priority)
        self.tasks[name] = task
        
        logger.info(f"Task added: {name} (interval={interval}s, schedule={schedule_time}, priority={priority.name})")
        return task
    
    def add_interval_task(self, name: str, func: Callable, interval_seconds: int,
                         timezone: str = 'Asia/Jakarta', priority: TaskPriority = TaskPriority.NORMAL):
        return self.add_task(name, func, interval=interval_seconds, timezone=timezone, priority=priority)
    
    def add_daily_task(self, name: str, func: Callable, hour: int, minute: int = 0,
                      timezone: str = 'Asia/Jakarta', priority: TaskPriority = TaskPriority.NORMAL):
        schedule_time = time(hour=hour, minute=minute)
        return self.add_task(name, func, schedule_time=schedule_time, timezone=timezone, priority=priority)
    
    def add_critical_task(self, name: str, func: Callable, interval_seconds: int,
                         timezone: str = 'Asia/Jakarta'):
        """Add a critical priority task that runs immediately when scheduled."""
        return self.add_task(name, func, interval=interval_seconds, timezone=timezone, 
                           priority=TaskPriority.CRITICAL)
    
    def set_task_priority(self, name: str, priority: TaskPriority) -> bool:
        """Set the priority of an existing task.
        
        Args:
            name: Name of the task
            priority: New priority level
            
        Returns:
            bool: True if task was found and updated, False otherwise
        """
        if name in self.tasks:
            self.tasks[name].set_priority(priority)
            return True
        logger.warning(f"Task not found for priority change: {name}")
        return False
    
    def remove_task(self, name: str) -> bool:
        if name in self.tasks:
            del self.tasks[name]
            logger.info(f"Task removed: {name}")
            return True
        
        logger.warning(f"Task not found: {name}")
        return False
    
    def enable_task(self, name: str) -> bool:
        if name in self.tasks:
            self.tasks[name].enable()
            return True
        return False
    
    def disable_task(self, name: str) -> bool:
        if name in self.tasks:
            self.tasks[name].disable()
            return True
        return False
    
    def get_task(self, name: str) -> Optional[ScheduledTask]:
        return self.tasks.get(name)
    
    def get_all_tasks(self) -> List[ScheduledTask]:
        return list(self.tasks.values())
    
    def get_pending_exceptions(self) -> Dict[str, BaseException]:
        """Get all exceptions from completed tasks.
        
        Returns:
            Dict mapping task names to their exceptions
        """
        return self._task_exceptions.copy()
    
    def clear_exceptions(self) -> None:
        """Clear all stored exceptions."""
        self._task_exceptions.clear()
    
    def record_exception(self, task_name: str, exception: Exception) -> None:
        """Record an exception with timestamp in scheduler-level history."""
        record = ExceptionRecord(
            exception=exception,
            timestamp=datetime.now(pytz.timezone('Asia/Jakarta'))
        )
        self._exception_history.append((task_name, record))
        self._task_exceptions[task_name] = exception
    
    def get_all_exception_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Get exception history across all tasks.
        
        Args:
            limit: Maximum number of recent exceptions to return
            
        Returns:
            List of exception records with task names
        """
        history = list(self._exception_history)
        if limit is not None:
            history = history[-limit:]
        return [
            {
                'task_name': task_name,
                **record.to_dict()
            }
            for task_name, record in history
        ]
    
    def get_task_exception_history(self, task_name: str, limit: Optional[int] = None) -> List[Dict]:
        """Get exception history for a specific task.
        
        Args:
            task_name: Name of the task
            limit: Maximum number of recent exceptions to return
            
        Returns:
            List of exception records for the task
        """
        task = self.tasks.get(task_name)
        if task:
            return task.get_exception_history(limit)
        return []
    
    async def cleanup_completed_tasks(self) -> int:
        """
        Cleanup completed tasks dari _active_task_executions dengan robust exception handling.
        
        Menggunakan timeout untuk acquire lock dan try-except-finally untuk memastikan
        cleanup selalu complete meskipun ada exception.
        
        Returns:
            Jumlah task yang dibersihkan
        """
        cleaned = 0
        lock_acquired = False
        
        try:
            # Acquire lock dengan timeout untuk mencegah deadlock
            lock_acquired = await _acquire_lock_with_timeout(self._lock, LOCK_ACQUIRE_TIMEOUT)
            
            if not lock_acquired:
                logger.warning("Gagal acquire lock untuk cleanup_completed_tasks, mencoba fallback")
                # Fallback: lakukan cleanup tanpa lock (less safe tapi lebih baik daripada stuck)
                return await self._fallback_cleanup_completed_tasks()
            
            try:
                # Cleanup active task executions
                completed_tasks = [t for t in self._active_task_executions if t.done()]
                for task in completed_tasks:
                    try:
                        self._active_task_executions.discard(task)
                        cleaned += 1
                    except Exception as e:
                        logger.error(f"Error saat discard task dari active executions: {e}")
                
                # Cleanup tracked tasks
                completed_tracked = [t for t in self._all_created_tasks if t.done()]
                for task in completed_tracked:
                    try:
                        self._all_created_tasks.discard(task)
                    except Exception as e:
                        logger.error(f"Error saat discard task dari tracked tasks: {e}")
                        
            except Exception as e:
                logger.error(f"Error selama proses cleanup completed tasks: {e}")
                
        except asyncio.CancelledError:
            logger.info("cleanup_completed_tasks dibatalkan")
            raise
        except Exception as e:
            logger.error(f"Error tidak terduga di cleanup_completed_tasks: {e}")
        finally:
            # Pastikan lock selalu di-release
            if lock_acquired:
                try:
                    self._lock.release()
                except RuntimeError:
                    # Lock sudah di-release atau tidak locked
                    pass
                except Exception as e:
                    logger.error(f"Error saat release lock: {e}")
        
        if cleaned > 0:
            logger.debug(f"Membersihkan {cleaned} completed tasks dari active executions")
            self._cleanup_stats['completed_tasks_cleaned'] += cleaned
        
        return cleaned
    
    async def _fallback_cleanup_completed_tasks(self) -> int:
        """
        Fallback cleanup untuk completed tasks tanpa lock.
        Digunakan jika acquire lock gagal.
        """
        cleaned = 0
        try:
            # Buat salinan untuk iterasi aman
            tasks_to_check = list(self._active_task_executions)
            for task in tasks_to_check:
                try:
                    if task.done():
                        self._active_task_executions.discard(task)
                        cleaned += 1
                except Exception:
                    pass
            
            tracked_to_check = list(self._all_created_tasks)
            for task in tracked_to_check:
                try:
                    if task.done():
                        self._all_created_tasks.discard(task)
                except Exception:
                    pass
                    
        except Exception as e:
            logger.error(f"Error di fallback cleanup: {e}")
        
        if cleaned > 0:
            logger.debug(f"Fallback cleanup: membersihkan {cleaned} tasks")
        
        return cleaned
    
    async def cleanup_orphaned_tasks(self) -> int:
        """
        Cleanup orphaned asyncio tasks yang tidak memiliki scheduled task terkait.
        
        Menggunakan try-except-finally dan timeout untuk memastikan cleanup
        selalu complete bahkan jika ada exception.
        
        Returns:
            Jumlah orphaned tasks yang dibersihkan
        """
        cleaned = 0
        lock_acquired = False
        orphaned = []
        
        try:
            valid_task_names = set(self.tasks.keys())
            
            # Acquire lock dengan timeout
            lock_acquired = await _acquire_lock_with_timeout(self._lock, LOCK_ACQUIRE_TIMEOUT)
            
            if not lock_acquired:
                logger.warning("Gagal acquire lock untuk cleanup_orphaned_tasks, mencoba fallback")
                return await self._fallback_cleanup_orphaned_tasks()
            
            try:
                # Identifikasi orphaned tasks
                for task in list(self._all_created_tasks):
                    try:
                        task_name_match = None
                        if hasattr(task, 'get_name'):
                            name = task.get_name()
                            for vtn in valid_task_names:
                                if vtn in name:
                                    task_name_match = vtn
                                    break
                        
                        if task_name_match is None and not task.done():
                            orphaned.append(task)
                    except Exception as e:
                        logger.error(f"Error saat memeriksa task untuk orphan detection: {e}")
                
                # Cancel dan cleanup orphaned tasks
                for task in orphaned:
                    try:
                        task.cancel()
                        cleaned += 1
                        self._all_created_tasks.discard(task)
                    except Exception as e:
                        logger.error(f"Error saat cancel/cleanup orphaned task: {e}")
                        # Tetap coba discard meski cancel gagal
                        try:
                            self._all_created_tasks.discard(task)
                        except Exception:
                            pass
                            
            except Exception as e:
                logger.error(f"Error selama proses cleanup orphaned tasks: {e}")
                
        except asyncio.CancelledError:
            logger.info("cleanup_orphaned_tasks dibatalkan")
            raise
        except Exception as e:
            logger.error(f"Error tidak terduga di cleanup_orphaned_tasks: {e}")
        finally:
            # Pastikan lock selalu di-release
            if lock_acquired:
                try:
                    self._lock.release()
                except RuntimeError:
                    pass
                except Exception as e:
                    logger.error(f"Error saat release lock di cleanup_orphaned_tasks: {e}")
        
        if cleaned > 0:
            logger.warning(f"Membersihkan {cleaned} orphaned tasks")
            self._cleanup_stats['orphaned_tasks_cleaned'] += cleaned
        
        return cleaned
    
    async def _fallback_cleanup_orphaned_tasks(self) -> int:
        """
        Fallback cleanup untuk orphaned tasks tanpa lock.
        Digunakan jika acquire lock gagal.
        """
        cleaned = 0
        try:
            valid_task_names = set(self.tasks.keys())
            tasks_to_check = list(self._all_created_tasks)
            
            for task in tasks_to_check:
                try:
                    if task.done():
                        continue
                    
                    task_name_match = None
                    if hasattr(task, 'get_name'):
                        name = task.get_name()
                        for vtn in valid_task_names:
                            if vtn in name:
                                task_name_match = vtn
                                break
                    
                    if task_name_match is None:
                        task.cancel()
                        self._all_created_tasks.discard(task)
                        cleaned += 1
                except Exception:
                    pass
                    
        except Exception as e:
            logger.error(f"Error di fallback cleanup orphaned: {e}")
        
        if cleaned > 0:
            logger.warning(f"Fallback cleanup: membersihkan {cleaned} orphaned tasks")
        
        return cleaned
    
    def detect_stale_tasks(self, threshold_minutes: int = STALE_TASK_THRESHOLD_MINUTES) -> List[str]:
        """Detect tasks that are stale (enabled but haven't run for extended period).
        
        Args:
            threshold_minutes: Minutes without running to be considered stale
            
        Returns:
            List of stale task names
        """
        stale_tasks = []
        for task_name, task in self.tasks.items():
            if task.is_stale(threshold_minutes):
                stale_tasks.append(task_name)
        
        if stale_tasks:
            logger.warning(f"Detected {len(stale_tasks)} stale tasks: {stale_tasks}")
            self._cleanup_stats['stale_tasks_detected'] = len(stale_tasks)
        
        return stale_tasks
    
    def detect_high_failure_tasks(self, threshold: int = 5) -> List[str]:
        """Detect tasks with high consecutive failure rates.
        
        Args:
            threshold: Number of consecutive failures to be considered high
            
        Returns:
            List of task names with high failure rates
        """
        failing_tasks = []
        for task_name, task in self.tasks.items():
            if task.has_high_failure_rate(threshold):
                failing_tasks.append(task_name)
        
        if failing_tasks:
            logger.warning(f"Detected {len(failing_tasks)} tasks with high failure rates: {failing_tasks}")
        
        return failing_tasks
    
    def get_auto_disabled_tasks(self) -> List[str]:
        """Get list of tasks that have been auto-disabled."""
        return [
            task_name for task_name, task in self.tasks.items()
            if task._auto_disabled
        ]
    
    def get_all_health_metrics(self) -> Dict[str, Dict]:
        """Get health metrics for all tasks.
        
        Returns:
            Dict mapping task names to their health metrics
        """
        return {
            task_name: task.get_health_metrics()
            for task_name, task in self.tasks.items()
        }
    
    def get_scheduler_health_summary(self) -> Dict:
        """Get overall scheduler health summary."""
        tasks = list(self.tasks.values())
        enabled_tasks = [t for t in tasks if t.enabled]
        executing_tasks = [t for t in tasks if t._is_executing]
        stale_tasks = [t for t in tasks if t.is_stale()]
        failing_tasks = [t for t in tasks if t.has_high_failure_rate()]
        auto_disabled = [t for t in tasks if t._auto_disabled]
        
        total_runs = sum(t.run_count for t in tasks)
        total_errors = sum(t.error_count for t in tasks)
        overall_success_rate = ((total_runs - total_errors) / total_runs * 100) if total_runs > 0 else 0
        
        return {
            'running': self.running,
            'lifecycle_state': self._state.value,
            'total_tasks': len(tasks),
            'enabled_tasks': len(enabled_tasks),
            'executing_tasks': len(executing_tasks),
            'stale_tasks': len(stale_tasks),
            'failing_tasks': len(failing_tasks),
            'auto_disabled_tasks': len(auto_disabled),
            'active_executions': len(self._active_task_executions),
            'tracked_tasks': len(self._all_created_tasks),
            'total_runs': total_runs,
            'total_errors': total_errors,
            'overall_success_rate': round(overall_success_rate, 2),
            'exception_history_count': len(self._exception_history),
            'cleanup_stats': self._cleanup_stats.copy()
        }
    
    def record_task_completion(self, task_name: str, success: bool, 
                               execution_time_ms: float = 0.0,
                               error_message: Optional[str] = None) -> None:
        """
        Record a completed task execution in history.
        
        Args:
            task_name: Name of the completed task
            success: Whether the task completed successfully
            execution_time_ms: Execution time in milliseconds
            error_message: Error message if task failed
        """
        try:
            record = CompletedTaskRecord(
                task_name=task_name,
                completion_time=datetime.now(pytz.timezone('Asia/Jakarta')),
                success=success,
                execution_time_ms=execution_time_ms,
                error_message=error_message
            )
            self._completed_tasks_history.append(record)
            
            # Also update the task's completion time
            if task_name in self.tasks:
                self.tasks[task_name].mark_completed()
                
        except Exception as e:
            logger.error(f"Error recording task completion for {task_name}: {e}")
    
    async def _cleanup_completed_tasks_history(self) -> int:
        """
        Cleanup completed task history based on age and count limits.
        
        Removes:
        - Tasks older than COMPLETED_TASK_MAX_AGE_MINUTES (30 minutes)
        - Tasks beyond MAX_COMPLETED_TASK_HISTORY (100)
        
        Returns:
            Number of entries cleaned
        """
        cleaned = 0
        now = datetime.now(pytz.timezone('Asia/Jakarta'))
        
        try:
            # Phase 1: Remove tasks older than 30 minutes
            old_size = len(self._completed_tasks_history)
            temp_history = []
            
            for record in self._completed_tasks_history:
                try:
                    age_minutes = record.age_minutes(now)
                    if age_minutes <= COMPLETED_TASK_MAX_AGE_MINUTES:
                        temp_history.append(record)
                    else:
                        cleaned += 1
                except Exception as e:
                    logger.debug(f"Error checking record age: {e}")
                    temp_history.append(record)  # Keep if can't check
            
            # Phase 2: Keep only last MAX_COMPLETED_TASK_HISTORY entries
            if len(temp_history) > MAX_COMPLETED_TASK_HISTORY:
                excess = len(temp_history) - MAX_COMPLETED_TASK_HISTORY
                temp_history = temp_history[-MAX_COMPLETED_TASK_HISTORY:]
                cleaned += excess
            
            # Update history
            self._completed_tasks_history.clear()
            for record in temp_history:
                self._completed_tasks_history.append(record)
            
            if cleaned > 0:
                logger.info(f"History cleanup: removed {cleaned} old entries (kept {len(self._completed_tasks_history)})")
                self._cleanup_stats['history_entries_cleaned'] += cleaned
            
            self._cleanup_stats['history_cleanups_run'] += 1
            self._last_history_cleanup = now
            
        except Exception as e:
            logger.error(f"Error in _cleanup_completed_tasks_history: {e}")
        
        return cleaned
    
    async def _history_cleanup_loop(self) -> None:
        """
        Background loop for periodic history cleanup (every 5 minutes).
        """
        logger.info(f"History cleanup loop started (interval: {HISTORY_CLEANUP_INTERVAL_SECONDS}s)")
        
        try:
            while self.running and not self._shutdown_flag.is_set():
                try:
                    # Wait for shutdown or timeout
                    try:
                        await asyncio.wait_for(
                            self._shutdown_flag.wait(),
                            timeout=float(HISTORY_CLEANUP_INTERVAL_SECONDS)
                        )
                        if self._shutdown_flag.is_set():
                            logger.info("Shutdown flag detected, exiting history cleanup loop")
                            break
                    except asyncio.TimeoutError:
                        pass
                    
                    # Run cleanup if not shutting down
                    if not self._shutdown_flag.is_set():
                        await self._cleanup_completed_tasks_history()
                        
                except asyncio.CancelledError:
                    logger.info("History cleanup loop cancelled")
                    raise
                except Exception as e:
                    logger.error(f"Error in history cleanup loop: {e}")
                    if not self._shutdown_flag.is_set():
                        await asyncio.sleep(5)
                        
        except asyncio.CancelledError:
            logger.info("History cleanup loop cancelled at top level")
        finally:
            logger.info("History cleanup loop finished")
    
    def get_scheduler_health(self) -> Dict:
        """
        Get scheduler health status including memory usage estimate.
        
        Returns:
            Dict containing:
            - active_tasks: Number of currently executing tasks
            - stale_tasks: Number of stale tasks
            - memory_usage_estimate_kb: Estimated memory usage in KB
            - last_cleanup_time: Timestamp of last cleanup
            - completed_history_size: Size of completed tasks history
            - exception_history_size: Size of exception history
            - lifecycle_state: Current lifecycle state
        """
        import sys
        
        tasks = list(self.tasks.values())
        active_tasks = len([t for t in tasks if t._is_executing])
        stale_tasks = len([t for t in tasks if t.is_stale()])
        
        # Estimate memory usage
        memory_estimate_bytes = 0
        try:
            # Estimate size of task objects
            memory_estimate_bytes += len(self.tasks) * 2048  # ~2KB per task
            
            # Estimate size of history
            memory_estimate_bytes += len(self._completed_tasks_history) * 256  # ~256B per record
            memory_estimate_bytes += len(self._exception_history) * 512  # ~512B per exception
            
            # Estimate size of active executions
            memory_estimate_bytes += len(self._active_task_executions) * 1024
            memory_estimate_bytes += len(self._all_created_tasks) * 1024
            
            # Add overhead for cleanup stats and other data
            memory_estimate_bytes += sys.getsizeof(self._cleanup_stats)
            memory_estimate_bytes += sys.getsizeof(self._task_exceptions)
            
        except Exception as e:
            logger.debug(f"Error estimating memory usage: {e}")
            memory_estimate_bytes = -1
        
        memory_estimate_kb = round(memory_estimate_bytes / 1024, 2) if memory_estimate_bytes >= 0 else -1
        
        return {
            'active_tasks': active_tasks,
            'stale_tasks': stale_tasks,
            'memory_usage_estimate_kb': memory_estimate_kb,
            'last_cleanup_time': self._last_cleanup.isoformat() if self._last_cleanup else None,
            'last_history_cleanup_time': self._last_history_cleanup.isoformat() if self._last_history_cleanup else None,
            'completed_history_size': len(self._completed_tasks_history),
            'exception_history_size': len(self._exception_history),
            'lifecycle_state': self._state.value,
            'total_tasks': len(self.tasks),
            'enabled_tasks': len([t for t in tasks if t.enabled]),
            'auto_disabled_tasks': len([t for t in tasks if t._auto_disabled]),
            'active_executions': len(self._active_task_executions),
            'tracked_tasks': len(self._all_created_tasks),
            'cleanup_stats': self._cleanup_stats.copy()
        }
    
    def recover_stale_tasks(self) -> List[str]:
        """
        Attempt to recover stale tasks by recalculating their next_run time.
        
        Returns:
            List of task names that were recovered
        """
        recovered = []
        stale_tasks = self.detect_stale_tasks()
        
        for task_name in stale_tasks:
            task = self.tasks.get(task_name)
            if task and task.enabled and not task._is_executing:
                try:
                    # Reset next_run to now to force immediate execution
                    task.next_run = datetime.now(task.timezone)
                    task._calculate_next_run()
                    recovered.append(task_name)
                    logger.info(f"Recovered stale task: {task_name}")
                except Exception as e:
                    logger.error(f"Error recovering stale task {task_name}: {e}")
        
        return recovered
    
    async def run_aggressive_cleanup(self) -> Dict:
        """
        Jalankan aggressive cleanup untuk semua resources terkait task.
        
        Menggunakan timeout untuk setiap operasi cleanup dan fallback mechanism
        jika cleanup utama gagal.
        
        Termasuk:
        - Cleanup completed tasks
        - Cleanup orphaned tasks
        - Auto-cancel stuck tasks (> TASK_AUTO_CANCEL_THRESHOLD detik)
        - Task rotation (setiap TASK_ROTATION_INTERVAL detik)
        - Cleanup by size (jika > COMPLETED_TASKS_MAX_SIZE entries)
        
        Returns:
            Dict dengan hasil cleanup
        """
        logger.info("Menjalankan aggressive cleanup...")
        
        completed_cleaned = 0
        orphaned_cleaned = 0
        stale_tasks = []
        failing_tasks = []
        cleanup_errors = []
        stuck_cancelled = 0
        size_cleaned = 0
        rotation_result = {}
        
        # Cleanup completed tasks dengan timeout dan fallback
        try:
            cleanup_success = await _safe_cleanup_with_timeout(
                cleanup_func=self.cleanup_completed_tasks,
                timeout=CLEANUP_TASK_TIMEOUT,
                fallback_func=self._fallback_cleanup_completed_tasks,
                operation_name="cleanup completed tasks"
            )
            if cleanup_success:
                completed_cleaned = self._cleanup_stats.get('completed_tasks_cleaned', 0)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            cleanup_errors.append(f"cleanup_completed: {e}")
            logger.error(f"Error saat cleanup completed tasks: {e}")
        
        # Cleanup orphaned tasks dengan timeout dan fallback
        try:
            cleanup_success = await _safe_cleanup_with_timeout(
                cleanup_func=self.cleanup_orphaned_tasks,
                timeout=CLEANUP_TASK_TIMEOUT,
                fallback_func=self._fallback_cleanup_orphaned_tasks,
                operation_name="cleanup orphaned tasks"
            )
            if cleanup_success:
                orphaned_cleaned = self._cleanup_stats.get('orphaned_tasks_cleaned', 0)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            cleanup_errors.append(f"cleanup_orphaned: {e}")
            logger.error(f"Error saat cleanup orphaned tasks: {e}")
        
        # Auto-cancel stuck tasks (Task Lifecycle Optimization)
        try:
            stuck_cancelled = await self.auto_cancel_stuck_tasks()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            cleanup_errors.append(f"auto_cancel_stuck: {e}")
            logger.error(f"Error saat auto-cancel stuck tasks: {e}")
        
        # Cleanup by size limit (Task Lifecycle Optimization)
        try:
            size_cleaned = await self.cleanup_completed_tasks_by_size()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            cleanup_errors.append(f"cleanup_by_size: {e}")
            logger.error(f"Error saat cleanup by size: {e}")
        
        # Task rotation (setiap TASK_ROTATION_INTERVAL detik)
        try:
            now = datetime.now(pytz.timezone('Asia/Jakarta'))
            time_since_rotation = (now - self._last_rotation).total_seconds()
            if time_since_rotation >= TASK_ROTATION_INTERVAL:
                rotation_result = await self.perform_task_rotation()
                self._last_rotation = now
                self._cleanup_stats['task_rotations'] = \
                    self._cleanup_stats.get('task_rotations', 0) + 1
        except asyncio.CancelledError:
            raise
        except Exception as e:
            cleanup_errors.append(f"task_rotation: {e}")
            logger.error(f"Error saat task rotation: {e}")
        
        # Detect stale dan failing tasks (operasi sync, tidak perlu timeout)
        try:
            stale_tasks = self.detect_stale_tasks()
        except Exception as e:
            cleanup_errors.append(f"detect_stale: {e}")
            logger.error(f"Error saat detect stale tasks: {e}")
        
        try:
            failing_tasks = self.detect_high_failure_tasks()
        except Exception as e:
            cleanup_errors.append(f"detect_failing: {e}")
            logger.error(f"Error saat detect failing tasks: {e}")
        
        # Garbage collection dengan protection
        try:
            gc.collect()
        except Exception as e:
            logger.warning(f"Error saat garbage collection: {e}")
        
        # Update stats
        try:
            now = datetime.now(pytz.timezone('Asia/Jakarta'))
            self._cleanup_stats['last_cleanup_time'] = now.isoformat()
            self._last_cleanup = now
        except Exception as e:
            logger.error(f"Error saat update cleanup stats: {e}")
            now = datetime.now(pytz.UTC)
        
        results = {
            'completed_tasks_cleaned': completed_cleaned,
            'orphaned_tasks_cleaned': orphaned_cleaned,
            'stuck_tasks_cancelled': stuck_cancelled,
            'size_entries_cleaned': size_cleaned,
            'task_rotation': rotation_result,
            'stale_tasks': stale_tasks,
            'failing_tasks': failing_tasks,
            'auto_disabled_tasks': self.get_auto_disabled_tasks(),
            'cleanup_errors': cleanup_errors,
            'timestamp': now.isoformat()
        }
        
        if cleanup_errors:
            logger.warning(f"Aggressive cleanup selesai dengan {len(cleanup_errors)} error: {results}")
        else:
            logger.info(f"Aggressive cleanup selesai: {results}")
        
        return results
    
    async def _cleanup_loop(self):
        """
        Background loop untuk periodic cleanup dengan robust exception handling.
        
        Loop ini akan terus berjalan meskipun ada exception di cleanup operations.
        Menggunakan fallback mechanism untuk memastikan cleanup selalu terjadi.
        """
        logger.info(f"Cleanup loop dimulai (interval: {self._cleanup_interval}s)")
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        try:
            while self.running and not self._shutdown_flag.is_set():
                try:
                    # Tunggu shutdown flag atau timeout
                    try:
                        await asyncio.wait_for(
                            self._shutdown_flag.wait(),
                            timeout=float(self._cleanup_interval)
                        )
                        if self._shutdown_flag.is_set():
                            logger.info("Shutdown flag terdeteksi, keluar dari cleanup loop")
                            break
                    except asyncio.TimeoutError:
                        # Timeout adalah kondisi normal, lanjutkan ke cleanup
                        pass
                    
                    # Jalankan cleanup jika tidak dalam shutdown
                    if not self._shutdown_flag.is_set():
                        try:
                            await asyncio.wait_for(
                                self.run_aggressive_cleanup(),
                                timeout=CLEANUP_TASK_TIMEOUT * 3  # Berikan waktu lebih untuk aggressive cleanup
                            )
                            consecutive_errors = 0  # Reset counter setelah sukses
                        except asyncio.TimeoutError:
                            consecutive_errors += 1
                            logger.warning(f"Aggressive cleanup timeout, consecutive errors: {consecutive_errors}")
                            await self._emergency_cleanup()
                        except asyncio.CancelledError:
                            logger.info("Aggressive cleanup dibatalkan")
                            raise
                        except RuntimeError as e:
                            consecutive_errors += 1
                            logger.error(f"Runtime error saat cleanup: {e}, consecutive errors: {consecutive_errors}")
                            await self._emergency_cleanup()
                        except Exception as e:
                            consecutive_errors += 1
                            logger.error(f"Error saat cleanup: {e}, consecutive errors: {consecutive_errors}")
                            await self._emergency_cleanup()
                        
                        # Jika terlalu banyak error berturut-turut, kurangi frekuensi cleanup
                        if consecutive_errors >= max_consecutive_errors:
                            logger.warning(f"Terlalu banyak cleanup errors ({consecutive_errors}), mengurangi frekuensi")
                            await asyncio.sleep(self._cleanup_interval * 2)  # Tambah delay
                            
                except asyncio.CancelledError:
                    logger.info("Cleanup loop iteration dibatalkan")
                    raise
                except Exception as e:
                    # Catch-all untuk exception yang tidak terduga
                    consecutive_errors += 1
                    logger.error(f"Exception tidak terduga di cleanup loop: {e}")
                    if not self._shutdown_flag.is_set():
                        await asyncio.sleep(5)  # Brief pause before retry
                        
        except asyncio.CancelledError:
            logger.info("Cleanup loop dibatalkan")
        except Exception as e:
            logger.error(f"Fatal error di cleanup loop: {e}")
        finally:
            # Final cleanup sebelum keluar
            try:
                await self._emergency_cleanup()
            except Exception as e:
                logger.error(f"Error saat final cleanup: {e}")
            logger.info("Cleanup loop selesai")
    
    async def _emergency_cleanup(self) -> None:
        """
        Emergency cleanup sederhana yang dijamin tidak akan stuck.
        Digunakan sebagai fallback jika cleanup utama gagal.
        """
        try:
            # Cleanup minimal tanpa lock
            tasks_to_check = list(self._active_task_executions)
            for task in tasks_to_check:
                try:
                    if task.done():
                        self._active_task_executions.discard(task)
                except Exception:
                    pass
            
            tracked_to_check = list(self._all_created_tasks)
            for task in tracked_to_check:
                try:
                    if task.done():
                        self._all_created_tasks.discard(task)
                except Exception:
                    pass
                    
            logger.debug("Emergency cleanup selesai")
        except Exception as e:
            logger.error(f"Error di emergency cleanup: {e}")
    
    def enable_task_with_reset(self, name: str) -> bool:
        """Enable a task and reset its failure counters.
        
        Args:
            name: Name of the task to enable
            
        Returns:
            bool: True if task was enabled, False if not found
        """
        if name in self.tasks:
            self.tasks[name].enable(reset_failures=True)
            return True
        return False
    
    async def auto_cancel_stuck_tasks(self) -> int:
        """
        Auto-cancel tasks yang stuck lebih dari TASK_AUTO_CANCEL_THRESHOLD detik.
        
        Task dianggap stuck jika sedang executing tapi tidak selesai dalam waktu threshold.
        
        Returns:
            Jumlah tasks yang di-cancel
        """
        cancelled_count = 0
        now = datetime.now(pytz.timezone('Asia/Jakarta'))
        threshold_seconds = TASK_AUTO_CANCEL_THRESHOLD
        
        try:
            for task_name, task in list(self.tasks.items()):
                try:
                    if not task._is_executing:
                        continue
                    
                    if task._execution_start_time is None:
                        continue
                    
                    execution_start = task._execution_start_time
                    if execution_start.tzinfo is None:
                        execution_start = pytz.timezone('Asia/Jakarta').localize(execution_start)
                    
                    elapsed_seconds = (now - execution_start).total_seconds()
                    
                    if elapsed_seconds > threshold_seconds:
                        logger.warning(
                            f"Task {task_name} stuck selama {elapsed_seconds:.1f}s "
                            f"(threshold: {threshold_seconds}s), melakukan auto-cancel"
                        )
                        
                        if task.current_execution_task and not task.current_execution_task.done():
                            task.current_execution_task.cancel()
                            cancelled_count += 1
                            
                            task._is_executing = False
                            task.error_count += 1
                            task.consecutive_failures += 1
                            task._calculate_next_run()
                            
                            self._cleanup_stats['auto_cancelled_stuck_tasks'] = \
                                self._cleanup_stats.get('auto_cancelled_stuck_tasks', 0) + 1
                            
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Error saat auto-cancel task {task_name}: {e}")
                    
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error di auto_cancel_stuck_tasks: {e}")
        
        if cancelled_count > 0:
            logger.info(f"Auto-cancelled {cancelled_count} stuck tasks")
        
        return cancelled_count
    
    async def perform_task_rotation(self) -> Dict[str, Any]:
        """
        Perform task rotation untuk memastikan task tidak stuck dalam state tertentu.
        
        Task rotation meliputi:
        - Reset task yang auto-disabled setelah interval tertentu
        - Re-enable task yang healthy tapi disabled
        - Refresh next_run untuk task yang stale
        
        Returns:
            Dict dengan hasil rotation
        """
        results = {
            'rotated_tasks': [],
            're_enabled_tasks': [],
            'refreshed_tasks': [],
            'errors': []
        }
        
        now = datetime.now(pytz.timezone('Asia/Jakarta'))
        
        try:
            for task_name, task in list(self.tasks.items()):
                try:
                    if task._auto_disabled:
                        if task.consecutive_failures < MAX_CONSECUTIVE_FAILURES_AUTO_DISABLE:
                            task._auto_disabled = False
                            task.enabled = True
                            task._calculate_next_run()
                            results['re_enabled_tasks'].append(task_name)
                            logger.info(f"Task {task_name} di-re-enable setelah rotation")
                    
                    if task.enabled and not task._is_executing:
                        if task.next_run and task.next_run < now - timedelta(seconds=TASK_ROTATION_INTERVAL):
                            old_next_run = task.next_run
                            task._calculate_next_run()
                            results['refreshed_tasks'].append({
                                'name': task_name,
                                'old_next_run': old_next_run.isoformat() if old_next_run else None,
                                'new_next_run': task.next_run.isoformat() if task.next_run else None
                            })
                            logger.debug(f"Task {task_name} next_run di-refresh: {old_next_run} -> {task.next_run}")
                    
                    results['rotated_tasks'].append(task_name)
                    
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    error_msg = f"Error rotating task {task_name}: {e}"
                    results['errors'].append(error_msg)
                    logger.error(error_msg)
                    
        except asyncio.CancelledError:
            raise
        except Exception as e:
            error_msg = f"Error di perform_task_rotation: {e}"
            results['errors'].append(error_msg)
            logger.error(error_msg)
        
        if results['re_enabled_tasks'] or results['refreshed_tasks']:
            logger.info(
                f"Task rotation selesai: {len(results['re_enabled_tasks'])} re-enabled, "
                f"{len(results['refreshed_tasks'])} refreshed"
            )
        
        return results
    
    def calculate_restart_backoff(self, consecutive_failures: int) -> float:
        """
        Hitung exponential backoff delay untuk restart task yang gagal.
        
        Formula: min(base * (2 ** failures), max_backoff)
        
        Args:
            consecutive_failures: Jumlah kegagalan berturut-turut
            
        Returns:
            Delay dalam detik untuk restart berikutnya
        """
        if consecutive_failures <= 0:
            return 0.0
        
        base = TASK_RESTART_BACKOFF_BASE
        max_backoff = TASK_RESTART_BACKOFF_MAX
        
        backoff = base * (2 ** (consecutive_failures - 1))
        
        return min(backoff, max_backoff)
    
    def get_task_restart_delay(self, task_name: str) -> float:
        """
        Dapatkan delay untuk restart task berdasarkan consecutive failures.
        
        Args:
            task_name: Nama task
            
        Returns:
            Delay dalam detik, 0 jika task tidak ada
        """
        task = self.tasks.get(task_name)
        if not task:
            return 0.0
        
        return self.calculate_restart_backoff(task.consecutive_failures)
    
    async def cleanup_completed_tasks_by_size(self) -> int:
        """
        Cleanup completed tasks jika jumlahnya melebihi COMPLETED_TASKS_MAX_SIZE.
        
        Method ini membatasi ukuran _task_exceptions dan _exception_history
        untuk mencegah memory leak.
        
        Returns:
            Jumlah entries yang dibersihkan
        """
        cleaned = 0
        max_size = COMPLETED_TASKS_MAX_SIZE
        
        try:
            if len(self._task_exceptions) > max_size:
                excess = len(self._task_exceptions) - max_size
                keys_to_remove = list(self._task_exceptions.keys())[:excess]
                for key in keys_to_remove:
                    try:
                        del self._task_exceptions[key]
                        cleaned += 1
                    except KeyError:
                        pass
                logger.info(f"Cleaned {excess} excess task exceptions (max: {max_size})")
            
            if len(self._exception_history) > max_size:
                excess = len(self._exception_history) - max_size
                for _ in range(excess):
                    try:
                        self._exception_history.popleft()
                        cleaned += 1
                    except IndexError:
                        break
                logger.info(f"Cleaned excess exception history entries (max: {max_size})")
            
            for task_name, task in list(self.tasks.items()):
                try:
                    if len(task._exception_history) > max_size:
                        excess = len(task._exception_history) - max_size
                        for _ in range(excess):
                            try:
                                task._exception_history.popleft()
                                cleaned += 1
                            except IndexError:
                                break
                except Exception as e:
                    logger.error(f"Error cleaning task {task_name} exception history: {e}")
                    
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error di cleanup_completed_tasks_by_size: {e}")
        
        if cleaned > 0:
            logger.debug(f"Cleaned {cleaned} entries by size limit ({max_size})")
            self._cleanup_stats['entries_cleaned_by_size'] = \
                self._cleanup_stats.get('entries_cleaned_by_size', 0) + cleaned
        
        return cleaned
    
    async def _track_task(self, task: asyncio.Task, task_name: str):
        """
        Track task dan handle completion dengan exception draining yang robust.
        
        Menggunakan lock dengan timeout untuk mencegah deadlock dan
        try-except-finally untuk memastikan cleanup selalu terjadi.
        """
        lock_acquired = False
        
        # Tambahkan task ke tracking dengan timeout lock
        try:
            lock_acquired = await _acquire_lock_with_timeout(self._lock, LOCK_ACQUIRE_TIMEOUT)
            if lock_acquired:
                try:
                    self._active_task_executions.add(task)
                    self._all_created_tasks.add(task)
                    self._completion_event.clear()
                finally:
                    try:
                        self._lock.release()
                        lock_acquired = False
                    except RuntimeError:
                        pass
            else:
                # Fallback tanpa lock
                logger.warning(f"Gagal acquire lock untuk track task {task_name}, menggunakan fallback")
                self._active_task_executions.add(task)
                self._all_created_tasks.add(task)
                self._completion_event.clear()
        except Exception as e:
            logger.error(f"Error saat menambahkan task {task_name} ke tracking: {e}")
        
        # Tunggu task selesai
        try:
            await task
        except asyncio.CancelledError:
            logger.debug(f"Tracked task {task_name} dibatalkan")
        except Exception as e:
            try:
                self._task_exceptions[task_name] = e
                self.record_exception(task_name, e)
            except Exception:
                pass
            logger.error(f"Tracked task {task_name} gagal: {e}")
        finally:
            # Cleanup tracking dengan timeout lock
            try:
                lock_acquired = await _acquire_lock_with_timeout(self._lock, LOCK_ACQUIRE_TIMEOUT)
                if lock_acquired:
                    try:
                        self._active_task_executions.discard(task)
                        self._all_created_tasks.discard(task)
                        if len(self._active_task_executions) == 0:
                            self._completion_event.set()
                    finally:
                        try:
                            self._lock.release()
                        except RuntimeError:
                            pass
                else:
                    # Fallback cleanup tanpa lock
                    self._active_task_executions.discard(task)
                    self._all_created_tasks.discard(task)
                    if len(self._active_task_executions) == 0:
                        self._completion_event.set()
            except Exception as e:
                logger.error(f"Error saat cleanup tracking untuk task {task_name}: {e}")
                # Emergency cleanup
                try:
                    self._active_task_executions.discard(task)
                    self._all_created_tasks.discard(task)
                except Exception:
                    pass
    
    async def flush_pending_tasks(self, timeout: float = FLUSH_PENDING_TIMEOUT) -> bool:
        """
        Tunggu semua pending task executions selesai dengan robust exception handling.
        
        Args:
            timeout: Waktu maksimum untuk menunggu dalam detik
            
        Returns:
            bool: True jika semua tasks selesai, False jika timeout terjadi
        """
        pending_count = 0
        
        # Check pending count dengan timeout lock
        try:
            lock_acquired = await _acquire_lock_with_timeout(self._lock, LOCK_ACQUIRE_TIMEOUT)
            if lock_acquired:
                try:
                    if not self._active_task_executions:
                        return True
                    pending_count = len(self._active_task_executions)
                finally:
                    try:
                        self._lock.release()
                    except RuntimeError:
                        pass
            else:
                # Fallback tanpa lock
                if not self._active_task_executions:
                    return True
                pending_count = len(self._active_task_executions)
        except Exception as e:
            logger.error(f"Error saat check pending tasks: {e}")
            pending_count = len(self._active_task_executions)
        
        logger.info(f"Flushing {pending_count} pending task executions (timeout={timeout}s)...")
        
        try:
            await asyncio.wait_for(self._completion_event.wait(), timeout=timeout)
            logger.info("Semua pending task executions selesai")
            return True
        except asyncio.TimeoutError:
            try:
                lock_acquired = await _acquire_lock_with_timeout(self._lock, LOCK_ACQUIRE_TIMEOUT)
                if lock_acquired:
                    try:
                        remaining = len(self._active_task_executions)
                    finally:
                        try:
                            self._lock.release()
                        except RuntimeError:
                            pass
                else:
                    remaining = len(self._active_task_executions)
            except Exception:
                remaining = len(self._active_task_executions)
            
            logger.warning(f"Flush timeout: {remaining} tasks masih pending setelah {timeout}s")
            return False
        except asyncio.CancelledError:
            logger.info("flush_pending_tasks dibatalkan")
            raise
        except Exception as e:
            logger.error(f"Error tidak terduga di flush_pending_tasks: {e}")
            return False
    
    async def _scheduler_loop(self):
        """
        Loop utama scheduler dengan robust exception handling.
        
        Loop ini akan terus berjalan dan menjadwalkan tasks meskipun
        ada exception di individual task executions.
        """
        logger.info("Scheduler loop dimulai")
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        try:
            while self.running and not self._shutdown_flag.is_set():
                try:
                    if self._shutdown_flag.is_set():
                        logger.info("Shutdown flag terdeteksi di scheduler loop")
                        break
                    
                    # Jalankan semua task yang siap dengan error handling per-task
                    for task in list(self.tasks.values()):
                        if self._shutdown_flag.is_set():
                            logger.info("Shutdown flag terdeteksi, menghentikan penjadwalan task")
                            break
                        
                        try:
                            if task.should_run():
                                def make_done_callback(task_name: str):
                                    def callback(t: asyncio.Task):
                                        try:
                                            self._on_task_done(t, task_name)
                                        except Exception as cb_err:
                                            logger.error(f"Error di done callback untuk {task_name}: {cb_err}")
                                    return callback
                                
                                try:
                                    execution_task = asyncio.create_task(
                                        task.execute(
                                            alert_system=self.alert_system,
                                            shutdown_flag=self._shutdown_flag,
                                            on_complete=lambda t, n: None
                                        )
                                    )
                                    execution_task.set_name(f"task_exec_{task.name}")
                                    execution_task.add_done_callback(make_done_callback(task.name))
                                    
                                    tracking_task = asyncio.create_task(
                                        self._track_task(execution_task, task.name)
                                    )
                                    tracking_task.set_name(f"track_{task.name}")
                                except Exception as e:
                                    logger.error(f"Error saat membuat task execution untuk {task.name}: {e}")
                        except Exception as e:
                            logger.error(f"Error saat memeriksa/menjalankan task {task.name}: {e}")
                            # Lanjutkan ke task berikutnya, jangan crash scheduler
                    
                    # Reset consecutive errors setelah iterasi sukses
                    consecutive_errors = 0
                    
                    # Tunggu shutdown flag atau timeout
                    try:
                        await asyncio.wait_for(
                            self._shutdown_flag.wait(),
                            timeout=1.0
                        )
                        if self._shutdown_flag.is_set():
                            logger.info("Shutdown flag di-set, keluar dari scheduler loop")
                            break
                    except asyncio.TimeoutError:
                        pass
                    
                except asyncio.CancelledError:
                    logger.info("Scheduler loop dibatalkan")
                    break
                except Exception as e:
                    consecutive_errors += 1
                    logger.error(f"Error di scheduler loop: {e} (consecutive errors: {consecutive_errors})")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        logger.critical(f"Terlalu banyak errors ({consecutive_errors}), mengurangi frekuensi scheduling")
                    
                    if not self._shutdown_flag.is_set():
                        # Exponential backoff untuk errors
                        sleep_time = min(5 * consecutive_errors, 60)
                        await asyncio.sleep(sleep_time)
                        
        except asyncio.CancelledError:
            logger.info("Scheduler loop menerima cancellation di top level")
        except Exception as e:
            logger.critical(f"Fatal error di scheduler loop: {e}")
        finally:
            logger.info("Scheduler loop selesai")
    
    async def start(self):
        """
        Mulai task scheduler dengan scheduler loop, cleanup loop, and history cleanup loop.
        
        Menggunakan lifecycle lock untuk serialize start/stop transitions
        dan mencegah race condition dengan concurrent stop.
        
        State Machine Transition:
        - IDLE -> STARTING -> RUNNING (sukses)
        - IDLE -> STARTING -> IDLE (gagal)
        
        Transition Guard:
        - start() hanya boleh dari state IDLE
        """
        async with self._lifecycle_lock:
            can_start, reason = self.can_start()
            if not can_start:
                logger.warning(f"Cannot start scheduler: {reason} (state: {self._state.value})")
                return
            
            self._transition_state(LifecycleState.STARTING, "start() called")
            
            try:
                self._shutdown_flag.clear()
                self._completion_event.clear()
                self._cleanup_results.clear()
                self.running = True
                
                self.scheduler_task = asyncio.create_task(self._scheduler_loop())
                self.scheduler_task.set_name("scheduler_loop")
                
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())
                self._cleanup_task.set_name("cleanup_loop")
                
                # Start history cleanup loop (runs every 5 minutes)
                self._history_cleanup_task = asyncio.create_task(self._history_cleanup_loop())
                self._history_cleanup_task.set_name("history_cleanup_loop")
                
                self._transition_state(LifecycleState.RUNNING, "Scheduler started successfully")
                logger.info("Task scheduler dimulai dengan cleanup loop dan history cleanup loop")
                
            except Exception as e:
                logger.error(f"Error saat memulai scheduler: {e}")
                self.running = False
                self._transition_state(LifecycleState.IDLE, f"Start failed: {e}")
                raise
    
    async def stop(self, graceful_timeout: float = SCHEDULER_STOP_TIMEOUT,
                   drain_timeout: float = DEFAULT_DRAIN_TIMEOUT):
        """
        Stop scheduler dengan graceful shutdown dan robust exception handling.
        
        Menggunakan lifecycle lock untuk serialize start/stop transitions.
        Lock di-hold sampai semua cleanup awaits selesai.
        
        State Machine Transition:
        - RUNNING -> STOPPING -> IDLE (sukses)
        - RUNNING -> STOPPING -> FAILED_STOP (gagal, perlu reset manual)
        
        Transition Guard:
        - stop() hanya boleh dari state RUNNING
        
        Args:
            graceful_timeout: Waktu maksimum untuk menunggu graceful shutdown loop
            drain_timeout: Waktu maksimum untuk menunggu running tasks selesai sebelum cancel (default 30s)
        """
        async with self._lifecycle_lock:
            can_stop, reason = self.can_stop()
            if not can_stop:
                logger.debug(f"Cannot stop scheduler: {reason} (state: {self._state.value})")
                return
            
            self._transition_state(LifecycleState.STOPPING, "stop() called")
            self._cleanup_results.clear()
            
            cleanup_failed = False
            
            active_count = len(self._active_task_executions)
            tracked_count = len(self._all_created_tasks)
            
            logger.info("=" * 50)
            logger.info("MENGHENTIKAN TASK SCHEDULER")
            logger.info(f"Active task executions: {active_count}")
            logger.info(f"Total tracked tasks: {tracked_count}")
            logger.info(f"Drain timeout: {drain_timeout}s")
            logger.info("=" * 50)
            
            self._shutdown_flag.set()
            self.running = False
            
            logger.info("Shutdown flag di-set, menunggu scheduler loop berhenti...")
            
            # Stop history cleanup loop first
            result = await self._stop_history_cleanup_loop_with_result()
            self._cleanup_results.append(result)
            if not result.success:
                cleanup_failed = True
            
            result = await self._stop_cleanup_loop_with_result()
            self._cleanup_results.append(result)
            if not result.success:
                cleanup_failed = True
            
            result = await self._stop_scheduler_loop_with_result()
            self._cleanup_results.append(result)
            if not result.success:
                cleanup_failed = True
            
            # Drain running tasks - wait for them to complete naturally
            if active_count > 0:
                result = await self._drain_running_tasks_with_result(drain_timeout)
                self._cleanup_results.append(result)
                if not result.success:
                    logger.warning("Some tasks did not complete during drain, proceeding with cancellation")
            
            result = await self._cancel_individual_tasks_with_result()
            self._cleanup_results.append(result)
            if not result.success:
                cleanup_failed = True
            
            result = await self._cancel_active_executions_with_result(graceful_timeout)
            self._cleanup_results.append(result)
            if not result.success:
                cleanup_failed = True
            
            result = await self._final_cleanup_with_result()
            self._cleanup_results.append(result)
            if not result.success:
                cleanup_failed = True
                await self._emergency_stop_cleanup()
            
            self._log_stop_summary()
            
            if cleanup_failed:
                self._transition_state(
                    LifecycleState.FAILED_STOP,
                    "Stop completed with cleanup failures - call reset_failed_stop() to recover"
                )
                logger.warning("⚠️ Stop selesai dengan error cleanup - panggil reset_failed_stop() untuk recovery")
            else:
                self._transition_state(LifecycleState.IDLE, "Stop completed successfully")
                logger.info("✅ Scheduler berhasil dihentikan")
    
    async def _stop_history_cleanup_loop_with_result(self) -> CleanupResult:
        """Stop history cleanup loop with structured result tracking."""
        start_time = datetime.now(pytz.timezone('Asia/Jakarta'))
        
        if not self._history_cleanup_task or self._history_cleanup_task.done():
            return CleanupResult(
                success=True,
                operation="stop_history_cleanup_loop",
                items_cleaned=0,
                duration_ms=0.0
            )
        
        try:
            logger.info("Stopping history cleanup loop...")
            self._history_cleanup_task.cancel()
            await asyncio.wait_for(self._history_cleanup_task, timeout=TASK_CANCEL_TIMEOUT)
            logger.info("✅ History cleanup loop stopped gracefully")
            
            duration_ms = (datetime.now(pytz.timezone('Asia/Jakarta')) - start_time).total_seconds() * 1000
            return CleanupResult(
                success=True,
                operation="stop_history_cleanup_loop",
                items_cleaned=1,
                duration_ms=duration_ms
            )
        except asyncio.TimeoutError:
            logger.warning(f"History cleanup loop cancellation timed out after {TASK_CANCEL_TIMEOUT}s")
            duration_ms = (datetime.now(pytz.timezone('Asia/Jakarta')) - start_time).total_seconds() * 1000
            return CleanupResult(
                success=False,
                operation="stop_history_cleanup_loop",
                error_message=f"Timeout after {TASK_CANCEL_TIMEOUT}s",
                duration_ms=duration_ms
            )
        except asyncio.CancelledError:
            logger.info("✅ History cleanup loop cancelled")
            duration_ms = (datetime.now(pytz.timezone('Asia/Jakarta')) - start_time).total_seconds() * 1000
            return CleanupResult(
                success=True,
                operation="stop_history_cleanup_loop",
                items_cleaned=1,
                duration_ms=duration_ms
            )
        except Exception as e:
            logger.error(f"Error stopping history cleanup loop: {e}")
            duration_ms = (datetime.now(pytz.timezone('Asia/Jakarta')) - start_time).total_seconds() * 1000
            return CleanupResult(
                success=False,
                operation="stop_history_cleanup_loop",
                error_message=str(e),
                duration_ms=duration_ms
            )
    
    async def _drain_running_tasks_with_result(self, drain_timeout: float) -> CleanupResult:
        """
        Wait for running tasks to complete naturally before cancellation.
        
        Args:
            drain_timeout: Maximum time to wait for tasks to complete
            
        Returns:
            CleanupResult with drain status
        """
        start_time = datetime.now(pytz.timezone('Asia/Jakarta'))
        
        try:
            active_count = len(self._active_task_executions)
            if active_count == 0:
                return CleanupResult(
                    success=True,
                    operation="drain_running_tasks",
                    items_cleaned=0,
                    duration_ms=0.0
                )
            
            logger.info(f"Draining {active_count} running tasks (timeout: {drain_timeout}s)...")
            
            completed_count = 0
            drain_start = datetime.now(pytz.timezone('Asia/Jakarta'))
            
            # Wait for tasks to complete, checking periodically
            check_interval = 0.5  # Check every 500ms
            elapsed = 0.0
            
            while elapsed < drain_timeout:
                # Count completed tasks
                still_running = [t for t in self._active_task_executions if not t.done()]
                completed_count = active_count - len(still_running)
                
                if not still_running:
                    logger.info(f"✅ All {completed_count} tasks completed during drain")
                    break
                
                # Log progress periodically
                if int(elapsed) % 5 == 0 and elapsed > 0:
                    logger.info(f"Drain progress: {completed_count}/{active_count} completed, {len(still_running)} still running")
                
                await asyncio.sleep(check_interval)
                elapsed = (datetime.now(pytz.timezone('Asia/Jakarta')) - drain_start).total_seconds()
            
            # Final count
            still_running_final = [t for t in self._active_task_executions if not t.done()]
            completed_count = active_count - len(still_running_final)
            
            duration_ms = (datetime.now(pytz.timezone('Asia/Jakarta')) - start_time).total_seconds() * 1000
            
            if still_running_final:
                logger.warning(f"Drain completed: {completed_count} finished, {len(still_running_final)} timed out")
                return CleanupResult(
                    success=False,
                    operation="drain_running_tasks",
                    items_cleaned=completed_count,
                    error_message=f"{len(still_running_final)} tasks did not complete",
                    duration_ms=duration_ms
                )
            else:
                return CleanupResult(
                    success=True,
                    operation="drain_running_tasks",
                    items_cleaned=completed_count,
                    duration_ms=duration_ms
                )
                
        except asyncio.CancelledError:
            duration_ms = (datetime.now(pytz.timezone('Asia/Jakarta')) - start_time).total_seconds() * 1000
            return CleanupResult(
                success=False,
                operation="drain_running_tasks",
                error_message="Drain cancelled",
                duration_ms=duration_ms
            )
        except Exception as e:
            logger.error(f"Error during task drain: {e}")
            duration_ms = (datetime.now(pytz.timezone('Asia/Jakarta')) - start_time).total_seconds() * 1000
            return CleanupResult(
                success=False,
                operation="drain_running_tasks",
                error_message=str(e),
                duration_ms=duration_ms
            )
    
    async def _stop_cleanup_loop(self) -> None:
        """Hentikan cleanup loop dengan timeout protection."""
        if self._cleanup_task and not self._cleanup_task.done():
            logger.info("Membatalkan cleanup loop task...")
            self._cleanup_task.cancel()
            try:
                await asyncio.wait_for(self._cleanup_task, timeout=TASK_CANCEL_TIMEOUT)
                logger.info("✅ Cleanup loop berhenti dengan graceful")
            except asyncio.TimeoutError:
                logger.warning(f"Pembatalan cleanup loop timeout setelah {TASK_CANCEL_TIMEOUT}s")
            except asyncio.CancelledError:
                logger.info("✅ Cleanup loop dibatalkan")
            except Exception as e:
                logger.error(f"Error saat menghentikan cleanup loop: {e}")
    
    async def _stop_cleanup_loop_with_result(self) -> CleanupResult:
        """Hentikan cleanup loop dengan structured result tracking."""
        start_time = datetime.now(pytz.timezone('Asia/Jakarta'))
        
        if not self._cleanup_task or self._cleanup_task.done():
            return CleanupResult(
                success=True,
                operation="stop_cleanup_loop",
                items_cleaned=0,
                duration_ms=0.0
            )
        
        try:
            logger.info("Membatalkan cleanup loop task...")
            self._cleanup_task.cancel()
            await asyncio.wait_for(self._cleanup_task, timeout=TASK_CANCEL_TIMEOUT)
            logger.info("✅ Cleanup loop berhenti dengan graceful")
            
            duration_ms = (datetime.now(pytz.timezone('Asia/Jakarta')) - start_time).total_seconds() * 1000
            return CleanupResult(
                success=True,
                operation="stop_cleanup_loop",
                items_cleaned=1,
                duration_ms=duration_ms
            )
        except asyncio.TimeoutError:
            logger.warning(f"Pembatalan cleanup loop timeout setelah {TASK_CANCEL_TIMEOUT}s")
            duration_ms = (datetime.now(pytz.timezone('Asia/Jakarta')) - start_time).total_seconds() * 1000
            return CleanupResult(
                success=False,
                operation="stop_cleanup_loop",
                error_message=f"Timeout setelah {TASK_CANCEL_TIMEOUT}s",
                duration_ms=duration_ms
            )
        except asyncio.CancelledError:
            logger.info("✅ Cleanup loop dibatalkan")
            duration_ms = (datetime.now(pytz.timezone('Asia/Jakarta')) - start_time).total_seconds() * 1000
            return CleanupResult(
                success=True,
                operation="stop_cleanup_loop",
                items_cleaned=1,
                duration_ms=duration_ms
            )
        except Exception as e:
            logger.error(f"Error saat menghentikan cleanup loop: {e}")
            duration_ms = (datetime.now(pytz.timezone('Asia/Jakarta')) - start_time).total_seconds() * 1000
            return CleanupResult(
                success=False,
                operation="stop_cleanup_loop",
                error_message=str(e),
                duration_ms=duration_ms
            )
    
    async def _stop_scheduler_loop_with_result(self) -> CleanupResult:
        """Hentikan scheduler loop dengan structured result tracking."""
        start_time = datetime.now(pytz.timezone('Asia/Jakarta'))
        
        if not self.scheduler_task or self.scheduler_task.done():
            return CleanupResult(
                success=True,
                operation="stop_scheduler_loop",
                items_cleaned=0,
                duration_ms=0.0
            )
        
        try:
            logger.info("Membatalkan scheduler loop task...")
            self.scheduler_task.cancel()
            await asyncio.wait_for(self.scheduler_task, timeout=TASK_CANCEL_TIMEOUT)
            logger.info("✅ Scheduler loop berhenti dengan graceful")
            
            duration_ms = (datetime.now(pytz.timezone('Asia/Jakarta')) - start_time).total_seconds() * 1000
            return CleanupResult(
                success=True,
                operation="stop_scheduler_loop",
                items_cleaned=1,
                duration_ms=duration_ms
            )
        except asyncio.TimeoutError:
            logger.warning(f"Pembatalan scheduler loop timeout setelah {TASK_CANCEL_TIMEOUT}s")
            duration_ms = (datetime.now(pytz.timezone('Asia/Jakarta')) - start_time).total_seconds() * 1000
            return CleanupResult(
                success=False,
                operation="stop_scheduler_loop",
                error_message=f"Timeout setelah {TASK_CANCEL_TIMEOUT}s",
                duration_ms=duration_ms
            )
        except asyncio.CancelledError:
            logger.info("✅ Scheduler loop dibatalkan")
            duration_ms = (datetime.now(pytz.timezone('Asia/Jakarta')) - start_time).total_seconds() * 1000
            return CleanupResult(
                success=True,
                operation="stop_scheduler_loop",
                items_cleaned=1,
                duration_ms=duration_ms
            )
        except Exception as e:
            logger.error(f"Error saat menghentikan scheduler loop: {e}")
            duration_ms = (datetime.now(pytz.timezone('Asia/Jakarta')) - start_time).total_seconds() * 1000
            return CleanupResult(
                success=False,
                operation="stop_scheduler_loop",
                error_message=str(e),
                duration_ms=duration_ms
            )
    
    async def _cancel_individual_tasks_with_result(self) -> CleanupResult:
        """Cancel individual task executions dengan structured result tracking."""
        start_time = datetime.now(pytz.timezone('Asia/Jakarta'))
        cancelled_count = 0
        failed_count = 0
        errors = []
        
        logger.info("Membatalkan individual task executions...")
        
        for task_name, task in list(self.tasks.items()):
            if task._is_executing:
                try:
                    logger.info(f"Membatalkan running task: {task_name}")
                    success = await asyncio.wait_for(
                        task.cancel_execution(timeout=TASK_CANCEL_TIMEOUT),
                        timeout=TASK_CANCEL_TIMEOUT + 1
                    )
                    if success:
                        cancelled_count += 1
                    else:
                        failed_count += 1
                        errors.append(f"{task_name}: cancellation returned false")
                except asyncio.TimeoutError:
                    failed_count += 1
                    errors.append(f"{task_name}: timeout")
                except Exception as e:
                    failed_count += 1
                    errors.append(f"{task_name}: {e}")
        
        logger.info(f"Membatalkan {cancelled_count} individual task executions")
        duration_ms = (datetime.now(pytz.timezone('Asia/Jakarta')) - start_time).total_seconds() * 1000
        
        return CleanupResult(
            success=failed_count == 0,
            operation="cancel_individual_tasks",
            items_cleaned=cancelled_count,
            error_message="; ".join(errors) if errors else None,
            duration_ms=duration_ms
        )
    
    async def _cancel_active_executions_with_result(self, graceful_timeout: float) -> CleanupResult:
        """Cancel active task executions dengan structured result tracking."""
        start_time = datetime.now(pytz.timezone('Asia/Jakarta'))
        
        lock_acquired = await _acquire_lock_with_timeout(self._lock, LOCK_ACQUIRE_TIMEOUT)
        
        if lock_acquired:
            try:
                active_tasks = list(self._active_task_executions)
            finally:
                try:
                    self._lock.release()
                except RuntimeError:
                    pass
        else:
            logger.warning("Gagal acquire lock, menggunakan fallback untuk active executions")
            active_tasks = list(self._active_task_executions)
        
        active_count = len(active_tasks)
        if active_count == 0:
            logger.info("Tidak ada active task executions untuk dibatalkan")
            return CleanupResult(
                success=True,
                operation="cancel_active_executions",
                items_cleaned=0,
                duration_ms=0.0
            )
        
        logger.info(f"Membatalkan {active_count} remaining active task executions...")
        
        for task_exec in active_tasks:
            try:
                if not task_exec.done():
                    task_exec.cancel()
            except Exception as e:
                logger.error(f"Error saat cancel task execution: {e}")
        
        try:
            done, pending = await asyncio.wait(
                active_tasks,
                timeout=graceful_timeout,
                return_when=asyncio.ALL_COMPLETED
            )
            
            completed = len(done)
            still_pending = len(pending)
            
            if still_pending > 0:
                logger.warning(f"{still_pending} task executions tidak selesai dalam timeout")
                for p in pending:
                    try:
                        p.cancel()
                    except Exception:
                        pass
            else:
                logger.info(f"✅ Semua {completed} task executions selesai")
            
            duration_ms = (datetime.now(pytz.timezone('Asia/Jakarta')) - start_time).total_seconds() * 1000
            return CleanupResult(
                success=still_pending == 0,
                operation="cancel_active_executions",
                items_cleaned=completed,
                error_message=f"{still_pending} tasks still pending" if still_pending > 0 else None,
                duration_ms=duration_ms
            )
        except Exception as e:
            logger.error(f"Error saat menunggu task executions: {e}")
            duration_ms = (datetime.now(pytz.timezone('Asia/Jakarta')) - start_time).total_seconds() * 1000
            return CleanupResult(
                success=False,
                operation="cancel_active_executions",
                error_message=str(e),
                duration_ms=duration_ms
            )
    
    async def _final_cleanup_with_result(self) -> CleanupResult:
        """Final cleanup dengan structured result tracking."""
        start_time = datetime.now(pytz.timezone('Asia/Jakarta'))
        items_cleaned = 0
        
        lock_acquired = await _acquire_lock_with_timeout(self._lock, LOCK_ACQUIRE_TIMEOUT)
        
        try:
            if lock_acquired:
                remaining = len(self._all_created_tasks)
                if remaining > 0:
                    logger.info(f"Membersihkan {remaining} remaining tracked tasks...")
                    for task in list(self._all_created_tasks):
                        try:
                            if not task.done():
                                task.cancel()
                                items_cleaned += 1
                        except Exception:
                            pass
                    self._all_created_tasks.clear()
                
                self._active_task_executions.clear()
            else:
                logger.warning("Final cleanup tanpa lock (fallback)")
                for task in list(self._all_created_tasks):
                    try:
                        if not task.done():
                            task.cancel()
                            items_cleaned += 1
                    except Exception:
                        pass
                self._all_created_tasks.clear()
                self._active_task_executions.clear()
            
            self._completion_event.set()
            
            duration_ms = (datetime.now(pytz.timezone('Asia/Jakarta')) - start_time).total_seconds() * 1000
            return CleanupResult(
                success=True,
                operation="final_cleanup",
                items_cleaned=items_cleaned,
                duration_ms=duration_ms
            )
        except Exception as e:
            logger.error(f"Error saat final cleanup: {e}")
            duration_ms = (datetime.now(pytz.timezone('Asia/Jakarta')) - start_time).total_seconds() * 1000
            return CleanupResult(
                success=False,
                operation="final_cleanup",
                items_cleaned=items_cleaned,
                error_message=str(e),
                duration_ms=duration_ms
            )
        finally:
            if lock_acquired:
                try:
                    self._lock.release()
                except RuntimeError:
                    pass
    
    async def _stop_scheduler_loop(self) -> None:
        """Hentikan scheduler loop dengan timeout protection."""
        if self.scheduler_task and not self.scheduler_task.done():
            logger.info("Membatalkan scheduler loop task...")
            self.scheduler_task.cancel()
            try:
                await asyncio.wait_for(self.scheduler_task, timeout=TASK_CANCEL_TIMEOUT)
                logger.info("✅ Scheduler loop berhenti dengan graceful")
            except asyncio.TimeoutError:
                logger.warning(f"Pembatalan scheduler loop timeout setelah {TASK_CANCEL_TIMEOUT}s")
            except asyncio.CancelledError:
                logger.info("✅ Scheduler loop dibatalkan")
            except Exception as e:
                logger.error(f"Error saat menghentikan scheduler loop: {e}")
    
    async def _cancel_individual_tasks(self) -> None:
        """Cancel semua individual task executions dengan error handling per-task."""
        logger.info("Membatalkan individual task executions...")
        cancelled_count = 0
        
        for task_name, task in list(self.tasks.items()):
            if task._is_executing:
                try:
                    logger.info(f"Membatalkan running task: {task_name}")
                    success = await asyncio.wait_for(
                        task.cancel_execution(timeout=TASK_CANCEL_TIMEOUT),
                        timeout=TASK_CANCEL_TIMEOUT + 1  # Sedikit lebih lama
                    )
                    if success:
                        cancelled_count += 1
                    else:
                        logger.warning(f"Gagal membatalkan task: {task_name}")
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout saat membatalkan task: {task_name}")
                except Exception as e:
                    logger.error(f"Error saat membatalkan task {task_name}: {e}")
        
        logger.info(f"Membatalkan {cancelled_count} individual task executions")
    
    async def _cancel_active_executions(self, graceful_timeout: float) -> None:
        """Cancel active task executions dengan timeout dan fallback."""
        # Gunakan lock dengan timeout
        lock_acquired = await _acquire_lock_with_timeout(self._lock, LOCK_ACQUIRE_TIMEOUT)
        
        if lock_acquired:
            try:
                active_tasks = list(self._active_task_executions)
            finally:
                try:
                    self._lock.release()
                except RuntimeError:
                    pass
        else:
            # Fallback tanpa lock
            logger.warning("Gagal acquire lock, menggunakan fallback untuk active executions")
            active_tasks = list(self._active_task_executions)
        
        active_count = len(active_tasks)
        if active_count > 0:
            logger.info(f"Membatalkan {active_count} remaining active task executions...")
            
            # Cancel semua task
            for task_exec in active_tasks:
                try:
                    if not task_exec.done():
                        task_exec.cancel()
                except Exception as e:
                    logger.error(f"Error saat cancel task execution: {e}")
            
            # Tunggu completion dengan timeout
            if active_tasks:
                try:
                    done, pending = await asyncio.wait(
                        active_tasks,
                        timeout=graceful_timeout,
                        return_when=asyncio.ALL_COMPLETED
                    )
                    
                    completed = len(done)
                    still_pending = len(pending)
                    
                    if still_pending > 0:
                        logger.warning(f"{still_pending} task executions tidak selesai dalam timeout")
                        # Force cancel pending tasks
                        for p in pending:
                            try:
                                p.cancel()
                            except Exception:
                                pass
                    else:
                        logger.info(f"✅ Semua {completed} task executions selesai")
                except Exception as e:
                    logger.error(f"Error saat menunggu task executions: {e}")
        else:
            logger.info("Tidak ada active task executions untuk dibatalkan")
    
    async def _final_cleanup(self) -> None:
        """Final cleanup dengan lock dan fallback."""
        lock_acquired = await _acquire_lock_with_timeout(self._lock, LOCK_ACQUIRE_TIMEOUT)
        
        try:
            if lock_acquired:
                remaining = len(self._all_created_tasks)
                if remaining > 0:
                    logger.info(f"Membersihkan {remaining} remaining tracked tasks...")
                    for task in list(self._all_created_tasks):
                        try:
                            if not task.done():
                                task.cancel()
                        except Exception:
                            pass
                    self._all_created_tasks.clear()
                
                self._active_task_executions.clear()
            else:
                # Fallback tanpa lock
                logger.warning("Final cleanup tanpa lock (fallback)")
                for task in list(self._all_created_tasks):
                    try:
                        if not task.done():
                            task.cancel()
                    except Exception:
                        pass
                self._all_created_tasks.clear()
                self._active_task_executions.clear()
        finally:
            if lock_acquired:
                try:
                    self._lock.release()
                except RuntimeError:
                    pass
        
        self._completion_event.set()
    
    async def _emergency_stop_cleanup(self) -> None:
        """Emergency cleanup yang dijamin tidak akan stuck."""
        try:
            # Clear semua tanpa lock
            for task in list(self._all_created_tasks):
                try:
                    if not task.done():
                        task.cancel()
                except Exception:
                    pass
            
            self._all_created_tasks.clear()
            self._active_task_executions.clear()
            self._completion_event.set()
            logger.info("Emergency stop cleanup selesai")
        except Exception as e:
            logger.error(f"Error di emergency stop cleanup: {e}")
    
    def _log_stop_summary(self) -> None:
        """Log summary setelah stop."""
        exceptions_count = len(self._task_exceptions)
        if exceptions_count > 0:
            logger.warning(f"{exceptions_count} tasks memiliki exceptions selama sesi ini")
            for task_name, exc in self._task_exceptions.items():
                logger.debug(f"  - {task_name}: {exc}")
        
        logger.info("=" * 50)
        logger.info("TASK SCHEDULER BERHASIL DIHENTIKAN")
        logger.info(f"Status akhir - Running: {self.running}, Shutdown flag: {self._shutdown_flag.is_set()}")
        logger.info("=" * 50)
    
    def get_status(self) -> Dict:
        tasks = list(self.tasks.values())
        stale_tasks = [t for t in tasks if t.is_stale()]
        failing_tasks = [t for t in tasks if t.has_high_failure_rate()]
        auto_disabled = [t for t in tasks if t._auto_disabled]
        
        can_start, can_start_reason = self.can_start()
        can_stop, can_stop_reason = self.can_stop()
        
        return {
            'running': self.running,
            'shutting_down': self._shutdown_flag.is_set(),
            'lifecycle_state': self._state.value,
            'can_start': can_start,
            'can_start_reason': can_start_reason,
            'can_stop': can_stop,
            'can_stop_reason': can_stop_reason,
            'total_tasks': len(self.tasks),
            'enabled_tasks': len([t for t in tasks if t.enabled]),
            'executing_tasks': len([t for t in tasks if t._is_executing]),
            'stale_tasks': len(stale_tasks),
            'failing_tasks': len(failing_tasks),
            'auto_disabled_tasks': len(auto_disabled),
            'active_executions': len(self._active_task_executions),
            'total_tracked_tasks': len(self._all_created_tasks),
            'pending_exceptions': len(self._task_exceptions),
            'exception_history_count': len(self._exception_history),
            'state_transition_history_count': len(self._state_transition_history),
            'cleanup_results': self.get_cleanup_results(),
            'cleanup_stats': self._cleanup_stats.copy(),
            'tasks': {name: task.to_dict() for name, task in self.tasks.items()}
        }
    
    def format_task_list(self) -> str:
        if not self.tasks:
            return "Tidak ada task yang dijadwalkan"
        
        msg = "📅 *Scheduled Tasks*\n\n"
        
        for task in self.tasks.values():
            if task._auto_disabled:
                status_icon = '🚫'
            elif task.enabled:
                status_icon = '✅'
            else:
                status_icon = '⛔'
            
            exec_icon = '🔄' if task._is_executing else ''
            stale_icon = '⏰' if task.is_stale() else ''
            fail_icon = '⚠️' if task.has_high_failure_rate() else ''
            
            msg += f"{status_icon}{exec_icon}{stale_icon}{fail_icon} *{task.name}*\n"
            
            if task.interval:
                msg += f"Interval: {task.interval}s\n"
            elif task.schedule_time:
                msg += f"Scheduled: {task.schedule_time.strftime('%H:%M')}\n"
            
            if task.last_run:
                msg += f"Last Run: {task.last_run.strftime('%Y-%m-%d %H:%M')}\n"
            
            if task.next_run:
                msg += f"Next Run: {task.next_run.strftime('%Y-%m-%d %H:%M')}\n"
            
            success_rate = task._health_metrics.success_rate
            msg += f"Runs: {task.run_count} | Errors: {task.error_count} | Success: {success_rate:.1f}%\n"
            
            if task.consecutive_failures > 0:
                msg += f"Consecutive Failures: {task.consecutive_failures}\n"
            
            if task._auto_disabled:
                msg += "⚠️ AUTO-DISABLED due to failures\n"
            
            msg += "\n"
        
        return msg

def setup_default_tasks(scheduler: TaskScheduler, bot_components: Dict):
    logger.info("Setting up default scheduled tasks")
    
    async def cleanup_old_charts():
        chart_generator = bot_components.get('chart_generator')
        if chart_generator:
            chart_generator.cleanup_old_charts(days=7)
    
    async def send_daily_summary():
        alert_system = bot_components.get('alert_system')
        if alert_system:
            await alert_system.send_daily_summary()
    
    async def cleanup_database():
        db_manager = bot_components.get('db_manager')
        if db_manager:
            session = db_manager.get_session()
            try:
                from bot.database import Trade
                from datetime import datetime, timedelta
                
                cutoff = datetime.utcnow() - timedelta(days=90)
                old_trades = session.query(Trade).filter(
                    Trade.signal_time < cutoff,
                    Trade.status == 'CLOSED'
                ).delete()
                
                session.commit()
                logger.info(f"Deleted {old_trades} old trade records")
            except (ValueError, TypeError) as e:
                logger.error(f"Validation error cleaning database: {e}")
                try:
                    session.rollback()
                except RuntimeError as rollback_error:
                    logger.error(f"Runtime error during rollback: {rollback_error}")
                except Exception as rollback_error:
                    logger.error(f"Error during rollback: {rollback_error}")
            except RuntimeError as e:
                logger.error(f"Runtime error cleaning database: {e}")
                try:
                    session.rollback()
                except RuntimeError as rollback_error:
                    logger.error(f"Runtime error during rollback: {rollback_error}")
                except Exception as rollback_error:
                    logger.error(f"Error during rollback: {rollback_error}")
            except Exception as e:
                logger.error(f"Error cleaning database: {e}")
                try:
                    session.rollback()
                except RuntimeError as rollback_error:
                    logger.error(f"Runtime error during rollback: {rollback_error}")
                except Exception as rollback_error:
                    logger.error(f"Error during rollback: {rollback_error}")
            finally:
                try:
                    session.close()
                except RuntimeError as close_error:
                    logger.error(f"Runtime error closing session: {close_error}")
                except Exception as close_error:
                    logger.error(f"Error closing session: {close_error}")
    
    async def health_check():
        logger.info("Running health check...")
        market_data = bot_components.get('market_data')
        if market_data:
            price = await market_data.get_current_price()
            if price:
                logger.info(f"Health check OK - Current price: {price}")
            else:
                logger.warning("Health check: No price data available")
    
    def is_market_closed_utc() -> bool:
        """
        Cek apakah market tutup berdasarkan waktu UTC.
        Market tutup: Sabtu 00:00 UTC - Minggu 21:00 UTC
        
        Returns:
            bool: True jika market tutup, False jika buka
        """
        now_utc = datetime.now(pytz.UTC)
        weekday = now_utc.weekday()  # 0=Monday, 5=Saturday, 6=Sunday
        hour = now_utc.hour
        
        # Saturday (weekday=5): Market tutup sepanjang hari
        if weekday == 5:
            return True
        
        # Sunday (weekday=6): Market tutup sampai 21:00 UTC
        if weekday == 6 and hour < 21:
            return True
        
        return False
    
    async def monitor_positions():
        position_tracker = bot_components.get('position_tracker')
        market_data = bot_components.get('market_data')
        
        # Early exit 1: Cek market status - skip jika market tutup
        if is_market_closed_utc():
            logger.debug("Position monitoring: Skip - market tutup (Sabtu 00:00 UTC - Minggu 21:00 UTC)")
            return
        
        # Early exit 2: Cek WebSocket connection dan data availability
        if market_data:
            if not market_data.is_connected():
                has_data = market_data.current_bid is not None and market_data.current_ask is not None
                if not has_data:
                    logger.debug("Position monitoring: Skip - WebSocket tidak terhubung dan tidak ada data")
                    return
        
        # Early exit 3: Tidak ada position_tracker
        if not position_tracker:
            logger.debug("Position monitoring: Skip - position_tracker tidak tersedia")
            return
        
        # Early exit 4: Tidak ada active positions
        total_active = sum(len(positions) for positions in position_tracker.active_positions.values())
        if total_active == 0:
            logger.debug("Position monitoring: Skip - tidak ada active positions")
            return
        
        # Proses monitoring
        updated = await position_tracker.monitor_active_positions()
        if updated:
            logger.info(f"Position monitoring: {len(updated)} positions updated")
        else:
            logger.debug("Position monitoring: Tidak ada perubahan")
    
    async def periodic_gc():
        gc.collect()
        logger.debug("Periodic garbage collection completed")
    
    async def save_candles_periodic():
        market_data = bot_components.get('market_data')
        db_manager = bot_components.get('db_manager')
        if market_data and db_manager:
            try:
                await market_data.save_candles_to_db(db_manager)
                market_data._prune_old_candles(db_manager, keep_count=150)
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError as e:
                logger.error(f"Timeout in periodic candle save: {e}")
            except (ValueError, TypeError) as e:
                logger.error(f"Validation error in periodic candle save: {e}")
            except RuntimeError as e:
                logger.error(f"Runtime error in periodic candle save: {e}")
            except Exception as e:
                logger.error(f"Error in periodic candle save: {e}")
    
    scheduler.add_interval_task(
        'cleanup_charts',
        cleanup_old_charts,
        interval_seconds=1800
    )
    
    async def aggressive_chart_cleanup():
        chart_generator = bot_components.get('chart_generator')
        if chart_generator:
            import os
            chart_dir = chart_generator.chart_dir
            max_charts = 10
            max_age_minutes = 30
            
            try:
                if os.path.exists(chart_dir):
                    import time
                    current_time = time.time()
                    files = []
                    
                    for f in os.listdir(chart_dir):
                        if f.endswith('.png'):
                            file_path = os.path.join(chart_dir, f)
                            file_age_minutes = (current_time - os.path.getmtime(file_path)) / 60
                            files.append((f, file_path, file_age_minutes))
                    
                    files.sort(key=lambda x: x[2], reverse=True)
                    
                    deleted_count = 0
                    for f, file_path, age in files:
                        if age > max_age_minutes or len(files) - deleted_count > max_charts:
                            try:
                                os.remove(file_path)
                                deleted_count += 1
                                logger.debug(f"Deleted chart: {f} (age: {age:.1f}min)")
                            except OSError as e:
                                logger.warning(f"OS error deleting chart {f}: {e}")
                            except Exception as e:
                                logger.warning(f"Failed to delete chart {f}: {e}")
                    
                    if deleted_count > 0:
                        logger.info(f"Aggressive cleanup: removed {deleted_count} old charts")
            except OSError as e:
                logger.error(f"OS error in aggressive chart cleanup: {e}")
            except Exception as e:
                logger.error(f"Error in aggressive chart cleanup: {e}")
    
    scheduler.add_interval_task(
        'aggressive_chart_cleanup',
        aggressive_chart_cleanup,
        interval_seconds=300
    )
    
    scheduler.add_daily_task(
        'daily_summary',
        send_daily_summary,
        hour=23,
        minute=55
    )
    
    scheduler.add_daily_task(
        'database_cleanup',
        cleanup_database,
        hour=3,
        minute=0
    )
    
    scheduler.add_interval_task(
        'health_check',
        health_check,
        interval_seconds=300
    )
    
    position_check_interval = 5 if Config.FREE_TIER_MODE else 10
    scheduler.add_interval_task(
        'position_monitoring',
        monitor_positions,
        interval_seconds=position_check_interval
    )
    logger.info(f"Position monitoring interval: {position_check_interval}s (FREE_TIER_MODE={Config.FREE_TIER_MODE})")
    
    scheduler.add_interval_task(
        'periodic_gc',
        periodic_gc,
        interval_seconds=300
    )
    
    scheduler.add_interval_task(
        'save_candles',
        save_candles_periodic,
        interval_seconds=60
    )
    
    # TASK 1: Register stale_position_cleanup - check dan auto-close posisi stale
    async def stale_position_cleanup():
        """Check dan auto-close posisi yang tidak aktif > 10 menit."""
        position_tracker = bot_components.get('position_tracker')
        if not position_tracker:
            logger.debug("Stale cleanup: Skip - position_tracker tidak tersedia")
            return
        
        try:
            closed = await position_tracker.check_stale_positions()
            if closed:
                logger.info(f"🧹 Stale cleanup: {len(closed)} positions closed")
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError as e:
            logger.error(f"Stale cleanup timeout: {e}")
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error(f"Stale cleanup validation error: {e}")
        except Exception as e:
            logger.error(f"Stale cleanup error: {e}")
    
    scheduler.add_interval_task(
        'stale_position_cleanup',
        stale_position_cleanup,
        interval_seconds=60  # Check every 60 seconds
    )
    logger.info("Stale position cleanup task registered (interval: 60s)")
    
    logger.info(f"Configured {len(scheduler.tasks)} default scheduled tasks")
