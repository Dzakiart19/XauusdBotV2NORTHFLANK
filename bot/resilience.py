"""
Resilience patterns: Circuit Breaker, Rate Limiter, Retry, dan Exponential Backoff

All exceptions logged before re-raise atau return:
- Semua exception di-log dengan level ERROR sebelum re-raise atau return default
- Retry attempts di-log dengan backoff duration
- Rate limiter sleep interruption di-log

Thread-safe: Semua patterns dapat digunakan dari multiple threads/coroutines
"""
import asyncio
import time
from typing import Optional, Callable, Any, TypeVar, Union
from enum import Enum
from collections import deque
from datetime import datetime, timedelta
import logging

logger = logging.getLogger('Resilience')

T = TypeVar('T')

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 60.0
DEFAULT_LOCK_TIMEOUT = 5.0


class ResilienceError(Exception):
    """Base exception for resilience-related errors"""
    pass


class RetryExhaustedError(ResilienceError):
    """Raised when all retry attempts are exhausted"""
    def __init__(self, attempts: int, last_exception: Optional[Exception] = None):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(f"Retry exhausted after {attempts} attempts: {last_exception}")


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    def __init__(self, name: str, retry_in: float):
        self.name = name
        self.retry_in = retry_in
        super().__init__(f"CircuitBreaker '{name}' is OPEN. Retry in {retry_in:.1f}s")


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """Circuit breaker pattern for preventing cascading failures
    
    Tracks failures and opens circuit after threshold is exceeded.
    Automatically attempts recovery after cooldown period.
    
    Features:
    - Half-open state requires consecutive successes before fully closing
    - Tracks state change statistics (open_count, half_open_count)
    - Auto-reset option after configurable timeout
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
        name: str = "CircuitBreaker",
        success_threshold: int = 3,
        auto_reset: bool = False,
        auto_reset_timeout: float = 300.0
    ):
        """Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to catch
            name: Name for logging
            success_threshold: Consecutive successes needed in half-open to close (default: 3)
            auto_reset: Enable automatic reset after timeout (default: False)
            auto_reset_timeout: Seconds before auto-reset when enabled (default: 300 = 5 minutes)
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name
        self.success_threshold = success_threshold
        self.auto_reset = auto_reset
        self.auto_reset_timeout = auto_reset_timeout
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED
        
        self.open_count = 0
        self.half_open_count = 0
        self.last_state_change: Optional[float] = None
        self._state_history: list = []
        
        logger.info(
            f"CircuitBreaker '{name}' initialized: "
            f"threshold={failure_threshold}, timeout={recovery_timeout}s, "
            f"success_threshold={success_threshold}, auto_reset={auto_reset}"
        )
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        self._check_auto_reset()
        
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                logger.info(f"CircuitBreaker '{self.name}': Attempting recovery (HALF_OPEN)")
                self._set_state(CircuitState.HALF_OPEN)
                self.success_count = 0
            else:
                remaining = self.recovery_timeout - (time.time() - (self.last_failure_time or 0))
                raise CircuitBreakerOpenException(self.name, remaining)
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except CircuitBreakerOpenException:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.warning(f"CircuitBreaker '{self.name}': Network error: {e}")
            if isinstance(e, self.expected_exception):
                self._on_failure()
            raise
        except Exception as e:
            logger.error(f"CircuitBreaker '{self.name}': Unexpected error type {type(e).__name__}: {e}")
            if isinstance(e, self.expected_exception):
                self._on_failure()
            raise
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection
        
        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        self._check_auto_reset()
        
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                logger.info(f"CircuitBreaker '{self.name}': Attempting recovery (HALF_OPEN)")
                self._set_state(CircuitState.HALF_OPEN)
                self.success_count = 0
            else:
                remaining = self.recovery_timeout - (time.time() - (self.last_failure_time or 0))
                raise CircuitBreakerOpenException(self.name, remaining)
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except CircuitBreakerOpenException:
            raise
        except (ResilienceError, Exception) as e:
            if isinstance(e, self.expected_exception):
                self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery"""
        if self.last_failure_time is None:
            return False
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _check_auto_reset(self) -> bool:
        """Check if auto-reset should be triggered
        
        Returns:
            True if circuit was auto-reset, False otherwise
        """
        if not self.auto_reset:
            return False
        
        if self.state != CircuitState.OPEN:
            return False
        
        if self.last_failure_time is None:
            return False
        
        elapsed = time.time() - self.last_failure_time
        if elapsed >= self.auto_reset_timeout:
            logger.info(
                f"CircuitBreaker '{self.name}': Auto-reset triggered after "
                f"{elapsed:.1f}s (timeout={self.auto_reset_timeout}s)"
            )
            self.reset()
            return True
        
        return False
    
    def _record_state_change(self, old_state: CircuitState, new_state: CircuitState):
        """Record state transition for statistics
        
        Args:
            old_state: Previous circuit state
            new_state: New circuit state
        """
        self.last_state_change = time.time()
        
        if new_state == CircuitState.OPEN:
            self.open_count += 1
        elif new_state == CircuitState.HALF_OPEN:
            self.half_open_count += 1
        
        self._state_history.append({
            'timestamp': self.last_state_change,
            'from_state': old_state.value,
            'to_state': new_state.value,
        })
        
        if len(self._state_history) > 100:
            self._state_history = self._state_history[-100:]
    
    def _set_state(self, new_state: CircuitState):
        """Set new state with tracking
        
        Args:
            new_state: New circuit state
        """
        if new_state != self.state:
            old_state = self.state
            self.state = new_state
            self._record_state_change(old_state, new_state)
    
    def _on_success(self):
        """Handle successful execution with success threshold for half-open state"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            logger.debug(
                f"CircuitBreaker '{self.name}': Half-open success "
                f"{self.success_count}/{self.success_threshold}"
            )
            
            if self.success_count >= self.success_threshold:
                logger.info(
                    f"CircuitBreaker '{self.name}': Recovery successful after "
                    f"{self.success_threshold} consecutive successes (CLOSED)"
                )
                self._set_state(CircuitState.CLOSED)
                self.failure_count = 0
                self.success_count = 0
        elif self.state == CircuitState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)
    
    def _on_failure(self):
        """Handle failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.success_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            logger.warning(
                f"CircuitBreaker '{self.name}': Recovery failed (OPEN again)"
            )
            self._set_state(CircuitState.OPEN)
        elif self.failure_count >= self.failure_threshold:
            logger.error(
                f"CircuitBreaker '{self.name}': Threshold exceeded "
                f"({self.failure_count}/{self.failure_threshold}) - Opening circuit"
            )
            self._set_state(CircuitState.OPEN)
        else:
            logger.warning(
                f"CircuitBreaker '{self.name}': Failure {self.failure_count}/{self.failure_threshold}"
            )
    
    def reset(self):
        """Manually reset circuit breaker"""
        logger.info(f"CircuitBreaker '{self.name}': Manual reset")
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        if old_state != CircuitState.CLOSED:
            self._record_state_change(old_state, CircuitState.CLOSED)
    
    def get_state(self) -> dict:
        """Get current circuit breaker state"""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'failure_threshold': self.failure_threshold,
            'last_failure_time': self.last_failure_time
        }
    
    def get_circuit_stats(self) -> dict:
        """Get circuit breaker statistics
        
        Returns:
            Dict with open_count, half_open_count, last_state_change, and more
        """
        self._check_auto_reset()
        
        return {
            'name': self.name,
            'state': self.state.value,
            'open_count': self.open_count,
            'half_open_count': self.half_open_count,
            'last_state_change': self.last_state_change,
            'last_state_change_iso': (
                datetime.fromtimestamp(self.last_state_change).isoformat()
                if self.last_state_change else None
            ),
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'failure_threshold': self.failure_threshold,
            'success_threshold': self.success_threshold,
            'recovery_timeout': self.recovery_timeout,
            'auto_reset': self.auto_reset,
            'auto_reset_timeout': self.auto_reset_timeout,
            'recent_transitions': self._state_history[-10:] if self._state_history else [],
        }


class RateLimiter:
    """Rate limiter using token bucket algorithm
    
    Limits number of operations within a time window.
    """
    
    def __init__(
        self,
        max_calls: int = 30,
        time_window: float = 60.0,
        name: str = "RateLimiter"
    ):
        """Initialize rate limiter
        
        Args:
            max_calls: Maximum number of calls allowed
            time_window: Time window in seconds
            name: Name for logging
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.name = name
        self.call_times: deque = deque(maxlen=max_calls)
        self.last_call_time: float = 0.0
        
        logger.info(
            f"RateLimiter '{name}' initialized: "
            f"{max_calls} calls per {time_window}s"
        )
    
    def acquire(self) -> bool:
        """Try to acquire permission to make a call
        
        Returns:
            True if call is allowed, False if rate limit exceeded
        """
        if self.max_calls <= 0:
            return False
        
        now = time.time()
        
        while self.call_times and now - self.call_times[0] > self.time_window:
            self.call_times.popleft()
        
        if len(self.call_times) < self.max_calls:
            self.call_times.append(now)
            self.last_call_time = now
            return True
        else:
            oldest_call = self.call_times[0]
            wait_time = self.time_window - (now - oldest_call)
            logger.warning(
                f"RateLimiter '{self.name}': Rate limit exceeded. "
                f"Wait {wait_time:.1f}s"
            )
            return False
    
    async def acquire_async(self, wait: bool = False) -> bool:
        """Try to acquire permission (async version)
        
        All exceptions logged before re-raise atau return:
        - Log exception jika sleep interrupted
        - Handle asyncio.CancelledError dengan proper cleanup
        
        Args:
            wait: If True, wait until permission is available
            
        Returns:
            True if call is allowed, False if rate limit exceeded (when wait=False)
        """
        while True:
            try:
                now = time.time()
                
                while self.call_times and now - self.call_times[0] > self.time_window:
                    self.call_times.popleft()
                
                if len(self.call_times) < self.max_calls:
                    self.call_times.append(now)
                    self.last_call_time = now
                    return True
                else:
                    if not wait:
                        oldest_call = self.call_times[0]
                        wait_time = self.time_window - (now - oldest_call)
                        logger.warning(
                            f"RateLimiter '{self.name}': Rate limit exceeded. "
                            f"Wait {wait_time:.1f}s"
                        )
                        return False
                    
                    oldest_call = self.call_times[0]
                    wait_time = self.time_window - (now - oldest_call)
                    logger.info(
                        f"RateLimiter '{self.name}': Waiting {wait_time:.1f}s for rate limit"
                    )
                    
                    try:
                        await asyncio.sleep(wait_time + 0.1)
                    except asyncio.CancelledError:
                        logger.warning(
                            f"RateLimiter '{self.name}': Sleep interrupted by cancellation"
                        )
                        raise
                    except Exception as sleep_error:
                        logger.error(
                            f"RateLimiter '{self.name}': Error saat waitlist sleep: "
                            f"{type(sleep_error).__name__}: {sleep_error}"
                        )
                        return False
                        
            except asyncio.CancelledError:
                logger.warning(f"RateLimiter '{self.name}': acquire_async dibatalkan")
                raise
            except Exception as e:
                logger.error(
                    f"RateLimiter '{self.name}': Error tidak terduga: "
                    f"{type(e).__name__}: {e}"
                )
                return False
    
    def get_remaining(self) -> int:
        """Get remaining calls in current window"""
        now = time.time()
        
        while self.call_times and now - self.call_times[0] > self.time_window:
            self.call_times.popleft()
        
        return self.max_calls - len(self.call_times)
    
    def get_wait_time(self) -> float:
        """Get wait time until next call is allowed (seconds)"""
        now = time.time()
        
        while self.call_times and now - self.call_times[0] > self.time_window:
            self.call_times.popleft()
        
        if len(self.call_times) < self.max_calls:
            return 0.0
        
        oldest_call = self.call_times[0]
        return max(0.0, self.time_window - (now - oldest_call))
    
    def reset(self):
        """Reset rate limiter"""
        logger.info(f"RateLimiter '{self.name}': Reset")
        self.call_times.clear()
    
    def get_state(self) -> dict:
        """Get current rate limiter state"""
        return {
            'name': self.name,
            'max_calls': self.max_calls,
            'time_window': self.time_window,
            'current_calls': len(self.call_times),
            'remaining': self.get_remaining(),
            'wait_time': self.get_wait_time()
        }
    
    def get_time_window(self) -> float:
        """Get time window in seconds (public accessor)"""
        return self.time_window
    
    def get_call_times(self) -> list:
        """Get list of call timestamps (public accessor, returns copy)"""
        return list(self.call_times)
    
    def set_call_times(self, call_times: list):
        """Set call times from restored state (public accessor)"""
        self.call_times.clear()
        for ts in call_times:
            self.call_times.append(ts)


def retry(
    func: Callable[..., T],
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
    default: Optional[T] = None,
    raise_on_exhaust: bool = True
) -> Callable[..., T]:
    """Retry decorator/wrapper untuk fungsi sync dengan exception handling.
    
    All exceptions logged before re-raise atau return:
    - Log exception dengan level ERROR jika callback fail
    - Increment failure counter dan log setiap retry attempt
    - Optionally re-raise atau return default (tergantung use case)
    
    Args:
        func: Fungsi yang akan di-retry
        max_attempts: Maksimum percobaan (default: 3)
        base_delay: Delay awal dalam detik (default: 1.0)
        max_delay: Delay maksimum dalam detik (default: 60.0)
        exceptions: Tuple exception types yang akan di-retry
        on_retry: Optional callback saat retry (attempt, exception, next_delay)
        default: Nilai default jika retry exhausted dan raise_on_exhaust=False
        raise_on_exhaust: Raise exception jika semua retry gagal (default: True)
        
    Returns:
        Hasil fungsi atau default value
        
    Raises:
        RetryExhaustedError: Jika semua retry gagal dan raise_on_exhaust=True
    """
    def wrapper(*args, **kwargs) -> T:
        last_exception: Optional[Exception] = None
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            try:
                result = func(*args, **kwargs)
                if attempt > 1:
                    logger.info(f"Retry berhasil pada percobaan ke-{attempt}")
                return result
            except exceptions as e:
                last_exception = e
                logger.error(
                    f"Retry attempt {attempt}/{max_attempts} gagal: "
                    f"{type(e).__name__}: {e}"
                )
                
                if attempt < max_attempts:
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    logger.info(f"Retry dalam {delay:.2f}s (exponential backoff)...")
                    
                    if on_retry:
                        try:
                            on_retry(attempt, e, delay)
                        except Exception as cb_error:
                            logger.error(f"Error dalam on_retry callback: {cb_error}")
                    
                    time.sleep(delay)
            except (KeyboardInterrupt, SystemExit):
                logger.warning("Retry interrupted oleh KeyboardInterrupt/SystemExit")
                raise
            except Exception as e:
                logger.error(f"Exception tidak terduga di retry: {type(e).__name__}: {e}")
                last_exception = e
                break
        
        logger.error(
            f"Retry exhausted setelah {attempt} percobaan. "
            f"Last error: {last_exception}"
        )
        
        if raise_on_exhaust:
            raise RetryExhaustedError(attempt, last_exception)
        
        return default  # type: ignore
    
    return wrapper


async def retry_async(
    func: Callable[..., Any],
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
    default: Optional[Any] = None,
    raise_on_exhaust: bool = True
) -> Any:
    """Async retry wrapper dengan exception handling.
    
    All exceptions logged before re-raise atau return.
    
    Args:
        func: Async fungsi yang akan di-retry
        max_attempts: Maksimum percobaan
        base_delay: Delay awal dalam detik
        max_delay: Delay maksimum dalam detik
        exceptions: Tuple exception types yang akan di-retry
        on_retry: Optional callback saat retry
        default: Nilai default jika retry exhausted
        raise_on_exhaust: Raise exception jika semua retry gagal
        
    Returns:
        Hasil fungsi atau default value
    """
    last_exception: Optional[Exception] = None
    attempt = 0
    
    while attempt < max_attempts:
        attempt += 1
        try:
            result = await func()
            if attempt > 1:
                logger.info(f"Async retry berhasil pada percobaan ke-{attempt}")
            return result
        except exceptions as e:
            last_exception = e
            logger.error(
                f"Async retry attempt {attempt}/{max_attempts} gagal: "
                f"{type(e).__name__}: {e}"
            )
            
            if attempt < max_attempts:
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                logger.info(f"Async retry dalam {delay:.2f}s (exponential backoff)...")
                
                if on_retry:
                    try:
                        on_retry(attempt, e, delay)
                    except Exception as cb_error:
                        logger.error(f"Error dalam on_retry callback: {cb_error}")
                
                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    logger.warning("Async retry sleep dibatalkan")
                    raise
                except Exception as sleep_error:
                    logger.error(f"Error saat async sleep dalam retry: {sleep_error}")
        except asyncio.CancelledError:
            logger.warning("Async retry dibatalkan")
            raise
        except (KeyboardInterrupt, SystemExit):
            logger.warning("Async retry interrupted")
            raise
        except Exception as e:
            logger.error(f"Exception tidak terduga di async retry: {type(e).__name__}: {e}")
            last_exception = e
            break
    
    logger.error(
        f"Async retry exhausted setelah {attempt} percobaan. "
        f"Last error: {last_exception}"
    )
    
    if raise_on_exhaust:
        raise RetryExhaustedError(attempt, last_exception)
    
    return default


def exponential_backoff(
    attempt: int,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    jitter: bool = True
) -> float:
    """Menghitung delay exponential backoff dengan max_attempts check.
    
    All exceptions logged before re-raise atau return:
    - Log setiap retry attempt dengan backoff duration
    - Ensure semua exception di-catch
    - max_attempts check untuk prevent infinite loop
    
    Args:
        attempt: Nomor percobaan saat ini (1-indexed)
        base_delay: Delay dasar dalam detik
        max_delay: Delay maksimum dalam detik
        max_attempts: Maksimum percobaan untuk validation
        jitter: Tambahkan jitter random untuk mencegah thundering herd
        
    Returns:
        Delay dalam detik
        
    Raises:
        ValueError: Jika attempt melebihi max_attempts
    """
    try:
        if attempt <= 0:
            logger.warning(f"Invalid attempt number: {attempt}, using 1")
            attempt = 1
        
        if attempt > max_attempts:
            logger.error(
                f"Attempt {attempt} melebihi max_attempts {max_attempts} - "
                f"prevent infinite loop"
            )
            raise ValueError(
                f"Attempt {attempt} melebihi max_attempts {max_attempts}"
            )
        
        delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
        
        if jitter:
            import random
            jitter_factor = random.uniform(0.8, 1.2)
            delay = delay * jitter_factor
        
        logger.debug(
            f"Exponential backoff: attempt={attempt}/{max_attempts}, "
            f"delay={delay:.2f}s (base={base_delay}s, max={max_delay}s)"
        )
        
        return delay
        
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Error menghitung exponential backoff: {type(e).__name__}: {e}")
        return min(base_delay, max_delay)


async def exponential_backoff_async(
    attempt: int,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    jitter: bool = True
) -> None:
    """Async exponential backoff dengan sleep.
    
    All exceptions logged before re-raise atau return.
    
    Args:
        attempt: Nomor percobaan saat ini
        base_delay: Delay dasar dalam detik
        max_delay: Delay maksimum dalam detik
        max_attempts: Maksimum percobaan
        jitter: Tambahkan jitter random
        
    Raises:
        ValueError: Jika attempt melebihi max_attempts
        asyncio.CancelledError: Jika sleep dibatalkan
    """
    try:
        delay = exponential_backoff(
            attempt, base_delay, max_delay, max_attempts, jitter
        )
        
        logger.info(f"Async backoff sleeping for {delay:.2f}s...")
        
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            logger.warning(f"Async backoff sleep dibatalkan pada attempt {attempt}")
            raise
        except Exception as sleep_error:
            logger.error(f"Error saat async backoff sleep: {sleep_error}")
            raise
            
    except ValueError:
        raise
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.error(f"Error dalam exponential_backoff_async: {type(e).__name__}: {e}")
        raise
