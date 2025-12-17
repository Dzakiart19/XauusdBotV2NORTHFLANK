"""
Lightweight resilience module - simplified for Koyeb free tier.
"""
import time
import asyncio
from collections import deque
from typing import Optional
from bot.logger import setup_logger

logger = setup_logger('Resilience')


class CircuitBreaker:
    """Simple circuit breaker - lightweight version"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0,
                 expected_exception: type = Exception, name: str = "CircuitBreaker", **kwargs):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.is_open = False
    
    def call(self, func, *args, **kwargs):
        if self.is_open:
            if time.time() - (self.last_failure_time or 0) > self.recovery_timeout:
                self.is_open = False
                self.failure_count = 0
            else:
                raise Exception(f"Circuit breaker {self.name} is open")
        
        try:
            result = func(*args, **kwargs)
            self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.is_open = True
                logger.warning(f"Circuit breaker {self.name} opened after {self.failure_count} failures")
            raise
    
    async def call_async(self, func, *args, **kwargs):
        if self.is_open:
            if time.time() - (self.last_failure_time or 0) > self.recovery_timeout:
                self.is_open = False
                self.failure_count = 0
            else:
                raise Exception(f"Circuit breaker {self.name} is open")
        
        try:
            result = await func(*args, **kwargs)
            self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.is_open = True
            raise
    
    def get_state(self) -> dict:
        """Get current circuit breaker state as dictionary"""
        if self.is_open:
            time_since_failure = time.time() - (self.last_failure_time or 0)
            if time_since_failure < self.recovery_timeout:
                state = 'OPEN'
            else:
                state = 'HALF_OPEN'
        else:
            state = 'CLOSED'
        
        return {
            'state': state,
            'failure_count': self.failure_count,
            'failure_threshold': self.failure_threshold,
            'is_open': self.is_open,
            'last_failure_time': self.last_failure_time,
            'recovery_timeout': self.recovery_timeout
        }
    
    def reset(self) -> None:
        """Reset circuit breaker to closed state"""
        self.is_open = False
        self.failure_count = 0
        self.last_failure_time = None
    
    def record_success(self) -> None:
        """Record successful call"""
        self.failure_count = 0
        if self.is_open:
            self.is_open = False
            logger.info(f"Circuit breaker {self.name} closed after successful recovery")
    
    def record_failure(self) -> None:
        """Record failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.is_open = True
            logger.warning(f"Circuit breaker {self.name} opened after {self.failure_count} failures")


class RateLimiter:
    """Simple rate limiter - lightweight version"""
    
    def __init__(self, max_calls: int = 30, time_window: float = 60.0, name: str = "RateLimiter"):
        self.max_calls = max_calls
        self.time_window = time_window
        self.name = name
        self.call_times: deque = deque(maxlen=max_calls)
        self.last_call_time: float = 0
    
    def acquire(self) -> bool:
        now = time.time()
        while self.call_times and now - self.call_times[0] > self.time_window:
            self.call_times.popleft()
        
        if len(self.call_times) < self.max_calls:
            self.call_times.append(now)
            self.last_call_time = now
            return True
        return False
    
    async def acquire_async(self, wait: bool = False) -> bool:
        while True:
            now = time.time()
            while self.call_times and now - self.call_times[0] > self.time_window:
                self.call_times.popleft()
            
            if len(self.call_times) < self.max_calls:
                self.call_times.append(now)
                self.last_call_time = now
                return True
            
            if not wait:
                return False
            
            await asyncio.sleep(0.5)
