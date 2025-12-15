"""
Security utilities for the Trading Bot.

Provides:
- HMAC webhook verification
- Enhanced secret masking
- Request validation
- Rate limiting decorators
"""
import hashlib
import hmac
import re
import time
import logging
from functools import wraps
from typing import Optional, Dict, Any, Callable
from datetime import datetime

logger = logging.getLogger('Security')


class SecurityError(Exception):
    """Base exception for security-related errors"""
    pass


class WebhookVerificationError(SecurityError):
    """Raised when webhook verification fails"""
    pass


class RateLimitExceededError(SecurityError):
    """Raised when rate limit is exceeded"""
    pass


def verify_telegram_webhook(token: str, request_body: bytes, 
                           secret_token: Optional[str] = None,
                           received_secret: Optional[str] = None) -> bool:
    """
    Verify Telegram webhook request using HMAC-SHA256.
    
    Telegram sends X-Telegram-Bot-Api-Secret-Token header when secret_token is set.
    
    Args:
        token: Bot token for verification
        request_body: Raw request body bytes
        secret_token: Secret token set when registering webhook
        received_secret: Secret received in X-Telegram-Bot-Api-Secret-Token header
    
    Returns:
        True if verification passes
    
    Raises:
        WebhookVerificationError: If verification fails
    """
    if not token:
        raise WebhookVerificationError("Bot token is required for verification")
    
    if not request_body:
        raise WebhookVerificationError("Request body is empty")
    
    if secret_token:
        if not received_secret:
            raise WebhookVerificationError("Missing secret token in request header")
        
        expected_mac = hmac.new(
            token.encode('utf-8'),
            secret_token.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        if not hmac.compare_digest(received_secret, expected_mac):
            if not hmac.compare_digest(received_secret, secret_token):
                raise WebhookVerificationError("Invalid secret token - webhook authentication failed")
    
    try:
        import json
        data = json.loads(request_body.decode('utf-8'))
        
        if 'update_id' not in data:
            raise WebhookVerificationError("Invalid Telegram update: missing update_id")
        
        if 'message' not in data and 'callback_query' not in data and 'edited_message' not in data:
            logger.warning("Webhook update without message/callback_query/edited_message")
        
        return True
        
    except json.JSONDecodeError as e:
        raise WebhookVerificationError(f"Invalid JSON in request body: {e}")


def verify_webhook_secret(received_secret: str, expected_secret: str) -> bool:
    """
    Verify webhook secret token using constant-time comparison.
    
    Args:
        received_secret: Secret received in X-Telegram-Bot-Api-Secret-Token header
        expected_secret: Expected secret token
    
    Returns:
        True if secrets match
    """
    if not received_secret or not expected_secret:
        return False
    
    return hmac.compare_digest(received_secret.encode(), expected_secret.encode())


def generate_webhook_secret(bot_token: str, length: int = 32) -> str:
    """
    Generate a secure webhook secret based on bot token.
    
    Args:
        bot_token: Telegram bot token
        length: Length of generated secret (max 256)
    
    Returns:
        Hex-encoded secret string
    """
    if not bot_token:
        raise ValueError("Bot token is required")
    
    length = min(length, 256)
    
    secret_data = f"{bot_token}:{int(time.time() // 3600)}"
    hash_bytes = hashlib.sha256(secret_data.encode()).hexdigest()
    
    return hash_bytes[:length]


SENSITIVE_PATTERNS = [
    (r'\b\d{8,10}:[A-Za-z0-9_-]{35,}\b', 'BOT_TOKEN'),
    (r'\bsk-[A-Za-z0-9]{32,}\b', 'OPENAI_KEY'),
    (r'\bAKIA[0-9A-Z]{16}\b', 'AWS_KEY'),
    (r'\b[A-Za-z0-9]{32,64}\b(?=.*(?:key|secret|token|password))', 'API_KEY'),
    (r'postgres(?:ql)?://[^:]+:[^@]+@[^\s]+', 'DATABASE_URL'),
    (r'mysql://[^:]+:[^@]+@[^\s]+', 'DATABASE_URL'),
    (r'\bBearer\s+[A-Za-z0-9\-._~+/]+=*\b', 'BEARER_TOKEN'),
    (r'\bpassword[=:\s]+[^\s]{8,}\b', 'PASSWORD'),
    (r'\bsecret[=:\s]+[^\s]{8,}\b', 'SECRET'),
]


def mask_sensitive_data(text: str, mask_char: str = '*') -> str:
    """
    Mask sensitive data in text with configurable patterns.
    
    Args:
        text: Text to sanitize
        mask_char: Character to use for masking
    
    Returns:
        Sanitized text with sensitive data masked
    """
    if not text or not isinstance(text, str):
        return text
    
    result = text
    
    for pattern, label in SENSITIVE_PATTERNS:
        try:
            matches = re.findall(pattern, result, re.IGNORECASE)
            for match in matches:
                if len(match) > 8:
                    masked = f"{match[:4]}{mask_char * 4}...{match[-4:]}"
                else:
                    masked = mask_char * len(match)
                result = result.replace(match, f"[{label}:{masked}]")
        except re.error:
            continue
    
    return result


def sanitize_user_input(text: str, max_length: int = 500, 
                        allowed_chars: Optional[str] = None) -> str:
    """
    Sanitize user input to prevent injection attacks.
    
    Args:
        text: User input text
        max_length: Maximum allowed length
        allowed_chars: Optional string of additional allowed characters
    
    Returns:
        Sanitized text
    """
    if not text or not isinstance(text, str):
        return ""
    
    sanitized = text.strip()
    
    sanitized = ''.join(c for c in sanitized if c.isprintable())
    
    dangerous_patterns = [
        '<script>', '</script>', 'javascript:', 
        'onerror=', 'onclick=', 'onload=',
        '${', '`', '$(', '#{',
        '../', '..\\',
        '; DROP', '; DELETE', '; UPDATE', "' OR '1'='1",
    ]
    
    lower_text = sanitized.lower()
    for pattern in dangerous_patterns:
        if pattern.lower() in lower_text:
            logger.warning(f"Dangerous pattern detected in input: {pattern}")
            sanitized = sanitized.replace(pattern, '')
            sanitized = re.sub(re.escape(pattern), '', sanitized, flags=re.IGNORECASE)
    
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized


def validate_telegram_user_id(user_id: Any) -> tuple:
    """
    Validate Telegram user ID.
    
    Args:
        user_id: User ID to validate
    
    Returns:
        Tuple of (is_valid, validated_id, error_message)
    """
    if user_id is None:
        return False, None, "User ID cannot be None"
    
    try:
        if isinstance(user_id, str):
            user_id = user_id.strip()
            if not user_id.lstrip('-').isdigit():
                return False, None, "User ID must be numeric"
            user_id = int(user_id)
        elif not isinstance(user_id, int):
            return False, None, f"Invalid user ID type: {type(user_id)}"
        
        if user_id <= 0:
            return False, None, "User ID must be positive"
        
        if user_id > 9999999999999:
            return False, None, "User ID too large"
        
        return True, user_id, None
        
    except (ValueError, TypeError) as e:
        return False, None, f"User ID validation error: {e}"


class SecureLogger:
    """
    Logger wrapper that automatically masks sensitive data.
    """
    
    def __init__(self, logger_name: str):
        self.logger = logging.getLogger(logger_name)
    
    def _sanitize(self, msg: str) -> str:
        return mask_sensitive_data(str(msg))
    
    def debug(self, msg: str, *args, **kwargs):
        self.logger.debug(self._sanitize(msg), *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        self.logger.info(self._sanitize(msg), *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        self.logger.warning(self._sanitize(msg), *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        self.logger.error(self._sanitize(msg), *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        self.logger.critical(self._sanitize(msg), *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs):
        self.logger.exception(self._sanitize(msg), *args, **kwargs)


def rate_limit(max_calls: int, period: float, key_func: Optional[Callable] = None):
    """
    Decorator for rate limiting function calls.
    
    Args:
        max_calls: Maximum number of calls allowed
        period: Time period in seconds
        key_func: Optional function to extract rate limit key from args
    
    Returns:
        Decorated function
    """
    call_times: Dict[str, list] = {}
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = key_func(*args, **kwargs) if key_func else 'default'
            now = time.time()
            
            if key not in call_times:
                call_times[key] = []
            
            call_times[key] = [t for t in call_times[key] if now - t < period]
            
            if len(call_times[key]) >= max_calls:
                raise RateLimitExceededError(
                    f"Rate limit exceeded: {max_calls} calls per {period}s"
                )
            
            call_times[key].append(now)
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def rate_limit_async(max_calls: int, period: float, key_func: Optional[Callable] = None):
    """
    Async decorator for rate limiting function calls.
    
    Args:
        max_calls: Maximum number of calls allowed
        period: Time period in seconds
        key_func: Optional function to extract rate limit key from args
    
    Returns:
        Decorated async function
    """
    call_times: Dict[str, list] = {}
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = key_func(*args, **kwargs) if key_func else 'default'
            now = time.time()
            
            if key not in call_times:
                call_times[key] = []
            
            call_times[key] = [t for t in call_times[key] if now - t < period]
            
            if len(call_times[key]) >= max_calls:
                raise RateLimitExceededError(
                    f"Rate limit exceeded: {max_calls} calls per {period}s"
                )
            
            call_times[key].append(now)
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator
