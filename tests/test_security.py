"""Unit tests for security module."""
import pytest
from bot.security import (
    mask_sensitive_data,
    sanitize_user_input,
    validate_telegram_user_id,
    verify_telegram_webhook,
    generate_webhook_secret,
    WebhookVerificationError,
)


class TestMaskSensitiveData:
    """Tests for mask_sensitive_data function."""
    
    def test_masks_bot_token(self):
        """Should mask Telegram bot tokens."""
        text = "Token: 1234567890:ABCdefGHIjklMNOpqrSTUvwxYZ123456789"
        result = mask_sensitive_data(text)
        assert "1234567890:ABCdefGHIjklMNOpqrSTUvwxYZ123456789" not in result
        assert "BOT_TOKEN" in result
    
    def test_masks_openai_key(self):
        """Should mask OpenAI API keys."""
        text = "Key: sk-abcdefghijklmnopqrstuvwxyz1234567890"
        result = mask_sensitive_data(text)
        assert "sk-abcdefghijklmnopqrstuvwxyz1234567890" not in result
    
    def test_masks_database_url(self):
        """Should mask database URLs with credentials."""
        text = "DB: postgres://user:password123@host.com:5432/dbname"
        result = mask_sensitive_data(text)
        assert "password123" not in result
    
    def test_preserves_safe_text(self):
        """Should not modify safe text."""
        text = "Hello, this is a normal message."
        result = mask_sensitive_data(text)
        assert result == text
    
    def test_handles_none_input(self):
        """Should handle None input gracefully."""
        result = mask_sensitive_data(None)
        assert result is None
    
    def test_handles_empty_string(self):
        """Should handle empty string."""
        result = mask_sensitive_data("")
        assert result == ""


class TestSanitizeUserInput:
    """Tests for sanitize_user_input function."""
    
    def test_removes_script_tags(self):
        """Should remove script tags."""
        text = "Hello <script>alert('xss')</script> World"
        result = sanitize_user_input(text)
        assert "<script>" not in result
        assert "</script>" not in result
    
    def test_removes_javascript_protocol(self):
        """Should remove javascript: protocol."""
        text = "javascript:alert(1)"
        result = sanitize_user_input(text)
        assert "javascript:" not in result
    
    def test_truncates_long_input(self):
        """Should truncate input exceeding max length."""
        text = "A" * 1000
        result = sanitize_user_input(text, max_length=100)
        assert len(result) == 100
    
    def test_strips_whitespace(self):
        """Should strip leading and trailing whitespace."""
        text = "  Hello World  "
        result = sanitize_user_input(text)
        assert result == "Hello World"
    
    def test_removes_sql_injection_patterns(self):
        """Should remove SQL injection patterns."""
        text = "'; DROP TABLE users; --"
        result = sanitize_user_input(text)
        assert "; DROP" not in result
    
    def test_handles_none_input(self):
        """Should handle None input."""
        result = sanitize_user_input(None)
        assert result == ""
    
    def test_handles_empty_string(self):
        """Should handle empty string."""
        result = sanitize_user_input("")
        assert result == ""


class TestValidateTelegramUserId:
    """Tests for validate_telegram_user_id function."""
    
    def test_valid_integer_id(self):
        """Should accept valid integer user ID."""
        is_valid, user_id, error = validate_telegram_user_id(123456789)
        assert is_valid is True
        assert user_id == 123456789
        assert error is None
    
    def test_valid_string_id(self):
        """Should accept valid string user ID."""
        is_valid, user_id, error = validate_telegram_user_id("123456789")
        assert is_valid is True
        assert user_id == 123456789
        assert error is None
    
    def test_invalid_negative_id(self):
        """Should reject negative user ID."""
        is_valid, user_id, error = validate_telegram_user_id(-123)
        assert is_valid is False
        assert user_id is None
        assert "positive" in error.lower()
    
    def test_invalid_zero_id(self):
        """Should reject zero user ID."""
        is_valid, user_id, error = validate_telegram_user_id(0)
        assert is_valid is False
        assert user_id is None
    
    def test_invalid_string_id(self):
        """Should reject non-numeric string."""
        is_valid, user_id, error = validate_telegram_user_id("abc123")
        assert is_valid is False
        assert user_id is None
    
    def test_none_input(self):
        """Should reject None input."""
        is_valid, user_id, error = validate_telegram_user_id(None)
        assert is_valid is False
        assert user_id is None
    
    def test_too_large_id(self):
        """Should reject extremely large user ID."""
        is_valid, user_id, error = validate_telegram_user_id(99999999999999999)
        assert is_valid is False


class TestWebhookVerification:
    """Tests for webhook verification functions."""
    
    def test_generate_webhook_secret(self):
        """Should generate consistent secret for same token."""
        token = "1234567890:ABCdefGHI"
        secret1 = generate_webhook_secret(token)
        secret2 = generate_webhook_secret(token)
        assert secret1 == secret2
        assert len(secret1) == 32
    
    def test_generate_webhook_secret_different_tokens(self):
        """Should generate different secrets for different tokens."""
        secret1 = generate_webhook_secret("token1:abc")
        secret2 = generate_webhook_secret("token2:xyz")
        assert secret1 != secret2
    
    def test_verify_webhook_empty_body(self):
        """Should reject empty request body."""
        with pytest.raises(WebhookVerificationError):
            verify_telegram_webhook("token", b"")
    
    def test_verify_webhook_invalid_json(self):
        """Should reject invalid JSON."""
        with pytest.raises(WebhookVerificationError):
            verify_telegram_webhook("token", b"not json")
    
    def test_verify_webhook_missing_update_id(self):
        """Should reject update without update_id."""
        with pytest.raises(WebhookVerificationError):
            verify_telegram_webhook("token", b'{"message": {}}')
    
    def test_verify_webhook_valid_update(self):
        """Should accept valid update."""
        body = b'{"update_id": 123, "message": {"text": "hello"}}'
        result = verify_telegram_webhook("token", body)
        assert result is True
