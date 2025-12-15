"""Unit tests for internationalization module."""
import pytest
from datetime import datetime
import pytz
from bot.i18n import (
    get_text, t,
    get_user_language, set_user_language,
    format_datetime, format_currency, format_percentage,
    SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE,
)


class TestGetText:
    """Tests for get_text function."""
    
    def test_get_text_default_language(self):
        """Should return Indonesian text by default."""
        result = get_text('welcome')
        assert "Selamat datang" in result
    
    def test_get_text_english(self):
        """Should return English text when specified."""
        result = get_text('welcome', language='en')
        assert "Welcome" in result
    
    def test_get_text_with_format_args(self):
        """Should format text with arguments."""
        result = get_text('welcome_user', language='en', name='John')
        assert "John" in result
    
    def test_get_text_missing_key(self):
        """Should return key if translation not found."""
        result = get_text('nonexistent_key_12345')
        assert result == 'nonexistent_key_12345'
    
    def test_get_text_unsupported_language(self):
        """Should fallback to default for unsupported language."""
        result = get_text('welcome', language='fr')
        assert "Selamat datang" in result
    
    def test_shorthand_function(self):
        """Should work with t() shorthand."""
        result = t('welcome')
        assert "Selamat datang" in result


class TestUserLanguagePreference:
    """Tests for user language preference functions."""
    
    def test_set_get_user_language(self):
        """Should set and get user language."""
        user_id = 12345
        set_user_language(user_id, 'en')
        assert get_user_language(user_id) == 'en'
    
    def test_default_language(self):
        """Should return default for unknown user."""
        assert get_user_language(999999) == DEFAULT_LANGUAGE
    
    def test_set_invalid_language(self):
        """Should reject unsupported language."""
        result = set_user_language(12345, 'invalid')
        assert result is False
    
    def test_supported_languages(self):
        """Should support Indonesian and English."""
        assert 'id' in SUPPORTED_LANGUAGES
        assert 'en' in SUPPORTED_LANGUAGES


class TestFormatFunctions:
    """Tests for formatting functions."""
    
    def test_format_currency_positive(self):
        """Should format positive currency."""
        result = format_currency(100.50)
        assert result == "+$100.50"
    
    def test_format_currency_negative(self):
        """Should format negative currency."""
        result = format_currency(-50.25)
        assert "$" in result
        assert "-50.25" in result
    
    def test_format_percentage_positive(self):
        """Should format positive percentage."""
        result = format_percentage(75.5)
        assert result == "+75.5%"
    
    def test_format_percentage_negative(self):
        """Should format negative percentage."""
        result = format_percentage(-10.0)
        assert result == "-10.0%"
    
    def test_format_datetime_full(self):
        """Should format datetime in full style."""
        dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=pytz.UTC)
        result = format_datetime(dt, style='full')
        assert "WIB" in result
    
    def test_format_datetime_english(self):
        """Should format datetime in English style."""
        set_user_language(99999, 'en')
        dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=pytz.UTC)
        result = format_datetime(dt, user_id=99999, style='full')
        assert "WIB" in result
