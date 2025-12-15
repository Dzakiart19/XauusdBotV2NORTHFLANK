"""Unit tests for config module."""
import pytest
import os
from unittest.mock import patch, MagicMock


class TestConfigAuthorization:
    """Tests for Config authorization methods."""
    
    @pytest.fixture
    def mock_env(self):
        """Setup mock environment variables."""
        with patch.dict(os.environ, {
            'TELEGRAM_BOT_TOKEN': '123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11',
            'AUTHORIZED_USER_IDS': '7879056603,123456789',
            'ID_USER_PUBLIC': '111111111',
            'BYPASS_SIGNAL_QUALITY_CHECK': 'false',
            'AUTO_SIGNAL_REPLACEMENT_ALLOWED': 'false'
        }):
            yield
    
    def test_parse_user_ids_valid(self):
        """Should parse valid user IDs correctly."""
        from config import _parse_user_ids
        
        result = _parse_user_ids('123,456,789')
        assert result == [123, 456, 789]
    
    def test_parse_user_ids_empty(self):
        """Should return empty list for empty input."""
        from config import _parse_user_ids
        
        result = _parse_user_ids('')
        assert result == []
    
    def test_parse_user_ids_invalid(self):
        """Should return empty list for invalid input."""
        from config import _parse_user_ids
        
        result = _parse_user_ids('abc,def')
        assert result == []
        
        result = _parse_user_ids('-1,0')
        assert result == []
    
    def test_is_owner_method(self, mock_env):
        """Should check owner status correctly."""
        from config import Config
        
        Config._refresh_secrets()
        
        assert Config.is_owner(7879056603) == True
        assert Config.is_owner(123456789) == True
        assert Config.is_owner(999999999) == False
    
    def test_is_public_user_method(self, mock_env):
        """Should check public user status correctly."""
        from config import Config
        
        Config._refresh_secrets()
        
        assert Config.is_public_user(111111111) == True
        assert Config.is_public_user(999999999) == False
    
    def test_has_full_access_method(self, mock_env):
        """Should check full access correctly."""
        from config import Config
        
        Config._refresh_secrets()
        
        assert Config.has_full_access(7879056603) == True
        assert Config.has_full_access(111111111) == True
        assert Config.has_full_access(999999999) == False
    
    def test_ensure_secrets_loaded(self, mock_env):
        """Should load secrets when called."""
        from config import Config
        
        Config._secrets_refreshed = False
        Config.AUTHORIZED_USER_IDS = []
        
        Config.ensure_secrets_loaded()
        
        assert Config._secrets_refreshed == True
        assert len(Config.AUTHORIZED_USER_IDS) > 0
    
    def test_audit_bypass_flag(self, mock_env):
        """Should audit bypass flags correctly."""
        from config import Config
        
        Config._bypass_flags_audit_log = []
        Config._audit_bypass_flag('TEST_FLAG', True, 'Test reason')
        
        assert len(Config._bypass_flags_audit_log) == 1
        assert Config._bypass_flags_audit_log[0]['flag'] == 'TEST_FLAG'
        assert Config._bypass_flags_audit_log[0]['value'] == True


class TestConfigRefreshSecrets:
    """Tests for Config._refresh_secrets method."""
    
    def test_refresh_loads_token(self):
        """Should load token from environment."""
        with patch.dict(os.environ, {
            'TELEGRAM_BOT_TOKEN': 'test_token_12345',
            'AUTHORIZED_USER_IDS': '123'
        }):
            from config import Config
            Config._refresh_secrets()
            
            assert Config.TELEGRAM_BOT_TOKEN == 'test_token_12345'
    
    def test_refresh_loads_user_ids(self):
        """Should load user IDs from environment."""
        with patch.dict(os.environ, {
            'TELEGRAM_BOT_TOKEN': 'test_token',
            'AUTHORIZED_USER_IDS': '123,456,789'
        }):
            from config import Config
            Config._refresh_secrets()
            
            assert 123 in Config.AUTHORIZED_USER_IDS
            assert 456 in Config.AUTHORIZED_USER_IDS
            assert 789 in Config.AUTHORIZED_USER_IDS
    
    def test_refresh_returns_debug_info(self):
        """Should return debug info dict."""
        with patch.dict(os.environ, {
            'TELEGRAM_BOT_TOKEN': 'test_token_12345',
            'AUTHORIZED_USER_IDS': '123,456'
        }):
            from config import Config
            result = Config._refresh_secrets()
            
            assert 'token_set' in result
            assert 'users_count' in result
            assert result['users_count'] == 2
    
    def test_refresh_handles_empty_env(self):
        """Should handle empty environment gracefully."""
        with patch.dict(os.environ, {}, clear=True):
            from config import Config
            result = Config._refresh_secrets()
            
            assert result['token_set'] == False
            assert result['users_count'] == 0
