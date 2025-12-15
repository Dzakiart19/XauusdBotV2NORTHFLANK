"""Unit tests for backup module with security focus."""
import pytest
from bot.backup import DatabaseBackupManager


class TestBackupSecurity:
    """Tests for backup security measures."""
    
    def test_sanitize_valid_hostname(self):
        """Should accept valid hostname."""
        manager = DatabaseBackupManager()
        manager.is_postgres = True
        result = manager._sanitize_pg_param("db.example.com", "hostname")
        assert result == "db.example.com"
    
    def test_sanitize_hostname_with_port(self):
        """Should reject hostname with injection attempt."""
        manager = DatabaseBackupManager()
        manager.is_postgres = True
        with pytest.raises(ValueError):
            manager._sanitize_pg_param("host; rm -rf /", "hostname")
    
    def test_sanitize_hostname_with_pipe(self):
        """Should reject hostname with pipe character."""
        manager = DatabaseBackupManager()
        manager.is_postgres = True
        with pytest.raises(ValueError):
            manager._sanitize_pg_param("host | cat /etc/passwd", "hostname")
    
    def test_sanitize_username_valid(self):
        """Should accept valid username."""
        manager = DatabaseBackupManager()
        manager.is_postgres = True
        result = manager._sanitize_pg_param("postgres_user", "username")
        assert result == "postgres_user"
    
    def test_sanitize_username_with_injection(self):
        """Should reject username with injection attempt."""
        manager = DatabaseBackupManager()
        manager.is_postgres = True
        with pytest.raises(ValueError):
            manager._sanitize_pg_param("user$(whoami)", "username")
    
    def test_sanitize_database_valid(self):
        """Should accept valid database name."""
        manager = DatabaseBackupManager()
        manager.is_postgres = True
        result = manager._sanitize_pg_param("my_database", "database")
        assert result == "my_database"
    
    def test_sanitize_database_with_backticks(self):
        """Should reject database name with backticks."""
        manager = DatabaseBackupManager()
        manager.is_postgres = True
        with pytest.raises(ValueError):
            manager._sanitize_pg_param("`whoami`", "database")
    
    def test_sanitize_database_with_semicolon(self):
        """Should reject database name with semicolon."""
        manager = DatabaseBackupManager()
        manager.is_postgres = True
        with pytest.raises(ValueError):
            manager._sanitize_pg_param("db; DROP TABLE users", "database")
    
    def test_sanitize_with_newline(self):
        """Should reject values with newlines."""
        manager = DatabaseBackupManager()
        manager.is_postgres = True
        with pytest.raises(ValueError):
            manager._sanitize_pg_param("host\nmalicious", "hostname")
    
    def test_sanitize_with_null_byte(self):
        """Should reject values with null bytes."""
        manager = DatabaseBackupManager()
        manager.is_postgres = True
        with pytest.raises(ValueError):
            manager._sanitize_pg_param("host\x00malicious", "hostname")


class TestBackupManagerInit:
    """Tests for DatabaseBackupManager initialization."""
    
    def test_default_initialization(self):
        """Should initialize with defaults."""
        manager = DatabaseBackupManager()
        assert manager.max_backups == 7
        assert manager.is_postgres is False
    
    def test_custom_max_backups(self):
        """Should accept custom max_backups."""
        manager = DatabaseBackupManager(max_backups=10)
        assert manager.max_backups == 10
    
    def test_configure_postgres(self):
        """Should configure PostgreSQL mode."""
        manager = DatabaseBackupManager()
        manager.configure_postgres("postgres://user:pass@host/db")
        assert manager.is_postgres is True
    
    def test_configure_postgres_invalid_url(self):
        """Should not enable postgres for invalid URL."""
        manager = DatabaseBackupManager()
        manager.configure_postgres("mysql://user:pass@host/db")
        assert manager.is_postgres is False
