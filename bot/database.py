from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, text, BigInteger, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import SQLAlchemyError, OperationalError, IntegrityError, TimeoutError as SATimeoutError
from sqlalchemy.exc import DatabaseError as SQLAlchemyDatabaseError
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from datetime import datetime
import os
import time
import threading
from typing import Callable, Any, Optional, Generator, Dict, List, cast
from functools import wraps
import logging

logger = logging.getLogger('DatabaseManager')

Base = declarative_base()

_transaction_lock = threading.Lock()

POOL_SIZE = 5
MAX_OVERFLOW = 10
POOL_TIMEOUT = 30
POOL_RECYCLE = 3600
POOL_PRE_PING = True
POOL_EXHAUSTED_MAX_RETRIES = 3
POOL_EXHAUSTED_INITIAL_DELAY = 0.5
POOL_HIGH_UTILIZATION_THRESHOLD = 80
TRANSACTION_MAX_RETRIES = 3
TRANSACTION_INITIAL_DELAY = 0.1
DEADLOCK_RETRY_DELAY = 0.2

class DatabaseError(Exception):
    """Base exception for database errors"""
    pass

class RetryableError(DatabaseError):
    """Database error that can be retried"""
    pass

class ConnectionPoolExhausted(DatabaseError):
    """Connection pool exhausted error after all retries failed"""
    pass

class PoolTimeoutError(DatabaseError):
    """Pool timeout error - could not acquire connection in time"""
    pass

class DeadlockError(DatabaseError):
    """Deadlock detected during transaction"""
    pass

class OrphanedRecordError(DatabaseError):
    """Orphaned record detected - trade/position mismatch"""
    pass

def _is_deadlock_error(error: Exception) -> bool:
    """Deteksi apakah error adalah deadlock.
    
    Args:
        error: Exception yang akan dicek
        
    Returns:
        True jika error adalah deadlock, False jika bukan
    """
    error_str = str(error).lower()
    deadlock_indicators = ['deadlock', 'lock wait timeout', 'database is locked', 'sqlite3.operationalerror: database is locked']
    return any(indicator in error_str for indicator in deadlock_indicators)

def retry_on_db_error(max_retries: int = 3, initial_delay: float = 0.1):
    """Decorator untuk retry operasi database dengan exponential backoff.
    
    Menangani pool timeout errors, operational errors, dan deadlock dengan retry logic.
    
    Args:
        max_retries: Jumlah maksimum percobaan retry
        initial_delay: Delay awal dalam detik sebelum retry pertama
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            delay = initial_delay
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except SATimeoutError as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"⚠️ Pool timeout pada {func.__name__} (percobaan {attempt + 1}/{max_retries}): {e}"
                        )
                        logger.info(f"🔄 Mencoba ulang dalam {delay:.2f} detik...")
                        time.sleep(delay)
                        delay *= 2
                    else:
                        logger.error(
                            f"❌ Pool timeout - batas retry tercapai untuk {func.__name__}: {e}"
                        )
                        raise ConnectionPoolExhausted(
                            f"Pool habis setelah {max_retries} percobaan di {func.__name__}"
                        ) from e
                except OperationalError as e:
                    last_exception = e
                    is_deadlock = _is_deadlock_error(e)
                    
                    if is_deadlock:
                        if attempt < max_retries - 1:
                            deadlock_delay = max(delay, DEADLOCK_RETRY_DELAY)
                            logger.warning(
                                f"🔒 Deadlock terdeteksi pada {func.__name__} "
                                f"(percobaan {attempt + 1}/{max_retries}): {e}"
                            )
                            logger.info(f"🔄 Mencoba ulang setelah deadlock dalam {deadlock_delay:.2f} detik...")
                            time.sleep(deadlock_delay)
                            delay *= 2
                        else:
                            logger.error(
                                f"❌ Deadlock persisten pada {func.__name__} setelah {max_retries} percobaan: {e}"
                            )
                            raise DeadlockError(
                                f"Deadlock tidak dapat diselesaikan setelah {max_retries} percobaan"
                            ) from e
                    elif attempt < max_retries - 1:
                        logger.warning(
                            f"⚠️ Error operasional database pada {func.__name__} "
                            f"(percobaan {attempt + 1}/{max_retries}): {e}"
                        )
                        logger.info(f"🔄 Mencoba ulang dalam {delay:.2f} detik...")
                        time.sleep(delay)
                        delay *= 2
                    else:
                        logger.error(
                            f"❌ Batas retry tercapai untuk {func.__name__}: {e}"
                        )
                        raise
                except IntegrityError as e:
                    logger.error(f"❌ Error integritas pada {func.__name__} (tidak dapat di-retry): {e}")
                    raise
                except SQLAlchemyDatabaseError as e:
                    logger.error(f"❌ Error database pada {func.__name__} (tidak dapat di-retry): {e}")
                    raise
                except (ValueError, TypeError, IOError, RuntimeError) as e:
                    logger.error(f"❌ Error tidak terduga pada {func.__name__}: {type(e).__name__}: {e}")
                    raise
            
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator

class Trade(Base):
    """Trade record with support for large Telegram user IDs (BigInteger)"""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(BigInteger, nullable=False)
    ticker = Column(String(20), nullable=False)
    signal_type = Column(String(10), nullable=False)
    signal_source = Column(String(10), default='auto')
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    take_profit = Column(Float, nullable=False)
    spread = Column(Float)
    estimated_pl = Column(Float)
    actual_pl = Column(Float)
    exit_price = Column(Float)
    status = Column(String(20), default='OPEN')
    signal_time = Column(DateTime, default=datetime.utcnow)
    close_time = Column(DateTime)
    timeframe = Column(String(10))
    result = Column(String(10))
    signal_quality_id = Column(Integer, nullable=True)
    
class SignalLog(Base):
    """Signal log with support for large Telegram user IDs (BigInteger)"""
    __tablename__ = 'signal_logs'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(BigInteger, nullable=False)
    ticker = Column(String(20), nullable=False)
    signal_type = Column(String(10), nullable=False)
    signal_source = Column(String(10), default='auto')
    entry_price = Column(Float, nullable=False)
    indicators = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    accepted = Column(Boolean, default=False)
    rejection_reason = Column(String(255))

class Position(Base):
    """Position tracking with support for large Telegram user IDs (BigInteger)"""
    __tablename__ = 'positions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(BigInteger, nullable=False)
    trade_id = Column(Integer, nullable=False)
    ticker = Column(String(20), nullable=False)
    signal_type = Column(String(10), nullable=False)
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    take_profit = Column(Float, nullable=False)
    current_price = Column(Float)
    unrealized_pl = Column(Float)
    status = Column(String(20), default='ACTIVE')
    opened_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime)
    original_sl = Column(Float)
    sl_adjustment_count = Column(Integer, default=0)
    max_profit_reached = Column(Float, default=0.0)
    last_price_update = Column(DateTime)
    signal_quality_id = Column(Integer, nullable=True)

class Performance(Base):
    __tablename__ = 'performance'
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, default=datetime.utcnow)
    total_trades = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    total_pl = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    equity = Column(Float, default=0.0)

class CandleData(Base):
    __tablename__ = 'candle_data'
    
    id = Column(Integer, primary_key=True)
    timeframe = Column(String(3), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, default=0)
    is_partial = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    """Database manager with connection pooling and rollback safety.
    
    Connection Pooling:
    - Uses SQLAlchemy QueuePool with configurable pool_size and max_overflow
    - pool_pre_ping ensures connections are valid before use
    - Pool monitoring via get_pool_status()
    
    Rollback Safety:
    - Per-operation rollback guarantees via try/except/finally
    - Safe session closure in finally blocks
    - transaction_scope() context manager for atomic operations
    """
    def __init__(self, db_path='data/bot.db', database_url=''):
        """Initialize database with PostgreSQL or SQLite support
        
        Args:
            db_path: Path to SQLite database (used if database_url is not provided)
            database_url: PostgreSQL connection URL (e.g., postgresql://user:pass@host:port/dbname)
        """
        self.is_postgres = False
        self.engine = None
        self.Session = None
        self._pool_stats = {
            'checkouts': 0,
            'checkins': 0,
            'connects': 0,
            'disconnects': 0,
            'overflow_connections': 0,
            'timeout_errors': 0,
            'high_utilization_warnings': 0,
            'total_wait_time_ms': 0.0,
            'max_wait_time_ms': 0.0,
            'checkout_attempts': 0
        }
        self._pool_stats_lock = threading.Lock()
        self._last_checkout_start = threading.local()
        
        try:
            if database_url and database_url.strip():
                logger.info(f"Using PostgreSQL from DATABASE_URL")
                db_url = database_url.strip()
                self.is_postgres = db_url.startswith('postgresql://') or db_url.startswith('postgres://')
                
                engine_kwargs = {
                    'echo': False,
                    'pool_pre_ping': POOL_PRE_PING,
                    'pool_recycle': POOL_RECYCLE,
                    'pool_size': POOL_SIZE,
                    'max_overflow': MAX_OVERFLOW,
                    'pool_timeout': POOL_TIMEOUT,
                    'poolclass': QueuePool
                }
                
                if not self.is_postgres:
                    engine_kwargs['connect_args'] = {
                        'check_same_thread': False,
                        'timeout': 30.0
                    }
                
                self.engine = create_engine(db_url, **engine_kwargs)
                logger.info(f"✅ Database engine created: {'PostgreSQL' if self.is_postgres else 'SQLite (from URL)'}")
                logger.info(f"   Pool config: size={POOL_SIZE}, max_overflow={MAX_OVERFLOW}, timeout={POOL_TIMEOUT}s")
                
            else:
                if not db_path or not isinstance(db_path, str):
                    raise ValueError(f"Invalid db_path: {db_path}")
                
                db_dir = os.path.dirname(db_path)
                if db_dir:
                    os.makedirs(db_dir, exist_ok=True)
                
                logger.info(f"Using SQLite database: {db_path}")
                
                self.engine = create_engine(
                    f'sqlite:///{db_path}',
                    connect_args={
                        'check_same_thread': False,
                        'timeout': 30.0
                    },
                    echo=False,
                    pool_pre_ping=POOL_PRE_PING,
                    pool_recycle=POOL_RECYCLE
                )
            
            self._setup_pool_event_listeners()
            
            self._configure_database()
            
            Base.metadata.create_all(self.engine)
            
            self._migrate_database()
            
            self.Session = scoped_session(sessionmaker(bind=self.engine))
            
            logger.info("✅ Database initialized successfully")
            
        except ValueError as e:
            logger.error(f"Configuration error during database initialization: {e}")
            raise DatabaseError(f"Database configuration failed: {e}")
        except OperationalError as e:
            logger.error(f"Operational error during database initialization (connection/timeout): {e}")
            raise DatabaseError(f"Database connection failed: {e}")
        except IntegrityError as e:
            logger.error(f"Integrity error during database initialization: {e}")
            raise DatabaseError(f"Database integrity error: {e}")
        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemy error during database initialization: {e}")
            raise DatabaseError(f"Database initialization failed: {e}")
        except OSError as e:
            logger.error(f"OS error during database initialization (file/directory): {e}")
            raise DatabaseError(f"Database file system error: {e}")
        except (IOError, RuntimeError, TypeError) as e:
            logger.error(f"Unexpected error during database initialization: {type(e).__name__}: {e}")
            raise DatabaseError(f"Database initialization failed: {e}")
    
    def _setup_pool_event_listeners(self):
        """Setup event listeners for connection pool monitoring."""
        db_manager = self
        
        @event.listens_for(self.engine, 'checkout')
        def on_checkout(dbapi_conn, connection_record, connection_proxy):
            checkout_start = getattr(db_manager._last_checkout_start, 'start_time', None)
            wait_time_ms = 0.0
            if checkout_start is not None:
                wait_time_ms = (time.time() - checkout_start) * 1000
            
            with db_manager._pool_stats_lock:
                db_manager._pool_stats['checkouts'] += 1
                if wait_time_ms > 0:
                    db_manager._pool_stats['total_wait_time_ms'] += wait_time_ms
                    if wait_time_ms > db_manager._pool_stats['max_wait_time_ms']:
                        db_manager._pool_stats['max_wait_time_ms'] = wait_time_ms
            
            db_manager._check_and_warn_high_utilization()
        
        @event.listens_for(self.engine, 'checkin')
        def on_checkin(dbapi_conn, connection_record):
            with db_manager._pool_stats_lock:
                db_manager._pool_stats['checkins'] += 1
        
        @event.listens_for(self.engine, 'connect')
        def on_connect(dbapi_conn, connection_record):
            with db_manager._pool_stats_lock:
                db_manager._pool_stats['connects'] += 1
        
        @event.listens_for(self.engine, 'close')
        def on_close(dbapi_conn, connection_record):
            with db_manager._pool_stats_lock:
                db_manager._pool_stats['disconnects'] += 1
        
        logger.debug("Pool event listeners configured")
    
    def _check_and_warn_high_utilization(self):
        """Check pool utilization and log warning if above threshold."""
        try:
            if self.engine is None:
                return
            pool = self.engine.pool
            if pool is None:
                return
            if hasattr(pool, 'checkedout') and hasattr(pool, 'size'):
                checked_out = pool.checkedout()  # type: ignore[union-attr]
                max_connections = pool.size() + MAX_OVERFLOW  # type: ignore[union-attr]
                if max_connections > 0:
                    utilization = (checked_out / max_connections) * 100
                    if utilization >= POOL_HIGH_UTILIZATION_THRESHOLD:
                        with self._pool_stats_lock:
                            self._pool_stats['high_utilization_warnings'] += 1
                        logger.warning(
                            f"⚠️ High pool utilization: {utilization:.1f}% "
                            f"(checked_out={checked_out}, max={max_connections})"
                        )
        except (AttributeError, TypeError, ZeroDivisionError) as e:
            logger.debug(f"Error checking pool utilization: {e}")
    
    def get_pool_status(self) -> Dict:
        """Get current connection pool status.
        
        Returns:
            Dict with pool statistics and current state
        """
        if self.engine is None:
            raise DatabaseError("Engine not initialized")
        pool = self.engine.pool
        if pool is None:
            raise DatabaseError("Connection pool not available")
        
        with self._pool_stats_lock:
            stats = self._pool_stats.copy()
        
        status = {
            'pool_size': pool.size() if hasattr(pool, 'size') else POOL_SIZE,  # type: ignore[union-attr]
            'checked_in': pool.checkedin() if hasattr(pool, 'checkedin') else 'N/A',  # type: ignore[union-attr]
            'checked_out': pool.checkedout() if hasattr(pool, 'checkedout') else 'N/A',  # type: ignore[union-attr]
            'overflow': pool.overflow() if hasattr(pool, 'overflow') else 'N/A',  # type: ignore[union-attr]
            'max_overflow': MAX_OVERFLOW,
            'pool_timeout': POOL_TIMEOUT,
            'total_checkouts': stats['checkouts'],
            'total_checkins': stats['checkins'],
            'total_connects': stats['connects'],
            'total_disconnects': stats['disconnects'],
            'is_postgres': self.is_postgres,
            'timeout_errors': stats['timeout_errors'],
            'high_utilization_warnings': stats['high_utilization_warnings'],
            'total_wait_time_ms': round(stats['total_wait_time_ms'], 2),
            'max_wait_time_ms': round(stats['max_wait_time_ms'], 2),
            'avg_wait_time_ms': round(stats['total_wait_time_ms'] / max(stats['checkout_attempts'], 1), 2)
        }
        
        active = stats['checkouts'] - stats['checkins']
        status['estimated_active_connections'] = max(0, active)
        
        if hasattr(pool, 'checkedout') and hasattr(pool, 'size'):
            pool_size = pool.size()  # type: ignore[union-attr]
            utilization = pool.checkedout() / (pool_size + MAX_OVERFLOW) * 100 if pool_size > 0 else 0  # type: ignore[union-attr]
            status['pool_utilization_percent'] = round(utilization, 1)
        
        return status
    
    def log_pool_status(self, level: str = 'info'):
        """Log current pool status for monitoring.
        
        Args:
            level: Log level - 'info', 'warning', or 'error'
        """
        status = self.get_pool_status()
        message = (
            f"Pool Status: checked_in={status['checked_in']}, "
            f"checked_out={status['checked_out']}, "
            f"overflow={status['overflow']}, "
            f"utilization={status.get('pool_utilization_percent', 'N/A')}%, "
            f"timeouts={status['timeout_errors']}, "
            f"avg_wait={status['avg_wait_time_ms']}ms"
        )
        
        if level == 'warning':
            logger.warning(message)
        elif level == 'error':
            logger.error(message)
        else:
            logger.info(message)
    
    def check_pool_health(self) -> Dict:
        """Perform periodic pool health check.
        
        Returns:
            Dict with health status and recommendations
        """
        status = self.get_pool_status()
        health = {
            'healthy': True,
            'status': status,
            'warnings': [],
            'recommendations': []
        }
        
        utilization = status.get('pool_utilization_percent', 0)
        if utilization >= POOL_HIGH_UTILIZATION_THRESHOLD:
            health['healthy'] = False
            health['warnings'].append(f"High pool utilization: {utilization}%")
            health['recommendations'].append("Consider increasing pool_size or max_overflow")
        
        if status['timeout_errors'] > 0:
            health['warnings'].append(f"Pool timeout errors occurred: {status['timeout_errors']}")
            if status['timeout_errors'] > 5:
                health['healthy'] = False
                health['recommendations'].append("Investigate connection leaks or increase pool timeout")
        
        avg_wait = status['avg_wait_time_ms']
        if avg_wait > 1000:
            health['warnings'].append(f"High average wait time: {avg_wait}ms")
            health['recommendations'].append("Pool may be undersized for current load")
        
        max_wait = status['max_wait_time_ms']
        if max_wait > 5000:
            health['warnings'].append(f"Very high max wait time: {max_wait}ms")
        
        checked_out = status.get('checked_out', 0)
        if isinstance(checked_out, int) and checked_out == status['pool_size'] + status['max_overflow']:
            health['healthy'] = False
            health['warnings'].append("Pool is at maximum capacity")
            health['recommendations'].append("All connections in use, requests may timeout")
        
        if health['warnings']:
            logger.warning(f"Pool health check warnings: {health['warnings']}")
        else:
            logger.debug("Pool health check: OK")
        
        return health
    
    def _configure_database(self):
        """Configure database with proper settings (SQLite only)"""
        if self.is_postgres:
            logger.info("PostgreSQL detected - skipping SQLite-specific configuration")
            return
        
        if self.engine is None:
            raise DatabaseError("Engine not initialized")
            
        try:
            with self.engine.connect() as conn:
                conn.execute(text('PRAGMA journal_mode=WAL'))
                conn.execute(text('PRAGMA synchronous=NORMAL'))
                conn.execute(text('PRAGMA temp_store=MEMORY'))
                conn.execute(text('PRAGMA mmap_size=30000000000'))
                conn.execute(text('PRAGMA page_size=4096'))
                conn.commit()
                logger.debug("SQLite configuration applied successfully")
        except OperationalError as e:
            logger.error(f"Operational error configuring SQLite database: {e}")
            raise
        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemy error configuring database: {e}")
            raise
        except (ValueError, TypeError, IOError, RuntimeError) as e:
            logger.error(f"Unexpected error configuring database: {type(e).__name__}: {e}")
            raise
    
    @retry_on_db_error(max_retries=3, initial_delay=0.1)
    def _migrate_database(self):
        """Auto-migrate database schema with error handling and validation"""
        logger.info("Checking database schema migrations...")
        
        if self.engine is None:
            raise DatabaseError("Engine not initialized")
        
        try:
            with self.engine.connect() as conn:
                try:
                    self._migrate_trades_table(conn)
                except (OperationalError, IntegrityError, SQLAlchemyError) as e:
                    logger.error(f"Migration error on trades table: {type(e).__name__}: {e}")
                    raise DatabaseError(f"Trades table migration failed: {e}")
                
                try:
                    self._migrate_signal_logs_table(conn)
                except (OperationalError, IntegrityError, SQLAlchemyError) as e:
                    logger.error(f"Migration error on signal_logs table: {type(e).__name__}: {e}")
                    raise DatabaseError(f"Signal logs table migration failed: {e}")
                
                try:
                    self._migrate_positions_table(conn)
                except (OperationalError, IntegrityError, SQLAlchemyError) as e:
                    logger.error(f"Migration error on positions table: {type(e).__name__}: {e}")
                    raise DatabaseError(f"Positions table migration failed: {e}")
                
                try:
                    self._migrate_candle_data_table(conn)
                except (OperationalError, IntegrityError, SQLAlchemyError) as e:
                    logger.error(f"Migration error on candle_data table: {type(e).__name__}: {e}")
                    raise DatabaseError(f"Candle data table migration failed: {e}")
                
            logger.info("✅ Database migrations completed successfully")
        
        except DatabaseError:
            raise
        except OperationalError as e:
            logger.error(f"Operational error during database migration: {e}")
            raise DatabaseError(f"Migration failed (connection/lock issue): {e}")
        except IntegrityError as e:
            logger.error(f"Integrity error during database migration: {e}")
            raise DatabaseError(f"Migration failed (data integrity issue): {e}")
        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemy error during database migration: {e}")
            raise DatabaseError(f"Migration failed: {e}")
        except (IOError, RuntimeError, TypeError, ValueError) as e:
            logger.error(f"Unexpected error during database migration: {type(e).__name__}: {e}")
            raise DatabaseError(f"Migration failed: {e}")
    
    def _migrate_trades_table(self, conn):
        """Migrate trades table schema - convert user_id to BIGINT for large Telegram IDs"""
        try:
            if self.is_postgres:
                result = conn.execute(text("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'trades'
                """))
                columns = [row[0] for row in result]
            else:
                result = conn.execute(text("PRAGMA table_info(trades)"))
                columns = [row[1] for row in result]
                
            if 'signal_source' not in columns:
                conn.execute(text("ALTER TABLE trades ADD COLUMN signal_source VARCHAR(10) DEFAULT 'auto'"))
                conn.commit()
                logger.info("✅ Added signal_source column to trades table")
            
            if 'user_id' not in columns:
                conn.execute(text("ALTER TABLE trades ADD COLUMN user_id BIGINT DEFAULT 0"))
                conn.commit()
                logger.info("✅ Added user_id column (BIGINT) to trades table")
            else:
                try:
                    if self.is_postgres:
                        conn.execute(text("""
                            ALTER TABLE trades 
                            ALTER COLUMN user_id TYPE BIGINT
                        """))
                    else:
                        result = conn.execute(text("PRAGMA table_info(trades)"))
                        columns = {row[1]: row[2] for row in result}
                        if 'user_id' in columns and columns['user_id'] != 'BIGINT':
                            logger.info("Migrating user_id from INTEGER to support larger Telegram IDs...")
                            conn.execute(text("ALTER TABLE trades ADD COLUMN user_id_new BIGINT"))
                            conn.execute(text("UPDATE trades SET user_id_new = user_id WHERE user_id IS NOT NULL"))
                            conn.execute(text("ALTER TABLE trades DROP COLUMN user_id"))
                            conn.execute(text("ALTER TABLE trades RENAME COLUMN user_id_new TO user_id"))
                            logger.info("✅ Migrated user_id to BIGINT")
                    conn.commit()
                except (OperationalError, IntegrityError, SQLAlchemyError, ValueError) as e:
                    logger.debug(f"Column type migration info: {e}")
                
        except (OperationalError, IntegrityError, SQLAlchemyError) as e:
            logger.error(f"Error migrating trades table: {e}")
            raise
    
    def _migrate_signal_logs_table(self, conn):
        """Migrate signal_logs table schema - convert user_id to BIGINT"""
        try:
            if self.is_postgres:
                result = conn.execute(text("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'signal_logs'
                """))
                columns = [row[0] for row in result]
            else:
                result = conn.execute(text("PRAGMA table_info(signal_logs)"))
                columns = [row[1] for row in result]
            
            if 'signal_source' not in columns:
                conn.execute(text("ALTER TABLE signal_logs ADD COLUMN signal_source VARCHAR(10) DEFAULT 'auto'"))
                conn.commit()
                logger.info("✅ Added signal_source column to signal_logs table")
            
            if 'user_id' not in columns:
                conn.execute(text("ALTER TABLE signal_logs ADD COLUMN user_id BIGINT DEFAULT 0"))
                conn.commit()
                logger.info("✅ Added user_id column (BIGINT) to signal_logs table")
            else:
                try:
                    if self.is_postgres:
                        conn.execute(text("""
                            ALTER TABLE signal_logs 
                            ALTER COLUMN user_id TYPE BIGINT
                        """))
                    conn.commit()
                except (OperationalError, IntegrityError, SQLAlchemyError, ValueError) as e:
                    logger.debug(f"Column type migration info: {e}")
                
        except (OperationalError, IntegrityError, SQLAlchemyError) as e:
            logger.error(f"Error migrating signal_logs table: {e}")
            raise
    
    def _migrate_positions_table(self, conn):
        """Migrate positions table schema - convert user_id to BIGINT"""
        try:
            if self.is_postgres:
                result = conn.execute(text("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'positions'
                """))
                columns = [row[0] for row in result]
            else:
                result = conn.execute(text("PRAGMA table_info(positions)"))
                columns = [row[1] for row in result]
            
            
            if 'user_id' not in columns:
                conn.execute(text("ALTER TABLE positions ADD COLUMN user_id BIGINT DEFAULT 0"))
                conn.commit()
                logger.info("✅ Added user_id column (BIGINT) to positions table")
            else:
                try:
                    if self.is_postgres:
                        conn.execute(text("""
                            ALTER TABLE positions 
                            ALTER COLUMN user_id TYPE BIGINT
                        """))
                    conn.commit()
                except (OperationalError, IntegrityError, SQLAlchemyError, ValueError) as e:
                    logger.debug(f"Column type migration info: {e}")
            
            if 'original_sl' not in columns:
                conn.execute(text("ALTER TABLE positions ADD COLUMN original_sl REAL"))
                conn.commit()
                conn.execute(text("UPDATE positions SET original_sl = stop_loss WHERE original_sl IS NULL"))
                conn.commit()
                logger.info("✅ Added original_sl column to positions table with backfill")
            
            if 'sl_adjustment_count' not in columns:
                conn.execute(text("ALTER TABLE positions ADD COLUMN sl_adjustment_count INTEGER DEFAULT 0"))
                conn.commit()
                conn.execute(text("UPDATE positions SET sl_adjustment_count = 0 WHERE sl_adjustment_count IS NULL"))
                conn.commit()
                logger.info("✅ Added sl_adjustment_count column to positions table")
            
            if 'max_profit_reached' not in columns:
                conn.execute(text("ALTER TABLE positions ADD COLUMN max_profit_reached REAL DEFAULT 0.0"))
                conn.commit()
                conn.execute(text("UPDATE positions SET max_profit_reached = 0.0 WHERE max_profit_reached IS NULL"))
                conn.commit()
                logger.info("✅ Added max_profit_reached column to positions table")
            
            if 'last_price_update' not in columns:
                conn.execute(text("ALTER TABLE positions ADD COLUMN last_price_update TIMESTAMP"))
                conn.commit()
                
                if self.is_postgres:
                    conn.execute(text("UPDATE positions SET last_price_update = NOW() WHERE last_price_update IS NULL"))
                else:
                    conn.execute(text("UPDATE positions SET last_price_update = datetime('now') WHERE last_price_update IS NULL"))
                
                conn.commit()
                logger.info("✅ Added last_price_update column to positions table")
                
        except (OperationalError, IntegrityError, SQLAlchemyError) as e:
            logger.error(f"Error migrating positions table: {e}")
            raise
    
    def _migrate_candle_data_table(self, conn):
        """Migrate candle_data table schema - ensure is_partial column exists"""
        try:
            if self.is_postgres:
                result = conn.execute(text("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'candle_data'
                """))
                columns = [row[0] for row in result]
            else:
                result = conn.execute(text("PRAGMA table_info(candle_data)"))
                columns = [row[1] for row in result]
            
            if 'is_partial' not in columns:
                conn.execute(text("ALTER TABLE candle_data ADD COLUMN is_partial BOOLEAN DEFAULT FALSE"))
                conn.commit()
                logger.info("✅ Added is_partial column to candle_data table")
            else:
                logger.debug("is_partial column already exists in candle_data table")
                
        except (OperationalError, IntegrityError, SQLAlchemyError) as e:
            logger.error(f"Error migrating candle_data table: {e}")
            raise
    
    def _get_session_with_pool_retry(
        self, 
        max_retries: int = POOL_EXHAUSTED_MAX_RETRIES,
        initial_delay: float = POOL_EXHAUSTED_INITIAL_DELAY
    ):
        """Mendapatkan session dengan retry logic untuk pool exhaustion.
        
        Mengimplementasikan exponential backoff ketika pool habis.
        
        Args:
            max_retries: Jumlah maksimum percobaan retry
            initial_delay: Delay awal dalam detik sebelum retry pertama
            
        Returns:
            Session object untuk operasi database
            
        Raises:
            ConnectionPoolExhausted: Jika pool habis setelah semua retry
            PoolTimeoutError: Jika timeout terjadi dan retry habis
            DatabaseError: Untuk kegagalan pembuatan session lainnya
        """
        delay = initial_delay
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                with self._pool_stats_lock:
                    self._pool_stats['checkout_attempts'] += 1
                
                self._last_checkout_start.start_time = time.time()
                if self.Session is None:
                    raise DatabaseError("Session factory belum diinisialisasi")
                session = self.Session()
                return session
                
            except SATimeoutError as e:
                last_exception = e
                with self._pool_stats_lock:
                    self._pool_stats['timeout_errors'] += 1
                
                self._log_pool_status_on_exhaustion()
                
                if attempt < max_retries:
                    logger.warning(
                        f"⚠️ Pool timeout pada percobaan {attempt + 1}/{max_retries + 1}: {e}. "
                        f"Mencoba ulang dalam {delay:.2f} detik..."
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(
                        f"❌ Pool habis setelah {max_retries + 1} percobaan. "
                        f"Error terakhir: {e}"
                    )
                    self._log_pool_status_on_exhaustion()
                    raise ConnectionPoolExhausted(
                        f"Connection pool habis setelah {max_retries + 1} percobaan. "
                        f"Pool timeout: {POOL_TIMEOUT}s. Pertimbangkan untuk menambah pool_size atau max_overflow."
                    ) from e
                    
            except OperationalError as e:
                last_exception = e
                error_str = str(e).lower()
                is_pool_error = 'timeout' in error_str or 'pool' in error_str
                is_deadlock = _is_deadlock_error(e)
                
                if is_deadlock:
                    with self._pool_stats_lock:
                        self._pool_stats['timeout_errors'] += 1
                    
                    if attempt < max_retries:
                        deadlock_delay = max(delay, DEADLOCK_RETRY_DELAY)
                        logger.warning(
                            f"🔒 Deadlock terdeteksi saat membuat session "
                            f"(percobaan {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Mencoba ulang dalam {deadlock_delay:.2f} detik..."
                        )
                        self.log_pool_status(level='warning')
                        time.sleep(deadlock_delay)
                        delay *= 2
                    else:
                        logger.error(
                            f"❌ Deadlock persisten saat membuat session setelah "
                            f"{max_retries + 1} percobaan: {e}"
                        )
                        self._log_pool_status_on_exhaustion()
                        raise DeadlockError(
                            f"Deadlock tidak dapat diselesaikan setelah {max_retries + 1} percobaan"
                        ) from e
                elif is_pool_error:
                    with self._pool_stats_lock:
                        self._pool_stats['timeout_errors'] += 1
                    
                    if attempt < max_retries:
                        logger.warning(
                            f"⚠️ Error operasional pool pada percobaan {attempt + 1}/{max_retries + 1}: {e}. "
                            f"Mencoba ulang dalam {delay:.2f} detik..."
                        )
                        self.log_pool_status(level='warning')
                        time.sleep(delay)
                        delay *= 2
                    else:
                        logger.error(
                            f"❌ Pool habis (error operasional) setelah {max_retries + 1} percobaan: {e}"
                        )
                        self._log_pool_status_on_exhaustion()
                        raise ConnectionPoolExhausted(
                            f"Connection pool habis (error operasional) setelah {max_retries + 1} percobaan"
                        ) from e
                else:
                    logger.error(f"❌ Error operasional saat membuat session: {e}")
                    self.log_pool_status(level='error')
                    raise
                    
            except (IOError, RuntimeError, TypeError, ValueError) as e:
                logger.error(f"❌ Error tidak terduga saat membuat session database: {type(e).__name__}: {e}")
                self.log_pool_status(level='error')
                raise DatabaseError(f"Gagal membuat session: {e}") from e
        
        if last_exception:
            raise ConnectionPoolExhausted(
                f"Connection pool habis setelah {max_retries + 1} percobaan"
            ) from last_exception
    
    def _log_pool_status_on_exhaustion(self):
        """Log status pool secara detail saat terjadi exhaustion."""
        try:
            status = self.get_pool_status()
            logger.error(
                f"📊 Status Pool saat Exhaustion:\n"
                f"   - Pool size: {status['pool_size']}\n"
                f"   - Checked in: {status['checked_in']}\n"
                f"   - Checked out: {status['checked_out']}\n"
                f"   - Overflow: {status['overflow']}\n"
                f"   - Utilisasi: {status.get('pool_utilization_percent', 'N/A')}%\n"
                f"   - Total timeout errors: {status['timeout_errors']}\n"
                f"   - Waktu tunggu rata-rata: {status['avg_wait_time_ms']}ms\n"
                f"   - Waktu tunggu maksimum: {status['max_wait_time_ms']}ms"
            )
        except (DatabaseError, AttributeError, KeyError) as e:
            logger.error(f"⚠️ Tidak dapat mengambil status pool: {e}")
    
    def get_session(self):
        """Get database session with pool timeout handling and retry logic.
        
        Returns:
            Session object for database operations
            
        Raises:
            ConnectionPoolExhausted: If pool is exhausted after all retries
            DatabaseError: If session creation fails
        """
        return self._get_session_with_pool_retry()
    
    @contextmanager
    def safe_session(self) -> Generator:
        """Context manager untuk penanganan session yang aman dengan rollback dan penutupan terjamin.
        
        Menyediakan jaminan rollback per-operasi via try/except dan penutupan session
        yang aman di blok finally. Termasuk penanganan pool timeout dengan degradasi graceful.
        
        Usage:
            with db.safe_session() as session:
                # operasi database
                session.add(...)
                # auto-commit saat sukses, auto-rollback saat gagal
                
        Raises:
            ConnectionPoolExhausted: Jika pool habis setelah semua retry
            DatabaseError: Untuk kegagalan pembuatan session lainnya
        """
        session = None
        try:
            session = self.get_session()
            if session is None:
                raise DatabaseError("Gagal mendapatkan session database")
            yield session
            session.commit()
        except ConnectionPoolExhausted:
            logger.error("❌ Safe session gagal: connection pool habis")
            raise
        except SATimeoutError as e:
            if session:
                try:
                    session.rollback()
                except (OperationalError, SQLAlchemyError) as rollback_error:
                    logger.error(f"⚠️ Error saat rollback setelah pool timeout: {rollback_error}")
            logger.error(f"❌ Pool timeout pada safe_session: {e}")
            self.log_pool_status(level='error')
            raise PoolTimeoutError(f"Pool timeout saat operasi session: {e}") from e
        except IntegrityError as e:
            if session:
                try:
                    session.rollback()
                except (OperationalError, SQLAlchemyError) as rollback_error:
                    logger.error(f"⚠️ Error saat rollback setelah IntegrityError: {rollback_error}")
            logger.error(f"❌ Error integritas pada safe_session: {e}")
            raise
        except OperationalError as e:
            if session:
                try:
                    session.rollback()
                except (OperationalError, SQLAlchemyError) as rollback_error:
                    logger.error(f"⚠️ Error saat rollback setelah OperationalError: {rollback_error}")
            logger.error(f"❌ Error operasional pada safe_session: {e}")
            raise
        except SQLAlchemyError as e:
            if session:
                try:
                    session.rollback()
                except (OperationalError, SQLAlchemyError) as rollback_error:
                    logger.error(f"⚠️ Error saat rollback setelah SQLAlchemyError: {rollback_error}")
            logger.error(f"❌ Error SQLAlchemy pada safe_session: {e}")
            raise
        except (ValueError, TypeError, IOError, RuntimeError) as e:
            if session:
                try:
                    session.rollback()
                except (OperationalError, SQLAlchemyError) as rollback_error:
                    logger.error(f"⚠️ Error saat rollback: {rollback_error}")
            logger.error(f"❌ Error tidak terduga pada safe_session: {type(e).__name__}: {e}")
            raise
        finally:
            if session:
                try:
                    session.close()
                except (OperationalError, SQLAlchemyError) as close_error:
                    logger.error(f"⚠️ Error saat menutup session: {close_error}")
    
    @contextmanager
    def transaction_scope(self, isolation_level: Optional[str] = None) -> Generator:
        """
        Menyediakan scope transaksional dengan isolasi dan penanganan pool timeout yang proper.
        
        Termasuk degradasi graceful dengan retry logic untuk skenario pool exhaustion.
        Koneksi selalu dikembalikan ke pool di blok finally.
        
        Args:
            isolation_level: Level isolasi opsional ('SERIALIZABLE', 'REPEATABLE READ', 'READ COMMITTED')
        
        Usage:
            with db.transaction_scope() as session:
                # operasi database
                session.add(...)
                # auto-commit saat sukses, auto-rollback saat gagal
                
        Raises:
            ConnectionPoolExhausted: Jika pool habis setelah semua retry
            PoolTimeoutError: Jika timeout terjadi saat operasi session
        """
        session = None
        transaction_exception = None
        
        try:
            session = self.get_session()
            if session is None:
                raise DatabaseError("Gagal mendapatkan session database")
            
            if isolation_level and self.is_postgres:
                session.execute(text(f"SET TRANSACTION ISOLATION LEVEL {isolation_level}"))
            
            yield session
            session.commit()
            
        except ConnectionPoolExhausted as e:
            transaction_exception = e
            logger.error(f"❌ Transaksi gagal: connection pool habis")
            raise
        except SATimeoutError as e:
            transaction_exception = e
            if session:
                try:
                    session.rollback()
                except (OperationalError, SQLAlchemyError) as rollback_error:
                    logger.error(f"⚠️ Error saat rollback setelah pool timeout: {rollback_error}")
            logger.error(f"❌ Transaksi di-rollback karena pool timeout: {e}")
            self.log_pool_status(level='error')
            raise PoolTimeoutError(f"Pool timeout saat transaksi: {e}") from e
        except IntegrityError as e:
            transaction_exception = e
            if session:
                try:
                    session.rollback()
                except (OperationalError, SQLAlchemyError) as rollback_error:
                    logger.error(f"⚠️ Error saat rollback setelah IntegrityError: {rollback_error}")
            logger.error(f"❌ Transaksi di-rollback karena error integritas: {e}")
            raise
        except OperationalError as e:
            transaction_exception = e
            if session:
                try:
                    session.rollback()
                except (OperationalError, SQLAlchemyError) as rollback_error:
                    logger.error(f"⚠️ Error saat rollback setelah OperationalError: {rollback_error}")
            logger.error(f"❌ Transaksi di-rollback karena error operasional: {e}")
            raise
        except SQLAlchemyError as e:
            transaction_exception = e
            if session:
                try:
                    session.rollback()
                except (OperationalError, SQLAlchemyError) as rollback_error:
                    logger.error(f"⚠️ Error saat rollback setelah SQLAlchemyError: {rollback_error}")
            logger.error(f"❌ Transaksi di-rollback karena error SQLAlchemy: {e}")
            raise
        except (ValueError, TypeError, IOError, RuntimeError) as e:
            transaction_exception = e
            if session:
                try:
                    session.rollback()
                except (OperationalError, SQLAlchemyError) as rollback_error:
                    logger.error(f"⚠️ Error saat rollback: {rollback_error}")
            logger.error(f"❌ Transaksi di-rollback: {type(e).__name__}: {e}")
            raise
        finally:
            if session:
                try:
                    session.close()
                except (OperationalError, SQLAlchemyError) as close_error:
                    logger.error(f"⚠️ Error saat menutup session: {close_error}")
                    if transaction_exception is None:
                        raise
    
    @contextmanager
    def serializable_transaction(self) -> Generator:
        """
        Menyediakan scope transaksi serializable untuk operasi pengguna konkuren.
        Mencegah race condition ketika banyak pengguna trading secara bersamaan.
        """
        with _transaction_lock:
            with self.transaction_scope('SERIALIZABLE' if self.is_postgres else None) as session:
                yield session
    
    @contextmanager
    def transaction_with_retry(
        self,
        max_retries: int = TRANSACTION_MAX_RETRIES,
        initial_delay: float = TRANSACTION_INITIAL_DELAY,
        use_savepoint: bool = False,
        isolation_level: Optional[str] = None
    ) -> Generator:
        """
        Context manager untuk transaksi dengan retry logic dan savepoint support.
        
        Menyediakan:
        - Retry otomatis untuk deadlock dan error sementara
        - Savepoint support untuk nested transaction
        - Rollback yang proper pada setiap kegagalan
        - Log detail dalam bahasa Indonesia
        
        Args:
            max_retries: Jumlah maksimum percobaan retry
            initial_delay: Delay awal dalam detik sebelum retry pertama
            use_savepoint: Gunakan savepoint untuk nested transaction
            isolation_level: Level isolasi opsional ('SERIALIZABLE', 'REPEATABLE READ', 'READ COMMITTED')
        
        Usage:
            with db.transaction_with_retry() as session:
                # operasi database
                session.add(...)
                # auto-commit saat sukses, auto-rollback saat gagal
                
        Raises:
            DeadlockError: Jika deadlock tidak dapat diselesaikan setelah semua retry
            ConnectionPoolExhausted: Jika pool habis setelah semua retry
            DatabaseError: Untuk error lainnya
        """
        delay = initial_delay
        last_exception = None
        
        for attempt in range(max_retries + 1):
            session = None
            savepoint = None
            
            try:
                session = self.get_session()
                if session is None:
                    raise DatabaseError("Gagal mendapatkan session database")
                
                if isolation_level and self.is_postgres:
                    session.execute(text(f"SET TRANSACTION ISOLATION LEVEL {isolation_level}"))
                
                if use_savepoint and self.is_postgres:
                    savepoint_name = f"sp_{int(time.time() * 1000)}_{attempt}"
                    session.execute(text(f"SAVEPOINT {savepoint_name}"))
                    savepoint = savepoint_name
                    logger.debug(f"🔖 Savepoint dibuat: {savepoint_name}")
                
                yield session
                
                if savepoint:
                    session.execute(text(f"RELEASE SAVEPOINT {savepoint}"))
                    logger.debug(f"🔖 Savepoint dirilis: {savepoint}")
                
                session.commit()
                logger.debug(f"✅ Transaksi berhasil pada percobaan {attempt + 1}")
                return
                
            except OperationalError as e:
                last_exception = e
                is_deadlock = _is_deadlock_error(e)
                
                if savepoint and session:
                    try:
                        session.execute(text(f"ROLLBACK TO SAVEPOINT {savepoint}"))
                        logger.info(f"🔖 Rollback ke savepoint: {savepoint}")
                    except (OperationalError, SQLAlchemyError) as sp_error:
                        logger.warning(f"⚠️ Gagal rollback ke savepoint: {sp_error}")
                elif session:
                    try:
                        session.rollback()
                    except (OperationalError, SQLAlchemyError) as rb_error:
                        logger.warning(f"⚠️ Error saat rollback: {rb_error}")
                
                if is_deadlock:
                    if attempt < max_retries:
                        deadlock_delay = max(delay, DEADLOCK_RETRY_DELAY)
                        logger.warning(
                            f"🔒 Deadlock terdeteksi (percobaan {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Mencoba ulang dalam {deadlock_delay:.2f} detik..."
                        )
                        time.sleep(deadlock_delay)
                        delay *= 2
                        continue
                    else:
                        logger.error(
                            f"❌ Deadlock persisten setelah {max_retries + 1} percobaan: {e}"
                        )
                        raise DeadlockError(
                            f"Deadlock tidak dapat diselesaikan setelah {max_retries + 1} percobaan"
                        ) from e
                elif attempt < max_retries:
                    logger.warning(
                        f"⚠️ Error operasional (percobaan {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Mencoba ulang dalam {delay:.2f} detik..."
                    )
                    time.sleep(delay)
                    delay *= 2
                    continue
                else:
                    logger.error(f"❌ Error operasional setelah {max_retries + 1} percobaan: {e}")
                    raise
                    
            except SATimeoutError as e:
                last_exception = e
                if session:
                    try:
                        session.rollback()
                    except (OperationalError, SQLAlchemyError) as rb_error:
                        logger.warning(f"⚠️ Error saat rollback setelah timeout: {rb_error}")
                
                if attempt < max_retries:
                    logger.warning(
                        f"⚠️ Pool timeout (percobaan {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Mencoba ulang dalam {delay:.2f} detik..."
                    )
                    self.log_pool_status(level='warning')
                    time.sleep(delay)
                    delay *= 2
                    continue
                else:
                    logger.error(f"❌ Pool timeout setelah {max_retries + 1} percobaan: {e}")
                    self._log_pool_status_on_exhaustion()
                    raise PoolTimeoutError(
                        f"Pool timeout tidak dapat diselesaikan setelah {max_retries + 1} percobaan"
                    ) from e
                    
            except IntegrityError as e:
                if session:
                    try:
                        session.rollback()
                    except (OperationalError, SQLAlchemyError) as rb_error:
                        logger.warning(f"⚠️ Error saat rollback setelah IntegrityError: {rb_error}")
                logger.error(f"❌ Error integritas (tidak dapat di-retry): {e}")
                raise
                
            except SQLAlchemyError as e:
                if session:
                    try:
                        session.rollback()
                    except (OperationalError, SQLAlchemyError) as rb_error:
                        logger.warning(f"⚠️ Error saat rollback setelah SQLAlchemyError: {rb_error}")
                logger.error(f"❌ Error SQLAlchemy: {e}")
                raise
                
            except (ValueError, TypeError, IOError, RuntimeError) as e:
                if session:
                    try:
                        session.rollback()
                    except (OperationalError, SQLAlchemyError) as rb_error:
                        logger.warning(f"⚠️ Error saat rollback: {rb_error}")
                logger.error(f"❌ Error tidak terduga: {type(e).__name__}: {e}")
                raise
                
            finally:
                if session:
                    try:
                        session.close()
                    except (OperationalError, SQLAlchemyError) as close_error:
                        logger.warning(f"⚠️ Error saat menutup session: {close_error}")
        
        if last_exception:
            raise last_exception
    
    def atomic_create_trade(self, session, trade_data: dict) -> Optional[int]:
        """
        Membuat trade secara atomik dengan locking yang proper.
        
        Args:
            session: Session database
            trade_data: Dictionary data trade
            
        Returns:
            Trade ID jika berhasil, None jika gagal
        """
        try:
            from bot.database import Trade
            
            trade = Trade(**trade_data)
            session.add(trade)
            session.flush()
            trade_id = cast(int, trade.id)
            
            return trade_id
            
        except IntegrityError as e:
            try:
                session.rollback()
            except (OperationalError, SQLAlchemyError) as rollback_error:
                logger.error(f"⚠️ Error saat rollback setelah IntegrityError: {rollback_error}")
            logger.error(f"❌ Error integritas saat membuat trade secara atomik: {e}")
            raise
        except OperationalError as e:
            try:
                session.rollback()
            except (OperationalError, SQLAlchemyError) as rollback_error:
                logger.error(f"⚠️ Error saat rollback setelah OperationalError: {rollback_error}")
            logger.error(f"❌ Error operasional saat membuat trade secara atomik: {e}")
            raise
        except SQLAlchemyError as e:
            try:
                session.rollback()
            except (OperationalError, SQLAlchemyError) as rollback_error:
                logger.error(f"⚠️ Error saat rollback setelah SQLAlchemyError: {rollback_error}")
            logger.error(f"❌ Error SQLAlchemy saat membuat trade secara atomik: {e}")
            raise
        except (ValueError, TypeError, KeyError) as e:
            try:
                session.rollback()
            except (OperationalError, SQLAlchemyError) as rollback_error:
                logger.error(f"⚠️ Error saat rollback: {rollback_error}")
            logger.error(f"❌ Error tidak terduga saat membuat trade secara atomik: {type(e).__name__}: {e}")
            raise
    
    def atomic_create_position(self, session, position_data: dict) -> Optional[int]:
        """
        Membuat posisi secara atomik dengan locking yang proper.
        
        Args:
            session: Session database  
            position_data: Dictionary data posisi
            
        Returns:
            Position ID jika berhasil, None jika gagal
        """
        try:
            from bot.database import Position
            
            position = Position(**position_data)
            session.add(position)
            session.flush()
            position_id = cast(int, position.id)
            
            return position_id
            
        except IntegrityError as e:
            try:
                session.rollback()
            except (OperationalError, SQLAlchemyError) as rollback_error:
                logger.error(f"⚠️ Error saat rollback setelah IntegrityError: {rollback_error}")
            logger.error(f"❌ Error integritas saat membuat posisi secara atomik: {e}")
            raise
        except OperationalError as e:
            try:
                session.rollback()
            except (OperationalError, SQLAlchemyError) as rollback_error:
                logger.error(f"⚠️ Error saat rollback setelah OperationalError: {rollback_error}")
            logger.error(f"❌ Error operasional saat membuat posisi secara atomik: {e}")
            raise
        except SQLAlchemyError as e:
            try:
                session.rollback()
            except (OperationalError, SQLAlchemyError) as rollback_error:
                logger.error(f"⚠️ Error saat rollback setelah SQLAlchemyError: {rollback_error}")
            logger.error(f"❌ Error SQLAlchemy saat membuat posisi secara atomik: {e}")
            raise
        except (ValueError, TypeError, KeyError) as e:
            try:
                session.rollback()
            except (OperationalError, SQLAlchemyError) as rollback_error:
                logger.error(f"⚠️ Error saat rollback: {rollback_error}")
            logger.error(f"❌ Error tidak terduga saat membuat posisi secara atomik: {type(e).__name__}: {e}")
            raise
    
    def atomic_close_position(
        self,
        user_id: int,
        position_id: int,
        trade_id: int,
        exit_price: float,
        actual_pl: float,
        close_time: datetime
    ) -> bool:
        """
        Menutup posisi secara atomik dengan memastikan konsistensi trade dan position.
        
        Menggunakan transaction_with_retry untuk memastikan:
        - Trade dan position diperbarui dalam satu transaksi atomik
        - Rollback otomatis jika salah satu operasi gagal
        - Tidak ada orphaned records (trade CLOSED tapi position masih ACTIVE atau sebaliknya)
        
        Args:
            user_id: ID pengguna Telegram
            position_id: ID posisi yang akan ditutup
            trade_id: ID trade terkait
            exit_price: Harga penutupan
            actual_pl: Profit/Loss aktual
            close_time: Waktu penutupan
            
        Returns:
            True jika berhasil, False jika gagal
            
        Raises:
            OrphanedRecordError: Jika ditemukan inkonsistensi data
            DatabaseError: Untuk error database lainnya
        """
        try:
            with self.transaction_with_retry(
                max_retries=TRANSACTION_MAX_RETRIES,
                use_savepoint=self.is_postgres
            ) as session:
                position = session.query(Position).filter(
                    Position.id == position_id,
                    Position.user_id == user_id
                ).with_for_update().first()
                
                trade = session.query(Trade).filter(
                    Trade.id == trade_id,
                    Trade.user_id == user_id
                ).with_for_update().first()
                
                if not position:
                    logger.warning(
                        f"⚠️ Posisi tidak ditemukan: position_id={position_id}, user_id={user_id}"
                    )
                    return False
                
                if not trade:
                    logger.error(
                        f"❌ Trade tidak ditemukan untuk posisi: trade_id={trade_id}, "
                        f"position_id={position_id}, user_id={user_id}"
                    )
                    raise OrphanedRecordError(
                        f"Posisi orphan terdeteksi: position_id={position_id} "
                        f"tanpa trade terkait trade_id={trade_id}"
                    )
                
                position.status = 'CLOSED'
                position.current_price = exit_price
                position.unrealized_pl = actual_pl
                position.closed_at = close_time
                
                trade.status = 'CLOSED'
                trade.exit_price = exit_price
                trade.actual_pl = actual_pl
                trade.close_time = close_time
                trade.result = 'WIN' if actual_pl >= 0 else 'LOSS'
                
                session.flush()
                
                logger.info(
                    f"✅ Posisi ditutup secara atomik: position_id={position_id}, "
                    f"trade_id={trade_id}, P/L=${actual_pl:.2f}, hasil={trade.result}"
                )
                
                return True
                
        except OrphanedRecordError:
            raise
        except DeadlockError as e:
            logger.error(f"❌ Deadlock saat menutup posisi {position_id}: {e}")
            raise
        except ConnectionPoolExhausted as e:
            logger.error(f"❌ Pool habis saat menutup posisi {position_id}: {e}")
            raise
        except (OperationalError, IntegrityError, SQLAlchemyError) as e:
            logger.error(
                f"❌ Error database saat menutup posisi {position_id}: {type(e).__name__}: {e}"
            )
            raise DatabaseError(f"Gagal menutup posisi secara atomik: {e}") from e
        except (ValueError, TypeError) as e:
            logger.error(f"❌ Error validasi saat menutup posisi {position_id}: {e}")
            raise DatabaseError(f"Error validasi: {e}") from e
    
    def cleanup_orphaned_trades(self) -> Dict[str, int]:
        """
        Membersihkan trade dan posisi orphan dalam database.
        
        Orphaned trades adalah:
        - Trade dengan status OPEN tapi tidak ada posisi ACTIVE yang terkait
        - Posisi dengan status ACTIVE tapi trade terkait sudah CLOSED
        
        Returns:
            Dict dengan jumlah record yang dibersihkan:
            {
                'orphaned_trades_fixed': int,
                'orphaned_positions_fixed': int,
                'mismatched_status_fixed': int
            }
        """
        result = {
            'orphaned_trades_fixed': 0,
            'orphaned_positions_fixed': 0,
            'mismatched_status_fixed': 0
        }
        
        try:
            with self.transaction_with_retry(max_retries=TRANSACTION_MAX_RETRIES) as session:
                orphaned_positions = session.query(Position).filter(
                    Position.status == 'ACTIVE'
                ).all()
                
                for position in orphaned_positions:
                    trade = session.query(Trade).filter(
                        Trade.id == position.trade_id,
                        Trade.user_id == position.user_id
                    ).first()
                    
                    if not trade:
                        logger.warning(
                            f"🔧 Posisi orphan ditemukan (tanpa trade): position_id={position.id}, "
                            f"trade_id={position.trade_id}, user_id={position.user_id}"
                        )
                        position.status = 'ORPHANED'
                        result['orphaned_positions_fixed'] += 1
                    elif trade.status == 'CLOSED' and position.status == 'ACTIVE':
                        logger.warning(
                            f"🔧 Status tidak cocok: position_id={position.id} ACTIVE "
                            f"tapi trade_id={trade.id} CLOSED"
                        )
                        position.status = 'CLOSED'
                        position.closed_at = trade.close_time
                        position.current_price = trade.exit_price
                        position.unrealized_pl = trade.actual_pl
                        result['mismatched_status_fixed'] += 1
                
                open_trades = session.query(Trade).filter(
                    Trade.status == 'OPEN'
                ).all()
                
                for trade in open_trades:
                    position = session.query(Position).filter(
                        Position.trade_id == trade.id,
                        Position.user_id == trade.user_id,
                        Position.status == 'ACTIVE'
                    ).first()
                    
                    if not position:
                        logger.warning(
                            f"🔧 Trade orphan ditemukan (tanpa posisi aktif): trade_id={trade.id}, "
                            f"user_id={trade.user_id}"
                        )
                        trade.status = 'ORPHANED'
                        result['orphaned_trades_fixed'] += 1
                
                session.flush()
                
                total_fixed = sum(result.values())
                if total_fixed > 0:
                    logger.info(
                        f"✅ Pembersihan orphaned records selesai: "
                        f"posisi={result['orphaned_positions_fixed']}, "
                        f"trade={result['orphaned_trades_fixed']}, "
                        f"status mismatch={result['mismatched_status_fixed']}"
                    )
                else:
                    logger.debug("✅ Tidak ada orphaned records ditemukan")
                
                return result
                
        except (DeadlockError, ConnectionPoolExhausted) as e:
            logger.error(f"❌ Error saat membersihkan orphaned records: {e}")
            raise
        except (OperationalError, IntegrityError, SQLAlchemyError) as e:
            logger.error(f"❌ Error database saat membersihkan orphaned records: {e}")
            raise DatabaseError(f"Gagal membersihkan orphaned records: {e}") from e
    
    def verify_trade_position_consistency(self, user_id: int, trade_id: int) -> Dict:
        """
        Memverifikasi konsistensi antara trade dan posisi terkait.
        
        Args:
            user_id: ID pengguna
            trade_id: ID trade yang akan diverifikasi
            
        Returns:
            Dict dengan status konsistensi:
            {
                'consistent': bool,
                'trade_status': str,
                'position_status': str or None,
                'issues': List[str]
            }
        """
        result = {
            'consistent': True,
            'trade_status': None,
            'position_status': None,
            'issues': []
        }
        
        try:
            with self.safe_session() as session:
                trade = session.query(Trade).filter(
                    Trade.id == trade_id,
                    Trade.user_id == user_id
                ).first()
                
                if not trade:
                    result['consistent'] = False
                    result['issues'].append(f"Trade tidak ditemukan: trade_id={trade_id}")
                    return result
                
                result['trade_status'] = trade.status
                
                position = session.query(Position).filter(
                    Position.trade_id == trade_id,
                    Position.user_id == user_id
                ).first()
                
                if not position:
                    result['consistent'] = False
                    result['issues'].append(
                        f"Posisi tidak ditemukan untuk trade_id={trade_id}"
                    )
                    return result
                
                result['position_status'] = position.status
                
                if trade.status == 'OPEN' and position.status != 'ACTIVE':
                    result['consistent'] = False
                    result['issues'].append(
                        f"Trade OPEN tapi posisi {position.status}"
                    )
                elif trade.status == 'CLOSED' and position.status not in ['CLOSED', 'ORPHANED']:
                    result['consistent'] = False
                    result['issues'].append(
                        f"Trade CLOSED tapi posisi {position.status}"
                    )
                
                if result['issues']:
                    logger.warning(
                        f"⚠️ Inkonsistensi terdeteksi untuk trade_id={trade_id}: "
                        f"{', '.join(result['issues'])}"
                    )
                
                return result
                
        except (OperationalError, SQLAlchemyError) as e:
            logger.error(f"❌ Error saat verifikasi konsistensi: {e}")
            result['consistent'] = False
            result['issues'].append(f"Error database: {e}")
            return result
    
    def clear_historical_data(self) -> Dict[str, Any]:
        """Hapus semua data history trading untuk fresh start di Koyeb deployment
        
        Returns:
            dict: Statistik data yang dihapus
        """
        result = {
            'trades_deleted': 0,
            'positions_deleted': 0,
            'signal_logs_deleted': 0,
            'performance_deleted': 0,
            'success': False,
            'message': ''
        }
        
        if not self.Session:
            result['message'] = 'Session not initialized'
            return result
        
        session = None
        try:
            session = self.Session()
            
            # Hapus semua trades
            trades_deleted = session.query(Trade).delete()
            result['trades_deleted'] = trades_deleted
            logger.info(f"🗑️  Dihapus {trades_deleted} trade records")
            
            # Hapus semua positions
            positions_deleted = session.query(Position).delete()
            result['positions_deleted'] = positions_deleted
            logger.info(f"🗑️  Dihapus {positions_deleted} position records")
            
            # Hapus semua signal logs
            signal_logs_deleted = session.query(SignalLog).delete()
            result['signal_logs_deleted'] = signal_logs_deleted
            logger.info(f"🗑️  Dihapus {signal_logs_deleted} signal log records")
            
            # Hapus semua performance records
            performance_deleted = session.query(Performance).delete()
            result['performance_deleted'] = performance_deleted
            logger.info(f"🗑️  Dihapus {performance_deleted} performance records")
            
            session.commit()
            result['success'] = True
            result['message'] = f'✅ History data cleared: {trades_deleted} trades, {positions_deleted} positions, {signal_logs_deleted} signal logs, {performance_deleted} performance records'
            logger.info(f"✅ Historical data cleared successfully: {result['message']}")
            
            return result
            
        except (OperationalError, SQLAlchemyError, Exception) as e:
            if session:
                session.rollback()
            logger.error(f"❌ Error saat menghapus historical data: {type(e).__name__}: {e}")
            result['success'] = False
            result['message'] = f'Error: {str(e)}'
            return result
        finally:
            if session:
                session.close()
    
    def close(self):
        """Menutup koneksi database dengan error handling dan pool cleanup."""
        try:
            logger.info("🔌 Menutup koneksi database...")
            self.log_pool_status()
            if self.Session is not None:
                self.Session.remove()
            if self.engine is not None:
                self.engine.dispose()
            logger.info("✅ Koneksi database berhasil ditutup")
        except (OperationalError, SQLAlchemyError, AttributeError) as e:
            logger.error(f"❌ Error saat menutup database: {type(e).__name__}: {e}")
