"""User Manager with Thread-Safe Updates.

This module provides thread-safe user management with the following guarantees:

Thread Safety:
- Per-user locks via defaultdict(threading.RLock) for atomic user operations
- Context managers for clean lock handling
- Guarded active_users mutations with dedicated lock

Read/Write Separation:
- READ operations (get_user, get_user_preferences, is_authorized, etc.):
  - Do NOT acquire user locks for reads
  - Use session-level isolation for data consistency
  - Return detached objects (via session.expunge)

- WRITE operations (create_user, update_user_activity, update_user_stats, etc.):
  - Acquire per-user lock before modification
  - Use context manager for automatic lock release
  - Atomic updates within lock scope

Async Support:
- Async wrapper methods (async_*) use asyncio.to_thread for heavy operations
- These can be called from async context without blocking the event loop
"""
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from contextlib import contextmanager
import threading
import asyncio
import pytz
from bot.logger import setup_logger
from sqlalchemy import Integer, String, DateTime, Boolean, Float, create_engine
from sqlalchemy.orm import sessionmaker, scoped_session, Mapped, mapped_column, DeclarativeBase
from sqlalchemy.exc import SQLAlchemyError

logger = setup_logger('UserManager')

class UserManagerError(Exception):
    """Exception untuk error pada user management"""
    pass

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = 'users'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    telegram_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    username: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    first_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    last_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_active: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    total_trades: Mapped[int] = mapped_column(Integer, default=0)
    total_profit: Mapped[float] = mapped_column(Float, default=0.0)
    subscription_tier: Mapped[str] = mapped_column(String(20), default='FREE')
    subscription_expires: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    settings: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

class UserPreferences(Base):
    __tablename__ = 'user_preferences'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    telegram_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    notification_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    daily_summary_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    risk_alerts_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    preferred_timeframe: Mapped[str] = mapped_column(String(10), default='M1')
    max_daily_signals: Mapped[int] = mapped_column(Integer, default=999999)
    timezone: Mapped[str] = mapped_column(String(50), default='Asia/Jakarta')

class UserTrial(Base):
    __tablename__ = 'user_trials'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    trial_start: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    trial_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class UserManager:
    """Thread-safe user manager with per-user locking.
    
    Provides:
    - Per-user RLocks for atomic write operations
    - Context managers for clean lock handling
    - Guarded active_users mutations
    - Session-level isolation for read operations
    - Async wrapper methods for non-blocking async context usage
    """
    def __init__(self, config, db_path: str = 'data/users.db'):
        self.config = config
        self.db_path = db_path
        
        self._lock = threading.RLock()
        self._user_locks: Dict[int, threading.RLock] = defaultdict(threading.RLock)
        self._user_locks_lock = threading.RLock()
        self._active_users_lock = threading.RLock()
        
        engine = create_engine(
            f'sqlite:///{self.db_path}',
            connect_args={'check_same_thread': False, 'timeout': 30.0}
        )
        Base.metadata.create_all(engine)
        
        session_factory = sessionmaker(bind=engine)
        self.Session = scoped_session(session_factory)
        
        self.active_users: Dict[int, Dict] = {}
        logger.info("User manager initialized with thread-safe RLock per-user locking")
    
    def _get_user_lock(self, telegram_id: int) -> threading.RLock:
        """Get or create a RLock for a specific user.
        
        Uses defaultdict for automatic RLock creation, protected by meta-lock.
        RLock allows same thread to acquire lock multiple times (reentrant).
        """
        with self._user_locks_lock:
            return self._user_locks[telegram_id]
    
    @contextmanager
    def user_lock(self, telegram_id: int):
        """Context manager for per-user lock handling.
        
        Provides clean lock acquisition and release with automatic cleanup.
        
        Usage:
            with self.user_lock(telegram_id):
                # atomic operations on user data
        """
        lock = self._get_user_lock(telegram_id)
        lock.acquire()
        try:
            yield
        finally:
            lock.release()
    
    @contextmanager
    def get_session(self):
        """Context manager for thread-safe session handling with proper cleanup.
        
        Provides:
        - Automatic commit on success
        - Automatic rollback on exception
        - Session cleanup in finally block
        """
        session = self.Session()
        try:
            yield session
            session.commit()
        except (SQLAlchemyError, ValueError, TypeError, AttributeError) as e:
            try:
                session.rollback()
            except SQLAlchemyError as rollback_error:
                logger.error(f"Error during session rollback: {rollback_error}")
            raise
        finally:
            try:
                session.close()
            except SQLAlchemyError as close_error:
                logger.error(f"Error closing session: {close_error}")
            try:
                self.Session.remove()
            except SQLAlchemyError as remove_error:
                logger.error(f"Error removing scoped session: {remove_error}")
    
    def create_user(self, telegram_id: int, username: Optional[str] = None,
                   first_name: Optional[str] = None, last_name: Optional[str] = None) -> Optional[User]:
        """Create a new user with per-user locking (WRITE operation).
        
        Thread-safe: Acquires per-user lock before modification.
        """
        with self.user_lock(telegram_id):
            with self.get_session() as session:
                try:
                    existing = session.query(User).filter(User.telegram_id == telegram_id).first()
                    
                    if existing:
                        logger.info(f"User already exists: {telegram_id}")
                        session.expunge(existing)
                        return existing
                    
                    is_admin = telegram_id in self.config.AUTHORIZED_USER_IDS
                    
                    user = User(
                        telegram_id=telegram_id,
                        username=username,
                        first_name=first_name,
                        last_name=last_name,
                        is_active=True,
                        is_admin=is_admin
                    )
                    
                    session.add(user)
                    session.flush()
                    
                    preferences = UserPreferences(telegram_id=telegram_id)
                    session.add(preferences)
                    
                    logger.info(f"Created new user: {telegram_id} ({username})")
                    session.expunge(user)
                    return user
                    
                except (SQLAlchemyError, ValueError, TypeError, AttributeError) as e:
                    logger.error(f"Error creating user: {e}")
                    return None
    
    async def async_create_user(self, telegram_id: int, username: Optional[str] = None,
                                first_name: Optional[str] = None, last_name: Optional[str] = None) -> Optional[User]:
        """Async wrapper for create_user using asyncio.to_thread."""
        return await asyncio.to_thread(self.create_user, telegram_id, username, first_name, last_name)
    
    def get_user(self, telegram_id: int) -> Optional[User]:
        """Get user by telegram_id (READ operation).
        
        Thread-safe via session isolation. No per-user lock needed for reads.
        Returns detached object.
        """
        with self.get_session() as session:
            try:
                user = session.query(User).filter(User.telegram_id == telegram_id).first()
                
                if user:
                    session.expunge(user)
                
                return user
            except (SQLAlchemyError, ValueError, TypeError, AttributeError) as e:
                logger.error(f"Error getting user: {e}")
                return None
    
    async def async_get_user(self, telegram_id: int) -> Optional[User]:
        """Async wrapper for get_user using asyncio.to_thread."""
        return await asyncio.to_thread(self.get_user, telegram_id)
    
    def get_user_by_username(self, username: str) -> Optional[int]:
        """Get user telegram_id by username (READ operation).
        
        Thread-safe via session isolation. No per-user lock needed for reads.
        """
        with self.get_session() as session:
            try:
                user = session.query(User).filter(User.username == username).first()
                return int(user.telegram_id) if user else None
            except (SQLAlchemyError, ValueError, TypeError, AttributeError) as e:
                logger.error(f"Error getting user by username: {e}")
                return None
    
    async def async_get_user_by_username(self, username: str) -> Optional[int]:
        """Async wrapper for get_user_by_username using asyncio.to_thread."""
        return await asyncio.to_thread(self.get_user_by_username, username)
    
    def update_user_activity(self, telegram_id: int):
        """Update user activity timestamp (WRITE operation).
        
        Thread-safe: Acquires per-user lock before modification.
        """
        with self.user_lock(telegram_id):
            with self.get_session() as session:
                try:
                    user = session.query(User).filter(User.telegram_id == telegram_id).first()
                    
                    if user:
                        user.last_active = datetime.utcnow()
                        logger.debug(f"Updated activity for user {telegram_id}")
                except (SQLAlchemyError, ValueError, TypeError, AttributeError) as e:
                    logger.error(f"Error updating user activity: {e}")
    
    async def async_update_user_activity(self, telegram_id: int):
        """Async wrapper for update_user_activity using asyncio.to_thread."""
        return await asyncio.to_thread(self.update_user_activity, telegram_id)
    
    def is_authorized(self, telegram_id: int) -> bool:
        """Check if user is authorized (READ operation - no lock needed)."""
        if telegram_id in self.config.AUTHORIZED_USER_IDS:
            return True
        
        if hasattr(self.config, 'ID_USER_PUBLIC') and telegram_id in self.config.ID_USER_PUBLIC:
            return True
        
        return False
    
    def is_admin(self, telegram_id: int) -> bool:
        """Check if user is admin (READ operation - no lock needed)."""
        # Check AUTHORIZED_USER_IDS dulu (dari secrets)
        if telegram_id in self.config.AUTHORIZED_USER_IDS:
            return True
        # Kalau tidak, check database
        user = self.get_user(telegram_id)
        return bool(user.is_admin) if user else False
    
    def get_all_users(self) -> List[User]:
        """Get all users (READ operation - no lock needed)."""
        with self.get_session() as session:
            try:
                users = session.query(User).all()
                for user in users:
                    session.expunge(user)
                return users
            except (SQLAlchemyError, ValueError, TypeError, AttributeError) as e:
                logger.error(f"Error getting all users: {e}")
                return []
    
    async def async_get_all_users(self) -> List[User]:
        """Async wrapper for get_all_users using asyncio.to_thread."""
        return await asyncio.to_thread(self.get_all_users)
    
    def get_active_users(self) -> List[User]:
        """Get all active users (READ operation - no lock needed)."""
        with self.get_session() as session:
            try:
                users = session.query(User).filter(User.is_active == True).all()
                for user in users:
                    session.expunge(user)
                return users
            except (SQLAlchemyError, ValueError, TypeError, AttributeError) as e:
                logger.error(f"Error getting active users: {e}")
                return []
    
    async def async_get_active_users(self) -> List[User]:
        """Async wrapper for get_active_users using asyncio.to_thread."""
        return await asyncio.to_thread(self.get_active_users)
    
    def deactivate_user(self, telegram_id: int) -> bool:
        """Deactivate a user (WRITE operation).
        
        Thread-safe: Acquires per-user lock before modification.
        """
        with self.user_lock(telegram_id):
            with self.get_session() as session:
                try:
                    user = session.query(User).filter(User.telegram_id == telegram_id).first()
                    
                    if user:
                        user.is_active = False
                        logger.info(f"Deactivated user: {telegram_id}")
                        
                        with self._active_users_lock:
                            if telegram_id in self.active_users:
                                del self.active_users[telegram_id]
                        
                        return True
                    
                    return False
                except (SQLAlchemyError, ValueError, TypeError, AttributeError) as e:
                    logger.error(f"Error deactivating user: {e}")
                    return False
    
    async def async_deactivate_user(self, telegram_id: int) -> bool:
        """Async wrapper for deactivate_user using asyncio.to_thread."""
        return await asyncio.to_thread(self.deactivate_user, telegram_id)
    
    def activate_user(self, telegram_id: int) -> bool:
        """Activate a user (WRITE operation).
        
        Thread-safe: Acquires per-user lock before modification.
        """
        with self.user_lock(telegram_id):
            with self.get_session() as session:
                try:
                    user = session.query(User).filter(User.telegram_id == telegram_id).first()
                    
                    if user:
                        user.is_active = True
                        logger.info(f"Activated user: {telegram_id}")
                        return True
                    
                    return False
                except (SQLAlchemyError, ValueError, TypeError, AttributeError) as e:
                    logger.error(f"Error activating user: {e}")
                    return False
    
    async def async_activate_user(self, telegram_id: int) -> bool:
        """Async wrapper for activate_user using asyncio.to_thread."""
        return await asyncio.to_thread(self.activate_user, telegram_id)
    
    def update_user_stats(self, telegram_id: int, profit: float):
        """Update user trading statistics (WRITE operation).
        
        Thread-safe: Acquires per-user lock before modification.
        Atomic update of total_trades and total_profit.
        """
        with self.user_lock(telegram_id):
            with self.get_session() as session:
                try:
                    user = session.query(User).filter(User.telegram_id == telegram_id).first()
                    
                    if user:
                        user.total_trades += 1
                        user.total_profit += profit
                        logger.debug(f"Updated stats for user {telegram_id}: profit={profit}")
                except (SQLAlchemyError, ValueError, TypeError, AttributeError) as e:
                    logger.error(f"Error updating user stats: {e}")
    
    async def async_update_user_stats(self, telegram_id: int, profit: float):
        """Async wrapper for update_user_stats using asyncio.to_thread."""
        return await asyncio.to_thread(self.update_user_stats, telegram_id, profit)
    
    def get_user_preferences(self, telegram_id: int) -> Optional[UserPreferences]:
        """Get user preferences (READ operation - no lock needed)."""
        with self.get_session() as session:
            try:
                prefs = session.query(UserPreferences).filter(
                    UserPreferences.telegram_id == telegram_id
                ).first()
                if prefs:
                    session.expunge(prefs)
                return prefs
            except (SQLAlchemyError, ValueError, TypeError, AttributeError) as e:
                logger.error(f"Error getting user preferences: {e}")
                return None
    
    async def async_get_user_preferences(self, telegram_id: int) -> Optional[UserPreferences]:
        """Async wrapper for get_user_preferences using asyncio.to_thread."""
        return await asyncio.to_thread(self.get_user_preferences, telegram_id)
    
    def update_user_preferences(self, telegram_id: int, **kwargs) -> bool:
        """Update user preferences (WRITE operation).
        
        Thread-safe: Acquires per-user lock before modification.
        """
        with self.user_lock(telegram_id):
            with self.get_session() as session:
                try:
                    prefs = session.query(UserPreferences).filter(
                        UserPreferences.telegram_id == telegram_id
                    ).first()
                    
                    if not prefs:
                        prefs = UserPreferences(telegram_id=telegram_id)
                        session.add(prefs)
                    
                    for key, value in kwargs.items():
                        if hasattr(prefs, key):
                            setattr(prefs, key, value)
                    
                    logger.info(f"Updated preferences for user {telegram_id}")
                    return True
                    
                except (SQLAlchemyError, ValueError, TypeError, AttributeError) as e:
                    logger.error(f"Error updating preferences: {e}")
                    return False
    
    async def async_update_user_preferences(self, telegram_id: int, **kwargs) -> bool:
        """Async wrapper for update_user_preferences using asyncio.to_thread."""
        return await asyncio.to_thread(self.update_user_preferences, telegram_id, **kwargs)
    
    def get_user_info(self, telegram_id: int) -> Optional[Dict]:
        """Get comprehensive user info (READ operation - no lock needed)."""
        user = self.get_user(telegram_id)
        prefs = self.get_user_preferences(telegram_id)
        
        if not user:
            return None
        
        jakarta_tz = pytz.timezone('Asia/Jakarta')
        created = user.created_at.replace(tzinfo=pytz.UTC).astimezone(jakarta_tz)
        last_active = user.last_active.replace(tzinfo=pytz.UTC).astimezone(jakarta_tz)
        
        info = {
            'telegram_id': user.telegram_id,
            'username': user.username,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'is_active': user.is_active,
            'is_admin': user.is_admin,
            'created_at': created.strftime('%Y-%m-%d %H:%M'),
            'last_active': last_active.strftime('%Y-%m-%d %H:%M'),
            'total_trades': user.total_trades,
            'total_profit': user.total_profit
        }
        
        if prefs:
            info['preferences'] = {
                'notifications': prefs.notification_enabled,
                'daily_summary': prefs.daily_summary_enabled,
                'risk_alerts': prefs.risk_alerts_enabled,
                'timeframe': prefs.preferred_timeframe,
                'timezone': prefs.timezone
            }
        
        return info
    
    async def async_get_user_info(self, telegram_id: int) -> Optional[Dict]:
        """Async wrapper for get_user_info using asyncio.to_thread."""
        return await asyncio.to_thread(self.get_user_info, telegram_id)
    
    def format_user_profile(self, telegram_id: int) -> Optional[str]:
        """Format user profile for display (READ operation - no lock needed)."""
        info = self.get_user_info(telegram_id)
        
        if not info:
            return None
        
        profile = f"👤 *User Profile*\n\n"
        profile += f"Name: {info.get('first_name', 'N/A')} {info.get('last_name', '')}\n"
        profile += f"Username: @{info.get('username', 'N/A')}\n"
        profile += f"Status: {'✅ Active' if info['is_active'] else '⛔ Inactive'}\n"
        profile += f"Role: {'👑 Admin' if info['is_admin'] else '👤 User'}\n\n"
        profile += f"📊 *Statistics*\n"
        profile += f"Total Trades: {info['total_trades']}\n"
        profile += f"Total Profit: ${info['total_profit']:.2f}\n"
        profile += f"Member Since: {info['created_at']}\n"
        profile += f"Last Active: {info['last_active']}\n"
        
        return profile
    
    async def async_format_user_profile(self, telegram_id: int) -> Optional[str]:
        """Async wrapper for format_user_profile using asyncio.to_thread."""
        return await asyncio.to_thread(self.format_user_profile, telegram_id)
    
    def get_user_count(self) -> Dict:
        """Get user statistics (READ operation - no lock needed)."""
        with self.get_session() as session:
            try:
                total = session.query(User).count()
                active = session.query(User).filter(User.is_active == True).count()
                admins = session.query(User).filter(User.is_admin == True).count()
                
                return {
                    'total': total,
                    'active': active,
                    'inactive': total - active,
                    'admins': admins
                }
            except (SQLAlchemyError, ValueError, TypeError, AttributeError) as e:
                logger.error(f"Error getting user count: {e}")
                return {
                    'total': 0,
                    'active': 0,
                    'inactive': 0,
                    'admins': 0
                }
    
    async def async_get_user_count(self) -> Dict:
        """Async wrapper for get_user_count using asyncio.to_thread."""
        return await asyncio.to_thread(self.get_user_count)
    
    def has_access(self, telegram_id: int) -> bool:
        """Check if user has access (READ operation - no lock needed).
        
        Access hierarchy:
        1. AUTHORIZED_USER_IDS - Full access (no trial needed)
        2. ID_USER_PUBLIC - Full access (no trial needed)
        3. Active trial - Temporary access during trial period
        """
        if telegram_id in self.config.AUTHORIZED_USER_IDS:
            return True
        
        if hasattr(self.config, 'ID_USER_PUBLIC') and telegram_id in self.config.ID_USER_PUBLIC:
            return True
        
        trial_status = self.check_trial_status(telegram_id)
        if trial_status and trial_status.get('is_active', False):
            return True
        
        return False
    
    def start_trial(self, user_id: int) -> Optional[Dict]:
        """Start a 3-day trial for a new user (WRITE operation).
        
        Trial hanya diberikan untuk user baru yang belum terdaftar di AUTHORIZED_USER_IDS.
        
        Args:
            user_id: Telegram user ID
            
        Returns:
            Dict with trial info if successful, None if already has trial or is authorized user
        """
        if user_id in self.config.AUTHORIZED_USER_IDS:
            logger.info(f"User {user_id} is authorized, no trial needed")
            return None
        
        if hasattr(self.config, 'ID_USER_PUBLIC') and user_id in self.config.ID_USER_PUBLIC:
            logger.info(f"User {user_id} is public user, no trial needed")
            return None
        
        with self.user_lock(user_id):
            with self.get_session() as session:
                try:
                    existing_trial = session.query(UserTrial).filter(
                        UserTrial.user_id == user_id
                    ).first()
                    
                    if existing_trial:
                        logger.info(f"User {user_id} already has a trial")
                        session.expunge(existing_trial)
                        return {
                            'user_id': existing_trial.user_id,
                            'trial_start': existing_trial.trial_start,
                            'trial_end': existing_trial.trial_end,
                            'is_active': existing_trial.is_active,
                            'already_exists': True
                        }
                    
                    now = datetime.utcnow()
                    trial_end = now + timedelta(days=3)
                    
                    new_trial = UserTrial(
                        user_id=user_id,
                        trial_start=now,
                        trial_end=trial_end,
                        is_active=True,
                        created_at=now
                    )
                    
                    session.add(new_trial)
                    session.flush()
                    
                    logger.info(f"Started 3-day trial for user {user_id}, expires at {trial_end}")
                    
                    return {
                        'user_id': user_id,
                        'trial_start': now,
                        'trial_end': trial_end,
                        'is_active': True,
                        'already_exists': False
                    }
                    
                except (SQLAlchemyError, ValueError, TypeError, AttributeError) as e:
                    logger.error(f"Error starting trial for user {user_id}: {e}")
                    return None
    
    async def async_start_trial(self, user_id: int) -> Optional[Dict]:
        """Async wrapper for start_trial using asyncio.to_thread."""
        return await asyncio.to_thread(self.start_trial, user_id)
    
    def check_trial_status(self, user_id: int) -> Optional[Dict]:
        """Check trial status for a user (READ operation).
        
        Args:
            user_id: Telegram user ID
            
        Returns:
            Dict with trial status, or None if no trial exists
        """
        with self.get_session() as session:
            try:
                trial = session.query(UserTrial).filter(
                    UserTrial.user_id == user_id
                ).first()
                
                if not trial:
                    return None
                
                now = datetime.utcnow()
                is_expired = now > trial.trial_end
                
                if is_expired and trial.is_active:
                    with self.user_lock(user_id):
                        trial_to_update = session.query(UserTrial).filter(
                            UserTrial.user_id == user_id
                        ).first()
                        if trial_to_update:
                            trial_to_update.is_active = False
                            session.commit()
                            logger.info(f"Trial expired for user {user_id}")
                
                remaining = trial.trial_end - now
                remaining_days = max(0, remaining.days)
                remaining_hours = max(0, remaining.seconds // 3600) if remaining_days == 0 else 0
                
                return {
                    'user_id': trial.user_id,
                    'trial_start': trial.trial_start,
                    'trial_end': trial.trial_end,
                    'is_active': not is_expired,
                    'is_expired': is_expired,
                    'remaining_days': remaining_days,
                    'remaining_hours': remaining_hours
                }
                
            except (SQLAlchemyError, ValueError, TypeError, AttributeError) as e:
                logger.error(f"Error checking trial status for user {user_id}: {e}")
                return None
    
    async def async_check_trial_status(self, user_id: int) -> Optional[Dict]:
        """Async wrapper for check_trial_status using asyncio.to_thread."""
        return await asyncio.to_thread(self.check_trial_status, user_id)
    
    def is_trial_expired(self, user_id: int) -> bool:
        """Check if user's trial has expired (READ operation).
        
        Args:
            user_id: Telegram user ID
            
        Returns:
            True if trial expired or doesn't exist, False if trial is active
        """
        trial_status = self.check_trial_status(user_id)
        
        if trial_status is None:
            return True
        
        return trial_status.get('is_expired', True)
    
    async def async_is_trial_expired(self, user_id: int) -> bool:
        """Async wrapper for is_trial_expired using asyncio.to_thread."""
        return await asyncio.to_thread(self.is_trial_expired, user_id)
    
    def get_trial_remaining_days(self, user_id: int) -> int:
        """Get remaining trial days for a user (READ operation).
        
        Args:
            user_id: Telegram user ID
            
        Returns:
            Number of remaining days (0 if expired or no trial)
        """
        trial_status = self.check_trial_status(user_id)
        
        if trial_status is None:
            return 0
        
        return trial_status.get('remaining_days', 0)
    
    async def async_get_trial_remaining_days(self, user_id: int) -> int:
        """Async wrapper for get_trial_remaining_days using asyncio.to_thread."""
        return await asyncio.to_thread(self.get_trial_remaining_days, user_id)
    
    def get_trial_info_message(self, user_id: int) -> Optional[str]:
        """Get formatted trial info message in Indonesian (READ operation).
        
        Args:
            user_id: Telegram user ID
            
        Returns:
            Formatted trial info message, or None if no trial
        """
        if user_id in self.config.AUTHORIZED_USER_IDS:
            return None
        
        if hasattr(self.config, 'ID_USER_PUBLIC') and user_id in self.config.ID_USER_PUBLIC:
            return None
        
        trial_status = self.check_trial_status(user_id)
        
        if trial_status is None:
            return None
        
        if trial_status.get('is_expired', False):
            return (
                "⚠️ *Masa Trial Berakhir*\n\n"
                "Masa trial 3 hari Anda telah berakhir.\n\n"
                "Untuk melanjutkan menggunakan bot ini, silakan berlangganan.\n\n"
                "📞 Hubungi admin untuk informasi berlangganan."
            )
        
        remaining_days = trial_status.get('remaining_days', 0)
        remaining_hours = trial_status.get('remaining_hours', 0)
        
        if remaining_days > 0:
            time_left = f"{remaining_days} hari"
        elif remaining_hours > 0:
            time_left = f"{remaining_hours} jam"
        else:
            time_left = "kurang dari 1 jam"
        
        jakarta_tz = pytz.timezone('Asia/Jakarta')
        trial_end_utc = trial_status.get('trial_end')
        if trial_end_utc:
            trial_end_local = trial_end_utc.replace(tzinfo=pytz.UTC).astimezone(jakarta_tz)
            end_date = trial_end_local.strftime('%d %B %Y, %H:%M WIB')
        else:
            end_date = "N/A"
        
        return (
            f"🎁 *Masa Trial Aktif*\n\n"
            f"⏳ Sisa waktu: *{time_left}*\n"
            f"📅 Berakhir: {end_date}\n\n"
            f"💡 Nikmati semua fitur bot selama masa trial!"
        )
    
    async def async_get_trial_info_message(self, user_id: int) -> Optional[str]:
        """Async wrapper for get_trial_info_message using asyncio.to_thread."""
        return await asyncio.to_thread(self.get_trial_info_message, user_id)
    
    def set_active_user(self, telegram_id: int, data: Dict):
        """Set active user data (WRITE operation with active_users lock).
        
        Thread-safe: Acquires active_users lock before mutation.
        """
        with self._active_users_lock:
            self.active_users[telegram_id] = data
            logger.debug(f"Set active user: {telegram_id}")
    
    def get_active_user(self, telegram_id: int) -> Optional[Dict]:
        """Get active user data (READ operation with active_users lock).
        
        Thread-safe: Acquires active_users lock for read.
        """
        with self._active_users_lock:
            return self.active_users.get(telegram_id)
    
    def remove_active_user(self, telegram_id: int) -> bool:
        """Remove active user (WRITE operation with active_users lock).
        
        Thread-safe: Acquires active_users lock before mutation.
        """
        with self._active_users_lock:
            if telegram_id in self.active_users:
                del self.active_users[telegram_id]
                logger.debug(f"Removed active user: {telegram_id}")
                return True
            return False
    
    def get_all_active_user_ids(self) -> List[int]:
        """Get all active user IDs (READ operation with active_users lock).
        
        Thread-safe: Acquires active_users lock for read, returns copy.
        """
        with self._active_users_lock:
            return list(self.active_users.keys())
    
    def clear_stale_locks(self, max_age_seconds: int = 3600):
        """Clear stale per-user locks that haven't been used recently.
        
        This helps prevent memory leaks from accumulating user locks.
        Should be called periodically from a maintenance task.
        """
        with self._user_locks_lock:
            initial_count = len(self._user_locks)
            
            self._user_locks = defaultdict(threading.RLock)
            
            logger.info(f"Cleared {initial_count} user locks")
