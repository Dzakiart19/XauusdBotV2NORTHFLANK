"""
Multi-position signal session manager - supports up to MAX_CONCURRENT_POSITIONS per user.
"""
from typing import Dict, Optional, Any, Callable, List, Tuple
from datetime import datetime
from bot.logger import setup_logger

logger = setup_logger('SignalSessionManager')

DEFAULT_MAX_CONCURRENT_POSITIONS = 4


class SignalSessionManager:
    """Multi-position signal session manager - supports multiple active signals per user"""
    
    def __init__(self, max_concurrent_positions: int = DEFAULT_MAX_CONCURRENT_POSITIONS):
        self.active_signals: Dict[int, Dict[Any, Dict[str, Any]]] = {}
        self.signal_history: list = []
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.max_concurrent_positions = max_concurrent_positions
        logger.info(f"✅ SignalSessionManager initialized with max_concurrent_positions={max_concurrent_positions}")
    
    def set_max_concurrent_positions(self, max_positions: int) -> None:
        """Update max concurrent positions setting"""
        self.max_concurrent_positions = max_positions
        logger.info(f"Updated max_concurrent_positions to {max_positions}")
    
    async def can_create_signal(self, user_id: int, signal_source: str = 'auto', 
                                 position_tracker: Any = None, risk_manager: Any = None) -> tuple:
        """Check if user can create a new signal based on max concurrent positions.
        
        Args:
            user_id: The user ID to check
            signal_source: Source of signal ('auto' or 'manual')
            position_tracker: Optional position tracker to check active positions
            risk_manager: Optional risk manager with can_open_position method
            
        Returns:
            tuple: (can_create: bool, block_reason: Optional[str])
        """
        if risk_manager and hasattr(risk_manager, 'can_open_position'):
            try:
                can_open, reason = risk_manager.can_open_position(user_id)
                if not can_open:
                    return (False, reason)
                return (True, None)
            except Exception as e:
                logger.warning(f"Error checking risk_manager.can_open_position: {e}")
        
        if position_tracker:
            try:
                active_count = await position_tracker.get_active_position_count_async(user_id)
                if active_count >= self.max_concurrent_positions:
                    return (False, f"Maksimum {self.max_concurrent_positions} posisi tercapai ({active_count} aktif). Tunggu hingga salah satu TP/SL tercapai.")
                return (True, None)
            except AttributeError:
                has_position = await position_tracker.has_active_position_async(user_id)
                if has_position:
                    try:
                        count = len(position_tracker.active_positions.get(user_id, {}))
                        if count >= self.max_concurrent_positions:
                            return (False, f"Maksimum {self.max_concurrent_positions} posisi tercapai. Tunggu hingga salah satu TP/SL tercapai.")
                    except Exception:
                        pass
                return (True, None)
            except Exception as e:
                logger.warning(f"Error checking position_tracker: {e}")
        
        current_sessions = self.get_active_session_count(user_id)
        if current_sessions >= self.max_concurrent_positions:
            return (False, f"Maksimum {self.max_concurrent_positions} sinyal aktif tercapai. Tunggu hingga salah satu selesai.")
        
        return (True, None)
    
    def get_active_session_count(self, user_id: int) -> int:
        """Get count of active sessions for a user"""
        if user_id not in self.active_signals:
            return 0
        return len(self.active_signals[user_id])
    
    async def create_session_async(self, user_id: int, session_id: Optional[str] = None, 
                                   signal_source: str = 'auto', signal_type: Optional[str] = None,
                                   entry_price: Optional[float] = None, stop_loss: Optional[float] = None,
                                   take_profit: Optional[float] = None, position_id: Optional[int] = None) -> str:
        """Create a new signal session with extended parameters (async version).
        
        Sessions are now keyed by position_id to support multiple concurrent positions.
        """
        signal_data: Dict[str, Any] = {
            'signal_source': signal_source,
            'signal_type': signal_type,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
        if session_id:
            signal_data['session_id'] = session_id
        if position_id:
            signal_data['position_id'] = position_id
        return self._create_session_internal(user_id, signal_data, position_id)
    
    def _create_session_internal(self, user_id: int, signal_data: Dict[str, Any], 
                                  position_id: Optional[int] = None) -> str:
        """Internal session creation method - now supports multiple sessions per user."""
        session_id = signal_data.get('session_id') or f"sig_{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        if user_id not in self.active_signals:
            self.active_signals[user_id] = {}
        
        key = position_id if position_id else session_id
        
        self.active_signals[user_id][key] = {
            'session_id': session_id,
            'signal': signal_data,
            'created_at': datetime.now(),
            'position_id': position_id
        }
        
        logger.info(f"📝 Session created: User:{user_id} Key:{key} SessionID:{session_id}")
        return session_id
    
    def register_event_handler(self, event_type: str, handler: Callable) -> None:
        """Register an event handler for a specific event type"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for event: {event_type}")
    
    def _trigger_event(self, event_type: str, *args, **kwargs) -> None:
        """Trigger all handlers for an event type"""
        handlers = self.event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in event handler for {event_type}: {e}")
    
    def create_session(self, user_id: int, signal_data: Dict[str, Any], position_id: Optional[int] = None) -> str:
        """Sync version of session creation"""
        return self._create_session_internal(user_id, signal_data, position_id)
    
    def get_active_signal(self, user_id: int, position_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get active signal for a user, optionally by position_id"""
        if user_id not in self.active_signals:
            return None
        
        user_sessions = self.active_signals[user_id]
        
        if position_id and position_id in user_sessions:
            return user_sessions[position_id].get('signal')
        
        if user_sessions:
            first_key = next(iter(user_sessions))
            return user_sessions[first_key].get('signal')
        
        return None
    
    def get_all_active_signals(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all active signals for a user"""
        if user_id not in self.active_signals:
            return []
        
        signals = []
        for key, session_data in self.active_signals[user_id].items():
            signal = session_data.get('signal', {}).copy()
            signal['session_key'] = key
            signal['session_id'] = session_data.get('session_id')
            signal['created_at'] = session_data.get('created_at')
            signals.append(signal)
        
        return signals
    
    def get_active_session(self, user_id: int, position_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get active session for a user, optionally by position_id"""
        if user_id not in self.active_signals:
            return None
        
        user_sessions = self.active_signals[user_id]
        
        if position_id and position_id in user_sessions:
            return user_sessions[position_id]
        
        if user_sessions:
            first_key = next(iter(user_sessions))
            return user_sessions[first_key]
        
        return None
    
    def close_session(self, user_id: int, result: str = 'closed', position_id: Optional[int] = None,
                      session_id: Optional[str] = None) -> bool:
        """Close a specific session by position_id, session_id, or the first session if not specified.
        
        Lookup priority:
        1. position_id (if provided and exists as key)
        2. session_id (search through sessions to find matching session_id)
        3. First available session (fallback)
        """
        if user_id not in self.active_signals:
            logger.debug(f"close_session: No sessions for user {user_id}")
            return False
        
        user_sessions = self.active_signals[user_id]
        
        if not user_sessions:
            return False
        
        key_to_remove = None
        
        # Priority 1: Look up by position_id
        if position_id is not None and position_id in user_sessions:
            key_to_remove = position_id
            logger.debug(f"close_session: Found by position_id {position_id}")
        
        # Priority 2: Look up by session_id if position_id not found
        if key_to_remove is None and session_id:
            for key, session_data in user_sessions.items():
                if session_data.get('session_id') == session_id:
                    key_to_remove = key
                    logger.debug(f"close_session: Found by session_id {session_id} -> key {key}")
                    break
        
        # Priority 3: Fallback to first session (only if no specific ID requested)
        if key_to_remove is None and position_id is None and session_id is None and user_sessions:
            key_to_remove = next(iter(user_sessions))
            logger.debug(f"close_session: Using first session, key {key_to_remove}")
        
        if key_to_remove is None:
            logger.warning(f"close_session: No matching session found for user {user_id} "
                          f"(position_id={position_id}, session_id={session_id})")
            return False
        
        session = user_sessions.pop(key_to_remove)
        session['result'] = result
        session['closed_at'] = datetime.now()
        self.signal_history.append(session)
        
        if not self.active_signals[user_id]:
            del self.active_signals[user_id]
        
        logger.info(f"📝 Session closed: User:{user_id} Key:{key_to_remove} Result:{result}")
        self._trigger_event('on_session_end', user_id, session)
        return True
    
    def get_session_count(self) -> int:
        """Get total count of all active sessions across all users"""
        total = 0
        for user_id in self.active_signals:
            total += len(self.active_signals[user_id])
        return total
    
    def get_history_count(self) -> int:
        return len(self.signal_history)
    
    def has_active_session(self, user_id: int) -> bool:
        """Check if user has any active sessions"""
        return user_id in self.active_signals and len(self.active_signals[user_id]) > 0
    
    def can_add_more_sessions(self, user_id: int) -> bool:
        """Check if user can add more sessions based on max limit"""
        current_count = self.get_active_session_count(user_id)
        return current_count < self.max_concurrent_positions
    
    async def end_session(self, user_id: int, result: str = 'closed', position_id: Optional[int] = None,
                          session_id: Optional[str] = None) -> bool:
        """End/close a signal session (async version).
        
        Args:
            user_id: The user ID
            result: Result reason for closing
            position_id: Optional position ID to close specific session
            session_id: Optional session ID as fallback lookup
            
        Returns:
            bool: True if session was closed, False if no session exists
        """
        return self.close_session(user_id, result, position_id, session_id)
    
    async def update_session(self, user_id: int, position_id: Optional[int] = None, 
                             trade_id: Optional[str] = None, **kwargs) -> bool:
        """Update an existing signal session with position/trade info.
        
        Args:
            user_id: The user ID
            position_id: Optional position ID to link
            trade_id: Optional trade ID to link
            **kwargs: Additional fields to update
            
        Returns:
            bool: True if session was updated, False if no session exists
        """
        if user_id not in self.active_signals:
            return False
        
        user_sessions = self.active_signals[user_id]
        if not user_sessions:
            return False
        
        session_key = None
        if position_id and position_id in user_sessions:
            session_key = position_id
        elif user_sessions:
            session_key = next(iter(user_sessions))
        
        if session_key is None:
            return False
        
        session = user_sessions[session_key]
        
        if position_id is not None:
            session['position_id'] = position_id
            if session_key != position_id:
                user_sessions[position_id] = session
                del user_sessions[session_key]
        
        if trade_id is not None:
            session['trade_id'] = trade_id
        
        for key, value in kwargs.items():
            session[key] = value
        
        session['updated_at'] = datetime.now()
        return True
    
    async def clear_all_sessions(self, reason: str = 'manual_clear') -> int:
        """Clear all active signal sessions.
        
        Args:
            reason: Reason for clearing sessions
            
        Returns:
            int: Number of sessions cleared
        """
        count = 0
        
        for user_id in list(self.active_signals.keys()):
            user_sessions = self.active_signals[user_id]
            for key in list(user_sessions.keys()):
                session = user_sessions.pop(key)
                session['result'] = reason
                session['closed_at'] = datetime.now()
                self.signal_history.append(session)
                self._trigger_event('on_session_end', user_id, session)
                count += 1
            
            if not self.active_signals[user_id]:
                del self.active_signals[user_id]
        
        logger.info(f"Cleared {count} active sessions (reason: {reason})")
        return count
    
    def get_recent_history(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent signal history for a user"""
        user_history = [s for s in self.signal_history if s.get('signal', {}).get('user_id') == user_id 
                       or (str(user_id) in str(s.get('session_id', '')))]
        return user_history[-limit:]
