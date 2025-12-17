"""
Lightweight signal session manager - simplified for Koyeb free tier.
"""
from typing import Dict, Optional, Any, Callable, List, Tuple
from datetime import datetime
from bot.logger import setup_logger

logger = setup_logger('SignalSessionManager')


class SignalSessionManager:
    """Simple signal session manager"""
    
    def __init__(self):
        self.active_signals: Dict[int, Dict[str, Any]] = {}
        self.signal_history: list = []
        self.event_handlers: Dict[str, List[Callable]] = {}
    
    async def can_create_signal(self, user_id: int, signal_source: str = 'auto', 
                                 position_tracker: Any = None) -> tuple:
        """Check if user can create a new signal.
        
        Args:
            user_id: The user ID to check
            signal_source: Source of signal ('auto' or 'manual')
            position_tracker: Optional position tracker to check active positions
            
        Returns:
            tuple: (can_create: bool, block_reason: Optional[str])
        """
        if self.has_active_session(user_id):
            return (False, "Anda sudah memiliki sinyal aktif. Tunggu hingga posisi ditutup.")
        
        if position_tracker:
            try:
                has_position = await position_tracker.has_active_position_async(user_id)
                if has_position:
                    return (False, "Anda memiliki posisi aktif. Tunggu hingga TP/SL tercapai.")
            except Exception:
                pass
        
        return (True, None)
    
    async def create_session_async(self, user_id: int, session_id: Optional[str] = None, 
                                   signal_source: str = 'auto', signal_type: Optional[str] = None,
                                   entry_price: Optional[float] = None, stop_loss: Optional[float] = None,
                                   take_profit: Optional[float] = None) -> str:
        """Create a new signal session with extended parameters (async version)."""
        signal_data: Dict[str, Any] = {
            'signal_source': signal_source,
            'signal_type': signal_type,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
        if session_id:
            signal_data['session_id'] = session_id
        return self._create_session_internal(user_id, signal_data)
    
    def _create_session_internal(self, user_id: int, signal_data: Dict[str, Any]) -> str:
        """Internal session creation method."""
        session_id = signal_data.get('session_id') or f"sig_{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.active_signals[user_id] = {
            'session_id': session_id,
            'signal': signal_data,
            'created_at': datetime.now()
        }
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
    
    def create_session(self, user_id: int, signal_data: Dict[str, Any]) -> str:
        session_id = f"sig_{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.active_signals[user_id] = {
            'session_id': session_id,
            'signal': signal_data,
            'created_at': datetime.now()
        }
        return session_id
    
    def get_active_signal(self, user_id: int) -> Optional[Dict[str, Any]]:
        return self.active_signals.get(user_id, {}).get('signal')
    
    def get_active_session(self, user_id: int) -> Optional[Dict[str, Any]]:
        return self.active_signals.get(user_id)
    
    def close_session(self, user_id: int, result: str = 'closed') -> bool:
        if user_id in self.active_signals:
            session = self.active_signals.pop(user_id)
            session['result'] = result
            session['closed_at'] = datetime.now()
            self.signal_history.append(session)
            self._trigger_event('on_session_end', user_id, session)
            return True
        return False
    
    def get_session_count(self) -> int:
        return len(self.active_signals)
    
    def get_history_count(self) -> int:
        return len(self.signal_history)
    
    def has_active_session(self, user_id: int) -> bool:
        return user_id in self.active_signals
    
    async def end_session(self, user_id: int, result: str = 'closed') -> bool:
        """End/close a signal session (async version).
        
        Args:
            user_id: The user ID
            result: Result reason for closing
            
        Returns:
            bool: True if session was closed, False if no session exists
        """
        return self.close_session(user_id, result)
    
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
        
        session = self.active_signals[user_id]
        if position_id is not None:
            session['position_id'] = position_id
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
        count = len(self.active_signals)
        
        for user_id in list(self.active_signals.keys()):
            session = self.active_signals.pop(user_id)
            session['result'] = reason
            session['closed_at'] = datetime.now()
            self.signal_history.append(session)
            self._trigger_event('on_session_end', user_id, session)
        
        logger.info(f"Cleared {count} active sessions (reason: {reason})")
        return count
