"""
Lightweight signal session manager - simplified for Koyeb free tier.
"""
from typing import Dict, Optional, Any, Callable, List
from datetime import datetime
from bot.logger import setup_logger

logger = setup_logger('SignalSessionManager')


class SignalSessionManager:
    """Simple signal session manager"""
    
    def __init__(self):
        self.active_signals: Dict[int, Dict[str, Any]] = {}
        self.signal_history: list = []
        self.event_handlers: Dict[str, List[Callable]] = {}
    
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
