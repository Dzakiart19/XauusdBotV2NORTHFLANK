"""
Lightweight signal event store - simplified for Koyeb free tier.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import deque
from bot.logger import setup_logger

logger = setup_logger('SignalEventStore')


class SignalEventStore:
    """Simple signal event store for dashboard sync"""
    
    def __init__(self, max_events: int = 100, ttl_seconds: int = 3600, **kwargs):
        self.max_events = max_events
        self.ttl_seconds = ttl_seconds
        self.events: deque = deque(maxlen=max_events)
        self.current_signal: Optional[Dict[str, Any]] = None
    
    def add_signal(self, signal_data: Dict[str, Any]) -> None:
        event = {
            'type': 'signal',
            'data': signal_data,
            'timestamp': datetime.now().isoformat()
        }
        self.events.append(event)
        self.current_signal = signal_data
    
    def add_trade_result(self, trade_data: Dict[str, Any]) -> None:
        event = {
            'type': 'trade_result',
            'data': trade_data,
            'timestamp': datetime.now().isoformat()
        }
        self.events.append(event)
        self.current_signal = None
    
    def add_event(self, event_type: str, data: Dict[str, Any]) -> None:
        event = {
            'type': event_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        self.events.append(event)
    
    def get_current_signal(self) -> Optional[Dict[str, Any]]:
        return self.current_signal
    
    def get_recent_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        return list(self.events)[-limit:]
    
    def get_events_since(self, timestamp: str) -> List[Dict[str, Any]]:
        """Get events since a specific timestamp"""
        try:
            since_dt = datetime.fromisoformat(timestamp)
            return [
                e for e in self.events 
                if datetime.fromisoformat(e['timestamp']) > since_dt
            ]
        except Exception:
            return list(self.events)
    
    def clear(self) -> None:
        self.events.clear()
        self.current_signal = None
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_events': len(self.events),
            'max_events': self.max_events,
            'has_current_signal': self.current_signal is not None
        }
    
    def record_signal(self, user_id_or_data = None, signal_data: Optional[Dict[str, Any]] = None) -> None:
        """Record a signal for a user. Supports both (user_id, data) and (data) signatures."""
        actual_data: Optional[Dict[str, Any]] = None
        actual_user_id: Optional[int] = None
        
        if signal_data is None and isinstance(user_id_or_data, dict):
            actual_data = user_id_or_data
        elif signal_data is not None:
            actual_data = signal_data
            if isinstance(user_id_or_data, int):
                actual_user_id = user_id_or_data
        
        if actual_data:
            if actual_user_id is not None:
                actual_data['user_id'] = actual_user_id
            self.add_signal(actual_data)
