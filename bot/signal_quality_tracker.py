"""
Lightweight signal quality tracker - simplified for Koyeb free tier.
"""
from typing import Dict, Optional, Any, List
from datetime import datetime
from collections import defaultdict
from bot.logger import setup_logger

logger = setup_logger('SignalQualityTracker')


class SignalQualityTracker:
    """Simple signal quality tracker"""
    
    def __init__(self, db_manager=None, config=None):
        self.db_manager = db_manager
        self.config = config
        self.signal_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {'wins': 0, 'losses': 0})
        self.recent_signals: List[Dict[str, Any]] = []
    
    def record_signal(self, signal_data: Dict[str, Any]) -> None:
        self.recent_signals.append({
            'signal': signal_data,
            'timestamp': datetime.now(),
            'result': None
        })
        if len(self.recent_signals) > 100:
            self.recent_signals = self.recent_signals[-100:]
    
    def record_result(self, signal_id: str, result: str, pnl: float = 0) -> None:
        rule_name = signal_id.split('_')[0] if '_' in signal_id else 'unknown'
        if result == 'win':
            self.signal_stats[rule_name]['wins'] += 1
        else:
            self.signal_stats[rule_name]['losses'] += 1
    
    def get_win_rate(self, rule_name: str = 'all') -> float:
        if rule_name == 'all':
            total_wins = sum(s['wins'] for s in self.signal_stats.values())
            total_losses = sum(s['losses'] for s in self.signal_stats.values())
        else:
            stats = self.signal_stats.get(rule_name, {'wins': 0, 'losses': 0})
            total_wins = stats['wins']
            total_losses = stats['losses']
        
        total = total_wins + total_losses
        return (total_wins / total * 100) if total > 0 else 0
    
    def get_stats(self) -> Dict[str, Any]:
        total_wins = sum(s['wins'] for s in self.signal_stats.values())
        total_losses = sum(s['losses'] for s in self.signal_stats.values())
        total = total_wins + total_losses
        
        return {
            'total_signals': total,
            'wins': total_wins,
            'losses': total_losses,
            'win_rate': (total_wins / total * 100) if total > 0 else 0,
            'by_rule': dict(self.signal_stats)
        }
