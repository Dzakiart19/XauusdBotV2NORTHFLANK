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
    
    def should_block_signal(self, signal_data: Dict[str, Any]) -> tuple:
        """Check if a signal should be blocked based on quality metrics.
        
        Quality filters:
        - Block signals with confidence < 50%
        - Block signals with grade D (lowest quality)
        - Block signals with confluence_count < 2
        - Block signals during high volatility with low confidence
        
        Returns:
            Tuple of (should_block: bool, reason: str or None)
        """
        if not signal_data:
            return (True, "Empty signal data")
        
        confidence = signal_data.get('confidence_score', signal_data.get('confidence', 0))
        if isinstance(confidence, str):
            try:
                confidence = float(confidence.replace('%', ''))
            except (ValueError, TypeError):
                confidence = 0
        
        if confidence < 50:
            reason = f"Low confidence: {confidence:.1f}% (min 50%)"
            logger.info(f"🚫 Signal blocked: {reason}")
            return (True, reason)
        
        grade = signal_data.get('quality_grade', signal_data.get('grade', 'B'))
        if grade == 'D':
            reason = f"Low quality grade: {grade} (min C)"
            logger.info(f"🚫 Signal blocked: {reason}")
            return (True, reason)
        
        confluence_count = signal_data.get('confluence_count', 0)
        if confluence_count < 2:
            reason = f"Insufficient confluence: {confluence_count} (min 2)"
            logger.info(f"🚫 Signal blocked: {reason}")
            return (True, reason)
        
        weighted_score = signal_data.get('weighted_confluence_score', 0)
        if weighted_score < 2.0 and confidence < 60:
            reason = f"Weak signal: score={weighted_score:.1f}, conf={confidence:.1f}%"
            logger.info(f"🚫 Signal blocked: {reason}")
            return (True, reason)
        
        logger.debug(f"✅ Signal passed quality check: grade={grade}, conf={confidence:.1f}%, confluence={confluence_count}")
        return (False, None)
    
    def track_blocked_signal(self, signal_data: Dict[str, Any] = None, reason: str = None, 
                               user_id: int = None, blocking_reason: str = None) -> None:
        """Track a blocked signal for analytics.
        
        Args:
            signal_data: Signal data dict
            reason: Blocking reason (legacy param)
            user_id: User ID (optional)
            blocking_reason: Blocking reason (alternative param name)
        """
        final_reason = blocking_reason or reason or "Unknown"
        logger.debug(f"Signal blocked for user {user_id}: {final_reason}")
    
    def get_overall_stats(self, days: int = None) -> Dict[str, Any]:
        """Get overall signal quality statistics.
        
        Args:
            days: Number of days to look back (optional, for compatibility)
        """
        stats = self.get_stats()
        stats['total_wins'] = stats.get('wins', 0)
        stats['total_losses'] = stats.get('losses', 0)
        stats['overall_accuracy'] = stats.get('win_rate', 0) / 100 if stats.get('win_rate', 0) > 0 else 0
        return stats
