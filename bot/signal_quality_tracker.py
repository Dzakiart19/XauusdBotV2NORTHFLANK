"""
Enhanced Signal Quality Tracker - Lightweight but smart for Koyeb free tier.
Features: Win rate tracking per rule, adaptive blocking, performance analytics.
"""
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field
from bot.logger import setup_logger

logger = setup_logger('SignalQualityTracker')


@dataclass
class SignalPerformance:
    """Track performance metrics for a signal type/rule"""
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    streak: int = 0
    last_result: str = ''
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def total_trades(self) -> int:
        return self.wins + self.losses
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.wins / self.total_trades) * 100
    
    def record_win(self, pnl: float = 0.0) -> None:
        self.wins += 1
        self.total_pnl += pnl
        if pnl > 0:
            self.avg_win = ((self.avg_win * (self.wins - 1)) + pnl) / self.wins
        if self.last_result == 'win':
            self.streak += 1
        else:
            self.streak = 1
        self.last_result = 'win'
        self.last_updated = datetime.now()
    
    def record_loss(self, pnl: float = 0.0) -> None:
        self.losses += 1
        self.total_pnl += pnl
        if pnl < 0:
            loss_count = self.losses
            self.avg_loss = ((self.avg_loss * (loss_count - 1)) + abs(pnl)) / loss_count
        if self.last_result == 'loss':
            self.streak -= 1
        else:
            self.streak = -1
        self.last_result = 'loss'
        self.last_updated = datetime.now()


class SignalQualityTracker:
    """
    Enhanced signal quality tracker with:
    - Win rate tracking per rule type
    - Adaptive blocking based on recent performance
    - Streak detection for momentum
    - Session-based performance (London, NY, Asian)
    """
    
    MIN_TRADES_FOR_ADAPTIVE = 5
    POOR_WIN_RATE_THRESHOLD = 35.0
    GOOD_WIN_RATE_THRESHOLD = 55.0
    MAX_LOSING_STREAK = 3
    MAX_RECENT_SIGNALS = 100
    
    def __init__(self, db_manager=None, config=None):
        self.db_manager = db_manager
        self.config = config
        
        self.rule_performance: Dict[str, SignalPerformance] = defaultdict(SignalPerformance)
        self.session_performance: Dict[str, SignalPerformance] = defaultdict(SignalPerformance)
        self.grade_performance: Dict[str, SignalPerformance] = defaultdict(SignalPerformance)
        
        self.recent_signals: List[Dict[str, Any]] = []
        self.blocked_signals_count: int = 0
        self.passed_signals_count: int = 0
        
        self._hourly_stats: Dict[int, SignalPerformance] = defaultdict(SignalPerformance)
        
        logger.info("SignalQualityTracker initialized with adaptive blocking")
    
    def _get_trading_session(self, hour: int = None) -> str:
        """Determine trading session based on UTC hour"""
        if hour is None:
            hour = datetime.utcnow().hour
        
        if 0 <= hour < 7:
            return 'asian'
        elif 7 <= hour < 12:
            return 'london'
        elif 12 <= hour < 17:
            return 'newyork'
        elif 17 <= hour < 21:
            return 'overlap'
        else:
            return 'late'
    
    def record_signal(self, signal_data: Dict[str, Any]) -> None:
        """Record a new signal for tracking"""
        signal_record = {
            'signal': signal_data,
            'timestamp': datetime.now(),
            'result': None,
            'pnl': 0.0,
            'rule': signal_data.get('rule_type', signal_data.get('signal_source', 'unknown')),
            'grade': signal_data.get('quality_grade', 'B'),
            'session': self._get_trading_session()
        }
        self.recent_signals.append(signal_record)
        
        if len(self.recent_signals) > self.MAX_RECENT_SIGNALS:
            self.recent_signals = self.recent_signals[-self.MAX_RECENT_SIGNALS:]
    
    def record_result(self, signal_id: str, result: str, pnl: float = 0.0) -> None:
        """Record the result of a signal (win/loss)"""
        rule_name = signal_id.split('_')[0] if '_' in signal_id else 'unknown'
        
        perf = self.rule_performance[rule_name]
        if result.lower() == 'win':
            perf.record_win(pnl)
        else:
            perf.record_loss(pnl)
        
        session = self._get_trading_session()
        session_perf = self.session_performance[session]
        if result.lower() == 'win':
            session_perf.record_win(pnl)
        else:
            session_perf.record_loss(pnl)
        
        hour = datetime.utcnow().hour
        hourly_perf = self._hourly_stats[hour]
        if result.lower() == 'win':
            hourly_perf.record_win(pnl)
        else:
            hourly_perf.record_loss(pnl)
        
        for sig in reversed(self.recent_signals):
            if sig.get('result') is None:
                sig['result'] = result
                sig['pnl'] = pnl
                break
        
        logger.debug(f"Recorded {result} for {rule_name}: WR={perf.win_rate:.1f}%, PnL=${pnl:.2f}")
    
    def get_win_rate(self, rule_name: str = 'all') -> float:
        """Get win rate for a specific rule or overall"""
        if rule_name == 'all':
            total_wins = sum(p.wins for p in self.rule_performance.values())
            total_losses = sum(p.losses for p in self.rule_performance.values())
            total = total_wins + total_losses
            return (total_wins / total * 100) if total > 0 else 0.0
        
        perf = self.rule_performance.get(rule_name)
        return perf.win_rate if perf else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        total_wins = sum(p.wins for p in self.rule_performance.values())
        total_losses = sum(p.losses for p in self.rule_performance.values())
        total_pnl = sum(p.total_pnl for p in self.rule_performance.values())
        total = total_wins + total_losses
        
        by_rule = {}
        for rule, perf in self.rule_performance.items():
            by_rule[rule] = {
                'wins': perf.wins,
                'losses': perf.losses,
                'win_rate': perf.win_rate,
                'total_pnl': perf.total_pnl,
                'streak': perf.streak
            }
        
        by_session = {}
        for session, perf in self.session_performance.items():
            by_session[session] = {
                'wins': perf.wins,
                'losses': perf.losses,
                'win_rate': perf.win_rate,
                'total_pnl': perf.total_pnl
            }
        
        return {
            'total_signals': total,
            'wins': total_wins,
            'losses': total_losses,
            'win_rate': (total_wins / total * 100) if total > 0 else 0,
            'total_pnl': total_pnl,
            'by_rule': by_rule,
            'by_session': by_session,
            'signals_blocked': self.blocked_signals_count,
            'signals_passed': self.passed_signals_count
        }
    
    def _check_adaptive_blocking(self, signal_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check if signal should be blocked based on adaptive performance analysis"""
        rule_type = signal_data.get('rule_type', signal_data.get('signal_source', 'unknown'))
        perf = self.rule_performance.get(rule_type)
        
        if perf and perf.total_trades >= self.MIN_TRADES_FOR_ADAPTIVE:
            if perf.win_rate < self.POOR_WIN_RATE_THRESHOLD:
                return (True, f"Rule {rule_type} has poor win rate: {perf.win_rate:.1f}%")
            
            if perf.streak <= -self.MAX_LOSING_STREAK:
                return (True, f"Rule {rule_type} on losing streak: {abs(perf.streak)} losses")
        
        session = self._get_trading_session()
        session_perf = self.session_performance.get(session)
        if session_perf and session_perf.total_trades >= self.MIN_TRADES_FOR_ADAPTIVE:
            if session_perf.win_rate < self.POOR_WIN_RATE_THRESHOLD:
                return (True, f"Session {session} has poor win rate: {session_perf.win_rate:.1f}%")
        
        return (False, None)
    
    def should_block_signal(self, signal_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Check if a signal should be blocked based on quality metrics.
        
        Quality filters:
        - Block signals with confidence < 50%
        - Block signals with grade D (lowest quality)
        - Block signals with confluence_count < 2
        - Block signals during poor performing sessions
        - Block signals from poor performing rules (adaptive)
        
        Returns:
            Tuple of (should_block: bool, reason: str or None)
        """
        if not signal_data:
            self.blocked_signals_count += 1
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
            self.blocked_signals_count += 1
            return (True, reason)
        
        grade = signal_data.get('quality_grade', signal_data.get('grade', 'B'))
        if grade == 'D':
            reason = f"Low quality grade: {grade} (min C)"
            logger.info(f"🚫 Signal blocked: {reason}")
            self.blocked_signals_count += 1
            return (True, reason)
        
        confluence_count = signal_data.get('confluence_count', 0)
        if confluence_count < 2:
            reason = f"Insufficient confluence: {confluence_count} (min 2)"
            logger.info(f"🚫 Signal blocked: {reason}")
            self.blocked_signals_count += 1
            return (True, reason)
        
        weighted_score = signal_data.get('weighted_confluence_score', 0)
        if weighted_score < 2.0 and confidence < 60:
            reason = f"Weak signal: score={weighted_score:.1f}, conf={confidence:.1f}%"
            logger.info(f"🚫 Signal blocked: {reason}")
            self.blocked_signals_count += 1
            return (True, reason)
        
        should_block, adaptive_reason = self._check_adaptive_blocking(signal_data)
        if should_block:
            logger.info(f"🚫 Signal blocked (adaptive): {adaptive_reason}")
            self.blocked_signals_count += 1
            return (True, adaptive_reason)
        
        self.passed_signals_count += 1
        logger.debug(f"✅ Signal passed: grade={grade}, conf={confidence:.1f}%, confluence={confluence_count}")
        return (False, None)
    
    def track_blocked_signal(self, signal_data: Dict[str, Any] = None, reason: str = None,
                             user_id: int = None, blocking_reason: str = None) -> None:
        """Track a blocked signal for analytics"""
        final_reason = blocking_reason or reason or "Unknown"
        logger.debug(f"Signal blocked for user {user_id}: {final_reason}")
    
    def get_overall_stats(self, days: int = None) -> Dict[str, Any]:
        """Get overall signal quality statistics"""
        stats = self.get_stats()
        stats['total_wins'] = stats.get('wins', 0)
        stats['total_losses'] = stats.get('losses', 0)
        stats['overall_accuracy'] = stats.get('win_rate', 0) / 100 if stats.get('win_rate', 0) > 0 else 0
        return stats
    
    def get_best_performing_rules(self, min_trades: int = 5) -> List[Dict[str, Any]]:
        """Get list of best performing rules sorted by win rate"""
        rules = []
        for rule, perf in self.rule_performance.items():
            if perf.total_trades >= min_trades:
                rules.append({
                    'rule': rule,
                    'win_rate': perf.win_rate,
                    'total_trades': perf.total_trades,
                    'total_pnl': perf.total_pnl,
                    'streak': perf.streak
                })
        
        return sorted(rules, key=lambda x: x['win_rate'], reverse=True)
    
    def get_best_trading_hours(self, min_trades: int = 3) -> List[Dict[str, Any]]:
        """Get best performing hours sorted by win rate"""
        hours = []
        for hour, perf in self._hourly_stats.items():
            if perf.total_trades >= min_trades:
                hours.append({
                    'hour': hour,
                    'win_rate': perf.win_rate,
                    'total_trades': perf.total_trades,
                    'total_pnl': perf.total_pnl
                })
        
        return sorted(hours, key=lambda x: x['win_rate'], reverse=True)
    
    def get_performance_summary(self) -> str:
        """Get human-readable performance summary"""
        stats = self.get_stats()
        
        summary = f"""📊 Signal Performance Summary
━━━━━━━━━━━━━━━━━━━━━━━━
Total Signals: {stats['total_signals']}
Win Rate: {stats['win_rate']:.1f}%
Total P/L: ${stats['total_pnl']:.2f}
Signals Blocked: {stats['signals_blocked']}
Signals Passed: {stats['signals_passed']}

By Session:"""
        
        for session, data in stats.get('by_session', {}).items():
            summary += f"\n  {session.title()}: {data['win_rate']:.1f}% ({data['wins']}W/{data['losses']}L)"
        
        return summary
