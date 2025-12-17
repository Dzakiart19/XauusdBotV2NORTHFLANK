"""
Lightweight analytics stub - simplified for Koyeb free tier.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from bot.logger import setup_logger

logger = setup_logger('Analytics')


class TradingAnalytics:
    """Simple trading analytics"""
    
    def __init__(self, db_manager=None, config=None):
        self.db_manager = db_manager
        self.config = config
    
    def get_performance_summary(self, days: int = 7) -> Dict[str, Any]:
        if self.db_manager:
            try:
                stats = self.db_manager.get_performance_stats()
                if stats:
                    return stats
            except Exception as e:
                logger.error(f"Error getting performance: {e}")
        
        return {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'period_days': days
        }
    
    def get_trade_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        if self.db_manager:
            try:
                trades = self.db_manager.get_recent_trades(limit=limit)
                if trades:
                    return [t.to_dict() if hasattr(t, 'to_dict') else dict(t) for t in trades]
            except Exception as e:
                logger.error(f"Error getting trade history: {e}")
        return []
    
    def get_daily_stats(self) -> Dict[str, Any]:
        return self.get_performance_summary(days=1)
    
    def get_weekly_stats(self) -> Dict[str, Any]:
        return self.get_performance_summary(days=7)
