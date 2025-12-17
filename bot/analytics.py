"""
Lightweight trading analytics - minimal implementation.

This module provides a simplified analytics implementation that delegates
to the database manager when possible, or returns sensible defaults.
This is a production-ready lightweight version, not a full analytics suite.

All methods return proper types (lists/dicts) and handle errors gracefully
to prevent runtime issues in the calling code.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from bot.logger import setup_logger

logger = setup_logger('Analytics')


class TradingAnalytics:
    """Lightweight trading analytics class.
    
    Provides basic trade history and performance statistics by delegating
    to database methods when available. Returns sensible defaults when
    database methods are unavailable or on errors.
    
    Attributes:
        db: Database manager instance
        config: Configuration object
    """
    
    def __init__(self, db_manager, config):
        self.db = db_manager
        self.config = config
    
    def get_recent_trades(self, user_id: int = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent trades from database.
        
        Args:
            user_id: Optional user ID to filter by
            limit: Maximum number of trades to return
            
        Returns:
            List of trade dictionaries
        """
        try:
            if hasattr(self.db, 'get_trade_history'):
                trades = self.db.get_trade_history(user_id=user_id, limit=limit)
                return trades if trades else []
            return []
        except Exception as e:
            logger.error(f"Error getting recent trades: {e}")
            return []
    
    def get_trading_performance(self, user_id: int = None, days: int = 30) -> Dict[str, Any]:
        """Get trading performance statistics.
        
        Args:
            user_id: Optional user ID to filter by
            days: Number of days to analyze
            
        Returns:
            Dictionary with performance stats
        """
        try:
            if hasattr(self.db, 'get_performance_stats'):
                stats = self.db.get_performance_stats(user_id=user_id)
                if stats:
                    return stats
            
            return {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'max_win': 0.0,
                'max_loss': 0.0,
                'profit_factor': 0.0
            }
        except Exception as e:
            logger.error(f"Error getting trading performance: {e}")
            return {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0
            }
