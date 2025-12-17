"""
Lightweight analytics stub - simplified for Koyeb free tier.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
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
    
    def get_recent_trades(self, user_id: Optional[int] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent completed trades for a user.
        
        Args:
            user_id: Optional user ID to filter by
            limit: Maximum number of trades to return
            
        Returns:
            List of trade dictionaries with fields:
            - signal_type, entry_price, exit_price, actual_pl, result, close_time
        """
        if not self.db_manager:
            logger.warning("No db_manager available for get_recent_trades")
            return []
        
        try:
            stats = self.db_manager.get_performance_stats(use_cache=False)
            if stats and 'recent_trades' in stats:
                trades = stats.get('recent_trades', [])
                return trades[:limit] if trades else []
        except Exception as e:
            logger.error(f"Error getting recent trades: {type(e).__name__}: {e}")
        
        return []
    
    def get_trading_performance(self, user_id: Optional[int] = None, days: int = 7) -> Dict[str, Any]:
        """Get trading performance statistics for a time period.
        
        Args:
            user_id: Optional user ID to filter by
            days: Number of days to look back
            
        Returns:
            Dictionary with performance statistics matching MessageFormatter format:
            - total_trades, wins, losses, winrate, total_pl, avg_pl, profit_factor
        """
        result = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'winrate': 0.0,
            'total_pl': 0.0,
            'avg_pl': 0.0,
            'profit_factor': 0.0,
            'period_days': days
        }
        
        if not self.db_manager:
            logger.warning("No db_manager available for get_trading_performance")
            return result
        
        try:
            stats = self.db_manager.get_performance_stats(use_cache=True)
            if stats and 'overall' in stats:
                overall = stats['overall']
                total_trades = overall.get('total_trades', 0)
                wins = overall.get('wins', 0)
                losses = overall.get('losses', 0)
                total_pnl = overall.get('total_pnl', 0.0)
                
                result['total_trades'] = total_trades
                result['wins'] = wins
                result['losses'] = losses
                result['winrate'] = overall.get('win_rate', 0.0)
                result['total_pl'] = total_pnl
                result['avg_pl'] = overall.get('avg_pnl', 0.0)
                
                if losses > 0 and wins > 0:
                    avg_win = total_pnl / wins if wins > 0 and total_pnl > 0 else 0
                    avg_loss = abs(total_pnl) / losses if losses > 0 and total_pnl < 0 else 1
                    result['profit_factor'] = abs(avg_win / avg_loss) if avg_loss > 0 else 0.0
                else:
                    result['profit_factor'] = 0.0
                    
        except Exception as e:
            logger.error(f"Error getting trading performance: {type(e).__name__}: {e}")
        
        return result
