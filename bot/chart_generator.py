"""
Lightweight chart generator stub - disabled for Koyeb free tier.

This module provides a no-op implementation of ChartGenerator.
All methods return sensible defaults without generating actual charts.
This allows the bot to run on resource-constrained environments.

To enable full chart generation, replace this stub with the full implementation.
"""
from typing import Optional, Any, Dict, Set
from bot.logger import setup_logger

logger = setup_logger('ChartGenerator')


class ChartGenerator:
    """Chart generator stub - charts disabled for lightweight deployment.
    
    This is a stub implementation that provides no-op methods for all
    chart generation functionality. All generate methods return None,
    cleanup methods are no-ops, and stats return empty/zero values.
    
    Attributes:
        config: Configuration object (may be None)
        enabled: Always False for stub implementation
        chart_dir: Default chart directory path (may not exist)
        _pending_charts: Set of chart paths pending deletion (always empty)
    """
    
    def __init__(self, config=None):
        """Initialize stub chart generator.
        
        Args:
            config: Optional configuration object
        """
        self.config = config
        self.enabled = False
        self.chart_dir = "charts"  # Default directory, may not exist
        self._pending_charts: Set[str] = set()  # Track pending charts (always empty in stub)
    
    def generate_chart(self, *args, **kwargs) -> Optional[str]:
        """Generate chart - stub returns None (charts disabled).
        
        Returns:
            None: Charts are disabled in this stub implementation
        """
        return None
    
    async def generate_chart_async(self, *args, **kwargs) -> Optional[str]:
        """Async version of generate_chart - returns None (charts disabled).
        
        Returns:
            None: Charts are disabled in this stub implementation
        """
        return None
    
    async def immediate_delete_chart_async(self, chart_path: str) -> None:
        """Async delete chart - no-op since charts are disabled.
        
        Args:
            chart_path: Path to chart file (ignored)
        """
        pass
    
    def delete_chart(self, chart_path: str) -> bool:
        """Delete a chart file - no-op since charts are disabled.
        
        Args:
            chart_path: Path to chart file (ignored)
            
        Returns:
            bool: Always True (no-op success)
        """
        return True
    
    def cleanup_old_charts(self, days: int = 7) -> int:
        """Cleanup old chart files - no-op since charts are disabled.
        
        Args:
            days: Age threshold in days (ignored)
            
        Returns:
            int: Always 0 (no charts to cleanup)
        """
        return 0
    
    async def cleanup_charts_aggressive_async(self, max_age_minutes: int = 15) -> int:
        """Aggressive async cleanup of old charts - no-op since charts are disabled.
        
        Args:
            max_age_minutes: Maximum age in minutes before cleanup (ignored)
            
        Returns:
            int: Always 0 (no charts cleaned)
        """
        return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chart generator statistics - returns empty/zero stats.
        
        Returns:
            Dict with zero/empty statistics for:
            - tracked_charts: Number of tracked charts (0)
            - lru_cache: LRU cache stats (empty)
            - current_memory_mb: Memory usage in MB (0)
            - cleanup_stats: Cleanup statistics (zeros)
        """
        return {
            'tracked_charts': 0,
            'lru_cache': {
                'size': 0,
                'max_size': 0,
                'hits': 0,
                'misses': 0
            },
            'current_memory_mb': 0.0,
            'cleanup_stats': {
                'total_deleted': 0,
                'last_cleanup': None
            }
        }
    
    def shutdown(self) -> None:
        """Shutdown chart generator - no-op for stub."""
        pass
    
    def cleanup(self) -> None:
        """Cleanup resources - no-op for stub."""
        pass
