"""
Minimal stub version of ChartGenerator - no-op implementation.
All chart generation methods return None without doing anything.
This removes heavy matplotlib/mplfinance dependencies.
"""

import os
from typing import Optional, Tuple
from bot.logger import setup_logger

logger = setup_logger('ChartGenerator')


class ChartError(Exception):
    """Base exception untuk chart generation errors"""
    pass


class ChartGeneratorError(ChartError):
    """Exception untuk chart generator errors"""
    pass


class DataValidationError(ChartError):
    """Chart data validation error"""
    pass


class ChartTimeoutError(ChartError):
    """Exception untuk chart generation timeout"""
    pass


class ChartCleanupError(ChartError):
    """Exception untuk chart cleanup errors"""
    pass


def validate_chart_data(df) -> Tuple[bool, Optional[str]]:
    """Stub: Always returns valid"""
    return True, None


class ChartGenerator:
    """Minimal stub ChartGenerator - all methods are no-op.
    
    This version removes matplotlib/mplfinance dependencies.
    All chart generation methods return None.
    """
    
    def __init__(self, config):
        self.config = config
        self.chart_dir = 'charts'
        os.makedirs(self.chart_dir, exist_ok=True)
        self._shutdown_requested = False
        self._shutdown_in_progress = False
        self._shutdown_complete = False
        self._pending_charts: set = set()
        self._timed_out_tasks: set = set()
        import asyncio
        self._chart_lock = asyncio.Lock()
        logger.info("ChartGenerator stub initialized (no-op mode)")
    
    def get_stats(self) -> dict:
        """Return empty stats for compatibility"""
        return {
            'pending_charts': 0,
            'timed_out_tasks': 0,
            'shutdown_requested': self._shutdown_requested,
            'mode': 'stub'
        }
    
    def _check_dataframe_size(self, df) -> Tuple[bool, Optional[str]]:
        """Stub: Always returns valid"""
        return True, None
    
    def generate_chart(self, df, signal: Optional[dict] = None,
                      timeframe: str = 'M1') -> Optional[str]:
        """Stub: Returns None without generating chart"""
        logger.debug("generate_chart called (stub - no-op)")
        return None
    
    async def generate_chart_async(self, df, signal: Optional[dict] = None,
                                   timeframe: str = 'M1', timeout: Optional[float] = None) -> Optional[str]:
        """Stub: Returns None without generating chart"""
        logger.debug("generate_chart_async called (stub - no-op)")
        return None
    
    def _delete_file_with_retry(self, filepath: str, max_retries: int = 3) -> bool:
        """Stub: Returns True"""
        return True
    
    async def _delete_file_with_retry_async(self, filepath: str, max_retries: int = 3) -> bool:
        """Stub: Returns True"""
        return True
    
    def delete_chart(self, filepath: str) -> bool:
        """Stub: Returns True"""
        logger.debug(f"delete_chart called for {filepath} (stub - no-op)")
        return True
    
    async def delete_chart_async(self, filepath: str) -> bool:
        """Stub: Returns True"""
        logger.debug(f"delete_chart_async called for {filepath} (stub - no-op)")
        return True
    
    def shutdown(self, timeout: Optional[float] = None):
        """Stub: Mark as shutdown"""
        if self._shutdown_complete:
            return
        self._shutdown_requested = True
        self._shutdown_in_progress = True
        self._shutdown_complete = True
        logger.info("ChartGenerator stub shutdown complete")
    
    async def shutdown_async(self, timeout: Optional[float] = None):
        """Stub: Mark as shutdown"""
        if self._shutdown_complete:
            return
        self._shutdown_requested = True
        self._shutdown_in_progress = True
        self._shutdown_complete = True
        logger.info("ChartGenerator stub shutdown complete (async)")
    
    def _cleanup_pending_charts(self):
        """Stub: No-op"""
        pass
    
    async def _cleanup_pending_charts_async(self):
        """Stub: No-op"""
        pass
    
    def cleanup_old_charts(self, max_age_minutes: int = 30, days: Optional[int] = None, **kwargs) -> int:
        """Stub: Returns 0 (no charts cleaned)"""
        logger.debug(f"cleanup_old_charts called (stub - no-op)")
        return 0
    
    async def cleanup_old_charts_async(self, max_age_minutes: int = 30, days: Optional[int] = None, **kwargs) -> int:
        """Stub: Returns 0 (no charts cleaned)"""
        logger.debug(f"cleanup_old_charts_async called (stub - no-op)")
        return 0
