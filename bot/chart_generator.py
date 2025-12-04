"""
Minimal stub version of ChartGenerator - no-op implementation.
All chart generation methods return None without doing anything.
This removes heavy matplotlib/mplfinance dependencies.

Includes aggressive cleanup infrastructure for when chart generation is enabled.
"""

import os
import time
import glob
from datetime import datetime, timedelta
from collections import OrderedDict
from typing import Optional, Tuple, Dict, Any, List
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


class ChartLRUCache:
    """LRU Cache untuk tracking chart files yang sudah di-generate.
    
    Menggunakan OrderedDict untuk maintain insertion order dan 
    evict oldest items ketika cache penuh.
    """
    
    def __init__(self, max_size: int = 10):
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        logger.debug(f"ChartLRUCache initialized with max_size={max_size}")
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get item from cache. Moves item to end (most recent) if found."""
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None
    
    def put(self, key: str, value: Dict[str, Any]) -> Optional[str]:
        """Add item to cache. Returns evicted key if cache was full."""
        evicted_key = None
        
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = value
        else:
            if len(self._cache) >= self._max_size:
                evicted_key, _ = self._cache.popitem(last=False)
                self._evictions += 1
                logger.debug(f"LRU cache evicted: {evicted_key}")
            
            self._cache[key] = value
        
        return evicted_key
    
    def remove(self, key: str) -> bool:
        """Remove item from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> int:
        """Clear all items from cache. Returns number of items cleared."""
        count = len(self._cache)
        self._cache.clear()
        return count
    
    def get_all_keys(self) -> List[str]:
        """Get all keys in cache (oldest first)."""
        return list(self._cache.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': len(self._cache),
            'max_size': self._max_size,
            'hits': self._hits,
            'misses': self._misses,
            'evictions': self._evictions,
            'hit_rate': self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0.0
        }
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        return key in self._cache


class ChartGenerator:
    """Minimal stub ChartGenerator - all methods are no-op.
    
    This version removes matplotlib/mplfinance dependencies.
    All chart generation methods return None.
    
    Includes aggressive cleanup infrastructure ready for when full mode is enabled.
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
        
        lru_cache_size = getattr(config, 'CHART_LRU_CACHE_SIZE', 10)
        self._lru_cache = ChartLRUCache(max_size=lru_cache_size)
        
        self._generated_charts: Dict[str, Dict[str, Any]] = {}
        
        self._memory_threshold_mb = getattr(config, 'CHART_MEMORY_THRESHOLD_MB', 400)
        self._expiry_minutes = getattr(config, 'CHART_EXPIRY_MINUTES', 15)
        
        self._cleanup_stats = {
            'total_deleted': 0,
            'immediate_deletes': 0,
            'aggressive_cleanups': 0,
            'memory_blocked': 0,
            'last_cleanup_time': None,
            'last_cleanup_count': 0
        }
        
        logger.info(f"ChartGenerator stub initialized (no-op mode) with LRU cache size={lru_cache_size}, "
                    f"memory_threshold={self._memory_threshold_mb}MB, expiry={self._expiry_minutes}min")
    
    def _get_current_memory_mb(self) -> float:
        """Get current process memory usage in MB.
        
        Returns:
            float: Memory usage in MB, or 0.0 if unable to determine
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)
        except ImportError:
            logger.debug("psutil not available, cannot check memory")
            return 0.0
        except Exception as e:
            logger.warning(f"Error getting memory usage: {e}")
            return 0.0
    
    def _check_memory_before_generate(self) -> bool:
        """Check if memory usage is below threshold before generating chart.
        
        Returns:
            bool: True if memory is OK to proceed, False if memory is too high
        """
        current_memory = self._get_current_memory_mb()
        
        if current_memory <= 0:
            return True
        
        if current_memory > self._memory_threshold_mb:
            self._cleanup_stats['memory_blocked'] += 1
            logger.warning(f"🚨 Memory too high for chart generation: {current_memory:.1f}MB > {self._memory_threshold_mb}MB threshold")
            return False
        
        logger.debug(f"Memory check OK: {current_memory:.1f}MB < {self._memory_threshold_mb}MB threshold")
        return True
    
    def _track_generated_chart(self, filepath: str, metadata: Optional[Dict] = None):
        """Track a generated chart file with timestamp and metadata.
        
        Args:
            filepath: Path to the chart file
            metadata: Optional metadata about the chart
        """
        now = datetime.now()
        chart_info = {
            'filepath': filepath,
            'created_at': now,
            'timestamp': now.timestamp(),
            'metadata': metadata or {},
            'sent': False,
            'deleted': False
        }
        
        self._generated_charts[filepath] = chart_info
        
        evicted = self._lru_cache.put(filepath, chart_info)
        if evicted:
            self._delete_chart_file(evicted)
            logger.info(f"🗑️ LRU eviction: deleted old chart {evicted}")
        
        logger.debug(f"Tracking chart: {filepath} (total tracked: {len(self._generated_charts)})")
    
    def _delete_chart_file(self, filepath: str) -> bool:
        """Delete a chart file from disk.
        
        Args:
            filepath: Path to the chart file
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        if not filepath:
            return False
            
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.debug(f"Deleted chart file: {filepath}")
                return True
            else:
                logger.debug(f"Chart file already deleted: {filepath}")
                return True
        except FileNotFoundError:
            return True
        except (PermissionError, OSError, IOError) as e:
            logger.warning(f"Failed to delete chart file {filepath}: {e}")
            return False
    
    def immediate_delete_chart(self, filepath: str) -> bool:
        """Immediately delete a chart file after it has been sent.
        
        This method should be called right after a chart is successfully 
        sent to Telegram to free up disk space immediately.
        
        Args:
            filepath: Path to the chart file to delete
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        if not filepath:
            logger.debug("immediate_delete_chart: No filepath provided")
            return False
        
        if filepath in self._generated_charts:
            self._generated_charts[filepath]['sent'] = True
            self._generated_charts[filepath]['deleted'] = True
        
        self._lru_cache.remove(filepath)
        
        success = self._delete_chart_file(filepath)
        
        if success:
            self._cleanup_stats['immediate_deletes'] += 1
            self._cleanup_stats['total_deleted'] += 1
            if filepath in self._generated_charts:
                del self._generated_charts[filepath]
            logger.info(f"🗑️ Immediate delete: {filepath}")
        
        return success
    
    async def immediate_delete_chart_async(self, filepath: str) -> bool:
        """Async version of immediate_delete_chart.
        
        Args:
            filepath: Path to the chart file to delete
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        return self.immediate_delete_chart(filepath)
    
    def cleanup_charts_aggressive(self, max_age_minutes: Optional[int] = None) -> int:
        """Aggressively cleanup old chart files.
        
        Deletes all chart files older than max_age_minutes.
        Also cleans up any orphaned files in the charts directory.
        
        Args:
            max_age_minutes: Maximum age in minutes before a chart is deleted.
                           Defaults to CHART_EXPIRY_MINUTES config value.
                           
        Returns:
            int: Number of charts deleted
        """
        if max_age_minutes is None:
            max_age_minutes = self._expiry_minutes
        
        deleted_count = 0
        now = datetime.now()
        cutoff_time = now - timedelta(minutes=max_age_minutes)
        
        logger.info(f"🧹 Starting aggressive chart cleanup (max_age={max_age_minutes}min)")
        
        expired_charts = []
        for filepath, info in list(self._generated_charts.items()):
            created_at = info.get('created_at', now)
            if created_at < cutoff_time:
                expired_charts.append(filepath)
        
        for filepath in expired_charts:
            if self._delete_chart_file(filepath):
                deleted_count += 1
                self._lru_cache.remove(filepath)
                del self._generated_charts[filepath]
                logger.debug(f"Deleted expired tracked chart: {filepath}")
        
        try:
            pattern = os.path.join(self.chart_dir, '*.png')
            for chart_file in glob.glob(pattern):
                try:
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(chart_file))
                    if file_mtime < cutoff_time:
                        if self._delete_chart_file(chart_file):
                            deleted_count += 1
                            if chart_file not in expired_charts:
                                logger.debug(f"Deleted orphaned chart: {chart_file}")
                except (OSError, IOError) as e:
                    logger.warning(f"Error checking chart file {chart_file}: {e}")
        except (OSError, IOError) as e:
            logger.warning(f"Error scanning chart directory: {e}")
        
        self._cleanup_stats['aggressive_cleanups'] += 1
        self._cleanup_stats['total_deleted'] += deleted_count
        self._cleanup_stats['last_cleanup_time'] = now
        self._cleanup_stats['last_cleanup_count'] = deleted_count
        
        if deleted_count > 0:
            logger.info(f"✅ Aggressive cleanup complete: deleted {deleted_count} charts")
        else:
            logger.debug("Aggressive cleanup: no charts to delete")
        
        return deleted_count
    
    async def cleanup_charts_aggressive_async(self, max_age_minutes: Optional[int] = None) -> int:
        """Async version of cleanup_charts_aggressive.
        
        Args:
            max_age_minutes: Maximum age in minutes before a chart is deleted.
                           
        Returns:
            int: Number of charts deleted
        """
        return self.cleanup_charts_aggressive(max_age_minutes)
    
    def get_stats(self) -> dict:
        """Return stats for compatibility and monitoring"""
        lru_stats = self._lru_cache.get_stats()
        
        return {
            'pending_charts': len(self._pending_charts),
            'timed_out_tasks': len(self._timed_out_tasks),
            'shutdown_requested': self._shutdown_requested,
            'mode': 'stub',
            'tracked_charts': len(self._generated_charts),
            'lru_cache': lru_stats,
            'memory_threshold_mb': self._memory_threshold_mb,
            'expiry_minutes': self._expiry_minutes,
            'cleanup_stats': self._cleanup_stats.copy(),
            'current_memory_mb': self._get_current_memory_mb()
        }
    
    def get_tracked_chart_count(self) -> int:
        """Get count of currently tracked charts."""
        return len(self._generated_charts)
    
    def get_lru_cache_stats(self) -> Dict[str, Any]:
        """Get LRU cache statistics."""
        return self._lru_cache.get_stats()
    
    def get_cleanup_stats(self) -> Dict[str, Any]:
        """Get cleanup statistics."""
        return self._cleanup_stats.copy()
    
    def _check_dataframe_size(self, df) -> Tuple[bool, Optional[str]]:
        """Stub: Always returns valid"""
        return True, None
    
    def generate_chart(self, df, signal: Optional[dict] = None,
                      timeframe: str = 'M1') -> Optional[str]:
        """Stub: Returns None without generating chart"""
        if not self._check_memory_before_generate():
            logger.warning("Chart generation blocked due to high memory usage")
            return None
        
        logger.debug("generate_chart called (stub - no-op)")
        return None
    
    async def generate_chart_async(self, df, signal: Optional[dict] = None,
                                   timeframe: str = 'M1', timeout: Optional[float] = None) -> Optional[str]:
        """Stub: Returns None without generating chart"""
        if not self._check_memory_before_generate():
            logger.warning("Chart generation blocked due to high memory usage")
            return None
        
        logger.debug("generate_chart_async called (stub - no-op)")
        return None
    
    def _delete_file_with_retry(self, filepath: str, max_retries: int = 3) -> bool:
        """Delete file with retry logic.
        
        Args:
            filepath: Path to file to delete
            max_retries: Maximum number of retry attempts
            
        Returns:
            bool: True if deleted successfully
        """
        for attempt in range(max_retries):
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    return True
                return True
            except FileNotFoundError:
                return True
            except (PermissionError, OSError, IOError) as e:
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))
                    continue
                logger.warning(f"Failed to delete {filepath} after {max_retries} attempts: {e}")
                return False
        return False
    
    async def _delete_file_with_retry_async(self, filepath: str, max_retries: int = 3) -> bool:
        """Async version of _delete_file_with_retry."""
        import asyncio
        
        for attempt in range(max_retries):
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    return True
                return True
            except FileNotFoundError:
                return True
            except (PermissionError, OSError, IOError) as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.1 * (attempt + 1))
                    continue
                logger.warning(f"Failed to delete {filepath} after {max_retries} attempts: {e}")
                return False
        return False
    
    def delete_chart(self, filepath: str) -> bool:
        """Delete chart file using immediate_delete_chart."""
        logger.debug(f"delete_chart called for {filepath}")
        return self.immediate_delete_chart(filepath)
    
    async def delete_chart_async(self, filepath: str) -> bool:
        """Async delete chart file."""
        logger.debug(f"delete_chart_async called for {filepath}")
        return await self.immediate_delete_chart_async(filepath)
    
    def shutdown(self, timeout: Optional[float] = None):
        """Shutdown with aggressive cleanup of all charts."""
        if self._shutdown_complete:
            return
        
        self._shutdown_requested = True
        self._shutdown_in_progress = True
        
        cleanup_count = self.cleanup_charts_aggressive(max_age_minutes=0)
        logger.info(f"Shutdown cleanup: deleted {cleanup_count} charts")
        
        self._lru_cache.clear()
        self._generated_charts.clear()
        
        self._shutdown_complete = True
        logger.info("ChartGenerator stub shutdown complete")
    
    async def shutdown_async(self, timeout: Optional[float] = None):
        """Async shutdown with aggressive cleanup."""
        if self._shutdown_complete:
            return
        
        self._shutdown_requested = True
        self._shutdown_in_progress = True
        
        cleanup_count = await self.cleanup_charts_aggressive_async(max_age_minutes=0)
        logger.info(f"Shutdown cleanup: deleted {cleanup_count} charts")
        
        self._lru_cache.clear()
        self._generated_charts.clear()
        
        self._shutdown_complete = True
        logger.info("ChartGenerator stub shutdown complete (async)")
    
    def _cleanup_pending_charts(self):
        """Cleanup pending charts - calls aggressive cleanup."""
        self.cleanup_charts_aggressive()
    
    async def _cleanup_pending_charts_async(self):
        """Async cleanup pending charts."""
        await self.cleanup_charts_aggressive_async()
    
    def cleanup_old_charts(self, max_age_minutes: int = 30, days: Optional[int] = None, **kwargs) -> int:
        """Cleanup old charts using aggressive cleanup method.
        
        Args:
            max_age_minutes: Maximum age in minutes (default: 30)
            days: Alternative way to specify age in days (ignored if max_age_minutes provided)
            
        Returns:
            int: Number of charts cleaned up
        """
        if days is not None:
            max_age_minutes = days * 24 * 60
        
        return self.cleanup_charts_aggressive(max_age_minutes=max_age_minutes)
    
    async def cleanup_old_charts_async(self, max_age_minutes: int = 30, days: Optional[int] = None, **kwargs) -> int:
        """Async cleanup old charts.
        
        Args:
            max_age_minutes: Maximum age in minutes (default: 30)
            days: Alternative way to specify age in days
            
        Returns:
            int: Number of charts cleaned up
        """
        if days is not None:
            max_age_minutes = days * 24 * 60
        
        return await self.cleanup_charts_aggressive_async(max_age_minutes=max_age_minutes)
