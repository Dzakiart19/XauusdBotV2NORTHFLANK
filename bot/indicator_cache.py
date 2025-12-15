"""
Indicator Caching System for Performance Optimization.

Provides:
- LRU cache for indicator calculations
- Time-based cache invalidation
- Memory-efficient storage
"""
import time
import logging
import hashlib
from typing import Dict, Any, Optional, Tuple
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
import pandas as pd
import numpy as np

logger = logging.getLogger('IndicatorCache')


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    created_at: float
    expires_at: float
    hits: int = 0
    size_bytes: int = 0


class IndicatorCache:
    """
    LRU cache for indicator calculations with TTL support.
    
    Features:
    - Configurable max size and TTL
    - LRU eviction policy
    - Memory usage tracking
    - Hit/miss statistics
    """
    
    def __init__(self, max_size: int = 100, default_ttl: float = 60.0,
                 max_memory_mb: float = 50.0):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of entries
            default_ttl: Default time-to-live in seconds
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = RLock()
        
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_evictions': 0,
            'total_memory_bytes': 0,
        }
        
        logger.info(
            f"IndicatorCache initialized: max_size={max_size}, "
            f"ttl={default_ttl}s, max_memory={max_memory_mb}MB"
        )
    
    def _generate_key(self, indicator_name: str, params: Dict[str, Any],
                     data_hash: Optional[str] = None) -> str:
        """Generate a unique cache key."""
        key_parts = [indicator_name]
        
        for k, v in sorted(params.items()):
            key_parts.append(f"{k}={v}")
        
        if data_hash:
            key_parts.append(data_hash)
        
        key_str = ":".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _hash_dataframe(self, df: pd.DataFrame) -> str:
        """Generate hash for DataFrame content."""
        if df is None or len(df) == 0:
            return "empty"
        
        try:
            last_rows = df.tail(10)
            hash_data = f"{len(df)}:{last_rows.values.tobytes()}"
            return hashlib.md5(hash_data.encode()).hexdigest()[:16]
        except Exception:
            return str(time.time())
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of a value in bytes."""
        if isinstance(value, pd.Series):
            return value.memory_usage(deep=True)
        elif isinstance(value, pd.DataFrame):
            return value.memory_usage(deep=True).sum()
        elif isinstance(value, np.ndarray):
            return value.nbytes
        elif isinstance(value, dict):
            return sum(self._estimate_size(v) for v in value.values())
        elif isinstance(value, (list, tuple)):
            return sum(self._estimate_size(v) for v in value)
        else:
            return 64
    
    def _evict_expired(self) -> int:
        """Remove expired entries. Returns count of evicted entries."""
        now = time.time()
        evicted = 0
        
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.expires_at < now
        ]
        
        for key in expired_keys:
            entry = self._cache.pop(key)
            self._stats['total_memory_bytes'] -= entry.size_bytes
            evicted += 1
        
        if evicted > 0:
            self._stats['evictions'] += evicted
            logger.debug(f"Evicted {evicted} expired entries")
        
        return evicted
    
    def _evict_lru(self, needed_bytes: int = 0) -> int:
        """Evict least recently used entries. Returns count of evicted entries."""
        evicted = 0
        
        while len(self._cache) >= self.max_size:
            key, entry = self._cache.popitem(last=False)
            self._stats['total_memory_bytes'] -= entry.size_bytes
            evicted += 1
        
        while (needed_bytes > 0 and 
               self._stats['total_memory_bytes'] + needed_bytes > self.max_memory_bytes and
               len(self._cache) > 0):
            key, entry = self._cache.popitem(last=False)
            self._stats['total_memory_bytes'] -= entry.size_bytes
            self._stats['memory_evictions'] += 1
            evicted += 1
        
        if evicted > 0:
            self._stats['evictions'] += evicted
        
        return evicted
    
    def get(self, indicator_name: str, params: Dict[str, Any],
            data_hash: Optional[str] = None) -> Tuple[bool, Any]:
        """
        Get cached indicator value.
        
        Args:
            indicator_name: Name of the indicator
            params: Indicator parameters
            data_hash: Optional hash of input data
        
        Returns:
            Tuple of (hit, value) where hit is True if found
        """
        key = self._generate_key(indicator_name, params, data_hash)
        
        with self._lock:
            self._evict_expired()
            
            if key in self._cache:
                entry = self._cache[key]
                
                self._cache.move_to_end(key)
                entry.hits += 1
                self._stats['hits'] += 1
                
                logger.debug(f"Cache hit: {indicator_name} (hits={entry.hits})")
                return True, entry.value
            
            self._stats['misses'] += 1
            return False, None
    
    def set(self, indicator_name: str, params: Dict[str, Any],
            value: Any, data_hash: Optional[str] = None,
            ttl: Optional[float] = None) -> bool:
        """
        Cache indicator value.
        
        Args:
            indicator_name: Name of the indicator
            params: Indicator parameters
            value: Value to cache
            data_hash: Optional hash of input data
            ttl: Optional TTL override
        
        Returns:
            True if cached successfully
        """
        if value is None:
            return False
        
        key = self._generate_key(indicator_name, params, data_hash)
        ttl = ttl if ttl is not None else self.default_ttl
        
        size_bytes = self._estimate_size(value)
        
        with self._lock:
            self._evict_expired()
            self._evict_lru(size_bytes)
            
            now = time.time()
            entry = CacheEntry(
                value=value,
                created_at=now,
                expires_at=now + ttl,
                hits=0,
                size_bytes=size_bytes
            )
            
            if key in self._cache:
                old_entry = self._cache[key]
                self._stats['total_memory_bytes'] -= old_entry.size_bytes
            
            self._cache[key] = entry
            self._stats['total_memory_bytes'] += size_bytes
            
            logger.debug(
                f"Cached: {indicator_name} "
                f"(size={size_bytes/1024:.1f}KB, ttl={ttl}s)"
            )
            
            return True
    
    def invalidate(self, indicator_name: Optional[str] = None) -> int:
        """
        Invalidate cache entries.
        
        Args:
            indicator_name: Optional indicator name to invalidate.
                          If None, clears all entries.
        
        Returns:
            Number of invalidated entries
        """
        with self._lock:
            if indicator_name is None:
                count = len(self._cache)
                self._cache.clear()
                self._stats['total_memory_bytes'] = 0
                logger.info(f"Invalidated all {count} cache entries")
                return count
            
            keys_to_remove = [
                key for key in self._cache.keys()
                if key.startswith(indicator_name)
            ]
            
            for key in keys_to_remove:
                entry = self._cache.pop(key)
                self._stats['total_memory_bytes'] -= entry.size_bytes
            
            logger.info(f"Invalidated {len(keys_to_remove)} entries for {indicator_name}")
            return len(keys_to_remove)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = (self._stats['hits'] / total_requests * 100
                       if total_requests > 0 else 0)
            
            return {
                'entries': len(self._cache),
                'max_size': self.max_size,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'hit_rate': f"{hit_rate:.1f}%",
                'evictions': self._stats['evictions'],
                'memory_evictions': self._stats['memory_evictions'],
                'memory_used_mb': self._stats['total_memory_bytes'] / 1024 / 1024,
                'max_memory_mb': self.max_memory_bytes / 1024 / 1024,
            }
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats['total_memory_bytes'] = 0
            logger.info("Cache cleared")


_default_cache: Optional[IndicatorCache] = None


def get_indicator_cache() -> IndicatorCache:
    """Get or create the default indicator cache."""
    global _default_cache
    if _default_cache is None:
        _default_cache = IndicatorCache()
    return _default_cache


def cached_indicator(ttl: float = 60.0):
    """
    Decorator for caching indicator calculations.
    
    Usage:
        @cached_indicator(ttl=120)
        def calculate_rsi(df, period):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache = get_indicator_cache()
            
            indicator_name = func.__name__
            
            params = {}
            if args:
                for i, arg in enumerate(args[1:], 1):
                    if not isinstance(arg, pd.DataFrame):
                        params[f'arg{i}'] = arg
            params.update(kwargs)
            
            df = args[0] if args and isinstance(args[0], pd.DataFrame) else None
            data_hash = cache._hash_dataframe(df) if df is not None else None
            
            hit, value = cache.get(indicator_name, params, data_hash)
            if hit:
                return value
            
            result = func(*args, **kwargs)
            
            cache.set(indicator_name, params, result, data_hash, ttl)
            
            return result
        
        return wrapper
    return decorator
