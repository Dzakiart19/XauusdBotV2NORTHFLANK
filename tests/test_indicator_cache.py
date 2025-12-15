"""Unit tests for indicator cache module."""
import pytest
import time
import pandas as pd
import numpy as np
from bot.indicator_cache import (
    IndicatorCache,
    get_indicator_cache,
    cached_indicator,
)


class TestIndicatorCache:
    """Tests for IndicatorCache class."""
    
    def test_cache_set_and_get(self):
        """Should store and retrieve values."""
        cache = IndicatorCache(max_size=10, default_ttl=60)
        
        cache.set('test_indicator', {'period': 14}, [1.0, 2.0, 3.0])
        hit, value = cache.get('test_indicator', {'period': 14})
        
        assert hit is True
        assert value == [1.0, 2.0, 3.0]
    
    def test_cache_miss(self):
        """Should return miss for non-existent key."""
        cache = IndicatorCache(max_size=10, default_ttl=60)
        
        hit, value = cache.get('nonexistent', {'period': 14})
        
        assert hit is False
        assert value is None
    
    def test_cache_ttl_expiry(self):
        """Should expire entries after TTL."""
        cache = IndicatorCache(max_size=10, default_ttl=0.1)
        
        cache.set('test', {}, 'value', ttl=0.1)
        
        time.sleep(0.2)
        
        hit, value = cache.get('test', {})
        assert hit is False
    
    def test_cache_lru_eviction(self):
        """Should evict least recently used entries."""
        cache = IndicatorCache(max_size=3, default_ttl=60)
        
        cache.set('ind1', {}, 'value1')
        cache.set('ind2', {}, 'value2')
        cache.set('ind3', {}, 'value3')
        
        cache.get('ind1', {})
        
        cache.set('ind4', {}, 'value4')
        
        hit2, _ = cache.get('ind2', {})
        assert hit2 is False
        
        hit1, _ = cache.get('ind1', {})
        assert hit1 is True
    
    def test_cache_with_dataframe(self):
        """Should cache DataFrame values."""
        cache = IndicatorCache(max_size=10, default_ttl=60)
        
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        cache.set('df_indicator', {'col': 'a'}, df)
        
        hit, value = cache.get('df_indicator', {'col': 'a'})
        assert hit is True
        assert isinstance(value, pd.DataFrame)
        assert len(value) == 3
    
    def test_cache_with_series(self):
        """Should cache Series values."""
        cache = IndicatorCache(max_size=10, default_ttl=60)
        
        series = pd.Series([1.0, 2.0, 3.0])
        cache.set('series_indicator', {}, series)
        
        hit, value = cache.get('series_indicator', {})
        assert hit is True
        assert isinstance(value, pd.Series)
    
    def test_cache_stats(self):
        """Should track hit/miss statistics."""
        cache = IndicatorCache(max_size=10, default_ttl=60)
        
        cache.set('test', {}, 'value')
        cache.get('test', {})
        cache.get('test', {})
        cache.get('missing', {})
        
        stats = cache.get_stats()
        assert stats['hits'] == 2
        assert stats['misses'] == 1
        assert stats['entries'] == 1
    
    def test_cache_invalidate_all(self):
        """Should clear all entries."""
        cache = IndicatorCache(max_size=10, default_ttl=60)
        
        cache.set('ind1', {}, 'v1')
        cache.set('ind2', {}, 'v2')
        
        count = cache.invalidate()
        
        assert count == 2
        assert cache.get_stats()['entries'] == 0
    
    def test_cache_invalidate_by_name(self):
        """Should invalidate all entries when no name specified."""
        cache = IndicatorCache(max_size=10, default_ttl=60)
        
        cache.set('rsi', {'period': 14}, 'v1')
        cache.set('macd', {'fast': 12}, 'v2')
        
        count = cache.invalidate()
        
        assert count == 2
        
        hit_rsi, _ = cache.get('rsi', {'period': 14})
        hit_macd, _ = cache.get('macd', {'fast': 12})
        
        assert hit_rsi is False
        assert hit_macd is False
    
    def test_cache_different_params(self):
        """Should distinguish between different parameters."""
        cache = IndicatorCache(max_size=10, default_ttl=60)
        
        cache.set('rsi', {'period': 14}, 'rsi_14')
        cache.set('rsi', {'period': 21}, 'rsi_21')
        
        _, v14 = cache.get('rsi', {'period': 14})
        _, v21 = cache.get('rsi', {'period': 21})
        
        assert v14 == 'rsi_14'
        assert v21 == 'rsi_21'
    
    def test_cache_data_hash(self):
        """Should distinguish between different input data."""
        cache = IndicatorCache(max_size=10, default_ttl=60)
        
        cache.set('ind', {}, 'value1', data_hash='hash1')
        cache.set('ind', {}, 'value2', data_hash='hash2')
        
        _, v1 = cache.get('ind', {}, data_hash='hash1')
        _, v2 = cache.get('ind', {}, data_hash='hash2')
        
        assert v1 == 'value1'
        assert v2 == 'value2'
    
    def test_cache_memory_tracking(self):
        """Should track memory usage."""
        cache = IndicatorCache(max_size=100, default_ttl=60, max_memory_mb=1.0)
        
        large_array = np.zeros(10000)
        cache.set('large', {}, large_array)
        
        stats = cache.get_stats()
        assert stats['memory_used_mb'] > 0


class TestCachedIndicatorDecorator:
    """Tests for cached_indicator decorator."""
    
    def test_decorator_caches_result(self):
        """Should cache function result."""
        call_count = 0
        
        @cached_indicator(ttl=60)
        def calculate_something(df):
            nonlocal call_count
            call_count += 1
            return df.sum()
        
        df = pd.DataFrame({'a': [1, 2, 3]})
        
        result1 = calculate_something(df)
        result2 = calculate_something(df)
        
        assert call_count == 1
    
    def test_decorator_with_different_params(self):
        """Should call function for different parameters."""
        call_count = 0
        
        @cached_indicator(ttl=60)
        def calculate_with_period(df, period):
            nonlocal call_count
            call_count += 1
            return df.rolling(period).mean()
        
        df = pd.DataFrame({'a': range(10)})
        
        calculate_with_period(df, period=5)
        calculate_with_period(df, period=10)
        
        assert call_count == 2


class TestGetIndicatorCache:
    """Tests for global cache singleton."""
    
    def test_singleton_pattern(self):
        """Should return same cache instance."""
        cache1 = get_indicator_cache()
        cache2 = get_indicator_cache()
        
        assert cache1 is cache2
