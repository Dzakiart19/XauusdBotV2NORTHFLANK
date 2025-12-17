"""
Market Data Cache & Fallback System - Lightweight for Koyeb free tier.
Features: Price caching, stale detection, fallback to last known good data.
"""
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import pandas as pd
from bot.logger import setup_logger

logger = setup_logger('MarketDataCache')


@dataclass
class CachedPrice:
    """Cached price data with metadata"""
    price: float
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = 'live'
    is_stale: bool = False
    
    def age_seconds(self) -> float:
        return (datetime.utcnow() - self.timestamp).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'price': self.price,
            'bid': self.bid,
            'ask': self.ask,
            'spread': self.spread,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'is_stale': self.is_stale,
            'age_seconds': self.age_seconds()
        }


@dataclass
class CachedCandle:
    """Cached OHLC candle data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    timeframe: str = 'M1'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'timeframe': self.timeframe
        }


class MarketDataCache:
    """
    Market data cache with fallback capabilities:
    - Caches live price data
    - Detects stale data
    - Provides fallback to last known good price
    - Maintains candle history for analysis
    """
    
    STALE_THRESHOLD_SECONDS = 30.0
    MAX_PRICE_HISTORY = 100
    MAX_CANDLE_CACHE = 60
    PRICE_CHANGE_ALERT_PERCENT = 1.0
    
    def __init__(self, symbol: str = 'XAUUSD'):
        self.symbol = symbol
        
        self._current_price: Optional[CachedPrice] = None
        self._last_good_price: Optional[CachedPrice] = None
        self._price_history: deque = deque(maxlen=self.MAX_PRICE_HISTORY)
        
        self._candle_cache: Dict[str, deque] = {
            'M1': deque(maxlen=self.MAX_CANDLE_CACHE),
            'M5': deque(maxlen=self.MAX_CANDLE_CACHE),
            'H1': deque(maxlen=self.MAX_CANDLE_CACHE)
        }
        
        self._data_gaps: List[Dict[str, Any]] = []
        self._fallback_count: int = 0
        self._total_updates: int = 0
        
        logger.info(f"MarketDataCache initialized for {symbol}")
    
    def update_price(self, price: float, bid: float = 0.0, ask: float = 0.0,
                     source: str = 'live') -> CachedPrice:
        """Update current price and maintain history"""
        if price <= 0:
            logger.warning(f"Invalid price: {price}")
            current = self.get_current_price()
            if current:
                return current
            return CachedPrice(price=0.0, is_stale=True)
        
        spread = ask - bid if bid > 0 and ask > 0 else 0.0
        
        cached = CachedPrice(
            price=price,
            bid=bid,
            ask=ask,
            spread=spread,
            timestamp=datetime.utcnow(),
            source=source,
            is_stale=False
        )
        
        if self._current_price:
            old_price = self._current_price.price
            change_percent = abs(price - old_price) / old_price * 100
            if change_percent > self.PRICE_CHANGE_ALERT_PERCENT:
                logger.info(f"📊 Price change alert: ${old_price:.2f} → ${price:.2f} ({change_percent:.2f}%)")
            
            time_gap = (cached.timestamp - self._current_price.timestamp).total_seconds()
            if time_gap > self.STALE_THRESHOLD_SECONDS * 2:
                self._data_gaps.append({
                    'start': self._current_price.timestamp,
                    'end': cached.timestamp,
                    'duration_seconds': time_gap
                })
                if len(self._data_gaps) > 20:
                    self._data_gaps = self._data_gaps[-20:]
        
        self._last_good_price = self._current_price or cached
        self._current_price = cached
        self._price_history.append(cached)
        self._total_updates += 1
        
        return cached
    
    def get_current_price(self) -> Optional[CachedPrice]:
        """Get current price with staleness check"""
        if not self._current_price:
            return None
        
        age = self._current_price.age_seconds()
        if age > self.STALE_THRESHOLD_SECONDS:
            self._current_price.is_stale = True
            logger.warning(f"Price data is stale ({age:.1f}s old)")
        
        return self._current_price
    
    def get_price_with_fallback(self) -> Tuple[float, str, bool]:
        """
        Get price with automatic fallback
        Returns: (price, source, is_fallback)
        """
        current = self.get_current_price()
        
        if current and not current.is_stale:
            return (current.price, current.source, False)
        
        if self._last_good_price:
            self._fallback_count += 1
            logger.warning(f"Using fallback price: ${self._last_good_price.price:.2f}")
            return (self._last_good_price.price, 'fallback', True)
        
        if self._price_history:
            last_known = self._price_history[-1]
            self._fallback_count += 1
            return (last_known.price, 'history', True)
        
        logger.error("No price data available!")
        return (0.0, 'none', True)
    
    def update_candle(self, timeframe: str, candle_data: Dict[str, Any]) -> bool:
        """Update candle cache for timeframe"""
        if timeframe not in self._candle_cache:
            return False
        
        try:
            candle = CachedCandle(
                timestamp=candle_data.get('timestamp', datetime.utcnow()),
                open=float(candle_data.get('open', 0)),
                high=float(candle_data.get('high', 0)),
                low=float(candle_data.get('low', 0)),
                close=float(candle_data.get('close', 0)),
                volume=float(candle_data.get('volume', 0)),
                timeframe=timeframe
            )
            
            if candle.open <= 0 or candle.close <= 0:
                return False
            
            self._candle_cache[timeframe].append(candle)
            return True
            
        except (ValueError, TypeError) as e:
            logger.error(f"Error updating candle: {e}")
            return False
    
    def get_candles_df(self, timeframe: str = 'M1') -> Optional[pd.DataFrame]:
        """Get candles as pandas DataFrame"""
        if timeframe not in self._candle_cache:
            return None
        
        candles = list(self._candle_cache[timeframe])
        if not candles:
            return None
        
        data = [c.to_dict() for c in candles]
        df = pd.DataFrame(data)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def get_price_stats(self) -> Dict[str, Any]:
        """Get price statistics from history"""
        if not self._price_history:
            return {}
        
        prices = [p.price for p in self._price_history]
        
        return {
            'current': prices[-1] if prices else 0,
            'high': max(prices),
            'low': min(prices),
            'avg': sum(prices) / len(prices),
            'count': len(prices),
            'range': max(prices) - min(prices),
            'range_percent': ((max(prices) - min(prices)) / min(prices) * 100) if min(prices) > 0 else 0
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get cache health status"""
        current = self.get_current_price()
        
        return {
            'symbol': self.symbol,
            'has_data': current is not None,
            'is_stale': current.is_stale if current else True,
            'last_update_age': current.age_seconds() if current else float('inf'),
            'total_updates': self._total_updates,
            'fallback_count': self._fallback_count,
            'fallback_rate': (self._fallback_count / self._total_updates * 100) if self._total_updates > 0 else 0,
            'data_gaps': len(self._data_gaps),
            'candle_counts': {tf: len(c) for tf, c in self._candle_cache.items()},
            'price_history_count': len(self._price_history)
        }
    
    def clear_cache(self) -> None:
        """Clear all cached data"""
        self._current_price = None
        self._last_good_price = None
        self._price_history.clear()
        for tf in self._candle_cache:
            self._candle_cache[tf].clear()
        self._data_gaps.clear()
        self._fallback_count = 0
        self._total_updates = 0
        logger.info("Market data cache cleared")
    
    def get_cache_summary(self) -> str:
        """Get human-readable cache summary"""
        status = self.get_health_status()
        current = self.get_current_price()
        
        price_str = f"${current.price:.2f}" if current else "N/A"
        stale_str = "⚠️ STALE" if (current and current.is_stale) else "✅ Live"
        age_str = f"{current.age_seconds():.1f}s ago" if current else "N/A"
        
        return f"""📦 Market Data Cache Status
━━━━━━━━━━━━━━━━━━━━━━━━
Symbol: {self.symbol}
Current Price: {price_str} {stale_str}
Last Update: {age_str}

Updates: {status['total_updates']}
Fallbacks: {status['fallback_count']} ({status['fallback_rate']:.1f}%)
Data Gaps: {status['data_gaps']}

Candles Cached:
  M1: {status['candle_counts']['M1']}
  M5: {status['candle_counts']['M5']}
  H1: {status['candle_counts']['H1']}"""
