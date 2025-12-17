"""
Enhanced Market Regime Detector - Lightweight but smart for Koyeb free tier.
Features: Multi-indicator regime detection, volatility analysis, trend strength.
"""
from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from bot.logger import setup_logger

logger = setup_logger('MarketRegime')


class RegimeType(str, Enum):
    STRONG_TREND = 'strong_trend'
    MODERATE_TREND = 'moderate_trend'
    WEAK_TREND = 'weak_trend'
    TRENDING_UP = 'trending_up'
    TRENDING_DOWN = 'trending_down'
    RANGE_BOUND = 'range_bound'
    RANGING = 'ranging'
    BREAKOUT = 'breakout'
    HIGH_VOLATILITY = 'high_volatility'
    LOW_VOLATILITY = 'low_volatility'
    VOLATILE = 'volatile'
    CONSOLIDATION = 'consolidation'
    UNKNOWN = 'unknown'


@dataclass
class MarketRegime:
    regime_type: RegimeType = RegimeType.UNKNOWN
    strength: float = 0.5
    confidence: float = 0.5
    volatility: float = 0.5
    trend_direction: str = 'neutral'
    atr_ratio: float = 1.0
    adx_value: float = 0.0
    recommendation: str = ''
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'regime_type': self.regime_type.value if isinstance(self.regime_type, RegimeType) else str(self.regime_type),
            'strength': self.strength,
            'confidence': self.confidence,
            'volatility': self.volatility,
            'trend_direction': self.trend_direction,
            'atr_ratio': self.atr_ratio,
            'adx_value': self.adx_value,
            'recommendation': self.recommendation
        }
    
    @property
    def is_trending(self) -> bool:
        return self.regime_type in [RegimeType.STRONG_TREND, RegimeType.MODERATE_TREND, 
                                    RegimeType.TRENDING_UP, RegimeType.TRENDING_DOWN]
    
    @property
    def is_ranging(self) -> bool:
        return self.regime_type in [RegimeType.RANGE_BOUND, RegimeType.RANGING, RegimeType.CONSOLIDATION]
    
    @property
    def is_volatile(self) -> bool:
        return self.volatility > 0.7 or self.regime_type == RegimeType.HIGH_VOLATILITY
    
    @property
    def bias(self) -> str:
        """Get market bias based on trend direction"""
        if self.trend_direction == 'bullish':
            return 'BUY'
        elif self.trend_direction == 'bearish':
            return 'SELL'
        return 'NEUTRAL'


class MarketRegimeDetector:
    """
    Enhanced market regime detector with:
    - EMA-based trend detection
    - ATR-based volatility analysis
    - ADX trend strength (simplified)
    - Multi-timeframe confirmation
    - Regime history for pattern detection
    """
    
    TREND_EMA_PERIODS = [10, 20, 50]
    ATR_PERIOD = 14
    ADX_PERIOD = 14
    VOLATILITY_LOOKBACK = 20
    MAX_REGIME_HISTORY = 20
    
    STRONG_TREND_THRESHOLD = 0.8
    MODERATE_TREND_THRESHOLD = 0.4
    HIGH_VOLATILITY_THRESHOLD = 1.5
    LOW_VOLATILITY_THRESHOLD = 0.5
    STRONG_ADX_THRESHOLD = 25
    WEAK_ADX_THRESHOLD = 15
    
    def __init__(self, config=None, indicator_engine=None):
        self.config = config
        self.indicator_engine = indicator_engine
        self.current_regime = MarketRegime()
        self.regime_history: List[MarketRegime] = []
        self._last_detection_time: Optional[datetime] = None
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate EMA for given prices"""
        if len(prices) < period:
            return prices[-1] if len(prices) > 0 else 0.0
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema
        return ema
    
    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        """Calculate Average True Range"""
        if len(highs) < 2:
            return 0.0
        
        tr_list = []
        for i in range(1, len(highs)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1])
            low_close = abs(lows[i] - closes[i-1])
            tr = max(high_low, high_close, low_close)
            tr_list.append(tr)
        
        if not tr_list:
            return 0.0
        
        return sum(tr_list[-self.ATR_PERIOD:]) / min(len(tr_list), self.ATR_PERIOD)
    
    def _calculate_adx_simplified(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        """Simplified ADX calculation"""
        if len(highs) < self.ADX_PERIOD + 1:
            return 0.0
        
        try:
            plus_dm_sum = 0
            minus_dm_sum = 0
            tr_sum = 0
            
            for i in range(1, min(len(highs), self.ADX_PERIOD + 1)):
                high_diff = highs[i] - highs[i-1]
                low_diff = lows[i-1] - lows[i]
                
                plus_dm = high_diff if high_diff > low_diff and high_diff > 0 else 0
                minus_dm = low_diff if low_diff > high_diff and low_diff > 0 else 0
                
                tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
                
                plus_dm_sum += plus_dm
                minus_dm_sum += minus_dm
                tr_sum += tr
            
            if tr_sum == 0:
                return 0.0
            
            plus_di = (plus_dm_sum / tr_sum) * 100
            minus_di = (minus_dm_sum / tr_sum) * 100
            
            di_sum = plus_di + minus_di
            if di_sum == 0:
                return 0.0
            
            dx = abs(plus_di - minus_di) / di_sum * 100
            return dx
            
        except (ZeroDivisionError, ValueError):
            return 0.0
    
    def _analyze_trend(self, closes: np.ndarray) -> tuple:
        """Analyze trend using EMAs"""
        if len(closes) < max(self.TREND_EMA_PERIODS):
            return ('neutral', 0.0)
        
        emas = [self._calculate_ema(closes, p) for p in self.TREND_EMA_PERIODS]
        current_price = closes[-1]
        
        bullish_count = sum(1 for ema in emas if current_price > ema)
        ema_aligned_bullish = all(emas[i] > emas[i+1] for i in range(len(emas)-1))
        ema_aligned_bearish = all(emas[i] < emas[i+1] for i in range(len(emas)-1))
        
        if bullish_count >= 2 and ema_aligned_bullish:
            trend = 'bullish'
            strength = min(1.0, bullish_count / len(emas) + 0.2)
        elif bullish_count <= 1 and ema_aligned_bearish:
            trend = 'bearish'
            strength = min(1.0, (len(emas) - bullish_count) / len(emas) + 0.2)
        else:
            trend = 'neutral'
            strength = 0.3
        
        return (trend, strength)
    
    def _analyze_volatility(self, closes: np.ndarray, atr: float) -> tuple:
        """Analyze volatility level"""
        if len(closes) < self.VOLATILITY_LOOKBACK or atr == 0:
            return (0.5, 1.0)
        
        avg_price = np.mean(closes[-self.VOLATILITY_LOOKBACK:])
        if avg_price == 0:
            return (0.5, 1.0)
        
        atr_percent = (atr / avg_price) * 100
        
        historical_changes = []
        for i in range(1, min(len(closes), self.VOLATILITY_LOOKBACK)):
            change = abs(closes[i] - closes[i-1]) / closes[i-1] * 100
            historical_changes.append(change)
        
        avg_change = np.mean(historical_changes) if historical_changes else 0
        
        if avg_change > 0:
            volatility_ratio = atr_percent / (avg_change * 10) if avg_change > 0 else 1.0
        else:
            volatility_ratio = 1.0
        
        volatility_normalized = min(1.0, max(0.0, atr_percent / 0.5))
        
        return (volatility_normalized, volatility_ratio)
    
    def detect_regime(self, df=None) -> MarketRegime:
        """Detect current market regime from dataframe"""
        if df is None or len(df) < 20:
            return MarketRegime()
        
        try:
            closes = df['close'].values if 'close' in df.columns else np.array([])
            highs = df['high'].values if 'high' in df.columns else closes
            lows = df['low'].values if 'low' in df.columns else closes
            
            if len(closes) < 20:
                return MarketRegime()
            
            trend_direction, trend_strength = self._analyze_trend(closes)
            atr = self._calculate_atr(highs, lows, closes)
            adx = self._calculate_adx_simplified(highs, lows, closes)
            volatility, atr_ratio = self._analyze_volatility(closes, atr)
            
            if adx >= self.STRONG_ADX_THRESHOLD and trend_strength >= self.STRONG_TREND_THRESHOLD:
                regime_type = RegimeType.STRONG_TREND
                confidence = 0.85
                recommendation = f"Strong {trend_direction} trend - follow the trend"
            elif adx >= self.WEAK_ADX_THRESHOLD and trend_strength >= self.MODERATE_TREND_THRESHOLD:
                regime_type = RegimeType.MODERATE_TREND
                confidence = 0.7
                recommendation = f"Moderate {trend_direction} trend - trade with caution"
            elif volatility >= 0.7:
                regime_type = RegimeType.HIGH_VOLATILITY
                confidence = 0.6
                recommendation = "High volatility - reduce position size"
            elif volatility <= 0.3:
                regime_type = RegimeType.LOW_VOLATILITY
                confidence = 0.65
                recommendation = "Low volatility - wait for breakout"
            elif adx < self.WEAK_ADX_THRESHOLD:
                regime_type = RegimeType.RANGE_BOUND
                confidence = 0.6
                recommendation = "Range bound - trade support/resistance"
            else:
                regime_type = RegimeType.WEAK_TREND
                confidence = 0.5
                recommendation = "Weak trend - be selective"
            
            regime = MarketRegime(
                regime_type=regime_type,
                strength=trend_strength,
                confidence=confidence,
                volatility=volatility,
                trend_direction=trend_direction,
                atr_ratio=atr_ratio,
                adx_value=adx,
                recommendation=recommendation
            )
            
            self.current_regime = regime
            self.regime_history.append(regime)
            if len(self.regime_history) > self.MAX_REGIME_HISTORY:
                self.regime_history = self.regime_history[-self.MAX_REGIME_HISTORY:]
            
            self._last_detection_time = datetime.now()
            
            logger.debug(f"Regime detected: {regime_type.value}, trend={trend_direction}, "
                        f"strength={trend_strength:.2f}, ADX={adx:.1f}, vol={volatility:.2f}")
            
            return regime
            
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return MarketRegime()
    
    def get_current_regime(self) -> MarketRegime:
        return self.current_regime
    
    def get_regime_multiplier(self, signal_type: str = 'BUY') -> float:
        """Get confidence multiplier based on regime alignment with signal"""
        regime = self.current_regime
        
        if regime.is_trending:
            if (regime.trend_direction == 'bullish' and signal_type == 'BUY') or \
               (regime.trend_direction == 'bearish' and signal_type == 'SELL'):
                return 1.2 if regime.regime_type == RegimeType.STRONG_TREND else 1.1
            else:
                return 0.7
        elif regime.is_volatile:
            return 0.8
        elif regime.is_ranging:
            return 0.9
        
        return 1.0
    
    def get_regime(self, indicators=None, df_m1=None, df_m5=None) -> MarketRegime:
        """Get market regime with optional indicators and dataframes"""
        df = df_m5 if df_m5 is not None else df_m1
        if df is not None:
            return self.detect_regime(df)
        return self.current_regime
    
    def get_adaptive_sl_multiplier(self) -> float:
        """Get SL multiplier based on volatility"""
        vol = self.current_regime.volatility
        if vol >= 0.7:
            return 1.3
        elif vol >= 0.5:
            return 1.1
        elif vol <= 0.3:
            return 0.9
        return 1.0
    
    def get_adaptive_tp_multiplier(self) -> float:
        """Get TP multiplier based on trend strength"""
        regime = self.current_regime
        if regime.is_trending and regime.strength >= 0.7:
            return 1.3
        elif regime.is_ranging:
            return 0.8
        return 1.0
    
    def is_favorable_for_trading(self) -> tuple:
        """Check if current regime is favorable for trading"""
        regime = self.current_regime
        
        if regime.is_volatile and regime.volatility > 0.8:
            return (False, "Extreme volatility - avoid trading")
        
        if regime.regime_type == RegimeType.UNKNOWN:
            return (False, "Unknown market conditions")
        
        if regime.is_trending and regime.strength >= 0.5:
            return (True, f"Good trending conditions ({regime.trend_direction})")
        
        if regime.is_ranging and regime.confidence >= 0.6:
            return (True, "Range-bound - good for reversal trades")
        
        return (True, "Acceptable conditions")
    
    def get_regime_summary(self) -> str:
        """Get human-readable regime summary"""
        regime = self.current_regime
        return f"""📈 Market Regime Analysis
━━━━━━━━━━━━━━━━━━━━━━━━
Type: {regime.regime_type.value.replace('_', ' ').title()}
Trend: {regime.trend_direction.title()} ({regime.strength*100:.0f}% strength)
Volatility: {'High' if regime.volatility > 0.7 else 'Normal' if regime.volatility > 0.3 else 'Low'}
ADX: {regime.adx_value:.1f}
Confidence: {regime.confidence*100:.0f}%

💡 {regime.recommendation}"""
