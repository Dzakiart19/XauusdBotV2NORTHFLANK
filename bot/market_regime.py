"""
Lightweight market regime detector - simplified for Koyeb free tier.
"""
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass
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
    VOLATILE = 'volatile'
    UNKNOWN = 'unknown'


@dataclass
class MarketRegime:
    regime_type: RegimeType = RegimeType.UNKNOWN
    strength: float = 0.5
    confidence: float = 0.5
    volatility: float = 0.5
    trend_direction: str = 'neutral'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'regime_type': self.regime_type.value if isinstance(self.regime_type, RegimeType) else str(self.regime_type),
            'strength': self.strength,
            'confidence': self.confidence,
            'volatility': self.volatility,
            'trend_direction': self.trend_direction
        }


class MarketRegimeDetector:
    """Simple market regime detector"""
    
    def __init__(self, config=None, indicator_engine=None):
        self.config = config
        self.indicator_engine = indicator_engine
        self.current_regime = MarketRegime()
    
    def detect_regime(self, df=None) -> MarketRegime:
        if df is None or len(df) < 20:
            return MarketRegime()
        
        try:
            close_prices = df['close'].values if 'close' in df.columns else []
            if len(close_prices) < 20:
                return MarketRegime()
            
            recent = close_prices[-20:]
            first_half = sum(recent[:10]) / 10
            second_half = sum(recent[10:]) / 10
            
            change_pct = (second_half - first_half) / first_half * 100
            
            if abs(change_pct) > 1.0:
                regime_type = RegimeType.STRONG_TREND
                trend = 'bullish' if change_pct > 0 else 'bearish'
            elif abs(change_pct) > 0.5:
                regime_type = RegimeType.MODERATE_TREND
                trend = 'bullish' if change_pct > 0 else 'bearish'
            elif abs(change_pct) > 0.2:
                regime_type = RegimeType.WEAK_TREND
                trend = 'bullish' if change_pct > 0 else 'bearish'
            else:
                regime_type = RegimeType.RANGE_BOUND
                trend = 'neutral'
            
            self.current_regime = MarketRegime(
                regime_type=regime_type,
                strength=min(abs(change_pct) / 2, 1.0),
                confidence=0.7,
                volatility=0.5,
                trend_direction=trend
            )
            return self.current_regime
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return MarketRegime()
    
    def get_current_regime(self) -> MarketRegime:
        return self.current_regime
    
    def get_regime_multiplier(self, signal_type: str = 'BUY') -> float:
        regime = self.current_regime
        if regime.regime_type in [RegimeType.STRONG_TREND, RegimeType.MODERATE_TREND]:
            if (regime.trend_direction == 'bullish' and signal_type == 'BUY') or \
               (regime.trend_direction == 'bearish' and signal_type == 'SELL'):
                return 1.2
        elif regime.regime_type == RegimeType.RANGE_BOUND:
            return 0.9
        return 1.0
    
    def get_regime(self, indicators=None, df_m1=None, df_m5=None) -> MarketRegime:
        """Get market regime with optional indicators and dataframes.
        
        Args:
            indicators: Dict of indicator values (optional)
            df_m1: M1 timeframe dataframe (optional)
            df_m5: M5 timeframe dataframe (preferred for detection)
        """
        df = df_m5 if df_m5 is not None else df_m1
        if df is not None:
            return self.detect_regime(df)
        return self.current_regime
