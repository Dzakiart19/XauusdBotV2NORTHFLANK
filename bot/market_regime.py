"""
Market Regime Detector untuk Bot Trading XAUUSD.

Modul ini menyediakan deteksi kondisi pasar untuk optimasi strategi trading:
- Trend Strength Detection (ADX + EMA slope + DI crossover)
- Volatility Assessment (ATR-based dengan 50-period baseline)
- Price Position Analysis (Price clustering untuk S/R)
- Breakout Detection (Bollinger Band squeeze)
- Regime Transition Detection
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any, Tuple
from enum import Enum
from datetime import datetime

from bot.logger import setup_logger
from bot.strategy import safe_float, is_valid_number
from bot.indicators import IndicatorEngine, safe_series_operation

logger = setup_logger('MarketRegime')


class MarketRegimeError(Exception):
    """Base exception for market regime detection errors"""
    pass


class RegimeType(str, Enum):
    """Enum untuk tipe market regime"""
    STRONG_TREND = 'strong_trend'
    MODERATE_TREND = 'moderate_trend'
    RANGE_BOUND = 'range_bound'
    BREAKOUT = 'breakout'
    HIGH_VOLATILITY = 'high_volatility'
    WEAK_TREND = 'weak_trend'
    UNKNOWN = 'unknown'


class BiasType(str, Enum):
    """Enum untuk bias arah trading"""
    BUY = 'BUY'
    SELL = 'SELL'
    NEUTRAL = 'NEUTRAL'


class VolatilityZone(str, Enum):
    """Enum untuk volatility zones"""
    HIGH = 'high'
    NORMAL = 'normal'
    LOW = 'low'


class VolatilityTrend(str, Enum):
    """Enum untuk volatility trend"""
    INCREASING = 'increasing'
    DECREASING = 'decreasing'
    STABLE = 'stable'


@dataclass
class DIAnalysis:
    """Data hasil analisis DI crossover"""
    plus_di: float = 0.0
    minus_di: float = 0.0
    di_diff: float = 0.0
    is_bullish_crossover: bool = False
    is_bearish_crossover: bool = False
    crossover_strength: str = 'none'
    direction: str = 'neutral'


@dataclass
class EMAAnalysis:
    """Data hasil analisis EMA untuk trend confirmation"""
    ema_short: float = 0.0
    ema_medium: float = 0.0
    ema_long: float = 0.0
    slope: float = 0.0
    slope_direction: str = 'neutral'
    is_aligned_bullish: bool = False
    is_aligned_bearish: bool = False
    alignment_score: float = 0.0


@dataclass
class TrendAnalysis:
    """Data hasil analisis trend dengan multi-indicator confirmation"""
    adx: float = 0.0
    plus_di: float = 0.0
    minus_di: float = 0.0
    trend_strength: str = 'none'
    trend_direction: str = 'neutral'
    is_strong_trend: bool = False
    is_trending: bool = False
    di_analysis: DIAnalysis = field(default_factory=DIAnalysis)
    ema_analysis: EMAAnalysis = field(default_factory=EMAAnalysis)
    momentum_direction: str = 'neutral'
    momentum_strength: str = 'weak'
    confirmation_count: int = 0
    confirmed_direction: str = 'neutral'


@dataclass
class VolatilityAnalysis:
    """Data hasil analisis volatility dengan enhanced tracking"""
    current_atr: float = 0.0
    average_atr: float = 0.0
    historical_atr_50: float = 0.0
    atr_ratio: float = 1.0
    atr_ratio_50: float = 1.0
    volatility_level: str = 'normal'
    volatility_zone: str = 'normal'
    volatility_trend: str = 'stable'
    is_high_volatility: bool = False
    is_low_volatility: bool = False
    suggested_sl_multiplier: float = 1.0
    bb_width_pct: float = 0.0
    is_bb_squeeze: bool = False
    atr_history: List[float] = field(default_factory=list)


@dataclass
class PriceCluster:
    """Data untuk price clustering S/R"""
    price: float = 0.0
    weight: float = 0.0
    touch_count: int = 0
    is_support: bool = False
    is_resistance: bool = False


@dataclass
class PricePositionAnalysis:
    """Data hasil analisis posisi harga dengan price clustering"""
    current_price: float = 0.0
    nearest_support: float = 0.0
    nearest_resistance: float = 0.0
    distance_to_support: float = 0.0
    distance_to_resistance: float = 0.0
    price_position: str = 'midpoint'
    position_bias: str = 'NEUTRAL'
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)
    congestion_zones: List[PriceCluster] = field(default_factory=list)
    range_high_20: float = 0.0
    range_low_20: float = 0.0
    range_pct: float = 0.0


@dataclass
class BreakoutAnalysis:
    """Data hasil analisis breakout dengan BB squeeze"""
    is_tight_range: bool = False
    range_compression_ratio: float = 1.0
    is_breakout_up: bool = False
    is_breakout_down: bool = False
    breakout_candles: int = 10
    volume_increasing: bool = False
    volume_ratio: float = 1.0
    breakout_probability: float = 0.0
    breakout_direction: str = 'none'
    is_bb_squeeze: bool = False
    bb_breakout_up: bool = False
    bb_breakout_down: bool = False


@dataclass
class RegimeTransition:
    """Data untuk regime transition tracking"""
    previous_regime: str = 'unknown'
    current_regime: str = 'unknown'
    transition_detected: bool = False
    transition_type: str = 'none'
    regime_duration_candles: int = 0
    regime_start_time: Optional[str] = None
    alerts: List[str] = field(default_factory=list)


@dataclass
class MultiIndicatorConfirmation:
    """Data untuk multi-indicator confirmation"""
    adx_confirms: bool = False
    ema_confirms: bool = False
    momentum_confirms: bool = False
    bb_confirms: bool = False
    confirmation_count: int = 0
    required_confirmations: int = 2
    is_confirmed: bool = False
    confirmed_regime: str = 'unknown'
    confirmation_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketRegime:
    """Dataclass utama untuk hasil analisis market regime"""
    regime_type: str = 'unknown'
    bias: str = 'NEUTRAL'
    strictness_level: float = 1.0
    
    trend_analysis: TrendAnalysis = field(default_factory=TrendAnalysis)
    volatility_analysis: VolatilityAnalysis = field(default_factory=VolatilityAnalysis)
    price_position: PricePositionAnalysis = field(default_factory=PricePositionAnalysis)
    breakout_analysis: BreakoutAnalysis = field(default_factory=BreakoutAnalysis)
    regime_transition: RegimeTransition = field(default_factory=RegimeTransition)
    multi_confirmation: MultiIndicatorConfirmation = field(default_factory=MultiIndicatorConfirmation)
    
    confidence: float = 0.0
    analysis_timestamp: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    recommended_strategies: List[str] = field(default_factory=list)
    avoid_strategies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert MarketRegime to dictionary"""
        return {
            'regime_type': self.regime_type,
            'bias': self.bias,
            'strictness_level': self.strictness_level,
            'trend': {
                'adx': self.trend_analysis.adx,
                'plus_di': self.trend_analysis.plus_di,
                'minus_di': self.trend_analysis.minus_di,
                'strength': self.trend_analysis.trend_strength,
                'direction': self.trend_analysis.trend_direction,
                'is_trending': self.trend_analysis.is_trending,
                'di_crossover': {
                    'is_bullish': self.trend_analysis.di_analysis.is_bullish_crossover,
                    'is_bearish': self.trend_analysis.di_analysis.is_bearish_crossover,
                    'strength': self.trend_analysis.di_analysis.crossover_strength
                },
                'ema': {
                    'slope': self.trend_analysis.ema_analysis.slope,
                    'slope_direction': self.trend_analysis.ema_analysis.slope_direction,
                    'alignment_score': self.trend_analysis.ema_analysis.alignment_score
                },
                'confirmation_count': self.trend_analysis.confirmation_count,
                'confirmed_direction': self.trend_analysis.confirmed_direction
            },
            'volatility': {
                'current_atr': self.volatility_analysis.current_atr,
                'average_atr': self.volatility_analysis.average_atr,
                'historical_atr_50': self.volatility_analysis.historical_atr_50,
                'atr_ratio': self.volatility_analysis.atr_ratio,
                'atr_ratio_50': self.volatility_analysis.atr_ratio_50,
                'level': self.volatility_analysis.volatility_level,
                'zone': self.volatility_analysis.volatility_zone,
                'trend': self.volatility_analysis.volatility_trend,
                'sl_multiplier': self.volatility_analysis.suggested_sl_multiplier,
                'bb_width_pct': self.volatility_analysis.bb_width_pct,
                'is_bb_squeeze': self.volatility_analysis.is_bb_squeeze
            },
            'price_position': {
                'current': self.price_position.current_price,
                'support': self.price_position.nearest_support,
                'resistance': self.price_position.nearest_resistance,
                'position': self.price_position.price_position,
                'bias': self.price_position.position_bias,
                'range_high_20': self.price_position.range_high_20,
                'range_low_20': self.price_position.range_low_20,
                'range_pct': self.price_position.range_pct
            },
            'breakout': {
                'is_tight_range': self.breakout_analysis.is_tight_range,
                'compression_ratio': self.breakout_analysis.range_compression_ratio,
                'is_breakout_up': self.breakout_analysis.is_breakout_up,
                'is_breakout_down': self.breakout_analysis.is_breakout_down,
                'volume_increasing': self.breakout_analysis.volume_increasing,
                'probability': self.breakout_analysis.breakout_probability,
                'is_bb_squeeze': self.breakout_analysis.is_bb_squeeze
            },
            'regime_transition': {
                'previous_regime': self.regime_transition.previous_regime,
                'current_regime': self.regime_transition.current_regime,
                'transition_detected': self.regime_transition.transition_detected,
                'transition_type': self.regime_transition.transition_type,
                'duration_candles': self.regime_transition.regime_duration_candles,
                'alerts': self.regime_transition.alerts
            },
            'multi_confirmation': {
                'adx_confirms': self.multi_confirmation.adx_confirms,
                'ema_confirms': self.multi_confirmation.ema_confirms,
                'momentum_confirms': self.multi_confirmation.momentum_confirms,
                'bb_confirms': self.multi_confirmation.bb_confirms,
                'confirmation_count': self.multi_confirmation.confirmation_count,
                'is_confirmed': self.multi_confirmation.is_confirmed
            },
            'confidence': self.confidence,
            'timestamp': self.analysis_timestamp,
            'recommendations': {
                'use': self.recommended_strategies,
                'avoid': self.avoid_strategies
            },
            'warnings': self.warnings
        }


class MarketRegimeDetector:
    """
    Market Regime Detector untuk mengidentifikasi kondisi pasar XAUUSD.
    
    Fitur utama:
    1. Trend Strength Detection (ADX + EMA slope + DI crossover)
    2. Volatility Assessment (ATR dengan 50-period baseline)
    3. Price Position Analysis (Price clustering untuk S/R)
    4. Breakout Detection (Bollinger Band squeeze)
    5. Multi-Indicator Confirmation
    6. Regime Transition Detection
    """
    
    ADX_STRONG_TREND = 35
    ADX_MODERATE_TREND_HIGH = 35
    ADX_MODERATE_TREND_LOW = 25
    ADX_WEAK_TREND_LOW = 15
    ADX_WEAK_TREND_HIGH = 25
    ADX_RANGING = 15
    
    ATR_HIGH_VOLATILITY_THRESHOLD = 1.5
    ATR_NORMAL_VOLATILITY_LOW = 0.8
    ATR_NORMAL_VOLATILITY_HIGH = 1.5
    ATR_LOW_VOLATILITY_THRESHOLD = 0.8
    
    ATR_AVERAGE_PERIOD = 20
    ATR_HISTORICAL_PERIOD = 50
    BREAKOUT_CANDLES = 10
    RANGE_TIGHT_THRESHOLD = 0.7
    RANGE_LOOKBACK = 20
    
    SUPPORT_PROXIMITY_PCT = 0.002
    RESISTANCE_PROXIMITY_PCT = 0.002
    
    EMA_SLOPE_THRESHOLD = 0.05
    DI_CROSSOVER_MIN_DIFF = 5.0
    
    CLUSTER_TOLERANCE_PCT = 0.001
    
    def __init__(self, config, indicator_engine: Optional[IndicatorEngine] = None):
        """
        Inisialisasi MarketRegimeDetector.
        
        Args:
            config: Objek konfigurasi bot
            indicator_engine: Instance IndicatorEngine (opsional, akan dibuat jika tidak ada)
        """
        self.config = config
        self.indicator_engine = indicator_engine or IndicatorEngine(config)
        self._last_regime: Optional[MarketRegime] = None
        self._regime_history: List[Tuple[str, str]] = []
        self._regime_start_candle: int = 0
        self._current_candle_count: int = 0
        logger.info("MarketRegimeDetector initialized with enhanced detection")
    
    def _safe_get_value(self, data: Any, index: int = -1, default: float = 0.0) -> float:
        """Safely get value from series/dataframe with NaN/Inf handling"""
        if data is None:
            return default
        if isinstance(data, pd.DataFrame):
            if len(data.columns) > 0:
                data = data.iloc[:, 0]
            else:
                return default
        if not isinstance(data, pd.Series):
            try:
                data = pd.Series(data)
            except (TypeError, ValueError):
                return default
        return safe_series_operation(data, 'value', index, default)
    
    def _analyze_di_crossover(self, df: pd.DataFrame, adx_period: int = 14) -> DIAnalysis:
        """
        Analisis DI+ dan DI- crossover untuk direction detection.
        
        Args:
            df: DataFrame dengan data OHLC
            adx_period: Periode untuk kalkulasi ADX
            
        Returns:
            DIAnalysis dengan hasil analisis crossover
        """
        result = DIAnalysis()
        
        try:
            if df is None or len(df) < adx_period + 2:
                return result
            
            _, plus_di_series, minus_di_series = self.indicator_engine.calculate_adx(df, adx_period)
            
            result.plus_di = safe_float(self._safe_get_value(plus_di_series, -1, 0.0), 0.0, "plus_di")
            result.minus_di = safe_float(self._safe_get_value(minus_di_series, -1, 0.0), 0.0, "minus_di")
            result.di_diff = abs(result.plus_di - result.minus_di)
            
            plus_di_prev = safe_float(self._safe_get_value(plus_di_series, -2, 0.0), 0.0, "plus_di_prev")
            minus_di_prev = safe_float(self._safe_get_value(minus_di_series, -2, 0.0), 0.0, "minus_di_prev")
            
            if result.plus_di > result.minus_di and plus_di_prev <= minus_di_prev:
                result.is_bullish_crossover = True
            elif result.minus_di > result.plus_di and minus_di_prev <= plus_di_prev:
                result.is_bearish_crossover = True
            
            if result.di_diff >= self.DI_CROSSOVER_MIN_DIFF * 2:
                result.crossover_strength = 'strong'
            elif result.di_diff >= self.DI_CROSSOVER_MIN_DIFF:
                result.crossover_strength = 'moderate'
            else:
                result.crossover_strength = 'weak'
            
            if result.plus_di > result.minus_di:
                result.direction = 'bullish'
            elif result.minus_di > result.plus_di:
                result.direction = 'bearish'
            else:
                result.direction = 'neutral'
            
        except Exception as e:
            logger.error(f"Error in DI crossover analysis: {str(e)}")
        
        return result
    
    def _analyze_ema_alignment(self, df: pd.DataFrame) -> EMAAnalysis:
        """
        Analisis EMA alignment dan slope untuk trend confirmation.
        
        Args:
            df: DataFrame dengan data OHLC
            
        Returns:
            EMAAnalysis dengan hasil analisis
        """
        result = EMAAnalysis()
        
        try:
            if df is None or len(df) < 50:
                return result
            
            ema_periods = getattr(self.config, 'EMA_PERIODS', [5, 20, 50])
            if len(ema_periods) < 3:
                ema_periods = [5, 20, 50]
            
            ema_short = self.indicator_engine.calculate_ema(df, ema_periods[0])
            ema_medium = self.indicator_engine.calculate_ema(df, ema_periods[1])
            ema_long = self.indicator_engine.calculate_ema(df, ema_periods[2] if len(ema_periods) > 2 else 50)
            
            result.ema_short = safe_float(self._safe_get_value(ema_short, -1, 0.0), 0.0, "ema_short")
            result.ema_medium = safe_float(self._safe_get_value(ema_medium, -1, 0.0), 0.0, "ema_medium")
            result.ema_long = safe_float(self._safe_get_value(ema_long, -1, 0.0), 0.0, "ema_long")
            
            ema_slope = self.indicator_engine.calculate_ema_slope(df, ema_periods[1], lookback=3)
            result.slope = safe_float(self._safe_get_value(ema_slope, -1, 0.0), 0.0, "ema_slope")
            
            if result.slope > self.EMA_SLOPE_THRESHOLD:
                result.slope_direction = 'bullish'
            elif result.slope < -self.EMA_SLOPE_THRESHOLD:
                result.slope_direction = 'bearish'
            else:
                result.slope_direction = 'neutral'
            
            if result.ema_short > result.ema_medium > result.ema_long:
                result.is_aligned_bullish = True
                result.alignment_score = 1.0
            elif result.ema_short < result.ema_medium < result.ema_long:
                result.is_aligned_bearish = True
                result.alignment_score = -1.0
            else:
                if result.ema_short > result.ema_medium:
                    result.alignment_score = 0.5
                elif result.ema_short < result.ema_medium:
                    result.alignment_score = -0.5
                else:
                    result.alignment_score = 0.0
            
        except Exception as e:
            logger.error(f"Error in EMA alignment analysis: {str(e)}")
        
        return result
    
    def _analyze_trend_strength(self, df: pd.DataFrame, adx_period: int = 14) -> TrendAnalysis:
        """
        Analisis kekuatan trend dengan multi-indicator confirmation.
        
        ADX Thresholds (updated):
        - ADX > 35: Strong trend
        - ADX 25-35: Moderate trend
        - ADX 15-25: Weak trend
        - ADX < 15: Ranging/No trend
        
        Confirmation sources:
        - ADX strength
        - EMA slope direction
        - DI crossover direction
        - Price momentum
        
        Args:
            df: DataFrame dengan data OHLC
            adx_period: Periode untuk kalkulasi ADX
            
        Returns:
            TrendAnalysis dengan hasil analisis lengkap
        """
        result = TrendAnalysis()
        
        try:
            if df is None or len(df) < adx_period + 1:
                logger.warning(f"Insufficient data for ADX calculation (need {adx_period + 1}, got {len(df) if df is not None else 0})")
                return result
            
            adx_series, plus_di_series, minus_di_series = self.indicator_engine.calculate_adx(df, adx_period)
            
            result.adx = safe_float(self._safe_get_value(adx_series, -1, 0.0), 0.0, "adx")
            result.plus_di = safe_float(self._safe_get_value(plus_di_series, -1, 0.0), 0.0, "plus_di")
            result.minus_di = safe_float(self._safe_get_value(minus_di_series, -1, 0.0), 0.0, "minus_di")
            
            if result.adx > self.ADX_STRONG_TREND:
                result.trend_strength = 'strong'
                result.is_strong_trend = True
                result.is_trending = True
            elif result.adx >= self.ADX_MODERATE_TREND_LOW:
                result.trend_strength = 'moderate'
                result.is_strong_trend = False
                result.is_trending = True
            elif result.adx >= self.ADX_WEAK_TREND_LOW:
                result.trend_strength = 'weak'
                result.is_strong_trend = False
                result.is_trending = False
            else:
                result.trend_strength = 'none'
                result.is_strong_trend = False
                result.is_trending = False
            
            if result.plus_di > result.minus_di:
                result.trend_direction = 'bullish'
            elif result.minus_di > result.plus_di:
                result.trend_direction = 'bearish'
            else:
                result.trend_direction = 'neutral'
            
            result.di_analysis = self._analyze_di_crossover(df, adx_period)
            result.ema_analysis = self._analyze_ema_alignment(df)
            
            momentum_data = self.indicator_engine.calculate_price_momentum(df, period=10)
            result.momentum_direction = momentum_data.get('direction', 'neutral')
            result.momentum_strength = momentum_data.get('strength', 'weak')
            
            confirmation_count = 0
            directions = []
            
            if result.trend_direction != 'neutral':
                directions.append(result.trend_direction)
                confirmation_count += 1
            
            if result.ema_analysis.slope_direction != 'neutral':
                directions.append(result.ema_analysis.slope_direction)
                if result.ema_analysis.slope_direction == result.trend_direction:
                    confirmation_count += 1
            
            if result.momentum_direction != 'neutral':
                directions.append(result.momentum_direction)
                if result.momentum_direction == result.trend_direction:
                    confirmation_count += 1
            
            result.confirmation_count = confirmation_count
            
            if directions:
                bullish_count = sum(1 for d in directions if d == 'bullish')
                bearish_count = sum(1 for d in directions if d == 'bearish')
                
                if bullish_count >= 2:
                    result.confirmed_direction = 'bullish'
                elif bearish_count >= 2:
                    result.confirmed_direction = 'bearish'
                else:
                    result.confirmed_direction = result.trend_direction
            else:
                result.confirmed_direction = 'neutral'
            
            logger.debug(f"Trend Analysis: ADX={result.adx:.2f}, +DI={result.plus_di:.2f}, -DI={result.minus_di:.2f}, "
                        f"Strength={result.trend_strength}, Confirmations={result.confirmation_count}")
            
        except Exception as e:
            logger.error(f"Error in trend strength analysis: {str(e)}")
        
        return result
    
    def _analyze_volatility(self, df: pd.DataFrame, atr_period: int = 14) -> VolatilityAnalysis:
        """
        Analisis volatilitas dengan 50-period baseline dan trend tracking.
        
        Features:
        - ATR comparison dengan 20 dan 50 period baseline
        - Volatility zones (high, normal, low)
        - Volatility trend (increasing/decreasing/stable)
        - Bollinger Band width analysis
        
        Args:
            df: DataFrame dengan data OHLC
            atr_period: Periode untuk kalkulasi ATR
            
        Returns:
            VolatilityAnalysis dengan hasil analisis lengkap
        """
        result = VolatilityAnalysis()
        
        try:
            if df is None or len(df) < max(atr_period, self.ATR_HISTORICAL_PERIOD):
                logger.warning("Insufficient data for volatility analysis")
                return result
            
            atr_series = self.indicator_engine.calculate_atr(df, atr_period)
            result.current_atr = safe_float(self._safe_get_value(atr_series, -1, 0.0), 0.0, "current_atr")
            
            if len(atr_series) >= self.ATR_AVERAGE_PERIOD:
                avg_atr = atr_series.rolling(window=self.ATR_AVERAGE_PERIOD, min_periods=1).mean()
                result.average_atr = safe_float(self._safe_get_value(avg_atr, -1, result.current_atr), result.current_atr, "average_atr")
            else:
                result.average_atr = safe_float(atr_series.mean(), result.current_atr, "average_atr")
            
            if len(atr_series) >= self.ATR_HISTORICAL_PERIOD:
                hist_atr = atr_series.rolling(window=self.ATR_HISTORICAL_PERIOD, min_periods=1).mean()
                result.historical_atr_50 = safe_float(self._safe_get_value(hist_atr, -1, result.average_atr), result.average_atr, "historical_atr_50")
            else:
                result.historical_atr_50 = result.average_atr
            
            if result.average_atr > 0:
                result.atr_ratio = safe_float(result.current_atr / result.average_atr, 1.0, "atr_ratio")
            else:
                result.atr_ratio = 1.0
            
            if result.historical_atr_50 > 0:
                result.atr_ratio_50 = safe_float(result.current_atr / result.historical_atr_50, 1.0, "atr_ratio_50")
            else:
                result.atr_ratio_50 = 1.0
            
            if result.atr_ratio_50 > self.ATR_HIGH_VOLATILITY_THRESHOLD:
                result.volatility_level = 'high'
                result.volatility_zone = VolatilityZone.HIGH.value
                result.is_high_volatility = True
                result.is_low_volatility = False
                result.suggested_sl_multiplier = 0.8
            elif result.atr_ratio_50 >= self.ATR_LOW_VOLATILITY_THRESHOLD:
                result.volatility_level = 'normal'
                result.volatility_zone = VolatilityZone.NORMAL.value
                result.is_high_volatility = False
                result.is_low_volatility = False
                result.suggested_sl_multiplier = 1.0
            else:
                result.volatility_level = 'low'
                result.volatility_zone = VolatilityZone.LOW.value
                result.is_high_volatility = False
                result.is_low_volatility = True
                result.suggested_sl_multiplier = 1.2
            
            if len(atr_series) >= 5:
                atr_recent = [safe_float(self._safe_get_value(atr_series, -i, 0.0), 0.0) for i in range(1, 6)]
                result.atr_history = atr_recent
                
                if len(atr_recent) >= 3:
                    atr_diff = atr_recent[0] - atr_recent[2]
                    if atr_diff > result.average_atr * 0.1:
                        result.volatility_trend = VolatilityTrend.INCREASING.value
                    elif atr_diff < -result.average_atr * 0.1:
                        result.volatility_trend = VolatilityTrend.DECREASING.value
                    else:
                        result.volatility_trend = VolatilityTrend.STABLE.value
            
            bb_data = self.indicator_engine.calculate_bollinger_bands(df, period=20, std_dev=2.0)
            result.bb_width_pct = bb_data.get('width_pct', 0.0)
            result.is_bb_squeeze = bb_data.get('squeeze', False)
            
            logger.debug(f"Volatility Analysis: ATR={result.current_atr:.2f}, HistATR50={result.historical_atr_50:.2f}, "
                        f"Ratio50={result.atr_ratio_50:.2f}, Zone={result.volatility_zone}, Trend={result.volatility_trend}")
            
        except Exception as e:
            logger.error(f"Error in volatility analysis: {str(e)}")
        
        return result
    
    def _calculate_price_clusters(self, df: pd.DataFrame, lookback: int = 50) -> List[PriceCluster]:
        """
        Calculate price clusters untuk identifikasi congestion zones sebagai S/R.
        
        Algorithm:
        1. Collect all swing highs/lows
        2. Cluster nearby prices within tolerance
        3. Weight recent prices higher
        4. Return sorted clusters by weight
        
        Args:
            df: DataFrame dengan data OHLC
            lookback: Periode lookback
            
        Returns:
            List of PriceCluster
        """
        clusters = []
        
        try:
            if df is None or len(df) < 10:
                return clusters
            
            high = df['high'].tail(lookback) if 'high' in df.columns else None
            low = df['low'].tail(lookback) if 'low' in df.columns else None
            close = df['close'].tail(lookback) if 'close' in df.columns else None
            
            if high is None or low is None or close is None:
                return clusters
            
            current_price = safe_float(self._safe_get_value(close, -1, 0.0), 0.0)
            if current_price <= 0:
                return clusters
            
            tolerance = current_price * self.CLUSTER_TOLERANCE_PCT
            
            price_points = []
            
            for i in range(2, len(high) - 2):
                try:
                    h = high.iloc[i]
                    h_prev1 = high.iloc[i-1]
                    h_prev2 = high.iloc[i-2]
                    h_next1 = high.iloc[i+1]
                    h_next2 = high.iloc[i+2]
                    
                    if h > h_prev1 and h > h_prev2 and h > h_next1 and h > h_next2:
                        recency_weight = (i / len(high)) * 2
                        price_points.append({'price': float(h), 'weight': recency_weight, 'type': 'high'})
                    
                    l = low.iloc[i]
                    l_prev1 = low.iloc[i-1]
                    l_prev2 = low.iloc[i-2]
                    l_next1 = low.iloc[i+1]
                    l_next2 = low.iloc[i+2]
                    
                    if l < l_prev1 and l < l_prev2 and l < l_next1 and l < l_next2:
                        recency_weight = (i / len(low)) * 2
                        price_points.append({'price': float(l), 'weight': recency_weight, 'type': 'low'})
                except (IndexError, KeyError):
                    continue
            
            if not price_points:
                return clusters
            
            price_points.sort(key=lambda x: x['price'])
            
            cluster_groups = []
            current_cluster = [price_points[0]]
            
            for i in range(1, len(price_points)):
                if abs(price_points[i]['price'] - current_cluster[-1]['price']) <= tolerance:
                    current_cluster.append(price_points[i])
                else:
                    if len(current_cluster) >= 1:
                        cluster_groups.append(current_cluster)
                    current_cluster = [price_points[i]]
            
            if current_cluster:
                cluster_groups.append(current_cluster)
            
            for group in cluster_groups:
                avg_price = sum(p['price'] for p in group) / len(group)
                total_weight = sum(p['weight'] for p in group)
                touch_count = len(group)
                
                is_support = avg_price < current_price
                is_resistance = avg_price > current_price
                
                clusters.append(PriceCluster(
                    price=avg_price,
                    weight=total_weight,
                    touch_count=touch_count,
                    is_support=is_support,
                    is_resistance=is_resistance
                ))
            
            clusters.sort(key=lambda x: x.weight, reverse=True)
            
        except Exception as e:
            logger.error(f"Error in price clustering: {str(e)}")
        
        return clusters[:10]
    
    def _analyze_price_position(self, df: pd.DataFrame, lookback: int = 50) -> PricePositionAnalysis:
        """
        Analisis posisi harga dengan price clustering untuk S/R.
        
        Features:
        - Price clustering algorithm
        - Congestion zones identification
        - 20-candle range tracking
        - Recent price weighting
        
        Args:
            df: DataFrame dengan data OHLC
            lookback: Periode lookback untuk support/resistance
            
        Returns:
            PricePositionAnalysis dengan hasil analisis
        """
        result = PricePositionAnalysis()
        
        try:
            if df is None or len(df) < 5:
                logger.warning("Insufficient data for price position analysis")
                return result
            
            close = df['close'] if 'close' in df.columns else None
            high = df['high'] if 'high' in df.columns else None
            low = df['low'] if 'low' in df.columns else None
            
            if close is None or len(close) == 0:
                return result
            
            result.current_price = safe_float(self._safe_get_value(close, -1, 0.0), 0.0, "current_price")
            
            if result.current_price <= 0:
                logger.warning("Invalid current price for position analysis")
                return result
            
            if high is not None and low is not None and len(df) >= self.RANGE_LOOKBACK:
                recent_high = high.tail(self.RANGE_LOOKBACK)
                recent_low = low.tail(self.RANGE_LOOKBACK)
                
                result.range_high_20 = safe_float(recent_high.max(), result.current_price * 1.01, "range_high_20")
                result.range_low_20 = safe_float(recent_low.min(), result.current_price * 0.99, "range_low_20")
                
                range_size = result.range_high_20 - result.range_low_20
                if result.current_price > 0:
                    result.range_pct = (range_size / result.current_price) * 100
            
            result.congestion_zones = self._calculate_price_clusters(df, lookback)
            
            if result.congestion_zones:
                support_clusters = [c for c in result.congestion_zones if c.is_support]
                resistance_clusters = [c for c in result.congestion_zones if c.is_resistance]
                
                if support_clusters:
                    support_clusters.sort(key=lambda x: x.price, reverse=True)
                    result.nearest_support = support_clusters[0].price
                    result.support_levels = [c.price for c in support_clusters[:5]]
                
                if resistance_clusters:
                    resistance_clusters.sort(key=lambda x: x.price)
                    result.nearest_resistance = resistance_clusters[0].price
                    result.resistance_levels = [c.price for c in resistance_clusters[:5]]
            
            if result.nearest_support == 0 or result.nearest_resistance == 0:
                sr_data = self.indicator_engine.calculate_micro_support_resistance(df, lookback)
                if result.nearest_support == 0:
                    result.nearest_support = safe_float(sr_data.get('nearest_support', 0.0), 0.0, "nearest_support")
                if result.nearest_resistance == 0:
                    result.nearest_resistance = safe_float(sr_data.get('nearest_resistance', 0.0), 0.0, "nearest_resistance")
                if not result.support_levels:
                    result.support_levels = sr_data.get('support_levels', [])
                if not result.resistance_levels:
                    result.resistance_levels = sr_data.get('resistance_levels', [])
            
            if result.current_price > 0:
                result.distance_to_support = safe_float(
                    (result.current_price - result.nearest_support) / result.current_price,
                    0.0, "distance_to_support"
                )
                result.distance_to_resistance = safe_float(
                    (result.nearest_resistance - result.current_price) / result.current_price,
                    0.0, "distance_to_resistance"
                )
            
            if result.distance_to_support <= self.SUPPORT_PROXIMITY_PCT:
                result.price_position = 'near_support'
                result.position_bias = BiasType.BUY.value
            elif result.distance_to_resistance <= self.RESISTANCE_PROXIMITY_PCT:
                result.price_position = 'near_resistance'
                result.position_bias = BiasType.SELL.value
            else:
                total_range = result.distance_to_support + result.distance_to_resistance
                if total_range > 0:
                    support_pct = result.distance_to_support / total_range
                    if support_pct < 0.3:
                        result.price_position = 'lower_zone'
                        result.position_bias = BiasType.BUY.value
                    elif support_pct > 0.7:
                        result.price_position = 'upper_zone'
                        result.position_bias = BiasType.SELL.value
                    else:
                        result.price_position = 'midpoint'
                        result.position_bias = BiasType.NEUTRAL.value
                else:
                    result.price_position = 'midpoint'
                    result.position_bias = BiasType.NEUTRAL.value
            
            logger.debug(f"Price Position: Current={result.current_price:.2f}, Support={result.nearest_support:.2f}, "
                        f"Resistance={result.nearest_resistance:.2f}, Position={result.price_position}, "
                        f"Clusters={len(result.congestion_zones)}")
            
        except Exception as e:
            logger.error(f"Error in price position analysis: {str(e)}")
        
        return result
    
    def _analyze_breakout(self, df: pd.DataFrame, breakout_candles: int = 10) -> BreakoutAnalysis:
        """
        Analisis potensi breakout dengan Bollinger Band squeeze.
        
        Features:
        - Range compression detection
        - Bollinger Band squeeze
        - Volume confirmation
        - Breakout probability calculation
        
        Args:
            df: DataFrame dengan data OHLC
            breakout_candles: Jumlah candle untuk analisis breakout
            
        Returns:
            BreakoutAnalysis dengan hasil analisis
        """
        result = BreakoutAnalysis()
        result.breakout_candles = breakout_candles
        
        try:
            if df is None or len(df) < breakout_candles + 1:
                logger.warning(f"Insufficient data for breakout analysis (need {breakout_candles + 1})")
                return result
            
            recent_df = df.tail(breakout_candles)
            
            high = recent_df['high'] if 'high' in recent_df.columns else None
            low = recent_df['low'] if 'low' in recent_df.columns else None
            close = df['close'] if 'close' in df.columns else None
            
            if high is None or low is None or close is None:
                return result
            
            period_high = safe_float(high.max(), 0.0, "period_high")
            period_low = safe_float(low.min(), 0.0, "period_low")
            current_close = safe_float(self._safe_get_value(close, -1, 0.0), 0.0, "current_close")
            
            period_range = period_high - period_low
            
            if len(df) > breakout_candles * 2:
                earlier_df = df.iloc[-(breakout_candles * 2):-breakout_candles]
                earlier_high = earlier_df['high'].max() if 'high' in earlier_df.columns else period_high
                earlier_low = earlier_df['low'].min() if 'low' in earlier_df.columns else period_low
                earlier_range = safe_float(earlier_high - earlier_low, period_range, "earlier_range")
                
                if earlier_range > 0:
                    result.range_compression_ratio = safe_float(period_range / earlier_range, 1.0, "compression_ratio")
                else:
                    result.range_compression_ratio = 1.0
            else:
                result.range_compression_ratio = 1.0
            
            result.is_tight_range = result.range_compression_ratio < self.RANGE_TIGHT_THRESHOLD
            
            bb_data = self.indicator_engine.calculate_bollinger_bands(df, period=20, std_dev=2.0)
            result.is_bb_squeeze = bb_data.get('squeeze', False)
            
            bb_upper = self._safe_get_value(bb_data.get('upper', pd.Series([0.0])), -1, 0.0)
            bb_lower = self._safe_get_value(bb_data.get('lower', pd.Series([0.0])), -1, 0.0)
            
            if bb_upper > 0 and current_close > bb_upper:
                result.bb_breakout_up = True
            if bb_lower > 0 and current_close < bb_lower:
                result.bb_breakout_down = True
            
            if len(df) > breakout_candles:
                prev_period_high = safe_float(df.iloc[:-1].tail(breakout_candles)['high'].max(), period_high, "prev_period_high")
                prev_period_low = safe_float(df.iloc[:-1].tail(breakout_candles)['low'].min(), period_low, "prev_period_low")
                
                result.is_breakout_up = current_close > prev_period_high
                result.is_breakout_down = current_close < prev_period_low
            
            vol_data = self.indicator_engine.calculate_volume_confirmation(df, period=breakout_candles)
            result.volume_increasing = vol_data.get('is_volume_strong', False)
            result.volume_ratio = safe_float(vol_data.get('volume_ratio', 1.0), 1.0, "volume_ratio")
            
            probability = 0.0
            
            if result.is_tight_range:
                probability += 0.2
            
            if result.is_bb_squeeze:
                probability += 0.2
            
            if result.is_breakout_up or result.is_breakout_down:
                probability += 0.3
            
            if result.bb_breakout_up or result.bb_breakout_down:
                probability += 0.15
            
            if result.volume_increasing:
                probability += 0.1
            
            if result.volume_ratio > 1.5:
                probability += 0.05
            
            result.breakout_probability = min(probability, 1.0)
            
            if result.is_breakout_up or result.bb_breakout_up:
                result.breakout_direction = 'up'
            elif result.is_breakout_down or result.bb_breakout_down:
                result.breakout_direction = 'down'
            elif result.is_tight_range or result.is_bb_squeeze:
                result.breakout_direction = 'pending'
            else:
                result.breakout_direction = 'none'
            
            logger.debug(f"Breakout Analysis: Tight={result.is_tight_range}, BBSqueeze={result.is_bb_squeeze}, "
                        f"BreakUp={result.is_breakout_up}, BreakDown={result.is_breakout_down}, "
                        f"VolInc={result.volume_increasing}, Prob={result.breakout_probability:.2f}")
            
        except Exception as e:
            logger.error(f"Error in breakout analysis: {str(e)}")
        
        return result
    
    def _check_range_bound(self, df: pd.DataFrame, volatility: VolatilityAnalysis) -> bool:
        """
        Check if market is range-bound using multiple criteria.
        
        Criteria:
        - ATR ratio < 0.8
        - Low ADX (< 15)
        - Bollinger Band squeeze
        - Tight high-low range
        
        Args:
            df: DataFrame dengan data OHLC
            volatility: VolatilityAnalysis hasil
            
        Returns:
            bool: True if range-bound
        """
        try:
            is_low_volatility = volatility.atr_ratio_50 < self.ATR_LOW_VOLATILITY_THRESHOLD
            is_bb_squeeze = volatility.is_bb_squeeze
            
            if df is not None and len(df) >= self.RANGE_LOOKBACK:
                high = df['high'].tail(self.RANGE_LOOKBACK)
                low = df['low'].tail(self.RANGE_LOOKBACK)
                close = df['close'].tail(self.RANGE_LOOKBACK)
                
                range_size = high.max() - low.min()
                avg_close = close.mean()
                
                if avg_close > 0:
                    range_pct = (range_size / avg_close) * 100
                    is_tight_range = range_pct < 0.5
                else:
                    is_tight_range = False
            else:
                is_tight_range = False
            
            criteria_met = sum([is_low_volatility, is_bb_squeeze, is_tight_range])
            
            return criteria_met >= 2
            
        except Exception as e:
            logger.error(f"Error in range-bound check: {str(e)}")
            return False
    
    def _get_multi_indicator_confirmation(self, 
                                          trend: TrendAnalysis, 
                                          volatility: VolatilityAnalysis,
                                          breakout: BreakoutAnalysis) -> MultiIndicatorConfirmation:
        """
        Get multi-indicator confirmation untuk regime detection.
        
        Requires at least 2 of 3 indicators to agree.
        
        Args:
            trend: TrendAnalysis hasil
            volatility: VolatilityAnalysis hasil
            breakout: BreakoutAnalysis hasil
            
        Returns:
            MultiIndicatorConfirmation dengan hasil
        """
        result = MultiIndicatorConfirmation()
        result.required_confirmations = 2
        
        try:
            result.adx_confirms = trend.is_trending
            
            ema_direction = trend.ema_analysis.slope_direction
            di_direction = trend.di_analysis.direction
            result.ema_confirms = (ema_direction == di_direction and ema_direction != 'neutral')
            
            momentum_direction = trend.momentum_direction
            result.momentum_confirms = (momentum_direction == trend.trend_direction and 
                                       trend.momentum_strength in ['moderate', 'strong'])
            
            result.bb_confirms = not volatility.is_bb_squeeze
            
            result.confirmation_count = sum([
                result.adx_confirms,
                result.ema_confirms,
                result.momentum_confirms
            ])
            
            result.is_confirmed = result.confirmation_count >= result.required_confirmations
            
            if result.is_confirmed:
                if trend.is_strong_trend:
                    result.confirmed_regime = RegimeType.STRONG_TREND.value
                elif trend.is_trending:
                    result.confirmed_regime = RegimeType.MODERATE_TREND.value
                else:
                    result.confirmed_regime = RegimeType.WEAK_TREND.value
            else:
                if volatility.is_bb_squeeze or breakout.is_tight_range:
                    result.confirmed_regime = RegimeType.RANGE_BOUND.value
                else:
                    result.confirmed_regime = RegimeType.UNKNOWN.value
            
            result.confirmation_details = {
                'adx': {'confirms': result.adx_confirms, 'value': trend.adx},
                'ema': {'confirms': result.ema_confirms, 'slope': trend.ema_analysis.slope},
                'momentum': {'confirms': result.momentum_confirms, 'direction': trend.momentum_direction},
                'bb': {'confirms': result.bb_confirms, 'squeeze': volatility.is_bb_squeeze}
            }
            
        except Exception as e:
            logger.error(f"Error in multi-indicator confirmation: {str(e)}")
        
        return result
    
    def _detect_regime_transition(self, current_regime: str) -> RegimeTransition:
        """
        Detect regime transitions dan track duration.
        
        Args:
            current_regime: Current regime type
            
        Returns:
            RegimeTransition dengan hasil analisis
        """
        result = RegimeTransition()
        result.current_regime = current_regime
        
        try:
            self._current_candle_count += 1
            
            if self._last_regime is not None:
                result.previous_regime = self._last_regime.regime_type
            else:
                result.previous_regime = 'unknown'
            
            if result.previous_regime != current_regime and result.previous_regime != 'unknown':
                result.transition_detected = True
                result.transition_type = f"{result.previous_regime}_to_{current_regime}"
                
                self._regime_history.append((result.previous_regime, current_regime))
                if len(self._regime_history) > 10:
                    self._regime_history = self._regime_history[-10:]
                
                old_duration = self._current_candle_count - self._regime_start_candle
                self._regime_start_candle = self._current_candle_count
                
                alert = f"Regime change detected: {result.previous_regime} → {current_regime} (after {old_duration} candles)"
                result.alerts.append(alert)
                logger.info(alert)
                
                if result.previous_regime == RegimeType.RANGE_BOUND.value and current_regime in [
                    RegimeType.STRONG_TREND.value, RegimeType.MODERATE_TREND.value
                ]:
                    result.alerts.append("Potential breakout: Range → Trend transition")
                
                if result.previous_regime in [RegimeType.STRONG_TREND.value, RegimeType.MODERATE_TREND.value] and \
                   current_regime == RegimeType.RANGE_BOUND.value:
                    result.alerts.append("Trend exhaustion: Trend → Range transition")
            
            result.regime_duration_candles = self._current_candle_count - self._regime_start_candle
            
            if self._regime_start_candle == 0:
                result.regime_start_time = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Error in regime transition detection: {str(e)}")
        
        return result
    
    def _determine_regime_type(self, 
                                trend: TrendAnalysis, 
                                volatility: VolatilityAnalysis, 
                                breakout: BreakoutAnalysis,
                                df: Optional[pd.DataFrame] = None) -> str:
        """
        Tentukan tipe market regime dengan multi-indicator confirmation.
        
        Priority:
        1. Breakout (jika ada breakout aktif dengan probabilitas tinggi)
        2. High Volatility (jika volatilitas sangat tinggi)
        3. Strong Trend (jika ADX > 35 dengan confirmation)
        4. Moderate Trend (jika ADX 25-35 dengan confirmation)
        5. Weak Trend (jika ADX 15-25)
        6. Range Bound (default untuk ADX < 15 atau BB squeeze)
        """
        confirmation = self._get_multi_indicator_confirmation(trend, volatility, breakout)
        
        if (breakout.is_breakout_up or breakout.is_breakout_down) and breakout.breakout_probability > 0.5:
            return RegimeType.BREAKOUT.value
        
        if volatility.is_high_volatility and volatility.atr_ratio_50 > 2.0:
            return RegimeType.HIGH_VOLATILITY.value
        
        is_range_bound = False
        if df is not None:
            is_range_bound = self._check_range_bound(df, volatility)
        if is_range_bound and trend.adx < self.ADX_RANGING:
            return RegimeType.RANGE_BOUND.value
        
        if confirmation.is_confirmed:
            if trend.is_strong_trend:
                return RegimeType.STRONG_TREND.value
            elif trend.is_trending and trend.trend_strength == 'moderate':
                return RegimeType.MODERATE_TREND.value
        
        if trend.is_strong_trend and trend.confirmation_count >= 2:
            return RegimeType.STRONG_TREND.value
        
        if trend.is_trending and trend.trend_strength == 'moderate':
            return RegimeType.MODERATE_TREND.value
        
        if trend.trend_strength == 'weak':
            return RegimeType.WEAK_TREND.value
        
        return RegimeType.RANGE_BOUND.value
    
    def _determine_bias(self, 
                        trend: TrendAnalysis, 
                        price_pos: PricePositionAnalysis, 
                        breakout: BreakoutAnalysis) -> str:
        """
        Tentukan bias arah trading berdasarkan confirmed direction.
        
        Priority:
        1. Breakout direction (jika ada breakout aktif)
        2. Confirmed trend direction (multi-indicator)
        3. Strong trend direction (jika ADX tinggi)
        4. Price position bias (support/resistance proximity)
        """
        if breakout.is_breakout_up and breakout.breakout_probability > 0.5:
            return BiasType.BUY.value
        if breakout.is_breakout_down and breakout.breakout_probability > 0.5:
            return BiasType.SELL.value
        
        if trend.confirmed_direction != 'neutral' and trend.confirmation_count >= 2:
            if trend.confirmed_direction == 'bullish':
                return BiasType.BUY.value
            elif trend.confirmed_direction == 'bearish':
                return BiasType.SELL.value
        
        if trend.is_strong_trend or (trend.is_trending and trend.trend_strength == 'moderate'):
            if trend.trend_direction == 'bullish':
                return BiasType.BUY.value
            elif trend.trend_direction == 'bearish':
                return BiasType.SELL.value
        
        return price_pos.position_bias
    
    def _calculate_strictness_level(self, 
                                    trend: TrendAnalysis, 
                                    volatility: VolatilityAnalysis,
                                    breakout: BreakoutAnalysis,
                                    confirmation: MultiIndicatorConfirmation) -> float:
        """
        Hitung level strictness untuk confluence requirement.
        
        Strictness Range: 0.5 - 2.0
        - 0.5: Sangat longgar (strong trend, confirmed, clear breakout)
        - 1.0: Normal (moderate conditions)
        - 2.0: Sangat ketat (weak trend, low volatility, no confirmation)
        """
        strictness = 1.0
        
        if trend.is_strong_trend:
            strictness -= 0.3
        elif trend.trend_strength == 'moderate':
            strictness -= 0.1
        elif trend.trend_strength == 'weak':
            strictness += 0.3
        else:
            strictness += 0.5
        
        if confirmation.is_confirmed:
            strictness -= 0.2
        else:
            strictness += 0.2
        
        if volatility.is_low_volatility:
            strictness += 0.3
        elif volatility.is_high_volatility:
            strictness -= 0.1
        
        if breakout.breakout_probability > 0.7:
            strictness -= 0.2
        elif breakout.is_tight_range or volatility.is_bb_squeeze:
            strictness += 0.1
        
        return max(0.5, min(2.0, strictness))
    
    def _generate_recommendations(self, regime_type: str, 
                                   trend: TrendAnalysis,
                                   volatility: VolatilityAnalysis) -> Tuple[List[str], List[str]]:
        """Generate strategi yang direkomendasikan dan yang harus dihindari."""
        recommended = []
        avoid = []
        
        if regime_type == RegimeType.STRONG_TREND.value:
            recommended = ['trend_following', 'breakout_continuation', 'pullback_entry']
            avoid = ['counter_trend', 'mean_reversion', 'range_trading']
        
        elif regime_type == RegimeType.MODERATE_TREND.value:
            recommended = ['trend_following', 'pullback_entry', 'swing_trading']
            avoid = ['aggressive_scalping', 'pure_range_trading']
        
        elif regime_type == RegimeType.RANGE_BOUND.value:
            recommended = ['mean_reversion', 'range_trading', 'support_resistance_bounce']
            avoid = ['trend_following', 'breakout_anticipation']
        
        elif regime_type == RegimeType.BREAKOUT.value:
            recommended = ['breakout_entry', 'momentum_trading', 'quick_scalp']
            avoid = ['mean_reversion', 'fade_the_move', 'range_trading']
        
        elif regime_type == RegimeType.HIGH_VOLATILITY.value:
            recommended = ['scalping_tight_sl', 'momentum_trading', 'quick_profit_taking']
            avoid = ['wide_stop_trades', 'overnight_positions', 'aggressive_sizing']
        
        elif regime_type == RegimeType.WEAK_TREND.value:
            recommended = ['high_confluence_only', 'small_position', 'quick_exit']
            avoid = ['aggressive_entry', 'large_positions', 'trend_following']
        
        else:
            recommended = ['conservative', 'wait_for_clarity']
            avoid = ['aggressive_trading']
        
        if volatility.is_high_volatility:
            if 'tight_sl' not in str(recommended):
                recommended.append('reduce_position_size')
            if 'wide_stop_trades' not in avoid:
                avoid.append('wide_stop_trades')
        
        if volatility.volatility_trend == VolatilityTrend.INCREASING.value:
            if 'monitor_volatility' not in recommended:
                recommended.append('monitor_volatility')
        
        return recommended, avoid
    
    def _calculate_confidence(self, 
                              trend: TrendAnalysis, 
                              volatility: VolatilityAnalysis,
                              price_pos: PricePositionAnalysis,
                              breakout: BreakoutAnalysis,
                              confirmation: MultiIndicatorConfirmation) -> float:
        """Hitung confidence level untuk analisis (0.0 - 1.0)."""
        confidence = 0.4
        
        if trend.adx > 10:
            confidence += 0.1
        if trend.adx > 25:
            confidence += 0.1
        
        if confirmation.is_confirmed:
            confidence += 0.15
        
        confidence += min(0.1, trend.confirmation_count * 0.05)
        
        if volatility.current_atr > 0 and volatility.historical_atr_50 > 0:
            confidence += 0.05
        
        if price_pos.nearest_support > 0 and price_pos.nearest_resistance > 0:
            confidence += 0.05
        
        if len(price_pos.congestion_zones) >= 3:
            confidence += 0.05
        
        if breakout.volume_ratio > 0:
            confidence += 0.05
        
        return min(1.0, confidence)
    
    def get_regime(self, 
                   indicators: Dict[str, Any], 
                   m1_df: Optional[pd.DataFrame] = None, 
                   m5_df: Optional[pd.DataFrame] = None) -> MarketRegime:
        """
        Analisis kondisi pasar dan return MarketRegime.
        
        Args:
            indicators: Dict dengan data indikator (dari IndicatorEngine)
            m1_df: DataFrame M1 untuk analisis detail
            m5_df: DataFrame M5 untuk konfirmasi (opsional)
            
        Returns:
            MarketRegime dengan hasil analisis lengkap
        """
        regime = MarketRegime()
        regime.analysis_timestamp = datetime.now().isoformat()
        
        try:
            df = m1_df
            if df is None or len(df) == 0:
                if m5_df is not None and len(m5_df) > 0:
                    df = m5_df
                    regime.warnings.append("Using M5 data as fallback (M1 not available)")
                else:
                    regime.warnings.append("No price data available for analysis")
                    logger.warning("No DataFrame provided for market regime analysis")
                    return regime
            
            regime.trend_analysis = self._analyze_trend_strength(df)
            regime.volatility_analysis = self._analyze_volatility(df)
            regime.price_position = self._analyze_price_position(df)
            regime.breakout_analysis = self._analyze_breakout(df)
            
            if m5_df is not None and len(m5_df) > 0:
                m5_trend = self._analyze_trend_strength(m5_df)
                
                if m5_trend.trend_strength != regime.trend_analysis.trend_strength:
                    regime.warnings.append(f"M1/M5 trend strength mismatch: M1={regime.trend_analysis.trend_strength}, M5={m5_trend.trend_strength}")
                
                if m5_trend.trend_direction != regime.trend_analysis.trend_direction:
                    regime.warnings.append(f"M1/M5 trend direction mismatch: M1={regime.trend_analysis.trend_direction}, M5={m5_trend.trend_direction}")
            
            if indicators:
                if 'adx' in indicators and is_valid_number(indicators.get('adx')):
                    indicator_adx = safe_float(indicators['adx'], 0.0)
                    if abs(indicator_adx - regime.trend_analysis.adx) > 5:
                        regime.trend_analysis.adx = (regime.trend_analysis.adx + indicator_adx) / 2
                
                if 'atr' in indicators and is_valid_number(indicators.get('atr')):
                    indicator_atr = safe_float(indicators['atr'], 0.0)
                    if indicator_atr > 0:
                        regime.volatility_analysis.current_atr = indicator_atr
            
            regime.multi_confirmation = self._get_multi_indicator_confirmation(
                regime.trend_analysis,
                regime.volatility_analysis,
                regime.breakout_analysis
            )
            
            regime.regime_type = self._determine_regime_type(
                regime.trend_analysis,
                regime.volatility_analysis,
                regime.breakout_analysis,
                df
            )
            
            regime.regime_transition = self._detect_regime_transition(regime.regime_type)
            
            if regime.regime_transition.alerts:
                regime.warnings.extend(regime.regime_transition.alerts)
            
            regime.bias = self._determine_bias(
                regime.trend_analysis,
                regime.price_position,
                regime.breakout_analysis
            )
            
            regime.strictness_level = self._calculate_strictness_level(
                regime.trend_analysis,
                regime.volatility_analysis,
                regime.breakout_analysis,
                regime.multi_confirmation
            )
            
            regime.recommended_strategies, regime.avoid_strategies = self._generate_recommendations(
                regime.regime_type,
                regime.trend_analysis,
                regime.volatility_analysis
            )
            
            regime.confidence = self._calculate_confidence(
                regime.trend_analysis,
                regime.volatility_analysis,
                regime.price_position,
                regime.breakout_analysis,
                regime.multi_confirmation
            )
            
            self._last_regime = regime
            
            logger.debug(f"Market Regime: {regime.regime_type} | Bias: {regime.bias} | "
                       f"Strictness: {regime.strictness_level:.2f} | Confidence: {regime.confidence:.2f} | "
                       f"Confirmations: {regime.multi_confirmation.confirmation_count}/3")
            
        except Exception as e:
            logger.error(f"Error in get_regime: {str(e)}")
            regime.warnings.append(f"Analysis error: {str(e)}")
        
        return regime
    
    def get_last_regime(self) -> Optional[MarketRegime]:
        """Return hasil analisis regime terakhir."""
        return self._last_regime
    
    def get_regime_history(self) -> List[Tuple[str, str]]:
        """Return history of regime transitions."""
        return self._regime_history.copy()
    
    def is_favorable_for_entry(self, signal_type: str) -> Tuple[bool, str]:
        """
        Cek apakah kondisi market favorable untuk entry signal tertentu.
        
        Args:
            signal_type: 'BUY' atau 'SELL'
            
        Returns:
            Tuple[bool, str]: (is_favorable, reason)
        """
        if self._last_regime is None:
            return False, "No market regime analysis available"
        
        regime = self._last_regime
        
        if regime.regime_type == RegimeType.UNKNOWN.value:
            return False, "Market regime unknown"
        
        if not regime.multi_confirmation.is_confirmed and regime.regime_type not in [
            RegimeType.RANGE_BOUND.value, RegimeType.BREAKOUT.value
        ]:
            return False, "Multi-indicator confirmation not met"
        
        if regime.strictness_level > 1.5:
            return False, f"High strictness required ({regime.strictness_level:.2f}), need more confluence"
        
        if regime.regime_type == RegimeType.RANGE_BOUND.value:
            if signal_type == BiasType.BUY.value and regime.price_position.price_position != 'near_support':
                return False, "Range-bound market: BUY only near support"
            if signal_type == BiasType.SELL.value and regime.price_position.price_position != 'near_resistance':
                return False, "Range-bound market: SELL only near resistance"
        
        if regime.regime_type == RegimeType.STRONG_TREND.value:
            if regime.bias != signal_type and regime.bias != BiasType.NEUTRAL.value:
                return False, f"Strong trend favors {regime.bias}, not {signal_type}"
        
        return True, f"Favorable for {signal_type} in {regime.regime_type} regime"
    
    def get_adjusted_sl_multiplier(self) -> float:
        """Return SL multiplier berdasarkan kondisi volatilitas."""
        if self._last_regime is None:
            return 1.0
        return self._last_regime.volatility_analysis.suggested_sl_multiplier
    
    def get_position_size_factor(self) -> float:
        """
        Return faktor untuk adjustment position size berdasarkan regime.
        
        Returns:
            float: Multiplier untuk position size (0.5 - 1.5)
        """
        if self._last_regime is None:
            return 1.0
        
        regime = self._last_regime
        factor = 1.0
        
        if regime.volatility_analysis.is_high_volatility:
            factor *= 0.7
        
        if regime.strictness_level > 1.3:
            factor *= 0.8
        
        if not regime.multi_confirmation.is_confirmed:
            factor *= 0.9
        
        if regime.regime_type == RegimeType.STRONG_TREND.value and regime.confidence > 0.7:
            factor *= 1.1
        
        return max(0.5, min(1.5, factor))
