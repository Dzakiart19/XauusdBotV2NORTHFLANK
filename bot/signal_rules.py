"""
Aggressive Signal Rules untuk Bot Trading XAUUSD.

Modul ini berisi 4 aturan sinyal untuk aggressive scalping:
1. M1 Quick Scalp Signal (20-50+ signals/hari)
2. M5 Swing Entry (5-15 signals/hari)
3. S/R Mean-Reversion (3-8 signals/hari)
4. Breakout Confirmation (2-5 signals/hari)

Enhanced Features:
- Weighted confluence scoring system
- Market regime awareness
- Secondary confirmation requirements
- Dynamic confidence calculation
- Signal quality grading (A, B, C)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Any
from enum import Enum
from datetime import datetime

from bot.logger import setup_logger
from bot.strategy import safe_float, is_valid_number, safe_divide
from bot.indicators import IndicatorEngine, safe_series_operation
from bot.market_regime import MarketRegimeDetector, MarketRegime, RegimeType

logger = setup_logger('SignalRules')


class SignalRuleError(Exception):
    """Base exception for signal rule errors"""
    pass


class RuleType(str, Enum):
    """Enum untuk tipe signal rule"""
    M1_SCALP = 'M1_SCALP'
    M5_SWING = 'M5_SWING'
    SR_REVERSION = 'SR_REVERSION'
    BREAKOUT = 'BREAKOUT'


class SignalType(str, Enum):
    """Enum untuk tipe signal"""
    BUY = 'BUY'
    SELL = 'SELL'
    NONE = 'NONE'


class QualityGrade(str, Enum):
    """Enum untuk signal quality grades"""
    A = 'A'
    B = 'B'
    C = 'C'
    D = 'D'


class ConfluenceType(str, Enum):
    """Enum untuk confluence types dengan weights"""
    TREND_ALIGNMENT = 'trend_alignment'
    RSI_CONFIRMATION = 'rsi_confirmation'
    VOLUME_CONFIRMATION = 'volume_confirmation'
    SR_PROXIMITY = 'sr_proximity'
    EMA_ALIGNMENT = 'ema_alignment'
    MACD_CONFIRMATION = 'macd_confirmation'
    ADX_CONFIRMATION = 'adx_confirmation'
    STOCHASTIC_CONFIRMATION = 'stochastic_confirmation'
    WICK_REJECTION = 'wick_rejection'
    BREAKOUT_CONFIRMATION = 'breakout_confirmation'


@dataclass
class WeightedConfluence:
    """Dataclass untuk weighted confluence tracking"""
    confluence_type: str = ''
    description: str = ''
    weight: float = 1.0
    is_active: bool = False
    
    def get_weighted_score(self) -> float:
        return self.weight if self.is_active else 0.0


@dataclass
class SignalResult:
    """Dataclass untuk hasil signal dari setiap rule"""
    signal_type: str = 'NONE'
    rule_name: str = ''
    confidence: float = 0.0
    sl_pips: float = 0.0
    tp_pips: float = 0.0
    reason: str = ''
    confluence_count: int = 0
    confluence_details: List[str] = field(default_factory=list)
    entry_price: float = 0.0
    quality_grade: str = 'D'
    weighted_confluence_score: float = 0.0
    regime_type: str = 'unknown'
    regime_multiplier: float = 1.0
    secondary_confirmations: Dict[str, bool] = field(default_factory=dict)
    time_factor: float = 1.0
    volatility_factor: float = 1.0
    raw_confidence: float = 0.0
    
    def is_valid(self) -> bool:
        """Check if signal is valid (not NONE)"""
        return self.signal_type in ('BUY', 'SELL') and self.confidence > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert SignalResult to dictionary"""
        return {
            'signal_type': self.signal_type,
            'rule_name': self.rule_name,
            'confidence': self.confidence,
            'sl_pips': self.sl_pips,
            'tp_pips': self.tp_pips,
            'reason': self.reason,
            'confluence_count': self.confluence_count,
            'confluence_details': self.confluence_details,
            'entry_price': self.entry_price,
            'quality_grade': self.quality_grade,
            'weighted_confluence_score': self.weighted_confluence_score,
            'regime_type': self.regime_type,
            'regime_multiplier': self.regime_multiplier,
            'secondary_confirmations': self.secondary_confirmations,
            'time_factor': self.time_factor,
            'volatility_factor': self.volatility_factor,
            'raw_confidence': self.raw_confidence
        }


class AggressiveSignalRules:
    """
    Class untuk menghitung 4 aggressive signal rules untuk XAUUSD scalping.
    
    Rules:
    1. M1 Quick Scalp (20-50+ signals/day) - Ultra short-term scalping
    2. M5 Swing Entry (5-15 signals/day) - Short-term swing trades
    3. S/R Mean-Reversion (3-8 signals/day) - Range-bound market reversal
    4. Breakout Confirmation (2-5 signals/day) - Breakout continuation
    
    Enhanced Features:
    - Weighted confluence scoring (1.0 - 2.0 weights)
    - Market regime awareness
    - Secondary confirmation requirements
    - Dynamic confidence calculation
    - Signal quality grading
    """
    
    PIP_VALUE_XAUUSD = 0.1
    
    CONFLUENCE_WEIGHTS = {
        ConfluenceType.TREND_ALIGNMENT.value: 2.0,
        ConfluenceType.SR_PROXIMITY.value: 1.8,
        ConfluenceType.RSI_CONFIRMATION.value: 1.5,
        ConfluenceType.EMA_ALIGNMENT.value: 1.5,
        ConfluenceType.VOLUME_CONFIRMATION.value: 1.3,
        ConfluenceType.MACD_CONFIRMATION.value: 1.3,
        ConfluenceType.ADX_CONFIRMATION.value: 1.4,
        ConfluenceType.STOCHASTIC_CONFIRMATION.value: 1.2,
        ConfluenceType.WICK_REJECTION.value: 1.3,
        ConfluenceType.BREAKOUT_CONFIRMATION.value: 1.6,
    }
    
    REGIME_MULTIPLIERS = {
        RegimeType.STRONG_TREND.value: 1.3,
        RegimeType.MODERATE_TREND.value: 1.1,
        RegimeType.WEAK_TREND.value: 0.5,
        RegimeType.RANGE_BOUND.value: 0.6,
        RegimeType.BREAKOUT.value: 1.2,
        RegimeType.HIGH_VOLATILITY.value: 0.7,
        RegimeType.UNKNOWN.value: 0.6,
    }
    
    MIN_WEIGHTED_CONFLUENCE_SCORE = 6.0
    
    MIN_CONFIDENCE_THRESHOLD = 0.80
    QUALITY_GRADE_FILTER = ['A', 'B']
    
    LOW_LIQUIDITY_HOURS = [21, 22, 23, 0, 1, 2, 3, 4, 5]
    HIGH_LIQUIDITY_HOURS = [8, 9, 10, 13, 14, 15, 16]
    
    M1_SL_MIN = 10.0
    M1_SL_MAX = 15.0
    M1_TP_MIN = 20.0
    M1_TP_MAX = 35.0
    M1_MIN_CONFLUENCE = 4
    M1_MIN_SECONDARY = 3
    
    M5_SL_MIN = 15.0
    M5_SL_MAX = 22.0
    M5_TP_MIN = 35.0
    M5_TP_MAX = 55.0
    M5_MIN_CONFLUENCE = 4
    M5_MIN_SECONDARY = 3
    M5_ADX_THRESHOLD = 22
    
    SR_SL_MIN = 10.0
    SR_SL_MAX = 15.0
    SR_TP_MIN = 20.0
    SR_TP_MAX = 40.0
    SR_MIN_CONFLUENCE = 2
    SR_ADX_MAX = 20
    SR_ATR_MAX_RATIO = 1.5
    SR_PROXIMITY_PIPS = 5.0
    
    BO_SL_MIN = 12.0
    BO_SL_MAX = 18.0
    BO_TP_MIN = 40.0
    BO_TP_MAX = 80.0
    BO_MIN_CONFLUENCE = 3
    BO_ADX_TARGET = 25
    BO_VOLUME_THRESHOLD = 1.5
    BO_SR_PROXIMITY_PIPS = 5.0
    
    def __init__(self, config, indicator_engine: Optional[IndicatorEngine] = None,
                 regime_detector: Optional[MarketRegimeDetector] = None):
        """
        Inisialisasi AggressiveSignalRules.
        
        Args:
            config: Objek konfigurasi bot
            indicator_engine: Instance IndicatorEngine (opsional)
            regime_detector: Instance MarketRegimeDetector (opsional)
        """
        self.config = config
        self.indicator_engine = indicator_engine or IndicatorEngine(config)
        self.regime_detector = regime_detector or MarketRegimeDetector(config, self.indicator_engine)
        self._current_regime: Optional[MarketRegime] = None
        self._quality_stats: Dict[str, int] = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        logger.info("AggressiveSignalRules initialized with enhanced confluence weighting and regime awareness")
    
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
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, 
                                    std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            df: DataFrame with OHLC data
            period: Period for moving average (default: 20)
            std_dev: Standard deviation multiplier (default: 2.0)
        
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        if df is None or len(df) < period:
            empty = pd.Series([0.0])
            return empty, empty, empty
        
        try:
            close = df['close'] if 'close' in df.columns else None
            if close is None:
                empty = pd.Series([0.0])
                return empty, empty, empty
            
            middle_band = pd.Series(close.rolling(window=period, min_periods=1).mean())
            std = pd.Series(close.rolling(window=period, min_periods=1).std())
            
            upper_band = pd.Series(middle_band + (std * std_dev))
            lower_band = pd.Series(middle_band - (std * std_dev))
            
            middle_band = pd.Series(middle_band.replace([np.inf, -np.inf], np.nan).fillna(close))
            upper_band = pd.Series(upper_band.replace([np.inf, -np.inf], np.nan).fillna(close))
            lower_band = pd.Series(lower_band.replace([np.inf, -np.inf], np.nan).fillna(close))
            
            return upper_band, middle_band, lower_band
            
        except Exception as e:
            logger.warning(f"Gagal menghitung Bollinger Bands: {str(e)}")
            empty = pd.Series([0.0])
            return empty, empty, empty
    
    def _get_candle_wick_data(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Get candle wick data for current candle.
        
        Returns:
            Dict with upper_wick, lower_wick, body_size, wick_ratio
        """
        result = {
            'upper_wick': 0.0,
            'lower_wick': 0.0,
            'body_size': 0.0,
            'wick_ratio': 0.0,
            'is_bullish': False
        }
        
        if df is None or len(df) < 1:
            return result
        
        try:
            last_candle = df.iloc[-1]
            open_price = safe_float(last_candle.get('open', 0), 0.0)
            high_price = safe_float(last_candle.get('high', 0), 0.0)
            low_price = safe_float(last_candle.get('low', 0), 0.0)
            close_price = safe_float(last_candle.get('close', 0), 0.0)
            
            if open_price <= 0 or close_price <= 0:
                return result
            
            result['is_bullish'] = close_price > open_price
            body_top = max(open_price, close_price)
            body_bottom = min(open_price, close_price)
            
            result['upper_wick'] = high_price - body_top
            result['lower_wick'] = body_bottom - low_price
            result['body_size'] = body_top - body_bottom
            
            total_range = high_price - low_price
            if total_range > 0:
                result['wick_ratio'] = (result['upper_wick'] + result['lower_wick']) / total_range
            
            return result
            
        except Exception as e:
            logger.warning(f"Error getting wick data: {str(e)}")
            return result
    
    def _calculate_sl_tp(self, base_sl: float, base_tp: float, 
                         sl_min: float, sl_max: float,
                         tp_min: float, tp_max: float,
                         volatility_multiplier: float = 1.0) -> Tuple[float, float]:
        """
        Calculate SL and TP with volatility adjustment.
        
        Args:
            base_sl: Base stop loss in pips
            base_tp: Base take profit in pips
            sl_min, sl_max: SL range limits
            tp_min, tp_max: TP range limits
            volatility_multiplier: Multiplier based on current volatility
        
        Returns:
            Tuple of (sl_pips, tp_pips)
        """
        adjusted_sl = base_sl * volatility_multiplier
        adjusted_tp = base_tp * volatility_multiplier
        
        sl_pips = max(sl_min, min(sl_max, adjusted_sl))
        tp_pips = max(tp_min, min(tp_max, adjusted_tp))
        
        return sl_pips, tp_pips
    
    def _calculate_weighted_confluence(self, confluences: List[WeightedConfluence]) -> Tuple[float, int, List[str]]:
        """
        Calculate weighted confluence score from list of confluences.
        
        Args:
            confluences: List of WeightedConfluence objects
        
        Returns:
            Tuple of (weighted_score, count, details)
        """
        active_confluences = [c for c in confluences if c.is_active]
        weighted_score = sum(c.get_weighted_score() for c in active_confluences)
        count = len(active_confluences)
        details = [f"{c.description} (w:{c.weight:.1f})" for c in active_confluences]
        
        return weighted_score, count, details
    
    def _get_confluence_weight(self, confluence_type: str) -> float:
        """Get weight for a specific confluence type."""
        return self.CONFLUENCE_WEIGHTS.get(confluence_type, 1.0)
    
    def _update_market_regime(self, df_m1: Optional[pd.DataFrame] = None,
                               df_m5: Optional[pd.DataFrame] = None,
                               indicators: Optional[Dict[str, Any]] = None) -> MarketRegime:
        """
        Update and return current market regime.
        
        Args:
            df_m1: DataFrame M1 timeframe
            df_m5: DataFrame M5 timeframe
            indicators: Current indicators dict
        
        Returns:
            MarketRegime object
        """
        try:
            if indicators is None:
                indicators = {}
            
            self._current_regime = self.regime_detector.get_regime(
                indicators=indicators,
                m1_df=df_m1,
                m5_df=df_m5
            )
            return self._current_regime
        except Exception as e:
            logger.warning(f"Failed to update market regime: {str(e)}")
            return MarketRegime()
    
    def _get_regime_multiplier(self, regime_type: str) -> float:
        """
        Get confidence multiplier based on market regime.
        
        Args:
            regime_type: Current market regime type
        
        Returns:
            Multiplier value (0.7 - 1.2)
        """
        return self.REGIME_MULTIPLIERS.get(regime_type, 1.0)
    
    def _is_regime_favorable_for_rule(self, rule_type: RuleType) -> Tuple[bool, str]:
        """
        Check if current regime is favorable for a specific rule type.
        
        Args:
            rule_type: The signal rule type
        
        Returns:
            Tuple of (is_favorable, reason)
        """
        if self._current_regime is None:
            return True, "No regime data available"
        
        regime = self._current_regime.regime_type
        
        if rule_type == RuleType.M1_SCALP:
            if regime == RegimeType.RANGE_BOUND.value:
                return False, f"M1_SCALP blocked in range_bound market - choppy conditions"
            if regime == RegimeType.HIGH_VOLATILITY.value:
                return False, f"M1_SCALP blocked in high_volatility market - too risky"
            if regime == RegimeType.UNKNOWN.value:
                return False, f"M1_SCALP blocked - unknown market conditions"
        
        if rule_type == RuleType.M5_SWING:
            if regime == RegimeType.RANGE_BOUND.value:
                return False, f"M5_SWING blocked in range_bound market"
            if regime == RegimeType.UNKNOWN.value:
                return False, f"M5_SWING blocked - unknown market conditions"
        
        if rule_type == RuleType.BREAKOUT:
            if regime == RegimeType.RANGE_BOUND.value:
                return False, f"BREAKOUT blocked in range_bound market"
            if regime == RegimeType.WEAK_TREND.value:
                return False, f"BREAKOUT blocked in weak_trend - insufficient momentum"
        
        if rule_type == RuleType.SR_REVERSION:
            if regime in [RegimeType.STRONG_TREND.value, RegimeType.BREAKOUT.value]:
                return False, f"SR_REVERSION not recommended in {regime} market"
        
        return True, f"Regime {regime} is compatible with {rule_type.value}"
    
    def _check_regime_alignment(self, signal_type: str, rule_type: RuleType) -> Tuple[bool, str]:
        """
        Check if signal direction aligns with market regime bias.
        
        IMPORTANT: Only BLOCKS signals with HIGH CONFIDENCE conflicting bias.
        Allows signals when regime uncertain or data unavailable (with warning).
        
        Args:
            signal_type: 'BUY' or 'SELL'
            rule_type: The signal rule type
            
        Returns:
            Tuple of (is_aligned, reason)
        """
        if self._current_regime is None:
            logger.warning(f"Regime unavailable - allowing {signal_type} with caution")
            return True, f"⚠️ Regime unavailable - {signal_type} allowed"
        
        bias = self._current_regime.bias
        regime = self._current_regime.regime_type
        confidence = self._current_regime.confidence
        
        if confidence < 0.6:
            return True, f"⚠️ Low confidence ({confidence:.2f}) - {signal_type} allowed with caution"
        
        if rule_type in [RuleType.M1_SCALP, RuleType.M5_SWING, RuleType.BREAKOUT]:
            if signal_type == 'BUY' and bias == 'SELL' and confidence >= 0.80:
                return False, f"🚫 BLOCKED: BUY conflicts with strong SELL bias (conf: {confidence:.2f})"
            if signal_type == 'SELL' and bias == 'BUY' and confidence >= 0.80:
                return False, f"🚫 BLOCKED: SELL conflicts with strong BUY bias (conf: {confidence:.2f})"
        
        if rule_type == RuleType.SR_REVERSION:
            if regime in [RegimeType.STRONG_TREND.value, RegimeType.MODERATE_TREND.value]:
                if signal_type == 'BUY' and bias == 'SELL' and confidence >= 0.85:
                    return False, f"🚫 BLOCKED: Reversion BUY in strong SELL trend"
                if signal_type == 'SELL' and bias == 'BUY' and confidence >= 0.85:
                    return False, f"🚫 BLOCKED: Reversion SELL in strong BUY trend"
        
        if signal_type == 'BUY' and bias == 'BUY':
            return True, f"✅ BUY aligned with BUY bias (regime: {regime})"
        elif signal_type == 'SELL' and bias == 'SELL':
            return True, f"✅ SELL aligned with SELL bias (regime: {regime})"
        elif bias == 'NEUTRAL':
            return True, f"⚠️ Neutral bias - {signal_type} allowed with caution"
        
        return True, f"Signal validated (regime: {regime}, bias: {bias})"
    
    def _validate_signal_quality(self, result: SignalResult) -> Tuple[bool, str]:
        """
        Final validation to ensure signal meets quality standards.
        
        NOTE: This is a soft filter - logs warnings for low quality but only blocks
        very poor quality signals. This ensures signals still flow while flagging concerns.
        
        Args:
            result: SignalResult to validate
            
        Returns:
            Tuple of (is_valid, reason)
        """
        warnings = []
        
        if result.confidence < 0.70:
            return False, f"Very low confidence {result.confidence:.2f} - blocked"
        elif result.confidence < self.MIN_CONFIDENCE_THRESHOLD:
            warnings.append(f"Low confidence {result.confidence:.2f}")
        
        if result.quality_grade == 'D':
            return False, f"Grade D signal blocked - too risky"
        elif result.quality_grade not in self.QUALITY_GRADE_FILTER:
            warnings.append(f"Grade {result.quality_grade} (below ideal)")
        
        if result.weighted_confluence_score < 4.0:
            return False, f"Confluence score {result.weighted_confluence_score:.1f} too low"
        elif result.weighted_confluence_score < self.MIN_WEIGHTED_CONFLUENCE_SCORE:
            warnings.append(f"Confluence {result.weighted_confluence_score:.1f} below ideal")
        
        if warnings:
            logger.info(f"⚠️ Signal allowed with warnings: {', '.join(warnings)}")
            return True, f"Signal allowed with caution: {', '.join(warnings)}"
        
        return True, "Signal passed all quality checks"
    
    def _get_time_of_day_factor(self) -> float:
        """
        Calculate time-of-day factor for confidence adjustment.
        
        Returns:
            Factor between 0.7 (low liquidity) and 1.1 (high liquidity)
        """
        try:
            current_hour = datetime.utcnow().hour
            
            if current_hour in self.LOW_LIQUIDITY_HOURS:
                return 0.7
            elif current_hour in self.HIGH_LIQUIDITY_HOURS:
                return 1.1
            else:
                return 1.0
        except Exception:
            return 1.0
    
    def _get_volatility_factor(self, atr_ratio: float) -> float:
        """
        Calculate volatility factor for confidence adjustment.
        
        Args:
            atr_ratio: Current ATR / Average ATR ratio
        
        Returns:
            Factor between 0.7 (extreme volatility) and 1.0 (normal)
        """
        if atr_ratio > 2.0:
            return 0.7
        elif atr_ratio > 1.5:
            return 0.85
        elif atr_ratio < 0.5:
            return 0.9
        else:
            return 1.0
    
    def _calculate_final_confidence(self, 
                                     weighted_score: float,
                                     base_confidence: float,
                                     regime_multiplier: float,
                                     time_factor: float,
                                     volatility_factor: float) -> float:
        """
        Calculate final confidence score with all factors.
        
        Args:
            weighted_score: Weighted confluence score
            base_confidence: Base confidence from rule logic
            regime_multiplier: Regime-based multiplier
            time_factor: Time-of-day factor
            volatility_factor: Volatility-based factor
        
        Returns:
            Final confidence score (0-100)
        """
        score_bonus = min(0.2, (weighted_score - self.MIN_WEIGHTED_CONFLUENCE_SCORE) * 0.05)
        
        adjusted_confidence = base_confidence + score_bonus
        
        adjusted_confidence *= regime_multiplier
        adjusted_confidence *= time_factor
        adjusted_confidence *= volatility_factor
        
        final_confidence = max(0.0, min(1.0, adjusted_confidence))
        
        return final_confidence * 100
    
    def _get_quality_grade(self, confidence_pct: float) -> str:
        """
        Determine signal quality grade based on confidence.
        
        Args:
            confidence_pct: Confidence percentage (0-100)
        
        Returns:
            Quality grade: 'A', 'B', 'C', or 'D'
        """
        if confidence_pct > 80:
            return QualityGrade.A.value
        elif confidence_pct >= 60:
            return QualityGrade.B.value
        elif confidence_pct >= 40:
            return QualityGrade.C.value
        else:
            return QualityGrade.D.value
    
    def _track_quality_grade(self, grade: str):
        """Track quality grade for statistics."""
        if grade in self._quality_stats:
            self._quality_stats[grade] += 1
    
    def get_quality_statistics(self) -> Dict[str, int]:
        """Return quality grade statistics."""
        return self._quality_stats.copy()
    
    def _check_m1_secondary_confirmations(self, 
                                           has_rsi: bool, 
                                           has_ema: bool, 
                                           has_volume: bool) -> Tuple[bool, Dict[str, bool], int]:
        """
        Check secondary confirmations for M1_SCALP.
        Requires: Minimal 2 dari (RSI, EMA, Volume)
        
        Returns:
            Tuple of (is_valid, confirmations_dict, count)
        """
        confirmations = {
            'rsi': has_rsi,
            'ema': has_ema,
            'volume': has_volume
        }
        count = sum(confirmations.values())
        is_valid = count >= self.M1_MIN_SECONDARY
        return is_valid, confirmations, count
    
    def _check_m5_secondary_confirmations(self,
                                           has_rsi: bool,
                                           has_ema: bool,
                                           has_volume: bool,
                                           has_trend: bool,
                                           has_sr: bool) -> Tuple[bool, Dict[str, bool], int]:
        """
        Check secondary confirmations for M5_SWING.
        Requires: Minimal 3 dari (RSI, EMA, Volume, Trend, S/R)
        
        Returns:
            Tuple of (is_valid, confirmations_dict, count)
        """
        confirmations = {
            'rsi': has_rsi,
            'ema': has_ema,
            'volume': has_volume,
            'trend': has_trend,
            'sr': has_sr
        }
        count = sum(confirmations.values())
        is_valid = count >= self.M5_MIN_SECONDARY
        return is_valid, confirmations, count
    
    def _check_sr_secondary_confirmations(self,
                                           has_rsi_extreme: bool,
                                           has_price_at_sr: bool) -> Tuple[bool, Dict[str, bool], int]:
        """
        Check secondary confirmations for SR_REVERSION.
        Requires: Wajib RSI extreme + Price di S/R zone
        
        Returns:
            Tuple of (is_valid, confirmations_dict, count)
        """
        confirmations = {
            'rsi_extreme': has_rsi_extreme,
            'price_at_sr': has_price_at_sr
        }
        count = sum(confirmations.values())
        is_valid = has_rsi_extreme and has_price_at_sr
        return is_valid, confirmations, count
    
    def _check_breakout_secondary_confirmations(self,
                                                  has_volume_spike: bool,
                                                  has_adx_rising: bool) -> Tuple[bool, Dict[str, bool], int]:
        """
        Check secondary confirmations for BREAKOUT.
        Requires: Wajib Volume spike + ADX rising
        
        Returns:
            Tuple of (is_valid, confirmations_dict, count)
        """
        confirmations = {
            'volume_spike': has_volume_spike,
            'adx_rising': has_adx_rising
        }
        count = sum(confirmations.values())
        is_valid = has_volume_spike and has_adx_rising
        return is_valid, confirmations, count
    
    def check_m1_scalp_signal(self, df_m1: pd.DataFrame, 
                               df_m5: Optional[pd.DataFrame] = None,
                               df_h1: Optional[pd.DataFrame] = None) -> SignalResult:
        """
        RULE 1: M1 Quick Scalp Signal (20-50+ signals/hari)
        
        Enhanced with:
        - Weighted confluence scoring
        - Market regime awareness
        - Secondary confirmation requirements (2+ dari RSI, EMA, Volume)
        - Quality grading
        
        Trigger Conditions:
        - M1 candle close above EMA5 (BUY) or below EMA5 (SELL)
        - RSI(14) > 50 AND RSI < 75 (BUY) or RSI < 50 AND RSI > 25 (SELL)
        - Volume > avg * 1.1
        - MACD histogram positive & increasing (BUY) or negative & decreasing (SELL)
        
        Confirmation:
        - M5 EMA5 > EMA20 (uptrend) for BUY, EMA5 < EMA20 for SELL
        - H1 not at resistance (BUY) or not at support (SELL)
        
        Execute IF: Weighted confluence >= 4.0 AND 2+ secondary confirmations
        
        Args:
            df_m1: DataFrame M1 timeframe
            df_m5: DataFrame M5 timeframe (optional)
            df_h1: DataFrame H1 timeframe (optional)
        
        Returns:
            SignalResult with enhanced data
        """
        result = SignalResult(rule_name=RuleType.M1_SCALP.value)
        
        if df_m1 is None or len(df_m1) < 30:
            result.reason = "Insufficient M1 data"
            return result
        
        try:
            indicators = self.indicator_engine.get_indicators(df_m1)
            if indicators is None:
                result.reason = "Failed to calculate M1 indicators"
                return result
            
            self._update_market_regime(df_m1, df_m5, indicators)
            result.regime_type = self._current_regime.regime_type if self._current_regime else 'unknown'
            result.regime_multiplier = self._get_regime_multiplier(result.regime_type)
            
            is_favorable, regime_reason = self._is_regime_favorable_for_rule(RuleType.M1_SCALP)
            if not is_favorable:
                result.reason = regime_reason
                return result
            
            close = safe_float(indicators.get('close', 0), 0.0)
            ema_5 = safe_float(indicators.get('ema_5', 0), 0.0)
            ema_20 = safe_float(indicators.get('ema_20', 0), 0.0)
            rsi = safe_float(indicators.get('rsi', 50), 50.0)
            rsi_prev = safe_float(indicators.get('rsi_prev', 50), 50.0)
            volume = safe_float(indicators.get('volume', 0), 0.0)
            volume_avg = safe_float(indicators.get('volume_avg', 1), 1.0)
            atr = safe_float(indicators.get('atr', 1.0), 1.0)
            atr_avg = safe_float(indicators.get('atr_avg', atr), atr)
            macd_histogram = safe_float(indicators.get('macd_histogram', 0), 0.0)
            
            macd_line, macd_signal, macd_hist_series = self.indicator_engine.calculate_macd(df_m1)
            macd_hist_current = self._safe_get_value(macd_hist_series, -1, 0.0)
            macd_hist_prev = self._safe_get_value(macd_hist_series, -2, 0.0)
            
            if close <= 0 or ema_5 <= 0:
                result.reason = "Invalid price data"
                return result
            
            atr_ratio = atr / atr_avg if atr_avg > 0 else 1.0
            result.volatility_factor = self._get_volatility_factor(atr_ratio)
            result.time_factor = self._get_time_of_day_factor()
            
            buy_confluences: List[WeightedConfluence] = []
            sell_confluences: List[WeightedConfluence] = []
            
            has_rsi_buy = False
            has_rsi_sell = False
            has_ema_buy = False
            has_ema_sell = False
            has_volume_buy = False
            has_volume_sell = False
            
            if close > ema_5:
                buy_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.EMA_ALIGNMENT.value,
                    description="M1 close > EMA5",
                    weight=self._get_confluence_weight(ConfluenceType.EMA_ALIGNMENT.value),
                    is_active=True
                ))
                has_ema_buy = True
            elif close < ema_5:
                sell_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.EMA_ALIGNMENT.value,
                    description="M1 close < EMA5",
                    weight=self._get_confluence_weight(ConfluenceType.EMA_ALIGNMENT.value),
                    is_active=True
                ))
                has_ema_sell = True
            
            if 50 < rsi < 75:
                if rsi > rsi_prev:
                    buy_confluences.append(WeightedConfluence(
                        confluence_type=ConfluenceType.RSI_CONFIRMATION.value,
                        description=f"RSI bullish momentum ({rsi:.1f})",
                        weight=self._get_confluence_weight(ConfluenceType.RSI_CONFIRMATION.value),
                        is_active=True
                    ))
                    has_rsi_buy = True
            elif 25 < rsi < 50:
                if rsi < rsi_prev:
                    sell_confluences.append(WeightedConfluence(
                        confluence_type=ConfluenceType.RSI_CONFIRMATION.value,
                        description=f"RSI bearish momentum ({rsi:.1f})",
                        weight=self._get_confluence_weight(ConfluenceType.RSI_CONFIRMATION.value),
                        is_active=True
                    ))
                    has_rsi_sell = True
            
            volume_ratio = volume / volume_avg if volume_avg > 0 else 1.0
            if volume_ratio > 1.1:
                if close > ema_5:
                    buy_confluences.append(WeightedConfluence(
                        confluence_type=ConfluenceType.VOLUME_CONFIRMATION.value,
                        description=f"Volume spike ({volume_ratio:.2f}x)",
                        weight=self._get_confluence_weight(ConfluenceType.VOLUME_CONFIRMATION.value),
                        is_active=True
                    ))
                    has_volume_buy = True
                else:
                    sell_confluences.append(WeightedConfluence(
                        confluence_type=ConfluenceType.VOLUME_CONFIRMATION.value,
                        description=f"Volume spike ({volume_ratio:.2f}x)",
                        weight=self._get_confluence_weight(ConfluenceType.VOLUME_CONFIRMATION.value),
                        is_active=True
                    ))
                    has_volume_sell = True
            
            if macd_hist_current > 0 and macd_hist_current > macd_hist_prev:
                buy_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.MACD_CONFIRMATION.value,
                    description="MACD histogram positive & increasing",
                    weight=self._get_confluence_weight(ConfluenceType.MACD_CONFIRMATION.value),
                    is_active=True
                ))
            elif macd_hist_current < 0 and macd_hist_current < macd_hist_prev:
                sell_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.MACD_CONFIRMATION.value,
                    description="MACD histogram negative & decreasing",
                    weight=self._get_confluence_weight(ConfluenceType.MACD_CONFIRMATION.value),
                    is_active=True
                ))
            
            if ema_5 > ema_20:
                buy_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.TREND_ALIGNMENT.value,
                    description="EMA5 > EMA20 alignment",
                    weight=self._get_confluence_weight(ConfluenceType.TREND_ALIGNMENT.value),
                    is_active=True
                ))
                has_ema_buy = True
            elif ema_5 < ema_20:
                sell_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.TREND_ALIGNMENT.value,
                    description="EMA5 < EMA20 alignment",
                    weight=self._get_confluence_weight(ConfluenceType.TREND_ALIGNMENT.value),
                    is_active=True
                ))
                has_ema_sell = True
            
            m5_confirmation_buy = True
            m5_confirmation_sell = True
            if df_m5 is not None and len(df_m5) >= 30:
                m5_indicators = self.indicator_engine.get_indicators(df_m5)
                if m5_indicators:
                    m5_ema5 = safe_float(m5_indicators.get('ema_5', 0), 0.0)
                    m5_ema20 = safe_float(m5_indicators.get('ema_20', 0), 0.0)
                    if m5_ema5 > m5_ema20:
                        buy_confluences.append(WeightedConfluence(
                            confluence_type=ConfluenceType.TREND_ALIGNMENT.value,
                            description="M5 EMA5 > EMA20 confirmation",
                            weight=self._get_confluence_weight(ConfluenceType.TREND_ALIGNMENT.value) * 0.8,
                            is_active=True
                        ))
                        m5_confirmation_sell = False
                    elif m5_ema5 < m5_ema20:
                        sell_confluences.append(WeightedConfluence(
                            confluence_type=ConfluenceType.TREND_ALIGNMENT.value,
                            description="M5 EMA5 < EMA20 confirmation",
                            weight=self._get_confluence_weight(ConfluenceType.TREND_ALIGNMENT.value) * 0.8,
                            is_active=True
                        ))
                        m5_confirmation_buy = False
            
            h1_at_resistance = False
            h1_at_support = False
            if df_h1 is not None and len(df_h1) >= 30:
                sr_data = self.indicator_engine.calculate_micro_support_resistance(df_h1, lookback=50)
                h1_close = self._safe_get_value(df_h1['close'], -1, 0.0) if 'close' in df_h1.columns else 0.0
                resistance = sr_data.get('nearest_resistance', 0)
                support = sr_data.get('nearest_support', 0)
                
                if h1_close > 0:
                    dist_to_resistance = abs(resistance - h1_close) / self.PIP_VALUE_XAUUSD if resistance > 0 else 999
                    dist_to_support = abs(h1_close - support) / self.PIP_VALUE_XAUUSD if support > 0 else 999
                    
                    h1_at_resistance = dist_to_resistance < 10
                    h1_at_support = dist_to_support < 10
                    
                    if not h1_at_resistance:
                        buy_confluences.append(WeightedConfluence(
                            confluence_type=ConfluenceType.SR_PROXIMITY.value,
                            description="H1 not at resistance",
                            weight=self._get_confluence_weight(ConfluenceType.SR_PROXIMITY.value) * 0.5,
                            is_active=True
                        ))
                    if not h1_at_support:
                        sell_confluences.append(WeightedConfluence(
                            confluence_type=ConfluenceType.SR_PROXIMITY.value,
                            description="H1 not at support",
                            weight=self._get_confluence_weight(ConfluenceType.SR_PROXIMITY.value) * 0.5,
                            is_active=True
                        ))
            
            buy_score, buy_count, buy_details = self._calculate_weighted_confluence(buy_confluences)
            sell_score, sell_count, sell_details = self._calculate_weighted_confluence(sell_confluences)
            
            volatility_mult = min(1.5, max(0.7, atr_ratio)) if atr_avg > 0 else 1.0
            
            buy_secondary_valid, buy_secondary, buy_sec_count = self._check_m1_secondary_confirmations(
                has_rsi_buy, has_ema_buy, has_volume_buy
            )
            sell_secondary_valid, sell_secondary, sell_sec_count = self._check_m1_secondary_confirmations(
                has_rsi_sell, has_ema_sell, has_volume_sell
            )
            
            if (buy_score >= self.MIN_WEIGHTED_CONFLUENCE_SCORE and 
                buy_count >= self.M1_MIN_CONFLUENCE and 
                buy_count > sell_count and 
                m5_confirmation_buy and 
                not h1_at_resistance and
                buy_secondary_valid):
                
                is_aligned, alignment_reason = self._check_regime_alignment('BUY', RuleType.M1_SCALP)
                if not is_aligned:
                    result.reason = alignment_reason
                    return result
                
                result.signal_type = SignalType.BUY.value
                result.confluence_count = buy_count
                result.confluence_details = buy_details
                result.weighted_confluence_score = buy_score
                result.secondary_confirmations = buy_secondary
                
                base_confidence = min(1.0, 0.55 + (buy_count * 0.08) + (buy_score * 0.03))
                result.raw_confidence = base_confidence
                
                confidence_pct = self._calculate_final_confidence(
                    buy_score, base_confidence,
                    result.regime_multiplier, result.time_factor, result.volatility_factor
                )
                result.confidence = confidence_pct / 100
                
                result.quality_grade = self._get_quality_grade(confidence_pct)
                self._track_quality_grade(result.quality_grade)
                
                is_quality_valid, quality_reason = self._validate_signal_quality(result)
                if not is_quality_valid:
                    result.signal_type = None
                    result.reason = quality_reason
                    return result
                
                sl, tp = self._calculate_sl_tp(
                    10.0, 22.0,
                    self.M1_SL_MIN, self.M1_SL_MAX,
                    self.M1_TP_MIN, self.M1_TP_MAX,
                    volatility_mult
                )
                result.sl_pips = sl
                result.tp_pips = tp
                result.entry_price = close
                result.reason = f"M1 Scalp BUY [Grade:{result.quality_grade}]: WScore={buy_score:.1f}, {buy_count} confluences, Regime={result.regime_type}, Bias=BUY"
                
            elif (sell_score >= self.MIN_WEIGHTED_CONFLUENCE_SCORE and 
                  sell_count >= self.M1_MIN_CONFLUENCE and 
                  sell_count > buy_count and 
                  m5_confirmation_sell and 
                  not h1_at_support and
                  sell_secondary_valid):
                
                is_aligned, alignment_reason = self._check_regime_alignment('SELL', RuleType.M1_SCALP)
                if not is_aligned:
                    result.reason = alignment_reason
                    return result
                
                result.signal_type = SignalType.SELL.value
                result.confluence_count = sell_count
                result.confluence_details = sell_details
                result.weighted_confluence_score = sell_score
                result.secondary_confirmations = sell_secondary
                
                base_confidence = min(1.0, 0.55 + (sell_count * 0.08) + (sell_score * 0.03))
                result.raw_confidence = base_confidence
                
                confidence_pct = self._calculate_final_confidence(
                    sell_score, base_confidence,
                    result.regime_multiplier, result.time_factor, result.volatility_factor
                )
                result.confidence = confidence_pct / 100
                
                result.quality_grade = self._get_quality_grade(confidence_pct)
                self._track_quality_grade(result.quality_grade)
                
                is_quality_valid, quality_reason = self._validate_signal_quality(result)
                if not is_quality_valid:
                    result.signal_type = None
                    result.reason = quality_reason
                    return result
                
                sl, tp = self._calculate_sl_tp(
                    10.0, 22.0,
                    self.M1_SL_MIN, self.M1_SL_MAX,
                    self.M1_TP_MIN, self.M1_TP_MAX,
                    volatility_mult
                )
                result.sl_pips = sl
                result.tp_pips = tp
                result.entry_price = close
                result.reason = f"M1 Scalp SELL [Grade:{result.quality_grade}]: WScore={sell_score:.1f}, {sell_count} confluences, Regime={result.regime_type}, Bias=SELL"
            else:
                result.reason = f"M1 Scalp: Buy(score={buy_score:.1f},cnt={buy_count},sec={buy_sec_count}), Sell(score={sell_score:.1f},cnt={sell_count},sec={sell_sec_count}), min={self.MIN_WEIGHTED_CONFLUENCE_SCORE}"
            
            return result
            
        except Exception as e:
            logger.error(f"Error in M1 Scalp signal: {str(e)}")
            result.reason = f"Error: {str(e)}"
            return result
    
    def check_m5_swing_signal(self, df_m5: pd.DataFrame,
                               df_h1: Optional[pd.DataFrame] = None) -> SignalResult:
        """
        RULE 2: M5 Swing Entry (5-15 signals/hari)
        
        Enhanced with:
        - Weighted confluence scoring
        - Market regime awareness (blocked in range_bound)
        - Secondary confirmation requirements (3+ dari RSI, EMA, Volume, Trend, S/R)
        - Quality grading
        
        Trigger Conditions:
        - M5 candle close
        - Price break above EMA20 (BUY) or below EMA20 (SELL)
        - Volume spike (> average)
        - RSI direction match (rising for BUY, falling for SELL)
        - MACD histogram match (positive for BUY, negative for SELL)
        
        Confirmation:
        - H1 above MA50 aligned (BUY) or below MA50 (SELL)
        - Last 5 H1 candles consistency
        
        Execute IF: Weighted confluence >= 4.0 AND (ADX > 20 OR Breakout) AND 3+ secondary confirmations AND regime != range_bound
        
        Args:
            df_m5: DataFrame M5 timeframe
            df_h1: DataFrame H1 timeframe (optional)
        
        Returns:
            SignalResult with enhanced data
        """
        result = SignalResult(rule_name=RuleType.M5_SWING.value)
        
        if df_m5 is None or len(df_m5) < 50:
            result.reason = "Insufficient M5 data"
            return result
        
        try:
            indicators = self.indicator_engine.get_indicators(df_m5)
            if indicators is None:
                result.reason = "Failed to calculate M5 indicators"
                return result
            
            self._update_market_regime(None, df_m5, indicators)
            result.regime_type = self._current_regime.regime_type if self._current_regime else 'unknown'
            result.regime_multiplier = self._get_regime_multiplier(result.regime_type)
            
            is_favorable, regime_reason = self._is_regime_favorable_for_rule(RuleType.M5_SWING)
            if not is_favorable:
                result.reason = regime_reason
                return result
            
            close = safe_float(indicators.get('close', 0), 0.0)
            close_prev = self._safe_get_value(df_m5['close'], -2, 0.0) if 'close' in df_m5.columns else 0.0
            ema_20 = safe_float(indicators.get('ema_20', 0), 0.0)
            ema_50 = safe_float(indicators.get('ema_50', 0), 0.0)
            rsi = safe_float(indicators.get('rsi', 50), 50.0)
            rsi_prev = safe_float(indicators.get('rsi_prev', 50), 50.0)
            volume = safe_float(indicators.get('volume', 0), 0.0)
            volume_avg = safe_float(indicators.get('volume_avg', 1), 1.0)
            macd_histogram = safe_float(indicators.get('macd_histogram', 0), 0.0)
            adx = safe_float(indicators.get('adx', 0), 0.0)
            adx_prev = safe_float(indicators.get('adx_prev', 0), 0.0)
            atr = safe_float(indicators.get('atr', 1.0), 1.0)
            atr_avg = safe_float(indicators.get('atr_avg', atr), atr)
            
            ema_20_series = self.indicator_engine.calculate_ema(df_m5, 20)
            ema_20_prev = self._safe_get_value(ema_20_series, -2, 0.0)
            
            if close <= 0 or ema_20 <= 0:
                result.reason = "Invalid price data"
                return result
            
            atr_ratio = atr / atr_avg if atr_avg > 0 else 1.0
            result.volatility_factor = self._get_volatility_factor(atr_ratio)
            result.time_factor = self._get_time_of_day_factor()
            
            buy_confluences: List[WeightedConfluence] = []
            sell_confluences: List[WeightedConfluence] = []
            
            has_rsi_buy = False
            has_rsi_sell = False
            has_ema_buy = False
            has_ema_sell = False
            has_volume_buy = False
            has_volume_sell = False
            has_trend_buy = False
            has_trend_sell = False
            has_sr_buy = False
            has_sr_sell = False
            
            price_break_above = close > ema_20 and close_prev <= ema_20_prev
            price_break_below = close < ema_20 and close_prev >= ema_20_prev
            
            if price_break_above:
                buy_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.BREAKOUT_CONFIRMATION.value,
                    description="Price break above EMA20",
                    weight=self._get_confluence_weight(ConfluenceType.BREAKOUT_CONFIRMATION.value),
                    is_active=True
                ))
                has_ema_buy = True
            elif price_break_below:
                sell_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.BREAKOUT_CONFIRMATION.value,
                    description="Price break below EMA20",
                    weight=self._get_confluence_weight(ConfluenceType.BREAKOUT_CONFIRMATION.value),
                    is_active=True
                ))
                has_ema_sell = True
            elif close > ema_20:
                buy_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.EMA_ALIGNMENT.value,
                    description="Price above EMA20",
                    weight=self._get_confluence_weight(ConfluenceType.EMA_ALIGNMENT.value),
                    is_active=True
                ))
                has_ema_buy = True
            elif close < ema_20:
                sell_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.EMA_ALIGNMENT.value,
                    description="Price below EMA20",
                    weight=self._get_confluence_weight(ConfluenceType.EMA_ALIGNMENT.value),
                    is_active=True
                ))
                has_ema_sell = True
            
            volume_ratio = volume / volume_avg if volume_avg > 0 else 1.0
            if volume_ratio > 1.2:
                if close > ema_20:
                    buy_confluences.append(WeightedConfluence(
                        confluence_type=ConfluenceType.VOLUME_CONFIRMATION.value,
                        description=f"Volume spike ({volume_ratio:.2f}x)",
                        weight=self._get_confluence_weight(ConfluenceType.VOLUME_CONFIRMATION.value),
                        is_active=True
                    ))
                    has_volume_buy = True
                else:
                    sell_confluences.append(WeightedConfluence(
                        confluence_type=ConfluenceType.VOLUME_CONFIRMATION.value,
                        description=f"Volume spike ({volume_ratio:.2f}x)",
                        weight=self._get_confluence_weight(ConfluenceType.VOLUME_CONFIRMATION.value),
                        is_active=True
                    ))
                    has_volume_sell = True
            
            if rsi > rsi_prev and 45 < rsi < 70:
                buy_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.RSI_CONFIRMATION.value,
                    description=f"RSI rising ({rsi:.1f})",
                    weight=self._get_confluence_weight(ConfluenceType.RSI_CONFIRMATION.value),
                    is_active=True
                ))
                has_rsi_buy = True
            elif rsi < rsi_prev and 30 < rsi < 55:
                sell_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.RSI_CONFIRMATION.value,
                    description=f"RSI falling ({rsi:.1f})",
                    weight=self._get_confluence_weight(ConfluenceType.RSI_CONFIRMATION.value),
                    is_active=True
                ))
                has_rsi_sell = True
            
            if macd_histogram > 0:
                buy_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.MACD_CONFIRMATION.value,
                    description="MACD histogram positive",
                    weight=self._get_confluence_weight(ConfluenceType.MACD_CONFIRMATION.value),
                    is_active=True
                ))
            elif macd_histogram < 0:
                sell_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.MACD_CONFIRMATION.value,
                    description="MACD histogram negative",
                    weight=self._get_confluence_weight(ConfluenceType.MACD_CONFIRMATION.value),
                    is_active=True
                ))
            
            if ema_20 > ema_50:
                buy_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.TREND_ALIGNMENT.value,
                    description="EMA20 > EMA50 trend",
                    weight=self._get_confluence_weight(ConfluenceType.TREND_ALIGNMENT.value),
                    is_active=True
                ))
                has_trend_buy = True
            elif ema_20 < ema_50:
                sell_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.TREND_ALIGNMENT.value,
                    description="EMA20 < EMA50 trend",
                    weight=self._get_confluence_weight(ConfluenceType.TREND_ALIGNMENT.value),
                    is_active=True
                ))
                has_trend_sell = True
            
            h1_confirmation_buy = True
            h1_confirmation_sell = True
            h1_candle_consistency = 0
            
            if df_h1 is not None and len(df_h1) >= 30:
                h1_indicators = self.indicator_engine.get_indicators(df_h1)
                if h1_indicators:
                    h1_close = safe_float(h1_indicators.get('close', 0), 0.0)
                    h1_ema50 = safe_float(h1_indicators.get('ema_50', 0), 0.0)
                    
                    if h1_close > h1_ema50:
                        buy_confluences.append(WeightedConfluence(
                            confluence_type=ConfluenceType.TREND_ALIGNMENT.value,
                            description="H1 above MA50",
                            weight=self._get_confluence_weight(ConfluenceType.TREND_ALIGNMENT.value) * 0.7,
                            is_active=True
                        ))
                        has_sr_buy = True
                        h1_confirmation_sell = False
                    elif h1_close < h1_ema50:
                        sell_confluences.append(WeightedConfluence(
                            confluence_type=ConfluenceType.TREND_ALIGNMENT.value,
                            description="H1 below MA50",
                            weight=self._get_confluence_weight(ConfluenceType.TREND_ALIGNMENT.value) * 0.7,
                            is_active=True
                        ))
                        has_sr_sell = True
                        h1_confirmation_buy = False
                
                if 'close' in df_h1.columns and 'open' in df_h1.columns:
                    last_5_h1 = df_h1.tail(5)
                    bullish_count = sum(1 for _, c in last_5_h1.iterrows() 
                                       if safe_float(c.get('close', 0), 0) > safe_float(c.get('open', 0), 0))
                    bearish_count = 5 - bullish_count
                    
                    if bullish_count >= 4:
                        buy_confluences.append(WeightedConfluence(
                            confluence_type=ConfluenceType.TREND_ALIGNMENT.value,
                            description="H1 bullish consistency (4/5+)",
                            weight=self._get_confluence_weight(ConfluenceType.TREND_ALIGNMENT.value) * 0.6,
                            is_active=True
                        ))
                        h1_candle_consistency = 1
                        has_trend_buy = True
                    elif bearish_count >= 4:
                        sell_confluences.append(WeightedConfluence(
                            confluence_type=ConfluenceType.TREND_ALIGNMENT.value,
                            description="H1 bearish consistency (4/5+)",
                            weight=self._get_confluence_weight(ConfluenceType.TREND_ALIGNMENT.value) * 0.6,
                            is_active=True
                        ))
                        h1_candle_consistency = -1
                        has_trend_sell = True
            
            has_adx_condition = adx > self.M5_ADX_THRESHOLD
            is_breakout = price_break_above or price_break_below
            has_momentum = has_adx_condition or is_breakout
            
            if has_adx_condition:
                if adx > adx_prev:
                    if close > ema_20:
                        buy_confluences.append(WeightedConfluence(
                            confluence_type=ConfluenceType.ADX_CONFIRMATION.value,
                            description=f"ADX rising ({adx:.1f})",
                            weight=self._get_confluence_weight(ConfluenceType.ADX_CONFIRMATION.value),
                            is_active=True
                        ))
                    else:
                        sell_confluences.append(WeightedConfluence(
                            confluence_type=ConfluenceType.ADX_CONFIRMATION.value,
                            description=f"ADX rising ({adx:.1f})",
                            weight=self._get_confluence_weight(ConfluenceType.ADX_CONFIRMATION.value),
                            is_active=True
                        ))
            
            buy_score, buy_count, buy_details = self._calculate_weighted_confluence(buy_confluences)
            sell_score, sell_count, sell_details = self._calculate_weighted_confluence(sell_confluences)
            
            volatility_mult = min(1.3, max(0.8, 1.0))
            
            buy_secondary_valid, buy_secondary, buy_sec_count = self._check_m5_secondary_confirmations(
                has_rsi_buy, has_ema_buy, has_volume_buy, has_trend_buy, has_sr_buy
            )
            sell_secondary_valid, sell_secondary, sell_sec_count = self._check_m5_secondary_confirmations(
                has_rsi_sell, has_ema_sell, has_volume_sell, has_trend_sell, has_sr_sell
            )
            
            if (buy_score >= self.MIN_WEIGHTED_CONFLUENCE_SCORE and
                buy_count >= self.M5_MIN_CONFLUENCE and 
                buy_count > sell_count and 
                h1_confirmation_buy and 
                has_momentum and
                buy_secondary_valid):
                
                is_aligned, alignment_reason = self._check_regime_alignment('BUY', RuleType.M5_SWING)
                if not is_aligned:
                    result.reason = alignment_reason
                    return result
                
                result.signal_type = SignalType.BUY.value
                result.confluence_count = buy_count
                result.confluence_details = buy_details
                result.weighted_confluence_score = buy_score
                result.secondary_confirmations = buy_secondary
                
                base_confidence = min(1.0, 0.55 + (buy_count * 0.07) + (buy_score * 0.03))
                result.raw_confidence = base_confidence
                
                if is_breakout:
                    base_confidence += 0.08
                
                confidence_pct = self._calculate_final_confidence(
                    buy_score, base_confidence,
                    result.regime_multiplier, result.time_factor, result.volatility_factor
                )
                result.confidence = confidence_pct / 100
                
                result.quality_grade = self._get_quality_grade(confidence_pct)
                self._track_quality_grade(result.quality_grade)
                
                is_quality_valid, quality_reason = self._validate_signal_quality(result)
                if not is_quality_valid:
                    result.signal_type = None
                    result.reason = quality_reason
                    return result
                
                sl, tp = self._calculate_sl_tp(
                    17.0, 40.0,
                    self.M5_SL_MIN, self.M5_SL_MAX,
                    self.M5_TP_MIN, self.M5_TP_MAX,
                    volatility_mult
                )
                result.sl_pips = sl
                result.tp_pips = tp
                result.entry_price = close
                result.reason = f"M5 Swing BUY [Grade:{result.quality_grade}]: WScore={buy_score:.1f}, ADX={adx:.1f}, Regime={result.regime_type}, Bias=BUY"
                
            elif (sell_score >= self.MIN_WEIGHTED_CONFLUENCE_SCORE and
                  sell_count >= self.M5_MIN_CONFLUENCE and 
                  sell_count > buy_count and 
                  h1_confirmation_sell and 
                  has_momentum and
                  sell_secondary_valid):
                
                is_aligned, alignment_reason = self._check_regime_alignment('SELL', RuleType.M5_SWING)
                if not is_aligned:
                    result.reason = alignment_reason
                    return result
                
                result.signal_type = SignalType.SELL.value
                result.confluence_count = sell_count
                result.confluence_details = sell_details
                result.weighted_confluence_score = sell_score
                result.secondary_confirmations = sell_secondary
                
                base_confidence = min(1.0, 0.55 + (sell_count * 0.07) + (sell_score * 0.03))
                result.raw_confidence = base_confidence
                
                if is_breakout:
                    base_confidence += 0.08
                
                confidence_pct = self._calculate_final_confidence(
                    sell_score, base_confidence,
                    result.regime_multiplier, result.time_factor, result.volatility_factor
                )
                result.confidence = confidence_pct / 100
                
                result.quality_grade = self._get_quality_grade(confidence_pct)
                self._track_quality_grade(result.quality_grade)
                
                is_quality_valid, quality_reason = self._validate_signal_quality(result)
                if not is_quality_valid:
                    result.signal_type = None
                    result.reason = quality_reason
                    return result
                
                sl, tp = self._calculate_sl_tp(
                    17.0, 40.0,
                    self.M5_SL_MIN, self.M5_SL_MAX,
                    self.M5_TP_MIN, self.M5_TP_MAX,
                    volatility_mult
                )
                result.sl_pips = sl
                result.tp_pips = tp
                result.entry_price = close
                result.reason = f"M5 Swing SELL [Grade:{result.quality_grade}]: WScore={sell_score:.1f}, ADX={adx:.1f}, Regime={result.regime_type}, Bias=SELL"
            else:
                momentum_status = f"ADX={adx:.1f}" if has_adx_condition else "No momentum"
                result.reason = f"M5 Swing: Buy(score={buy_score:.1f},sec={buy_sec_count}), Sell(score={sell_score:.1f},sec={sell_sec_count}), {momentum_status}, Regime={result.regime_type}"
            
            return result
            
        except Exception as e:
            logger.error(f"Error in M5 Swing signal: {str(e)}")
            result.reason = f"Error: {str(e)}"
            return result
    
    def check_sr_reversion_signal(self, df_m5: pd.DataFrame,
                                   df_h1: Optional[pd.DataFrame] = None) -> SignalResult:
        """
        RULE 3: S/R Mean-Reversion (3-8 signals/hari, ranging market)
        
        Enhanced with:
        - Weighted confluence scoring
        - Market regime awareness (better in range_bound)
        - Secondary confirmation requirements (Wajib RSI extreme + Price di S/R zone)
        - Quality grading
        
        Trigger Conditions:
        - Price within 5 pips of S/R level
        - Wick test level (rejection candle)
        - RSI extreme (< 30 for BUY, > 70 for SELL)
        - Volume decrease (consolidation)
        
        Confirmation:
        - M5 close within 1 pip of S/R
        - Bollinger mid nearby
        - Stochastic extreme (< 20 for BUY, > 80 for SELL)
        
        Execute IF: ADX < 20 AND ATR < 150% average AND weighted confluence >= 4.0 AND RSI extreme + at S/R
        
        Args:
            df_m5: DataFrame M5 timeframe
            df_h1: DataFrame H1 timeframe (optional for S/R levels)
        
        Returns:
            SignalResult with enhanced data
        """
        result = SignalResult(rule_name=RuleType.SR_REVERSION.value)
        
        if df_m5 is None or len(df_m5) < 50:
            result.reason = "Insufficient M5 data"
            return result
        
        try:
            indicators = self.indicator_engine.get_indicators(df_m5)
            if indicators is None:
                result.reason = "Failed to calculate M5 indicators"
                return result
            
            self._update_market_regime(None, df_m5, indicators)
            result.regime_type = self._current_regime.regime_type if self._current_regime else 'unknown'
            result.regime_multiplier = self._get_regime_multiplier(result.regime_type)
            
            if result.regime_type == RegimeType.RANGE_BOUND.value:
                result.regime_multiplier = 1.1
            
            close = safe_float(indicators.get('close', 0), 0.0)
            rsi = safe_float(indicators.get('rsi', 50), 50.0)
            stoch_k = safe_float(indicators.get('stoch_k', 50), 50.0)
            stoch_d = safe_float(indicators.get('stoch_d', 50), 50.0)
            volume = safe_float(indicators.get('volume', 0), 0.0)
            volume_avg = safe_float(indicators.get('volume_avg', 1), 1.0)
            adx = safe_float(indicators.get('adx', 25), 25.0)
            atr = safe_float(indicators.get('atr', 1.0), 1.0)
            atr_avg = safe_float(indicators.get('atr_avg', atr), atr)
            
            if close <= 0:
                result.reason = "Invalid price data"
                return result
            
            atr_ratio = atr / atr_avg if atr_avg > 0 else 1.0
            result.volatility_factor = self._get_volatility_factor(atr_ratio)
            result.time_factor = self._get_time_of_day_factor()
            
            sr_source = df_h1 if df_h1 is not None and len(df_h1) >= 30 else df_m5
            sr_data = self.indicator_engine.calculate_micro_support_resistance(sr_source, lookback=50)
            
            support = sr_data.get('nearest_support', 0)
            resistance = sr_data.get('nearest_resistance', 0)
            support_levels = sr_data.get('support_levels', [])
            resistance_levels = sr_data.get('resistance_levels', [])
            
            dist_to_support_pips = abs(close - support) / self.PIP_VALUE_XAUUSD if support > 0 else 999
            dist_to_resistance_pips = abs(resistance - close) / self.PIP_VALUE_XAUUSD if resistance > 0 else 999
            
            upper_bb, middle_bb, lower_bb = self._calculate_bollinger_bands(df_m5, period=20, std_dev=2.0)
            bb_mid = self._safe_get_value(middle_bb, -1, close)
            dist_to_bb_mid_pips = abs(close - bb_mid) / self.PIP_VALUE_XAUUSD
            
            atr_series = self.indicator_engine.calculate_atr(df_m5, 14)
            atr_avg_calc = self._safe_get_value(atr_series.rolling(window=20, min_periods=1).mean(), -1, atr)
            atr_ratio_calc = atr / atr_avg_calc if atr_avg_calc > 0 else 1.0
            
            wick_data = self._get_candle_wick_data(df_m5)
            
            if adx >= self.SR_ADX_MAX:
                result.reason = f"ADX too high ({adx:.1f} >= {self.SR_ADX_MAX}) - Trending market, not suitable for SR Reversion"
                return result
            
            if atr_ratio_calc >= self.SR_ATR_MAX_RATIO:
                result.reason = f"ATR ratio too high ({atr_ratio_calc:.2f} >= {self.SR_ATR_MAX_RATIO}) - High volatility"
                return result
            
            buy_confluences: List[WeightedConfluence] = []
            sell_confluences: List[WeightedConfluence] = []
            
            near_support = dist_to_support_pips <= self.SR_PROXIMITY_PIPS
            near_resistance = dist_to_resistance_pips <= self.SR_PROXIMITY_PIPS
            
            has_rsi_extreme_buy = rsi < 30
            has_rsi_extreme_sell = rsi > 70
            has_price_at_sr_buy = near_support
            has_price_at_sr_sell = near_resistance
            
            if near_support:
                buy_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.SR_PROXIMITY.value,
                    description=f"Price near support ({dist_to_support_pips:.1f} pips)",
                    weight=self._get_confluence_weight(ConfluenceType.SR_PROXIMITY.value),
                    is_active=True
                ))
            if near_resistance:
                sell_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.SR_PROXIMITY.value,
                    description=f"Price near resistance ({dist_to_resistance_pips:.1f} pips)",
                    weight=self._get_confluence_weight(ConfluenceType.SR_PROXIMITY.value),
                    is_active=True
                ))
            
            if wick_data['lower_wick'] > wick_data['body_size'] and wick_data['is_bullish']:
                buy_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.WICK_REJECTION.value,
                    description="Bullish rejection wick",
                    weight=self._get_confluence_weight(ConfluenceType.WICK_REJECTION.value),
                    is_active=True
                ))
            elif wick_data['upper_wick'] > wick_data['body_size'] and not wick_data['is_bullish']:
                sell_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.WICK_REJECTION.value,
                    description="Bearish rejection wick",
                    weight=self._get_confluence_weight(ConfluenceType.WICK_REJECTION.value),
                    is_active=True
                ))
            
            if has_rsi_extreme_buy:
                buy_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.RSI_CONFIRMATION.value,
                    description=f"RSI oversold ({rsi:.1f})",
                    weight=self._get_confluence_weight(ConfluenceType.RSI_CONFIRMATION.value) * 1.2,
                    is_active=True
                ))
            elif has_rsi_extreme_sell:
                sell_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.RSI_CONFIRMATION.value,
                    description=f"RSI overbought ({rsi:.1f})",
                    weight=self._get_confluence_weight(ConfluenceType.RSI_CONFIRMATION.value) * 1.2,
                    is_active=True
                ))
            
            volume_ratio = volume / volume_avg if volume_avg > 0 else 1.0
            if volume_ratio < 0.9:
                if near_support:
                    buy_confluences.append(WeightedConfluence(
                        confluence_type=ConfluenceType.VOLUME_CONFIRMATION.value,
                        description=f"Volume decreasing ({volume_ratio:.2f}x)",
                        weight=self._get_confluence_weight(ConfluenceType.VOLUME_CONFIRMATION.value),
                        is_active=True
                    ))
                elif near_resistance:
                    sell_confluences.append(WeightedConfluence(
                        confluence_type=ConfluenceType.VOLUME_CONFIRMATION.value,
                        description=f"Volume decreasing ({volume_ratio:.2f}x)",
                        weight=self._get_confluence_weight(ConfluenceType.VOLUME_CONFIRMATION.value),
                        is_active=True
                    ))
            
            if dist_to_support_pips <= 1.0:
                buy_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.SR_PROXIMITY.value,
                    description="M5 close at support level",
                    weight=self._get_confluence_weight(ConfluenceType.SR_PROXIMITY.value) * 0.5,
                    is_active=True
                ))
            elif dist_to_resistance_pips <= 1.0:
                sell_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.SR_PROXIMITY.value,
                    description="M5 close at resistance level",
                    weight=self._get_confluence_weight(ConfluenceType.SR_PROXIMITY.value) * 0.5,
                    is_active=True
                ))
            
            if dist_to_bb_mid_pips <= 15:
                if near_support:
                    buy_confluences.append(WeightedConfluence(
                        confluence_type=ConfluenceType.EMA_ALIGNMENT.value,
                        description="Bollinger mid nearby",
                        weight=self._get_confluence_weight(ConfluenceType.EMA_ALIGNMENT.value) * 0.5,
                        is_active=True
                    ))
                elif near_resistance:
                    sell_confluences.append(WeightedConfluence(
                        confluence_type=ConfluenceType.EMA_ALIGNMENT.value,
                        description="Bollinger mid nearby",
                        weight=self._get_confluence_weight(ConfluenceType.EMA_ALIGNMENT.value) * 0.5,
                        is_active=True
                    ))
            
            if stoch_k < 20 and stoch_d < 20:
                buy_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.STOCHASTIC_CONFIRMATION.value,
                    description=f"Stochastic oversold ({stoch_k:.1f})",
                    weight=self._get_confluence_weight(ConfluenceType.STOCHASTIC_CONFIRMATION.value),
                    is_active=True
                ))
            elif stoch_k > 80 and stoch_d > 80:
                sell_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.STOCHASTIC_CONFIRMATION.value,
                    description=f"Stochastic overbought ({stoch_k:.1f})",
                    weight=self._get_confluence_weight(ConfluenceType.STOCHASTIC_CONFIRMATION.value),
                    is_active=True
                ))
            
            buy_confluences.append(WeightedConfluence(
                confluence_type=ConfluenceType.ADX_CONFIRMATION.value,
                description=f"ADX low ({adx:.1f} - ranging)",
                weight=self._get_confluence_weight(ConfluenceType.ADX_CONFIRMATION.value) * 0.5,
                is_active=True
            ))
            sell_confluences.append(WeightedConfluence(
                confluence_type=ConfluenceType.ADX_CONFIRMATION.value,
                description=f"ADX low ({adx:.1f} - ranging)",
                weight=self._get_confluence_weight(ConfluenceType.ADX_CONFIRMATION.value) * 0.5,
                is_active=True
            ))
            
            buy_score, buy_count, buy_details = self._calculate_weighted_confluence(buy_confluences)
            sell_score, sell_count, sell_details = self._calculate_weighted_confluence(sell_confluences)
            
            buy_secondary_valid, buy_secondary, _ = self._check_sr_secondary_confirmations(
                has_rsi_extreme_buy, has_price_at_sr_buy
            )
            sell_secondary_valid, sell_secondary, _ = self._check_sr_secondary_confirmations(
                has_rsi_extreme_sell, has_price_at_sr_sell
            )
            
            if (buy_score >= self.MIN_WEIGHTED_CONFLUENCE_SCORE and
                buy_count >= self.SR_MIN_CONFLUENCE + 1 and 
                buy_count > sell_count and 
                near_support and
                rsi < 40 and
                buy_secondary_valid):
                
                result.signal_type = SignalType.BUY.value
                result.confluence_count = buy_count
                result.confluence_details = buy_details
                result.weighted_confluence_score = buy_score
                result.secondary_confirmations = buy_secondary
                
                base_confidence = min(1.0, 0.5 + (buy_count * 0.08) + (buy_score * 0.02))
                result.raw_confidence = base_confidence
                
                if has_rsi_extreme_buy:
                    base_confidence += 0.1
                if stoch_k < 20:
                    base_confidence += 0.05
                
                confidence_pct = self._calculate_final_confidence(
                    buy_score, base_confidence,
                    result.regime_multiplier, result.time_factor, result.volatility_factor
                )
                result.confidence = confidence_pct / 100
                
                result.quality_grade = self._get_quality_grade(confidence_pct)
                self._track_quality_grade(result.quality_grade)
                
                volatility_mult = min(1.2, max(0.8, atr_ratio_calc))
                sl, tp = self._calculate_sl_tp(
                    12.0, 30.0,
                    self.SR_SL_MIN, self.SR_SL_MAX,
                    self.SR_TP_MIN, self.SR_TP_MAX,
                    volatility_mult
                )
                result.sl_pips = sl
                result.tp_pips = tp
                result.entry_price = close
                result.reason = f"S/R Reversion BUY [Grade:{result.quality_grade}]: WScore={buy_score:.1f}, at support, Regime={result.regime_type}"
                
            elif (sell_score >= self.MIN_WEIGHTED_CONFLUENCE_SCORE and
                  sell_count >= self.SR_MIN_CONFLUENCE + 1 and 
                  sell_count > buy_count and 
                  near_resistance and
                  rsi > 60 and
                  sell_secondary_valid):
                
                result.signal_type = SignalType.SELL.value
                result.confluence_count = sell_count
                result.confluence_details = sell_details
                result.weighted_confluence_score = sell_score
                result.secondary_confirmations = sell_secondary
                
                base_confidence = min(1.0, 0.5 + (sell_count * 0.08) + (sell_score * 0.02))
                result.raw_confidence = base_confidence
                
                if has_rsi_extreme_sell:
                    base_confidence += 0.1
                if stoch_k > 80:
                    base_confidence += 0.05
                
                confidence_pct = self._calculate_final_confidence(
                    sell_score, base_confidence,
                    result.regime_multiplier, result.time_factor, result.volatility_factor
                )
                result.confidence = confidence_pct / 100
                
                result.quality_grade = self._get_quality_grade(confidence_pct)
                self._track_quality_grade(result.quality_grade)
                
                volatility_mult = min(1.2, max(0.8, atr_ratio_calc))
                sl, tp = self._calculate_sl_tp(
                    12.0, 30.0,
                    self.SR_SL_MIN, self.SR_SL_MAX,
                    self.SR_TP_MIN, self.SR_TP_MAX,
                    volatility_mult
                )
                result.sl_pips = sl
                result.tp_pips = tp
                result.entry_price = close
                result.reason = f"S/R Reversion SELL [Grade:{result.quality_grade}]: WScore={sell_score:.1f}, at resistance, Regime={result.regime_type}"
            else:
                result.reason = f"S/R Reversion: Buy(score={buy_score:.1f}), Sell(score={sell_score:.1f}), Support={dist_to_support_pips:.1f}p, Resistance={dist_to_resistance_pips:.1f}p"
            
            return result
            
        except Exception as e:
            logger.error(f"Error in S/R Reversion signal: {str(e)}")
            result.reason = f"Error: {str(e)}"
            return result
    
    def check_breakout_signal(self, df_m5: pd.DataFrame,
                               df_h1: Optional[pd.DataFrame] = None) -> SignalResult:
        """
        RULE 4: Breakout Confirmation (2-5 signals/hari)
        
        Enhanced with:
        - Weighted confluence scoring
        - Market regime awareness (blocked in range_bound)
        - Secondary confirmation requirements (Wajib Volume spike + ADX rising)
        - Quality grading
        
        Trigger Conditions:
        - Price break 5/10 candle high (BUY) or low (SELL)
        - Close outside breakout level
        - Volume > 150% average
        - RSI momentum match (above 50 for BUY, below 50 for SELL)
        
        Confirmation:
        - M5 confirm breakout (close outside level)
        - H1 no immediate S/R ahead (room to run)
        
        Execute IF: ADX increasing toward 25+ AND weighted confluence >= 4.0 AND Volume spike + ADX rising AND regime != range_bound
        
        Args:
            df_m5: DataFrame M5 timeframe
            df_h1: DataFrame H1 timeframe (optional)
        
        Returns:
            SignalResult with enhanced data
        """
        result = SignalResult(rule_name=RuleType.BREAKOUT.value)
        
        if df_m5 is None or len(df_m5) < 50:
            result.reason = "Insufficient M5 data"
            return result
        
        try:
            indicators = self.indicator_engine.get_indicators(df_m5)
            if indicators is None:
                result.reason = "Failed to calculate M5 indicators"
                return result
            
            self._update_market_regime(None, df_m5, indicators)
            result.regime_type = self._current_regime.regime_type if self._current_regime else 'unknown'
            result.regime_multiplier = self._get_regime_multiplier(result.regime_type)
            
            is_favorable, regime_reason = self._is_regime_favorable_for_rule(RuleType.BREAKOUT)
            if not is_favorable:
                result.reason = regime_reason
                return result
            
            close = safe_float(indicators.get('close', 0), 0.0)
            rsi = safe_float(indicators.get('rsi', 50), 50.0)
            volume = safe_float(indicators.get('volume', 0), 0.0)
            volume_avg = safe_float(indicators.get('volume_avg', 1), 1.0)
            adx = safe_float(indicators.get('adx', 0), 0.0)
            adx_prev = safe_float(indicators.get('adx_prev', 0), 0.0)
            macd_histogram = safe_float(indicators.get('macd_histogram', 0), 0.0)
            atr = safe_float(indicators.get('atr', 1.0), 1.0)
            atr_avg = safe_float(indicators.get('atr_avg', atr), atr)
            
            if close <= 0:
                result.reason = "Invalid price data"
                return result
            
            atr_ratio = atr / atr_avg if atr_avg > 0 else 1.0
            result.volatility_factor = self._get_volatility_factor(atr_ratio)
            result.time_factor = self._get_time_of_day_factor()
            
            high_series = df_m5['high'] if 'high' in df_m5.columns else None
            low_series = df_m5['low'] if 'low' in df_m5.columns else None
            
            if high_series is None or low_series is None:
                result.reason = "Missing high/low data"
                return result
            
            lookback_5 = min(5, len(df_m5) - 1)
            lookback_10 = min(10, len(df_m5) - 1)
            
            prev_high_5 = high_series.iloc[-lookback_5-1:-1].max() if lookback_5 > 0 else close
            prev_low_5 = low_series.iloc[-lookback_5-1:-1].min() if lookback_5 > 0 else close
            prev_high_10 = high_series.iloc[-lookback_10-1:-1].max() if lookback_10 > 0 else close
            prev_low_10 = low_series.iloc[-lookback_10-1:-1].min() if lookback_10 > 0 else close
            
            breakout_up_5 = close > prev_high_5
            breakout_down_5 = close < prev_low_5
            breakout_up_10 = close > prev_high_10
            breakout_down_10 = close < prev_low_10
            
            h1_sr_ahead = False
            h1_sr_distance = 999
            if df_h1 is not None and len(df_h1) >= 30:
                sr_data = self.indicator_engine.calculate_micro_support_resistance(df_h1, lookback=50)
                h1_resistance = sr_data.get('nearest_resistance', 0)
                h1_support = sr_data.get('nearest_support', 0)
                
                if close > 0:
                    dist_to_h1_resistance = abs(h1_resistance - close) / self.PIP_VALUE_XAUUSD if h1_resistance > 0 else 999
                    dist_to_h1_support = abs(close - h1_support) / self.PIP_VALUE_XAUUSD if h1_support > 0 else 999
                    
                    if breakout_up_5 or breakout_up_10:
                        h1_sr_distance = dist_to_h1_resistance
                        h1_sr_ahead = dist_to_h1_resistance < self.BO_SR_PROXIMITY_PIPS
                    elif breakout_down_5 or breakout_down_10:
                        h1_sr_distance = dist_to_h1_support
                        h1_sr_ahead = dist_to_h1_support < self.BO_SR_PROXIMITY_PIPS
            
            buy_confluences: List[WeightedConfluence] = []
            sell_confluences: List[WeightedConfluence] = []
            
            volume_ratio = volume / volume_avg if volume_avg > 0 else 1.0
            has_volume_spike = volume_ratio >= self.BO_VOLUME_THRESHOLD
            adx_increasing = adx > adx_prev
            has_adx_rising = adx_increasing and adx >= 20
            
            if breakout_up_10:
                buy_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.BREAKOUT_CONFIRMATION.value,
                    description="Break 10-candle high",
                    weight=self._get_confluence_weight(ConfluenceType.BREAKOUT_CONFIRMATION.value) * 1.2,
                    is_active=True
                ))
            elif breakout_up_5:
                buy_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.BREAKOUT_CONFIRMATION.value,
                    description="Break 5-candle high",
                    weight=self._get_confluence_weight(ConfluenceType.BREAKOUT_CONFIRMATION.value),
                    is_active=True
                ))
            
            if breakout_down_10:
                sell_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.BREAKOUT_CONFIRMATION.value,
                    description="Break 10-candle low",
                    weight=self._get_confluence_weight(ConfluenceType.BREAKOUT_CONFIRMATION.value) * 1.2,
                    is_active=True
                ))
            elif breakout_down_5:
                sell_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.BREAKOUT_CONFIRMATION.value,
                    description="Break 5-candle low",
                    weight=self._get_confluence_weight(ConfluenceType.BREAKOUT_CONFIRMATION.value),
                    is_active=True
                ))
            
            if (breakout_up_5 or breakout_up_10) and close > prev_high_5:
                buy_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.BREAKOUT_CONFIRMATION.value,
                    description="Close confirmed above breakout level",
                    weight=self._get_confluence_weight(ConfluenceType.BREAKOUT_CONFIRMATION.value) * 0.5,
                    is_active=True
                ))
            if (breakout_down_5 or breakout_down_10) and close < prev_low_5:
                sell_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.BREAKOUT_CONFIRMATION.value,
                    description="Close confirmed below breakout level",
                    weight=self._get_confluence_weight(ConfluenceType.BREAKOUT_CONFIRMATION.value) * 0.5,
                    is_active=True
                ))
            
            if has_volume_spike:
                if breakout_up_5 or breakout_up_10:
                    buy_confluences.append(WeightedConfluence(
                        confluence_type=ConfluenceType.VOLUME_CONFIRMATION.value,
                        description=f"Volume breakout ({volume_ratio:.2f}x)",
                        weight=self._get_confluence_weight(ConfluenceType.VOLUME_CONFIRMATION.value) * 1.3,
                        is_active=True
                    ))
                elif breakout_down_5 or breakout_down_10:
                    sell_confluences.append(WeightedConfluence(
                        confluence_type=ConfluenceType.VOLUME_CONFIRMATION.value,
                        description=f"Volume breakout ({volume_ratio:.2f}x)",
                        weight=self._get_confluence_weight(ConfluenceType.VOLUME_CONFIRMATION.value) * 1.3,
                        is_active=True
                    ))
            
            if rsi > 50 and (breakout_up_5 or breakout_up_10):
                buy_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.RSI_CONFIRMATION.value,
                    description=f"RSI bullish momentum ({rsi:.1f})",
                    weight=self._get_confluence_weight(ConfluenceType.RSI_CONFIRMATION.value),
                    is_active=True
                ))
            elif rsi < 50 and (breakout_down_5 or breakout_down_10):
                sell_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.RSI_CONFIRMATION.value,
                    description=f"RSI bearish momentum ({rsi:.1f})",
                    weight=self._get_confluence_weight(ConfluenceType.RSI_CONFIRMATION.value),
                    is_active=True
                ))
            
            if macd_histogram > 0:
                buy_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.MACD_CONFIRMATION.value,
                    description="MACD histogram positive",
                    weight=self._get_confluence_weight(ConfluenceType.MACD_CONFIRMATION.value),
                    is_active=True
                ))
            elif macd_histogram < 0:
                sell_confluences.append(WeightedConfluence(
                    confluence_type=ConfluenceType.MACD_CONFIRMATION.value,
                    description="MACD histogram negative",
                    weight=self._get_confluence_weight(ConfluenceType.MACD_CONFIRMATION.value),
                    is_active=True
                ))
            
            if has_adx_rising:
                if breakout_up_5 or breakout_up_10:
                    buy_confluences.append(WeightedConfluence(
                        confluence_type=ConfluenceType.ADX_CONFIRMATION.value,
                        description=f"ADX increasing ({adx:.1f})",
                        weight=self._get_confluence_weight(ConfluenceType.ADX_CONFIRMATION.value) * 1.2,
                        is_active=True
                    ))
                elif breakout_down_5 or breakout_down_10:
                    sell_confluences.append(WeightedConfluence(
                        confluence_type=ConfluenceType.ADX_CONFIRMATION.value,
                        description=f"ADX increasing ({adx:.1f})",
                        weight=self._get_confluence_weight(ConfluenceType.ADX_CONFIRMATION.value) * 1.2,
                        is_active=True
                    ))
            
            if not h1_sr_ahead:
                if breakout_up_5 or breakout_up_10:
                    buy_confluences.append(WeightedConfluence(
                        confluence_type=ConfluenceType.SR_PROXIMITY.value,
                        description=f"Room to run ({h1_sr_distance:.1f} pips)",
                        weight=self._get_confluence_weight(ConfluenceType.SR_PROXIMITY.value) * 0.7,
                        is_active=True
                    ))
                elif breakout_down_5 or breakout_down_10:
                    sell_confluences.append(WeightedConfluence(
                        confluence_type=ConfluenceType.SR_PROXIMITY.value,
                        description=f"Room to run ({h1_sr_distance:.1f} pips)",
                        weight=self._get_confluence_weight(ConfluenceType.SR_PROXIMITY.value) * 0.7,
                        is_active=True
                    ))
            
            buy_score, buy_count, buy_details = self._calculate_weighted_confluence(buy_confluences)
            sell_score, sell_count, sell_details = self._calculate_weighted_confluence(sell_confluences)
            
            has_breakout = breakout_up_5 or breakout_up_10 or breakout_down_5 or breakout_down_10
            
            buy_secondary_valid, buy_secondary, _ = self._check_breakout_secondary_confirmations(
                has_volume_spike and (breakout_up_5 or breakout_up_10), 
                has_adx_rising and (breakout_up_5 or breakout_up_10)
            )
            sell_secondary_valid, sell_secondary, _ = self._check_breakout_secondary_confirmations(
                has_volume_spike and (breakout_down_5 or breakout_down_10), 
                has_adx_rising and (breakout_down_5 or breakout_down_10)
            )
            
            if (buy_score >= self.MIN_WEIGHTED_CONFLUENCE_SCORE and
                buy_count >= self.BO_MIN_CONFLUENCE and 
                buy_count > sell_count and 
                (breakout_up_5 or breakout_up_10) and
                adx_increasing and
                not h1_sr_ahead and
                buy_secondary_valid):
                
                result.signal_type = SignalType.BUY.value
                result.confluence_count = buy_count
                result.confluence_details = buy_details
                result.weighted_confluence_score = buy_score
                result.secondary_confirmations = buy_secondary
                
                base_confidence = min(1.0, 0.5 + (buy_count * 0.08) + (buy_score * 0.02))
                result.raw_confidence = base_confidence
                
                if breakout_up_10:
                    base_confidence += 0.1
                if volume_ratio >= 2.0:
                    base_confidence += 0.1
                if adx >= self.BO_ADX_TARGET:
                    base_confidence += 0.1
                
                confidence_pct = self._calculate_final_confidence(
                    buy_score, base_confidence,
                    result.regime_multiplier, result.time_factor, result.volatility_factor
                )
                result.confidence = confidence_pct / 100
                
                result.quality_grade = self._get_quality_grade(confidence_pct)
                self._track_quality_grade(result.quality_grade)
                
                volatility_mult = min(1.5, max(0.8, 1.0))
                sl, tp = self._calculate_sl_tp(
                    15.0, 60.0,
                    self.BO_SL_MIN, self.BO_SL_MAX,
                    self.BO_TP_MIN, self.BO_TP_MAX,
                    volatility_mult
                )
                result.sl_pips = sl
                result.tp_pips = tp
                result.entry_price = close
                result.reason = f"Breakout BUY [Grade:{result.quality_grade}]: WScore={buy_score:.1f}, ADX={adx:.1f}, Vol={volume_ratio:.1f}x, Regime={result.regime_type}"
                
            elif (sell_score >= self.MIN_WEIGHTED_CONFLUENCE_SCORE and
                  sell_count >= self.BO_MIN_CONFLUENCE and 
                  sell_count > buy_count and 
                  (breakout_down_5 or breakout_down_10) and
                  adx_increasing and
                  not h1_sr_ahead and
                  sell_secondary_valid):
                
                result.signal_type = SignalType.SELL.value
                result.confluence_count = sell_count
                result.confluence_details = sell_details
                result.weighted_confluence_score = sell_score
                result.secondary_confirmations = sell_secondary
                
                base_confidence = min(1.0, 0.5 + (sell_count * 0.08) + (sell_score * 0.02))
                result.raw_confidence = base_confidence
                
                if breakout_down_10:
                    base_confidence += 0.1
                if volume_ratio >= 2.0:
                    base_confidence += 0.1
                if adx >= self.BO_ADX_TARGET:
                    base_confidence += 0.1
                
                confidence_pct = self._calculate_final_confidence(
                    sell_score, base_confidence,
                    result.regime_multiplier, result.time_factor, result.volatility_factor
                )
                result.confidence = confidence_pct / 100
                
                result.quality_grade = self._get_quality_grade(confidence_pct)
                self._track_quality_grade(result.quality_grade)
                
                volatility_mult = min(1.5, max(0.8, 1.0))
                sl, tp = self._calculate_sl_tp(
                    15.0, 60.0,
                    self.BO_SL_MIN, self.BO_SL_MAX,
                    self.BO_TP_MIN, self.BO_TP_MAX,
                    volatility_mult
                )
                result.sl_pips = sl
                result.tp_pips = tp
                result.entry_price = close
                result.reason = f"Breakout SELL [Grade:{result.quality_grade}]: WScore={sell_score:.1f}, ADX={adx:.1f}, Vol={volume_ratio:.1f}x, Regime={result.regime_type}"
            else:
                bo_status = "Breakout detected" if has_breakout else "No breakout"
                vol_status = f"Vol={volume_ratio:.2f}x" if has_volume_spike else "Low volume"
                adx_status = f"ADX={adx:.1f}" if adx_increasing else f"ADX flat ({adx:.1f})"
                result.reason = f"Breakout: Buy(score={buy_score:.1f}), Sell(score={sell_score:.1f}), {bo_status}, {vol_status}, {adx_status}"
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Breakout signal: {str(e)}")
            result.reason = f"Error: {str(e)}"
            return result
    
    def check_all_signals(self, df_m1: Optional[pd.DataFrame] = None,
                          df_m5: Optional[pd.DataFrame] = None,
                          df_h1: Optional[pd.DataFrame] = None) -> List[SignalResult]:
        """
        Check all 4 signal rules and return list of valid signals.
        
        Enhanced with market regime tracking and quality grading.
        
        Args:
            df_m1: DataFrame M1 timeframe (optional)
            df_m5: DataFrame M5 timeframe (optional)
            df_h1: DataFrame H1 timeframe (optional)
        
        Returns:
            List of SignalResult objects for valid signals only
        """
        results = []
        
        try:
            if df_m1 is not None and len(df_m1) >= 30:
                m1_signal = self.check_m1_scalp_signal(df_m1, df_m5, df_h1)
                if m1_signal.is_valid():
                    results.append(m1_signal)
                    logger.info(f"M1 Scalp signal generated: {m1_signal.signal_type}, Grade={m1_signal.quality_grade}, Confidence={m1_signal.confidence:.2f}")
            
            if df_m5 is not None and len(df_m5) >= 50:
                m5_signal = self.check_m5_swing_signal(df_m5, df_h1)
                if m5_signal.is_valid():
                    results.append(m5_signal)
                    logger.info(f"M5 Swing signal generated: {m5_signal.signal_type}, Grade={m5_signal.quality_grade}, Confidence={m5_signal.confidence:.2f}")
                
                sr_signal = self.check_sr_reversion_signal(df_m5, df_h1)
                if sr_signal.is_valid():
                    results.append(sr_signal)
                    logger.info(f"S/R Reversion signal generated: {sr_signal.signal_type}, Grade={sr_signal.quality_grade}, Confidence={sr_signal.confidence:.2f}")
                
                bo_signal = self.check_breakout_signal(df_m5, df_h1)
                if bo_signal.is_valid():
                    results.append(bo_signal)
                    logger.info(f"Breakout signal generated: {bo_signal.signal_type}, Grade={bo_signal.quality_grade}, Confidence={bo_signal.confidence:.2f}")
            
            if results:
                results.sort(key=lambda x: (
                    0 if x.quality_grade == 'A' else (1 if x.quality_grade == 'B' else (2 if x.quality_grade == 'C' else 3)),
                    -x.weighted_confluence_score,
                    -x.confidence
                ))
                
                logger.info(f"Generated {len(results)} valid signals. Quality stats: {self._quality_stats}")
            
        except Exception as e:
            logger.error(f"Error in check_all_signals: {str(e)}")
        
        return results
    
    def get_best_signal(self, df_m1: Optional[pd.DataFrame] = None,
                        df_m5: Optional[pd.DataFrame] = None,
                        df_h1: Optional[pd.DataFrame] = None) -> Optional[SignalResult]:
        """
        Get the best quality signal from all rules.
        
        Priority:
        1. Quality Grade (A > B > C > D)
        2. Weighted Confluence Score (higher is better)
        3. Confidence (higher is better)
        
        Args:
            df_m1: DataFrame M1 timeframe (optional)
            df_m5: DataFrame M5 timeframe (optional)
            df_h1: DataFrame H1 timeframe (optional)
        
        Returns:
            Best SignalResult or None if no valid signals
        """
        signals = self.check_all_signals(df_m1, df_m5, df_h1)
        if signals:
            return signals[0]
        return None
    
    def get_current_regime(self) -> Optional[MarketRegime]:
        """Return the current market regime."""
        return self._current_regime
    
    def reset_quality_statistics(self):
        """Reset quality grade statistics."""
        self._quality_stats = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        logger.info("Quality statistics reset")
