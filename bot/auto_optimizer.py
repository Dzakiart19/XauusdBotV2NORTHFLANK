"""
Auto-Optimization Engine untuk Bot Trading XAUUSD.

Modul ini menyediakan optimasi otomatis parameter trading berdasarkan:
- Dynamic Adjustment Rules berdasarkan signal performance
- Auto-optimization logic dengan gradual adjustments
- Safety guards untuk mencegah over-optimization
- Integration dengan SignalQualityTracker dan TradingStrategy
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any, Tuple, Callable
from enum import Enum
from copy import deepcopy
import threading
import json
import pytz

from bot.logger import setup_logger
from bot.signal_quality_tracker import SignalQualityTracker, RuleType, MarketRegimeType

logger = setup_logger('AutoOptimizer')


class AutoOptimizerError(Exception):
    """Base exception for auto optimizer errors"""
    pass


class OptimizationStatus(str, Enum):
    """Enum untuk status optimization"""
    SUCCESS = 'SUCCESS'
    PARTIAL = 'PARTIAL'
    SKIPPED = 'SKIPPED'
    FAILED = 'FAILED'
    ROLLED_BACK = 'ROLLED_BACK'


class AdjustmentType(str, Enum):
    """Enum untuk tipe adjustment"""
    TIGHTEN_CONFLUENCE = 'tighten_confluence'
    LOOSEN_CONFLUENCE = 'loosen_confluence'
    DISABLE_M5_LOW_ADX = 'disable_m5_low_adx'
    ENABLE_M5_LOW_ADX = 'enable_m5_low_adx'
    INCREASE_3CONF_WEIGHT = 'increase_3conf_weight'
    REDUCE_HOUR_FREQUENCY = 'reduce_hour_frequency'
    ADJUST_VOLUME_THRESHOLD = 'adjust_volume_threshold'
    ADJUST_EMA_STRICTNESS = 'adjust_ema_strictness'
    ADJUST_SIGNAL_COOLDOWN = 'adjust_signal_cooldown'
    RESET_TO_DEFAULT = 'reset_to_default'
    AGGRESSIVE_BOOST = 'aggressive_boost'
    AGGRESSIVE_REDUCE = 'aggressive_reduce'
    FAST_TUNE_CONFLUENCE = 'fast_tune_confluence'
    EMERGENCY_TIGHTEN = 'emergency_tighten'


class AggressiveTuningMode(str, Enum):
    """Enum untuk mode aggressive tuning"""
    CONSERVATIVE = 'conservative'
    MODERATE = 'moderate'
    AGGRESSIVE = 'aggressive'
    ULTRA_AGGRESSIVE = 'ultra_aggressive'


@dataclass
class OptimizationParameters:
    """
    Dataclass untuk parameter yang dapat dioptimasi.
    
    Attributes:
        min_confluence_required: Minimum confluence level (2-4)
        volume_threshold_multiplier: Multiplier untuk volume threshold (0.8-1.5)
        ema_strictness: Strictness level untuk EMA alignment (0.5-2.0)
        signal_cooldown: Cooldown antar signal dalam detik (0-60)
        enable_m5_low_adx: Enable M5 signals saat ADX rendah
        three_confluence_weight: Weight multiplier untuk 3+ confluence (1.0-2.0)
        reduced_signal_hours: List jam dengan reduced frequency
        adx_threshold_for_m5: ADX threshold untuk enable M5 (default 15)
    """
    min_confluence_required: int = 2
    volume_threshold_multiplier: float = 1.1
    ema_strictness: float = 1.0
    signal_cooldown: int = 0
    enable_m5_low_adx: bool = True
    three_confluence_weight: float = 1.0
    reduced_signal_hours: List[int] = field(default_factory=list)
    adx_threshold_for_m5: float = 15.0
    
    MIN_CONFLUENCE = 2
    MAX_CONFLUENCE = 4
    MIN_VOLUME_MULTIPLIER = 0.8
    MAX_VOLUME_MULTIPLIER = 1.5
    MIN_EMA_STRICTNESS = 0.5
    MAX_EMA_STRICTNESS = 2.0
    MIN_SIGNAL_COOLDOWN = 0
    MAX_SIGNAL_COOLDOWN = 60
    MIN_3CONF_WEIGHT = 1.0
    MAX_3CONF_WEIGHT = 2.0
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate parameter ranges and return (is_valid, errors)"""
        errors = []
        
        if not self.MIN_CONFLUENCE <= self.min_confluence_required <= self.MAX_CONFLUENCE:
            errors.append(f"min_confluence_required {self.min_confluence_required} out of range [{self.MIN_CONFLUENCE}, {self.MAX_CONFLUENCE}]")
        
        if not self.MIN_VOLUME_MULTIPLIER <= self.volume_threshold_multiplier <= self.MAX_VOLUME_MULTIPLIER:
            errors.append(f"volume_threshold_multiplier {self.volume_threshold_multiplier} out of range [{self.MIN_VOLUME_MULTIPLIER}, {self.MAX_VOLUME_MULTIPLIER}]")
        
        if not self.MIN_EMA_STRICTNESS <= self.ema_strictness <= self.MAX_EMA_STRICTNESS:
            errors.append(f"ema_strictness {self.ema_strictness} out of range [{self.MIN_EMA_STRICTNESS}, {self.MAX_EMA_STRICTNESS}]")
        
        if not self.MIN_SIGNAL_COOLDOWN <= self.signal_cooldown <= self.MAX_SIGNAL_COOLDOWN:
            errors.append(f"signal_cooldown {self.signal_cooldown} out of range [{self.MIN_SIGNAL_COOLDOWN}, {self.MAX_SIGNAL_COOLDOWN}]")
        
        if not self.MIN_3CONF_WEIGHT <= self.three_confluence_weight <= self.MAX_3CONF_WEIGHT:
            errors.append(f"three_confluence_weight {self.three_confluence_weight} out of range [{self.MIN_3CONF_WEIGHT}, {self.MAX_3CONF_WEIGHT}]")
        
        for hour in self.reduced_signal_hours:
            if not 0 <= hour <= 23:
                errors.append(f"Invalid hour in reduced_signal_hours: {hour}")
        
        return len(errors) == 0, errors
    
    def clamp_values(self) -> 'OptimizationParameters':
        """Clamp all values to valid ranges"""
        self.min_confluence_required = max(self.MIN_CONFLUENCE, min(self.MAX_CONFLUENCE, self.min_confluence_required))
        self.volume_threshold_multiplier = max(self.MIN_VOLUME_MULTIPLIER, min(self.MAX_VOLUME_MULTIPLIER, self.volume_threshold_multiplier))
        self.ema_strictness = max(self.MIN_EMA_STRICTNESS, min(self.MAX_EMA_STRICTNESS, self.ema_strictness))
        self.signal_cooldown = max(self.MIN_SIGNAL_COOLDOWN, min(self.MAX_SIGNAL_COOLDOWN, self.signal_cooldown))
        self.three_confluence_weight = max(self.MIN_3CONF_WEIGHT, min(self.MAX_3CONF_WEIGHT, self.three_confluence_weight))
        self.reduced_signal_hours = [h for h in self.reduced_signal_hours if 0 <= h <= 23]
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationParameters':
        """Create from dictionary"""
        valid_keys = ['min_confluence_required', 'volume_threshold_multiplier', 'ema_strictness',
                      'signal_cooldown', 'enable_m5_low_adx', 'three_confluence_weight',
                      'reduced_signal_hours', 'adx_threshold_for_m5']
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)
    
    @staticmethod
    def get_default() -> 'OptimizationParameters':
        """Get default parameters"""
        return OptimizationParameters()


@dataclass
class Adjustment:
    """Dataclass untuk single adjustment record"""
    adjustment_type: str
    parameter_name: str
    old_value: Any
    new_value: Any
    reason: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(pytz.UTC))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'adjustment_type': self.adjustment_type,
            'parameter_name': self.parameter_name,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'reason': self.reason,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class OptimizationResult:
    """
    Dataclass untuk hasil optimization run.
    
    Attributes:
        status: Status optimization (SUCCESS, PARTIAL, SKIPPED, FAILED, ROLLED_BACK)
        adjustments: List adjustment yang dilakukan
        parameters_before: Parameter sebelum optimization
        parameters_after: Parameter setelah optimization
        analysis_summary: Ringkasan analisis yang dilakukan
        timestamp: Waktu optimization
        signals_analyzed: Jumlah signal yang dianalisis
        recommendations: Rekomendasi tambahan
    """
    status: str
    adjustments: List[Adjustment] = field(default_factory=list)
    parameters_before: Optional[Dict[str, Any]] = None
    parameters_after: Optional[Dict[str, Any]] = None
    analysis_summary: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(pytz.UTC))
    signals_analyzed: int = 0
    recommendations: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': self.status,
            'adjustments': [adj.to_dict() for adj in self.adjustments],
            'parameters_before': self.parameters_before,
            'parameters_after': self.parameters_after,
            'analysis_summary': self.analysis_summary,
            'timestamp': self.timestamp.isoformat(),
            'signals_analyzed': self.signals_analyzed,
            'recommendations': self.recommendations,
            'error_message': self.error_message
        }


@dataclass
class PerformanceSnapshot:
    """Snapshot performa untuk tracking rollback"""
    timestamp: datetime
    overall_accuracy: float
    parameters: OptimizationParameters
    signals_count: int
    accuracy_by_type: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'overall_accuracy': self.overall_accuracy,
            'parameters': self.parameters.to_dict(),
            'signals_count': self.signals_count,
            'accuracy_by_type': self.accuracy_by_type
        }


class AutoOptimizer:
    """
    Auto-Optimization Engine untuk bot trading XAUUSD.
    
    Fitur utama:
    1. Dynamic Adjustment Rules berdasarkan signal performance
    2. Auto-optimization logic dengan gradual adjustments
    3. Safety guards untuk mencegah over-optimization
    4. Integration dengan SignalQualityTracker
    
    Thread-safe dengan locking untuk operasi konkuren.
    """
    
    M1_SCALP_ACCURACY_THRESHOLD = 0.45
    M5_SWING_RANGING_THRESHOLD = 0.50
    THREE_CONF_WIN_RATE_THRESHOLD = 0.70
    HOUR_ACCURACY_THRESHOLD = 0.50
    ADX_LOW_THRESHOLD = 15.0
    
    MINIMUM_SIGNALS_FOR_ADJUSTMENT = 20
    DEFAULT_ANALYSIS_SIGNALS = 50
    DEFAULT_OPTIMIZATION_INTERVAL_HOURS = 6
    
    MAX_ADJUSTMENT_STEP_CONFLUENCE = 1
    MAX_ADJUSTMENT_STEP_MULTIPLIER = 0.1
    MAX_ADJUSTMENT_STEP_STRICTNESS = 0.2
    MAX_ADJUSTMENT_STEP_COOLDOWN = 15
    MAX_ADJUSTMENT_STEP_WEIGHT = 0.2
    
    ROLLBACK_PERFORMANCE_THRESHOLD = 0.05
    
    MAX_OPTIMIZATION_HISTORY = 100
    
    AGGRESSIVE_THRESHOLDS = {
        AggressiveTuningMode.CONSERVATIVE: {
            'accuracy_drop_trigger': 0.10,
            'adjustment_multiplier': 1.0,
            'min_signals': 30,
            'cooldown_hours': 8
        },
        AggressiveTuningMode.MODERATE: {
            'accuracy_drop_trigger': 0.08,
            'adjustment_multiplier': 1.5,
            'min_signals': 20,
            'cooldown_hours': 4
        },
        AggressiveTuningMode.AGGRESSIVE: {
            'accuracy_drop_trigger': 0.05,
            'adjustment_multiplier': 2.0,
            'min_signals': 15,
            'cooldown_hours': 2
        },
        AggressiveTuningMode.ULTRA_AGGRESSIVE: {
            'accuracy_drop_trigger': 0.03,
            'adjustment_multiplier': 3.0,
            'min_signals': 10,
            'cooldown_hours': 1
        }
    }
    
    EMERGENCY_ACCURACY_THRESHOLD = 0.30
    FAST_TUNE_THRESHOLD = 0.40
    AGGRESSIVE_BOOST_THRESHOLD = 0.65
    
    BLOCKING_RATE_THRESHOLD = 0.70  # 70%
    BLOCKING_RATE_WINDOW_HOURS = 1
    
    def __init__(self, 
                 signal_quality_tracker: Optional[SignalQualityTracker] = None,
                 config = None,
                 strategy_update_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
                 optimization_interval_hours: int = DEFAULT_OPTIMIZATION_INTERVAL_HOURS):
        """
        Inisialisasi AutoOptimizer.
        
        Args:
            signal_quality_tracker: SignalQualityTracker instance untuk data quality
            config: Konfigurasi bot (opsional)
            strategy_update_callback: Callback untuk update TradingStrategy parameters
            optimization_interval_hours: Interval antar optimization runs
        """
        self.signal_quality_tracker = signal_quality_tracker
        self.config = config
        self.strategy_update_callback = strategy_update_callback
        self.optimization_interval_hours = optimization_interval_hours
        
        self._lock = threading.RLock()
        self._current_parameters = OptimizationParameters()
        self._default_parameters = OptimizationParameters()
        self._optimization_history: List[OptimizationResult] = []
        self._performance_snapshots: List[PerformanceSnapshot] = []
        self._last_optimization_time: Optional[datetime] = None
        self._rollback_parameters: Optional[OptimizationParameters] = None
        
        self._is_enabled = True
        self._consecutive_failures = 0
        self._max_consecutive_failures = 3
        
        self._tuning_mode: AggressiveTuningMode = AggressiveTuningMode.MODERATE
        self._aggressive_tuning_enabled = True
        self._last_aggressive_tune_time: Optional[datetime] = None
        self._fast_tune_count = 0
        self._emergency_mode = False
        
        logger.info(f"AutoOptimizer initialized with {optimization_interval_hours}h interval and aggressive tuning support")
    
    def run_optimization(self, signal_quality_data: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """
        Jalankan optimization cycle berdasarkan signal quality data.
        
        Args:
            signal_quality_data: Data kualitas signal (opsional, akan diambil dari tracker jika None)
        
        Returns:
            OptimizationResult dengan detail adjustments
        """
        with self._lock:
            result = OptimizationResult(
                status=OptimizationStatus.SKIPPED.value,
                parameters_before=self._current_parameters.to_dict()
            )
            
            try:
                if not self._is_enabled:
                    result.error_message = "AutoOptimizer is disabled"
                    logger.info("Optimization skipped: AutoOptimizer is disabled")
                    return result
                
                if signal_quality_data is None:
                    if self.signal_quality_tracker is None:
                        result.error_message = "No signal quality data available and no tracker configured"
                        result.status = OptimizationStatus.FAILED.value
                        logger.warning(result.error_message)
                        return result
                    signal_quality_data = self.signal_quality_tracker.get_overall_stats(days=7)
                
                if signal_quality_data is None:
                    result.error_message = "Failed to retrieve signal quality data from tracker"
                    result.status = OptimizationStatus.FAILED.value
                    logger.error(result.error_message)
                    return result
                
                if isinstance(signal_quality_data, dict) and 'error' in signal_quality_data:
                    result.error_message = f"Error getting signal quality data: {signal_quality_data['error']}"
                    result.status = OptimizationStatus.FAILED.value
                    logger.error(result.error_message)
                    return result
                
                total_signals = signal_quality_data.get('total_signals', 0) if isinstance(signal_quality_data, dict) else 0
                result.signals_analyzed = total_signals
                
                if total_signals < self.MINIMUM_SIGNALS_FOR_ADJUSTMENT:
                    result.error_message = f"Insufficient signals for optimization: {total_signals} < {self.MINIMUM_SIGNALS_FOR_ADJUSTMENT}"
                    result.recommendations.append(f"Wait for at least {self.MINIMUM_SIGNALS_FOR_ADJUSTMENT} completed signals before optimization")
                    logger.info(result.error_message)
                    return result
                
                self._save_performance_snapshot(signal_quality_data)
                
                adjustments = self._analyze_and_generate_adjustments(signal_quality_data)
                
                if not adjustments:
                    result.status = OptimizationStatus.SKIPPED.value
                    result.analysis_summary = self._generate_analysis_summary(signal_quality_data)
                    result.recommendations.append("No adjustments needed - current parameters performing well")
                    logger.info("Optimization skipped: No adjustments needed")
                    return result
                
                success = self._apply_adjustments_internal(adjustments)
                
                if success:
                    result.status = OptimizationStatus.SUCCESS.value
                    result.adjustments = adjustments
                    self._consecutive_failures = 0
                else:
                    result.status = OptimizationStatus.PARTIAL.value
                    self._consecutive_failures += 1
                
                result.parameters_after = self._current_parameters.to_dict()
                result.analysis_summary = self._generate_analysis_summary(signal_quality_data)
                
                self._last_optimization_time = datetime.now(pytz.UTC)
                self._add_to_history(result)
                
                logger.info(f"Optimization completed: {result.status} with {len(adjustments)} adjustments")
                for adj in adjustments:
                    logger.info(f"  📊 {adj.adjustment_type}: {adj.parameter_name} {adj.old_value} -> {adj.new_value} ({adj.reason})")
                
                return result
                
            except Exception as e:
                result.status = OptimizationStatus.FAILED.value
                result.error_message = f"Optimization failed with exception: {str(e)}"
                self._consecutive_failures += 1
                logger.error(f"Optimization exception: {e}")
                
                if self._consecutive_failures >= self._max_consecutive_failures:
                    logger.warning(f"Disabling AutoOptimizer after {self._consecutive_failures} consecutive failures")
                    self._is_enabled = False
                
                return result
    
    def _analyze_and_generate_adjustments(self, data: Dict[str, Any]) -> List[Adjustment]:
        """Analisis data dan generate list adjustments yang diperlukan"""
        adjustments = []
        
        adjustments.extend(self._check_m1_scalp_accuracy(data))
        adjustments.extend(self._check_m5_swing_ranging(data))
        adjustments.extend(self._check_three_confluence_win_rate(data))
        adjustments.extend(self._check_hourly_accuracy(data))
        adjustments.extend(self._check_performance_improvement(data))
        
        # Check blocking rate
        adjustments.extend(self._check_signal_blocking_rate(data))
        
        return adjustments
    
    def _check_m1_scalp_accuracy(self, data: Dict[str, Any]) -> List[Adjustment]:
        """
        Dynamic Rule 1: If M1 scalp accuracy < 45% dalam 50 signals -> tighten confluence
        """
        adjustments = []
        accuracy_by_type = data.get('accuracy_by_type', {})
        m1_data = accuracy_by_type.get('M1_SCALP', {})
        
        if not m1_data:
            return adjustments
        
        m1_accuracy = m1_data.get('accuracy', 0.0)
        m1_total = m1_data.get('total', 0)
        
        if m1_total >= 20 and m1_accuracy < self.M1_SCALP_ACCURACY_THRESHOLD:
            current_confluence = self._current_parameters.min_confluence_required
            if current_confluence < OptimizationParameters.MAX_CONFLUENCE:
                new_confluence = min(current_confluence + self.MAX_ADJUSTMENT_STEP_CONFLUENCE, 
                                    OptimizationParameters.MAX_CONFLUENCE)
                
                adjustments.append(Adjustment(
                    adjustment_type=AdjustmentType.TIGHTEN_CONFLUENCE.value,
                    parameter_name='min_confluence_required',
                    old_value=current_confluence,
                    new_value=new_confluence,
                    reason=f"M1_SCALP accuracy {m1_accuracy*100:.1f}% < {self.M1_SCALP_ACCURACY_THRESHOLD*100}% threshold (n={m1_total})"
                ))
                
                logger.info(f"🔧 M1_SCALP accuracy low ({m1_accuracy*100:.1f}%), tightening confluence: {current_confluence} -> {new_confluence}")
        
        elif m1_total >= 30 and m1_accuracy >= 0.55:
            current_confluence = self._current_parameters.min_confluence_required
            if current_confluence > OptimizationParameters.MIN_CONFLUENCE:
                new_confluence = max(current_confluence - self.MAX_ADJUSTMENT_STEP_CONFLUENCE,
                                    OptimizationParameters.MIN_CONFLUENCE)
                
                adjustments.append(Adjustment(
                    adjustment_type=AdjustmentType.LOOSEN_CONFLUENCE.value,
                    parameter_name='min_confluence_required',
                    old_value=current_confluence,
                    new_value=new_confluence,
                    reason=f"M1_SCALP accuracy {m1_accuracy*100:.1f}% >= 55%, loosening confluence (n={m1_total})"
                ))
                
                logger.info(f"🔧 M1_SCALP accuracy high ({m1_accuracy*100:.1f}%), loosening confluence: {current_confluence} -> {new_confluence}")
        
        return adjustments
    
    def _check_m5_swing_ranging(self, data: Dict[str, Any]) -> List[Adjustment]:
        """
        Dynamic Rule 2: If M5 swing accuracy di ranging market < 50% -> disable M5 saat ADX < 15
        """
        adjustments = []
        accuracy_by_regime = data.get('accuracy_by_regime', {})
        ranging_data = accuracy_by_regime.get('ranging', {})
        accuracy_by_type = data.get('accuracy_by_type', {})
        m5_data = accuracy_by_type.get('M5_SWING', {})
        
        m5_total = m5_data.get('total', 0)
        m5_accuracy = m5_data.get('accuracy', 0.0)
        ranging_accuracy = ranging_data.get('accuracy', 0.0)
        ranging_total = ranging_data.get('total', 0)
        
        if m5_total >= 15 and ranging_total >= 10:
            combined_poor = (m5_accuracy < self.M5_SWING_RANGING_THRESHOLD and 
                           ranging_accuracy < self.M5_SWING_RANGING_THRESHOLD)
            
            if combined_poor and self._current_parameters.enable_m5_low_adx:
                adjustments.append(Adjustment(
                    adjustment_type=AdjustmentType.DISABLE_M5_LOW_ADX.value,
                    parameter_name='enable_m5_low_adx',
                    old_value=True,
                    new_value=False,
                    reason=f"M5_SWING accuracy {m5_accuracy*100:.1f}% and ranging {ranging_accuracy*100:.1f}% < {self.M5_SWING_RANGING_THRESHOLD*100}%"
                ))
                
                logger.info(f"🔧 M5_SWING poor in ranging market, disabling M5 for ADX < {self.ADX_LOW_THRESHOLD}")
            
            elif m5_accuracy >= 0.55 and ranging_accuracy >= 0.50 and not self._current_parameters.enable_m5_low_adx:
                adjustments.append(Adjustment(
                    adjustment_type=AdjustmentType.ENABLE_M5_LOW_ADX.value,
                    parameter_name='enable_m5_low_adx',
                    old_value=False,
                    new_value=True,
                    reason=f"M5_SWING accuracy {m5_accuracy*100:.1f}% improved, re-enabling M5 for low ADX"
                ))
                
                logger.info(f"🔧 M5_SWING accuracy improved, re-enabling M5 for low ADX")
        
        return adjustments
    
    def _check_three_confluence_win_rate(self, data: Dict[str, Any]) -> List[Adjustment]:
        """
        Dynamic Rule 3: If 3-confluence win rate > 70% -> increase weighting
        """
        adjustments = []
        accuracy_by_confluence = data.get('accuracy_by_confluence', {})
        three_conf_data = accuracy_by_confluence.get('3+_confluence', {})
        
        if not three_conf_data:
            return adjustments
        
        three_conf_accuracy = three_conf_data.get('accuracy', 0.0)
        three_conf_total = three_conf_data.get('total', 0)
        
        if three_conf_total >= 15:
            current_weight = self._current_parameters.three_confluence_weight
            
            if three_conf_accuracy > self.THREE_CONF_WIN_RATE_THRESHOLD:
                if current_weight < OptimizationParameters.MAX_3CONF_WEIGHT:
                    new_weight = min(current_weight + self.MAX_ADJUSTMENT_STEP_WEIGHT,
                                    OptimizationParameters.MAX_3CONF_WEIGHT)
                    
                    adjustments.append(Adjustment(
                        adjustment_type=AdjustmentType.INCREASE_3CONF_WEIGHT.value,
                        parameter_name='three_confluence_weight',
                        old_value=current_weight,
                        new_value=round(new_weight, 2),
                        reason=f"3+ confluence win rate {three_conf_accuracy*100:.1f}% > {self.THREE_CONF_WIN_RATE_THRESHOLD*100}% (n={three_conf_total})"
                    ))
                    
                    logger.info(f"🔧 3+ confluence performing well ({three_conf_accuracy*100:.1f}%), increasing weight: {current_weight} -> {new_weight}")
            
            elif three_conf_accuracy < 0.50 and current_weight > OptimizationParameters.MIN_3CONF_WEIGHT:
                new_weight = max(current_weight - self.MAX_ADJUSTMENT_STEP_WEIGHT,
                                OptimizationParameters.MIN_3CONF_WEIGHT)
                
                adjustments.append(Adjustment(
                    adjustment_type=AdjustmentType.INCREASE_3CONF_WEIGHT.value,
                    parameter_name='three_confluence_weight',
                    old_value=current_weight,
                    new_value=round(new_weight, 2),
                    reason=f"3+ confluence underperforming {three_conf_accuracy*100:.1f}%, reducing weight (n={three_conf_total})"
                ))
                
                logger.info(f"🔧 3+ confluence underperforming ({three_conf_accuracy*100:.1f}%), reducing weight: {current_weight} -> {new_weight}")
        
        return adjustments
    
    def _check_hourly_accuracy(self, data: Dict[str, Any]) -> List[Adjustment]:
        """
        Dynamic Rule 4: If certain hour accuracy < 50% consistently -> reduce signal frequency
        """
        adjustments = []
        accuracy_by_hour = data.get('accuracy_by_hour', {})
        
        if not accuracy_by_hour:
            return adjustments
        
        bad_hours = []
        for hour_str, hour_data in accuracy_by_hour.items():
            try:
                hour = int(hour_str) if isinstance(hour_str, str) else hour_str
                accuracy = hour_data.get('accuracy', 0.0)
                total = hour_data.get('total', 0)
                
                if total >= 10 and accuracy < self.HOUR_ACCURACY_THRESHOLD:
                    bad_hours.append(hour)
            except (ValueError, TypeError):
                continue
        
        current_reduced_hours = set(self._current_parameters.reduced_signal_hours)
        new_reduced_hours = set(bad_hours)
        
        hours_to_add = new_reduced_hours - current_reduced_hours
        hours_to_remove = current_reduced_hours - new_reduced_hours
        
        if hours_to_add:
            updated_hours = sorted(list(current_reduced_hours | hours_to_add))
            adjustments.append(Adjustment(
                adjustment_type=AdjustmentType.REDUCE_HOUR_FREQUENCY.value,
                parameter_name='reduced_signal_hours',
                old_value=list(current_reduced_hours),
                new_value=updated_hours,
                reason=f"Hours {list(hours_to_add)} accuracy < {self.HOUR_ACCURACY_THRESHOLD*100}%"
            ))
            
            logger.info(f"🔧 Adding reduced frequency for hours: {list(hours_to_add)}")
        
        if hours_to_remove and len(self._performance_snapshots) >= 2:
            updated_hours = sorted(list(current_reduced_hours - hours_to_remove))
            adjustments.append(Adjustment(
                adjustment_type=AdjustmentType.REDUCE_HOUR_FREQUENCY.value,
                parameter_name='reduced_signal_hours',
                old_value=list(current_reduced_hours),
                new_value=updated_hours,
                reason=f"Hours {list(hours_to_remove)} accuracy improved, removing from reduced list"
            ))
            
            logger.info(f"🔧 Removing reduced frequency for improved hours: {list(hours_to_remove)}")
        
        return adjustments
    
    def _check_performance_improvement(self, data: Dict[str, Any]) -> List[Adjustment]:
        """
        Check if performance improved significantly to reset parameters
        """
        adjustments = []
        
        if len(self._performance_snapshots) < 2:
            return adjustments
        
        current_accuracy = data.get('overall_accuracy', 0.0)
        last_snapshot = self._performance_snapshots[-1]
        
        if current_accuracy > 0.60 and current_accuracy > last_snapshot.overall_accuracy + 0.10:
            if self._current_parameters != self._default_parameters:
                logger.info(f"🔧 Performance improved significantly ({current_accuracy*100:.1f}%), considering reset to defaults")
        
        return adjustments
    
    def _check_signal_blocking_rate(self, data: Dict[str, Any]) -> List[Adjustment]:
        """
        Check signal blocking rate dan generate adjustments jika terlalu tinggi.
        
        Jika blocking rate > 70% dalam 1 jam:
        - Recommend enabling AUTO_SIGNAL_REPLACEMENT_ALLOWED
        - Recommend reducing cooldowns
        
        Args:
            data: Signal quality data termasuk blocking stats
        
        Returns:
            List of Adjustments untuk mengurangi blocking
        """
        adjustments = []
        
        blocking_stats = data.get('blocking_stats', {})
        total_blocked = blocking_stats.get('total_blocked', 0)
        window_hours = blocking_stats.get('window_hours', 1)
        
        estimated_total_signals = 50 * window_hours
        
        if total_blocked > 0 and estimated_total_signals > 0:
            blocking_rate = total_blocked / (total_blocked + estimated_total_signals)
            
            if blocking_rate >= self.BLOCKING_RATE_THRESHOLD:
                logger.warning(
                    f"⚠️ High signal blocking rate detected: {blocking_rate*100:.1f}% "
                    f"({total_blocked} blocked in {window_hours}h)"
                )
                
                current_cooldown = self._current_parameters.signal_cooldown
                if current_cooldown > 0:
                    adjustments.append(Adjustment(
                        adjustment_type='reduce_blocking_cooldown',
                        parameter_name='signal_cooldown',
                        old_value=current_cooldown,
                        new_value=0,
                        reason=f"Blocking rate {blocking_rate*100:.1f}% > {self.BLOCKING_RATE_THRESHOLD*100}% threshold"
                    ))
                
                if data.get('auto_replacement_disabled', False):
                    adjustments.append(Adjustment(
                        adjustment_type='enable_auto_replacement',
                        parameter_name='auto_signal_replacement',
                        old_value=False,
                        new_value=True,
                        reason=f"Enable auto replacement due to high blocking rate {blocking_rate*100:.1f}%"
                    ))
                
                if adjustments:
                    logger.info(
                        f"🔧 Auto-optimizer recommends: Enable AUTO_SIGNAL_REPLACEMENT_ALLOWED "
                        f"and set TICK_COOLDOWN_FOR_SAME_SIGNAL=0 to reduce blocking rate"
                    )
        
        return adjustments
    
    def set_tuning_mode(self, mode: AggressiveTuningMode) -> None:
        """Set the aggressive tuning mode."""
        with self._lock:
            old_mode = self._tuning_mode
            self._tuning_mode = mode
            logger.info(f"🔧 Tuning mode changed: {old_mode.value} -> {mode.value}")
    
    def get_tuning_mode(self) -> AggressiveTuningMode:
        """Get current tuning mode."""
        with self._lock:
            return self._tuning_mode
    
    def run_aggressive_optimization(self, signal_quality_data: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """
        Run aggressive optimization cycle with faster and larger adjustments.
        
        This method applies more aggressive thresholds and adjustments based on
        the current tuning mode. Suitable for rapidly changing market conditions.
        
        Args:
            signal_quality_data: Signal quality data (optional, will be fetched from tracker if None)
        
        Returns:
            OptimizationResult with aggressive adjustments
        """
        with self._lock:
            result = OptimizationResult(
                status=OptimizationStatus.SKIPPED.value,
                parameters_before=self._current_parameters.to_dict()
            )
            
            try:
                if not self._aggressive_tuning_enabled:
                    result.error_message = "Aggressive tuning is disabled"
                    return result
                
                thresholds = self.AGGRESSIVE_THRESHOLDS.get(self._tuning_mode, 
                                                           self.AGGRESSIVE_THRESHOLDS[AggressiveTuningMode.MODERATE])
                
                if self._last_aggressive_tune_time:
                    hours_since = (datetime.now(pytz.UTC) - self._last_aggressive_tune_time).total_seconds() / 3600
                    if hours_since < thresholds['cooldown_hours']:
                        result.error_message = f"Aggressive tune cooldown: {thresholds['cooldown_hours'] - hours_since:.1f}h remaining"
                        return result
                
                if signal_quality_data is None:
                    if self.signal_quality_tracker is not None:
                        signal_quality_data = self.signal_quality_tracker.get_overall_stats(days=3)
                    
                    if signal_quality_data is None or not isinstance(signal_quality_data, dict):
                        signal_quality_data = {
                            'total_signals': 0,
                            'overall_accuracy': 0.5,
                            'accuracy_by_type': {},
                            'accuracy_by_regime': {}
                        }
                        logger.warning("No signal quality data available, using defaults for aggressive optimization")
                
                if not isinstance(signal_quality_data, dict):
                    signal_quality_data = {
                        'total_signals': 0,
                        'overall_accuracy': 0.5,
                        'accuracy_by_type': {},
                        'accuracy_by_regime': {}
                    }
                
                total_signals = signal_quality_data.get('total_signals', 0)
                overall_accuracy = signal_quality_data.get('overall_accuracy', 0.5)
                
                force_emergency = False
                if total_signals < thresholds['min_signals']:
                    if overall_accuracy < self.EMERGENCY_ACCURACY_THRESHOLD:
                        force_emergency = True
                        logger.warning(f"⚠️ Emergency mode: Accuracy {overall_accuracy*100:.1f}% critical, forcing adjustments with {total_signals} signals")
                    elif self._tuning_mode in [AggressiveTuningMode.AGGRESSIVE, AggressiveTuningMode.ULTRA_AGGRESSIVE]:
                        force_emergency = overall_accuracy < self.FAST_TUNE_THRESHOLD
                        if force_emergency:
                            logger.warning(f"⚠️ Aggressive override: Proceeding with {total_signals} signals (accuracy: {overall_accuracy*100:.1f}%)")
                    
                    if not force_emergency:
                        result.error_message = f"Insufficient signals: {total_signals} < {thresholds['min_signals']}"
                        return result
                
                if force_emergency:
                    signal_quality_data['force_emergency'] = True
                
                adjustments = self._generate_aggressive_adjustments(signal_quality_data, thresholds)
                
                if not adjustments:
                    result.status = OptimizationStatus.SKIPPED.value
                    result.recommendations.append("No aggressive adjustments needed")
                    return result
                
                success = self._apply_adjustments_internal(adjustments)
                
                if success:
                    result.status = OptimizationStatus.SUCCESS.value
                    result.adjustments = adjustments
                    self._last_aggressive_tune_time = datetime.now(pytz.UTC)
                    self._fast_tune_count += 1
                    logger.info(f"🚀 Aggressive optimization completed with {len(adjustments)} adjustments (mode: {self._tuning_mode.value})")
                else:
                    result.status = OptimizationStatus.PARTIAL.value
                
                result.parameters_after = self._current_parameters.to_dict()
                result.signals_analyzed = total_signals
                self._add_to_history(result)
                
                return result
                
            except Exception as e:
                result.status = OptimizationStatus.FAILED.value
                result.error_message = f"Aggressive optimization failed: {str(e)}"
                logger.error(f"Aggressive optimization exception: {e}")
                return result
    
    def _generate_aggressive_adjustments(self, data: Dict[str, Any], thresholds: Dict[str, Any]) -> List[Adjustment]:
        """Generate aggressive adjustments based on current mode thresholds."""
        adjustments = []
        multiplier = thresholds.get('adjustment_multiplier', 1.0)
        
        overall_accuracy = data.get('overall_accuracy', 0.5)
        force_emergency = data.get('force_emergency', False)
        
        if force_emergency or overall_accuracy < self.EMERGENCY_ACCURACY_THRESHOLD:
            self._emergency_mode = True
            adjustments.extend(self._apply_emergency_tightening(multiplier))
            logger.warning(f"🚨 EMERGENCY MODE: Accuracy {overall_accuracy*100:.1f}% (force={force_emergency})")
        
        elif overall_accuracy < self.FAST_TUNE_THRESHOLD:
            adjustments.extend(self._apply_fast_tune_confluence(multiplier))
            logger.info(f"⚡ Fast tuning: Accuracy {overall_accuracy*100:.1f}% below {self.FAST_TUNE_THRESHOLD*100}%")
        
        elif overall_accuracy > self.AGGRESSIVE_BOOST_THRESHOLD:
            adjustments.extend(self._apply_aggressive_boost(multiplier))
            logger.info(f"🚀 Aggressive boost: Accuracy {overall_accuracy*100:.1f}% above {self.AGGRESSIVE_BOOST_THRESHOLD*100}%")
        
        accuracy_by_type = data.get('accuracy_by_type', {})
        for rule_type, rule_data in accuracy_by_type.items():
            rule_accuracy = rule_data.get('accuracy', 0.5)
            rule_total = rule_data.get('total', 0)
            
            if rule_total >= 10 and rule_accuracy < self.M1_SCALP_ACCURACY_THRESHOLD - thresholds['accuracy_drop_trigger']:
                adjustments.extend(self._apply_rule_specific_aggressive_tune(rule_type, multiplier))
        
        return adjustments
    
    def _apply_emergency_tightening(self, multiplier: float) -> List[Adjustment]:
        """Apply emergency tightening when performance is critically low."""
        adjustments = []
        
        current_confluence = self._current_parameters.min_confluence_required
        new_confluence = min(current_confluence + int(2 * multiplier), OptimizationParameters.MAX_CONFLUENCE)
        
        if new_confluence > current_confluence:
            adjustments.append(Adjustment(
                adjustment_type=AdjustmentType.EMERGENCY_TIGHTEN.value,
                parameter_name='min_confluence_required',
                old_value=current_confluence,
                new_value=new_confluence,
                reason=f"Emergency tightening: confluence {current_confluence} -> {new_confluence}"
            ))
        
        current_ema = self._current_parameters.ema_strictness
        new_ema = min(current_ema + (0.4 * multiplier), OptimizationParameters.MAX_EMA_STRICTNESS)
        
        if new_ema > current_ema:
            adjustments.append(Adjustment(
                adjustment_type=AdjustmentType.EMERGENCY_TIGHTEN.value,
                parameter_name='ema_strictness',
                old_value=current_ema,
                new_value=round(new_ema, 2),
                reason=f"Emergency EMA strictness increase"
            ))
        
        if self._current_parameters.enable_m5_low_adx:
            adjustments.append(Adjustment(
                adjustment_type=AdjustmentType.DISABLE_M5_LOW_ADX.value,
                parameter_name='enable_m5_low_adx',
                old_value=True,
                new_value=False,
                reason="Emergency disable M5 low ADX"
            ))
        
        return adjustments
    
    def _apply_fast_tune_confluence(self, multiplier: float) -> List[Adjustment]:
        """Apply fast tuning to confluence requirements."""
        adjustments = []
        
        current_confluence = self._current_parameters.min_confluence_required
        adjustment_step = max(1, int(self.MAX_ADJUSTMENT_STEP_CONFLUENCE * multiplier))
        new_confluence = min(current_confluence + adjustment_step, OptimizationParameters.MAX_CONFLUENCE)
        
        if new_confluence > current_confluence:
            adjustments.append(Adjustment(
                adjustment_type=AdjustmentType.FAST_TUNE_CONFLUENCE.value,
                parameter_name='min_confluence_required',
                old_value=current_confluence,
                new_value=new_confluence,
                reason=f"Fast tune: confluence {current_confluence} -> {new_confluence} (mult: {multiplier}x)"
            ))
        
        current_volume = self._current_parameters.volume_threshold_multiplier
        new_volume = min(current_volume + (0.15 * multiplier), OptimizationParameters.MAX_VOLUME_MULTIPLIER)
        
        if new_volume > current_volume:
            adjustments.append(Adjustment(
                adjustment_type=AdjustmentType.FAST_TUNE_CONFLUENCE.value,
                parameter_name='volume_threshold_multiplier',
                old_value=current_volume,
                new_value=round(new_volume, 2),
                reason=f"Fast tune: volume multiplier increase"
            ))
        
        return adjustments
    
    def _apply_aggressive_boost(self, multiplier: float) -> List[Adjustment]:
        """Apply aggressive boost when performance is excellent."""
        adjustments = []
        
        current_weight = self._current_parameters.three_confluence_weight
        new_weight = min(current_weight + (0.3 * multiplier), OptimizationParameters.MAX_3CONF_WEIGHT)
        
        if new_weight > current_weight:
            adjustments.append(Adjustment(
                adjustment_type=AdjustmentType.AGGRESSIVE_BOOST.value,
                parameter_name='three_confluence_weight',
                old_value=current_weight,
                new_value=round(new_weight, 2),
                reason=f"Aggressive boost: 3-conf weight increase"
            ))
        
        current_confluence = self._current_parameters.min_confluence_required
        if current_confluence > OptimizationParameters.MIN_CONFLUENCE:
            new_confluence = max(current_confluence - 1, OptimizationParameters.MIN_CONFLUENCE)
            adjustments.append(Adjustment(
                adjustment_type=AdjustmentType.AGGRESSIVE_BOOST.value,
                parameter_name='min_confluence_required',
                old_value=current_confluence,
                new_value=new_confluence,
                reason=f"Aggressive boost: loosen confluence for more signals"
            ))
        
        return adjustments
    
    def _apply_rule_specific_aggressive_tune(self, rule_type: str, multiplier: float) -> List[Adjustment]:
        """Apply rule-specific aggressive tuning."""
        adjustments = []
        
        if rule_type == 'M1_SCALP':
            current_cooldown = self._current_parameters.signal_cooldown
            new_cooldown = min(current_cooldown + int(10 * multiplier), OptimizationParameters.MAX_SIGNAL_COOLDOWN)
            
            if new_cooldown > current_cooldown:
                adjustments.append(Adjustment(
                    adjustment_type=AdjustmentType.AGGRESSIVE_REDUCE.value,
                    parameter_name='signal_cooldown',
                    old_value=current_cooldown,
                    new_value=new_cooldown,
                    reason=f"M1_SCALP underperforming: increase cooldown"
                ))
        
        elif rule_type == 'M5_SWING':
            current_adx = self._current_parameters.adx_threshold_for_m5
            new_adx = min(current_adx + (3 * multiplier), 25.0)
            
            if new_adx > current_adx:
                adjustments.append(Adjustment(
                    adjustment_type=AdjustmentType.AGGRESSIVE_REDUCE.value,
                    parameter_name='adx_threshold_for_m5',
                    old_value=current_adx,
                    new_value=round(new_adx, 1),
                    reason=f"M5_SWING underperforming: increase ADX threshold"
                ))
        
        return adjustments
    
    def enable_aggressive_tuning(self) -> None:
        """Enable aggressive tuning mode."""
        with self._lock:
            self._aggressive_tuning_enabled = True
            logger.info("🚀 Aggressive tuning enabled")
    
    def disable_aggressive_tuning(self) -> None:
        """Disable aggressive tuning mode."""
        with self._lock:
            self._aggressive_tuning_enabled = False
            self._emergency_mode = False
            logger.info("⏹️ Aggressive tuning disabled")
    
    def get_aggressive_tuning_status(self) -> Dict[str, Any]:
        """Get status of aggressive tuning."""
        with self._lock:
            return {
                'enabled': self._aggressive_tuning_enabled,
                'mode': self._tuning_mode.value,
                'emergency_mode': self._emergency_mode,
                'fast_tune_count': self._fast_tune_count,
                'last_aggressive_tune': self._last_aggressive_tune_time.isoformat() if self._last_aggressive_tune_time else None,
                'thresholds': self.AGGRESSIVE_THRESHOLDS.get(self._tuning_mode, {})
            }
    
    def _apply_adjustments_internal(self, adjustments: List[Adjustment]) -> bool:
        """Apply list of adjustments to current parameters"""
        try:
            self._rollback_parameters = deepcopy(self._current_parameters)
            
            for adj in adjustments:
                param_name = adj.parameter_name
                new_value = adj.new_value
                
                if hasattr(self._current_parameters, param_name):
                    setattr(self._current_parameters, param_name, new_value)
                    logger.debug(f"Applied adjustment: {param_name} = {new_value}")
                else:
                    logger.warning(f"Unknown parameter in adjustment: {param_name}")
            
            self._current_parameters.clamp_values()
            
            is_valid, errors = self._current_parameters.validate()
            if not is_valid:
                logger.error(f"Invalid parameters after adjustment: {errors}")
                self._current_parameters = self._rollback_parameters
                self._rollback_parameters = None
                return False
            
            if self.strategy_update_callback:
                callback_success = self.strategy_update_callback(self._current_parameters.to_dict())
                if not callback_success:
                    logger.warning("Strategy update callback returned False")
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying adjustments: {e}")
            if self._rollback_parameters:
                self._current_parameters = self._rollback_parameters
                self._rollback_parameters = None
            return False
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """Dapatkan parameter saat ini sebagai dictionary"""
        with self._lock:
            return self._current_parameters.to_dict()
    
    def apply_adjustments(self, adjustments_dict: Dict[str, Any]) -> bool:
        """
        Apply adjustments dari external source.
        
        Args:
            adjustments_dict: Dictionary dengan parameter adjustments
        
        Returns:
            True jika berhasil, False jika gagal
        """
        with self._lock:
            try:
                adjustments = []
                
                for param_name, new_value in adjustments_dict.items():
                    if hasattr(self._current_parameters, param_name):
                        old_value = getattr(self._current_parameters, param_name)
                        adjustments.append(Adjustment(
                            adjustment_type='external_adjustment',
                            parameter_name=param_name,
                            old_value=old_value,
                            new_value=new_value,
                            reason="External adjustment"
                        ))
                
                if not adjustments:
                    logger.warning("No valid parameters in adjustments_dict")
                    return False
                
                success = self._apply_adjustments_internal(adjustments)
                
                if success:
                    logger.info(f"Applied {len(adjustments)} external adjustments")
                    for adj in adjustments:
                        logger.info(f"  📊 {adj.parameter_name}: {adj.old_value} -> {adj.new_value}")
                
                return success
                
            except Exception as e:
                logger.error(f"Error applying external adjustments: {e}")
                return False
    
    def reset_to_defaults(self) -> None:
        """Reset semua parameter ke default values"""
        with self._lock:
            old_params = self._current_parameters.to_dict()
            self._current_parameters = OptimizationParameters()
            
            result = OptimizationResult(
                status=OptimizationStatus.SUCCESS.value,
                adjustments=[Adjustment(
                    adjustment_type=AdjustmentType.RESET_TO_DEFAULT.value,
                    parameter_name='all_parameters',
                    old_value=old_params,
                    new_value=self._current_parameters.to_dict(),
                    reason="Manual reset to defaults"
                )],
                parameters_before=old_params,
                parameters_after=self._current_parameters.to_dict()
            )
            
            self._add_to_history(result)
            
            if self.strategy_update_callback:
                self.strategy_update_callback(self._current_parameters.to_dict())
            
            logger.info("🔄 All parameters reset to defaults")
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Dapatkan history optimization sebagai list of dictionaries"""
        with self._lock:
            return [result.to_dict() for result in self._optimization_history]
    
    def rollback_last_adjustment(self) -> bool:
        """
        Rollback ke parameter sebelum adjustment terakhir.
        
        Returns:
            True jika berhasil rollback, False jika tidak ada rollback point
        """
        with self._lock:
            if self._rollback_parameters is None:
                logger.warning("No rollback point available")
                return False
            
            old_params = self._current_parameters.to_dict()
            self._current_parameters = deepcopy(self._rollback_parameters)
            self._rollback_parameters = None
            
            result = OptimizationResult(
                status=OptimizationStatus.ROLLED_BACK.value,
                adjustments=[Adjustment(
                    adjustment_type='rollback',
                    parameter_name='all_parameters',
                    old_value=old_params,
                    new_value=self._current_parameters.to_dict(),
                    reason="Manual rollback to previous state"
                )],
                parameters_before=old_params,
                parameters_after=self._current_parameters.to_dict()
            )
            
            self._add_to_history(result)
            
            if self.strategy_update_callback:
                self.strategy_update_callback(self._current_parameters.to_dict())
            
            logger.info("🔙 Rolled back to previous parameters")
            return True
    
    def check_and_rollback_if_needed(self, current_stats: Dict[str, Any]) -> bool:
        """
        Check performa setelah adjustment dan rollback jika memburuk.
        
        Args:
            current_stats: Current signal quality stats
        
        Returns:
            True jika rollback dilakukan, False jika tidak perlu
        """
        with self._lock:
            if not self._performance_snapshots or self._rollback_parameters is None:
                return False
            
            last_snapshot = self._performance_snapshots[-1]
            current_accuracy = current_stats.get('overall_accuracy', 0.0)
            
            performance_drop = last_snapshot.overall_accuracy - current_accuracy
            
            if performance_drop > self.ROLLBACK_PERFORMANCE_THRESHOLD:
                logger.warning(f"Performance dropped by {performance_drop*100:.1f}%, triggering rollback")
                return self.rollback_last_adjustment()
            
            return False
    
    def should_run_optimization(self) -> Tuple[bool, str]:
        """
        Check apakah sudah waktunya menjalankan optimization.
        
        Returns:
            Tuple (should_run, reason)
        """
        with self._lock:
            if not self._is_enabled:
                return False, "AutoOptimizer is disabled"
            
            if self._last_optimization_time is None:
                return True, "First optimization run"
            
            time_since_last = datetime.now(pytz.UTC) - self._last_optimization_time
            hours_since = time_since_last.total_seconds() / 3600
            
            if hours_since >= self.optimization_interval_hours:
                return True, f"{hours_since:.1f} hours since last optimization"
            
            remaining = self.optimization_interval_hours - hours_since
            return False, f"{remaining:.1f} hours until next optimization"
    
    def is_market_closed(self) -> bool:
        """Check if market is closed (weekend)"""
        now = datetime.now(pytz.UTC)
        return now.weekday() >= 5
    
    def reset_on_market_close(self) -> bool:
        """
        Reset parameters ke default saat market close (weekend).
        
        Returns:
            True jika reset dilakukan
        """
        with self._lock:
            if self.is_market_closed():
                logger.info("Market is closed, resetting to defaults")
                self.reset_to_defaults()
                return True
            return False
    
    def enable(self) -> None:
        """Enable AutoOptimizer"""
        with self._lock:
            self._is_enabled = True
            self._consecutive_failures = 0
            logger.info("AutoOptimizer enabled")
    
    def disable(self) -> None:
        """Disable AutoOptimizer"""
        with self._lock:
            self._is_enabled = False
            logger.info("AutoOptimizer disabled")
    
    def is_enabled(self) -> bool:
        """Check if AutoOptimizer is enabled"""
        with self._lock:
            return self._is_enabled
    
    def _save_performance_snapshot(self, data: Dict[str, Any]) -> None:
        """Save current performance snapshot for comparison"""
        accuracy_by_type = {}
        for rule, rule_data in data.get('accuracy_by_type', {}).items():
            accuracy_by_type[rule] = rule_data.get('accuracy', 0.0)
        
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(pytz.UTC),
            overall_accuracy=data.get('overall_accuracy', 0.0),
            parameters=deepcopy(self._current_parameters),
            signals_count=data.get('total_signals', 0),
            accuracy_by_type=accuracy_by_type
        )
        
        self._performance_snapshots.append(snapshot)
        
        if len(self._performance_snapshots) > 20:
            self._performance_snapshots = self._performance_snapshots[-20:]
    
    def _add_to_history(self, result: OptimizationResult) -> None:
        """Add result to optimization history"""
        self._optimization_history.append(result)
        
        if len(self._optimization_history) > self.MAX_OPTIMIZATION_HISTORY:
            self._optimization_history = self._optimization_history[-self.MAX_OPTIMIZATION_HISTORY:]
    
    def _generate_analysis_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of the analysis performed"""
        return {
            'overall_accuracy': data.get('overall_accuracy', 0.0),
            'total_signals': data.get('total_signals', 0),
            'profit_factor': data.get('profit_factor', 0.0),
            'best_performing_type': self._get_best_performing(data.get('accuracy_by_type', {})),
            'worst_performing_type': self._get_worst_performing(data.get('accuracy_by_type', {})),
            'best_performing_hour': data.get('best_performing_hour'),
            'worst_performing_hour': data.get('worst_performing_hour'),
            'current_parameters': self._current_parameters.to_dict()
        }
    
    def _get_best_performing(self, accuracy_dict: Dict[str, Dict]) -> Optional[str]:
        """Get best performing item from accuracy dict"""
        if not accuracy_dict:
            return None
        best = None
        best_acc = -1
        for name, data in accuracy_dict.items():
            acc = data.get('accuracy', 0) if isinstance(data, dict) else 0
            if acc > best_acc:
                best_acc = acc
                best = name
        return best
    
    def _get_worst_performing(self, accuracy_dict: Dict[str, Dict]) -> Optional[str]:
        """Get worst performing item from accuracy dict"""
        if not accuracy_dict:
            return None
        worst = None
        worst_acc = 2
        for name, data in accuracy_dict.items():
            acc = data.get('accuracy', 1) if isinstance(data, dict) else 1
            total = data.get('total', 0) if isinstance(data, dict) else 0
            if total >= 10 and acc < worst_acc:
                worst_acc = acc
                worst = name
        return worst
    
    def get_status(self) -> Dict[str, Any]:
        """
        Return status dict untuk command /optimize.
        
        Returns:
            Dict dengan status optimization
        """
        with self._lock:
            last_opt = self._optimization_history[-1] if self._optimization_history else None
            return {
                'enabled': self._is_enabled,
                'optimization_count': len(self._optimization_history),
                'last_optimization': str(self._last_optimization_time.isoformat()) if self._last_optimization_time else None,
                'next_optimization': 'Soon' if self._is_enabled else 'N/A',
                'current_parameters': self._current_parameters.to_dict(),
                'recent_changes': [adj.to_dict() for adj in (last_opt.adjustments if last_opt else [])],
                'performance_impact': {
                    'winrate_before': last_opt.analysis_summary.get('overall_accuracy', 0) * 100 if last_opt and last_opt.analysis_summary else 0,
                    'winrate_after': last_opt.analysis_summary.get('overall_accuracy', 0) * 100 if last_opt and last_opt.analysis_summary else 0
                } if last_opt else {}
            }
    
    def get_status_report(self) -> str:
        """
        Generate status report untuk Telegram.
        
        Returns:
            Formatted string untuk Telegram
        """
        with self._lock:
            report = "🤖 *Auto-Optimizer Status*\n\n"
            
            report += f"• Status: {'✅ Enabled' if self._is_enabled else '❌ Disabled'}\n"
            
            if self._last_optimization_time:
                time_since = datetime.now(pytz.UTC) - self._last_optimization_time
                hours_since = time_since.total_seconds() / 3600
                report += f"• Last Run: {hours_since:.1f}h ago\n"
            else:
                report += "• Last Run: Never\n"
            
            report += f"• Total Optimizations: {len(self._optimization_history)}\n\n"
            
            report += "📊 *Current Parameters*\n"
            report += f"• Min Confluence: {self._current_parameters.min_confluence_required}\n"
            report += f"• Volume Multiplier: {self._current_parameters.volume_threshold_multiplier:.2f}\n"
            report += f"• EMA Strictness: {self._current_parameters.ema_strictness:.2f}\n"
            report += f"• Signal Cooldown: {self._current_parameters.signal_cooldown}s\n"
            report += f"• M5 Low ADX: {'✅' if self._current_parameters.enable_m5_low_adx else '❌'}\n"
            report += f"• 3-Conf Weight: {self._current_parameters.three_confluence_weight:.2f}\n"
            
            if self._current_parameters.reduced_signal_hours:
                report += f"• Reduced Hours: {self._current_parameters.reduced_signal_hours}\n"
            
            if self._optimization_history:
                last_result = self._optimization_history[-1]
                report += f"\n📈 *Last Optimization*\n"
                report += f"• Status: {last_result.status}\n"
                report += f"• Adjustments: {len(last_result.adjustments)}\n"
                if last_result.analysis_summary:
                    acc = last_result.analysis_summary.get('overall_accuracy', 0)
                    report += f"• Accuracy: {acc*100:.1f}%\n"
            
            return report
    
    def set_signal_quality_tracker(self, tracker: SignalQualityTracker) -> None:
        """Set the SignalQualityTracker instance"""
        with self._lock:
            self.signal_quality_tracker = tracker
            logger.info("SignalQualityTracker set for AutoOptimizer")
    
    def set_strategy_update_callback(self, callback: Callable[[Dict[str, Any]], bool]) -> None:
        """Set the callback for updating TradingStrategy"""
        with self._lock:
            self.strategy_update_callback = callback
            logger.info("Strategy update callback set for AutoOptimizer")
