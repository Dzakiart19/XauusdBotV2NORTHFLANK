"""
Lightweight Auto Optimizer - Smart parameter tuning for Koyeb free tier.
Features: Performance-based adjustments, simple backtesting, threshold tuning.
"""
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from bot.logger import setup_logger

logger = setup_logger('AutoOptimizer')


class AdjustmentType(str, Enum):
    INCREASE = 'increase'
    DECREASE = 'decrease'
    NO_CHANGE = 'no_change'


class OptimizationStatus(str, Enum):
    IDLE = 'idle'
    RUNNING = 'running'
    COMPLETED = 'completed'


@dataclass
class OptimizationParameters:
    confidence_threshold: float = 60.0
    min_confluence: int = 2
    min_rr_ratio: float = 1.5
    trailing_distance: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'confidence_threshold': self.confidence_threshold,
            'min_confluence': self.min_confluence,
            'min_rr_ratio': self.min_rr_ratio,
            'trailing_distance': self.trailing_distance
        }


@dataclass
class Adjustment:
    parameter: str
    type: AdjustmentType
    old_value: float
    new_value: float
    reason: str


@dataclass
class OptimizationResult:
    success: bool = True
    message: str = ''
    parameters: Optional[OptimizationParameters] = None
    adjustments: List[Adjustment] = field(default_factory=list)
    status: str = 'idle'
    win_rate_before: float = 0.0
    win_rate_after: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'message': self.message,
            'parameters': self.parameters.to_dict() if self.parameters else None,
            'adjustments': [{'parameter': a.parameter, 'type': a.type.value,
                           'old': a.old_value, 'new': a.new_value, 'reason': a.reason}
                          for a in self.adjustments],
            'status': self.status,
            'win_rate_before': self.win_rate_before,
            'win_rate_after': self.win_rate_after
        }


@dataclass
class PerformanceSnapshot:
    win_rate: float = 0.0
    total_trades: int = 0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class AutoOptimizerError(Exception):
    pass


class AutoOptimizer:
    """
    Lightweight auto optimizer with:
    - Performance-based parameter adjustment
    - Simple threshold tuning
    - Win rate optimization
    - Conservative adjustment strategy
    """
    
    MIN_TRADES_FOR_OPTIMIZATION = 10
    OPTIMIZATION_COOLDOWN_HOURS = 6
    
    WIN_RATE_TARGET = 55.0
    WIN_RATE_POOR = 40.0
    WIN_RATE_EXCELLENT = 65.0
    
    CONFIDENCE_ADJUSTMENT_STEP = 5.0
    CONFLUENCE_ADJUSTMENT_STEP = 1
    RR_ADJUSTMENT_STEP = 0.1
    
    MAX_CONFIDENCE_THRESHOLD = 80.0
    MIN_CONFIDENCE_THRESHOLD = 45.0
    MAX_CONFLUENCE = 5
    MIN_CONFLUENCE = 2
    MAX_RR_RATIO = 2.5
    MIN_RR_RATIO = 1.2
    
    def __init__(self, signal_quality_tracker=None, config=None):
        self.signal_quality_tracker = signal_quality_tracker
        self.config = config
        self.enabled = True
        self.status = OptimizationStatus.IDLE
        
        self.current_parameters = OptimizationParameters(
            confidence_threshold=getattr(config, 'SIGNAL_HIGH_CONFIDENCE_THRESHOLD', 60.0) if config else 60.0,
            min_confluence=2,
            min_rr_ratio=getattr(config, 'TP_RR_RATIO', 1.5) if config else 1.5,
            trailing_distance=getattr(config, 'TRAILING_STOP_TRAIL_DISTANCE_PIPS', 0.5) if config else 0.5
        )
        
        self.performance_history: List[PerformanceSnapshot] = []
        self.optimization_history: List[OptimizationResult] = []
        self.last_optimization_time: Optional[datetime] = None
        
        logger.info("AutoOptimizer initialized with smart parameter tuning")
    
    def _get_current_performance(self) -> PerformanceSnapshot:
        """Get current performance snapshot from signal quality tracker"""
        if not self.signal_quality_tracker:
            return PerformanceSnapshot()
        
        stats = self.signal_quality_tracker.get_stats()
        return PerformanceSnapshot(
            win_rate=stats.get('win_rate', 0.0),
            total_trades=stats.get('total_signals', 0),
            total_pnl=stats.get('total_pnl', 0.0),
            timestamp=datetime.now()
        )
    
    def _analyze_performance(self, snapshot: PerformanceSnapshot) -> List[Adjustment]:
        """Analyze performance and suggest adjustments"""
        adjustments = []
        
        if snapshot.total_trades < self.MIN_TRADES_FOR_OPTIMIZATION:
            return adjustments
        
        if snapshot.win_rate < self.WIN_RATE_POOR:
            if self.current_parameters.confidence_threshold < self.MAX_CONFIDENCE_THRESHOLD:
                new_conf = min(self.MAX_CONFIDENCE_THRESHOLD,
                              self.current_parameters.confidence_threshold + self.CONFIDENCE_ADJUSTMENT_STEP)
                adjustments.append(Adjustment(
                    parameter='confidence_threshold',
                    type=AdjustmentType.INCREASE,
                    old_value=self.current_parameters.confidence_threshold,
                    new_value=new_conf,
                    reason=f"Win rate ({snapshot.win_rate:.1f}%) below {self.WIN_RATE_POOR}%"
                ))
            
            if self.current_parameters.min_confluence < self.MAX_CONFLUENCE:
                new_conf = min(self.MAX_CONFLUENCE,
                              self.current_parameters.min_confluence + self.CONFLUENCE_ADJUSTMENT_STEP)
                adjustments.append(Adjustment(
                    parameter='min_confluence',
                    type=AdjustmentType.INCREASE,
                    old_value=self.current_parameters.min_confluence,
                    new_value=new_conf,
                    reason=f"Increasing signal quality requirements"
                ))
        
        elif snapshot.win_rate > self.WIN_RATE_EXCELLENT:
            if self.current_parameters.confidence_threshold > self.MIN_CONFIDENCE_THRESHOLD:
                new_conf = max(self.MIN_CONFIDENCE_THRESHOLD,
                              self.current_parameters.confidence_threshold - self.CONFIDENCE_ADJUSTMENT_STEP)
                adjustments.append(Adjustment(
                    parameter='confidence_threshold',
                    type=AdjustmentType.DECREASE,
                    old_value=self.current_parameters.confidence_threshold,
                    new_value=new_conf,
                    reason=f"Win rate ({snapshot.win_rate:.1f}%) above {self.WIN_RATE_EXCELLENT}% - can be more aggressive"
                ))
        
        return adjustments
    
    def _apply_adjustments(self, adjustments: List[Adjustment]) -> None:
        """Apply adjustments to current parameters"""
        for adj in adjustments:
            if adj.parameter == 'confidence_threshold':
                self.current_parameters.confidence_threshold = adj.new_value
            elif adj.parameter == 'min_confluence':
                self.current_parameters.min_confluence = int(adj.new_value)
            elif adj.parameter == 'min_rr_ratio':
                self.current_parameters.min_rr_ratio = adj.new_value
            elif adj.parameter == 'trailing_distance':
                self.current_parameters.trailing_distance = adj.new_value
            
            logger.info(f"📊 Parameter adjusted: {adj.parameter} {adj.old_value} → {adj.new_value} ({adj.reason})")
    
    def should_run_optimization(self) -> Tuple[bool, str]:
        """Check if optimization should run"""
        if not self.enabled:
            return (False, "Optimization disabled")
        
        if self.status == OptimizationStatus.RUNNING:
            return (False, "Optimization already running")
        
        if self.last_optimization_time:
            hours_since = (datetime.now() - self.last_optimization_time).total_seconds() / 3600
            if hours_since < self.OPTIMIZATION_COOLDOWN_HOURS:
                return (False, f"Cooldown: {self.OPTIMIZATION_COOLDOWN_HOURS - hours_since:.1f}h remaining")
        
        perf = self._get_current_performance()
        if perf.total_trades < self.MIN_TRADES_FOR_OPTIMIZATION:
            return (False, f"Need {self.MIN_TRADES_FOR_OPTIMIZATION - perf.total_trades} more trades")
        
        return (True, "Ready for optimization")
    
    def optimize(self) -> OptimizationResult:
        """Run optimization cycle"""
        self.status = OptimizationStatus.RUNNING
        
        try:
            perf_before = self._get_current_performance()
            
            if perf_before.total_trades < self.MIN_TRADES_FOR_OPTIMIZATION:
                return OptimizationResult(
                    success=False,
                    message=f"Insufficient trades ({perf_before.total_trades}/{self.MIN_TRADES_FOR_OPTIMIZATION})",
                    status='insufficient_data'
                )
            
            adjustments = self._analyze_performance(perf_before)
            
            if not adjustments:
                result = OptimizationResult(
                    success=True,
                    message="No adjustments needed - performance is optimal",
                    parameters=self.current_parameters,
                    status='optimal',
                    win_rate_before=perf_before.win_rate
                )
            else:
                self._apply_adjustments(adjustments)
                
                result = OptimizationResult(
                    success=True,
                    message=f"Applied {len(adjustments)} adjustments",
                    parameters=self.current_parameters,
                    adjustments=adjustments,
                    status='adjusted',
                    win_rate_before=perf_before.win_rate
                )
            
            self.performance_history.append(perf_before)
            if len(self.performance_history) > 50:
                self.performance_history = self.performance_history[-50:]
            
            self.optimization_history.append(result)
            if len(self.optimization_history) > 20:
                self.optimization_history = self.optimization_history[-20:]
            
            self.last_optimization_time = datetime.now()
            self.status = OptimizationStatus.COMPLETED
            
            return result
            
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            self.status = OptimizationStatus.IDLE
            return OptimizationResult(
                success=False,
                message=f"Error: {str(e)}",
                status='error'
            )
    
    def run_optimization(self) -> OptimizationResult:
        """Alias for optimize()"""
        return self.optimize()
    
    def get_status(self) -> Dict[str, Any]:
        """Get optimizer status"""
        perf = self._get_current_performance()
        should_run, reason = self.should_run_optimization()
        
        return {
            'enabled': self.enabled,
            'status': self.status.value,
            'current_parameters': self.current_parameters.to_dict(),
            'current_win_rate': perf.win_rate,
            'total_trades': perf.total_trades,
            'should_optimize': should_run,
            'optimization_reason': reason,
            'last_optimization': self.last_optimization_time.isoformat() if self.last_optimization_time else None,
            'total_optimizations': len(self.optimization_history)
        }
    
    def get_status_report(self) -> str:
        """Get human-readable status report"""
        status = self.get_status()
        
        return f"""🔧 Auto-Optimizer Status
━━━━━━━━━━━━━━━━━━━━━━━━
Status: {status['status'].title()}
Win Rate: {status['current_win_rate']:.1f}%
Total Trades: {status['total_trades']}

Current Parameters:
  • Confidence: {status['current_parameters']['confidence_threshold']:.0f}%
  • Min Confluence: {status['current_parameters']['min_confluence']}
  • R:R Ratio: {status['current_parameters']['min_rr_ratio']:.1f}
  • Trail Distance: {status['current_parameters']['trailing_distance']} pips

Ready to Optimize: {'Yes' if status['should_optimize'] else 'No'}
{status['optimization_reason']}"""
    
    def get_recommended_parameters(self) -> Dict[str, Any]:
        """Get currently recommended parameters"""
        return self.current_parameters.to_dict()
