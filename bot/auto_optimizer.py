"""
Lightweight auto optimizer stub - disabled for Koyeb free tier.
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
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
    confidence_threshold: float = 0.6
    min_rr_ratio: float = 1.5


@dataclass
class OptimizationResult:
    success: bool = True
    message: str = ''
    parameters: OptimizationParameters = None


@dataclass
class Adjustment:
    type: AdjustmentType = AdjustmentType.NO_CHANGE
    value: float = 0


@dataclass
class PerformanceSnapshot:
    win_rate: float = 0
    total_trades: int = 0


class AutoOptimizerError(Exception):
    pass


class AutoOptimizer:
    """Stub auto optimizer - disabled for lightweight deployment"""
    
    def __init__(self, signal_quality_tracker=None, config=None):
        self.enabled = False
        self.status = OptimizationStatus.IDLE
    
    def optimize(self) -> OptimizationResult:
        return OptimizationResult(success=True, message="Auto-optimization disabled")
    
    def get_status(self) -> Dict[str, Any]:
        return {
            'enabled': False,
            'status': 'disabled',
            'message': 'Auto-optimization disabled for lightweight deployment'
        }
