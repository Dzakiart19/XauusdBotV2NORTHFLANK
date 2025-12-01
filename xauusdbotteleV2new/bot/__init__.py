"""
Bot Trading XAUUSD Module.

This package provides trading bot functionality for XAUUSD including:
- Signal generation and quality tracking
- Auto-optimization engine
- Market regime detection
- Risk management
"""

from bot.auto_optimizer import (
    AutoOptimizer,
    OptimizationParameters,
    OptimizationResult,
    Adjustment,
    AdjustmentType,
    OptimizationStatus,
    PerformanceSnapshot,
    AutoOptimizerError
)

__all__ = [
    'AutoOptimizer',
    'OptimizationParameters',
    'OptimizationResult',
    'Adjustment',
    'AdjustmentType',
    'OptimizationStatus',
    'PerformanceSnapshot',
    'AutoOptimizerError'
]
