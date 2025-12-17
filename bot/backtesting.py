"""
Lightweight backtesting stub - disabled for Koyeb free tier.
"""
from typing import Dict, Any
from bot.logger import setup_logger

logger = setup_logger('Backtesting')


class BacktestEngine:
    """Backtest engine stub - disabled"""
    
    def __init__(self, *args, **kwargs):
        pass
    
    def run(self) -> Dict[str, Any]:
        return {'enabled': False, 'message': 'Backtesting disabled for lightweight deployment'}


class StrategyOptimizer:
    """Strategy optimizer stub - disabled"""
    
    def __init__(self, *args, **kwargs):
        pass
    
    def optimize(self) -> Dict[str, Any]:
        return {'enabled': False}
