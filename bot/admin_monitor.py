"""
Lightweight admin monitor stub - disabled for Koyeb free tier.
"""
from typing import Dict, Any
from bot.logger import setup_logger

logger = setup_logger('AdminMonitor')


class AdminDashboard:
    """Admin dashboard stub - disabled"""
    
    def __init__(self, *args, **kwargs):
        pass
    
    def get_status(self) -> Dict[str, Any]:
        return {'enabled': False}


class MetricsCollector:
    """Metrics collector stub - disabled"""
    
    def __init__(self, *args, **kwargs):
        pass
    
    def collect(self) -> Dict[str, Any]:
        return {}


class AlertManager:
    """Alert manager stub - disabled"""
    
    def __init__(self, *args, **kwargs):
        pass
