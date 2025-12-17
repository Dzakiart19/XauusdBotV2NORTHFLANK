"""
Lightweight chart generator stub - disabled for Koyeb free tier.
"""
from typing import Optional, Any
from bot.logger import setup_logger

logger = setup_logger('ChartGenerator')


class ChartGenerator:
    """Chart generator stub - charts disabled for lightweight deployment"""
    
    def __init__(self, config=None):
        self.config = config
        self.enabled = False
    
    def generate_chart(self, *args, **kwargs) -> Optional[str]:
        return None
    
    def shutdown(self):
        pass
    
    def cleanup(self):
        pass
