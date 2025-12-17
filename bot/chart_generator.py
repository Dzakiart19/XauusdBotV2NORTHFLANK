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
    
    async def generate_chart_async(self, *args, **kwargs) -> Optional[str]:
        """Async version of generate_chart - returns None (charts disabled)"""
        return None
    
    async def immediate_delete_chart_async(self, chart_path: str) -> None:
        """Async delete chart - no-op since charts are disabled"""
        pass
    
    def shutdown(self):
        pass
    
    def cleanup(self):
        pass
