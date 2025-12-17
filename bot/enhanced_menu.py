"""
Lightweight enhanced menu stub - simplified for Koyeb free tier.
"""
from typing import Dict, Any, Optional
from bot.logger import setup_logger

logger = setup_logger('EnhancedMenu')


class EnhancedMenuHandler:
    """Simple menu handler stub"""
    
    def __init__(self, *args, **kwargs):
        pass
    
    def get_main_menu(self) -> Dict[str, Any]:
        return {}
