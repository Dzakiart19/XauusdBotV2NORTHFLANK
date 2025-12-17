"""
Lightweight report generator stub - simplified for Koyeb free tier.
"""
from typing import Dict, Any, Optional
from datetime import datetime
from bot.logger import setup_logger

logger = setup_logger('ReportGenerator')


class ReportGenerator:
    """Simple report generator stub"""
    
    def __init__(self, db_manager=None, config=None):
        self.db_manager = db_manager
        self.config = config
    
    def generate_daily_report(self) -> str:
        return "Daily report generation disabled for lightweight deployment"
    
    def generate_weekly_report(self) -> str:
        return "Weekly report generation disabled for lightweight deployment"


class ScheduledReportManager:
    """Stub for scheduled reports - disabled"""
    
    def __init__(self, *args, **kwargs):
        pass
    
    async def start(self):
        pass
    
    async def stop(self):
        pass
