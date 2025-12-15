"""
Admin Monitoring Dashboard Module untuk Bot Trading XAUUSD.

Modul ini menyediakan:
- Health metrics monitoring
- Performance benchmarking
- System resource tracking
- Error rate monitoring
- Alerting untuk admin
"""

import asyncio
import os
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import pytz

from bot.logger import setup_logger

logger = setup_logger('AdminMonitor')


class AdminMonitorError(Exception):
    """Exception untuk admin monitor errors"""
    pass


@dataclass
class SystemMetrics:
    """Metrics sistem"""
    timestamp: datetime
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    disk_percent: float = 0.0
    open_files: int = 0
    threads: int = 0
    connections: int = 0


@dataclass
class BotMetrics:
    """Metrics bot"""
    timestamp: datetime
    signals_generated: int = 0
    signals_sent: int = 0
    active_positions: int = 0
    active_users: int = 0
    websocket_connected: bool = False
    telegram_connected: bool = False
    database_connected: bool = False
    error_count_1h: int = 0
    warning_count_1h: int = 0


@dataclass
class PerformanceMetrics:
    """Performance metrics"""
    timestamp: datetime
    avg_signal_latency_ms: float = 0.0
    avg_api_response_ms: float = 0.0
    avg_db_query_ms: float = 0.0
    tick_processing_rate: float = 0.0
    message_delivery_rate: float = 0.0


class MetricsCollector:
    """Collector untuk berbagai metrics"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.system_metrics: deque = deque(maxlen=max_history)
        self.bot_metrics: deque = deque(maxlen=max_history)
        self.performance_metrics: deque = deque(maxlen=max_history)
        self.error_log: deque = deque(maxlen=500)
        self.jakarta_tz = pytz.timezone('Asia/Jakarta')
        
        self._latency_samples: List[float] = []
        self._api_response_samples: List[float] = []
        self._db_query_samples: List[float] = []
        
        logger.info("MetricsCollector initialized")
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect system resource metrics"""
        try:
            process = psutil.Process()
            
            metrics = SystemMetrics(
                timestamp=datetime.now(self.jakarta_tz),
                cpu_percent=psutil.cpu_percent(interval=0.1),
                memory_percent=psutil.virtual_memory().percent,
                memory_used_mb=psutil.virtual_memory().used / (1024 * 1024),
                memory_available_mb=psutil.virtual_memory().available / (1024 * 1024),
                disk_percent=psutil.disk_usage('/').percent,
                open_files=len(process.open_files()),
                threads=process.num_threads(),
                connections=len(process.connections())
            )
            
            self.system_metrics.append(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(timestamp=datetime.now(self.jakarta_tz))
    
    def collect_bot_metrics(self, signal_rules=None, position_tracker=None,
                            market_data=None, telegram_bot=None,
                            db_manager=None) -> BotMetrics:
        """Collect bot-specific metrics"""
        try:
            metrics = BotMetrics(
                timestamp=datetime.now(self.jakarta_tz),
                websocket_connected=market_data.is_connected() if market_data else False,
                telegram_connected=telegram_bot is not None,
                database_connected=db_manager is not None
            )
            
            if signal_rules:
                metrics.signals_generated = getattr(signal_rules, 'total_signals_generated', 0)
            
            if position_tracker:
                metrics.active_positions = len(getattr(position_tracker, 'active_positions', {}))
            
            self.bot_metrics.append(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting bot metrics: {e}")
            return BotMetrics(timestamp=datetime.now(self.jakarta_tz))
    
    def record_latency(self, latency_ms: float, category: str = 'signal'):
        """Record latency sample"""
        if category == 'signal':
            self._latency_samples.append(latency_ms)
            if len(self._latency_samples) > 100:
                self._latency_samples = self._latency_samples[-100:]
        elif category == 'api':
            self._api_response_samples.append(latency_ms)
            if len(self._api_response_samples) > 100:
                self._api_response_samples = self._api_response_samples[-100:]
        elif category == 'db':
            self._db_query_samples.append(latency_ms)
            if len(self._db_query_samples) > 100:
                self._db_query_samples = self._db_query_samples[-100:]
    
    def collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect performance metrics"""
        try:
            metrics = PerformanceMetrics(
                timestamp=datetime.now(self.jakarta_tz),
                avg_signal_latency_ms=sum(self._latency_samples) / len(self._latency_samples) if self._latency_samples else 0,
                avg_api_response_ms=sum(self._api_response_samples) / len(self._api_response_samples) if self._api_response_samples else 0,
                avg_db_query_ms=sum(self._db_query_samples) / len(self._db_query_samples) if self._db_query_samples else 0
            )
            
            self.performance_metrics.append(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
            return PerformanceMetrics(timestamp=datetime.now(self.jakarta_tz))
    
    def log_error(self, error_type: str, message: str, details: Optional[Dict] = None):
        """Log error untuk tracking"""
        self.error_log.append({
            'timestamp': datetime.now(self.jakarta_tz).isoformat(),
            'type': error_type,
            'message': message,
            'details': details or {}
        })
    
    def get_error_count(self, hours: int = 1) -> int:
        """Get error count dalam N jam terakhir"""
        cutoff = datetime.now(self.jakarta_tz) - timedelta(hours=hours)
        count = 0
        for error in self.error_log:
            try:
                error_time = datetime.fromisoformat(error['timestamp'])
                if error_time > cutoff:
                    count += 1
            except (ValueError, KeyError):
                pass
        return count


class AlertManager:
    """Manager untuk admin alerts"""
    
    def __init__(self, telegram_bot=None, admin_chat_ids: Optional[List[int]] = None):
        self.telegram_bot = telegram_bot
        self.admin_chat_ids = admin_chat_ids or []
        self.alert_history: deque = deque(maxlen=100)
        self.suppressed_alerts: Dict[str, datetime] = {}
        self.suppression_duration = timedelta(minutes=30)
        self.jakarta_tz = pytz.timezone('Asia/Jakarta')
        
        self.thresholds = {
            'cpu_critical': 90.0,
            'cpu_warning': 75.0,
            'memory_critical': 90.0,
            'memory_warning': 80.0,
            'disk_critical': 95.0,
            'disk_warning': 85.0,
            'error_rate_critical': 10,
            'error_rate_warning': 5
        }
        
        logger.info("AlertManager initialized")
    
    def set_threshold(self, name: str, value: float):
        """Set alert threshold"""
        if name in self.thresholds:
            self.thresholds[name] = value
            logger.info(f"Threshold {name} set to {value}")
    
    def _is_suppressed(self, alert_key: str) -> bool:
        """Check if alert is suppressed"""
        if alert_key in self.suppressed_alerts:
            if datetime.now(self.jakarta_tz) < self.suppressed_alerts[alert_key]:
                return True
            else:
                del self.suppressed_alerts[alert_key]
        return False
    
    def _suppress_alert(self, alert_key: str):
        """Suppress alert untuk durasi tertentu"""
        self.suppressed_alerts[alert_key] = datetime.now(self.jakarta_tz) + self.suppression_duration
    
    async def check_and_alert(self, system_metrics: SystemMetrics, 
                               bot_metrics: BotMetrics,
                               error_count: int):
        """Check metrics dan kirim alert jika perlu"""
        alerts = []
        
        if system_metrics.cpu_percent >= self.thresholds['cpu_critical']:
            alerts.append(('critical', 'cpu', f"CPU usage critical: {system_metrics.cpu_percent:.1f}%"))
        elif system_metrics.cpu_percent >= self.thresholds['cpu_warning']:
            alerts.append(('warning', 'cpu', f"CPU usage high: {system_metrics.cpu_percent:.1f}%"))
        
        if system_metrics.memory_percent >= self.thresholds['memory_critical']:
            alerts.append(('critical', 'memory', f"Memory usage critical: {system_metrics.memory_percent:.1f}%"))
        elif system_metrics.memory_percent >= self.thresholds['memory_warning']:
            alerts.append(('warning', 'memory', f"Memory usage high: {system_metrics.memory_percent:.1f}%"))
        
        if system_metrics.disk_percent >= self.thresholds['disk_critical']:
            alerts.append(('critical', 'disk', f"Disk usage critical: {system_metrics.disk_percent:.1f}%"))
        elif system_metrics.disk_percent >= self.thresholds['disk_warning']:
            alerts.append(('warning', 'disk', f"Disk usage high: {system_metrics.disk_percent:.1f}%"))
        
        if error_count >= self.thresholds['error_rate_critical']:
            alerts.append(('critical', 'errors', f"High error rate: {error_count} errors/hour"))
        elif error_count >= self.thresholds['error_rate_warning']:
            alerts.append(('warning', 'errors', f"Elevated error rate: {error_count} errors/hour"))
        
        if not bot_metrics.websocket_connected:
            alerts.append(('critical', 'websocket', "WebSocket disconnected"))
        
        for severity, alert_type, message in alerts:
            alert_key = f"{alert_type}_{severity}"
            
            if not self._is_suppressed(alert_key):
                await self._send_alert(severity, alert_type, message)
                self._suppress_alert(alert_key)
    
    async def _send_alert(self, severity: str, alert_type: str, message: str):
        """Kirim alert ke admin"""
        emoji = "🚨" if severity == 'critical' else "⚠️"
        
        alert_text = f"{emoji} *Admin Alert: {severity.upper()}*\n\n"
        alert_text += f"Type: {alert_type}\n"
        alert_text += f"Message: {message}\n"
        alert_text += f"Time: {datetime.now(self.jakarta_tz).strftime('%H:%M:%S WIB')}"
        
        self.alert_history.append({
            'timestamp': datetime.now(self.jakarta_tz).isoformat(),
            'severity': severity,
            'type': alert_type,
            'message': message
        })
        
        if self.telegram_bot and self.admin_chat_ids:
            for chat_id in self.admin_chat_ids:
                try:
                    await self.telegram_bot.bot.send_message(
                        chat_id=chat_id,
                        text=alert_text,
                        parse_mode='Markdown'
                    )
                except Exception as e:
                    logger.error(f"Failed to send admin alert: {e}")
        
        logger.warning(f"Admin Alert [{severity}]: {message}")


class AdminDashboard:
    """Admin dashboard untuk monitoring"""
    
    def __init__(self, config=None):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.jakarta_tz = pytz.timezone('Asia/Jakarta')
        self._monitoring_task = None
        self._is_monitoring = False
        
        logger.info("AdminDashboard initialized")
    
    def set_components(self, signal_rules=None, position_tracker=None,
                       market_data=None, telegram_bot=None, db_manager=None):
        """Set komponen untuk monitoring"""
        self.signal_rules = signal_rules
        self.position_tracker = position_tracker
        self.market_data = market_data
        self.telegram_bot = telegram_bot
        self.db_manager = db_manager
        
        if telegram_bot:
            self.alert_manager.telegram_bot = telegram_bot
    
    def set_admin_chat_ids(self, chat_ids: List[int]):
        """Set admin chat IDs untuk alerting"""
        self.alert_manager.admin_chat_ids = chat_ids
    
    async def start_monitoring(self, interval_seconds: int = 60):
        """Start background monitoring"""
        if self._is_monitoring:
            logger.warning("Monitoring already running")
            return
        
        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop(interval_seconds))
        logger.info(f"Admin monitoring started (interval: {interval_seconds}s)")
    
    async def stop_monitoring(self):
        """Stop background monitoring"""
        self._is_monitoring = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Admin monitoring stopped")
    
    async def _monitoring_loop(self, interval: int):
        """Background monitoring loop"""
        while self._is_monitoring:
            try:
                system_metrics = self.metrics_collector.collect_system_metrics()
                bot_metrics = self.metrics_collector.collect_bot_metrics(
                    self.signal_rules,
                    self.position_tracker,
                    self.market_data,
                    self.telegram_bot,
                    self.db_manager
                )
                self.metrics_collector.collect_performance_metrics()
                
                error_count = self.metrics_collector.get_error_count(hours=1)
                
                await self.alert_manager.check_and_alert(
                    system_metrics, bot_metrics, error_count
                )
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data untuk dashboard display"""
        system_metrics = list(self.metrics_collector.system_metrics)[-1] if self.metrics_collector.system_metrics else None
        bot_metrics = list(self.metrics_collector.bot_metrics)[-1] if self.metrics_collector.bot_metrics else None
        perf_metrics = list(self.metrics_collector.performance_metrics)[-1] if self.metrics_collector.performance_metrics else None
        
        return {
            'timestamp': datetime.now(self.jakarta_tz).isoformat(),
            'system': {
                'cpu_percent': system_metrics.cpu_percent if system_metrics else 0,
                'memory_percent': system_metrics.memory_percent if system_metrics else 0,
                'memory_used_mb': round(system_metrics.memory_used_mb, 1) if system_metrics else 0,
                'disk_percent': system_metrics.disk_percent if system_metrics else 0,
                'threads': system_metrics.threads if system_metrics else 0
            } if system_metrics else {},
            'bot': {
                'websocket_connected': bot_metrics.websocket_connected if bot_metrics else False,
                'telegram_connected': bot_metrics.telegram_connected if bot_metrics else False,
                'database_connected': bot_metrics.database_connected if bot_metrics else False,
                'active_positions': bot_metrics.active_positions if bot_metrics else 0,
                'signals_generated': bot_metrics.signals_generated if bot_metrics else 0
            } if bot_metrics else {},
            'performance': {
                'avg_signal_latency_ms': round(perf_metrics.avg_signal_latency_ms, 2) if perf_metrics else 0,
                'avg_api_response_ms': round(perf_metrics.avg_api_response_ms, 2) if perf_metrics else 0,
                'avg_db_query_ms': round(perf_metrics.avg_db_query_ms, 2) if perf_metrics else 0
            } if perf_metrics else {},
            'errors': {
                'count_1h': self.metrics_collector.get_error_count(hours=1),
                'recent': list(self.metrics_collector.error_log)[-5:]
            },
            'alerts': {
                'recent': list(self.alert_manager.alert_history)[-10:]
            },
            'uptime': self._get_uptime()
        }
    
    def _get_uptime(self) -> str:
        """Get bot uptime"""
        try:
            process = psutil.Process()
            create_time = datetime.fromtimestamp(process.create_time())
            uptime = datetime.now() - create_time
            
            days = uptime.days
            hours = uptime.seconds // 3600
            minutes = (uptime.seconds % 3600) // 60
            
            if days > 0:
                return f"{days}d {hours}h {minutes}m"
            elif hours > 0:
                return f"{hours}h {minutes}m"
            else:
                return f"{minutes}m"
        except Exception:
            return "Unknown"
    
    def get_metrics_history(self, metric_type: str = 'system', 
                            limit: int = 100) -> List[Dict[str, Any]]:
        """Get historical metrics"""
        if metric_type == 'system':
            metrics = list(self.metrics_collector.system_metrics)[-limit:]
            return [
                {
                    'timestamp': m.timestamp.isoformat(),
                    'cpu_percent': m.cpu_percent,
                    'memory_percent': m.memory_percent
                }
                for m in metrics
            ]
        elif metric_type == 'performance':
            metrics = list(self.metrics_collector.performance_metrics)[-limit:]
            return [
                {
                    'timestamp': m.timestamp.isoformat(),
                    'signal_latency': m.avg_signal_latency_ms,
                    'api_response': m.avg_api_response_ms
                }
                for m in metrics
            ]
        return []
    
    def format_status_message(self) -> str:
        """Format status message untuk Telegram"""
        data = self.get_dashboard_data()
        
        system = data.get('system', {})
        bot = data.get('bot', {})
        errors = data.get('errors', {})
        
        ws_status = "🟢" if bot.get('websocket_connected') else "🔴"
        tg_status = "🟢" if bot.get('telegram_connected') else "🔴"
        db_status = "🟢" if bot.get('database_connected') else "🔴"
        
        text = "📊 *Admin Status Dashboard*\n\n"
        text += "━━━━━━━━━━━━━━━━━━━━\n"
        text += "*System Resources:*\n"
        text += f"💻 CPU: {system.get('cpu_percent', 0):.1f}%\n"
        text += f"🧠 Memory: {system.get('memory_percent', 0):.1f}% ({system.get('memory_used_mb', 0):.0f} MB)\n"
        text += f"💾 Disk: {system.get('disk_percent', 0):.1f}%\n"
        text += f"🧵 Threads: {system.get('threads', 0)}\n"
        text += "━━━━━━━━━━━━━━━━━━━━\n"
        text += "*Connections:*\n"
        text += f"{ws_status} WebSocket: {'Connected' if bot.get('websocket_connected') else 'Disconnected'}\n"
        text += f"{tg_status} Telegram: {'Connected' if bot.get('telegram_connected') else 'Disconnected'}\n"
        text += f"{db_status} Database: {'Connected' if bot.get('database_connected') else 'Disconnected'}\n"
        text += "━━━━━━━━━━━━━━━━━━━━\n"
        text += "*Bot Status:*\n"
        text += f"📍 Active Positions: {bot.get('active_positions', 0)}\n"
        text += f"📡 Signals Generated: {bot.get('signals_generated', 0)}\n"
        text += f"⚠️ Errors (1h): {errors.get('count_1h', 0)}\n"
        text += "━━━━━━━━━━━━━━━━━━━━\n"
        text += f"⏱️ Uptime: {data.get('uptime', 'Unknown')}\n"
        text += f"🕐 {datetime.now(self.jakarta_tz).strftime('%H:%M:%S WIB')}"
        
        return text
