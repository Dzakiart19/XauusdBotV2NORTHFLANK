import asyncio
import signal
import sys
import os
import fcntl
import atexit
from aiohttp import web
from typing import Optional, Dict, Set, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from sqlalchemy import text

LOCK_FILE_PATH = '/tmp/xauusd_trading_bot.lock'

from config import Config, ConfigError
from bot.logger import setup_logger, mask_token, sanitize_log_message
from bot.database import DatabaseManager
from bot.sentry_integration import initialize_sentry, get_sentry_manager
from bot.backup import DatabaseBackupManager
from bot.market_data import MarketDataClient
from bot.strategy import TradingStrategy
from bot.risk_manager import RiskManager
from bot.position_tracker import PositionTracker
from bot.chart_generator import ChartGenerator
from bot.telegram_bot import TradingBot
from bot.alert_system import AlertSystem
from bot.error_handler import ErrorHandler
from bot.user_manager import UserManager
from bot.task_scheduler import TaskScheduler, setup_default_tasks
from bot.signal_session_manager import SignalSessionManager
from bot.market_regime import MarketRegimeDetector
from bot.signal_rules import AggressiveSignalRules
from bot.signal_quality_tracker import SignalQualityTracker
from bot.auto_optimizer import AutoOptimizer
from bot.indicators import IndicatorEngine

logger = setup_logger('Main')


class TaskPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class TaskInfo:
    name: str
    task: asyncio.Task
    priority: TaskPriority = TaskPriority.NORMAL
    critical: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    status: TaskStatus = TaskStatus.RUNNING
    cancel_timeout: float = 5.0
    restart_count: int = 0
    max_restarts: int = 3
    last_restart_at: Optional[datetime] = None
    
    def is_done(self) -> bool:
        return self.task.done()
    
    def is_cancelled(self) -> bool:
        return self.task.cancelled()
    
    def can_restart(self) -> bool:
        if self.max_restarts < 0:
            return True
        return self.restart_count < self.max_restarts


class SingleInstanceLock:
    """Prevent multiple bot instances from running simultaneously using file lock"""
    
    def __init__(self, lock_file_path: str = LOCK_FILE_PATH):
        self.lock_file_path = lock_file_path
        self.lock_file = None
        self.acquired = False
    
    def acquire(self) -> bool:
        """Try to acquire exclusive lock on the lock file
        
        Returns:
            True if lock acquired successfully, False if another instance is running
        """
        try:
            self.lock_file = open(self.lock_file_path, 'w')
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            
            self.lock_file.write(f"{os.getpid()}\n")
            self.lock_file.write(f"{datetime.now().isoformat()}\n")
            self.lock_file.flush()
            
            self.acquired = True
            atexit.register(self.release)
            logger.info(f"Bot instance lock acquired (PID: {os.getpid()})")
            return True
            
        except (IOError, OSError) as e:
            if self.lock_file:
                self.lock_file.close()
            
            existing_pid = "unknown"
            try:
                with open(self.lock_file_path, 'r') as f:
                    existing_pid = f.readline().strip()
            except (IOError, OSError):
                pass
            
            logger.error(f"Failed to acquire bot lock - another instance may be running (PID: {existing_pid})")
            return False
    
    def release(self):
        """Release the lock file"""
        if self.lock_file and self.acquired:
            try:
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
                self.lock_file.close()
                
                if os.path.exists(self.lock_file_path):
                    os.unlink(self.lock_file_path)
                
                self.acquired = False
                logger.info("Bot instance lock released")
            except (IOError, OSError) as e:
                logger.error(f"Error releasing bot lock: {e}")
    
    def __enter__(self):
        if not self.acquire():
            raise RuntimeError("Failed to acquire bot instance lock - another instance may be running")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


class TradingBotOrchestrator:
    SHUTDOWN_TOTAL_TIMEOUT = 30
    SHUTDOWN_PHASE_TIMEOUT = 8
    TASK_CANCEL_TIMEOUT = 5
    
    def __init__(self):
        self.instance_lock = SingleInstanceLock()
        if not self.instance_lock.acquire():
            raise RuntimeError(
                "Cannot start bot: Another instance is already running. "
                "Check for zombie processes or wait for the other instance to terminate."
            )
        self.config = Config()
        self.config_valid = False
        
        sentry_dsn = os.getenv('SENTRY_DSN')
        environment = os.getenv('ENVIRONMENT', 'production')
        initialize_sentry(sentry_dsn, environment)
        logger.info(f"Sentry error tracking initialized (environment: {environment})")
        
        # Refresh secrets dari environment (penting untuk Koyeb deployment)
        # Secrets mungkin di-inject setelah module import
        logger.info("Refreshing secrets dari environment...")
        refresh_result = Config._refresh_secrets()
        logger.info(f"Secrets refresh: token_set={refresh_result['token_set']}, users={refresh_result['users_count']}")
        
        logger.info("Validating configuration...")
        try:
            self.config.validate()
            logger.info("✅ Configuration validated successfully")
            self.config_valid = True
        except ConfigError as e:
            logger.warning(f"⚠️ Configuration validation issues: {e}")
            logger.warning("Bot will start in limited mode - health check will be available")
            logger.warning("Set missing environment variables and restart to enable full functionality")
            self.config_valid = False
            
            sentry = get_sentry_manager()
            sentry.capture_exception(e, {'context': 'Configuration validation'})
        
        self.running = False
        self._shutdown_in_progress = False
        self._shutdown_count = 0
        self._shutdown_lock = asyncio.Lock()
        self.shutdown_event = asyncio.Event()
        self.health_server = None
        
        self._task_registry: Dict[str, TaskInfo] = {}
        self._task_registry_lock = asyncio.Lock()
        self._completed_tasks: Set[str] = set()
        self._task_failure_counts: Dict[str, int] = {}
        self._task_restart_callbacks: Dict[str, Callable] = {}
        
        self.db_manager = DatabaseManager(
            db_path=self.config.DATABASE_PATH,
            database_url=self.config.DATABASE_URL
        )
        logger.info("Database initialized")
        
        # Reset historical data jika RESET_HISTORICAL_DATA_ON_START=true (untuk Koyeb deployment)
        if self.config.RESET_HISTORICAL_DATA_ON_START:
            logger.info("🔄 RESET_HISTORICAL_DATA_ON_START aktif - menghapus data history...")
            clear_result = self.db_manager.clear_historical_data()
            logger.info(f"Reset result: {clear_result['message']}")
        
        self.backup_manager = DatabaseBackupManager(
            db_path=self.config.DATABASE_PATH,
            backup_dir='backups',
            max_backups=7
        )
        if self.config.DATABASE_URL:
            self.backup_manager.configure_postgres(self.config.DATABASE_URL)
        logger.info("Database backup manager initialized")
        
        self.task_scheduler = TaskScheduler(self.config)
        logger.info("Task scheduler initialized (available in all modes)")
        
        if not self.config_valid:
            logger.warning("Skipping full component initialization - running in limited mode")
            self.error_handler = None
            self.user_manager = None
            self.market_data = None
            self.strategy = None
            self.indicator_engine = None
            self.signal_quality_tracker = None
            self.market_regime_detector = None
            self.signal_rules = None
            self.auto_optimizer = None
            self.risk_manager = None
            self.chart_generator = None
            self.alert_system = None
            self.position_tracker = None
            self.telegram_bot = None
            logger.info("Limited mode: Only database, task scheduler, and health server will be initialized")
            return
        
        logger.info("Initializing Trading Bot components...")
        
        self.error_handler = ErrorHandler(self.config)
        logger.info("Error handler initialized")
        
        self.user_manager = UserManager(self.config)
        logger.info("User manager initialized")
        
        self.market_data = MarketDataClient(self.config)
        logger.info("Market data client initialized")
        
        self.strategy = TradingStrategy(self.config)
        logger.info("Trading strategy initialized")
        
        self.indicator_engine = IndicatorEngine(self.config)
        logger.info("Indicator engine initialized")
        
        self.signal_quality_tracker = SignalQualityTracker(self.db_manager, self.config)
        logger.info("Signal quality tracker initialized")
        
        self.market_regime_detector = MarketRegimeDetector(self.config, self.indicator_engine)
        logger.info("Market regime detector initialized")
        
        self.signal_rules = AggressiveSignalRules(self.config, self.indicator_engine)
        logger.info("Aggressive signal rules initialized")
        
        self.auto_optimizer = AutoOptimizer(
            signal_quality_tracker=self.signal_quality_tracker,
            config=self.config
        )
        logger.info("Auto optimizer initialized")
        
        self.risk_manager = RiskManager(self.config, self.db_manager)
        logger.info("Risk manager initialized")
        
        self.chart_generator = ChartGenerator(self.config)
        logger.info("Chart generator initialized")
        
        self.alert_system = AlertSystem(self.config, self.db_manager)
        logger.info("Alert system initialized")
        
        self.signal_session_manager = SignalSessionManager()
        logger.info("Signal session manager initialized")
        
        self.position_tracker = PositionTracker(
            self.config, 
            self.db_manager, 
            self.risk_manager,
            self.alert_system,
            self.user_manager,
            self.chart_generator,
            self.market_data,
            signal_session_manager=self.signal_session_manager,
            signal_quality_tracker=self.signal_quality_tracker
        )
        logger.info("Position tracker initialized")
        
        self.telegram_bot = TradingBot(
            self.config,
            self.db_manager,
            self.strategy,
            self.risk_manager,
            self.market_data,
            self.position_tracker,
            self.chart_generator,
            self.alert_system,
            self.error_handler,
            self.user_manager,
            self.signal_session_manager,
            self.task_scheduler,
            market_regime_detector=self.market_regime_detector,
            signal_rules=self.signal_rules,
            signal_quality_tracker=self.signal_quality_tracker,
            auto_optimizer=self.auto_optimizer
        )
        logger.info("Telegram bot initialized")
        
        logger.info("All components initialized successfully")
    
    @property
    def shutdown_in_progress(self) -> bool:
        return self._shutdown_in_progress
    
    async def register_task(
        self,
        name: str,
        task: asyncio.Task,
        priority: TaskPriority = TaskPriority.NORMAL,
        critical: bool = False,
        cancel_timeout: float = 5.0,
        max_restarts: int = 3
    ) -> TaskInfo:
        async with self._task_registry_lock:
            if name in self._task_registry:
                old_task = self._task_registry[name]
                if not old_task.is_done():
                    logger.warning(f"Replacing existing running task: {name}")
                    old_task.task.cancel()
            
            task_info = TaskInfo(
                name=name,
                task=task,
                priority=priority,
                critical=critical,
                cancel_timeout=cancel_timeout,
                max_restarts=max_restarts
            )
            self._task_registry[name] = task_info
            logger.debug(f"Task registered: {name} (priority={priority.name}, critical={critical}, max_restarts={max_restarts})")
            return task_info
    
    async def unregister_task(self, name: str, cancel: bool = False) -> bool:
        async with self._task_registry_lock:
            if name not in self._task_registry:
                logger.warning(f"Task not found for unregister: {name}")
                return False
            
            task_info = self._task_registry[name]
            
            if cancel and not task_info.is_done():
                task_info.task.cancel()
                try:
                    await asyncio.wait_for(
                        asyncio.shield(task_info.task),
                        timeout=task_info.cancel_timeout
                    )
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            
            task_info.status = TaskStatus.COMPLETED if task_info.is_done() else TaskStatus.CANCELLED
            self._completed_tasks.add(name)
            del self._task_registry[name]
            logger.debug(f"Task unregistered: {name} (status={task_info.status.value})")
            return True
    
    def get_task_status(self, name: str) -> Optional[TaskStatus]:
        if name in self._task_registry:
            task_info = self._task_registry[name]
            if task_info.is_done():
                if task_info.is_cancelled():
                    return TaskStatus.CANCELLED
                elif task_info.task.exception():
                    return TaskStatus.FAILED
                return TaskStatus.COMPLETED
            return TaskStatus.RUNNING
        elif name in self._completed_tasks:
            return TaskStatus.COMPLETED
        return None
    
    def get_registered_tasks(self) -> Dict[str, Dict[str, Any]]:
        result = {}
        for name, info in self._task_registry.items():
            result[name] = {
                'priority': info.priority.name,
                'critical': info.critical,
                'created_at': info.created_at.isoformat(),
                'is_done': info.is_done(),
                'is_cancelled': info.is_cancelled(),
                'status': status.value if (status := self.get_task_status(name)) else 'unknown'
            }
        return result
    
    async def _cancel_task_with_shield(
        self,
        task_info: TaskInfo,
        timeout: float
    ) -> bool:
        if task_info.is_done():
            return True
        
        name = task_info.name
        
        if task_info.critical:
            logger.info(f"[SHUTDOWN] Shielding critical task: {name}")
            try:
                await asyncio.wait_for(
                    asyncio.shield(task_info.task),
                    timeout=timeout
                )
                logger.info(f"[SHUTDOWN] Critical task completed: {name}")
                return True
            except asyncio.TimeoutError:
                logger.warning(f"[SHUTDOWN] Critical task {name} timeout after {timeout}s, forcing cancel")
                task_info.task.cancel()
                try:
                    await asyncio.wait_for(task_info.task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                return False
            except asyncio.CancelledError:
                logger.info(f"[SHUTDOWN] Critical task {name} cancelled")
                return True
        else:
            logger.debug(f"[SHUTDOWN] Cancelling task: {name}")
            task_info.task.cancel()
            try:
                await asyncio.wait_for(task_info.task, timeout=timeout)
                return True
            except (asyncio.CancelledError, asyncio.TimeoutError):
                return False
    
    async def _cancel_all_registered_tasks(self, timeout: float = 10.0) -> int:
        async with self._task_registry_lock:
            if not self._task_registry:
                return 0
            
            sorted_tasks = sorted(
                self._task_registry.values(),
                key=lambda t: (t.critical, t.priority.value),
                reverse=True
            )
            
            cancelled_count = 0
            logger.info(f"[SHUTDOWN] Cancelling {len(sorted_tasks)} registered tasks...")
            
            for task_info in sorted_tasks:
                if not task_info.is_done():
                    per_task_timeout = min(timeout / len(sorted_tasks), task_info.cancel_timeout)
                    success = await self._cancel_task_with_shield(task_info, per_task_timeout)
                    if success:
                        cancelled_count += 1
                        task_info.status = TaskStatus.COMPLETED
                    else:
                        task_info.status = TaskStatus.CANCELLED
            
            return cancelled_count
    
    async def _restart_task(self, task_name: str, task_info: TaskInfo) -> bool:
        """Restart a stuck or failed task
        
        Args:
            task_name: Name of the task to restart
            task_info: TaskInfo object containing task details
            
        Returns:
            True if restart was successful, False otherwise
        """
        if not task_info.can_restart():
            logger.error(f"❌ Task {task_name} exceeded max restarts ({task_info.max_restarts})")
            self._task_failure_counts[task_name] = self._task_failure_counts.get(task_name, 0) + 1
            return False
        
        try:
            if not task_info.is_done():
                task_info.task.cancel()
                try:
                    await asyncio.wait_for(task_info.task, timeout=3.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            
            task_info.restart_count += 1
            task_info.last_restart_at = datetime.now()
            
            if task_name == "market_data_websocket" and self.market_data:
                logger.warning(f"🔄 Restarting market_data_websocket (attempt {task_info.restart_count}/{task_info.max_restarts})")
                self.market_data.reconnect_attempts = 0
                new_task = asyncio.create_task(self.market_data.connect_websocket())
                task_info.task = new_task
                task_info.created_at = datetime.now()
                task_info.status = TaskStatus.RUNNING
                logger.info(f"✅ market_data_websocket restarted successfully")
                return True
            
            elif task_name == "position_tracker" and self.position_tracker:
                logger.warning(f"🔄 Restarting position_tracker (attempt {task_info.restart_count}/{task_info.max_restarts})")
                new_task = asyncio.create_task(
                    self.position_tracker.monitor_positions(self.market_data)
                )
                task_info.task = new_task
                task_info.created_at = datetime.now()
                task_info.status = TaskStatus.RUNNING
                logger.info(f"✅ position_tracker restarted successfully")
                return True
            
            elif task_name == "telegram_bot" and self.telegram_bot:
                logger.warning(f"🔄 Restarting telegram_bot (attempt {task_info.restart_count}/{task_info.max_restarts})")
                new_task = asyncio.create_task(self.telegram_bot.run())
                task_info.task = new_task
                task_info.created_at = datetime.now()
                task_info.status = TaskStatus.RUNNING
                logger.info(f"✅ telegram_bot restarted successfully")
                return True
            
            elif task_name == "health_check_long_running":
                logger.warning(f"🔄 Restarting health_check_long_running (attempt {task_info.restart_count}/{task_info.max_restarts})")
                new_task = asyncio.create_task(self._health_check_long_running())
                task_info.task = new_task
                task_info.created_at = datetime.now()
                task_info.status = TaskStatus.RUNNING
                logger.info(f"✅ health_check_long_running restarted successfully")
                return True
            
            elif task_name == "self_ping_keep_alive":
                logger.warning(f"🔄 Restarting self_ping_keep_alive (attempt {task_info.restart_count}/{task_info.max_restarts})")
                new_task = asyncio.create_task(self._self_ping_keep_alive())
                task_info.task = new_task
                task_info.created_at = datetime.now()
                task_info.status = TaskStatus.RUNNING
                logger.info(f"✅ self_ping_keep_alive restarted successfully")
                return True
            
            else:
                logger.warning(f"⚠️ No restart handler for task: {task_name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to restart task {task_name}: {e}")
            self._task_failure_counts[task_name] = self._task_failure_counts.get(task_name, 0) + 1
            return False
    
    def _convert_timestamp_to_string(self, ts: Any, pd: Any, np: Any) -> str:
        """Convert various timestamp formats to ISO string format.
        
        Handles numpy datetime64, pandas Timestamp, datetime, and other formats.
        
        Args:
            ts: Timestamp value (can be numpy datetime64, pandas Timestamp, datetime, etc)
            pd: pandas module reference
            np: numpy module reference
            
        Returns:
            str: ISO formatted timestamp string
        """
        ts_str = str(ts)
        try:
            if ts is None:
                return ts_str
            
            if isinstance(ts, (np.datetime64, pd.Timestamp)):
                ts_converted = pd.to_datetime(ts)
                if pd.isna(ts_converted):
                    return str(ts)
                return ts_converted.strftime('%Y-%m-%dT%H:%M:%S')
            
            if isinstance(ts, datetime):
                return ts.isoformat()
            
            if hasattr(ts, 'isoformat'):
                iso_method = getattr(ts, 'isoformat', None)
                if callable(iso_method):
                    return str(iso_method())
            
            if hasattr(ts, 'strftime'):
                strftime_method = getattr(ts, 'strftime', None)
                if callable(strftime_method):
                    return str(strftime_method('%Y-%m-%dT%H:%M:%S'))
            
            return str(ts)
        except Exception:
            return str(ts)
    
    async def _check_market_data_health(self) -> bool:
        """Check if market_data_websocket is healthy and reconnecting properly
        
        Returns:
            True if healthy or recovering, False if stuck
        """
        if not self.market_data:
            return True
        
        try:
            reconnect_attempts = self.market_data.reconnect_attempts
            max_attempts = self.market_data.max_reconnect_attempts
            is_connected = self.market_data.is_connected()
            use_simulator = self.market_data.use_simulator
            
            if is_connected or use_simulator:
                return True
            
            if reconnect_attempts >= max_attempts:
                logger.warning(f"⚠️ market_data reached max reconnect attempts ({reconnect_attempts}/{max_attempts})")
                
                if "market_data_websocket" in self._task_registry:
                    task_info = self._task_registry["market_data_websocket"]
                    if not task_info.is_done() and task_info.can_restart():
                        logger.warning("🔄 Forcing market_data_websocket restart due to max reconnect attempts")
                        return await self._restart_task("market_data_websocket", task_info)
                
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking market data health: {e}")
            return True
    
    async def _self_ping_keep_alive(self):
        """Self-ping task to keep Koyeb service awake on free tier.
        
        Koyeb free tier puts services to sleep after 5 minutes of inactivity.
        This task sends HTTP requests to the health endpoint every SELF_PING_INTERVAL
        seconds to keep the service active 24/7.
        """
        if not self.config.SELF_PING_ENABLED:
            logger.info("🔇 Self-ping keep-alive is DISABLED (SELF_PING_ENABLED=false)")
            return
        
        logger.info(f"🏓 Starting self-ping keep-alive task (interval: {self.config.SELF_PING_INTERVAL}s)")
        
        ping_count = 0
        success_count = 0
        fail_count = 0
        
        self_url = None
        koyeb_public_domain = os.getenv('KOYEB_PUBLIC_DOMAIN', '')
        replit_dev_domain = os.getenv('REPLIT_DEV_DOMAIN', '')
        
        if koyeb_public_domain:
            self_url = f"https://{koyeb_public_domain.strip()}/health"
            logger.info(f"🏓 Self-ping URL (Koyeb): {self_url}")
        elif replit_dev_domain:
            self_url = f"https://{replit_dev_domain.strip()}/health"
            logger.info(f"🏓 Self-ping URL (Replit): {self_url}")
        else:
            self_url = f"http://localhost:{self.config.HEALTH_CHECK_PORT}/health"
            logger.info(f"🏓 Self-ping URL (localhost): {self_url}")
        
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                while self.running and not self._shutdown_in_progress:
                    try:
                        await asyncio.sleep(self.config.SELF_PING_INTERVAL)
                        
                        if not self.running or self._shutdown_in_progress:
                            break
                        
                        ping_count += 1
                        
                        try:
                            async with session.get(
                                self_url,
                                timeout=aiohttp.ClientTimeout(total=self.config.SELF_PING_TIMEOUT),
                                ssl=False
                            ) as response:
                                if response.status == 200:
                                    success_count += 1
                                    if ping_count % 15 == 0:
                                        logger.info(
                                            f"🏓 Self-ping #{ping_count} OK "
                                            f"(success: {success_count}, fail: {fail_count})"
                                        )
                                    else:
                                        logger.debug(f"🏓 Self-ping #{ping_count} OK (status: {response.status})")
                                else:
                                    fail_count += 1
                                    logger.warning(f"⚠️ Self-ping #{ping_count} returned status {response.status}")
                        
                        except asyncio.TimeoutError:
                            fail_count += 1
                            logger.warning(f"⚠️ Self-ping #{ping_count} timeout after {self.config.SELF_PING_TIMEOUT}s")
                        
                        except aiohttp.ClientError as e:
                            fail_count += 1
                            logger.warning(f"⚠️ Self-ping #{ping_count} failed: {type(e).__name__}: {e}")
                    
                    except asyncio.CancelledError:
                        logger.info(f"🏓 Self-ping cancelled after {ping_count} pings (success: {success_count}, fail: {fail_count})")
                        raise
                    
                    except Exception as e:
                        fail_count += 1
                        logger.error(f"❌ Self-ping error: {type(e).__name__}: {e}")
                        await asyncio.sleep(30)
            
            logger.info(f"🏓 Self-ping stopped (total: {ping_count}, success: {success_count}, fail: {fail_count})")
        
        except asyncio.CancelledError:
            logger.info("🏓 Self-ping task cancelled")
        except Exception as e:
            logger.error(f"❌ Self-ping task fatal error: {e}")
    
    def _auto_detect_webhook_url(self) -> Optional[str]:
        if self.config.WEBHOOK_URL and self.config.WEBHOOK_URL.strip():
            return None
        
        import json
        from urllib.parse import urlparse
        
        domain = None
        
        koyeb_app_name = os.getenv('KOYEB_APP_NAME')
        koyeb_service_name = os.getenv('KOYEB_SERVICE_NAME')
        koyeb_public_domain = os.getenv('KOYEB_PUBLIC_DOMAIN')
        
        if koyeb_public_domain:
            domain = koyeb_public_domain.strip()
            logger.info(f"Detected Koyeb domain from KOYEB_PUBLIC_DOMAIN: {domain}")
        elif koyeb_app_name or koyeb_service_name:
            app_name = koyeb_app_name or koyeb_service_name
            domain = f"{app_name}.koyeb.app"
            logger.info(f"Constructed Koyeb domain from app/service name: {domain}")
        
        if not domain:
            replit_domains = os.getenv('REPLIT_DOMAINS')
            replit_dev_domain = os.getenv('REPLIT_DEV_DOMAIN')
            
            if replit_domains:
                try:
                    domains_list = json.loads(replit_domains)
                    if isinstance(domains_list, list) and len(domains_list) > 0:
                        domain = str(domains_list[0]).strip()
                        logger.info(f"Detected Replit deployment domain from REPLIT_DOMAINS: {domain}")
                    else:
                        logger.warning(f"REPLIT_DOMAINS is not a valid array: {replit_domains}")
                except json.JSONDecodeError:
                    domain = replit_domains.strip().strip('[]"\'').split(',')[0].strip().strip('"\'')
                    logger.warning(f"Failed to parse REPLIT_DOMAINS as JSON, using fallback: {domain}")
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Error parsing REPLIT_DOMAINS: {e}")
            
            if not domain and replit_dev_domain:
                domain = replit_dev_domain.strip()
                logger.info(f"Detected Replit dev domain from REPLIT_DEV_DOMAIN: {domain}")
        
        if domain:
            domain = domain.strip().strip('"\'')
            
            if not domain or domain.startswith('[') or domain.startswith('{') or '"' in domain or "'" in domain or '://' in domain:
                logger.error(f"Invalid domain detected after parsing: {domain}")
                return None
            
            try:
                test_url = f"https://{domain}"
                parsed = urlparse(test_url)
                if not parsed.netloc or parsed.netloc != domain:
                    logger.error(f"Domain validation failed - invalid structure: {domain}")
                    return None
            except (ValueError, TypeError) as e:
                logger.error(f"Domain validation error: {e}")
                return None
            
            bot_token = self.config.TELEGRAM_BOT_TOKEN
            webhook_url = f"https://{domain}/bot{bot_token}"
            
            logger.info(f"✅ Auto-constructed webhook URL: {webhook_url}")
            return webhook_url
        
        logger.warning("Could not auto-detect webhook URL - no Koyeb/Replit domain found")
        logger.warning("Set WEBHOOK_URL environment variable manually or KOYEB_PUBLIC_DOMAIN")
        return None
        
    async def start_health_server(self):
        try:
            import socket
            
            def is_port_in_use(port: int) -> bool:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    return s.connect_ex(('localhost', port)) == 0
            
            port = self.config.HEALTH_CHECK_PORT
            max_port_attempts = 5
            
            for attempt in range(max_port_attempts):
                if is_port_in_use(port):
                    logger.warning(f"Port {port} is already in use (attempt {attempt + 1}/{max_port_attempts})")
                    port += 1
                    logger.info(f"Trying alternative port: {port}")
                else:
                    logger.info(f"✅ Port {port} is available")
                    self.config.HEALTH_CHECK_PORT = port
                    break
            else:
                logger.error(f"Could not find available port after {max_port_attempts} attempts")
                raise Exception(f"All ports from {self.config.HEALTH_CHECK_PORT} to {port} are in use")
            
            async def health_check(request):
                missing_config = []
                if not self.config.TELEGRAM_BOT_TOKEN:
                    missing_config.append('TELEGRAM_BOT_TOKEN')
                if not self.config.AUTHORIZED_USER_IDS:
                    missing_config.append('AUTHORIZED_USER_IDS')
                
                market_status = 'not_initialized' if not self.config_valid else (self.market_data.get_status() if self.market_data else 'not_initialized')
                
                db_status = 'unknown'
                position_count = 0
                try:
                    session = self.db_manager.get_session()
                    if session:
                        result = session.execute(text('SELECT 1'))
                        result.fetchone()
                        
                        try:
                            count_result = session.execute(text("SELECT COUNT(*) FROM positions WHERE status = 'open'"))
                            position_count = count_result.scalar() or 0
                        except Exception:
                            position_count = 0
                        
                        session.close()
                        db_status = 'connected'
                    else:
                        db_status = 'error: session is None'
                except Exception as e:
                    db_status = f'error: {str(e)[:50]}'
                    logger.error(f"Database health check failed: {e}")
                
                memory_status = self.config.check_memory_status()
                
                cache_stats = {}
                chart_stats = {}
                if self.config_valid and self.telegram_bot:
                    try:
                        cache_stats = self.telegram_bot.get_cache_stats()
                    except Exception:
                        cache_stats = {'error': 'unavailable'}
                
                if self.config_valid and self.chart_generator:
                    try:
                        chart_stats = self.chart_generator.get_stats()
                    except Exception:
                        chart_stats = {'error': 'unavailable'}
                
                mode = 'full' if self.config_valid else 'limited'
                
                is_degraded = self.config.should_degrade_gracefully()
                if is_degraded:
                    mode = 'degraded'
                
                overall_status = 'healthy' if self.config_valid and self.running and not is_degraded else 'degraded' if is_degraded else 'limited' if not self.config_valid else 'stopped'
                
                task_registry_info = self.get_registered_tasks()
                
                health_status = {
                    'status': overall_status,
                    'mode': mode,
                    'config_valid': self.config_valid,
                    'missing_config': missing_config,
                    'market_data': market_status,
                    'telegram_bot': 'running' if self.config_valid and self.telegram_bot and self.telegram_bot.app else 'not_initialized',
                    'scheduler': 'running' if self.config_valid and self.task_scheduler and self.task_scheduler.running else 'not_initialized',
                    'database': db_status,
                    'open_positions': position_count,
                    'webhook_mode': self.config.TELEGRAM_WEBHOOK_MODE if self.config_valid else False,
                    'memory': memory_status,
                    'cache_stats': cache_stats,
                    'chart_stats': chart_stats,
                    'registered_tasks': len(task_registry_info),
                    'task_details': task_registry_info,
                    'shutdown_in_progress': self._shutdown_in_progress,
                    'free_tier_mode': self.config.FREE_TIER_MODE,
                    'message': 'Bot running in degraded mode - memory critical' if is_degraded else 'Bot running in limited mode - set missing environment variables to enable full functionality' if not self.config_valid else 'Bot running normally'
                }
                
                status_code = 200 if self.config_valid and self.running else 503
                
                return web.json_response(health_status, status=status_code)
            
            async def telegram_webhook(request):
                if not self.config.TELEGRAM_WEBHOOK_MODE:
                    logger.warning("⚠️ Webhook endpoint called but webhook mode is disabled")
                    return web.json_response({'error': 'Webhook mode is disabled'}, status=400)
                
                if not self.telegram_bot or not self.telegram_bot.app:
                    logger.error("❌ Webhook called but telegram bot not initialized (running in limited mode?)")
                    logger.error("Check if TELEGRAM_BOT_TOKEN and AUTHORIZED_USER_IDS are set in environment variables")
                    return web.json_response({'error': 'Bot not initialized'}, status=503)
                
                try:
                    update_data = await request.json()
                    update_id = update_data.get('update_id', 'unknown')
                    message_text = update_data.get('message', {}).get('text', 'no text')
                    user_id = update_data.get('message', {}).get('from', {}).get('id', 'unknown')
                    
                    logger.info(f"📨 Webhook received: update_id={update_id}, user={user_id}, message='{message_text}'")
                    
                    await self.telegram_bot.process_update(update_data)
                    
                    logger.info(f"✅ Webhook processed successfully: update_id={update_id}")
                    return web.json_response({'ok': True})
                    
                except Exception as e:
                    logger.error(f"❌ Error processing webhook request: {e}")
                    logger.error(f"Request path: {request.path}, Method: {request.method}")
                    if self.error_handler:
                        self.error_handler.log_exception(e, "webhook_endpoint")
                    return web.json_response({'error': str(e)}, status=500)
            
            async def dashboard_page(request):
                """Serve dashboard web app HTML"""
                try:
                    import os
                    html_path = os.path.join('webapp', 'templates', 'dashboard.html')
                    if os.path.exists(html_path):
                        with open(html_path, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                        return web.Response(text=html_content, content_type='text/html')
                    else:
                        return web.Response(text='Dashboard not found', status=404)
                except Exception as e:
                    logger.error(f"Error serving dashboard: {e}")
                    return web.Response(text='Error loading dashboard', status=500)
            
            async def static_files(request):
                """Serve static files (CSS, JS)"""
                try:
                    import os
                    filename = request.match_info.get('filename', '')
                    safe_filename = os.path.basename(filename)
                    
                    file_path = os.path.join('webapp', 'static', safe_filename)
                    if os.path.exists(file_path) and os.path.isfile(file_path):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        content_type = 'text/plain'
                        if safe_filename.endswith('.css'):
                            content_type = 'text/css'
                        elif safe_filename.endswith('.js'):
                            content_type = 'application/javascript'
                        
                        return web.Response(
                            text=content, 
                            content_type=content_type,
                            headers={'Cache-Control': 'no-cache'}
                        )
                    else:
                        return web.Response(text='File not found', status=404)
                except Exception as e:
                    logger.error(f"Error serving static file: {e}")
                    return web.Response(text='Error loading file', status=500)
            
            async def api_dashboard(request):
                """API endpoint for real-time dashboard data with per-user filtering"""
                try:
                    import pytz
                    
                    wib = pytz.timezone('Asia/Jakarta')
                    now = datetime.now(wib)
                    
                    user_id_str = request.query.get('user_id', None)
                    user_id = None
                    user_mode = 'guest'
                    is_authorized = False
                    
                    if user_id_str:
                        try:
                            user_id = int(user_id_str)
                            if user_id in self.config.AUTHORIZED_USER_IDS:
                                is_authorized = True
                                user_mode = 'authorized'
                                logger.debug(f"Dashboard accessed by authorized user: {user_id}")
                            else:
                                user_mode = 'limited'
                                logger.debug(f"Dashboard accessed by non-authorized user: {user_id}")
                        except (ValueError, TypeError):
                            user_id = None
                            user_mode = 'guest'
                    
                    price_data: Dict[str, Any] = {
                        'mid': 0.0,
                        'bid': 0.0,
                        'ask': 0.0,
                        'spread': 0.0,
                        'high': 0.0,
                        'low': 0.0,
                        'change_percent': 0.0
                    }
                    
                    if self.config_valid and self.market_data:
                        try:
                            bid = self.market_data.current_bid
                            ask = self.market_data.current_ask
                            
                            m1_df = self.market_data.m1_builder.get_dataframe(limit=1440)
                            
                            if bid and ask and isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and bid > 0 and ask > 0:
                                price_data['bid'] = float(bid)
                                price_data['ask'] = float(ask)
                                price_data['mid'] = float((bid + ask) / 2)
                                price_data['spread'] = float(round((ask - bid) * 10, 1))
                            elif m1_df is not None and len(m1_df) > 0:
                                last_candle = m1_df.iloc[-1]
                                last_close = float(last_candle['close'])
                                last_open = float(last_candle['open'])
                                last_high = float(last_candle['high'])
                                last_low = float(last_candle['low'])
                                
                                if last_close > 0:
                                    price_data['mid'] = last_close
                                    price_data['bid'] = last_close - 0.05
                                    price_data['ask'] = last_close + 0.05
                                    price_data['spread'] = 1.0
                            
                            if m1_df is not None and len(m1_df) > 0:
                                high_val = m1_df['high'].max()
                                low_val = m1_df['low'].min()
                                price_data['high'] = float(high_val) if high_val is not None else 0.0
                                price_data['low'] = float(low_val) if low_val is not None else 0.0
                                
                                if len(m1_df) > 1:
                                    first_close = float(m1_df.iloc[0]['close'])
                                    last_close = float(m1_df.iloc[-1]['close'])
                                    if first_close > 0:
                                        change = ((last_close - first_close) / first_close) * 100
                                        price_data['change_percent'] = float(round(change, 2))
                        except Exception as e:
                            logger.debug(f"Error getting price data: {e}")
                    
                    last_signal = None
                    if self.config_valid and self.telegram_bot:
                        try:
                            for type_key, signal_data in self.telegram_bot.last_signal_per_type.items():
                                if signal_data and signal_data.get('timestamp'):
                                    ts = signal_data['timestamp']
                                    if last_signal is None or ts > last_signal.get('_ts', datetime.min):
                                        last_signal = {
                                            'direction': signal_data.get('signal_type', 'UNKNOWN'),
                                            'entry_price': signal_data.get('entry_price'),
                                            'sl': signal_data.get('stop_loss'),
                                            'tp': signal_data.get('take_profit'),
                                            'timestamp': ts.isoformat(),
                                            '_ts': ts
                                        }
                            if last_signal:
                                last_signal.pop('_ts', None)
                                
                            if last_signal and (last_signal.get('sl') is None or last_signal.get('tp') is None):
                                if hasattr(self, 'signal_session_manager') and self.signal_session_manager:
                                    sessions = self.signal_session_manager.get_all_active_sessions()
                                    for user_id, session in sessions.items():
                                        if session and hasattr(session, 'stop_loss') and hasattr(session, 'take_profit'):
                                            if last_signal.get('sl') is None:
                                                last_signal['sl'] = session.stop_loss
                                            if last_signal.get('tp') is None:
                                                last_signal['tp'] = session.take_profit
                                            break
                        except Exception as e:
                            logger.debug(f"Error getting last signal: {e}")
                    
                    active_position = {'active': False}
                    if self.config_valid and self.position_tracker:
                        try:
                            positions = self.position_tracker.get_active_positions()
                            if positions:
                                for pos_user_id, user_positions in positions.items():
                                    if user_id is not None and pos_user_id != user_id:
                                        continue
                                    
                                    if isinstance(user_positions, dict):
                                        for pos_id, pos in user_positions.items():
                                            if isinstance(pos, dict):
                                                entry_price = pos.get('entry_price')
                                                stop_loss = pos.get('stop_loss')
                                                take_profit = pos.get('take_profit')
                                                direction = pos.get('signal_type', 'BUY')
                                                
                                                current_pnl_pips = None
                                                distance_to_tp_pips = None
                                                distance_to_sl_pips = None
                                                unrealized_pnl_usd = 0.0
                                                
                                                current_price = price_data.get('mid')
                                                pip_value = getattr(self.config, 'XAUUSD_PIP_VALUE', 10.0)
                                                lot_size = pos.get('lot_size', getattr(self.config, 'LOT_SIZE', 0.01))
                                                
                                                if current_price and entry_price:
                                                    if direction == 'BUY':
                                                        current_pnl_pips = round((current_price - entry_price) * pip_value, 1)
                                                    else:
                                                        current_pnl_pips = round((entry_price - current_price) * pip_value, 1)
                                                    
                                                    unrealized_pnl_usd = round(current_pnl_pips * lot_size * pip_value, 2)
                                                
                                                if current_price and take_profit:
                                                    if direction == 'BUY':
                                                        distance_to_tp_pips = round((take_profit - current_price) * pip_value, 1)
                                                    else:
                                                        distance_to_tp_pips = round((current_price - take_profit) * pip_value, 1)
                                                
                                                if current_price and stop_loss:
                                                    if direction == 'BUY':
                                                        distance_to_sl_pips = round((current_price - stop_loss) * pip_value, 1)
                                                    else:
                                                        distance_to_sl_pips = round((stop_loss - current_price) * pip_value, 1)
                                                
                                                active_position = {
                                                    'active': True,
                                                    'direction': direction,
                                                    'entry_price': entry_price,
                                                    'sl': stop_loss,
                                                    'tp': take_profit,
                                                    'unrealized_pnl': unrealized_pnl_usd,
                                                    'current_pnl_pips': current_pnl_pips,
                                                    'distance_to_tp_pips': distance_to_tp_pips,
                                                    'distance_to_sl_pips': distance_to_sl_pips,
                                                    'lot_size': lot_size
                                                }
                                                break
                                    if active_position['active']:
                                        break
                        except Exception as e:
                            logger.debug(f"Error getting positions: {e}")
                    
                    regime_data = None
                    if self.config_valid and self.market_regime_detector and self.market_data:
                        try:
                            m1_df = self.market_data.m1_builder.get_dataframe(limit=100)
                            if m1_df is not None and len(m1_df) >= 50:
                                regime = self.market_regime_detector.get_regime({}, m1_df)
                                if regime:
                                    regime_data = {
                                        'trend': regime.regime_type if hasattr(regime, 'regime_type') else 'Unknown',
                                        'volatility': regime.volatility_analysis.volatility_zone if hasattr(regime, 'volatility_analysis') and hasattr(regime.volatility_analysis, 'volatility_zone') else 'Normal',
                                        'bias': regime.bias if hasattr(regime, 'bias') else 'NEUTRAL',
                                        'confidence': regime.confidence if hasattr(regime, 'confidence') else 0
                                    }
                        except Exception as e:
                            logger.debug(f"Error getting regime: {e}")
                    
                    stats: Dict[str, Any] = {
                        'win_rate': 0.0,
                        'total_pnl': 0.0,
                        'signals_today': 0,
                        'total_trades': 0
                    }
                    
                    if self.config_valid and self.db_manager:
                        try:
                            session = self.db_manager.get_session()
                            if session:
                                today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                                
                                if user_id is not None and is_authorized:
                                    result = session.execute(text(
                                        "SELECT COUNT(*) as total, "
                                        "SUM(CASE WHEN actual_pl > 0 THEN 1 ELSE 0 END) as wins, "
                                        "COALESCE(SUM(actual_pl), 0) as total_pnl "
                                        "FROM trades WHERE status = 'CLOSED' AND user_id = :user_id"
                                    ), {'user_id': user_id})
                                else:
                                    result = session.execute(text(
                                        "SELECT COUNT(*) as total, "
                                        "SUM(CASE WHEN actual_pl > 0 THEN 1 ELSE 0 END) as wins, "
                                        "COALESCE(SUM(actual_pl), 0) as total_pnl "
                                        "FROM trades WHERE status = 'CLOSED'"
                                    ))
                                row = result.fetchone()
                                if row:
                                    total = int(row[0] or 0)
                                    wins = int(row[1] or 0)
                                    stats['total_trades'] = total
                                    stats['total_pnl'] = float(row[2] or 0)
                                    stats['win_rate'] = float(wins / total * 100) if total > 0 else 0.0
                                
                                if user_id is not None and is_authorized:
                                    today_result = session.execute(text(
                                        "SELECT COUNT(*) FROM trades WHERE signal_time >= :today AND user_id = :user_id"
                                    ), {'today': today_start, 'user_id': user_id})
                                else:
                                    today_result = session.execute(text(
                                        "SELECT COUNT(*) FROM trades WHERE signal_time >= :today"
                                    ), {'today': today_start})
                                signals_today = today_result.scalar()
                                stats['signals_today'] = int(signals_today or 0)
                                
                                session.close()
                        except Exception as e:
                            logger.debug(f"Error getting stats: {e}")
                    
                    response_data = {
                        'price': price_data,
                        'last_signal': last_signal,
                        'active_position': active_position,
                        'regime': regime_data,
                        'stats': stats,
                        'timestamp': now.isoformat(),
                        'connected': self.config_valid and self.market_data is not None and self.market_data.is_connected(),
                        'user_mode': user_mode,
                        'user_id': user_id
                    }
                    
                    return web.json_response(response_data, headers={
                        'Cache-Control': 'no-cache, no-store, must-revalidate',
                        'Pragma': 'no-cache',
                        'Expires': '0',
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                        'Access-Control-Allow-Headers': 'Content-Type'
                    })
                    
                except Exception as e:
                    logger.error(f"Error in API dashboard: {e}")
                    return web.json_response({'error': str(e)}, status=500)
            
            async def api_candles(request):
                """API endpoint for candle data with per-user position filtering"""
                try:
                    timeframe = request.query.get('timeframe', 'M1').upper()
                    if timeframe not in ('M1', 'M5', 'H1'):
                        timeframe = 'M1'
                    
                    try:
                        limit = int(request.query.get('limit', 50))
                        limit = max(1, min(limit, 200))
                    except (ValueError, TypeError):
                        limit = 50
                    
                    user_id_str = request.query.get('user_id', None)
                    request_user_id = None
                    is_authorized = False
                    
                    if user_id_str:
                        try:
                            request_user_id = int(user_id_str)
                            if request_user_id in self.config.AUTHORIZED_USER_IDS:
                                is_authorized = True
                        except (ValueError, TypeError):
                            request_user_id = None
                    
                    candles_data = []
                    current_price = None
                    
                    if self.config_valid and self.market_data:
                        if timeframe == 'M1':
                            builder = self.market_data.m1_builder
                        elif timeframe == 'M5':
                            builder = self.market_data.m5_builder
                        else:
                            builder = self.market_data.h1_builder
                        
                        df = builder.get_dataframe(limit=limit)
                        
                        if df is not None and len(df) > 0:
                            df_reset = df.reset_index()
                            import pandas as pd
                            import numpy as np
                            for _, row in df_reset.iterrows():
                                ts = row['timestamp']
                                ts_str = self._convert_timestamp_to_string(ts, pd, np)
                                candle = {
                                    'timestamp': ts_str,
                                    'open': float(row['open']),
                                    'high': float(row['high']),
                                    'low': float(row['low']),
                                    'close': float(row['close']),
                                    'volume': int(row['volume']) if 'volume' in row else 0
                                }
                                candles_data.append(candle)
                        
                        current_price = await self.market_data.get_current_price()
                    
                    active_position = None
                    if self.config_valid and self.position_tracker:
                        try:
                            positions = self.position_tracker.get_active_positions()
                            if positions:
                                if request_user_id is not None and is_authorized:
                                    user_positions = positions.get(request_user_id, {})
                                    if isinstance(user_positions, dict):
                                        for pos_id, pos in user_positions.items():
                                            if isinstance(pos, dict):
                                                active_position = {
                                                    'entry_price': pos.get('entry_price'),
                                                    'stop_loss': pos.get('stop_loss'),
                                                    'take_profit': pos.get('take_profit'),
                                                    'trailing_sl': pos.get('trailing_sl'),
                                                    'direction': pos.get('signal_type', 'BUY')
                                                }
                                                break
                                else:
                                    for user_id, user_positions in positions.items():
                                        if isinstance(user_positions, dict):
                                            for pos_id, pos in user_positions.items():
                                                if isinstance(pos, dict):
                                                    active_position = {
                                                        'entry_price': pos.get('entry_price'),
                                                        'stop_loss': pos.get('stop_loss'),
                                                        'take_profit': pos.get('take_profit'),
                                                        'trailing_sl': pos.get('trailing_sl'),
                                                        'direction': pos.get('signal_type', 'BUY')
                                                    }
                                                    break
                                        if active_position:
                                            break
                        except Exception as e:
                            logger.debug(f"Error getting positions for candles API: {e}")
                    
                    response_data = {
                        'candles': candles_data,
                        'current_price': current_price,
                        'active_position': active_position
                    }
                    
                    return web.json_response(response_data, headers={
                        'Cache-Control': 'no-cache, no-store, must-revalidate',
                        'Pragma': 'no-cache',
                        'Expires': '0',
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                        'Access-Control-Allow-Headers': 'Content-Type'
                    })
                    
                except Exception as e:
                    logger.error(f"Error in API candles: {e}")
                    return web.json_response({'error': str(e)}, status=500)
            
            async def api_trade_history(request):
                """API endpoint for trade history with pagination and per-user filtering"""
                try:
                    try:
                        page = int(request.query.get('page', 1))
                        page = max(1, page)
                    except (ValueError, TypeError):
                        page = 1
                    
                    try:
                        limit = int(request.query.get('limit', 20))
                        limit = max(1, min(limit, 50))
                    except (ValueError, TypeError):
                        limit = 20
                    
                    status_filter = request.query.get('status', 'all').lower()
                    
                    user_id_str = request.query.get('user_id', None)
                    user_id = None
                    is_authorized = False
                    
                    if user_id_str:
                        try:
                            user_id = int(user_id_str)
                            if user_id in self.config.AUTHORIZED_USER_IDS:
                                is_authorized = True
                        except (ValueError, TypeError):
                            user_id = None
                    
                    trades_list = []
                    total_trades = 0
                    
                    if self.config_valid and self.db_manager:
                        try:
                            session = self.db_manager.get_session()
                            if session:
                                offset = (page - 1) * limit
                                
                                if status_filter == 'closed':
                                    status_condition = "status = 'CLOSED'"
                                elif status_filter == 'open':
                                    status_condition = "status = 'OPEN'"
                                else:
                                    status_condition = "1=1"
                                
                                if user_id is not None and is_authorized:
                                    count_result = session.execute(text(
                                        f"SELECT COUNT(*) FROM trades WHERE {status_condition} AND user_id = :user_id"
                                    ), {'user_id': user_id})
                                else:
                                    count_result = session.execute(text(
                                        f"SELECT COUNT(*) FROM trades WHERE {status_condition}"
                                    ))
                                total_trades = count_result.scalar() or 0
                                
                                if user_id is not None and is_authorized:
                                    result = session.execute(text(
                                        f"SELECT id, user_id, ticker, signal_type, entry_price, stop_loss, "
                                        f"take_profit, exit_price, status, signal_time, close_time, result, actual_pl "
                                        f"FROM trades WHERE {status_condition} AND user_id = :user_id "
                                        f"ORDER BY COALESCE(close_time, signal_time) DESC "
                                        f"LIMIT :limit OFFSET :offset"
                                    ), {'limit': limit, 'offset': offset, 'user_id': user_id})
                                else:
                                    result = session.execute(text(
                                        f"SELECT id, user_id, ticker, signal_type, entry_price, stop_loss, "
                                        f"take_profit, exit_price, status, signal_time, close_time, result, actual_pl "
                                        f"FROM trades WHERE {status_condition} "
                                        f"ORDER BY COALESCE(close_time, signal_time) DESC "
                                        f"LIMIT :limit OFFSET :offset"
                                    ), {'limit': limit, 'offset': offset})
                                
                                for row in result:
                                    trade_data = {
                                        'id': row[0],
                                        'user_id': row[1],
                                        'ticker': row[2],
                                        'signal_type': row[3],
                                        'entry_price': float(row[4]) if row[4] else None,
                                        'stop_loss': float(row[5]) if row[5] else None,
                                        'take_profit': float(row[6]) if row[6] else None,
                                        'exit_price': float(row[7]) if row[7] else None,
                                        'status': row[8],
                                        'signal_time': row[9].isoformat() if row[9] else None,
                                        'close_time': row[10].isoformat() if row[10] else None,
                                        'result': row[11],
                                        'pnl': float(row[12]) if row[12] else 0.0
                                    }
                                    trades_list.append(trade_data)
                                
                                session.close()
                        except Exception as e:
                            logger.error(f"Error fetching trade history: {e}")
                    
                    total_pages = (total_trades + limit - 1) // limit if total_trades > 0 else 1
                    
                    response_data = {
                        'trades': trades_list,
                        'pagination': {
                            'page': page,
                            'limit': limit,
                            'total_trades': total_trades,
                            'total_pages': total_pages,
                            'has_next': page < total_pages,
                            'has_prev': page > 1
                        }
                    }
                    
                    return web.json_response(response_data, headers={
                        'Cache-Control': 'no-cache, no-store, must-revalidate',
                        'Pragma': 'no-cache',
                        'Expires': '0',
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                        'Access-Control-Allow-Headers': 'Content-Type'
                    })
                    
                except Exception as e:
                    logger.error(f"Error in API trade-history: {e}")
                    return web.json_response({'error': str(e)}, status=500)
            
            async def ws_dashboard(request):
                """WebSocket endpoint for real-time dashboard updates"""
                ws = web.WebSocketResponse()
                await ws.prepare(request)
                
                user_id_str = request.query.get('user_id', None)
                user_id = None
                if user_id_str:
                    try:
                        user_id = int(user_id_str)
                        if self.config_valid and user_id not in self.config.AUTHORIZED_USER_IDS:
                            user_id = None
                    except (ValueError, TypeError):
                        user_id = None
                
                logger.info(f"🔌 WebSocket client connected to /ws/dashboard (user_id: {user_id})")
                
                try:
                    while not ws.closed and self.running:
                        try:
                            import pytz
                            wib = pytz.timezone('Asia/Jakarta')
                            now = datetime.now(wib)
                            
                            price_data: Dict[str, Any] = {
                                'mid': 0.0,
                                'bid': 0.0,
                                'ask': 0.0,
                                'spread': 0.0,
                                'high': 0.0,
                                'low': 0.0,
                                'change_percent': 0.0
                            }
                            
                            if self.config_valid and self.market_data:
                                try:
                                    bid = self.market_data.current_bid
                                    ask = self.market_data.current_ask
                                    if bid and ask and isinstance(bid, (int, float)) and isinstance(ask, (int, float)):
                                        price_data['bid'] = float(bid)
                                        price_data['ask'] = float(ask)
                                        price_data['mid'] = float((bid + ask) / 2)
                                        price_data['spread'] = float(round((ask - bid) * 10, 1))
                                    
                                    m1_df = self.market_data.m1_builder.get_dataframe(limit=1440)
                                    if m1_df is not None and len(m1_df) > 0:
                                        high_val = m1_df['high'].max()
                                        low_val = m1_df['low'].min()
                                        price_data['high'] = float(high_val) if high_val is not None else 0.0
                                        price_data['low'] = float(low_val) if low_val is not None else 0.0
                                        
                                        if len(m1_df) > 1:
                                            first_close = float(m1_df.iloc[0]['close'])
                                            last_close = float(m1_df.iloc[-1]['close'])
                                            if first_close > 0:
                                                change = ((last_close - first_close) / first_close) * 100
                                                price_data['change_percent'] = float(round(change, 2))
                                except Exception as e:
                                    logger.debug(f"WebSocket: Error getting price data: {e}")
                            
                            active_position = {'active': False}
                            if user_id is not None and self.config_valid and self.position_tracker:
                                try:
                                    positions = self.position_tracker.get_active_positions()
                                    if positions:
                                        for pos_user_id, user_positions in positions.items():
                                            if pos_user_id != user_id:
                                                continue
                                            if isinstance(user_positions, dict):
                                                for pos_id, pos in user_positions.items():
                                                    if isinstance(pos, dict):
                                                        entry_price = pos.get('entry_price')
                                                        stop_loss = pos.get('stop_loss')
                                                        take_profit = pos.get('take_profit')
                                                        direction = pos.get('signal_type', 'BUY')
                                                        
                                                        current_pnl_pips = None
                                                        distance_to_tp_pips = None
                                                        distance_to_sl_pips = None
                                                        unrealized_pnl_usd = 0.0
                                                        
                                                        current_price = price_data.get('mid')
                                                        pip_value = getattr(self.config, 'XAUUSD_PIP_VALUE', 10.0)
                                                        lot_size = pos.get('lot_size', getattr(self.config, 'LOT_SIZE', 0.01))
                                                        
                                                        if current_price and entry_price:
                                                            if direction == 'BUY':
                                                                current_pnl_pips = round((current_price - entry_price) * pip_value, 1)
                                                            else:
                                                                current_pnl_pips = round((entry_price - current_price) * pip_value, 1)
                                                            
                                                            unrealized_pnl_usd = round(current_pnl_pips * lot_size * pip_value, 2)
                                                        
                                                        if current_price and take_profit:
                                                            if direction == 'BUY':
                                                                distance_to_tp_pips = round((take_profit - current_price) * pip_value, 1)
                                                            else:
                                                                distance_to_tp_pips = round((current_price - take_profit) * pip_value, 1)
                                                        
                                                        if current_price and stop_loss:
                                                            if direction == 'BUY':
                                                                distance_to_sl_pips = round((current_price - stop_loss) * pip_value, 1)
                                                            else:
                                                                distance_to_sl_pips = round((stop_loss - current_price) * pip_value, 1)
                                                        
                                                        active_position = {
                                                            'active': True,
                                                            'direction': direction,
                                                            'entry_price': entry_price,
                                                            'sl': stop_loss,
                                                            'tp': take_profit,
                                                            'unrealized_pnl': unrealized_pnl_usd,
                                                            'current_pnl_pips': current_pnl_pips,
                                                            'distance_to_tp_pips': distance_to_tp_pips,
                                                            'distance_to_sl_pips': distance_to_sl_pips,
                                                            'lot_size': lot_size
                                                        }
                                                        break
                                            if active_position['active']:
                                                break
                                except Exception as e:
                                    logger.debug(f"WebSocket: Error getting positions: {e}")
                            
                            stats: Dict[str, Any] = {
                                'win_rate': 0.0,
                                'total_pnl': 0.0,
                                'signals_today': 0,
                                'total_trades': 0
                            }
                            
                            if self.config_valid and self.db_manager:
                                try:
                                    session = self.db_manager.get_session()
                                    if session:
                                        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                                        
                                        if user_id is not None:
                                            result = session.execute(text(
                                                "SELECT COUNT(*) as total, "
                                                "SUM(CASE WHEN actual_pl > 0 THEN 1 ELSE 0 END) as wins, "
                                                "SUM(COALESCE(actual_pl, 0)) as total_pnl "
                                                "FROM trades WHERE status = 'CLOSED' AND user_id = :user_id"
                                            ), {'user_id': user_id})
                                        else:
                                            result = session.execute(text(
                                                "SELECT COUNT(*) as total, "
                                                "SUM(CASE WHEN actual_pl > 0 THEN 1 ELSE 0 END) as wins, "
                                                "SUM(COALESCE(actual_pl, 0)) as total_pnl "
                                                "FROM trades WHERE status = 'CLOSED'"
                                            ))
                                        row = result.fetchone()
                                        if row:
                                            total = int(row[0] or 0)
                                            wins = int(row[1] or 0)
                                            stats['total_trades'] = total
                                            stats['total_pnl'] = float(row[2] or 0)
                                            stats['win_rate'] = float(wins / total * 100) if total > 0 else 0.0
                                        
                                        if user_id is not None:
                                            today_result = session.execute(text(
                                                "SELECT COUNT(*) FROM trades WHERE signal_time >= :today AND user_id = :user_id"
                                            ), {'today': today_start, 'user_id': user_id})
                                        else:
                                            today_result = session.execute(text(
                                                "SELECT COUNT(*) FROM trades WHERE signal_time >= :today"
                                            ), {'today': today_start})
                                        signals_today = today_result.scalar()
                                        stats['signals_today'] = int(signals_today or 0)
                                        
                                        session.close()
                                except Exception as e:
                                    logger.debug(f"WebSocket: Error getting stats: {e}")
                            
                            ws_data = {
                                'price': price_data,
                                'active_position': active_position,
                                'stats': stats,
                                'timestamp': now.isoformat(),
                                'connected': self.config_valid and self.market_data is not None and self.market_data.is_connected()
                            }
                            
                            await ws.send_json(ws_data)
                            await asyncio.sleep(1)
                            
                        except ConnectionResetError:
                            logger.debug("WebSocket: Client connection reset")
                            break
                        except Exception as e:
                            logger.debug(f"WebSocket: Error in data loop: {e}")
                            await asyncio.sleep(1)
                            
                except asyncio.CancelledError:
                    logger.debug("WebSocket: Connection cancelled")
                except Exception as e:
                    logger.debug(f"WebSocket error: {e}")
                finally:
                    if not ws.closed:
                        await ws.close()
                    logger.info("🔌 WebSocket client disconnected from /ws/dashboard")
                
                return ws
            
            app = web.Application()
            app.router.add_get('/health', health_check)
            app.router.add_get('/', health_check)
            app.router.add_get('/dashboard', dashboard_page)
            app.router.add_get('/static/{filename}', static_files)
            app.router.add_get('/api/dashboard', api_dashboard)
            app.router.add_get('/api/candles', api_candles)
            app.router.add_get('/api/trade-history', api_trade_history)
            app.router.add_get('/ws/dashboard', ws_dashboard)
            logger.info("Dashboard web app endpoints registered: /dashboard, /api/dashboard, /api/candles, /api/trade-history, /ws/dashboard, /static/*")
            
            webhook_path = None
            if self.config_valid and self.config.TELEGRAM_BOT_TOKEN:
                # Register both /webhook (simple) and /bot{token} (standard Telegram format)
                webhook_path = f"/bot{self.config.TELEGRAM_BOT_TOKEN}"
                app.router.add_post(webhook_path, telegram_webhook)
                app.router.add_post('/webhook', telegram_webhook)
                logger.info(f"Webhook routes registered: /webhook and {webhook_path[:20]}...")
                
                # Also register GET for webhook status check (Koyeb health check)
                async def webhook_status(request):
                    return web.json_response({
                        'ok': True,
                        'webhook_mode': self.config.TELEGRAM_WEBHOOK_MODE,
                        'is_koyeb': self.config.IS_KOYEB,
                        'bot_initialized': self.telegram_bot is not None and self.telegram_bot.app is not None
                    })
                app.router.add_get('/webhook', webhook_status)
            else:
                logger.info("Webhook route not registered - limited mode or missing bot token")
            
            runner = web.AppRunner(app)
            await runner.setup()
            
            site = web.TCPSite(runner, '0.0.0.0', self.config.HEALTH_CHECK_PORT)
            await site.start()
            
            self.health_server = runner
            logger.info(f"Health check server started on port {self.config.HEALTH_CHECK_PORT}")
            if self.config.TELEGRAM_WEBHOOK_MODE and webhook_path:
                logger.info(f"Webhook endpoint available at: http://0.0.0.0:{self.config.HEALTH_CHECK_PORT}{webhook_path}")
            elif self.config.TELEGRAM_WEBHOOK_MODE:
                logger.info("Webhook mode enabled but endpoint not available (limited mode)")
            
        except Exception as e:
            logger.error(f"Failed to start health server: {e}")
    
    async def _quick_check_market_data(self) -> dict:
        """Quick check for market data WebSocket status (2 sec timeout)"""
        try:
            async with asyncio.timeout(2):
                if not self.market_data:
                    return {'status': 'ok', 'message': 'No market data client'}
                return {
                    'status': 'ok',
                    'connected': self.market_data.is_connected(),
                    'simulator': self.market_data.use_simulator,
                    'reconnects': self.market_data.reconnect_attempts
                }
        except asyncio.TimeoutError:
            return {'status': 'timeout', 'message': 'Market data check timed out'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    async def _quick_check_database(self) -> dict:
        """Quick check for database connection pool (3 sec timeout)"""
        try:
            async with asyncio.timeout(3):
                if not self.db_manager:
                    return {'status': 'ok', 'message': 'No database manager'}
                if hasattr(self.db_manager, 'engine') and self.db_manager.engine:
                    engine = self.db_manager.engine
                    def sync_check():
                        with engine.connect() as conn:
                            conn.execute(text("SELECT 1"))
                        return True
                    
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, sync_check)
                    return {'status': 'ok', 'message': 'Database connection healthy'}
                return {'status': 'ok', 'message': 'Using SQLite'}
        except asyncio.TimeoutError:
            return {'status': 'timeout', 'message': 'Database check timed out'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    async def _quick_check_telegram(self) -> dict:
        """Quick check for Telegram bot alive (2 sec timeout)"""
        try:
            async with asyncio.timeout(2):
                if not self.telegram_bot:
                    return {'status': 'ok', 'message': 'No telegram bot'}
                if hasattr(self.telegram_bot, 'app') and self.telegram_bot.app:
                    return {'status': 'ok', 'message': 'Telegram bot running'}
                return {'status': 'warning', 'message': 'Telegram bot not initialized'}
        except asyncio.TimeoutError:
            return {'status': 'timeout', 'message': 'Telegram check timed out'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    async def _quick_check_memory(self) -> dict:
        """Quick check for memory/CPU usage (1 sec timeout)"""
        try:
            async with asyncio.timeout(1):
                mem_status = self.config.check_memory_status()
                return {'status': 'ok', **mem_status}
        except asyncio.TimeoutError:
            return {'status': 'timeout', 'message': 'Memory check timed out'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    async def _run_all_quick_checks(self) -> dict:
        """Run all quick health checks concurrently with total timeout"""
        try:
            async with asyncio.timeout(self.config.HEALTH_CHECK_LONG_TIMEOUT):
                results = await asyncio.gather(
                    self._quick_check_market_data(),
                    self._quick_check_database(),
                    self._quick_check_telegram(),
                    self._quick_check_memory(),
                    return_exceptions=True
                )
                
                check_names = ['market_data', 'database', 'telegram', 'memory']
                health_report = {}
                
                for name, result in zip(check_names, results):
                    if isinstance(result, Exception):
                        health_report[name] = {'status': 'error', 'message': str(result)}
                        logger.warning(f"⚠️ Health check {name} failed: {result}")
                    elif isinstance(result, dict):
                        if result.get('status') != 'ok':
                            health_report[name] = result
                            logger.warning(f"⚠️ Health check {name}: {result.get('message', 'Unknown issue')}")
                        else:
                            health_report[name] = result
                    else:
                        health_report[name] = {'status': 'error', 'message': f'Unexpected result type: {type(result)}'}
                
                return health_report
        except asyncio.TimeoutError:
            logger.warning(f"⚠️ Quick health checks exceeded total timeout ({self.config.HEALTH_CHECK_LONG_TIMEOUT}s)")
            return {'status': 'timeout', 'message': 'Quick checks timed out'}
        except Exception as e:
            logger.warning(f"⚠️ Error running quick health checks: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def _health_check_long_running(self):
        """Monitor for stuck/long-running tasks with automatic restart capability
        
        Optimized for quick checks with short timeouts to never block main loop.
        Uses config-based intervals and runs checks concurrently.
        """
        LONG_RUNNING_THRESHOLD_NORMAL = 300
        LONG_RUNNING_THRESHOLD_CRITICAL = 600
        WARNING_THRESHOLD = 120
        
        while self.running and not self._shutdown_in_progress:
            try:
                await asyncio.sleep(self.config.HEALTH_CHECK_INTERVAL)
                
                now = datetime.now()
                
                await self._run_all_quick_checks()
                
                await self._check_market_data_health()
                
                if hasattr(self, '_task_registry'):
                    tasks_to_restart = []
                    
                    for task_name, task_info in list(self._task_registry.items()):
                        if task_info.is_done():
                            try:
                                exc = task_info.task.exception()
                                if exc:
                                    logger.error(f"💥 Task {task_name} failed with exception: {exc}")
                                    self._task_failure_counts[task_name] = self._task_failure_counts.get(task_name, 0) + 1
                                    
                                    if task_info.can_restart():
                                        tasks_to_restart.append((task_name, task_info))
                            except asyncio.CancelledError:
                                pass
                            except asyncio.InvalidStateError:
                                pass
                            continue
                        
                        task_age = (now - task_info.created_at).total_seconds()
                        
                        threshold = LONG_RUNNING_THRESHOLD_CRITICAL if task_info.critical else LONG_RUNNING_THRESHOLD_NORMAL
                        
                        if task_age > WARNING_THRESHOLD and task_age <= threshold:
                            max_restarts_str = "∞" if task_info.max_restarts < 0 else str(task_info.max_restarts)
                            logger.warning(
                                f"🔥 LONG TASK: {task_name} running for {task_age:.0f}s "
                                f"(priority={task_info.priority.name}, critical={task_info.critical}, "
                                f"restarts={task_info.restart_count}/{max_restarts_str})"
                            )
                        
                        if task_age > threshold:
                            failure_count = self._task_failure_counts.get(task_name, 0)
                            max_restarts_str = "∞" if task_info.max_restarts < 0 else str(task_info.max_restarts)
                            logger.error(
                                f"💀 STUCK TASK: {task_name} exceeded {threshold}s threshold! "
                                f"(age={task_age:.0f}s, critical={task_info.critical}, "
                                f"failures={failure_count}, restarts={task_info.restart_count}/{max_restarts_str})"
                            )
                            
                            if not task_info.critical:
                                if task_info.can_restart():
                                    tasks_to_restart.append((task_name, task_info))
                                else:
                                    logger.error(f"❌ Task {task_name} cannot restart - max restarts exceeded, cancelling")
                                    try:
                                        task_info.task.cancel()
                                    except Exception as cancel_err:
                                        logger.error(f"Failed to cancel stuck task {task_name}: {cancel_err}")
                            else:
                                logger.warning(
                                    f"⚠️ Critical task {task_name} is stuck but will not be cancelled. "
                                    f"Monitoring for recovery..."
                                )
                    
                    for task_name, task_info in tasks_to_restart:
                        logger.warning(f"🔄 Attempting restart of task: {task_name}")
                        restart_success = await self._restart_task(task_name, task_info)
                        if restart_success:
                            logger.info(f"✅ Task {task_name} restarted successfully")
                        else:
                            logger.error(f"❌ Failed to restart task {task_name}")
                
                if hasattr(self, '_last_health_log'):
                    if (now - self._last_health_log).total_seconds() > 600:
                        active_tasks = sum(1 for t in self._task_registry.values() if not t.is_done())
                        failed_tasks = sum(1 for t in self._task_registry.values() if t.is_done())
                        failure_counts = dict(self._task_failure_counts)
                        
                        logger.info(
                            f"📊 Health check: {active_tasks} active, {failed_tasks} done/failed, "
                            f"running={self.running}, failures={failure_counts}"
                        )
                        
                        if self.market_data:
                            md_status = {
                                'connected': self.market_data.is_connected(),
                                'simulator': self.market_data.use_simulator,
                                'reconnects': self.market_data.reconnect_attempts
                            }
                            logger.info(f"📡 Market data status: {md_status}")
                        
                        self._last_health_log = now
                else:
                    self._last_health_log = now
                    
            except asyncio.CancelledError:
                logger.debug("Health check task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in health check: {e}")
                if self.error_handler:
                    self.error_handler.log_exception(e, "health_check_long_running")
                await asyncio.sleep(10)
    
    async def setup_scheduled_tasks(self):
        if not self.config_valid or not self.task_scheduler:
            logger.warning("Skipping scheduled tasks setup - limited mode or scheduler not initialized")
            return
            
        bot_components = {
            'chart_generator': self.chart_generator,
            'alert_system': self.alert_system,
            'db_manager': self.db_manager,
            'market_data': self.market_data,
            'position_tracker': self.position_tracker
        }
        
        setup_default_tasks(self.task_scheduler, bot_components)
        logger.info("Scheduled tasks configured")
    
    async def start(self):
        logger.info("=" * 60)
        logger.info("XAUUSD TRADING BOT STARTING")
        logger.info("=" * 60)
        logger.info(f"Mode: {'DRY RUN (Simulation)' if self.config.DRY_RUN else 'LIVE'}")
        logger.info(f"Config Valid: {'YES ✅' if self.config_valid else 'NO ⚠️ (Limited Mode)'}")
        
        if self.config.TELEGRAM_BOT_TOKEN:
            logger.info(f"Telegram Bot Token: Configured ({self.config.get_masked_token()})")
            
            if ':' in self.config.TELEGRAM_BOT_TOKEN and len(self.config.TELEGRAM_BOT_TOKEN) > 40:
                logger.warning("⚠️ Bot token detected - ensure it's never logged in plain text")
        else:
            logger.warning("Telegram Bot Token: NOT CONFIGURED ⚠️")
        
        logger.info(f"Authorized Users: {len(self.config.AUTHORIZED_USER_IDS)}")
        logger.info(f"LOT_SIZE: {self.config.LOT_SIZE} | XAUUSD_PIP_VALUE: {self.config.XAUUSD_PIP_VALUE}")
        logger.info(f"DYNAMIC_SL_THRESHOLD: ${self.config.DYNAMIC_SL_LOSS_THRESHOLD} | FIXED_RISK: ${self.config.FIXED_RISK_AMOUNT}")
        
        if self.config.TELEGRAM_WEBHOOK_MODE:
            webhook_url = self._auto_detect_webhook_url()
            if webhook_url:
                self.config.WEBHOOK_URL = webhook_url
                logger.info(f"Webhook URL auto-detected: {webhook_url}")
            else:
                logger.info(f"Webhook mode enabled with URL: {self.config.WEBHOOK_URL}")
        
        logger.info("=" * 60)
        
        if not self.config_valid:
            logger.warning("=" * 60)
            logger.warning("RUNNING IN LIMITED MODE")
            logger.warning("=" * 60)
            logger.warning("Bot functionality will be limited due to missing configuration.")
            logger.warning("")
            logger.warning("To enable full functionality, set these environment variables:")
            if not self.config.TELEGRAM_BOT_TOKEN:
                logger.warning("  - TELEGRAM_BOT_TOKEN (get from @BotFather on Telegram)")
            if not self.config.AUTHORIZED_USER_IDS:
                logger.warning("  - AUTHORIZED_USER_IDS (your Telegram user ID)")
            logger.warning("")
            logger.warning("Health check endpoint will remain available at /health")
            logger.warning("=" * 60)
            
            logger.info("Starting health check server only...")
            await self.start_health_server()
            
            logger.info("=" * 60)
            logger.info("BOT RUNNING IN LIMITED MODE - HEALTH CHECK AVAILABLE")
            logger.info("=" * 60)
            logger.info("Set environment variables and restart to enable trading functionality")
            
            await self.shutdown_event.wait()
            return
        
        self.running = True
        
        assert self.market_data is not None, "Market data should be initialized in full mode"
        assert self.telegram_bot is not None, "Telegram bot should be initialized in full mode"
        assert self.task_scheduler is not None, "Task scheduler should be initialized early (available in all modes)"
        assert self.position_tracker is not None, "Position tracker should be initialized in full mode"
        assert self.alert_system is not None, "Alert system should be initialized in full mode"
        
        try:
            logger.info("Starting health check server...")
            await self.start_health_server()
            
            if self.config.SELF_PING_ENABLED:
                logger.info("Starting self-ping keep-alive task (prevents Koyeb sleeping)...")
                self_ping_task = asyncio.create_task(self._self_ping_keep_alive())
                await self.register_task(
                    name="self_ping_keep_alive",
                    task=self_ping_task,
                    priority=TaskPriority.LOW,
                    critical=False,
                    cancel_timeout=3.0,
                    max_restarts=-1
                )
                logger.info(f"✅ Self-ping task started (interval: {self.config.SELF_PING_INTERVAL}s)")
            else:
                logger.info("🔇 Self-ping keep-alive is DISABLED")
            
            logger.info("Starting health check for long-running tasks...")
            health_check_task = asyncio.create_task(self._health_check_long_running())
            await self.register_task(
                name="health_check_long_running",
                task=health_check_task,
                priority=TaskPriority.LOW,
                critical=False,
                cancel_timeout=3.0,
                max_restarts=-1
            )
            
            logger.info("Loading candles from database...")
            await self.market_data.load_candles_from_db(self.db_manager)
            
            logger.info("Connecting to market data feed...")
            market_task = asyncio.create_task(self.market_data.connect_websocket())
            await self.register_task(
                name="market_data_websocket",
                task=market_task,
                priority=TaskPriority.CRITICAL,
                critical=True,
                cancel_timeout=10.0
            )
            
            logger.info("Waiting for initial market data (max 10s)...")
            for i in range(10):
                await asyncio.sleep(1)
                if self.market_data.is_connected():
                    logger.info("✅ Market data connection established")
                    break
                if i % 3 == 0:
                    logger.info(f"Connecting to market data... ({i}s)")
            
            if not self.market_data.is_connected():
                logger.warning("⚠️ Market data not connected - using cached candles or simulator mode")
            
            logger.info("Setting up scheduled tasks...")
            await self.setup_scheduled_tasks()
            
            logger.info("Starting task scheduler...")
            await self.task_scheduler.start()
            
            logger.info("Starting position tracker...")
            logger.info("Reloading active positions from database...")
            await self.position_tracker.reload_active_positions()
            
            position_task = asyncio.create_task(
                self.position_tracker.monitor_positions(self.market_data)
            )
            await self.register_task(
                name="position_tracker",
                task=position_task,
                priority=TaskPriority.HIGH,
                critical=False,
                cancel_timeout=5.0
            )
            
            logger.info("Initializing Telegram bot...")
            bot_initialized = await self.telegram_bot.initialize()
            
            if not bot_initialized:
                logger.error("Failed to initialize Telegram bot!")
                return
            
            if self.telegram_bot.app and self.config.AUTHORIZED_USER_IDS:
                self.alert_system.set_telegram_app(
                    self.telegram_bot.app,
                    self.config.AUTHORIZED_USER_IDS,
                    send_message_callback=self.telegram_bot._send_telegram_message
                )
                self.alert_system.telegram_app = self.telegram_bot.app
                self.position_tracker.telegram_app = self.telegram_bot.app
                logger.info("Telegram app set for alert system and position tracker with rate-limited callback")
            
            if self.config.TELEGRAM_WEBHOOK_MODE:
                if self.config.WEBHOOK_URL:
                    logger.info(f"Setting up webhook: {self.config.WEBHOOK_URL}")
                    try:
                        success = await self.telegram_bot.setup_webhook(self.config.WEBHOOK_URL)
                        if success:
                            logger.info("✅ Webhook setup completed successfully")
                        else:
                            logger.error("❌ Webhook setup failed!")
                    except Exception as e:
                        logger.error(f"❌ Failed to setup webhook: {e}")
                        if self.error_handler:
                            self.error_handler.log_exception(e, "webhook_setup")
                else:
                    logger.error("=" * 60)
                    logger.error("⚠️ WEBHOOK MODE ENABLED BUT NO WEBHOOK_URL!")
                    logger.error("=" * 60)
                    logger.error("Webhook mode is enabled but WEBHOOK_URL is not set.")
                    logger.error("This means bot CANNOT receive Telegram updates!")
                    logger.error("")
                    logger.error("To fix this:")
                    logger.error("1. Set WEBHOOK_URL environment variable in Koyeb, OR")
                    logger.error("2. Set KOYEB_PUBLIC_DOMAIN environment variable, OR")
                    logger.error("3. Run this command to set webhook manually:")
                    logger.error("   python3 fix_webhook.py")
                    logger.error("")
                    logger.error("Bot will continue but WILL NOT respond to commands!")
                    logger.error("=" * 60)
            
            logger.info("Starting background cleanup tasks...")
            await self.telegram_bot.start_background_cleanup_tasks()
            
            logger.info("Starting Telegram bot polling...")
            bot_task = asyncio.create_task(self.telegram_bot.run())
            await self.register_task(
                name="telegram_bot",
                task=bot_task,
                priority=TaskPriority.HIGH,
                critical=False,
                cancel_timeout=8.0
            )
            
            logger.info("Waiting for candles to build (minimal 30 candles, max 20s)...")
            candle_ready = False
            for i in range(20):
                await asyncio.sleep(1)
                try:
                    df_check = await asyncio.wait_for(
                        self.market_data.get_historical_data('M1', 100),
                        timeout=3.0
                    )
                    if df_check is not None and len(df_check) >= 30:
                        logger.info(f"✅ Got {len(df_check)} candles, ready for trading!")
                        candle_ready = True
                        break
                except asyncio.TimeoutError:
                    pass
                if i % 5 == 0 and i > 0:
                    logger.info(f"Building candles... {i}s elapsed")
            
            if not candle_ready:
                logger.warning("⚠️ Candles not fully built yet, but continuing - bot will use available data")
            
            if self.telegram_bot.app and self.config.AUTHORIZED_USER_IDS:
                startup_msg = (
                    "🤖 *Bot Started Successfully*\n\n"
                    f"Mode: {'DRY RUN' if self.config.DRY_RUN else 'LIVE'}\n"
                    f"Market: {'Connected' if self.market_data.is_connected() else 'Connecting...'}\n"
                    f"Status: Auto-monitoring AKTIF ✅\n\n"
                    "Bot akan otomatis mendeteksi sinyal trading.\n"
                    "Gunakan /help untuk list command"
                )
                
                valid_user_ids = []
                for user_id in self.config.AUTHORIZED_USER_IDS:
                    try:
                        chat_info = await asyncio.wait_for(
                            self.telegram_bot.app.bot.get_chat(user_id),
                            timeout=5.0
                        )
                        if chat_info.type == 'bot':
                            logger.warning(f"Skipping bot ID {user_id} - cannot send messages to bots")
                            continue
                        
                        valid_user_ids.append(user_id)
                        await asyncio.wait_for(
                            self.telegram_bot.app.bot.send_message(
                                chat_id=user_id,
                                text=startup_msg,
                                parse_mode='Markdown'
                            ),
                            timeout=5.0
                        )
                        logger.debug(f"Startup message sent successfully to user {user_id}")
                    except Exception as telegram_error:
                        error_type = type(telegram_error).__name__
                        error_msg = str(telegram_error).lower()
                        
                        if 'bot' in error_msg and 'forbidden' in error_msg:
                            logger.warning(f"Skipping bot ID {user_id} - Telegram bots cannot receive messages")
                        else:
                            logger.error(f"Failed to send startup message to user {user_id}: [{error_type}] {telegram_error}")
                            if self.error_handler:
                                self.error_handler.log_exception(telegram_error, f"startup_message_user_{user_id}")
                
                if valid_user_ids:
                    logger.info(f"Auto-starting monitoring for {len(valid_user_ids)} valid users...")
                    await self.telegram_bot.auto_start_monitoring(valid_user_ids)
                else:
                    logger.warning("No valid user IDs found - all IDs are either bots or invalid")
            
            logger.info("=" * 60)
            logger.info("BOT IS NOW RUNNING")
            logger.info(f"Registered tasks: {list(self._task_registry.keys())}")
            logger.info("=" * 60)
            logger.info("Press Ctrl+C to stop")
            
            await self.shutdown_event.wait()
            
        except asyncio.CancelledError:
            logger.info("Bot tasks cancelled")
        except Exception as e:
            logger.error(f"Error during bot operation: {e}")
            if self.error_handler:
                self.error_handler.log_exception(e, "main_loop")
            if self.alert_system:
                await self.alert_system.send_system_error(f"Bot crashed: {str(e)}")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        async with self._shutdown_lock:
            if self._shutdown_in_progress:
                self._shutdown_count += 1
                logger.warning(f"[SHUTDOWN] Shutdown already in progress (signal count: {self._shutdown_count})")
                if self._shutdown_count >= 3:
                    logger.error("[SHUTDOWN] Received 3+ shutdown signals, forcing immediate exit")
                    sys.exit(1)
                return
            
            self._shutdown_in_progress = True
        
        if not self.running and not self.health_server:
            logger.debug("[SHUTDOWN] Bot not running and no health server, skipping shutdown")
            self._shutdown_in_progress = False
            return
        
        logger.info("=" * 60)
        logger.info("[SHUTDOWN] GRACEFUL SHUTDOWN INITIATED")
        logger.info("=" * 60)
        
        self.running = False
        loop = asyncio.get_running_loop()
        shutdown_start_time = loop.time()
        
        def log_progress(phase: str, status: str = "started"):
            elapsed = loop.time() - shutdown_start_time
            logger.info(f"[SHUTDOWN] [{elapsed:.1f}s] Phase: {phase} - {status}")
        
        try:
            log_progress("MarketData", "saving candles and disconnecting")
            if self.market_data:
                try:
                    await asyncio.wait_for(
                        self.market_data.save_candles_to_db(self.db_manager),
                        timeout=self.SHUTDOWN_PHASE_TIMEOUT
                    )
                    log_progress("MarketData", "candles saved")
                except asyncio.TimeoutError:
                    logger.warning("[SHUTDOWN] Market data save timed out")
                except Exception as e:
                    logger.error(f"[SHUTDOWN] Error saving candles: {e}")
                
                try:
                    self.market_data.disconnect()
                    log_progress("MarketData", "disconnected ✓")
                except Exception as e:
                    logger.error(f"[SHUTDOWN] Error disconnecting market data: {e}")
            
            log_progress("Telegram", "stopping bot")
            if self.telegram_bot:
                try:
                    await asyncio.wait_for(
                        self.telegram_bot.stop_background_cleanup_tasks(),
                        timeout=5
                    )
                except asyncio.TimeoutError:
                    logger.warning("[SHUTDOWN] Background cleanup tasks shutdown timed out")
                except Exception as e:
                    logger.error(f"[SHUTDOWN] Error stopping background cleanup tasks: {e}")
                
                try:
                    await asyncio.wait_for(
                        self.telegram_bot.stop(),
                        timeout=self.SHUTDOWN_PHASE_TIMEOUT
                    )
                    log_progress("Telegram", "stopped ✓")
                except asyncio.TimeoutError:
                    logger.warning("[SHUTDOWN] Telegram bot shutdown timed out")
                except Exception as e:
                    logger.error(f"[SHUTDOWN] Error stopping Telegram bot: {e}")
            
            log_progress("TaskScheduler", "stopping")
            if self.task_scheduler:
                try:
                    await asyncio.wait_for(
                        self.task_scheduler.stop(),
                        timeout=self.SHUTDOWN_PHASE_TIMEOUT
                    )
                    log_progress("TaskScheduler", "stopped ✓")
                except asyncio.TimeoutError:
                    logger.warning("[SHUTDOWN] Task scheduler shutdown timed out")
                except Exception as e:
                    logger.error(f"[SHUTDOWN] Error stopping task scheduler: {e}")
            
            log_progress("PositionTracker", "stopping")
            if self.position_tracker:
                try:
                    self.position_tracker.stop_monitoring()
                    log_progress("PositionTracker", "stopped ✓")
                except Exception as e:
                    logger.error(f"[SHUTDOWN] Error stopping position tracker: {e}")
            
            log_progress("RegisteredTasks", "cancelling all")
            cancelled = await self._cancel_all_registered_tasks(timeout=10.0)
            log_progress("RegisteredTasks", f"cancelled {cancelled} tasks ✓")
            
            log_progress("ChartGenerator", "shutting down")
            if self.chart_generator:
                try:
                    self.chart_generator.shutdown()
                    log_progress("ChartGenerator", "shutdown ✓")
                except Exception as e:
                    logger.error(f"[SHUTDOWN] Error shutting down chart generator: {e}")
            
            log_progress("HTTPServer", "stopping health server")
            if self.health_server:
                try:
                    await asyncio.wait_for(
                        self.health_server.cleanup(),
                        timeout=5
                    )
                    log_progress("HTTPServer", "stopped ✓")
                except asyncio.TimeoutError:
                    logger.warning("[SHUTDOWN] Health server shutdown timed out")
                except Exception as e:
                    logger.error(f"[SHUTDOWN] Error stopping health server: {e}")
            
            log_progress("Database", "closing connections")
            if self.db_manager:
                try:
                    self.db_manager.close()
                    log_progress("Database", "closed ✓")
                except Exception as e:
                    logger.error(f"[SHUTDOWN] Error closing database: {e}")
            
            shutdown_duration = loop.time() - shutdown_start_time
            
            logger.info("=" * 60)
            logger.info(f"[SHUTDOWN] COMPLETE in {shutdown_duration:.2f}s")
            if shutdown_duration > self.SHUTDOWN_TOTAL_TIMEOUT:
                logger.warning(f"[SHUTDOWN] Exceeded timeout ({shutdown_duration:.2f}s > {self.SHUTDOWN_TOTAL_TIMEOUT}s)")
            logger.info("=" * 60)
            
            import logging
            logging.shutdown()
            
        except Exception as e:
            logger.error(f"[SHUTDOWN] Error during shutdown: {e}")
            import logging
            logging.shutdown()
            raise
        finally:
            self._shutdown_in_progress = False
            
            if hasattr(self, 'instance_lock') and self.instance_lock:
                try:
                    logger.info("[SHUTDOWN] Releasing instance lock...")
                    self.instance_lock.release()
                    logger.info("[SHUTDOWN] Instance lock released ✓")
                except Exception as e:
                    logger.error(f"[SHUTDOWN] Error releasing instance lock: {e}")


async def main():
    orchestrator = TradingBotOrchestrator()
    loop = asyncio.get_running_loop()
    
    shutdown_requested = False
    
    def signal_handler(sig, frame):
        nonlocal shutdown_requested
        signame = signal.Signals(sig).name
        
        if shutdown_requested:
            orchestrator._shutdown_count += 1
            try:
                loop.call_soon_threadsafe(
                    lambda: logger.warning(f"[SIGNAL] Received {signame} again (count: {orchestrator._shutdown_count})")
                )
            except RuntimeError:
                pass
            if orchestrator._shutdown_count >= 3:
                sys.exit(1)
            return
        
        shutdown_requested = True
        try:
            loop.call_soon_threadsafe(
                lambda: logger.info(f"[SIGNAL] Received {signame} ({sig}), initiating graceful shutdown...")
            )
        except RuntimeError:
            pass
        
        try:
            loop.call_soon_threadsafe(orchestrator.shutdown_event.set)
        except RuntimeError:
            pass
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await orchestrator.start()
        return 0
    except KeyboardInterrupt:
        logger.info("[SIGNAL] KeyboardInterrupt received")
        return 0
    except Exception as e:
        logger.error(f"[MAIN] Unhandled exception: {e}")
        return 1
    finally:
        if not orchestrator.shutdown_in_progress:
            await orchestrator.shutdown()


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        exit_code = 0
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        exit_code = 1
    
    sys.exit(exit_code)
