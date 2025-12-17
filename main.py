"""
XAUUSD Trading Bot - Lightweight Version for Koyeb Free Tier

Core features:
- Trading signals (BUY/SELL)
- Position tracking with TP/SL (Win/Lose)
- Performance statistics and win rate
- Simple web dashboard
"""

import asyncio
import signal
import sys
import os
import fcntl
import atexit
import threading
import pytz
import hmac
import time
from datetime import datetime
from aiohttp import web
from typing import Optional, Dict, Any
from collections import defaultdict


class WebhookRateLimiter:
    """Simple in-memory rate limiter for webhook requests"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, list] = defaultdict(list)
        self._lock = threading.Lock()
    
    def is_allowed(self, ip: str) -> bool:
        """Check if request from IP is allowed"""
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds
        
        with self._lock:
            self._requests[ip] = [t for t in self._requests[ip] if t > cutoff_time]
            
            if len(self._requests[ip]) >= self.max_requests:
                return False
            
            self._requests[ip].append(current_time)
            return True
    
    def cleanup(self):
        """Remove expired entries to prevent memory growth"""
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds
        
        with self._lock:
            expired_ips = [ip for ip, times in self._requests.items() 
                          if not times or all(t <= cutoff_time for t in times)]
            for ip in expired_ips:
                del self._requests[ip]


def verify_webhook_signature(request_token: str, expected_secret: str) -> bool:
    """Verify Telegram webhook signature using constant-time comparison"""
    if not expected_secret:
        return True
    if not request_token:
        return False
    return hmac.compare_digest(request_token, expected_secret)


def get_client_ip(request) -> str:
    """Extract client IP from request, handling proxies"""
    forwarded = request.headers.get('X-Forwarded-For', '')
    if forwarded:
        return forwarded.split(',')[0].strip()
    real_ip = request.headers.get('X-Real-IP', '')
    if real_ip:
        return real_ip.strip()
    peername = request.transport.get_extra_info('peername')
    if peername:
        return peername[0]
    return 'unknown'


# Quick health server for Koyeb health checks
_quick_health_started = False
_quick_health_thread = None
_quick_health_stop_event = None
_initialization_status = {"phase": "starting", "ready": False, "error": None}

def start_quick_health_server():
    """Start minimal health server immediately"""
    global _quick_health_thread, _quick_health_stop_event, _quick_health_started
    
    if _quick_health_thread is not None and _quick_health_thread.is_alive():
        return
    
    port = int(os.getenv('PORT', '5000'))
    _quick_health_stop_event = threading.Event()
    
    def run_health_server():
        global _quick_health_started
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def health_handler(request):
            return web.json_response({
                'status': 'ok',
                'phase': _initialization_status.get('phase', 'starting'),
                'ready': _initialization_status.get('ready', False)
            })
        
        async def run():
            global _quick_health_started
            app = web.Application()
            async def ping_handler(request):
                return web.Response(text='pong')
            
            app.router.add_get('/health', health_handler)
            app.router.add_get('/ping', ping_handler)
            app.router.add_get('/', health_handler)
            
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, '0.0.0.0', port)
            await site.start()
            _quick_health_started = True
            print(f"[QUICK-START] Health server on port {port}")
            
            while _quick_health_stop_event is None or not _quick_health_stop_event.is_set():
                await asyncio.sleep(0.5)
                if _quick_health_stop_event is not None and _quick_health_stop_event.is_set():
                    break
            
            await runner.cleanup()
        
        try:
            loop.run_until_complete(run())
        except Exception as e:
            print(f"[QUICK-START] Error: {e}")
        finally:
            loop.close()
    
    _quick_health_thread = threading.Thread(target=run_health_server, daemon=True)
    _quick_health_thread.start()
    
    import time
    for _ in range(20):
        if _quick_health_started:
            break
        time.sleep(0.1)

def stop_quick_health_server():
    global _quick_health_stop_event, _quick_health_thread
    if _quick_health_stop_event:
        _quick_health_stop_event.set()
    if _quick_health_thread and _quick_health_thread.is_alive():
        _quick_health_thread.join(timeout=3.0)
    print("[QUICK-START] Health server stopped")

def update_init_status(phase: str, ready: bool = False, error=None):
    global _initialization_status
    _initialization_status = {"phase": phase, "ready": ready, "error": error}

# Start health server immediately
try:
    start_quick_health_server()
except Exception as e:
    print(f"[QUICK-START] Warning: {e}")

update_init_status("importing")

# Core imports
from config import Config, ConfigError
from bot.logger import setup_logger
from bot.database import DatabaseManager
from bot.market_data import MarketDataClient
from bot.strategy import TradingStrategy
from bot.risk_manager import RiskManager
from bot.position_tracker import PositionTracker
from bot.telegram_bot import TradingBot
from bot.alert_system import AlertSystem
from bot.error_handler import ErrorHandler
from bot.user_manager import UserManager
from bot.task_scheduler import TaskScheduler
from bot.indicators import IndicatorEngine
from bot.signal_rules import AggressiveSignalRules
from bot.signal_session_manager import SignalSessionManager
from bot.signal_quality_tracker import SignalQualityTracker
from bot.market_regime import MarketRegimeDetector
from bot.auto_optimizer import AutoOptimizer
from bot.chart_generator import ChartGenerator

update_init_status("imports_complete")
logger = setup_logger('Main')

LOCK_FILE_PATH = '/tmp/xauusd_trading_bot.lock'


class SingleInstanceLock:
    """Prevent multiple bot instances"""
    
    def __init__(self, lock_file_path: str = LOCK_FILE_PATH):
        self.lock_file_path = lock_file_path
        self.lock_file = None
        self.acquired = False
    
    def acquire(self) -> bool:
        try:
            self.lock_file = open(self.lock_file_path, 'w')
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.lock_file.write(f"{os.getpid()}\n{datetime.now().isoformat()}\n")
            self.lock_file.flush()
            self.acquired = True
            atexit.register(self.release)
            logger.info(f"Bot instance lock acquired (PID: {os.getpid()})")
            return True
        except (IOError, OSError):
            if self.lock_file:
                self.lock_file.close()
            logger.error("Failed to acquire bot lock - another instance may be running")
            return False
    
    def release(self):
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


class TradingBotOrchestrator:
    """Simplified bot orchestrator for Koyeb free tier"""
    
    def __init__(self):
        self.instance_lock = SingleInstanceLock()
        if not self.instance_lock.acquire():
            raise RuntimeError("Another instance is already running")
        
        self.config = Config()
        self.config_valid = False
        self.running = False
        self._shutdown_in_progress = False
        self.shutdown_event = asyncio.Event()
        self.health_runner = None
        
        # Initialize webhook rate limiter
        self.webhook_rate_limiter = WebhookRateLimiter(
            max_requests=self.config.WEBHOOK_RATE_LIMIT,
            window_seconds=self.config.WEBHOOK_RATE_LIMIT_WINDOW
        )
        
        # Refresh and validate config
        logger.info("=" * 50)
        logger.info("XAUUSD TRADING BOT - LIGHTWEIGHT VERSION")
        logger.info("=" * 50)
        
        refresh_result = Config._refresh_secrets()
        logger.info(f"Token: {'Set' if refresh_result['token_set'] else 'NOT SET'}")
        logger.info(f"Authorized Users: {refresh_result['users_count']}")
        
        try:
            self.config.validate()
            self.config_valid = True
            logger.info("Configuration validated successfully")
        except ConfigError as e:
            logger.warning(f"Configuration issues: {e}")
            self.config_valid = False
        
        # Initialize database
        self.db_manager = DatabaseManager(
            db_path=self.config.DATABASE_PATH,
            database_url=self.config.DATABASE_URL
        )
        logger.info("Database initialized")
        
        # Initialize task scheduler
        self.task_scheduler = TaskScheduler(self.config)
        logger.info("Task scheduler initialized")
        
        if not self.config_valid:
            logger.warning("Running in limited mode - set environment variables to enable full functionality")
            self._init_null_components()
            return
        
        # Initialize core components
        self._init_components()
        logger.info("All components initialized")
    
    def _init_null_components(self):
        """Initialize null components for limited mode"""
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
        self.signal_session_manager = None
    
    def _init_components(self):
        """Initialize trading components"""
        self.error_handler = ErrorHandler(self.config)
        self.user_manager = UserManager(self.config)
        self.market_data = MarketDataClient(self.config)
        self.strategy = TradingStrategy(self.config)
        self.indicator_engine = IndicatorEngine(self.config)
        self.signal_quality_tracker = SignalQualityTracker(self.db_manager, self.config)
        self.market_regime_detector = MarketRegimeDetector(self.config, self.indicator_engine)
        self.signal_rules = AggressiveSignalRules(self.config, self.indicator_engine)
        self.auto_optimizer = AutoOptimizer(self.signal_quality_tracker, self.config)
        self.risk_manager = RiskManager(self.config, self.db_manager)
        self.chart_generator = ChartGenerator(self.config)
        self.alert_system = AlertSystem(self.config, self.db_manager)
        self.signal_session_manager = SignalSessionManager()
        
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
    
    async def _create_health_app(self) -> web.Application:
        """Create health check and dashboard web app"""
        app = web.Application()
        
        async def health_handler(request):
            return web.json_response({
                'status': 'ok',
                'ready': self.config_valid,
                'timestamp': datetime.now(pytz.UTC).isoformat()
            })
        
        async def ping_handler(request):
            return web.Response(text='pong')
        
        async def dashboard_handler(request):
            stats = self._get_stats()
            html = self._generate_dashboard_html(stats)
            return web.Response(text=html, content_type='text/html')
        
        async def api_stats_handler(request):
            return web.json_response(self._get_stats())
        
        app.router.add_get('/health', health_handler)
        app.router.add_get('/api/health', health_handler)
        app.router.add_get('/ping', ping_handler)
        app.router.add_get('/', dashboard_handler)
        app.router.add_get('/dashboard', dashboard_handler)
        app.router.add_get('/api/stats', api_stats_handler)
        
        # Add webhook if telegram bot is configured
        if self.telegram_bot and self.config_valid:
            async def webhook_handler(request):
                try:
                    client_ip = get_client_ip(request)
                    
                    # Rate limiting check
                    if not self.webhook_rate_limiter.is_allowed(client_ip):
                        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
                        return web.Response(
                            text='Too Many Requests',
                            status=429,
                            headers={'Retry-After': str(self.config.WEBHOOK_RATE_LIMIT_WINDOW)}
                        )
                    
                    # Webhook signature verification
                    if self.config.WEBHOOK_SECRET:
                        request_token = request.headers.get('X-Telegram-Bot-Api-Secret-Token', '')
                        if not verify_webhook_signature(request_token, self.config.WEBHOOK_SECRET):
                            logger.warning(f"Invalid webhook signature from IP: {client_ip}")
                            return web.Response(text='Unauthorized', status=401)
                    
                    data = await request.json()
                    
                    # Telegram user ID validation
                    user_id = None
                    if 'message' in data and 'from' in data['message']:
                        user_id = data['message']['from'].get('id')
                    elif 'callback_query' in data and 'from' in data['callback_query']:
                        user_id = data['callback_query']['from'].get('id')
                    elif 'edited_message' in data and 'from' in data['edited_message']:
                        user_id = data['edited_message']['from'].get('id')
                    
                    if user_id is not None:
                        Config.ensure_secrets_loaded()
                        if not Config.has_full_access(user_id):
                            logger.warning(f"Unauthorized user attempted access: {user_id} from IP: {client_ip}")
                            return web.Response(text='ok')
                    
                    if self.telegram_bot:
                        await self.telegram_bot.process_update(data)
                    return web.Response(text='ok')
                except Exception as e:
                    logger.error(f"Webhook error: {e}")
                    return web.Response(text='error', status=500)
            
            app.router.add_post('/bot', webhook_handler)
            app.router.add_post('/webhook', webhook_handler)
        
        return app
    
    def _get_stats(self) -> Dict[str, Any]:
        """Get bot statistics"""
        stats = {
            'status': 'running' if self.running else 'stopped',
            'config_valid': self.config_valid,
            'timestamp': datetime.now(pytz.UTC).isoformat()
        }
        
        if self.db_manager:
            try:
                perf = self.db_manager.get_performance_stats()
                if perf:
                    stats.update({
                        'total_trades': perf.get('total_trades', 0),
                        'wins': perf.get('wins', 0),
                        'losses': perf.get('losses', 0),
                        'win_rate': perf.get('win_rate', 0),
                        'total_pnl': perf.get('total_pnl', 0)
                    })
            except Exception as e:
                logger.error(f"Error getting stats: {e}")
        
        return stats
    
    def _generate_dashboard_html(self, stats: Dict[str, Any]) -> str:
        """Generate simple dashboard HTML"""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>XAUUSD Trading Bot</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               max-width: 800px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eee; }}
        .card {{ background: #16213e; border-radius: 10px; padding: 20px; margin: 10px 0; }}
        .stat {{ display: inline-block; margin: 10px 20px; text-align: center; }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #00d4ff; }}
        .stat-label {{ color: #888; }}
        .status {{ padding: 5px 15px; border-radius: 20px; display: inline-block; }}
        .status-ok {{ background: #00c853; }}
        .status-error {{ background: #ff5252; }}
        h1 {{ color: #00d4ff; }}
    </style>
</head>
<body>
    <h1>XAUUSD Trading Bot</h1>
    
    <div class="card">
        <h2>Status</h2>
        <span class="status {'status-ok' if stats.get('config_valid') else 'status-error'}">
            {'Running' if stats.get('config_valid') else 'Limited Mode'}
        </span>
        <p>Last updated: {stats.get('timestamp', 'N/A')}</p>
    </div>
    
    <div class="card">
        <h2>Performance</h2>
        <div class="stat">
            <div class="stat-value">{stats.get('total_trades', 0)}</div>
            <div class="stat-label">Total Trades</div>
        </div>
        <div class="stat">
            <div class="stat-value">{stats.get('wins', 0)}</div>
            <div class="stat-label">Wins</div>
        </div>
        <div class="stat">
            <div class="stat-value">{stats.get('losses', 0)}</div>
            <div class="stat-label">Losses</div>
        </div>
        <div class="stat">
            <div class="stat-value">{stats.get('win_rate', 0):.1f}%</div>
            <div class="stat-label">Win Rate</div>
        </div>
    </div>
    
    <script>
        setTimeout(() => location.reload(), 30000);
    </script>
</body>
</html>
"""
    
    @property
    def shutdown_in_progress(self) -> bool:
        return self._shutdown_in_progress
    
    async def start(self):
        """Start the bot"""
        self.running = True
        port = int(os.getenv('PORT', '5000'))
        
        logger.info("=" * 50)
        logger.info("STARTING BOT")
        logger.info(f"Config Valid: {'YES' if self.config_valid else 'NO'}")
        logger.info(f"Port: {port}")
        logger.info("=" * 50)
        
        # Stop quick health server
        stop_quick_health_server()
        
        # Start main health server
        app = await self._create_health_app()
        self.health_runner = web.AppRunner(app)
        await self.health_runner.setup()
        site = web.TCPSite(self.health_runner, '0.0.0.0', port)
        await site.start()
        logger.info(f"Web server started on port {port}")
        
        # Initialize telegram bot if configured
        if self.telegram_bot and self.config_valid:
            try:
                await self.telegram_bot.initialize()
                logger.info("Telegram bot initialized")
                
                # Start WebSocket connection to Deriv in background
                if self.market_data:
                    asyncio.create_task(self._run_market_data_websocket())
                    logger.info("Market data WebSocket connection started")
                
                # Start webhook mode in background
                asyncio.create_task(self._run_telegram_bot())
            except Exception as e:
                logger.error(f"Failed to initialize Telegram bot: {e}")
        
        # Wait for shutdown
        await self.shutdown_event.wait()
    
    async def _run_telegram_bot(self):
        """Run telegram bot webhook mode in background"""
        try:
            if self.telegram_bot:
                await self.telegram_bot.run_webhook()
        except Exception as e:
            logger.error(f"Telegram bot error: {e}")
    
    async def _run_market_data_websocket(self):
        """Run market data WebSocket connection to Deriv in background"""
        try:
            if self.market_data:
                logger.info("Starting Deriv WebSocket connection...")
                await self.market_data.connect_websocket()
        except Exception as e:
            logger.error(f"Market data WebSocket error: {e}", exc_info=True)
    
    async def shutdown(self):
        """Shutdown the bot"""
        if self._shutdown_in_progress:
            return
        
        self._shutdown_in_progress = True
        self.running = False
        logger.info("Shutting down...")
        
        try:
            # Stop market data WebSocket first
            if self.market_data:
                try:
                    self.market_data.running = False
                    self.market_data._is_shutting_down = True
                    logger.info("Market data WebSocket stopped")
                except Exception as e:
                    logger.error(f"Error stopping market data: {e}")
            
            if self.telegram_bot:
                try:
                    await asyncio.wait_for(self.telegram_bot.stop(), timeout=10)
                except asyncio.TimeoutError:
                    logger.warning("Telegram bot shutdown timed out")
                except Exception as e:
                    logger.error(f"Error stopping Telegram bot: {e}")
            
            if self.task_scheduler:
                try:
                    await asyncio.wait_for(self.task_scheduler.stop(), timeout=5)
                except Exception as e:
                    logger.error(f"Error stopping task scheduler: {e}")
            
            if self.health_runner:
                await self.health_runner.cleanup()
            
            if self.instance_lock:
                self.instance_lock.release()
            
            logger.info("Shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


async def main():
    """Main entry point"""
    orchestrator = TradingBotOrchestrator()
    shutdown_requested = False
    
    def signal_handler(sig, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            return
        shutdown_requested = True
        logger.info(f"Received signal {sig}, shutting down...")
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(orchestrator.shutdown_event.set)
        except RuntimeError:
            pass
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        update_init_status("starting", ready=False)
        await orchestrator.start()
        update_init_status("running", ready=True)
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted")
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        update_init_status("error", ready=False, error=str(e))
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
