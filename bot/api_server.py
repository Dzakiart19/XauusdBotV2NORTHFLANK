"""
REST API Server untuk Bot Trading XAUUSD.

Modul ini menyediakan:
- REST API endpoints untuk akses eksternal
- Webhook endpoints untuk integrasi
- Authentication dan rate limiting
- API documentation
"""

import asyncio
import hashlib
import hmac
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from functools import wraps
import pytz

from aiohttp import web
from bot.logger import setup_logger
from bot.analytics import TradingAnalytics
from bot.report_generator import ReportGenerator

logger = setup_logger('APIServer')


class APIError(Exception):
    """Exception untuk API errors"""
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class RateLimiter:
    """Simple rate limiter untuk API"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed"""
        now = time.time()
        
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        self.requests[client_id] = [
            t for t in self.requests[client_id]
            if now - t < self.window_seconds
        ]
        
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        self.requests[client_id].append(now)
        return True
    
    def get_reset_time(self, client_id: str) -> int:
        """Get seconds until rate limit resets"""
        if client_id not in self.requests or not self.requests[client_id]:
            return 0
        
        oldest = min(self.requests[client_id])
        return int(self.window_seconds - (time.time() - oldest))


class APIKeyManager:
    """Manager untuk API keys"""
    
    def __init__(self):
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.key_usage: Dict[str, int] = {}
    
    def generate_key(self, user_id: int, name: str = "default") -> str:
        """Generate new API key for user"""
        import secrets
        key = f"xau_{secrets.token_hex(32)}"
        
        self.api_keys[key] = {
            'user_id': user_id,
            'name': name,
            'created_at': datetime.now(pytz.UTC).isoformat(),
            'active': True,
            'permissions': ['read']
        }
        
        logger.info(f"API key generated for user {user_id}")
        return key
    
    def validate_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return user info"""
        if api_key in self.api_keys:
            key_info = self.api_keys[api_key]
            if key_info.get('active', False):
                self.key_usage[api_key] = self.key_usage.get(api_key, 0) + 1
                return key_info
        return None
    
    def revoke_key(self, api_key: str) -> bool:
        """Revoke an API key"""
        if api_key in self.api_keys:
            self.api_keys[api_key]['active'] = False
            logger.info(f"API key revoked")
            return True
        return False
    
    def get_user_keys(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all API keys for a user"""
        keys = []
        for key, info in self.api_keys.items():
            if info.get('user_id') == user_id:
                keys.append({
                    'key_preview': f"{key[:8]}...{key[-4:]}",
                    'name': info.get('name'),
                    'created_at': info.get('created_at'),
                    'active': info.get('active'),
                    'usage': self.key_usage.get(key, 0)
                })
        return keys


def require_api_key(func):
    """Decorator untuk require API key authentication"""
    @wraps(func)
    async def wrapper(self, request):
        api_key = request.headers.get('X-API-Key') or request.query.get('api_key')
        
        if not api_key:
            raise web.HTTPUnauthorized(
                text=json.dumps({'error': 'API key required'}),
                content_type='application/json'
            )
        
        key_info = self.api_key_manager.validate_key(api_key)
        if not key_info:
            raise web.HTTPUnauthorized(
                text=json.dumps({'error': 'Invalid API key'}),
                content_type='application/json'
            )
        
        request['user_id'] = key_info.get('user_id')
        request['permissions'] = key_info.get('permissions', [])
        
        return await func(self, request)
    return wrapper


def rate_limit(func):
    """Decorator untuk rate limiting"""
    @wraps(func)
    async def wrapper(self, request):
        client_id = request.headers.get('X-API-Key') or request.remote
        
        if not self.rate_limiter.is_allowed(client_id):
            reset_time = self.rate_limiter.get_reset_time(client_id)
            raise web.HTTPTooManyRequests(
                text=json.dumps({
                    'error': 'Rate limit exceeded',
                    'reset_in_seconds': reset_time
                }),
                content_type='application/json'
            )
        
        return await func(self, request)
    return wrapper


class TradingAPIServer:
    """REST API Server untuk trading bot"""
    
    def __init__(self, config, db_manager, market_data=None, 
                 signal_rules=None, position_tracker=None):
        self.config = config
        self.db = db_manager
        self.market_data = market_data
        self.signal_rules = signal_rules
        self.position_tracker = position_tracker
        
        self.analytics = TradingAnalytics(db_manager, config)
        self.report_generator = ReportGenerator(db_manager, config)
        self.api_key_manager = APIKeyManager()
        self.rate_limiter = RateLimiter(max_requests=100, window_seconds=60)
        
        self.jakarta_tz = pytz.timezone('Asia/Jakarta')
        self.webhooks: Dict[str, Dict[str, Any]] = {}
        
        logger.info("TradingAPIServer initialized")
    
    def create_routes(self) -> List[web.RouteDef]:
        """Create API routes"""
        return [
            web.get('/api/v1/status', self.get_status),
            web.get('/api/v1/price', self.get_price),
            web.get('/api/v1/signals', self.get_signals),
            web.get('/api/v1/signals/latest', self.get_latest_signal),
            web.get('/api/v1/positions', self.get_positions),
            web.get('/api/v1/trades', self.get_trades),
            web.get('/api/v1/performance', self.get_performance),
            web.get('/api/v1/performance/hourly', self.get_hourly_performance),
            web.get('/api/v1/reports/daily', self.get_daily_report),
            web.get('/api/v1/reports/weekly', self.get_weekly_report),
            web.get('/api/v1/reports/monthly', self.get_monthly_report),
            web.post('/api/v1/webhooks', self.register_webhook),
            web.delete('/api/v1/webhooks/{webhook_id}', self.delete_webhook),
            web.get('/api/v1/webhooks', self.list_webhooks),
            web.get('/api/v1/health', self.health_check),
            web.get('/api/v1/docs', self.get_docs),
        ]
    
    async def get_status(self, request: web.Request) -> web.Response:
        """Get bot status"""
        try:
            status = {
                'status': 'online',
                'timestamp': datetime.now(self.jakarta_tz).isoformat(),
                'version': '2.0.0',
                'market_connected': self.market_data.is_connected() if self.market_data else False,
                'components': {
                    'market_data': 'online' if self.market_data else 'offline',
                    'signal_rules': 'online' if self.signal_rules else 'offline',
                    'position_tracker': 'online' if self.position_tracker else 'offline',
                    'database': 'online' if self.db else 'offline'
                }
            }
            return web.json_response(status)
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    @rate_limit
    async def get_price(self, request: web.Request) -> web.Response:
        """Get current XAUUSD price"""
        try:
            if not self.market_data:
                return web.json_response({'error': 'Market data not available'}, status=503)
            
            price_data = self.market_data.get_current_price()
            
            if not price_data:
                return web.json_response({'error': 'Price data not available'}, status=503)
            
            response = {
                'symbol': 'XAUUSD',
                'bid': price_data.get('bid'),
                'ask': price_data.get('ask'),
                'mid': price_data.get('mid'),
                'spread': price_data.get('spread'),
                'timestamp': datetime.now(self.jakarta_tz).isoformat()
            }
            
            return web.json_response(response)
        except Exception as e:
            logger.error(f"Error getting price: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    @rate_limit
    async def get_signals(self, request: web.Request) -> web.Response:
        """Get recent signals"""
        try:
            limit = int(request.query.get('limit', 10))
            limit = min(limit, 100)
            
            signals = []
            
            return web.json_response({
                'signals': signals,
                'count': len(signals),
                'limit': limit
            })
        except Exception as e:
            logger.error(f"Error getting signals: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    @rate_limit
    async def get_latest_signal(self, request: web.Request) -> web.Response:
        """Get latest signal"""
        try:
            signal = None
            
            if signal:
                return web.json_response({'signal': signal})
            else:
                return web.json_response({'signal': None, 'message': 'No active signal'})
        except Exception as e:
            logger.error(f"Error getting latest signal: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    @rate_limit
    async def get_positions(self, request: web.Request) -> web.Response:
        """Get open positions"""
        try:
            user_id = request.query.get('user_id')
            
            if not self.position_tracker:
                return web.json_response({'error': 'Position tracker not available'}, status=503)
            
            positions = []
            
            return web.json_response({
                'positions': positions,
                'count': len(positions)
            })
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    @rate_limit
    async def get_trades(self, request: web.Request) -> web.Response:
        """Get trade history"""
        try:
            user_id = request.query.get('user_id')
            limit = int(request.query.get('limit', 20))
            limit = min(limit, 100)
            
            if user_id:
                trades = self.analytics.get_recent_trades(int(user_id), limit)
            else:
                trades = []
            
            return web.json_response({
                'trades': trades,
                'count': len(trades),
                'limit': limit
            })
        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    @rate_limit
    async def get_performance(self, request: web.Request) -> web.Response:
        """Get trading performance"""
        try:
            user_id = request.query.get('user_id')
            days = int(request.query.get('days', 30))
            
            user_id_int = int(user_id) if user_id else None
            performance = self.analytics.get_trading_performance(user_id_int, days)
            
            return web.json_response(performance)
        except Exception as e:
            logger.error(f"Error getting performance: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    @rate_limit
    async def get_hourly_performance(self, request: web.Request) -> web.Response:
        """Get hourly performance breakdown"""
        try:
            user_id = request.query.get('user_id')
            days = int(request.query.get('days', 30))
            
            user_id_int = int(user_id) if user_id else None
            hourly = self.analytics.get_hourly_stats(user_id_int, days)
            
            return web.json_response(hourly)
        except Exception as e:
            logger.error(f"Error getting hourly performance: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    @rate_limit
    async def get_daily_report(self, request: web.Request) -> web.Response:
        """Get daily report"""
        try:
            user_id = request.query.get('user_id')
            date_str = request.query.get('date')
            format_type = request.query.get('format', 'json')
            
            user_id_int = int(user_id) if user_id else None
            date = datetime.fromisoformat(date_str) if date_str else None
            
            report = self.report_generator.generate_daily_report(user_id_int, date)
            
            if format_type == 'text':
                text = self.report_generator.format_report_text(report)
                return web.Response(text=text, content_type='text/plain')
            elif format_type == 'csv':
                csv_data = self.report_generator.export_to_csv(report)
                return web.Response(
                    text=csv_data, 
                    content_type='text/csv',
                    headers={'Content-Disposition': 'attachment; filename="daily_report.csv"'}
                )
            else:
                return web.json_response(report)
        except Exception as e:
            logger.error(f"Error getting daily report: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    @rate_limit
    async def get_weekly_report(self, request: web.Request) -> web.Response:
        """Get weekly report"""
        try:
            user_id = request.query.get('user_id')
            format_type = request.query.get('format', 'json')
            
            user_id_int = int(user_id) if user_id else None
            report = self.report_generator.generate_weekly_report(user_id_int)
            
            if format_type == 'text':
                text = self.report_generator.format_report_text(report)
                return web.Response(text=text, content_type='text/plain')
            elif format_type == 'csv':
                csv_data = self.report_generator.export_to_csv(report)
                return web.Response(
                    text=csv_data,
                    content_type='text/csv',
                    headers={'Content-Disposition': 'attachment; filename="weekly_report.csv"'}
                )
            else:
                return web.json_response(report)
        except Exception as e:
            logger.error(f"Error getting weekly report: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    @rate_limit
    async def get_monthly_report(self, request: web.Request) -> web.Response:
        """Get monthly report"""
        try:
            user_id = request.query.get('user_id')
            year = request.query.get('year')
            month = request.query.get('month')
            format_type = request.query.get('format', 'json')
            
            user_id_int = int(user_id) if user_id else None
            year_int = int(year) if year else None
            month_int = int(month) if month else None
            
            report = self.report_generator.generate_monthly_report(user_id_int, year_int, month_int)
            
            if format_type == 'text':
                text = self.report_generator.format_report_text(report)
                return web.Response(text=text, content_type='text/plain')
            elif format_type == 'csv':
                csv_data = self.report_generator.export_to_csv(report)
                return web.Response(
                    text=csv_data,
                    content_type='text/csv',
                    headers={'Content-Disposition': 'attachment; filename="monthly_report.csv"'}
                )
            else:
                return web.json_response(report)
        except Exception as e:
            logger.error(f"Error getting monthly report: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def register_webhook(self, request: web.Request) -> web.Response:
        """Register a webhook for signal notifications"""
        try:
            data = await request.json()
            
            url = data.get('url')
            events = data.get('events', ['signal', 'trade_close'])
            secret = data.get('secret')
            
            if not url:
                return web.json_response({'error': 'URL required'}, status=400)
            
            import secrets
            webhook_id = secrets.token_hex(16)
            
            self.webhooks[webhook_id] = {
                'url': url,
                'events': events,
                'secret': secret,
                'created_at': datetime.now(self.jakarta_tz).isoformat(),
                'active': True,
                'delivery_count': 0,
                'last_delivery': None
            }
            
            logger.info(f"Webhook registered: {webhook_id}")
            
            return web.json_response({
                'webhook_id': webhook_id,
                'url': url,
                'events': events,
                'message': 'Webhook registered successfully'
            }, status=201)
        except Exception as e:
            logger.error(f"Error registering webhook: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def delete_webhook(self, request: web.Request) -> web.Response:
        """Delete a webhook"""
        try:
            webhook_id = request.match_info['webhook_id']
            
            if webhook_id in self.webhooks:
                del self.webhooks[webhook_id]
                logger.info(f"Webhook deleted: {webhook_id}")
                return web.json_response({'message': 'Webhook deleted'})
            else:
                return web.json_response({'error': 'Webhook not found'}, status=404)
        except Exception as e:
            logger.error(f"Error deleting webhook: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def list_webhooks(self, request: web.Request) -> web.Response:
        """List all webhooks"""
        try:
            webhooks = []
            for wh_id, wh_data in self.webhooks.items():
                webhooks.append({
                    'id': wh_id,
                    'url': wh_data['url'],
                    'events': wh_data['events'],
                    'active': wh_data['active'],
                    'delivery_count': wh_data['delivery_count'],
                    'created_at': wh_data['created_at']
                })
            
            return web.json_response({'webhooks': webhooks, 'count': len(webhooks)})
        except Exception as e:
            logger.error(f"Error listing webhooks: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def health_check(self, request: web.Request) -> web.Response:
        """API health check"""
        return web.json_response({
            'status': 'healthy',
            'timestamp': datetime.now(self.jakarta_tz).isoformat(),
            'api_version': 'v1'
        })
    
    async def get_docs(self, request: web.Request) -> web.Response:
        """Get API documentation"""
        docs = {
            'api_name': 'XAUUSD Trading Bot API',
            'version': 'v1',
            'base_url': '/api/v1',
            'endpoints': [
                {
                    'path': '/status',
                    'method': 'GET',
                    'description': 'Get bot status',
                    'authentication': False
                },
                {
                    'path': '/price',
                    'method': 'GET',
                    'description': 'Get current XAUUSD price',
                    'authentication': False,
                    'rate_limit': '100/minute'
                },
                {
                    'path': '/signals',
                    'method': 'GET',
                    'description': 'Get recent trading signals',
                    'parameters': [
                        {'name': 'limit', 'type': 'int', 'default': 10}
                    ],
                    'authentication': False
                },
                {
                    'path': '/signals/latest',
                    'method': 'GET',
                    'description': 'Get latest signal',
                    'authentication': False
                },
                {
                    'path': '/positions',
                    'method': 'GET',
                    'description': 'Get open positions',
                    'parameters': [
                        {'name': 'user_id', 'type': 'int', 'required': False}
                    ],
                    'authentication': False
                },
                {
                    'path': '/trades',
                    'method': 'GET',
                    'description': 'Get trade history',
                    'parameters': [
                        {'name': 'user_id', 'type': 'int', 'required': False},
                        {'name': 'limit', 'type': 'int', 'default': 20}
                    ],
                    'authentication': False
                },
                {
                    'path': '/performance',
                    'method': 'GET',
                    'description': 'Get trading performance',
                    'parameters': [
                        {'name': 'user_id', 'type': 'int', 'required': False},
                        {'name': 'days', 'type': 'int', 'default': 30}
                    ],
                    'authentication': False
                },
                {
                    'path': '/performance/hourly',
                    'method': 'GET',
                    'description': 'Get hourly performance breakdown',
                    'authentication': False
                },
                {
                    'path': '/reports/daily',
                    'method': 'GET',
                    'description': 'Get daily report',
                    'parameters': [
                        {'name': 'date', 'type': 'string', 'format': 'YYYY-MM-DD'},
                        {'name': 'format', 'type': 'string', 'enum': ['json', 'text', 'csv']}
                    ],
                    'authentication': False
                },
                {
                    'path': '/reports/weekly',
                    'method': 'GET',
                    'description': 'Get weekly report',
                    'authentication': False
                },
                {
                    'path': '/reports/monthly',
                    'method': 'GET',
                    'description': 'Get monthly report',
                    'parameters': [
                        {'name': 'year', 'type': 'int'},
                        {'name': 'month', 'type': 'int'}
                    ],
                    'authentication': False
                },
                {
                    'path': '/webhooks',
                    'method': 'POST',
                    'description': 'Register webhook for notifications',
                    'body': {
                        'url': 'string (required)',
                        'events': 'array (optional)',
                        'secret': 'string (optional)'
                    },
                    'authentication': False
                },
                {
                    'path': '/webhooks/{webhook_id}',
                    'method': 'DELETE',
                    'description': 'Delete a webhook',
                    'authentication': False
                },
                {
                    'path': '/webhooks',
                    'method': 'GET',
                    'description': 'List all webhooks',
                    'authentication': False
                },
                {
                    'path': '/health',
                    'method': 'GET',
                    'description': 'API health check',
                    'authentication': False
                }
            ],
            'authentication': {
                'type': 'API Key',
                'header': 'X-API-Key',
                'query_param': 'api_key'
            },
            'rate_limits': {
                'default': '100 requests per minute',
                'authenticated': '1000 requests per minute'
            }
        }
        
        return web.json_response(docs)
    
    async def trigger_webhook(self, event_type: str, data: Dict[str, Any]):
        """Trigger webhooks for an event"""
        import aiohttp
        
        for webhook_id, webhook in self.webhooks.items():
            if not webhook.get('active'):
                continue
            
            if event_type not in webhook.get('events', []):
                continue
            
            try:
                payload = {
                    'event': event_type,
                    'data': data,
                    'timestamp': datetime.now(self.jakarta_tz).isoformat()
                }
                
                headers = {'Content-Type': 'application/json'}
                
                if webhook.get('secret'):
                    signature = hmac.new(
                        webhook['secret'].encode(),
                        json.dumps(payload).encode(),
                        hashlib.sha256
                    ).hexdigest()
                    headers['X-Signature'] = signature
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        webhook['url'],
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        if response.status >= 200 and response.status < 300:
                            webhook['delivery_count'] += 1
                            webhook['last_delivery'] = datetime.now(self.jakarta_tz).isoformat()
                            logger.info(f"Webhook delivered: {webhook_id}")
                        else:
                            logger.warning(f"Webhook delivery failed: {webhook_id}, status: {response.status}")
                
            except Exception as e:
                logger.error(f"Webhook delivery error for {webhook_id}: {e}")
