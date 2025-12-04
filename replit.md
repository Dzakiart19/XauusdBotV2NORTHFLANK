# XAUUSD Trading Bot Pro V2

## Overview
This project is an automated Telegram-based trading bot for XAUUSD, providing real-time signals, automatic position tracking, and trade outcome notifications. It aims to deliver 24/7 unlimited signals, robust risk management, and performance tracking. Key capabilities include advanced chart generation with technical indicators, a refined dual-mode (Auto/Manual) trading strategy utilizing advanced filtering, and a Trend-Plus-Pullback approach. The bot's vision is to be a professional, informative, and accessible trading assistant for XAUUSD, with a strong focus on private access control and enhanced precision. It includes a real-time web dashboard for monitoring positions, trade history, and market regimes, with strict per-user data isolation.

## User Preferences
- Bahasa komunikasi: **Bahasa Indonesia** (100% tidak ada bahasa Inggris)
- Data source: **Deriv WebSocket** (gratis, tanpa API key)
- Trading pair: **XAUUSD** (Gold)
- Notifikasi: **Telegram** dengan foto chart + indikator
- Tracking: **Real-time** sampai TP/SL
- Mode: **24/7 unlimited** untuk user terdaftar
- Akurasi: Strategi multi-indicator dengan validasi ketat
- Chart: Menampilkan indikator EMA, RSI, Stochastic (tidak polos)
- Akses Bot: **Privat** - hanya untuk user yang terdaftar di AUTHORIZED_USER_IDS atau ID_USER_PUBLIC

## System Architecture
The bot features a modular architecture for scalability and maintainability, designed around core components for market data, strategy execution, and user interaction.

**Core Components & System Design:**
- **Orchestrator:** Manages overall bot operations.
- **Market Data:** Handles Deriv WebSocket connection, OHLC candle construction, and persistence.
- **Strategy:** Implements dual-mode signal detection using indicators like Twin Range Filter, Market Bias CEREBR, EMA 50, and RSI, with a weighted scoring system, Market Regime Detector, and Confluence Scoring System. Features adaptive volume filters, dynamic ADX thresholds, parallel timeframe signal generation, and smart signal cooldown.
- **Position Tracker:** Monitors real-time trade positions per user, including dynamic SL settings and trailing stops, with grade-based auto-closure for stale positions.
- **Telegram Bot:** Manages command handling, notifications, and features a real-time dashboard with auto-updates.
- **Chart Generator:** Creates professional charts with integrated technical indicators using `mplfinance` and `matplotlib`.
- **Risk Manager:** Calculates lot sizes, P/L, enforces per-user risk limits, and optimizes TP/SL based on session strength.
- **Database:** PostgreSQL (with SQLite support) for persistent data, featuring auto-migration, connection pooling, and robust transaction management, with BIGINT support for Telegram user IDs.
- **User Manager:** Handles user authentication, access control, and a 3-day trial system with auto-expiration.
- **Resilience:** Incorporates CircuitBreaker for WebSocket, global rate limiting for Telegram API, retry mechanisms, and advanced WebSocket recovery.
- **System Health:** Includes port conflict detection, bot instance locking, Sentry integration, health checks, and OOM graceful degradation.
- **Web Dashboard:** A real-time, per-user web dashboard with virtual scrolling, EMA overlay charts, candle data caching, live market regime tags, detailed position monitoring (P/L, TP/SL progress), signal card enhancements (TTL, win rate, confidence), and toast notifications.
- **Enhanced Signal Features:** Includes enhanced inside bar pattern detection and a breakout confirmation system.
- **Win Rate Tracking:** Provides enhanced win rate statistics per signal type, session, pattern, and streak information.
- **Deployment:** Optimized for Koyeb and Replit, supporting webhook mode, memory optimization, and self-ping for free-tier services.

**Technical Implementations & Feature Specifications:**
- **Indicators:** EMA (5, 10, 20, 50), RSI (14), Stochastic (K=14, D=3), ATR (14), MACD (12,26,9), Volume, Twin Range Filter, Market Bias CEREBR. Includes advanced features like RSI Divergence, ATR Volatility Zones, and Adaptive Smoothed RSI.
- **Risk Management:** Fixed SL ($1 per trade), dynamic TP (1.45x-2.50x R:R), max spread (5 pips), risk per trade (0.5%), and fixed lot size at 0.01.
- **Access Control:** Private bot with dual-tier access and a trial system, with strict per-user data isolation across all endpoints.
- **Telegram Commands:** Comprehensive admin and user commands, including `/winstats`.
- **Anti-Duplicate Protection:** Two-phase cache pattern with hash-based tracking for signal deduplication.
- **Candle Data Persistence:** Stores M1, M5, and H1 candles, including a smart H1 candle bootstrap.
- **Bot Stability:** Hang detection, health monitors, optimized Telegram polling, and a global error handler.

## Recent Changes (December 2025)

**Quick Health Server Pattern for Koyeb (Dec 4):**
- Added background thread-based quick health server that starts IMMEDIATELY at module load
- Runs in separate thread with its own event loop before any heavy imports
- Responds to /health, /ping, /api/health, and / endpoints instantly
- Allows Koyeb health checks to pass while bot initializes (60+ second initialization)
- Main health server takes over once fully initialized with proper thread cleanup
- Uses threading.Event for clean stop signaling and join(timeout=3.0) for cleanup

**Non-Blocking Webhook Handler (Dec 4):**
- Webhook handler now responds immediately with 200 OK to Telegram
- All update processing happens in background via asyncio.create_task
- Eliminates timeout issues during high-latency operations
- Logs minimal info before responding for quick turnaround

**Blocking Database Operations Fix (Dec 4):**
- Fixed critical issue where synchronous database operations blocked the event loop, causing 20+ second response times
- Health check handler now uses run_in_executor() with 3-second timeout for database queries
- Dashboard stats endpoint uses same async pattern for database queries
- Session lifecycle properly managed with try/finally blocks to prevent connection pool exhaustion
- Inner try/except for session acquisition prevents executor crashes from bubbling up
- All database operations now gracefully degrade with safe default values on errors

**Webhook Verification Enhancement (Dec 4):**
- Added robust webhook verification after setup with retry logic (3 attempts with exponential backoff)
- Webhook success only marked True after URL is confirmed registered with Telegram API
- Detailed logging of webhook status including URL, pending updates, max connections, and last error info
- Proper null checks for telegram_bot.app.bot before accessing webhook info
- Verification failures trigger retries with actionable warning logs

**Koyeb Deployment Environment Fix (Dec 4):**
- Fixed critical bug where environment variables not loading correctly on Koyeb
- Root cause: Config class attributes evaluated at import time before Koyeb injects env vars
- Fix: `_refresh_secrets()` now aggressively re-reads ALL env vars, not just non-empty ones
- Webhook routes now always register even in "limited mode" (prevents chicken-egg problem)
- Health check always returns HTTP 200 (allows Koyeb health probe to pass during initialization)
- Dockerfile HEALTHCHECK uses `${PORT:-8000}` for port flexibility
- Webhook handler returns 200 to Telegram even on errors (prevents retry loops)
- Updated KOYEB_ENV.md with improved troubleshooting documentation

**Webhook URL Format Fix (Dec 4):**
- Fixed Koyeb webhook not receiving Telegram updates
- Changed auto-generated webhook URL from `/webhook` to `/bot{token}` format
- This matches the standard Telegram webhook path and the aiohttp route exposed by the server
- Now config.py generates `https://{KOYEB_PUBLIC_DOMAIN}/bot{TELEGRAM_BOT_TOKEN}` when both are available
- Fallback to `/webhook` if token not available during config init

**Critical Shutdown Flag Reset Fix (Dec 4):**
- Fixed critical bug where `_is_shutting_down` flag was not reset in `initialize()` method
- This caused bot to enter infinite restart loop after first restart
- Bot polling would start, keep-alive loop would immediately exit (flag still True from previous stop())
- Main.py would detect task completion and restart, creating endless cycle
- Fix: Added `self._is_shutting_down = False` reset at start of `initialize()`
- Bot now properly restarts and maintains stable polling connection

**Polling Mode Keep-Alive Fix (Dec 4):**
- Fixed critical bug where Telegram bot task completes unexpectedly in polling mode
- Added keep-alive loop for polling mode similar to webhook mode implementation
- Loop monitors updater health every 30 seconds with consecutive error tracking
- Prevents premature task completion and ensures 24/7 bot availability

**Bot Restart Fix:**
- Fixed "Updater is already running" error in main.py restart logic
- Restart now follows proper sequence: stop() -> wait -> initialize() -> start_background_cleanup_tasks() -> run()
- Prevents Telegram Application state conflicts during auto-restart

**Per-User Isolation Tests:**
- Added 20 comprehensive unit tests in tests/test_user_isolation.py
- Tests cover SignalEventStore isolation: get_latest_signal, get_recent_signals, clear_user_signals, get_user_signal_count
- All tests validate strict per-user data separation

**Memory Monitoring Enhancement:**
- Added get_memory_stats() method to SignalEventStore
- Enhanced telemetry with cleanup_runs and average_signals_per_user tracking
- Memory stats logged after each cleanup cycle for monitoring

**Koyeb Anti-Sleep Optimization:**
- Unified Koyeb detection across config and main (KOYEB_PUBLIC_DOMAIN, KOYEB_REGION, KOYEB_SERVICE_NAME, KOYEB_APP_NAME)
- Aggressive self-ping interval: 55s for Koyeb, 240s for Replit
- Multi-endpoint ping strategy: /health, /, /api/health with retry fallback
- Ultra-light health endpoints: /api/health and /ping for fast response
- Burst ping mode: extra pings every 10 cycles for Koyeb anti-idle
- Warning log when Koyeb detected but aggressive mode disabled

## External Dependencies
- **Deriv WebSocket API:** For real-time XAUUSD market data.
- **Telegram Bot API (`python-telegram-bot`):** For all Telegram interactions.
- **SQLAlchemy:** ORM for database interactions.
- **Pandas & NumPy:** For data manipulation and numerical operations.
- **mplfinance & Matplotlib:** For generating financial charts.
- **pytz:** For timezone handling.
- **aiohttp:** For asynchronous HTTP server and client operations.
- **python-dotenv:** For managing environment variables.
- **Sentry:** For advanced error tracking and monitoring.