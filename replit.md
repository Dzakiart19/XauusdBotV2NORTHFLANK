# XAUUSD Trading Bot - Enhanced Version 2.3

## Overview
This project is a comprehensive Telegram-based trading bot for XAUUSD, optimized for Koyeb Free Tier deployment. It provides real-time trading signals with Take Profit/Stop Loss levels, offers a 3-day trial system, and supports paid subscriptions. Key capabilities include a REST API, backtesting engine, report generation, admin monitoring dashboard, and enhanced interactive menus. The bot is designed for 24/7 operation, delivering accurate signals through a multi-indicator strategy with strict validation.

## Recent Changes (December 2025)
- **Trailing Stop $0.5**: Trailing stop sekarang aktif setiap profit $0.5 dengan jarak trail 0.5 pips - auto-detect tanpa perlu env manual
- **Signal Quality Filter AKTIF**: Filter kualitas sinyal sekarang aktif - hanya sinyal dengan confidence >50%, grade C+, dan confluence >2 yang dikirim
- **Koyeb Auto-Detect**: Semua konfigurasi optimal sudah di-hardcode, hanya perlu 3 env vars: TELEGRAM_BOT_TOKEN, AUTHORIZED_USER_IDS, WEBHOOK_URL
- **Auto-Resume Monitoring**: Bot now automatically resumes monitoring for users when restarted, no need to click /monitor again
- **Pin Message on /start**: Welcome message and monitoring status are now automatically pinned in chat
- **Real-Time Price Accuracy**: Signals now use real-time market price instead of candle close price for more accurate entry points
- **Monitoring State Persistence**: Added `is_monitoring` column to User model to persist monitoring state across bot restarts
- **Fixed Critical Bug - can_create_signal()**: Added missing async method to SignalSessionManager that was causing all /getsignal and auto signal commands to fail
- **Fixed AutoOptimizer API**: Changed `should_run_optimization()` to return tuple `(bool, reason)` matching caller expectations
- **Fixed OptimizationResult**: Added `adjustments` and `status` attributes that were missing
- **Fixed MarketRegime Usage**: Updated telegram_bot.py to use correct attributes (`volatility`, `trend_direction`) instead of non-existent (`volatility_analysis`, `bias`)
- **Fixed SignalEventStore**: Updated `record_signal()` to handle both sync call and (user_id, data) signature
- **Synchronized Codebase**: Ensured all component APIs match their caller expectations after lightweight simplification
- **Fixed Command Errors**: Added missing methods to analytics.py, signal_quality_tracker.py, market_regime.py, auto_optimizer.py for /riwayat, /performa, /getsignal, /status commands
- **Stub Classes**: Lightweight implementations for SignalQualityTracker, MarketRegimeDetector, AutoOptimizer optimized for free-tier deployment
- **Fixed Owner Recognition Bug**: AUTHORIZED_USER_IDS now properly loaded via `Config.ensure_secrets_loaded()` before bot initialization
- **Governance Audit**: Added audit logging for bypass flags (BYPASS_SIGNAL_QUALITY_CHECK, AUTO_SIGNAL_REPLACEMENT_ALLOWED)

## Stub/Disabled Modules (Lightweight Deployment)
The following modules are stubs (disabled) for Koyeb free-tier optimization:
- `admin_monitor.py` - AdminDashboard, MetricsCollector, AlertManager (stubs)
- `backtesting.py` - BacktestEngine, StrategyOptimizer (stubs)
- `enhanced_menu.py` - EnhancedMenuHandler (stub)
- `report_generator.py` - ReportGenerator, ScheduledReportManager (stubs)
- `chart_generator.py` - ChartGenerator (stub, returns None)
- `sentry_integration.py` - SentryIntegrationManager (optional, requires SENTRY_DSN)

## User Preferences
- Bahasa komunikasi: **Bahasa Indonesia** (100% tidak ada bahasa Inggris)
- Data source: **Deriv WebSocket** (gratis, tanpa API key)
- Trading pair: **XAUUSD** (Gold)
- Notifikasi: **Telegram** dengan sinyal text (chart tersedia di webapp)
- Tracking: **Real-time** sampai TP/SL
- Mode: **24/7 unlimited** untuk user terdaftar
- Akurasi: Strategi multi-indicator dengan validasi ketat
- Akses Bot: **Privat** - hanya untuk user yang terdaftar di AUTHORIZED_USER_IDS atau ID_USER_PUBLIC

## Access Control Hierarchy
1. **Owner/Admin** (AUTHORIZED_USER_IDS): Full access, no trial needed, all admin commands available
2. **Premium User** (ID_USER_PUBLIC): Full access, no trial needed
3. **Trial User**: 3-day trial access, limited features after expiration

## System Architecture
The bot employs a modular architecture for scalability and maintainability, focusing on market data processing, strategy execution, and user interaction.

**Core Components & System Design:**
- **Orchestrator:** Manages overall bot operations.
- **Market Data:** Handles Deriv WebSocket connection, OHLC candle construction, and persistence.
- **Strategy:** Implements dual-mode signal detection using indicators like Twin Range Filter, Market Bias CEREBR, EMA 50, and RSI, with a weighted scoring system, Market Regime Detector, and Confluence Scoring System. Features adaptive volume filters, dynamic ADX thresholds, parallel timeframe signal generation, and smart signal cooldown.
- **Position Tracker:** Monitors real-time trade positions per user, including dynamic SL and trailing stops, with grade-based auto-closure.
- **Telegram Bot:** Manages commands, notifications, and a real-time dashboard with auto-updates.
- **Risk Manager:** Calculates lot sizes, P/L, enforces per-user risk limits, and optimizes TP/SL based on session strength.
- **Database:** PostgreSQL (with SQLite support) for persistent data, featuring auto-migration, connection pooling, and robust transaction management.
- **User Manager:** Handles user authentication, access control, and a 3-day trial system with auto-expiration.
- **Resilience:** Incorporates CircuitBreaker for WebSocket, global rate limiting for Telegram API, retry mechanisms, and advanced WebSocket recovery.
- **System Health:** Includes port conflict detection, bot instance locking, Sentry integration, health checks, and OOM graceful degradation.
- **Web Dashboard:** A real-time, per-user web dashboard with virtual scrolling, EMA overlay charts, candle data caching, live market regime tags, detailed position monitoring, and toast notifications.
- **Enhanced Signal Features:** Includes enhanced inside bar pattern detection and a breakout confirmation system.
- **Win Rate Tracking:** Provides enhanced win rate statistics per signal type, session, pattern, and streak information.
- **Deployment:** Optimized for Koyeb and Replit, supporting webhook mode, memory optimization, and self-ping for free-tier services.
- **Report Generator:** Provides daily, weekly, monthly trading reports with export options (CSV, JSON, text) and scheduled delivery.
- **Backtesting Engine:** Allows historical data simulation, comprehensive backtest results, and strategy optimization.
- **REST API Server:** Offers full REST API endpoints for external access, including status, price, signals, positions, trades, performance, and reports, with API key authentication and rate limiting.
- **Admin Monitoring Dashboard:** Displays real-time system and bot metrics, performance metrics, and automated alerting with Telegram notifications.
- **Enhanced Interactive Menu:** Provides a context-aware main menu, settings, reports, signal action, position management, user onboarding, and help menus via inline keyboards.
- **Security Module:** Implements webhook HMAC verification, enhanced secret masking, input sanitization, and rate limiting.
- **Multi-Language Support:** Supports Indonesian and English translations with user language preferences.
- **Indicator Caching:** Utilizes LRU cache with TTL for indicator calculations.

**Technical Implementations & Feature Specifications:**
- **Indicators:** EMA (5, 10, 20, 50), RSI (14), Stochastic (K=14, D=3), ATR (14), MACD (12,26,9), Volume, Twin Range Filter, Market Bias CEREBR. Includes RSI Divergence, ATR Volatility Zones, and Adaptive Smoothed RSI.
- **Risk Management:** Fixed SL ($1 per trade), dynamic TP (1.45x-2.50x R:R), max spread (5 pips), risk per trade (0.5%), and fixed lot size at 0.01.
- **Access Control:** Private bot with dual-tier access and a trial system.
- **Telegram Commands:** `/start`, `/help`, `/monitor`, `/stopmonitor`, `/getsignal`, `/status`, `/riwayat`, `/performa`, `/dashboard`, `/stopdashboard`, `/trialstatus`, `/buyaccess`, `/riset` (Admin only), `/optimize`.
- **Anti-Duplicate Protection:** Two-phase cache pattern with hash-based tracking for signal deduplication.
- **Candle Data Persistence:** Stores M1, M5, and H1 candles, including a smart H1 candle bootstrap.
- **Bot Stability:** Hang detection, health monitors, optimized Telegram polling, and a global error handler.
- **Polling Mode Keep-Alive:** Ensures 24/7 bot availability.
- **Koyeb Anti-Sleep Optimization:** Aggressive self-ping and multi-endpoint ping strategy.
- **Background Task Health Management:** Smart stuck-task detection with whitelisting for continuous tasks.
- **Timezone:** Web dashboard displays time in WIB (UTC+7).

## External Dependencies
- **Deriv WebSocket API:** Real-time XAUUSD market data.
- **Telegram Bot API (`python-telegram-bot`):** All Telegram interactions.
- **SQLAlchemy:** ORM for database interactions.
- **Pandas & NumPy:** Data manipulation and numerical operations.
- **pytz:** Timezone handling.
- **aiohttp:** Asynchronous HTTP server and client operations.
- **python-dotenv:** Environment variable management.
- **Sentry:** Advanced error tracking and monitoring.
- **psutil:** System resource monitoring.