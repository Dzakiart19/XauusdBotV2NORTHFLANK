# XAUUSD Trading Bot Pro V2

## Overview
This project is an automated Telegram-based trading bot for XAUUSD, providing real-time signals, automatic position tracking, and trade outcome notifications. It aims to deliver 24/7 unlimited signals, robust risk management, and performance tracking. Key capabilities include advanced chart generation with technical indicators, a refined dual-mode (Auto/Manual) trading strategy utilizing advanced filtering, and a Trend-Plus-Pullback approach. The bot's vision is to be a professional, informative, and accessible trading assistant for XAUUSD, with a strong focus on private access control and enhanced precision.

## Recent Changes (December 2025)
- **Database**: Migrated to PostgreSQL with BIGINT support for Telegram user IDs
- **API Dashboard Stats Query Fix**: Fixed SQL query to use correct column names (`actual_pl` instead of `pnl`, `signal_time` instead of `created_at`, and `status = 'CLOSED'` uppercase)
- **Trade History Sync**: Verified - trades properly save with `exit_price`, `actual_pl`, `close_time`, and `result` when positions are closed
- **Web Dashboard**: Real-time sync confirmed working with live price data, active positions, and market regime display
- **Position Tracking**: Full lifecycle working - open, monitor with dynamic SL/trailing stop, and close with P/L calculation

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
- **Strategy:** Implements dual-mode signal detection using indicators like Twin Range Filter, Market Bias CEREBR, EMA 50, and RSI, with a weighted scoring system and an aggressive scalping system featuring a Market Regime Detector and Confluence Scoring System.
- **Position Tracker:** Monitors real-time trade positions per user, including dynamic SL settings and trailing stops.
- **Telegram Bot:** Manages command handling, notifications, and features a real-time dashboard with auto-updates.
- **Chart Generator:** Creates professional charts with integrated technical indicators (EMA, RSI, Stochastic, TRF, CEREBR).
- **Risk Manager:** Calculates lot sizes, P/L, and enforces per-user risk limits (fixed SL, dynamic TP, signal cooldown).
- **Database:** SQLite (with PostgreSQL support) for persistent data, featuring auto-migration, connection pooling, and robust transaction management.
- **User Manager:** Handles user authentication, access control, and a 3-day trial system with auto-expiration.
- **Resilience:** Incorporates CircuitBreaker for WebSocket, global rate limiting for Telegram API, retry mechanisms, and advanced WebSocket recovery.
- **System Health:** Includes port conflict detection, bot instance locking, Sentry integration, health checks, and OOM graceful degradation.
- **Thread Safety:** Utilizes `asyncio.Lock` for critical operations.

**Technical Implementations & Feature Specifications:**
- **Indicators:** EMA (5, 10, 20, 50), RSI (14), Stochastic (K=14, D=3), ATR (14), MACD (12,26,9), Volume, Twin Range Filter, Market Bias CEREBR. Features include RSI Divergence Detection, ATR Volatility Zones, Adaptive Smoothed RSI, and Hull Moving Average Smoothing.
- **Risk Management:** Fixed SL ($1 per trade), dynamic TP (1.45x-2.50x R:R), max spread (5 pips), risk per trade (0.5%). Lot size fixed at 0.01.
- **Access Control:** Private bot with dual-tier access and a trial system.
- **Telegram Commands:** Comprehensive set of admin and user commands for status, analytics, signal retrieval, performance tracking, and dashboard interaction.
- **Anti-Duplicate Protection:** Two-phase cache pattern with hash-based tracking and thread-safe locking for signal deduplication.
- **Candle Data Persistence:** Stores M1, M5, and H1 candles in the database with immediate H1 save on candle close and partial restore capabilities.
- **Bot Stability:** Includes hang detection, health monitors, optimized Telegram polling, and a global error handler for prolonged stability.
- **Chart Generation:** Uses `mplfinance` and `matplotlib` for multi-panel charts, configured for headless operation.
- **Multi-User Support:** Implemented for per-user position tracking and risk management.
- **Deployment:** Optimized for Koyeb and Replit, featuring an HTTP server for health checks and a Telegram Web App Dashboard for real-time data display.
- **Performance Optimization:** Unlimited mode features no global signal cooldown, no tick throttling, and optimized notifications.
- **Logging & Error Handling:** Rate-limited logging, log rotation, type-safe indicator computations, and comprehensive exception handling.
- **Task Management:** Centralized task registry with shielded cancellation for graceful shutdown.

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