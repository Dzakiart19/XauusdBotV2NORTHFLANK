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