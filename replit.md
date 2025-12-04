# XAUUSD Trading Bot Lite

## Overview
This project is a lightweight Telegram-based trading bot for XAUUSD, optimized for Koyeb Free Tier deployment. It delivers real-time trading signals with Take Profit/Stop Loss levels, offers a 3-day trial system for new users, and includes a mechanism for paid subscriptions. The bot focuses on core functionality: signal delivery without heavy chart generation (charts are available in a separate web application).

## User Preferences
- Bahasa komunikasi: **Bahasa Indonesia** (100% tidak ada bahasa Inggris)
- Data source: **Deriv WebSocket** (gratis, tanpa API key)
- Trading pair: **XAUUSD** (Gold)
- Notifikasi: **Telegram** dengan sinyal text (chart tersedia di webapp)
- Tracking: **Real-time** sampai TP/SL
- Mode: **24/7 unlimited** untuk user terdaftar
- Akurasi: Strategi multi-indicator dengan validasi ketat
- Akses Bot: **Privat** - hanya untuk user yang terdaftar di AUTHORIZED_USER_IDS atau ID_USER_PUBLIC

## System Architecture
The bot features a modular architecture designed for scalability and maintainability, centered on components for market data processing, strategy execution, and user interaction.

**Core Components & System Design:**
- **Orchestrator:** Manages overall bot operations.
- **Market Data:** Handles Deriv WebSocket connection, OHLC candle construction, and persistence.
- **Strategy:** Implements dual-mode signal detection using indicators like Twin Range Filter, Market Bias CEREBR, EMA 50, and RSI, with a weighted scoring system, Market Regime Detector, and Confluence Scoring System. Includes adaptive volume filters, dynamic ADX thresholds, parallel timeframe signal generation, and smart signal cooldown.
- **Position Tracker:** Monitors real-time trade positions per user, including dynamic SL settings and trailing stops, with grade-based auto-closure.
- **Telegram Bot:** Manages command handling, notifications, and features a real-time dashboard with auto-updates.
- **Chart Generator:** Operates in a stub mode (no-op) to conserve memory.
- **Risk Manager:** Calculates lot sizes, P/L, enforces per-user risk limits, and optimizes TP/SL based on session strength.
- **Database:** PostgreSQL (with SQLite support) for persistent data, featuring auto-migration, connection pooling, and robust transaction management.
- **User Manager:** Handles user authentication, access control, and a 3-day trial system with auto-expiration.
- **Resilience:** Incorporates CircuitBreaker for WebSocket, global rate limiting for Telegram API, retry mechanisms, and advanced WebSocket recovery.
- **System Health:** Includes port conflict detection, bot instance locking, Sentry integration, health checks, and OOM graceful degradation.
- **Web Dashboard:** A real-time, per-user web dashboard with virtual scrolling, EMA overlay charts, candle data caching, live market regime tags, detailed position monitoring, and toast notifications.
- **Enhanced Signal Features:** Includes enhanced inside bar pattern detection and a breakout confirmation system.
- **Win Rate Tracking:** Provides enhanced win rate statistics per signal type, session, pattern, and streak information.
- **Deployment:** Optimized for Koyeb and Replit, supporting webhook mode, memory optimization, and self-ping for free-tier services.

**Technical Implementations & Feature Specifications:**
- **Indicators:** EMA (5, 10, 20, 50), RSI (14), Stochastic (K=14, D=3), ATR (14), MACD (12,26,9), Volume, Twin Range Filter, Market Bias CEREBR. Includes advanced features like RSI Divergence, ATR Volatility Zones, and Adaptive Smoothed RSI.
- **Risk Management:** Fixed SL ($1 per trade), dynamic TP (1.45x-2.50x R:R), max spread (5 pips), risk per trade (0.5%), and fixed lot size at 0.01.
- **Access Control:** Private bot with dual-tier access and a trial system, with strict per-user data isolation.
- **Telegram Commands:** 12 command yang tersedia:
  - `/start` - Memulai bot dan aktivasi user
  - `/help` - Menampilkan bantuan dan daftar command
  - `/monitor` - Mulai monitoring sinyal trading real-time
  - `/stopmonitor` - Menghentikan monitoring sinyal
  - `/getsignal` - Mendapatkan sinyal trading manual
  - `/status` - Melihat posisi aktif dan status koneksi
  - `/riwayat` - Melihat riwayat trading terakhir
  - `/performa` - Statistik performa trading (7d, 30d, all-time)
  - `/trialstatus` - Melihat status trial/akses user
  - `/buyaccess` - Informasi berlangganan premium
  - `/riset` - Reset database trading (Admin only)
  - `/optimize` - Melihat status auto-optimizer dan parameter trading
- **Anti-Duplicate Protection:** Two-phase cache pattern with hash-based tracking for signal deduplication.
- **Candle Data Persistence:** Stores M1, M5, and H1 candles, including a smart H1 candle bootstrap.
- **Bot Stability:** Hang detection, health monitors, optimized Telegram polling, and a global error handler.
- **Polling Mode Keep-Alive:** Implemented to prevent unexpected task completion and ensure 24/7 bot availability.
- **Koyeb Anti-Sleep Optimization:** Aggressive self-ping interval and multi-endpoint ping strategy with burst ping mode to prevent idle shutdown.
- **Background Task Health Management:** Smart stuck-task detection with whitelist for continuous background tasks (self_ping, health_check, memory_monitor, etc.) to prevent false-positive restarts and warning spam.
- **Timezone WIB:** Web dashboard menampilkan waktu dalam zona waktu WIB (UTC+7) untuk user Indonesia.

## Recent Changes (December 2024)
- **[TERBARU - 4 Des] Perbaikan Bug Auto-Monitor Mati Sendiri:**
  - ROOT CAUSE: Duplicate `end_session` call di `position_tracker.py` menyebabkan event handler dipanggil 2x
  - FIX: Hapus duplicate `end_session` di method `_close_position_internal` - sekarang hanya dipanggil sekali
  - Tambah dokumentasi jelas di `_resolve_session_state` bahwa cleanup session TIDAK menghentikan monitoring
  - Perbaiki `_on_session_end_handler` dengan logging detail - HANYA stop dashboard dan clear cache, TIDAK stop monitoring
  - Tambah logging detail saat monitoring loop exit untuk diagnosa kondisi yang menyebabkan exit
  - Perbaiki `_on_monitoring_task_done` dengan logging lengkap untuk debug
  - PENTING: Monitoring sekarang TETAP BERJALAN setelah posisi ditutup (session end)
- **[TERBARU] Perbaikan Unlimited Signal Trading:**
  - Signal Quality Grade minimum diubah dari 'C' ke 'D' untuk membolehkan sinyal lebih banyak
  - Ditambahkan BYPASS_SIGNAL_QUALITY_CHECK=true mode untuk menonaktifkan blocking signal quality sepenuhnya
  - Sinyal trading sekarang benar-benar unlimited tanpa blocking berdasarkan grade
- **Perbaikan Auto-Monitoring Sebelumnya:**
  - Ditambahkan heartbeat logging setiap 30 detik untuk memantau kesehatan monitoring loop
  - Ditambahkan exception handling catch-all untuk mencegah monitoring loop berhenti tiba-tiba
  - Ditambahkan health check untuk auto-restart monitoring tasks yang mati (_check_and_restart_dead_monitoring_tasks)
  - Enhanced logging untuk monitoring lifecycle (start/stop dengan detail iterasi dan reason)
- Implementasi command `/status` untuk melihat posisi aktif dan status koneksi
- Implementasi command `/optimize` untuk melihat status auto-optimizer dan parameter trading
- Auto-monitor sekarang aktif otomatis untuk SEMUA user (AUTHORIZED_USER_IDS + ID_USER_PUBLIC) saat bot restart
- Emergency brake drawdown dinonaktifkan (DRAWDOWN_EMERGENCY_BRAKE_ENABLED=false) untuk sinyal unlimited
- Perbaikan duplicate method `should_run_optimization` di auto_optimizer.py
- Optimasi ukuran log files untuk menghemat storage Koyeb (dari 2.2M ke 524K)
- Update dokumentasi command (12 command tersedia)
- Fixed stuck task detection to skip background tasks that are designed to run continuously
- Reduced "LONG TASK" warning spam for legitimate background processes
- Added WIB timezone conversion to web dashboard (formatTime/formatTimeShort functions)
- Fixed datetime serialization for health check endpoints
- Improved import handling (pytz)

## External Dependencies
- **Deriv WebSocket API:** For real-time XAUUSD market data.
- **Telegram Bot API (`python-telegram-bot`):** For all Telegram interactions.
- **SQLAlchemy:** ORM for database interactions.
- **Pandas & NumPy:** For data manipulation and numerical operations.
- **pytz:** For timezone handling.
- **aiohttp:** For asynchronous HTTP server and client operations.
- **python-dotenv:** For managing environment variables.
- **Sentry:** For advanced error tracking and monitoring.