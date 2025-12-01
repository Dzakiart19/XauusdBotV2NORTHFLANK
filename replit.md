# XAUUSD Trading Bot Pro V2

## Overview
This project is an automated Telegram-based trading bot for XAUUSD, designed to provide real-time signals, automatic position tracking, and trade outcome notifications. It offers 24/7 unlimited signals, robust risk management, and performance tracking via a database. Key features include advanced chart generation with technical indicators, a refined dual-mode (Auto/Manual) trading strategy utilizing advanced filtering with Twin Range Filter (TRF) and Market Bias CEREBR, and a Trend-Plus-Pullback approach for enhanced precision. The bot aims to be a professional, informative, and accessible trading assistant for XAUUSD, with a focus on private access control.

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
The bot features a modular architecture for scalability and maintainability.

**Core Components & System Design:**
- **Orchestrator:** Manages bot operations.
- **Market Data:** Handles Deriv WebSocket connection, OHLC candle construction, and persistence.
- **Strategy:** Implements dual-mode signal detection (Auto/Manual) using indicators like Twin Range Filter, Market Bias CEREBR, EMA 50, and RSI, with a weighted scoring system.
- **Position Tracker:** Monitors real-time trade positions per user.
- **Telegram Bot:** Manages command handling and notifications.
- **Chart Generator:** Creates professional charts with integrated technical indicators.
- **Risk Manager:** Calculates lot sizes, P/L, and enforces per-user risk limits (fixed SL, dynamic TP, signal cooldown).
- **Database:** SQLite (with PostgreSQL support) for persistent data, featuring auto-migration, connection pooling, and robust transaction management.
- **User Manager:** Handles user authentication and access control.
- **Resilience:** Incorporates CircuitBreaker for WebSocket, global rate limiting for Telegram API, retry mechanisms, and advanced WebSocket recovery.
- **System Health:** Includes port conflict detection, bot instance locking, Sentry integration, health checks, and OOM graceful degradation.
- **Thread Safety:** Utilizes `asyncio.Lock` for position tracking, signal session management, and atomic command execution.

**Aggressive Scalping System (NEW):**
- **Market Regime Detector:** ADX-based trend detection, ATR volatility analysis, S/R proximity, breakout detection. Returns regime type (strong_trend, moderate_trend, range_bound, breakout, high_volatility) dengan bias (BUY/SELL/NEUTRAL).
- **Confluence Scoring System:** 2-3-4+ confluence levels dengan adaptive weighting per market regime. SCALP (2 conf, 15-30 pips), SHORT_TERM (3 conf, 30-50 pips), OPTIMAL (4+ conf, 50+ pips).
- **4 Signal Rules:** M1_SCALP (20-50 signals/day), M5_SWING (5-15/day), SR_REVERSION (3-8/day, ranging), BREAKOUT (2-5/day).
- **Dynamic Risk Calculator:** Daily exposure threshold, max concurrent positions (3-5), partial exit strategy (40%-35%-25%), trailing stop.
- **Signal Quality Tracker:** Real-time statistics per rule/regime/hour, hit rate tracking, alert system untuk penurunan performa.
- **Auto-Optimization Engine:** Dynamic adjustment rules, gradual parameter updates, safety guards, rollback mechanism.

**Advanced Technical Features (Nov 2025):**
- **Aggressive Tuning Modes:** 4 tuning levels (CONSERVATIVE, MODERATE, AGGRESSIVE, ULTRA_AGGRESSIVE) with different thresholds and adjustment multipliers.
- **RSI Divergence Detection:** Detects regular and hidden bullish/bearish divergences using swing pivot analysis with min_pivot_distance filtering.
- **ATR Volatility Zones:** Classifies market into 5 zones (EXTREME_LOW, LOW, NORMAL, HIGH, EXTREME_HIGH) with recommended SL/TP multipliers.
- **Adaptive Smoothed RSI:** RSI with volatility-adaptive smoothing periods.
- **Hull Moving Average Smoothing:** Enhanced smoothing options (EMA, SMA, WMA, Hull) for indicator series.
- **Emergency Mode Optimization:** Force adjustments when accuracy drops below 30%, bypassing min_signals requirement.
- **New Telegram Commands:** /analyze (technical analysis), /backtest (signal quality statistics).

**Symbol Configuration Update (Nov 30, 2025):**
- SYMBOL_FALLBACKS disederhanakan menjadi hanya ["XAUUSD"] - menghapus WLDXAU dan symbol fallback lainnya
- Default symbol diubah dari "WLDXAU" ke "XAUUSD" untuk koneksi yang lebih bersih
- Mengurangi error log dari subscription attempts yang gagal ke symbol yang tidak perlu
- Catatan: Saat market tutup (weekend), Deriv API akan mengembalikan error "Symbol is invalid" - ini adalah perilaku normal

**Position Tracking Verification (Dec 1, 2025):**
- ✅ Bug "Position not closing on TP/SL hit" sudah DIPERBAIKI dan terverifikasi
- ✅ Trailing stop activation berfungsi dengan baik (lock-in profit)
- ✅ TP/SL trigger detection akurat dengan logging lengkap
- ✅ Position close notification terkirim ke Telegram
- ✅ Signal session management bekerja (create, end, cleanup)
- ✅ Dashboard update dan cleanup berfungsi
- ✅ Chart cleanup otomatis setelah session end
- Catatan: Position monitoring interval 5 detik (FREE_TIER_MODE) dengan tick-based real-time monitoring untuk deteksi TP/SL

**Signal Accuracy Improvements (Dec 1, 2025):**
Perbaikan untuk meningkatkan akurasi sinyal dari ~50% ke target 80-90%:

*Regime Alignment Check:*
- ✅ Tambah method `_check_regime_alignment()` di signal_rules.py dan strategy.py
- ✅ Blok sinyal BUY ketika market bias SELL (dengan confidence >= 0.75-0.80)
- ✅ Blok sinyal SELL ketika market bias BUY (dengan confidence >= 0.75-0.80)
- ✅ Izinkan sinyal dengan peringatan jika regime unavailable atau low confidence

*Quality Filtering:*
- ✅ Tambah method `_validate_signal_quality()` dengan soft filtering
- ✅ Blok sinyal dengan confidence < 0.70 atau Grade D
- ✅ Warning untuk sinyal dengan confluence < 6.0 atau grade C

*Parameter Updates (signal_rules.py):*
- MIN_WEIGHTED_CONFLUENCE_SCORE: 4.0 → 6.0
- M1_MIN_CONFLUENCE: 2 → 4
- M5_MIN_CONFLUENCE: 3 → 4
- REGIME_MULTIPLIERS diperketat (weak_trend: 0.7→0.5, range_bound: 0.9→0.6)
- Blok M1_SCALP di range_bound, high_volatility, dan unknown regime

*Dynamic SL Settings (config.py):*
- DYNAMIC_SL_LOSS_THRESHOLD: $1.0 → $3.0 (mencegah SL terlalu ketat)
- BREAKEVEN_PROFIT_THRESHOLD: $0.5 → $2.0
- TRAILING_STOP_PROFIT_THRESHOLD: $1.0 → $2.5
- TRAILING_STOP_DISTANCE_PIPS: 3.0 → 5.0
- SIGNAL_SCORE_THRESHOLD_AUTO: 55 → 75

**Real-Time Telegram Dashboard (Dec 1, 2025):**
- ✅ Dashboard real-time yang ditampilkan langsung di chat Telegram
- ✅ Auto-update setiap **5 detik** dengan MD5 hash untuk deteksi perubahan
- ✅ Commands: `/dashboard` (start), `/stopdashboard` (stop), `/refresh` (manual refresh)
- Panel yang ditampilkan:
  - 💰 **HARGA XAUUSD (REAL-TIME)**: Status LIVE indicator, Mid Price, Bid/Ask, Spread, 24h Range, Change %
  - 📊 **MARKET REGIME**: Tren, Volatilitas, Bias, Confidence
  - 📡 **SINYAL TERAKHIR**: Tipe, Entry price, waktu
  - 📈 **POSISI AKTIF**: Entry, SL, TP, Unrealized P/L real-time
  - 📉 **STATISTIK**: Win Rate, Total P/L, Sinyal Hari Ini
- Fitur: Rate limiting, error handling, thread-safe dengan asyncio.Lock
- Data harga diambil langsung dari WebSocket tick (bukan candle)
- Timezone: WIB (Asia/Jakarta)

**Multi-Platform Deployment Support (Dec 1, 2025):**
- ✅ **Koyeb**: Sudah didukung (lihat DEPLOYMENT_KOYEB.md)
- ✅ **Northflank**: Baru ditambahkan (lihat DEPLOYMENT_NORTHFLANK.md)
- Kedua platform support auto-redeploy dari GitHub
- Northflank: Gratis tanpa kartu kredit, 2 services, always-on 24/7
- File konfigurasi: Dockerfile, Procfile, northflank.json

**UI/UX Decisions:**
- Telegram serves as the primary user interface.
- Signal messages are enriched with icons, source labels, and confidence reasons.
- Charts display candlesticks, volume, EMA, RSI, Stochastic, TRF bands, and CEREBR in a multi-panel layout.
- Exit notifications (WIN/LOSE) are text-only, while initial signal activations include photo charts.
- All timestamps are displayed in WIB (Asia/Jakarta) timezone.

**Technical Implementations & Feature Specifications:**
- **Indicators:** EMA (5, 10, 20, 50), RSI (14), Stochastic (K=14, D=3), ATR (14), MACD (12,26,9), Volume, Twin Range Filter, Market Bias CEREBR.
- **Risk Management:** Fixed SL ($1 per trade), dynamic TP (1.45x-2.50x R:R), max spread (5 pips), risk per trade (0.5%). Includes dynamic SL tightening and trailing stop activation. Lot size is fixed at 0.01.
- **Access Control:** Private bot with dual-tier access.
- **Commands:** Admin commands (`/riset`, `/status`, `/tasks`, `/analytics`, `/systemhealth`) and User commands (`/start`, `/help`, `/monitor`, `/getsignal`, `/status`, `/riwayat`, `/performa`, `/regime`, `/optimize`, `/rules`, `/dashboard`, `/stopdashboard`, `/refresh`).
- **Anti-Duplicate Protection:** Employs a two-phase cache pattern (pending/confirmed status, hash-based tracking, thread-safe locking, TTL-backed signal cache with async cleanup) for race-condition-safe signal deduplication and anti-spam.
- **Candle Data Persistence:** Stores M1, M5, and H1 candles in the database with immediate H1 save on candle close. Supports partial candle restore on restart.
- **H1 Historical Loading (Nov 28, 2025):** H1 timeframe dimuat lengkap saat startup dengan immediate persistence, pending queue mechanism, dan thread-safe session management.
- **Bot Stability Improvements (Nov 28, 2025):** Hang detection di OHLCBuilder (detect_hang, recover_from_hang), builder health monitor task, long-running task health check, optimized Telegram polling (120s timeout), dan global error handler untuk stabilitas 12+ jam.
- **Chart Generation:** Uses `mplfinance` and `matplotlib` for multi-panel charts, configured for headless operation.
- **Multi-User Support:** Implements per-user position tracking and risk management.
- **Deployment:** Optimized for Koyeb and Replit, featuring an HTTP server for health checks and webhooks, and `FREE_TIER_MODE` for resource efficiency.
- **Performance Optimization:** Unlimited mode features no global signal cooldown, no tick throttling, early exit for position monitoring, optimized Telegram timeout handling, and fast text-only exit notifications.
- **Logging & Error Handling:** Rate-limited logging, log rotation, type-safe indicator computations, and comprehensive exception handling.
- **Task Management:** Centralized task registry with shielded cancellation for graceful shutdown and background task callbacks.

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
```