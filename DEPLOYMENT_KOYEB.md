# 🚀 Deploy Trading Bot ke Koyeb

Panduan lengkap untuk deploy XAUUSD Trading Bot ke Koyeb.

## ⚠️ PENTING: Bot Tidak Merespon?

**Jika bot tidak merespon command sama sekali:**

Bot kemungkinan besar running dalam **"limited mode"** karena environment variables belum di-set dengan benar di Koyeb.

**Cek status bot:**
1. Buka URL Koyeb service Anda: `https://<your-service>.koyeb.app/health`
2. Lihat field `"mode"`:
   - ✅ `"mode": "full"` → Bot berjalan normal, siap menerima command
   - ❌ `"mode": "limited"` → Bot TIDAK akan merespon, perlu set environment variables!

**Jika limited mode, lihat field `"missing_config"`:**
```json
{
  "mode": "limited",
  "missing_config": [
    "TELEGRAM_BOT_TOKEN",
    "AUTHORIZED_USER_IDS"
  ]
}
```

**Solusi: Set environment variables yang kurang → Restart service!**

Scroll ke bagian **"4. Environment Variables"** di bawah untuk panduan lengkap.

---

## 📋 Prerequisites

1. **Akun Koyeb** (gratis): https://www.koyeb.com/
2. **Telegram Bot Token** dari @BotFather
3. **Telegram User ID** Anda

## 🔧 Step-by-Step Deployment

### 1. Persiapan Repository

Pastikan repository Anda sudah di GitHub/GitLab dan code sudah ter-push.

### 2. Buat Service di Koyeb

1. Login ke **Koyeb Dashboard**: https://app.koyeb.com/
2. Klik **"Create Service"**
3. Pilih **"GitHub"** atau **"GitLab"** sebagai source
4. Connect dan pilih repository trading bot Anda
5. Branch: **main** atau **master**

### 3. Konfigurasi Build

Di bagian **"Build"**:

- **Build command**: (kosongkan, atau isi `pip install -r requirements.txt`)
- **Run command**: `python main.py`

### 4. Environment Variables ⚡ WAJIB

**TANPA ENVIRONMENT VARIABLES INI, BOT TIDAK AKAN MERESPON COMMAND!**

#### Cara Set Environment Variables di Koyeb:

1. Di Koyeb Dashboard, klik service trading bot Anda
2. Klik tab **"Settings"**
3. Scroll ke bagian **"Environment variables"**
4. Klik **"Add variable"**
5. Masukkan NAME dan VALUE untuk setiap variable
6. Klik **"Save"** setelah semua variable ditambahkan
7. **WAJIB: Klik "Redeploy"** untuk apply perubahan!

#### Variable WAJIB (Bot tidak akan jalan tanpa ini):

**1. TELEGRAM_BOT_TOKEN**
```
NAME:  TELEGRAM_BOT_TOKEN
VALUE: 123456789:ABCdefGHIjklMNOpqrsTUVwxyz1234567890
```
- Dapatkan dari @BotFather di Telegram
- Kirim `/newbot` ke @BotFather untuk buat bot baru
- Copy token yang diberikan (format: angka:huruf-angka)

**2. AUTHORIZED_USER_IDS** (Owner/Admin Bot)
```
NAME:  AUTHORIZED_USER_IDS
VALUE: 123456789
```
- **INI ADALAH ID PEMILIK BOT** - akan muncul sebagai "👑 Owner/Admin"
- Dapatkan user ID Telegram Anda dari @userinfobot
- Kirim pesan apa saja ke @userinfobot untuk dapatkan ID
- Jika lebih dari 1 admin, pisahkan dengan koma: `123456,789012,345678`

**3. ID_USER_PUBLIC** (User Premium - Optional)
```
NAME:  ID_USER_PUBLIC
VALUE: 111222333,444555666
```
- User dengan akses premium tanpa trial - akan muncul sebagai "✅ User Premium"
- Pisahkan dengan koma jika lebih dari 1 user

**⚠️ PENTING untuk Developer Open Source:**
```
┌─────────────────────────────────────────────────────────────────┐
│ JANGAN GUNAKAN ID ORANG LAIN!                                   │
│                                                                 │
│ Saat deploy bot ini, GANTI AUTHORIZED_USER_IDS dengan           │
│ Telegram ID ANDA SENDIRI agar Anda dikenali sebagai Owner.      │
│                                                                 │
│ Cara dapat Telegram ID:                                         │
│ 1. Buka Telegram → cari @userinfobot                            │
│ 2. Kirim /start → bot akan balas dengan ID Anda                 │
│ 3. Copy ID tersebut ke AUTHORIZED_USER_IDS                      │
└─────────────────────────────────────────────────────────────────┘
```

**Perbedaan Role:**
| Role | Environment Variable | Status di Bot |
|------|---------------------|---------------|
| Owner/Admin | AUTHORIZED_USER_IDS | 👑 Owner/Admin |
| User Premium | ID_USER_PUBLIC | ✅ User Premium |
| User Biasa | (tidak perlu) | 🎁 Trial User |

**WEBHOOK MODE (Recommended untuk Koyeb):**
```
TELEGRAM_WEBHOOK_MODE=true
WEBHOOK_URL=https://<your-koyeb-domain>/webhook
```

**Contoh:**
```
TELEGRAM_WEBHOOK_MODE=true
WEBHOOK_URL=https://united-zorana-dzeckyete-7e3e7caa.koyeb.app/webhook
```

**Catatan Webhook:**
- ✅ Webhook mode lebih efisien dan reliable untuk deployment cloud
- ✅ Format WEBHOOK_URL harus berakhir dengan `/webhook` (PENTING!)
- ✅ Pastikan `TELEGRAM_WEBHOOK_MODE=true` untuk enable webhook
- ✅ Server otomatis listen ke PORT dari environment Koyeb
- ✅ Healthcheck endpoint: `/health` (port 8080)
- ✅ Webhook endpoint: `/webhook` (auto-registered)

**Trading Hours (Optional):**
```
TRADING_HOURS_START=0          # Jam mulai trading (0 = tengah malam)
TRADING_HOURS_END=23           # Jam akhir trading (23 = 23:59 - HARUS 0-23, bukan 24!)
FRIDAY_CUTOFF_HOUR=20          # Stop trading Jumat jam 20:00+
```

**Unlimited Mode (Optional - untuk unlimited signals):**
```
SIGNAL_COOLDOWN_SECONDS=0      # Tidak ada cooldown antar sinyal
MAX_TRADES_PER_DAY=0           # Unlimited jumlah trades per hari
DAILY_LOSS_PERCENT=0.0         # Unlimited, tidak ada batas kerugian harian
```
**Catatan:** Time filter (weekday/weekend/trading hours) tetap aktif untuk keamanan!

**Indicators & Risk (Optional - sudah ada default yang bagus):**
```
EMA_PERIODS=5,10,20
RSI_PERIOD=14
STOCH_K_PERIOD=14
ATR_PERIOD=14
MAX_SPREAD_PIPS=15.0
SL_ATR_MULTIPLIER=1.2
TP_RR_RATIO=1.5
DEFAULT_SL_PIPS=20.0
DEFAULT_TP_PIPS=30.0
FIXED_RISK_AMOUNT=1.0
```

### 5. Instance Configuration

- **Instance type**: Pilih **"Nano"** atau **"Micro"** (gratis tier cukup)
- **Regions**: Pilih region terdekat (e.g., Frankfurt, Singapore)
- **Scaling**: 1 instance (cukup untuk bot)

### 6. Health Check (Wajib)

- **Health check port**: `8080`
- **Health check path**: `/health`
- **Health check protocol**: HTTP

**Status yang dicek:**
- ✅ Market data connection
- ✅ Database status
- ✅ Telegram bot status
- ✅ Task scheduler status
- ✅ Webhook mode status

### 7. Deploy!

1. Klik **"Deploy"**
2. Tunggu 2-5 menit untuk build & deploy
3. Status akan berubah jadi **"Healthy"** kalau berhasil

## ⚠️ CATATAN PENTING: TRADING_HOURS_END

**JANGAN GUNAKAN NILAI 24!**
- ❌ `TRADING_HOURS_END=24` → ERROR! (hanya accept 0-23)
- ✅ `TRADING_HOURS_END=23` → BENAR (hampir 24/7, sampai 23:59)

Config hanya accept jam 0-23. Untuk trading 24/7, gunakan:
```
TRADING_HOURS_START=0
TRADING_HOURS_END=23
```

---

## ✅ Verifikasi Deployment

### Test Bot di Telegram

1. Buka Telegram, cari bot Anda
2. Ketik `/start` - harus ada respons
3. Ketik `/getsignal` - harus kirim sinyal trading dengan chart
4. Ketik `/monitor` - mulai monitoring otomatis
5. Ketik `/settings` - lihat konfigurasi

### Cek Logs di Koyeb

1. Buka service Anda di Koyeb Dashboard
2. Tab **"Logs"**
3. Harus lihat:
   ```
   ✅ Connected to Deriv WebSocket
   📡 Subscribed to frxXAUUSD
   Telegram bot is running!
   BOT IS NOW RUNNING
   ```

## 🔍 Troubleshooting

### ❌ Bot Tidak Merespon Command Sama Sekali

**Gejala**: Bot tidak reply command `/start`, `/help`, atau command lainnya, meskipun logs di Koyeb "aman-aman saja" (tidak ada error).

**Root Cause**: Bot running dalam **limited mode** karena environment variables tidak di-set.

**Cara Diagnosa:**
1. Buka browser, akses: `https://<your-service>.koyeb.app/health`
2. Cek field `"mode"` di response JSON:
   ```json
   {
     "status": "limited",
     "mode": "limited",
     "config_valid": false,
     "missing_config": ["TELEGRAM_BOT_TOKEN", "AUTHORIZED_USER_IDS"],
     "message": "Bot running in limited mode - set missing environment variables"
   }
   ```

**Solusi Step-by-Step:**

1. **Set Environment Variables di Koyeb:**
   - Klik service Anda → Tab "Settings"
   - Scroll ke "Environment variables"
   - Tambahkan variable `TELEGRAM_BOT_TOKEN` dan `AUTHORIZED_USER_IDS`
   - Lihat section **"4. Environment Variables"** di atas untuk detail lengkap

2. **Restart Service:**
   - Klik tombol **"Redeploy"** di Koyeb Dashboard
   - Tunggu 2-3 menit hingga status jadi "Healthy"

3. **Verify Bot Sudah Full Mode:**
   - Akses lagi: `https://<your-service>.koyeb.app/health`
   - Pastikan `"mode": "full"` dan `"config_valid": true`
   - HTTP Status Code harus **200** (bukan 503)

4. **Test di Telegram:**
   - Kirim `/start` ke bot Anda
   - Bot harus balas dengan welcome message
   - Jika masih belum ada balasan, cek logs di Koyeb untuk error

**Cek Logs di Koyeb:**
```
# Logs yang BENAR (full mode):
✅ All components initialized successfully
✅ Webhook route registered: /bot123456789:ABC...
✅ Telegram bot is running!
✅ BOT IS NOW RUNNING

# Logs yang SALAH (limited mode):
⚠️ Configuration validation issues: TELEGRAM_BOT_TOKEN is required
⚠️ Bot will start in limited mode
⚠️ RUNNING IN LIMITED MODE
⚠️ Webhook route not registered - limited mode
```

**Webhook Logging Enhancement:**
Mulai v2.9, bot akan log setiap webhook yang diterima:
```
📨 Webhook received: update_id=123456, user=789012, message='/start'
🔄 Processing webhook update 123456 from user 789012: '/start'
✅ Webhook processed successfully: update_id=123456
```

Jika Anda TIDAK melihat logs ini saat kirim command, berarti:
- Bot dalam limited mode (environment variables kurang), ATAU
- Webhook tidak setup dengan benar (Telegram tidak bisa kirim updates ke bot)

### Webhook Mode Tidak Aktif

**Problem**: Logs menunjukkan "Webhook mode: FALSE" di health check
**Solusi:**
1. Pastikan environment variable `TELEGRAM_WEBHOOK_MODE=true` sudah diset
2. Set `WEBHOOK_URL` atau `KOYEB_PUBLIC_DOMAIN` untuk auto-generate URL
3. Restart service di Koyeb Dashboard
4. Check logs untuk konfirmasi: "✅ Webhook configured successfully!"
5. Test dengan mengirim pesan ke bot di Telegram

**Verifikasi webhook aktif:**
```
curl https://<your-koyeb-domain>/health
```
Response harus menunjukkan `"webhook_mode": true`

### 🔧 Fix Webhook Manual (Jika Bot Tidak Merespon)

Jika bot sudah deployed tapi tidak merespon command, gunakan script `fix_webhook.py`:

```bash
# Cek status webhook saat ini
python fix_webhook.py --status

# Setup webhook manual
python fix_webhook.py https://your-app.koyeb.app/webhook

# Hapus webhook (untuk switch ke polling mode)
python fix_webhook.py --delete
```

**Catatan:**
- Script ini akan register webhook URL ke Telegram API
- Pastikan `TELEGRAM_BOT_TOKEN` sudah diset di environment
- Jalankan di environment dimana bot token tersedia

### Docker Build Failed - libgl1-mesa-glx Error

**Problem**: Error saat build Docker - "Package 'libgl1-mesa-glx' has no installation candidate"
**Solusi**: ✅ **SUDAH DIPERBAIKI!**
- Dockerfile sudah diupdate untuk menggunakan `libgl1` (Debian Trixie compatible)
- Package dependencies sudah dioptimalkan
- Build sekarang lebih cepat dan lebih kecil

### Bot tidak response di Telegram

**Problem**: Bot tidak merespons command
**Solusi**:
1. Cek Koyeb logs untuk error
2. Pastikan `TELEGRAM_BOT_TOKEN` benar
3. Pastikan `AUTHORIZED_USER_IDS` sesuai dengan user ID Anda

### Database Error

**Problem**: "database is locked" atau error database
**Solusi**:
1. Koyeb menggunakan ephemeral storage
2. Data akan hilang saat redeploy
3. Untuk persistent data, gunakan PostgreSQL external (optional)

### WebSocket Connection Failed

**Problem**: "Failed to connect to Deriv WebSocket"
**Solusi**:
1. Biasanya temporary, tunggu beberapa detik
2. Cek internet connection Koyeb instance
3. Bot auto-reconnect setiap 3 detik

### Health Check Failed

**Problem**: Service status "Unhealthy"
**Solusi**:
1. Pastikan health check port `8080` sudah benar
2. Pastikan bot sudah fully started (tunggu 30 detik)
3. Check logs untuk error saat startup

## 📊 Commands Tersedia

```
/start       - Tampilkan menu utama
/help        - Bantuan lengkap
/monitor     - Mulai monitoring sinyal otomatis
/stopmonitor - Stop monitoring
/getsignal   - Generate sinyal manual sekarang
/riwayat     - Lihat riwayat trading
/performa    - Statistik performa
/settings    - Lihat konfigurasi bot
```

## 🎯 Fitur Bot (UPDATED v2.4)

- ✅ **Webhook Mode** - Telegram webhook untuk Koyeb deployment
- ✅ **Auto-detect domain** - Otomatis detect Koyeb/Replit domain
- ✅ **Real-time data** dari Deriv (XAUUSD/Gold)
- ✅ **Zero API key** required untuk market data
- ✅ **Dual signal modes**: 🤖 Auto (strict) & 👤 Manual (relaxed)
- ✅ **Enhanced strategy**: RSI crossover + EMA trend + volume confirmation
- ✅ **No signal spam**: Pemisahan jelas auto vs manual
- ✅ **Chart visualization** setiap sinyal
- ✅ **Position tracking** hingga TP/SL tercapai
- ✅ **Risk management** dengan cooldown & daily loss limit
- ✅ **24/7 monitoring** tanpa henti
- ✅ **Signal source tracking**: Setiap sinyal ter-label sumbernya
- ✅ **Premium subscription**: Weekly & Monthly packages
- ✅ **Admin commands**: User management & database control

## 🆓 Optimasi untuk Koyeb Free Tier

### Resource Limits Free Tier:
- ✅ 1 web service gratis
- ✅ 24/7 uptime
- ✅ 512MB RAM, 0.1 vCPU (shared)
- ⚠️ Ephemeral storage (data hilang saat redeploy)

### ⚡ Bot Sudah Dioptimalkan untuk Free Tier:

**1. Automatic Free Tier Mode**
Bot otomatis mendeteksi free tier dan optimize resource usage:
```bash
# Set di Koyeb Environment Variables (OPSIONAL - default sudah TRUE)
FREE_TIER_MODE=true
```

**Optimasi yang diterapkan:**
- ✅ Chart generation dengan ThreadPoolExecutor (max_workers=1)
- ✅ Signal detection interval: 3 detik (optimal balance speed vs CPU)
- ✅ Dashboard update: 6 detik (smooth real-time tanpa overhead)
- ✅ Global signal cooldown: 1 detik (minimal throttling)
- ✅ Tick log sampling: 1 dari 30 ticks di-log (reduce I/O)
- ✅ Database candle persistence: Instant startup, no API fetch
- ✅ Webhook mode: Lebih efisien dari polling
- ✅ **Position monitoring: 5 detik** (lebih agresif vs 10 detik normal)
- ✅ **HTTP fallback untuk harga** (backup saat WebSocket tidak stabil)

**2. Reduce Logging (Optional)**
Untuk mengurangi I/O operations di free tier:
```bash
# Set di Koyeb Environment Variables
TICK_LOG_SAMPLE_RATE=50  # Log hanya 1 dari 50 ticks (default: 30)
```

**3. PostgreSQL External (Optional - Recommended)**
Free tier Koyeb menggunakan ephemeral storage. Untuk persistent data:
```bash
# Gunakan PostgreSQL external (Neon, Supabase, dll)
DATABASE_URL=postgresql://user:pass@host:5432/dbname
```

Bot otomatis detect dan migrate ke PostgreSQL!

**4. Monitoring Resource Usage**
Cek health endpoint untuk monitor performa:
```bash
curl https://<your-app>.koyeb.app/health
```

Response akan menunjukkan:
- `market_connected`: Status koneksi Deriv WebSocket
- `telegram_running`: Status Telegram bot
- `mode`: "full" atau "limited"
- `webhook_mode`: true/false

### 🎯 Performance Tips:

**DO:**
- ✅ Gunakan webhook mode (lebih efisien dari polling)
- ✅ Biarkan `FREE_TIER_MODE=true` (default)
- ✅ Set `TICK_LOG_SAMPLE_RATE=50` untuk reduce logging
- ✅ Gunakan PostgreSQL external untuk data persistence
- ✅ Monitor logs di Koyeb dashboard untuk catch issues early

**DON'T:**
- ❌ Disable FREE_TIER_MODE di free tier (akan consume banyak resource)
- ❌ Set signal detection interval < 3 detik (CPU intensive)
- ❌ Set dashboard update < 5 detik (Telegram API intensive)
- ❌ Enable debug logging di production (I/O overhead)

### 📊 Expected Performance:

Dengan optimasi ini, bot dapat handle:
- ✅ 1-3 concurrent users tanpa lag
- ✅ Signal detection dalam 3-6 detik
- ✅ Dashboard update setiap 6 detik
- ✅ 24/7 uptime di Koyeb free tier
- ✅ <300MB RAM usage (masih dalam 512MB limit)
- ✅ <5% CPU usage average (shared vCPU)

**Bot sudah production-ready untuk Koyeb Free Tier!** 🚀

## 🔄 Update Bot

Untuk update bot setelah deployment:

1. Push code baru ke GitHub/GitLab
2. Koyeb auto-redeploy (jika auto-deploy enabled)
3. Atau manual redeploy di Dashboard

## 📞 Support

Jika ada masalah:
1. Cek Koyeb logs dulu
2. Cek Telegram bot dengan `/settings`
3. Restart service di Koyeb Dashboard

---

**Happy Trading! 🚀📈**
