# Environment Variables untuk Koyeb Deployment

## WAJIB DIISI

Ini adalah environment variables yang **HARUS** diisi agar bot berfungsi di Koyeb:

### 1. TELEGRAM_BOT_TOKEN
Token bot dari @BotFather
```
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrSTUvwxYZ
```

### 2. AUTHORIZED_USER_IDS  
Telegram user ID yang diizinkan (dapatkan dari @userinfobot)
```
AUTHORIZED_USER_IDS=123456789
```
Untuk multiple user: `AUTHORIZED_USER_IDS=123456789,987654321`

### 3. TELEGRAM_WEBHOOK_MODE
WAJIB `true` untuk Koyeb (polling tidak bekerja di Koyeb)
```
TELEGRAM_WEBHOOK_MODE=true
```

### 4. KOYEB_PUBLIC_DOMAIN atau WEBHOOK_URL
Pilih salah satu:

**Option A - KOYEB_PUBLIC_DOMAIN (Recommended)**
```
KOYEB_PUBLIC_DOMAIN=nama-app-xxx.koyeb.app
```
Dapatkan dari Koyeb dashboard setelah deploy.

**Option B - WEBHOOK_URL (Manual)**
```
WEBHOOK_URL=https://nama-app-xxx.koyeb.app/webhook
```

### 5. PORT
Port yang digunakan Koyeb (biasanya 8000)
```
PORT=8000
```

## RECOMMENDED (Untuk Free Tier)

```
FREE_TIER_MODE=true
SELF_PING_ENABLED=true
SELF_PING_INTERVAL=240
MEMORY_WARNING_THRESHOLD_MB=400
MEMORY_CRITICAL_THRESHOLD_MB=450
```

## DATABASE_URL (WAJIB untuk Koyeb!)

⚠️ **PENTING**: Koyeb filesystem bersifat ephemeral (data hilang saat restart). 
Anda HARUS menggunakan PostgreSQL eksternal.

### Cara Mendapatkan DATABASE_URL Gratis (Neon):

1. **Buka https://neon.tech** dan buat akun (gratis)
2. **Buat project baru** (nama bebas, misal: `trading-bot`)
3. **Copy Connection String** dari dashboard:
   ```
   DATABASE_URL=postgresql://user:password@ep-xxx.neon.tech/neondb?sslmode=require
   ```

### Alternatif PostgreSQL Gratis:
- **Neon** (https://neon.tech) - Recommended, serverless
- **Supabase** (https://supabase.com) - All-in-one platform
- **Railway** (https://railway.app) - Kuota terbatas
- **ElephantSQL** (https://elephantsql.com) - Shared instance

## OPSIONAL

```
DRY_RUN=false
ENVIRONMENT=production
```

## Contoh Lengkap untuk Koyeb

```env
# ============================================
# WAJIB (Bot tidak akan jalan tanpa ini)
# ============================================
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrSTUvwxYZ
AUTHORIZED_USER_IDS=123456789
TELEGRAM_WEBHOOK_MODE=true
KOYEB_PUBLIC_DOMAIN=painful-koral-dzeckyete-ee811ac5.koyeb.app
PORT=8000

# DATABASE (Dari Neon/Supabase - WAJIB untuk Koyeb)
DATABASE_URL=postgresql://user:password@ep-xxx.neon.tech/neondb?sslmode=require

# ============================================
# FREE TIER OPTIMIZATION (Recommended)
# ============================================
FREE_TIER_MODE=true
SELF_PING_ENABLED=true
SELF_PING_INTERVAL=240
MEMORY_WARNING_THRESHOLD_MB=400
MEMORY_CRITICAL_THRESHOLD_MB=450

# ============================================
# UNLIMITED SIGNALS (Jangan ubah!)
# ============================================
SIGNAL_COOLDOWN_SECONDS=0
OPPOSITE_SIGNAL_COOLDOWN_SECONDS=0
MAX_TRADES_PER_DAY=0

# ============================================
# TRADING SETTINGS (Opsional)
# ============================================
ACCOUNT_BALANCE=1000
RISK_PER_TRADE_PERCENT=1.0
```

## Bot Commands (Telegram)

| Command | Deskripsi |
|---------|-----------|
| `/start` | Memulai bot |
| `/help` | Menampilkan bantuan |
| `/monitor` | Memulai monitoring harga XAUUSD |
| `/stopmonitor` | Menghentikan monitoring |
| `/getsignal` | Mendapatkan sinyal trading terbaru |
| `/riwayat` | Menampilkan 10 trade terakhir |
| `/performa` | Statistik performa (7d, 30d, all-time) |
| `/trialstatus` | Cek status akses trial |
| `/buyaccess` | Informasi pembelian akses |
| `/riset` | Mode riset/analisis pasar |

## Web Dashboard

Dashboard dapat diakses melalui Telegram WebApp atau URL langsung:
```
https://YOUR-DOMAIN.koyeb.app/
```

Fitur Dashboard:
- Real-time harga XAUUSD
- Grafik candlestick dengan indikator
- Status posisi aktif
- Riwayat trading
- Sinkronisasi dengan Telegram Bot via SignalEventStore

## Troubleshooting

### Bot tidak merespon command sama sekali?

**Penyebab umum:**
1. Environment variables tidak terbaca dengan benar
2. Webhook belum ter-register ke Telegram
3. Bot dalam "limited mode"

**Langkah fix:**

1. **Cek log di Koyeb dashboard**
   - Buka Koyeb > Service > Logs
   - Cari pesan "REFRESHING ENVIRONMENT CONFIGURATION"
   - Pastikan terlihat:
     - Token: ✅ Set
     - Authorized Users: (harus > 0)
     - Webhook Mode: ✅ Enabled

2. **Pastikan format environment variables BENAR:**
   ```
   TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrSTUvwxYZ
   AUTHORIZED_USER_IDS=123456789
   TELEGRAM_WEBHOOK_MODE=true
   WEBHOOK_URL=https://nama-app.koyeb.app/bot1234567890:ABCdefGHIjklMNOpqrSTUvwxYZ
   PORT=8000
   ```

3. **Format WEBHOOK_URL yang BENAR:**
   ```
   WEBHOOK_URL=https://DOMAIN/bot<TELEGRAM_TOKEN>
   ```
   Contoh:
   ```
   WEBHOOK_URL=https://logical-krill-dzeckyete-3968ba8c.koyeb.app/bot7879056603:AAE_hnemqmyUVpl8bPvZLWMSOLbauvWADjQ
   ```

4. **Re-deploy service** setelah mengubah environment variables

### Website/Dashboard lambat?
- Ini normal jika bot dalam "limited mode"
- Cek log apakah ada error saat startup
- Pastikan semua environment variables sudah di-set

### Bot dalam "limited mode"?
Ini berarti TELEGRAM_BOT_TOKEN atau AUTHORIZED_USER_IDS tidak terbaca.

**Fix:**
1. Pastikan TIDAK ada spasi di awal/akhir value
2. Pastikan TIDAK ada kutip (`"` atau `'`) di value
3. Re-deploy setelah mengubah environment variables

### Cara mendapatkan KOYEB_PUBLIC_DOMAIN:
1. Buka Koyeb dashboard
2. Pilih service/app anda
3. Lihat di bagian "Public URL" atau "Domain"
4. Copy hanya bagian domain (tanpa https://)

### Cara test webhook:
Buka browser dan akses:
```
https://YOUR-DOMAIN.koyeb.app/webhook
```
Response yang benar:
```json
{"ok": true, "webhook_mode": true, "is_koyeb": true, "config_valid": true, "token_set": true, "bot_initialized": true}
```

Jika `config_valid: false` atau `bot_initialized: false`, cek environment variables.
