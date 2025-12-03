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

## OPSIONAL

```
DATABASE_URL=postgresql://user:pass@host:5432/dbname
DRY_RUN=false
ENVIRONMENT=production
```

## Contoh Lengkap untuk Koyeb

```env
# WAJIB
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrSTUvwxYZ
AUTHORIZED_USER_IDS=123456789
TELEGRAM_WEBHOOK_MODE=true
KOYEB_PUBLIC_DOMAIN=trading-bot-xyz.koyeb.app
PORT=8000

# FREE TIER OPTIMIZATION
FREE_TIER_MODE=true
SELF_PING_ENABLED=true
SELF_PING_INTERVAL=240

# TRADING
SIGNAL_COOLDOWN_SECONDS=0
MAX_TRADES_PER_DAY=0
```

## Troubleshooting

### Bot tidak merespon command?
1. Cek apakah `TELEGRAM_WEBHOOK_MODE=true`
2. Cek apakah `KOYEB_PUBLIC_DOMAIN` atau `WEBHOOK_URL` sudah di-set
3. Pastikan domain sesuai dengan yang di Koyeb dashboard
4. Cek log di Koyeb untuk error message

### Bot masih bisa kirim signal tapi tidak merespon?
Ini berarti webhook tidak ter-setup dengan benar. Pastikan:
- `KOYEB_PUBLIC_DOMAIN` sudah benar (contoh: `trading-bot-abc123.koyeb.app`)
- BUKAN `https://...` - hanya domain saja

### Cara mendapatkan KOYEB_PUBLIC_DOMAIN:
1. Buka Koyeb dashboard
2. Pilih service/app anda
3. Lihat di bagian "Public URL" atau "Domain"
4. Copy hanya bagian domain (tanpa https://)
