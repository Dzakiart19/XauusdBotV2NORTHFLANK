# 🚀 Deploy Trading Bot ke Northflank

Panduan lengkap untuk deploy XAUUSD Trading Bot ke Northflank (GRATIS tanpa kartu kredit!).

## ✅ Kenapa Northflank?

| Fitur | Northflank | Koyeb |
|-------|-----------|-------|
| **Free Tier** | 2 services + 1 database | 1 service + limited DB |
| **Credit Card** | ❌ TIDAK PERLU | ✅ Perlu |
| **Apps Sleep** | ❌ Selalu online 24/7 | Bisa sleep jika idle |
| **Auto-Redeploy** | ✅ Ya, dari GitHub | ✅ Ya |
| **Build Minutes** | 500/bulan gratis | Terbatas |

---

## 📋 Prerequisites

1. **Akun Northflank** (gratis): https://app.northflank.com/
2. **Repository GitHub** dengan code trading bot
3. **Telegram Bot Token** dari @BotFather
4. **Telegram User ID** Anda (dari @userinfobot)

---

## 🔧 Step-by-Step Deployment

### 1. Signup di Northflank

1. Buka https://app.northflank.com/
2. Klik **"Sign Up"**
3. Pilih signup dengan **GitHub** (recommended) atau Email
4. **TIDAK PERLU kartu kredit!**

### 2. Buat Project Baru

1. Setelah login, klik **"Create Project"**
2. Beri nama project, misal: `trading-bot`
3. Pilih region terdekat (Frankfurt/Singapore/dll)
4. Klik **"Create Project"**

### 3. Connect GitHub Repository

1. Di project dashboard, klik **"Create Service"**
2. Pilih **"Deploy from Git"**
3. Klik **"Connect GitHub"**
4. Authorize Northflank untuk akses repository Anda
5. Pilih repository trading bot
6. Branch: **main** (atau branch utama Anda)

### 4. Konfigurasi Build

Di halaman **"Build"**:

- **Build Type**: Docker
- **Dockerfile**: `Dockerfile` (otomatis terdeteksi)
- **Build Context**: `/` (root)

Klik **"Next"**

### 5. Konfigurasi Runtime

Di halaman **"Resources"**:

- **Plan**: Free (atau Starter jika mau upgrade)
- **CPU**: 0.1 vCPU
- **Memory**: 512 MB

Di halaman **"Networking"**:

- **Port**: `8080`
- **Protocol**: HTTP
- **Public**: ✅ Enable

### 6. Environment Variables ⚡ WAJIB

**TANPA ENVIRONMENT VARIABLES INI, BOT TIDAK AKAN MERESPON COMMAND!**

Tambahkan di halaman **"Environment"**:

#### Variable WAJIB:

```
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz1234567890
AUTHORIZED_USER_IDS=123456789
```

- **TELEGRAM_BOT_TOKEN**: Dapatkan dari @BotFather
- **AUTHORIZED_USER_IDS**: Dapatkan dari @userinfobot

#### Webhook Mode (RECOMMENDED):

```
TELEGRAM_WEBHOOK_MODE=true
WEBHOOK_URL=https://<your-app>.northflank.app/webhook
```

> **Catatan**: Setelah deploy, ganti `<your-app>` dengan URL service Anda yang diberikan Northflank.

#### Variable Opsional:

```
FREE_TIER_MODE=true
TRADING_HOURS_START=0
TRADING_HOURS_END=23
SIGNAL_COOLDOWN_SECONDS=0
MAX_TRADES_PER_DAY=0
```

### 7. Health Check

Di halaman **"Health Checks"**:

- **Path**: `/health`
- **Port**: `8080`
- **Protocol**: HTTP
- **Initial Delay**: 30 seconds
- **Period**: 30 seconds

### 8. Deploy!

1. Review semua konfigurasi
2. Klik **"Create Service"**
3. Tunggu 3-5 menit untuk build & deploy
4. Status akan berubah jadi **"Running"** kalau berhasil

---

## 🔄 Auto-Redeploy dari GitHub

**Ya, Northflank otomatis redeploy saat Anda push ke GitHub!**

### Cara Kerja:

1. Push code baru ke GitHub branch yang di-connect
2. Northflank otomatis detect perubahan
3. Build baru dimulai
4. Deploy otomatis setelah build sukses

### Disable Auto-Redeploy (Opsional):

Jika Anda ingin kontrol manual:
1. Buka service Anda di Northflank
2. Klik **"Settings"** → **"Build Settings"**
3. Disable **"Auto Build"**

---

## ✅ Verifikasi Deployment

### 1. Test Health Check

Buka browser:
```
https://<your-app>.northflank.app/health
```

Response yang benar:
```json
{
  "status": "healthy",
  "mode": "full",
  "config_valid": true,
  "market_connected": true,
  "telegram_running": true,
  "webhook_mode": true
}
```

### 2. Test Bot di Telegram

1. Buka Telegram, cari bot Anda
2. Ketik `/start` - harus ada respons welcome
3. Ketik `/getsignal` - harus kirim sinyal trading
4. Ketik `/monitor` - mulai monitoring otomatis

### 3. Cek Logs di Northflank

1. Buka service Anda
2. Tab **"Logs"**
3. Harus lihat:
   ```
   ✅ Connected to Deriv WebSocket
   ✅ Telegram bot is running!
   ✅ BOT IS NOW RUNNING
   ```

---

## 🔍 Troubleshooting

### ❌ Bot Tidak Merespon

**Gejala**: Bot tidak reply command
**Solusi**:
1. Cek health endpoint: `https://<your-app>.northflank.app/health`
2. Pastikan `"mode": "full"` (bukan "limited")
3. Jika "limited", set environment variables yang kurang
4. Restart service di Northflank

### ❌ Webhook Tidak Aktif

**Gejala**: Health menunjukkan `"webhook_mode": false`
**Solusi**:
1. Set `TELEGRAM_WEBHOOK_MODE=true`
2. Set `WEBHOOK_URL=https://<your-app>.northflank.app/webhook`
3. Restart service

### ❌ Build Failed

**Gejala**: Build error di Northflank
**Solusi**:
1. Cek Dockerfile syntax
2. Pastikan requirements.txt valid
3. Cek logs build untuk error spesifik

### ❌ WebSocket Connection Failed

**Gejala**: `"market_connected": false`
**Solusi**:
1. Biasanya temporary, tunggu beberapa detik
2. Bot auto-reconnect setiap 3 detik
3. Cek logs untuk error detail

---

## 🆓 Optimasi Free Tier Northflank

### Resource Limits:

- ✅ 2 services gratis
- ✅ 1 database addon gratis (PostgreSQL)
- ✅ 500 build minutes/bulan
- ✅ 2GB persistent storage
- ✅ 24/7 uptime (tidak ada sleep!)

### Tips Performa:

1. **Biarkan `FREE_TIER_MODE=true`** - Sudah dioptimalkan untuk resource minimal
2. **Gunakan webhook mode** - Lebih efisien dari polling
3. **Gunakan PostgreSQL addon** - Untuk persistent data (opsional)

---

## 🔄 Migrasi dari Koyeb ke Northflank

Jika Anda sudah deploy di Koyeb dan ingin migrasi:

1. **Export environment variables** dari Koyeb
2. **Buat service baru** di Northflank
3. **Copy environment variables** ke Northflank
4. **Deploy** dan test
5. **Update WEBHOOK_URL** ke domain Northflank baru

**Catatan**: Anda bisa menjalankan keduanya bersamaan untuk testing, tapi pastikan hanya satu yang aktif sebagai bot Telegram utama.

---

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

---

## 🎯 Fitur Bot

- ✅ Real-time data dari Deriv (XAUUSD/Gold)
- ✅ Zero API key required untuk market data
- ✅ Dual signal modes: 🤖 Auto & 👤 Manual
- ✅ Chart visualization setiap sinyal
- ✅ Position tracking hingga TP/SL tercapai
- ✅ Risk management dengan trailing stop
- ✅ 24/7 monitoring tanpa henti
- ✅ Webhook mode untuk deployment cloud

---

## 🔧 PostgreSQL Database (Opsional)

Northflank menawarkan PostgreSQL addon gratis:

1. Di project dashboard, klik **"Add-Ons"**
2. Klik **"Create Add-on"**
3. Pilih **"PostgreSQL"**
4. Pilih plan **"Free"**
5. Copy connection string
6. Tambahkan ke environment variables:
   ```
   DATABASE_URL=postgresql://...
   ```

---

## 📞 Support

Jika ada masalah:
1. Cek Northflank logs dulu
2. Cek health endpoint
3. Restart service di Dashboard
4. Cek dokumentasi: https://northflank.com/docs

---

**Happy Trading! 🚀📈**

*Bot ini support deployment di Koyeb DAN Northflank. Pilih platform sesuai preferensi Anda!*
