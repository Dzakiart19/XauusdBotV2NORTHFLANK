#!/usr/bin/env python3
"""
Script untuk fix/setup webhook Telegram secara manual.
Jalankan ini jika bot tidak merespon command di Koyeb.

Cara pakai:
1. Set environment variable WEBHOOK_URL atau KOYEB_PUBLIC_DOMAIN
2. Jalankan: python fix_webhook.py

Atau langsung dengan argument:
    python fix_webhook.py https://your-app.koyeb.app/webhook
"""

import asyncio
import os
import sys
from telegram import Bot
from telegram.error import TelegramError

def get_webhook_url():
    """Dapatkan webhook URL dari environment atau argument"""
    if len(sys.argv) > 1:
        return sys.argv[1]
    
    webhook_url = os.getenv('WEBHOOK_URL', '')
    if webhook_url:
        return webhook_url
    
    koyeb_domain = os.getenv('KOYEB_PUBLIC_DOMAIN', '')
    if koyeb_domain:
        return f"https://{koyeb_domain}/webhook"
    
    return None

async def setup_webhook():
    """Setup webhook ke Telegram API"""
    token = os.getenv('TELEGRAM_BOT_TOKEN', '')
    
    if not token:
        print("❌ ERROR: TELEGRAM_BOT_TOKEN tidak diset!")
        print("Set environment variable: export TELEGRAM_BOT_TOKEN='your_token'")
        return False
    
    webhook_url = get_webhook_url()
    
    if not webhook_url:
        print("❌ ERROR: Webhook URL tidak tersedia!")
        print("")
        print("Cara menggunakan:")
        print("  1. Set WEBHOOK_URL environment variable:")
        print("     export WEBHOOK_URL='https://your-app.koyeb.app/webhook'")
        print("")
        print("  2. Atau gunakan argument:")
        print("     python fix_webhook.py https://your-app.koyeb.app/webhook")
        print("")
        print("  3. Atau set KOYEB_PUBLIC_DOMAIN:")
        print("     export KOYEB_PUBLIC_DOMAIN='your-app.koyeb.app'")
        return False
    
    if not webhook_url.startswith('https://'):
        print(f"⚠️ WARNING: Webhook URL harus HTTPS: {webhook_url}")
        print("Telegram tidak menerima webhook HTTP (hanya HTTPS)")
        return False
    
    print(f"🔧 Setting up webhook...")
    print(f"   URL: {webhook_url}")
    print(f"   Token: {token[:15]}...{token[-5:]}")
    
    try:
        bot = Bot(token=token)
        
        print("\n📋 Menghapus webhook lama...")
        await bot.delete_webhook(drop_pending_updates=True)
        print("   ✅ Webhook lama dihapus")
        
        print("\n📋 Mendaftarkan webhook baru...")
        success = await bot.set_webhook(
            url=webhook_url,
            allowed_updates=['message', 'callback_query', 'edited_message'],
            drop_pending_updates=True
        )
        
        if success:
            print("   ✅ Webhook berhasil didaftarkan!")
        else:
            print("   ❌ Webhook gagal didaftarkan")
            return False
        
        print("\n📋 Verifikasi webhook...")
        webhook_info = await bot.get_webhook_info()
        
        print(f"   URL: {webhook_info.url}")
        print(f"   Pending Updates: {webhook_info.pending_update_count}")
        print(f"   Max Connections: {webhook_info.max_connections}")
        
        if webhook_info.last_error_message:
            print(f"   ⚠️ Last Error: {webhook_info.last_error_message}")
            print(f"   Error Date: {webhook_info.last_error_date}")
        
        if webhook_info.url == webhook_url:
            print("\n✅ SUKSES! Webhook sudah aktif.")
            print(f"   Bot akan menerima update di: {webhook_url}")
            return True
        else:
            print(f"\n❌ ERROR: URL mismatch!")
            print(f"   Expected: {webhook_url}")
            print(f"   Got: {webhook_info.url}")
            return False
            
    except TelegramError as e:
        print(f"\n❌ Telegram Error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        return False

async def check_webhook_status():
    """Cek status webhook saat ini"""
    token = os.getenv('TELEGRAM_BOT_TOKEN', '')
    
    if not token:
        print("❌ TELEGRAM_BOT_TOKEN tidak diset!")
        return
    
    try:
        bot = Bot(token=token)
        webhook_info = await bot.get_webhook_info()
        
        print("=" * 50)
        print("📊 WEBHOOK STATUS")
        print("=" * 50)
        print(f"URL: {webhook_info.url or '(not set)'}")
        print(f"Pending Updates: {webhook_info.pending_update_count}")
        print(f"Max Connections: {webhook_info.max_connections}")
        print(f"IP Address: {webhook_info.ip_address or '(not set)'}")
        
        if webhook_info.allowed_updates:
            print(f"Allowed Updates: {', '.join(webhook_info.allowed_updates)}")
        
        if webhook_info.last_error_message:
            print(f"\n⚠️ LAST ERROR:")
            print(f"   Message: {webhook_info.last_error_message}")
            print(f"   Date: {webhook_info.last_error_date}")
        
        print("=" * 50)
        
        if webhook_info.url:
            print("✅ Webhook aktif")
        else:
            print("❌ Webhook TIDAK aktif - bot tidak bisa menerima command!")
            print("   Jalankan: python fix_webhook.py <URL> untuk setup")
            
    except TelegramError as e:
        print(f"❌ Telegram Error: {e}")
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")

async def delete_webhook():
    """Hapus webhook dan switch ke polling mode"""
    token = os.getenv('TELEGRAM_BOT_TOKEN', '')
    
    if not token:
        print("❌ TELEGRAM_BOT_TOKEN tidak diset!")
        return
    
    try:
        bot = Bot(token=token)
        await bot.delete_webhook(drop_pending_updates=True)
        print("✅ Webhook dihapus - bot sekarang bisa menggunakan polling mode")
    except TelegramError as e:
        print(f"❌ Telegram Error: {e}")

def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg in ['--status', '-s', 'status']:
            asyncio.run(check_webhook_status())
            return
        
        if arg in ['--delete', '-d', 'delete']:
            asyncio.run(delete_webhook())
            return
        
        if arg in ['--help', '-h', 'help']:
            print(__doc__)
            print("\nOptions:")
            print("  --status, -s    Cek status webhook saat ini")
            print("  --delete, -d    Hapus webhook (untuk polling mode)")
            print("  --help, -h      Tampilkan bantuan ini")
            print("  <URL>           Setup webhook dengan URL tertentu")
            return
    
    success = asyncio.run(setup_webhook())
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
