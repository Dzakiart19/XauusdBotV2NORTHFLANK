"""
Internationalization (i18n) module for multi-language support.

Supports:
- Indonesian (id) - Default
- English (en)

Usage:
    from bot.i18n import get_text, set_user_language
    
    set_user_language(user_id, 'en')
    msg = get_text('welcome', user_id)
"""
import logging
from typing import Dict, Optional, Any
from datetime import datetime
import pytz

logger = logging.getLogger('I18n')

SUPPORTED_LANGUAGES = ['id', 'en']
DEFAULT_LANGUAGE = 'id'

_user_languages: Dict[int, str] = {}

TRANSLATIONS = {
    'id': {
        'welcome': 'Selamat datang di XAUUSD Trading Bot!',
        'welcome_user': 'Selamat datang, {name}!',
        'help_title': 'Bantuan Trading Bot',
        'help_commands': '''
Perintah yang tersedia:

📊 *Trading*
/monitor - Mulai monitoring sinyal otomatis
/stop - Hentikan monitoring
/getsignal - Dapatkan sinyal manual
/status - Cek status posisi aktif

📈 *Analisis*
/performa - Lihat statistik performa
/riwayat - Lihat riwayat trading
/stats - Statistik hari ini

⚙️ *Pengaturan*
/settings - Buka pengaturan
/bahasa - Ganti bahasa (ID/EN)
/notifikasi - Atur notifikasi

ℹ️ *Informasi*
/help - Tampilkan bantuan ini
/about - Tentang bot
''',
        'monitoring_started': '🔄 Monitoring sinyal XAUUSD dimulai...',
        'monitoring_stopped': '⏹️ Monitoring dihentikan',
        'no_active_position': '📭 Tidak ada posisi aktif',
        'signal_buy': '🟢 SINYAL BUY - XAUUSD',
        'signal_sell': '🔴 SINYAL SELL - XAUUSD',
        'entry_price': 'Entry',
        'stop_loss': 'Stop Loss',
        'take_profit': 'Take Profit',
        'risk_reward': 'Risk:Reward',
        'lot_size': 'Lot Size',
        'risk_amount': 'Risiko',
        'trend': 'Trend',
        'momentum': 'Momentum',
        'volume': 'Volume',
        'time': 'Waktu',
        'trade_closed': 'Trade Ditutup',
        'tp_hit': 'TP Hit - Profit',
        'sl_hit': 'SL Hit',
        'profit': 'Profit',
        'loss': 'Loss',
        'duration': 'Durasi',
        'performance_title': 'Performa Trading',
        'total_trades': 'Total Trade',
        'wins': 'Menang',
        'losses': 'Kalah',
        'win_rate': 'Win Rate',
        'total_pl': 'Total P/L',
        'profit_factor': 'Profit Factor',
        'last_7_days': '7 Hari Terakhir',
        'last_30_days': '30 Hari Terakhir',
        'all_time': 'Semua Waktu',
        'no_data': 'Belum ada data',
        'loading': 'Memuat...',
        'error': 'Terjadi kesalahan',
        'unauthorized': 'Anda tidak memiliki akses',
        'rate_limited': 'Terlalu banyak permintaan, coba lagi nanti',
        'settings_title': 'Pengaturan',
        'language_changed': 'Bahasa berhasil diubah ke Indonesia',
        'notification_on': 'Notifikasi diaktifkan',
        'notification_off': 'Notifikasi dinonaktifkan',
        'onboarding_step1': '''
👋 *Selamat Datang di Trading Bot!*

Saya akan membantu Anda trading XAUUSD dengan sinyal otomatis.

📌 *Langkah 1/3: Mengenal Fitur*

• Sinyal BUY/SELL otomatis
• Monitoring posisi real-time
• Manajemen risiko otomatis
• Statistik performa

Ketik /next untuk melanjutkan...
''',
        'onboarding_step2': '''
⚙️ *Langkah 2/3: Pengaturan Risiko*

Sebelum mulai trading, atur toleransi risiko Anda:

• Risk per trade: 1-2% modal
• Stop Loss: Selalu digunakan
• Take Profit: Rasio minimal 1:1.5

Ketik /next untuk melanjutkan...
''',
        'onboarding_step3': '''
🚀 *Langkah 3/3: Mulai Trading*

Anda sudah siap! Berikut cara memulai:

1. Ketik /monitor untuk mulai monitoring
2. Bot akan kirim sinyal otomatis
3. Gunakan /status untuk cek posisi

💡 Tips: Gunakan /help kapan saja untuk bantuan.

Ketik /monitor untuk memulai!
''',
        'menu_main': 'Menu Utama',
        'menu_trading': 'Trading',
        'menu_analysis': 'Analisis',
        'menu_settings': 'Pengaturan',
        'btn_start_monitoring': '🔄 Mulai Monitor',
        'btn_stop_monitoring': '⏹️ Stop Monitor',
        'btn_get_signal': '📡 Sinyal Manual',
        'btn_status': '📊 Status',
        'btn_performance': '📈 Performa',
        'btn_history': '📜 Riwayat',
        'btn_settings': '⚙️ Pengaturan',
        'btn_help': '❓ Bantuan',
        'btn_language': '🌐 Bahasa',
        'btn_back': '⬅️ Kembali',
        'btn_notifications': '🔔 Notifikasi',
        'btn_risk_level': '⚠️ Risk Level',
        'btn_timeframe': '📊 Timeframe',
    },
    'en': {
        'welcome': 'Welcome to XAUUSD Trading Bot!',
        'welcome_user': 'Welcome, {name}!',
        'help_title': 'Trading Bot Help',
        'help_commands': '''
Available commands:

📊 *Trading*
/monitor - Start automatic signal monitoring
/stop - Stop monitoring
/getsignal - Get manual signal
/status - Check active position status

📈 *Analysis*
/performa - View performance statistics
/riwayat - View trading history
/stats - Today's statistics

⚙️ *Settings*
/settings - Open settings
/language - Change language (ID/EN)
/notifications - Set notifications

ℹ️ *Information*
/help - Show this help
/about - About the bot
''',
        'monitoring_started': '🔄 XAUUSD signal monitoring started...',
        'monitoring_stopped': '⏹️ Monitoring stopped',
        'no_active_position': '📭 No active position',
        'signal_buy': '🟢 BUY SIGNAL - XAUUSD',
        'signal_sell': '🔴 SELL SIGNAL - XAUUSD',
        'entry_price': 'Entry',
        'stop_loss': 'Stop Loss',
        'take_profit': 'Take Profit',
        'risk_reward': 'Risk:Reward',
        'lot_size': 'Lot Size',
        'risk_amount': 'Risk',
        'trend': 'Trend',
        'momentum': 'Momentum',
        'volume': 'Volume',
        'time': 'Time',
        'trade_closed': 'Trade Closed',
        'tp_hit': 'TP Hit - Profit',
        'sl_hit': 'SL Hit',
        'profit': 'Profit',
        'loss': 'Loss',
        'duration': 'Duration',
        'performance_title': 'Trading Performance',
        'total_trades': 'Total Trades',
        'wins': 'Wins',
        'losses': 'Losses',
        'win_rate': 'Win Rate',
        'total_pl': 'Total P/L',
        'profit_factor': 'Profit Factor',
        'last_7_days': 'Last 7 Days',
        'last_30_days': 'Last 30 Days',
        'all_time': 'All Time',
        'no_data': 'No data available',
        'loading': 'Loading...',
        'error': 'An error occurred',
        'unauthorized': 'You do not have access',
        'rate_limited': 'Too many requests, please try again later',
        'settings_title': 'Settings',
        'language_changed': 'Language changed to English',
        'notification_on': 'Notifications enabled',
        'notification_off': 'Notifications disabled',
        'onboarding_step1': '''
👋 *Welcome to Trading Bot!*

I will help you trade XAUUSD with automatic signals.

📌 *Step 1/3: Getting to Know Features*

• Automatic BUY/SELL signals
• Real-time position monitoring
• Automatic risk management
• Performance statistics

Type /next to continue...
''',
        'onboarding_step2': '''
⚙️ *Step 2/3: Risk Settings*

Before trading, set your risk tolerance:

• Risk per trade: 1-2% of capital
• Stop Loss: Always used
• Take Profit: Minimum ratio 1:1.5

Type /next to continue...
''',
        'onboarding_step3': '''
🚀 *Step 3/3: Start Trading*

You're ready! Here's how to start:

1. Type /monitor to start monitoring
2. Bot will send automatic signals
3. Use /status to check positions

💡 Tip: Use /help anytime for assistance.

Type /monitor to begin!
''',
        'menu_main': 'Main Menu',
        'menu_trading': 'Trading',
        'menu_analysis': 'Analysis',
        'menu_settings': 'Settings',
        'btn_start_monitoring': '🔄 Start Monitor',
        'btn_stop_monitoring': '⏹️ Stop Monitor',
        'btn_get_signal': '📡 Manual Signal',
        'btn_status': '📊 Status',
        'btn_performance': '📈 Performance',
        'btn_history': '📜 History',
        'btn_settings': '⚙️ Settings',
        'btn_help': '❓ Help',
        'btn_language': '🌐 Language',
        'btn_back': '⬅️ Back',
        'btn_notifications': '🔔 Notifications',
        'btn_risk_level': '⚠️ Risk Level',
        'btn_timeframe': '📊 Timeframe',
    }
}


def get_user_language(user_id: int) -> str:
    """Get user's preferred language."""
    return _user_languages.get(user_id, DEFAULT_LANGUAGE)


def set_user_language(user_id: int, language: str) -> bool:
    """
    Set user's preferred language.
    
    Args:
        user_id: Telegram user ID
        language: Language code ('id' or 'en')
    
    Returns:
        True if successful
    """
    if language not in SUPPORTED_LANGUAGES:
        logger.warning(f"Unsupported language: {language}")
        return False
    
    _user_languages[user_id] = language
    logger.info(f"Language set for user {user_id}: {language}")
    return True


def get_text(key: str, user_id: Optional[int] = None, 
             language: Optional[str] = None, **kwargs) -> str:
    """
    Get translated text for a key.
    
    Args:
        key: Translation key
        user_id: Optional user ID to get their preferred language
        language: Optional language override
        **kwargs: Format arguments
    
    Returns:
        Translated text
    """
    lang = language
    if not lang and user_id:
        lang = get_user_language(user_id)
    if not lang:
        lang = DEFAULT_LANGUAGE
    
    if lang not in TRANSLATIONS:
        lang = DEFAULT_LANGUAGE
    
    text = TRANSLATIONS[lang].get(key)
    if not text:
        text = TRANSLATIONS[DEFAULT_LANGUAGE].get(key, key)
    
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, ValueError) as e:
            logger.warning(f"Error formatting text '{key}': {e}")
    
    return text


def t(key: str, user_id: Optional[int] = None, **kwargs) -> str:
    """Shorthand for get_text()."""
    return get_text(key, user_id, **kwargs)


def format_datetime(dt: datetime, user_id: Optional[int] = None, 
                   style: str = 'full') -> str:
    """
    Format datetime according to user's language.
    
    Args:
        dt: Datetime to format
        user_id: User ID for language preference
        style: 'full', 'date', 'time', or 'short'
    
    Returns:
        Formatted datetime string
    """
    lang = get_user_language(user_id) if user_id else DEFAULT_LANGUAGE
    
    jakarta = pytz.timezone('Asia/Jakarta')
    if dt.tzinfo is None:
        dt = pytz.UTC.localize(dt)
    local_dt = dt.astimezone(jakarta)
    
    if lang == 'en':
        if style == 'full':
            return local_dt.strftime('%b %d, %Y %I:%M:%S %p WIB')
        elif style == 'date':
            return local_dt.strftime('%b %d, %Y')
        elif style == 'time':
            return local_dt.strftime('%I:%M:%S %p WIB')
        else:
            return local_dt.strftime('%m/%d %I:%M %p')
    else:
        if style == 'full':
            return local_dt.strftime('%d/%m/%Y %H:%M:%S WIB')
        elif style == 'date':
            return local_dt.strftime('%d/%m/%Y')
        elif style == 'time':
            return local_dt.strftime('%H:%M:%S WIB')
        else:
            return local_dt.strftime('%d/%m %H:%M')


def format_currency(amount: float, user_id: Optional[int] = None) -> str:
    """
    Format currency according to user's language.
    
    Args:
        amount: Amount to format
        user_id: User ID for language preference
    
    Returns:
        Formatted currency string
    """
    sign = '+' if amount >= 0 else ''
    return f"{sign}${amount:.2f}"


def format_percentage(value: float, user_id: Optional[int] = None) -> str:
    """
    Format percentage according to user's language.
    
    Args:
        value: Percentage value
        user_id: User ID for language preference
    
    Returns:
        Formatted percentage string
    """
    sign = '+' if value >= 0 else ''
    return f"{sign}{value:.1f}%"
