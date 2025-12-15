"""
Enhanced Interactive Menu System untuk Bot Trading XAUUSD.

Modul ini menyediakan:
- Menu inline keyboard yang lebih interaktif
- Quick action buttons setelah signal
- Custom notification preferences
- User onboarding flow
"""

import logging
from typing import Optional, List, Dict, Any, Callable
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes, CallbackQueryHandler
from bot.i18n import get_text, t, set_user_language

logger = logging.getLogger('EnhancedMenu')


class CallbackData:
    """Konstanta untuk callback data"""
    MAIN_MENU = 'menu:main'
    MONITOR_START = 'menu:monitor:start'
    MONITOR_STOP = 'menu:monitor:stop'
    GET_SIGNAL = 'menu:signal'
    STATUS = 'menu:status'
    PERFORMANCE = 'menu:performance'
    HISTORY = 'menu:history'
    SETTINGS = 'menu:settings'
    HELP = 'menu:help'
    DASHBOARD = 'menu:dashboard'
    REPORTS = 'menu:reports'
    
    SETTINGS_LANG = 'settings:lang'
    SETTINGS_NOTIF = 'settings:notif'
    SETTINGS_RISK = 'settings:risk'
    SETTINGS_TIMEFRAME = 'settings:timeframe'
    
    LANG_ID = 'lang:id'
    LANG_EN = 'lang:en'
    
    NOTIF_ALL = 'notif:all'
    NOTIF_SIGNALS = 'notif:signals'
    NOTIF_CRITICAL = 'notif:critical'
    NOTIF_OFF = 'notif:off'
    
    REPORT_DAILY = 'report:daily'
    REPORT_WEEKLY = 'report:weekly'
    REPORT_MONTHLY = 'report:monthly'
    REPORT_EXPORT = 'report:export'
    
    SIGNAL_ACTION_CLOSE = 'signal:close'
    SIGNAL_ACTION_UPDATE = 'signal:update'
    SIGNAL_ACTION_MODIFY = 'signal:modify'
    
    CONFIRM_YES = 'confirm:yes'
    CONFIRM_NO = 'confirm:no'
    
    BACK = 'nav:back'
    REFRESH = 'nav:refresh'


class EnhancedMenuBuilder:
    """Builder untuk membuat menu yang lebih interaktif"""
    
    def __init__(self, user_id: Optional[int] = None):
        self.user_id = user_id
        self.buttons: List[List[InlineKeyboardButton]] = []
        self.current_row: List[InlineKeyboardButton] = []
    
    def add_button(self, text: str, callback_data: str, 
                   url: Optional[str] = None) -> 'EnhancedMenuBuilder':
        """Add button ke current row"""
        if url:
            btn = InlineKeyboardButton(text, url=url)
        else:
            btn = InlineKeyboardButton(text, callback_data=callback_data)
        self.current_row.append(btn)
        return self
    
    def add_localized_button(self, key: str, callback_data: str) -> 'EnhancedMenuBuilder':
        """Add button dengan teks yang di-localize"""
        text = t(key, self.user_id)
        return self.add_button(text, callback_data)
    
    def add_emoji_button(self, emoji: str, text: str, callback_data: str) -> 'EnhancedMenuBuilder':
        """Add button dengan emoji"""
        return self.add_button(f"{emoji} {text}", callback_data)
    
    def new_row(self) -> 'EnhancedMenuBuilder':
        """Start row baru"""
        if self.current_row:
            self.buttons.append(self.current_row)
            self.current_row = []
        return self
    
    def add_back_button(self, callback_data: str = CallbackData.MAIN_MENU) -> 'EnhancedMenuBuilder':
        """Add back button"""
        self.new_row()
        return self.add_button("⬅️ Kembali", callback_data)
    
    def add_refresh_button(self, callback_data: str = CallbackData.REFRESH) -> 'EnhancedMenuBuilder':
        """Add refresh button"""
        return self.add_button("🔄 Refresh", callback_data)
    
    def build(self) -> InlineKeyboardMarkup:
        """Build keyboard markup"""
        if self.current_row:
            self.buttons.append(self.current_row)
        return InlineKeyboardMarkup(self.buttons)


def get_enhanced_main_menu(user_id: int, is_monitoring: bool = False,
                           has_active_position: bool = False) -> InlineKeyboardMarkup:
    """Get main menu dengan status awareness"""
    builder = EnhancedMenuBuilder(user_id)
    
    if is_monitoring:
        builder.add_emoji_button("⏹️", "Stop Monitor", CallbackData.MONITOR_STOP)
    else:
        builder.add_emoji_button("▶️", "Start Monitor", CallbackData.MONITOR_START)
    
    builder.add_emoji_button("📡", "Get Signal", CallbackData.GET_SIGNAL)
    builder.new_row()
    
    if has_active_position:
        builder.add_emoji_button("📊", "Posisi Aktif", CallbackData.STATUS)
    else:
        builder.add_emoji_button("📈", "Status", CallbackData.STATUS)
    
    builder.add_emoji_button("📋", "Performa", CallbackData.PERFORMANCE)
    builder.new_row()
    
    builder.add_emoji_button("📜", "Riwayat", CallbackData.HISTORY)
    builder.add_emoji_button("📊", "Reports", CallbackData.REPORTS)
    builder.new_row()
    
    builder.add_emoji_button("⚙️", "Settings", CallbackData.SETTINGS)
    builder.add_emoji_button("❓", "Help", CallbackData.HELP)
    
    return builder.build()


def get_settings_menu(user_id: int) -> InlineKeyboardMarkup:
    """Get settings menu"""
    return (EnhancedMenuBuilder(user_id)
        .add_emoji_button("🌐", "Bahasa", CallbackData.SETTINGS_LANG)
        .add_emoji_button("🔔", "Notifikasi", CallbackData.SETTINGS_NOTIF)
        .new_row()
        .add_emoji_button("⚠️", "Risk Level", CallbackData.SETTINGS_RISK)
        .add_emoji_button("⏱️", "Timeframe", CallbackData.SETTINGS_TIMEFRAME)
        .add_back_button()
        .build())


def get_language_menu(user_id: int) -> InlineKeyboardMarkup:
    """Get language selection menu"""
    return (EnhancedMenuBuilder(user_id)
        .add_button("🇮🇩 Bahasa Indonesia", CallbackData.LANG_ID)
        .new_row()
        .add_button("🇬🇧 English", CallbackData.LANG_EN)
        .add_back_button(CallbackData.SETTINGS)
        .build())


def get_notification_menu(user_id: int, current_setting: str = 'all') -> InlineKeyboardMarkup:
    """Get notification settings menu"""
    builder = EnhancedMenuBuilder(user_id)
    
    settings = [
        ("📢 Semua Notifikasi", CallbackData.NOTIF_ALL, 'all'),
        ("📡 Sinyal Saja", CallbackData.NOTIF_SIGNALS, 'signals'),
        ("🚨 Critical Only", CallbackData.NOTIF_CRITICAL, 'critical'),
        ("🔇 Matikan", CallbackData.NOTIF_OFF, 'off')
    ]
    
    for text, callback, value in settings:
        if value == current_setting:
            builder.add_button(f"✅ {text}", callback)
        else:
            builder.add_button(text, callback)
        builder.new_row()
    
    builder.add_back_button(CallbackData.SETTINGS)
    
    return builder.build()


def get_reports_menu(user_id: int) -> InlineKeyboardMarkup:
    """Get reports menu"""
    return (EnhancedMenuBuilder(user_id)
        .add_emoji_button("📅", "Laporan Harian", CallbackData.REPORT_DAILY)
        .add_emoji_button("📆", "Laporan Mingguan", CallbackData.REPORT_WEEKLY)
        .new_row()
        .add_emoji_button("📊", "Laporan Bulanan", CallbackData.REPORT_MONTHLY)
        .add_emoji_button("📥", "Export Data", CallbackData.REPORT_EXPORT)
        .add_back_button()
        .build())


def get_signal_action_menu(user_id: int, signal_id: str, 
                            signal_type: str) -> InlineKeyboardMarkup:
    """Get quick action menu setelah signal"""
    type_emoji = "🟢" if signal_type == 'BUY' else "🔴"
    
    return (EnhancedMenuBuilder(user_id)
        .add_emoji_button("📊", "Update Status", f"signal:update:{signal_id}")
        .add_emoji_button("✏️", "Modify SL/TP", f"signal:modify:{signal_id}")
        .new_row()
        .add_emoji_button("❌", "Close Position", f"signal:close:{signal_id}")
        .new_row()
        .add_emoji_button("📡", "New Signal", CallbackData.GET_SIGNAL)
        .add_back_button()
        .build())


def get_position_menu(user_id: int, position_id: str) -> InlineKeyboardMarkup:
    """Get position management menu"""
    return (EnhancedMenuBuilder(user_id)
        .add_emoji_button("🔄", "Refresh", f"pos:refresh:{position_id}")
        .add_emoji_button("📊", "Detail", f"pos:detail:{position_id}")
        .new_row()
        .add_emoji_button("🎯", "Modify TP", f"pos:modify_tp:{position_id}")
        .add_emoji_button("⛔", "Modify SL", f"pos:modify_sl:{position_id}")
        .new_row()
        .add_emoji_button("❌", "Close", f"pos:close:{position_id}")
        .add_back_button()
        .build())


def get_confirmation_menu(user_id: int, action: str, 
                          action_data: Optional[str] = None) -> InlineKeyboardMarkup:
    """Get confirmation dialog"""
    confirm_callback = f"confirm:yes:{action}"
    cancel_callback = f"confirm:no:{action}"
    
    if action_data:
        confirm_callback += f":{action_data}"
        cancel_callback += f":{action_data}"
    
    return (EnhancedMenuBuilder(user_id)
        .add_button("✅ Ya, Lanjutkan", confirm_callback)
        .add_button("❌ Batal", cancel_callback)
        .build())


def get_onboarding_menu(user_id: int, step: int) -> InlineKeyboardMarkup:
    """Get onboarding step menu"""
    builder = EnhancedMenuBuilder(user_id)
    
    if step == 1:
        builder.add_button("Mulai ➡️", "onboard:step:2")
        builder.new_row()
        builder.add_button("⏭️ Skip", "onboard:skip")
    elif step == 2:
        builder.add_button("⬅️ Kembali", "onboard:step:1")
        builder.add_button("Lanjut ➡️", "onboard:step:3")
        builder.new_row()
        builder.add_button("⏭️ Skip", "onboard:skip")
    elif step == 3:
        builder.add_button("⬅️ Kembali", "onboard:step:2")
        builder.add_button("🚀 Selesai", "onboard:complete")
    
    return builder.build()


def get_help_menu(user_id: int) -> InlineKeyboardMarkup:
    """Get help menu dengan kategori"""
    return (EnhancedMenuBuilder(user_id)
        .add_emoji_button("📡", "Tentang Sinyal", "help:signals")
        .add_emoji_button("📊", "Cara Baca Chart", "help:charts")
        .new_row()
        .add_emoji_button("⚠️", "Risk Management", "help:risk")
        .add_emoji_button("💡", "Tips Trading", "help:tips")
        .new_row()
        .add_emoji_button("🆘", "FAQ", "help:faq")
        .add_emoji_button("📞", "Kontak", "help:contact")
        .add_back_button()
        .build())


def get_export_format_menu(user_id: int, report_type: str) -> InlineKeyboardMarkup:
    """Get export format selection menu"""
    return (EnhancedMenuBuilder(user_id)
        .add_emoji_button("📄", "JSON", f"export:json:{report_type}")
        .add_emoji_button("📊", "CSV", f"export:csv:{report_type}")
        .new_row()
        .add_emoji_button("📝", "Text", f"export:text:{report_type}")
        .add_back_button(CallbackData.REPORTS)
        .build())


class EnhancedMenuHandler:
    """Handler untuk menu callbacks"""
    
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.user_states: Dict[int, Dict[str, Any]] = {}
        self.user_preferences: Dict[int, Dict[str, Any]] = {}
    
    def get_user_preference(self, user_id: int, key: str, default: Any = None) -> Any:
        """Get user preference"""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        return self.user_preferences[user_id].get(key, default)
    
    def set_user_preference(self, user_id: int, key: str, value: Any):
        """Set user preference"""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        self.user_preferences[user_id][key] = value
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback query dari inline keyboard"""
        query = update.callback_query
        if not query:
            return
        
        await query.answer()
        
        user_id = query.from_user.id
        callback_data = query.data
        
        try:
            parts = callback_data.split(':')
            category = parts[0]
            action = parts[1] if len(parts) > 1 else None
            params = parts[2:] if len(parts) > 2 else []
            
            if category == 'menu':
                await self._handle_menu_action(query, user_id, action, params)
            elif category == 'settings':
                await self._handle_settings_action(query, user_id, action, params)
            elif category == 'lang':
                await self._handle_language_action(query, user_id, action)
            elif category == 'notif':
                await self._handle_notification_action(query, user_id, action)
            elif category == 'report':
                await self._handle_report_action(query, user_id, action)
            elif category == 'signal':
                await self._handle_signal_action(query, user_id, action, params)
            elif category == 'pos':
                await self._handle_position_action(query, user_id, action, params)
            elif category == 'confirm':
                await self._handle_confirmation(query, user_id, action, params)
            elif category == 'help':
                await self._handle_help_action(query, user_id, action)
            elif category == 'onboard':
                await self._handle_onboarding(query, user_id, action, params)
            elif category == 'export':
                await self._handle_export(query, user_id, action, params)
            elif category == 'nav':
                await self._handle_navigation(query, user_id, action)
            else:
                logger.warning(f"Unknown callback category: {category}")
                
        except Exception as e:
            logger.error(f"Error handling callback {callback_data}: {e}")
            await query.edit_message_text("❌ Terjadi error. Silakan coba lagi.")
    
    async def _handle_menu_action(self, query, user_id: int, action: str, params: List[str]):
        """Handle menu navigation"""
        if action == 'main':
            is_monitoring = self.get_user_preference(user_id, 'is_monitoring', False)
            has_position = self.get_user_preference(user_id, 'has_active_position', False)
            
            await query.edit_message_text(
                "🤖 *XAUUSD Trading Bot*\n\nPilih menu di bawah:",
                reply_markup=get_enhanced_main_menu(user_id, is_monitoring, has_position),
                parse_mode='Markdown'
            )
        elif action == 'settings':
            await query.edit_message_text(
                "⚙️ *Pengaturan*\n\nPilih pengaturan yang ingin diubah:",
                reply_markup=get_settings_menu(user_id),
                parse_mode='Markdown'
            )
        elif action == 'reports':
            await query.edit_message_text(
                "📊 *Laporan Trading*\n\nPilih jenis laporan:",
                reply_markup=get_reports_menu(user_id),
                parse_mode='Markdown'
            )
        elif action == 'help':
            await query.edit_message_text(
                "❓ *Bantuan*\n\nPilih topik bantuan:",
                reply_markup=get_help_menu(user_id),
                parse_mode='Markdown'
            )
    
    async def _handle_settings_action(self, query, user_id: int, action: str, params: List[str]):
        """Handle settings actions"""
        if action == 'lang':
            await query.edit_message_text(
                "🌐 *Pilih Bahasa*",
                reply_markup=get_language_menu(user_id),
                parse_mode='Markdown'
            )
        elif action == 'notif':
            current = self.get_user_preference(user_id, 'notification_level', 'all')
            await query.edit_message_text(
                "🔔 *Pengaturan Notifikasi*\n\nPilih level notifikasi:",
                reply_markup=get_notification_menu(user_id, current),
                parse_mode='Markdown'
            )
    
    async def _handle_language_action(self, query, user_id: int, lang: str):
        """Handle language selection"""
        set_user_language(user_id, lang)
        self.set_user_preference(user_id, 'language', lang)
        
        lang_name = "Bahasa Indonesia" if lang == 'id' else "English"
        
        await query.edit_message_text(
            f"✅ Bahasa diubah ke *{lang_name}*",
            reply_markup=get_settings_menu(user_id),
            parse_mode='Markdown'
        )
    
    async def _handle_notification_action(self, query, user_id: int, level: str):
        """Handle notification level change"""
        self.set_user_preference(user_id, 'notification_level', level)
        
        level_names = {
            'all': 'Semua Notifikasi',
            'signals': 'Sinyal Saja',
            'critical': 'Critical Only',
            'off': 'Dimatikan'
        }
        
        await query.edit_message_text(
            f"✅ Notifikasi diubah ke *{level_names.get(level, level)}*",
            reply_markup=get_notification_menu(user_id, level),
            parse_mode='Markdown'
        )
    
    async def _handle_report_action(self, query, user_id: int, report_type: str):
        """Handle report requests"""
        await query.edit_message_text(
            f"📊 Generating {report_type} report...",
            parse_mode='Markdown'
        )
    
    async def _handle_signal_action(self, query, user_id: int, action: str, params: List[str]):
        """Handle signal-related actions"""
        signal_id = params[0] if params else None
        
        if action == 'update' and signal_id:
            await query.edit_message_text("🔄 Updating signal status...")
        elif action == 'close' and signal_id:
            await query.edit_message_text(
                "⚠️ *Konfirmasi Close Position*\n\nApakah Anda yakin ingin menutup posisi ini?",
                reply_markup=get_confirmation_menu(user_id, 'close_signal', signal_id),
                parse_mode='Markdown'
            )
    
    async def _handle_position_action(self, query, user_id: int, action: str, params: List[str]):
        """Handle position management actions"""
        position_id = params[0] if params else None
        
        if action == 'refresh' and position_id:
            await query.edit_message_text("🔄 Refreshing position...")
        elif action == 'close' and position_id:
            await query.edit_message_text(
                "⚠️ *Konfirmasi Close*\n\nYakin ingin menutup posisi?",
                reply_markup=get_confirmation_menu(user_id, 'close_position', position_id),
                parse_mode='Markdown'
            )
    
    async def _handle_confirmation(self, query, user_id: int, result: str, params: List[str]):
        """Handle confirmation dialogs"""
        action = params[0] if params else None
        action_data = params[1] if len(params) > 1 else None
        
        if result == 'yes':
            await query.edit_message_text(f"✅ Action {action} confirmed")
        else:
            await query.edit_message_text(
                "❌ Dibatalkan",
                reply_markup=get_enhanced_main_menu(user_id)
            )
    
    async def _handle_help_action(self, query, user_id: int, topic: str):
        """Handle help topics"""
        help_texts = {
            'signals': (
                "📡 *Tentang Sinyal Trading*\n\n"
                "Bot ini menggunakan kombinasi indikator:\n"
                "• EMA (5, 10, 20)\n"
                "• RSI (14)\n"
                "• MACD (12, 26, 9)\n"
                "• Stochastic (14, 3)\n\n"
                "Sinyal digenerate saat semua kondisi terpenuhi."
            ),
            'charts': (
                "📊 *Cara Membaca Chart*\n\n"
                "• 🟢 Hijau = Bullish/BUY\n"
                "• 🔴 Merah = Bearish/SELL\n"
                "• Garis biru = EMA\n"
                "• Entry line = Harga masuk\n"
                "• SL/TP lines = Stop Loss/Take Profit"
            ),
            'risk': (
                "⚠️ *Risk Management*\n\n"
                "• Gunakan lot size kecil (0.01)\n"
                "• Max risk 1-2% per trade\n"
                "• Selalu pasang Stop Loss\n"
                "• Jangan overtrade"
            ),
            'tips': (
                "💡 *Tips Trading*\n\n"
                "• Ikuti trend utama\n"
                "• Trading di session aktif\n"
                "• Hindari news high impact\n"
                "• Sabar menunggu setup"
            ),
            'faq': (
                "🆘 *FAQ*\n\n"
                "Q: Berapa sinyal per hari?\n"
                "A: Unlimited, tergantung kondisi market\n\n"
                "Q: Akurasi sinyal?\n"
                "A: Tergantung market, biasanya 60-70%"
            ),
            'contact': (
                "📞 *Kontak*\n\n"
                "Untuk bantuan lebih lanjut, hubungi admin."
            )
        }
        
        text = help_texts.get(topic, "Topik tidak ditemukan.")
        
        await query.edit_message_text(
            text,
            reply_markup=get_help_menu(user_id),
            parse_mode='Markdown'
        )
    
    async def _handle_onboarding(self, query, user_id: int, action: str, params: List[str]):
        """Handle onboarding flow"""
        onboarding_texts = {
            1: (
                "👋 *Selamat Datang!*\n\n"
                "Bot ini akan membantu Anda mendapatkan sinyal trading XAUUSD secara real-time.\n\n"
                "Tekan *Mulai* untuk melanjutkan."
            ),
            2: (
                "📡 *Cara Mendapatkan Sinyal*\n\n"
                "1. Gunakan /monitor untuk auto signal\n"
                "2. Gunakan /getsignal untuk manual signal\n"
                "3. Setiap sinyal dilengkapi SL/TP\n\n"
                "Tekan *Lanjut* untuk tips penting."
            ),
            3: (
                "⚠️ *Penting!*\n\n"
                "• Trading adalah aktivitas berisiko tinggi\n"
                "• Gunakan modal yang siap hilang\n"
                "• Bot hanya memberikan sinyal, keputusan ada di Anda\n\n"
                "Tekan *Selesai* untuk mulai trading!"
            )
        }
        
        if action == 'step':
            step = int(params[0]) if params else 1
            text = onboarding_texts.get(step, onboarding_texts[1])
            
            await query.edit_message_text(
                text,
                reply_markup=get_onboarding_menu(user_id, step),
                parse_mode='Markdown'
            )
        elif action == 'complete' or action == 'skip':
            self.set_user_preference(user_id, 'onboarding_complete', True)
            
            await query.edit_message_text(
                "🎉 *Setup Selesai!*\n\nAnda siap menggunakan bot. Gunakan menu di bawah untuk memulai.",
                reply_markup=get_enhanced_main_menu(user_id),
                parse_mode='Markdown'
            )
    
    async def _handle_export(self, query, user_id: int, format_type: str, params: List[str]):
        """Handle data export"""
        report_type = params[0] if params else 'daily'
        
        await query.edit_message_text(
            f"📥 Generating {report_type} report in {format_type.upper()} format..."
        )
    
    async def _handle_navigation(self, query, user_id: int, action: str):
        """Handle navigation actions"""
        if action == 'back':
            await query.edit_message_text(
                "🤖 *XAUUSD Trading Bot*\n\nPilih menu di bawah:",
                reply_markup=get_enhanced_main_menu(user_id),
                parse_mode='Markdown'
            )
        elif action == 'refresh':
            await query.answer("🔄 Refreshing...", show_alert=False)
