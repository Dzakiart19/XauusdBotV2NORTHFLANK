"""
Interactive Telegram Menu System.

Provides inline keyboard menus for better user experience.
"""
import logging
from typing import Optional, List, Dict, Any
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes
from bot.i18n import get_text, t

logger = logging.getLogger('Menu')


class MenuBuilder:
    """Builder for creating Telegram inline keyboard menus."""
    
    def __init__(self, user_id: Optional[int] = None):
        self.user_id = user_id
        self.buttons: List[List[InlineKeyboardButton]] = []
        self.current_row: List[InlineKeyboardButton] = []
    
    def add_button(self, text: str, callback_data: str, 
                   url: Optional[str] = None) -> 'MenuBuilder':
        """Add a button to the current row."""
        if url:
            btn = InlineKeyboardButton(text, url=url)
        else:
            btn = InlineKeyboardButton(text, callback_data=callback_data)
        self.current_row.append(btn)
        return self
    
    def add_localized_button(self, key: str, callback_data: str) -> 'MenuBuilder':
        """Add a button with localized text."""
        text = t(key, self.user_id)
        return self.add_button(text, callback_data)
    
    def new_row(self) -> 'MenuBuilder':
        """Start a new row of buttons."""
        if self.current_row:
            self.buttons.append(self.current_row)
            self.current_row = []
        return self
    
    def build(self) -> InlineKeyboardMarkup:
        """Build and return the keyboard markup."""
        if self.current_row:
            self.buttons.append(self.current_row)
        return InlineKeyboardMarkup(self.buttons)


def get_main_menu(user_id: int) -> InlineKeyboardMarkup:
    """
    Get the main menu keyboard.
    
    Args:
        user_id: Telegram user ID for localization
    
    Returns:
        InlineKeyboardMarkup
    """
    return (MenuBuilder(user_id)
        .add_localized_button('btn_start_monitoring', 'menu:monitor:start')
        .add_localized_button('btn_status', 'menu:status')
        .new_row()
        .add_localized_button('btn_get_signal', 'menu:signal')
        .add_localized_button('btn_performance', 'menu:performance')
        .new_row()
        .add_localized_button('btn_history', 'menu:history')
        .add_localized_button('btn_settings', 'menu:settings')
        .new_row()
        .add_localized_button('btn_help', 'menu:help')
        .build())


def get_trading_menu(user_id: int, is_monitoring: bool = False) -> InlineKeyboardMarkup:
    """
    Get the trading menu keyboard.
    
    Args:
        user_id: Telegram user ID for localization
        is_monitoring: Whether monitoring is currently active
    
    Returns:
        InlineKeyboardMarkup
    """
    builder = MenuBuilder(user_id)
    
    if is_monitoring:
        builder.add_localized_button('btn_stop_monitoring', 'menu:monitor:stop')
    else:
        builder.add_localized_button('btn_start_monitoring', 'menu:monitor:start')
    
    builder.add_localized_button('btn_get_signal', 'menu:signal')
    builder.new_row()
    builder.add_localized_button('btn_status', 'menu:status')
    builder.add_localized_button('btn_back', 'menu:main')
    
    return builder.build()


def get_settings_menu(user_id: int) -> InlineKeyboardMarkup:
    """
    Get the settings menu keyboard.
    
    Args:
        user_id: Telegram user ID for localization
    
    Returns:
        InlineKeyboardMarkup
    """
    return (MenuBuilder(user_id)
        .add_localized_button('btn_language', 'menu:settings:language')
        .add_localized_button('btn_notifications', 'menu:settings:notifications')
        .new_row()
        .add_localized_button('btn_risk_level', 'menu:settings:risk')
        .add_localized_button('btn_timeframe', 'menu:settings:timeframe')
        .new_row()
        .add_localized_button('btn_back', 'menu:main')
        .build())


def get_language_menu(user_id: int) -> InlineKeyboardMarkup:
    """
    Get the language selection menu.
    
    Args:
        user_id: Telegram user ID
    
    Returns:
        InlineKeyboardMarkup
    """
    return (MenuBuilder(user_id)
        .add_button('🇮🇩 Indonesia', 'menu:lang:id')
        .add_button('🇬🇧 English', 'menu:lang:en')
        .new_row()
        .add_localized_button('btn_back', 'menu:settings')
        .build())


def get_confirmation_menu(user_id: int, action: str) -> InlineKeyboardMarkup:
    """
    Get a confirmation menu.
    
    Args:
        user_id: Telegram user ID
        action: Action to confirm
    
    Returns:
        InlineKeyboardMarkup
    """
    return (MenuBuilder(user_id)
        .add_button('✅ Ya', f'confirm:{action}:yes')
        .add_button('❌ Tidak', f'confirm:{action}:no')
        .build())


def get_position_menu(user_id: int, position_id: str) -> InlineKeyboardMarkup:
    """
    Get position management menu.
    
    Args:
        user_id: Telegram user ID
        position_id: Position identifier
    
    Returns:
        InlineKeyboardMarkup
    """
    return (MenuBuilder(user_id)
        .add_button('📊 Update', f'pos:{position_id}:update')
        .add_button('❌ Close', f'pos:{position_id}:close')
        .new_row()
        .add_button('🔄 Modify SL', f'pos:{position_id}:modify_sl')
        .add_button('🎯 Modify TP', f'pos:{position_id}:modify_tp')
        .new_row()
        .add_localized_button('btn_back', 'menu:main')
        .build())


def get_analysis_menu(user_id: int) -> InlineKeyboardMarkup:
    """
    Get analysis menu keyboard.
    
    Args:
        user_id: Telegram user ID
    
    Returns:
        InlineKeyboardMarkup
    """
    return (MenuBuilder(user_id)
        .add_localized_button('btn_performance', 'menu:analysis:performance')
        .add_localized_button('btn_history', 'menu:analysis:history')
        .new_row()
        .add_button('📅 Hari Ini', 'menu:analysis:today')
        .add_button('📆 Minggu Ini', 'menu:analysis:week')
        .new_row()
        .add_localized_button('btn_back', 'menu:main')
        .build())


def get_onboarding_menu(user_id: int, step: int) -> InlineKeyboardMarkup:
    """
    Get onboarding step menu.
    
    Args:
        user_id: Telegram user ID
        step: Current onboarding step (1-3)
    
    Returns:
        InlineKeyboardMarkup
    """
    builder = MenuBuilder(user_id)
    
    if step < 3:
        builder.add_button('Lanjut ➡️', f'onboard:next:{step + 1}')
    else:
        builder.add_localized_button('btn_start_monitoring', 'menu:monitor:start')
    
    if step > 1:
        builder.add_button('⬅️ Kembali', f'onboard:back:{step - 1}')
    
    builder.new_row()
    builder.add_button('⏭️ Lewati', 'onboard:skip')
    
    return builder.build()


async def handle_menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE,
                               callback_data: str, bot_instance) -> None:
    """
    Handle menu callback queries.
    
    Args:
        update: Telegram update
        context: Callback context
        callback_data: Callback data string
        bot_instance: TradingBot instance
    """
    query = update.callback_query
    user_id = query.from_user.id
    
    await query.answer()
    
    parts = callback_data.split(':')
    menu_type = parts[0]
    action = parts[1] if len(parts) > 1 else None
    param = parts[2] if len(parts) > 2 else None
    
    try:
        if menu_type == 'menu':
            await _handle_menu_action(query, user_id, action, param, bot_instance)
        elif menu_type == 'confirm':
            await _handle_confirmation(query, user_id, action, param, bot_instance)
        elif menu_type == 'pos':
            await _handle_position_action(query, user_id, action, param, bot_instance)
        elif menu_type == 'onboard':
            await _handle_onboarding(query, user_id, action, param)
        else:
            logger.warning(f"Unknown menu type: {menu_type}")
    except Exception as e:
        logger.error(f"Error handling menu callback: {e}")
        await query.edit_message_text(t('error', user_id))


async def _handle_menu_action(query, user_id: int, action: str, 
                              param: Optional[str], bot_instance) -> None:
    """Handle menu navigation actions."""
    if action == 'main':
        await query.edit_message_text(
            t('menu_main', user_id),
            reply_markup=get_main_menu(user_id),
            parse_mode='Markdown'
        )
    elif action == 'settings':
        await query.edit_message_text(
            t('settings_title', user_id),
            reply_markup=get_settings_menu(user_id),
            parse_mode='Markdown'
        )
    elif action == 'help':
        await query.edit_message_text(
            t('help_commands', user_id),
            reply_markup=get_main_menu(user_id),
            parse_mode='Markdown'
        )
    elif action == 'lang':
        await query.edit_message_text(
            t('btn_language', user_id),
            reply_markup=get_language_menu(user_id),
            parse_mode='Markdown'
        )


async def _handle_confirmation(query, user_id: int, action: str,
                               confirm: str, bot_instance) -> None:
    """Handle confirmation dialogs."""
    if confirm == 'yes':
        await query.edit_message_text(f"✅ Confirmed: {action}")
    else:
        await query.edit_message_text(
            t('menu_main', user_id),
            reply_markup=get_main_menu(user_id)
        )


async def _handle_position_action(query, user_id: int, position_id: str,
                                  action: str, bot_instance) -> None:
    """Handle position management actions."""
    await query.edit_message_text(f"Position {position_id}: {action}")


async def _handle_onboarding(query, user_id: int, action: str,
                            step: Optional[str]) -> None:
    """Handle onboarding flow."""
    from bot.i18n import get_text
    
    if action == 'skip':
        await query.edit_message_text(
            t('menu_main', user_id),
            reply_markup=get_main_menu(user_id)
        )
        return
    
    step_num = int(step) if step else 1
    step_key = f'onboarding_step{step_num}'
    
    await query.edit_message_text(
        get_text(step_key, user_id),
        reply_markup=get_onboarding_menu(user_id, step_num),
        parse_mode='Markdown'
    )
