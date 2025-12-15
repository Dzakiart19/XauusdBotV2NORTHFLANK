"""Unit tests for menu module."""
import pytest
from unittest.mock import MagicMock
from telegram import InlineKeyboardMarkup, InlineKeyboardButton
from bot.menu import (
    MenuBuilder,
    get_main_menu,
    get_settings_menu,
    get_language_menu,
    get_trading_menu,
    get_analysis_menu,
    get_onboarding_menu,
    get_confirmation_menu,
)
from bot.i18n import set_user_language


class TestMenuBuilder:
    """Tests for MenuBuilder class."""
    
    def test_add_button(self):
        """Should add a callback button."""
        builder = MenuBuilder()
        builder.add_button('Test', 'callback:test')
        markup = builder.build()
        
        assert isinstance(markup, InlineKeyboardMarkup)
        assert len(markup.inline_keyboard) == 1
        assert len(markup.inline_keyboard[0]) == 1
        assert markup.inline_keyboard[0][0].text == 'Test'
        assert markup.inline_keyboard[0][0].callback_data == 'callback:test'
    
    def test_add_url_button(self):
        """Should add a URL button."""
        builder = MenuBuilder()
        builder.add_button('Link', 'ignored', url='https://example.com')
        markup = builder.build()
        
        assert markup.inline_keyboard[0][0].url == 'https://example.com'
    
    def test_new_row(self):
        """Should create multiple rows."""
        builder = MenuBuilder()
        builder.add_button('Btn1', 'cb1')
        builder.new_row()
        builder.add_button('Btn2', 'cb2')
        markup = builder.build()
        
        assert len(markup.inline_keyboard) == 2
        assert len(markup.inline_keyboard[0]) == 1
        assert len(markup.inline_keyboard[1]) == 1
    
    def test_multiple_buttons_per_row(self):
        """Should add multiple buttons to same row."""
        builder = MenuBuilder()
        builder.add_button('Btn1', 'cb1')
        builder.add_button('Btn2', 'cb2')
        markup = builder.build()
        
        assert len(markup.inline_keyboard) == 1
        assert len(markup.inline_keyboard[0]) == 2
    
    def test_add_localized_button(self):
        """Should add button with localized text."""
        user_id = 12345
        set_user_language(user_id, 'en')
        
        builder = MenuBuilder(user_id)
        builder.add_localized_button('btn_help', 'menu:help')
        markup = builder.build()
        
        assert 'Help' in markup.inline_keyboard[0][0].text


class TestMainMenu:
    """Tests for main menu generation."""
    
    def test_main_menu_structure(self):
        """Should create main menu with correct structure."""
        user_id = 12345
        markup = get_main_menu(user_id)
        
        assert isinstance(markup, InlineKeyboardMarkup)
        assert len(markup.inline_keyboard) >= 3
    
    def test_main_menu_indonesian(self):
        """Should use Indonesian labels."""
        user_id = 12346
        set_user_language(user_id, 'id')
        markup = get_main_menu(user_id)
        
        all_text = ' '.join(
            btn.text for row in markup.inline_keyboard for btn in row
        )
        assert 'Monitor' in all_text or 'Bantuan' in all_text
    
    def test_main_menu_english(self):
        """Should use English labels."""
        user_id = 12347
        set_user_language(user_id, 'en')
        markup = get_main_menu(user_id)
        
        all_text = ' '.join(
            btn.text for row in markup.inline_keyboard for btn in row
        )
        assert 'Monitor' in all_text or 'Help' in all_text


class TestSettingsMenu:
    """Tests for settings menu."""
    
    def test_settings_menu_has_language_button(self):
        """Should include language button."""
        markup = get_settings_menu(12345)
        
        all_data = [
            btn.callback_data 
            for row in markup.inline_keyboard 
            for btn in row
        ]
        assert 'menu:settings:language' in all_data
    
    def test_settings_menu_has_back_button(self):
        """Should include back button."""
        markup = get_settings_menu(12345)
        
        all_data = [
            btn.callback_data 
            for row in markup.inline_keyboard 
            for btn in row
        ]
        assert 'menu:main' in all_data


class TestLanguageMenu:
    """Tests for language menu."""
    
    def test_language_menu_has_both_options(self):
        """Should have both Indonesian and English options."""
        markup = get_language_menu(12345)
        
        all_data = [
            btn.callback_data 
            for row in markup.inline_keyboard 
            for btn in row
        ]
        assert 'menu:lang:id' in all_data
        assert 'menu:lang:en' in all_data


class TestTradingMenu:
    """Tests for trading menu."""
    
    def test_trading_menu_start_monitoring(self):
        """Should show start button when not monitoring."""
        markup = get_trading_menu(12345, is_monitoring=False)
        
        all_data = [
            btn.callback_data 
            for row in markup.inline_keyboard 
            for btn in row
        ]
        assert 'menu:monitor:start' in all_data
    
    def test_trading_menu_stop_monitoring(self):
        """Should show stop button when monitoring."""
        markup = get_trading_menu(12345, is_monitoring=True)
        
        all_data = [
            btn.callback_data 
            for row in markup.inline_keyboard 
            for btn in row
        ]
        assert 'menu:monitor:stop' in all_data


class TestOnboardingMenu:
    """Tests for onboarding menu."""
    
    def test_onboarding_step1(self):
        """Should show next button on step 1."""
        markup = get_onboarding_menu(12345, step=1)
        
        all_data = [
            btn.callback_data 
            for row in markup.inline_keyboard 
            for btn in row
        ]
        assert 'onboard:next:2' in all_data
        assert 'onboard:skip' in all_data
    
    def test_onboarding_step3(self):
        """Should show start monitoring on step 3."""
        markup = get_onboarding_menu(12345, step=3)
        
        all_data = [
            btn.callback_data 
            for row in markup.inline_keyboard 
            for btn in row
        ]
        assert 'menu:monitor:start' in all_data


class TestConfirmationMenu:
    """Tests for confirmation menu."""
    
    def test_confirmation_menu(self):
        """Should have yes and no buttons."""
        markup = get_confirmation_menu(12345, 'delete')
        
        all_data = [
            btn.callback_data 
            for row in markup.inline_keyboard 
            for btn in row
        ]
        assert 'confirm:delete:yes' in all_data
        assert 'confirm:delete:no' in all_data
