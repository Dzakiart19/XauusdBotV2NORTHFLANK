"""
Message Templates untuk Telegram
Format pesan yang konsisten dan rapi
"""
from typing import Optional, Union, Any
from datetime import datetime
import pytz


def _safe_numeric(value: Any, default: Union[int, float] = 0) -> Union[int, float]:
    """Helper untuk memastikan nilai adalah numeric yang valid"""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_string(value: Any, default: str = 'N/A') -> str:
    """Helper untuk memastikan nilai adalah string yang valid"""
    if value is None:
        return default
    return str(value) if value else default


class MessageFormatter:
    """Helper untuk format pesan Telegram yang rapi"""
    
    @staticmethod
    def escape_markdown(text: str) -> str:
        """Escape karakter special untuk Markdown"""
        special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        for char in special_chars:
            text = text.replace(char, f'\\{char}')
        return text
    
    @staticmethod
    def progress_bar(current: float, target: float, total_length: int = 10) -> str:
        """Buat progress bar visual"""
        if target == 0:
            return "▱" * total_length
        
        percentage = min(abs(current / target), 1.0)
        filled = int(percentage * total_length)
        empty = total_length - filled
        
        return "▰" * filled + "▱" * empty
    
    @staticmethod
    def signal_alert(signal: dict, signal_source: str = 'auto', config=None) -> str:
        """Format pesan alert sinyal trading - Format Profesional"""
        signal_type = signal.get('signal', 'UNKNOWN')
        signal_type = _safe_string(signal_type, 'UNKNOWN')
        direction_icon = "🟢" if signal_type == 'BUY' else "🔴"
        
        entry = _safe_numeric(signal.get('entry_price', 0), 0)
        sl = _safe_numeric(signal.get('stop_loss', 0), 0)
        tp = _safe_numeric(signal.get('take_profit', 0), 0)
        
        sl_pips = _safe_numeric(signal.get('sl_pips', 0), 0)
        tp_pips = _safe_numeric(signal.get('tp_pips', 0), 0)
        rr_ratio = _safe_numeric(signal.get('rr_ratio', 0), 0)
        
        lot_size = _safe_numeric(signal.get('lot_size', 0.01), 0.01)
        risk_percent = _safe_numeric(signal.get('risk_percent', 1.0), 1.0)
        risk_amount = _safe_numeric(signal.get('risk_amount', 0), 0)
        account_balance = _safe_numeric(signal.get('account_balance', 0), 0)
        
        if config and account_balance == 0:
            account_balance = _safe_numeric(getattr(config, 'ACCOUNT_BALANCE', 0), 0)
        if config and risk_percent == 1.0:
            risk_percent = _safe_numeric(getattr(config, 'RISK_PER_TRADE_PERCENT', 1.0), 1.0)
        if risk_amount == 0 and account_balance > 0:
            risk_amount = account_balance * risk_percent / 100
        
        trend_desc = signal.get('trend_description', 'N/A')
        trend_status_val = signal.get('trend_status')
        if trend_status_val is not None:
            trend_status = _safe_string(trend_status_val, 'N/A')
        else:
            trend_status = _safe_string(trend_desc, 'N/A')
        
        momentum_status = _safe_string(signal.get('momentum_status'), 'N/A')
        volume_status = _safe_string(signal.get('volume_status'), 'N/A')
        vwap_status = _safe_string(signal.get('vwap_status'), 'N/A')
        
        timeframe = _safe_string(signal.get('timeframe'), 'M1')
        timestamp = datetime.now(pytz.timezone('Asia/Jakarta')).strftime('%H:%M:%S WIB')
        
        msg = (
            f"{direction_icon} *SIGNAL {signal_type} - XAUUSD*\n"
            f"{'━' * 22}\n"
            f"📊 Entry: `${entry:.2f}`\n"
            f"🛡️ Stop Loss: `${sl:.2f}` ({sl_pips:.1f} pips)\n"
            f"🎯 Take Profit: `${tp:.2f}` ({tp_pips:.1f} pips)\n"
            f"📈 Risk:Reward = 1:{rr_ratio:.1f}\n\n"
            f"💰 *Risk Management:*\n"
            f"• Lot Size: {lot_size:.2f}\n"
            f"• Risk: {risk_percent:.1f}% (${risk_amount:.2f})\n"
            f"• Modal: ${account_balance:.2f}\n\n"
            f"📋 *Konfirmasi:*\n"
            f"• Trend: {trend_status}\n"
            f"• Momentum: {momentum_status}\n"
            f"• Volume: {volume_status}\n"
            f"• VWAP: {vwap_status}\n\n"
            f"⏰ Waktu: {timestamp}\n"
            f"📊 Timeframe: {timeframe}\n"
            f"{'━' * 22}"
        )
        
        return msg
    
    @staticmethod
    def position_update(position_data: dict) -> str:
        """Format update posisi real-time dengan info trailing stop dan dynamic SL"""
        signal_type = _safe_string(position_data.get('signal_type'), 'UNKNOWN')
        direction_icon = "🟢" if signal_type == 'BUY' else "🔴"
        
        entry = _safe_numeric(position_data.get('entry_price', 0), 0)
        current = _safe_numeric(position_data.get('current_price', 0), 0)
        sl = _safe_numeric(position_data.get('stop_loss', 0), 0)
        tp = _safe_numeric(position_data.get('take_profit', 0), 0)
        pl = _safe_numeric(position_data.get('unrealized_pl', 0), 0)
        original_sl = _safe_numeric(position_data.get('original_stop_loss', sl), sl)
        sl_adjustment_count = int(_safe_numeric(position_data.get('sl_adjustment_count', 0), 0))
        max_profit = _safe_numeric(position_data.get('max_profit_reached', 0), 0)
        
        if entry == 0:
            price_change = 0
            price_change_pct = 0
        else:
            price_change = current - entry
            price_change_pct = (price_change / entry) * 100
        
        if signal_type == 'BUY':
            tp_distance_total = tp - entry
            tp_distance_current = current - entry
            sl_distance_total = entry - sl
            sl_distance_current = entry - current
        else:
            tp_distance_total = entry - tp
            tp_distance_current = entry - current
            sl_distance_total = sl - entry
            sl_distance_current = current - entry
        
        tp_progress = max(0, min(100, (tp_distance_current / tp_distance_total * 100) if tp_distance_total > 0 else 0))
        sl_progress = max(0, min(100, (sl_distance_current / sl_distance_total * 100) if sl_distance_total > 0 else 0))
        
        tp_bar = MessageFormatter.progress_bar(tp_distance_current, tp_distance_total, 10)
        
        pl_icon = "💰" if pl >= 0 else "📉"
        pl_text = f"+${pl:.2f}" if pl >= 0 else f"-${abs(pl):.2f}"
        
        sl_status = ""
        trailing_info = ""
        if sl_adjustment_count > 0:
            if max_profit > 0:
                sl_status = f"\n💎 _Trailing Stop ({sl_adjustment_count}x)_"
                trailing_info = f"\n🔒 *Profit Terkunci:* `+${max_profit:.2f}`"
            else:
                sl_status = f"\n🛡️ _Dynamic SL ({sl_adjustment_count}x)_"
        
        msg = (
            f"{direction_icon} *POSISI {signal_type} AKTIF*\n"
            f"{'━' * 32}\n\n"
            f"📍 *Entry:* `${entry:.2f}`\n"
            f"📊 *Current:* `${current:.2f}` ({price_change_pct:+.3f}%)\n"
            f"{pl_icon} *P/L:* `{pl_text}`{trailing_info}\n\n"
            f"🎯 *Progress ke TP:*\n"
            f"{tp_bar} {tp_progress:.1f}%\n"
            f"Target: `${tp:.2f}`\n\n"
            f"🛡️ *Stop Loss:* `${sl:.2f}`{sl_status}\n"
            f"⚠️ Risk: {sl_progress:.1f}%\n\n"
            f"⏰ {datetime.now(pytz.timezone('Asia/Jakarta')).strftime('%H:%M:%S WIB')}"
        )
        
        return msg
    
    @staticmethod
    def trade_exit(exit_data: dict, pip_value: float = 10.0) -> str:
        """Format pesan trade exit - Format Profesional"""
        result = _safe_string(exit_data.get('result'), 'UNKNOWN')
        signal_type = _safe_string(exit_data.get('signal_type'), 'UNKNOWN')
        entry = _safe_numeric(exit_data.get('entry_price', 0), 0)
        exit_price = _safe_numeric(exit_data.get('exit_price', 0), 0)
        pl = _safe_numeric(exit_data.get('actual_pl', 0), 0)
        reason = _safe_string(exit_data.get('reason'), 'CLOSED')
        duration = exit_data.get('duration', 'N/A')
        
        result_icon = "✅" if result == 'WIN' else "❌"
        result_text = "TP HIT" if reason == 'TP_HIT' else ("SL HIT" if reason in ['SL_HIT', 'DYNAMIC_SL_HIT'] else result)
        
        price_diff = abs(exit_price - entry)
        pl_pips = price_diff * _safe_numeric(pip_value, 10.0)
        
        pl_emoji = "💰" if pl >= 0 else "📉"
        pl_text = f"+${pl:.2f}" if pl >= 0 else f"-${abs(pl):.2f}"
        
        if isinstance(duration, (int, float)):
            duration = _safe_numeric(duration, 0)
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            if hours > 0:
                duration_str = f"{hours}h {minutes}m"
            else:
                duration_str = f"{minutes}m"
        else:
            duration_str = str(duration) if duration else "N/A"
        
        msg = (
            f"{result_icon} *TRADE CLOSED - {result_text}*\n"
            f"{'━' * 22}\n"
            f"📊 Entry: `${entry:.2f}`\n"
            f"📊 Exit: `${exit_price:.2f}`\n"
            f"💰 P/L: {pl_emoji} {pl_text} ({pl_pips:.1f} pips)\n"
            f"⏱️ Duration: {duration_str}\n"
            f"{'━' * 22}"
        )
        
        return msg
    
    @staticmethod
    def waiting_for_signal(signal_source: str = 'auto') -> str:
        """Format pesan menunggu sinyal"""
        icon = "🤖" if signal_source == 'auto' else "👤"
        mode = "otomatis" if signal_source == 'auto' else "manual"
        
        return (
            f"{icon} *Monitoring Aktif*\n"
            f"{'━' * 32}\n\n"
            f"⏳ Menunggu sinyal {mode}...\n"
            f"📊 Menganalisis market XAUUSD\n\n"
            f"⏰ {datetime.now(pytz.timezone('Asia/Jakarta')).strftime('%H:%M:%S WIB')}"
        )
    
    @staticmethod
    def session_blocked(active_source: str, requested_source: str) -> str:
        """Format pesan ketika sinyal diblok karena ada sesi aktif"""
        active_source = _safe_string(active_source, 'unknown')
        requested_source = _safe_string(requested_source, 'unknown')
        active_icon = "🤖" if active_source == 'auto' else "👤"
        requested_icon = "🤖" if requested_source == 'auto' else "👤"
        
        return (
            f"⛔ *Sinyal Diblok*\n"
            f"{'━' * 32}\n\n"
            f"{active_icon} Ada sinyal *{active_source.upper()}* yang masih aktif!\n\n"
            f"{requested_icon} Tidak bisa buat sinyal *{requested_source.upper()}* sekarang.\n\n"
            f"⏳ Tunggu sampai posisi selesai dulu\n"
            f"(TP/SL tercapai)\n\n"
            f"💡 Cek status: /status"
        )
    
    @staticmethod
    def no_active_position() -> str:
        """Format pesan tidak ada posisi aktif"""
        return (
            f"ℹ️ *Status Posisi*\n"
            f"{'━' * 32}\n\n"
            f"📭 Tidak ada posisi aktif\n\n"
            f"💡 Gunakan:\n"
            f"  • /monitor - Monitoring otomatis\n"
            f"  • /getsignal - Sinyal manual\n\n"
            f"⏰ {datetime.now(pytz.timezone('Asia/Jakarta')).strftime('%H:%M:%S WIB')}"
        )
    
    @staticmethod
    def statistics_summary(stats: dict) -> str:
        """Format statistik trading - untuk /performa"""
        total_trades = _safe_numeric(stats.get('total_trades', 0), 0)
        wins = _safe_numeric(stats.get('wins', 0), 0)
        losses = _safe_numeric(stats.get('losses', 0), 0)
        total_profit = _safe_numeric(stats.get('total_profit', 0), 0)
        win_rate = _safe_numeric(stats.get('win_rate', 0), 0)
        
        win_rate_emoji = "🔥" if win_rate >= 70 else "💪" if win_rate >= 50 else "📊"
        pl_emoji = "💰" if total_profit >= 0 else "📉"
        pl_text = f"+${total_profit:.2f}" if total_profit >= 0 else f"-${abs(total_profit):.2f}"
        
        return (
            f"📊 *Statistik Trading*\n"
            f"{'━' * 22}\n\n"
            f"📈 *Total Trades:* {int(total_trades)}\n"
            f"✅ *Wins:* {int(wins)}\n"
            f"❌ *Losses:* {int(losses)}\n"
            f"{win_rate_emoji} *Win Rate:* {win_rate:.1f}%\n\n"
            f"{pl_emoji} *Total P/L:* `{pl_text}`\n\n"
            f"⏰ {datetime.now(pytz.timezone('Asia/Jakarta')).strftime('%Y-%m-%d %H:%M WIB')}"
        )
    
    @staticmethod
    def daily_stats(stats: dict) -> str:
        """Format statistik harian - untuk /stats command"""
        total_trades = _safe_numeric(stats.get('total_trades', 0), 0)
        wins = _safe_numeric(stats.get('wins', 0), 0)
        losses = _safe_numeric(stats.get('losses', 0), 0)
        win_rate = _safe_numeric(stats.get('win_rate', 0), 0)
        net_pl = _safe_numeric(stats.get('total_pl', 0), 0)
        profit_factor = stats.get('profit_factor', 'N/A')
        
        avg_win = _safe_numeric(stats.get('avg_win', 0), 0)
        avg_loss = _safe_numeric(stats.get('avg_loss', 0), 0)
        avg_rr = (avg_win / avg_loss) if avg_loss > 0 else 0
        
        pl_emoji = "💰" if net_pl >= 0 else "📉"
        pl_text = f"+${net_pl:.2f}" if net_pl >= 0 else f"-${abs(net_pl):.2f}"
        
        if isinstance(profit_factor, (int, float)):
            profit_factor_val = _safe_numeric(profit_factor, 0)
            profit_factor_str = f"{profit_factor_val:.2f}"
        else:
            profit_factor_str = _safe_string(profit_factor, 'N/A')
        
        avg_rr_str = f"1:{avg_rr:.1f}" if avg_rr > 0 else "N/A"
        
        date_str = stats.get('date')
        if date_str is None:
            date_str = datetime.now(pytz.timezone('Asia/Jakarta')).strftime('%Y-%m-%d')
        else:
            date_str = _safe_string(date_str, datetime.now(pytz.timezone('Asia/Jakarta')).strftime('%Y-%m-%d'))
        
        return (
            f"📊 *STATISTIK HARI INI*\n"
            f"{'━' * 22}\n"
            f"📈 Total Trade: {int(total_trades)}\n"
            f"✅ Win: {int(wins)} | ❌ Loss: {int(losses)}\n"
            f"📊 Win Rate: {win_rate:.1f}%\n"
            f"💰 Net P/L: {pl_emoji} {pl_text}\n"
            f"📈 Profit Factor: {profit_factor_str}\n"
            f"🎯 Avg RR: {avg_rr_str}\n"
            f"{'━' * 22}\n\n"
            f"📅 Tanggal: {date_str}"
        )
    
    @staticmethod
    def error_message(error_text: str, context: str = "") -> str:
        """Format pesan error"""
        error_text = _safe_string(error_text, 'Unknown error')
        context = _safe_string(context, '') if context else ''
        
        return (
            f"⚠️ *Error*\n"
            f"{'━' * 32}\n\n"
            f"{error_text}\n\n"
            f"{f'Context: {context}' if context else ''}"
            f"⏰ {datetime.now(pytz.timezone('Asia/Jakarta')).strftime('%H:%M:%S WIB')}"
        )
