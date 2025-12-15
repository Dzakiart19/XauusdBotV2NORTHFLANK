"""
Report Generator Module untuk Bot Trading XAUUSD.

Modul ini menyediakan pembuatan laporan trading:
- Laporan harian, mingguan, bulanan
- Export ke PDF dan Excel
- Visualisasi performa trading
- Statistik per jam, hari, dan session
"""

import io
import csv
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import pytz

from bot.logger import setup_logger
from bot.analytics import TradingAnalytics

logger = setup_logger('ReportGenerator')


class ReportGeneratorError(Exception):
    """Exception untuk error pada report generator"""
    pass


class ReportPeriod:
    """Konstanta untuk periode laporan"""
    DAILY = 'daily'
    WEEKLY = 'weekly'
    MONTHLY = 'monthly'
    CUSTOM = 'custom'


class ReportFormat:
    """Konstanta untuk format laporan"""
    TEXT = 'text'
    JSON = 'json'
    CSV = 'csv'
    HTML = 'html'


class ReportGenerator:
    """Generator laporan trading komprehensif"""
    
    def __init__(self, db_manager, config=None):
        self.db = db_manager
        self.config = config
        self.analytics = TradingAnalytics(db_manager, config)
        self.jakarta_tz = pytz.timezone('Asia/Jakarta')
        logger.info("ReportGenerator initialized")
    
    def generate_daily_report(self, user_id: Optional[int] = None, 
                               date: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate laporan harian
        
        Args:
            user_id: Filter by user (None for all)
            date: Tanggal laporan (default: hari ini)
        
        Returns:
            Dict berisi laporan lengkap
        """
        try:
            if date is None:
                date = datetime.now(self.jakarta_tz)
            
            start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = start_date + timedelta(days=1)
            
            performance = self.analytics.get_trading_performance(user_id, days=1)
            hourly = self.analytics.get_hourly_stats(user_id, days=1)
            source_perf = self.analytics.get_signal_source_performance(user_id, days=1)
            
            report = {
                'report_type': ReportPeriod.DAILY,
                'report_date': date.strftime('%Y-%m-%d'),
                'generated_at': datetime.now(self.jakarta_tz).isoformat(),
                'user_id': user_id,
                'summary': {
                    'total_trades': performance.get('total_trades', 0),
                    'wins': performance.get('wins', 0),
                    'losses': performance.get('losses', 0),
                    'winrate': performance.get('winrate', 0.0),
                    'total_pl': performance.get('total_pl', 0.0),
                    'avg_pl': performance.get('avg_pl', 0.0),
                    'largest_win': performance.get('largest_win', 0.0),
                    'largest_loss': performance.get('largest_loss', 0.0),
                    'profit_factor': performance.get('profit_factor', 0.0)
                },
                'hourly_breakdown': hourly.get('hourly_breakdown', {}),
                'best_hour': hourly.get('best_hour', {}),
                'worst_hour': hourly.get('worst_hour', {}),
                'signal_source': source_perf
            }
            
            logger.info(f"Daily report generated for {date.strftime('%Y-%m-%d')}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating daily report: {e}")
            return {'error': str(e)}
    
    def generate_weekly_report(self, user_id: Optional[int] = None,
                                week_start: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate laporan mingguan
        
        Args:
            user_id: Filter by user
            week_start: Awal minggu (default: minggu ini)
        
        Returns:
            Dict berisi laporan lengkap
        """
        try:
            if week_start is None:
                today = datetime.now(self.jakarta_tz)
                week_start = today - timedelta(days=today.weekday())
            
            week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
            week_end = week_start + timedelta(days=7)
            
            performance = self.analytics.get_trading_performance(user_id, days=7)
            hourly = self.analytics.get_hourly_stats(user_id, days=7)
            source_perf = self.analytics.get_signal_source_performance(user_id, days=7)
            position_stats = self.analytics.get_position_tracking_stats(user_id, days=7)
            risk_metrics = self.analytics.get_risk_metrics(user_id, days=7)
            
            daily_breakdown = {}
            for i in range(7):
                day = week_start + timedelta(days=i)
                day_report = self.generate_daily_report(user_id, day)
                daily_breakdown[day.strftime('%A')] = day_report.get('summary', {})
            
            report = {
                'report_type': ReportPeriod.WEEKLY,
                'week_start': week_start.strftime('%Y-%m-%d'),
                'week_end': week_end.strftime('%Y-%m-%d'),
                'generated_at': datetime.now(self.jakarta_tz).isoformat(),
                'user_id': user_id,
                'summary': {
                    'total_trades': performance.get('total_trades', 0),
                    'wins': performance.get('wins', 0),
                    'losses': performance.get('losses', 0),
                    'winrate': performance.get('winrate', 0.0),
                    'total_pl': performance.get('total_pl', 0.0),
                    'avg_pl': performance.get('avg_pl', 0.0),
                    'largest_win': performance.get('largest_win', 0.0),
                    'largest_loss': performance.get('largest_loss', 0.0),
                    'profit_factor': performance.get('profit_factor', 0.0)
                },
                'daily_breakdown': daily_breakdown,
                'hourly_breakdown': hourly.get('hourly_breakdown', {}),
                'best_hour': hourly.get('best_hour', {}),
                'worst_hour': hourly.get('worst_hour', {}),
                'signal_source': source_perf,
                'position_stats': position_stats,
                'risk_metrics': risk_metrics
            }
            
            logger.info(f"Weekly report generated for week starting {week_start.strftime('%Y-%m-%d')}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating weekly report: {e}")
            return {'error': str(e)}
    
    def generate_monthly_report(self, user_id: Optional[int] = None,
                                 year: Optional[int] = None,
                                 month: Optional[int] = None) -> Dict[str, Any]:
        """Generate laporan bulanan
        
        Args:
            user_id: Filter by user
            year: Tahun (default: tahun ini)
            month: Bulan (default: bulan ini)
        
        Returns:
            Dict berisi laporan lengkap
        """
        try:
            now = datetime.now(self.jakarta_tz)
            if year is None:
                year = now.year
            if month is None:
                month = now.month
            
            month_start = datetime(year, month, 1, tzinfo=self.jakarta_tz)
            if month == 12:
                month_end = datetime(year + 1, 1, 1, tzinfo=self.jakarta_tz)
            else:
                month_end = datetime(year, month + 1, 1, tzinfo=self.jakarta_tz)
            
            days_in_month = (month_end - month_start).days
            
            performance = self.analytics.get_trading_performance(user_id, days=days_in_month)
            hourly = self.analytics.get_hourly_stats(user_id, days=days_in_month)
            source_perf = self.analytics.get_signal_source_performance(user_id, days=days_in_month)
            position_stats = self.analytics.get_position_tracking_stats(user_id, days=days_in_month)
            risk_metrics = self.analytics.get_risk_metrics(user_id, days=days_in_month)
            
            weekly_breakdown = {}
            current = month_start
            week_num = 1
            while current < month_end:
                week_end = min(current + timedelta(days=7), month_end)
                weekly_breakdown[f'Week {week_num}'] = {
                    'start': current.strftime('%Y-%m-%d'),
                    'end': week_end.strftime('%Y-%m-%d')
                }
                current = week_end
                week_num += 1
            
            report = {
                'report_type': ReportPeriod.MONTHLY,
                'year': year,
                'month': month,
                'month_name': month_start.strftime('%B'),
                'generated_at': datetime.now(self.jakarta_tz).isoformat(),
                'user_id': user_id,
                'summary': {
                    'total_trades': performance.get('total_trades', 0),
                    'wins': performance.get('wins', 0),
                    'losses': performance.get('losses', 0),
                    'winrate': performance.get('winrate', 0.0),
                    'total_pl': performance.get('total_pl', 0.0),
                    'avg_pl': performance.get('avg_pl', 0.0),
                    'largest_win': performance.get('largest_win', 0.0),
                    'largest_loss': performance.get('largest_loss', 0.0),
                    'profit_factor': performance.get('profit_factor', 0.0),
                    'avg_trades_per_day': round(performance.get('total_trades', 0) / days_in_month, 2)
                },
                'weekly_breakdown': weekly_breakdown,
                'hourly_breakdown': hourly.get('hourly_breakdown', {}),
                'best_hour': hourly.get('best_hour', {}),
                'worst_hour': hourly.get('worst_hour', {}),
                'signal_source': source_perf,
                'position_stats': position_stats,
                'risk_metrics': risk_metrics
            }
            
            logger.info(f"Monthly report generated for {month_start.strftime('%B %Y')}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating monthly report: {e}")
            return {'error': str(e)}
    
    def format_report_text(self, report: Dict[str, Any]) -> str:
        """Format laporan ke format teks untuk Telegram
        
        Args:
            report: Data laporan
        
        Returns:
            String formatted untuk Telegram
        """
        try:
            report_type = report.get('report_type', 'unknown')
            summary = report.get('summary', {})
            
            if report_type == ReportPeriod.DAILY:
                title = f"📊 Laporan Harian - {report.get('report_date', '')}"
            elif report_type == ReportPeriod.WEEKLY:
                title = f"📊 Laporan Mingguan\n{report.get('week_start', '')} - {report.get('week_end', '')}"
            elif report_type == ReportPeriod.MONTHLY:
                title = f"📊 Laporan Bulanan - {report.get('month_name', '')} {report.get('year', '')}"
            else:
                title = "📊 Laporan Trading"
            
            total_pl = summary.get('total_pl', 0)
            pl_emoji = "🟢" if total_pl >= 0 else "🔴"
            
            text = f"*{title}*\n\n"
            text += "━━━━━━━━━━━━━━━━━━━━\n"
            text += f"📈 Total Trade: {summary.get('total_trades', 0)}\n"
            text += f"✅ Win: {summary.get('wins', 0)}\n"
            text += f"❌ Loss: {summary.get('losses', 0)}\n"
            text += f"📊 Win Rate: {summary.get('winrate', 0):.1f}%\n"
            text += "━━━━━━━━━━━━━━━━━━━━\n"
            text += f"{pl_emoji} Total P/L: ${total_pl:.2f}\n"
            text += f"📉 Avg P/L: ${summary.get('avg_pl', 0):.2f}\n"
            text += f"🏆 Largest Win: ${summary.get('largest_win', 0):.2f}\n"
            text += f"💔 Largest Loss: ${summary.get('largest_loss', 0):.2f}\n"
            text += f"⚖️ Profit Factor: {summary.get('profit_factor', 0):.2f}\n"
            text += "━━━━━━━━━━━━━━━━━━━━\n"
            
            best_hour = report.get('best_hour', {})
            worst_hour = report.get('worst_hour', {})
            
            if best_hour.get('hour') is not None:
                text += f"\n🕐 *Jam Terbaik*: {best_hour['hour']:02d}:00 WIB\n"
                text += f"   P/L: ${best_hour.get('stats', {}).get('total_pl', 0):.2f}\n"
            
            if worst_hour.get('hour') is not None:
                text += f"\n🕐 *Jam Terburuk*: {worst_hour['hour']:02d}:00 WIB\n"
                text += f"   P/L: ${worst_hour.get('stats', {}).get('total_pl', 0):.2f}\n"
            
            source = report.get('signal_source', {})
            if source:
                text += "\n*Performa per Sumber Sinyal:*\n"
                auto = source.get('auto', {})
                manual = source.get('manual', {})
                
                if auto.get('total_trades', 0) > 0:
                    text += f"🤖 Auto: {auto.get('winrate', 0):.1f}% WR, ${auto.get('total_pl', 0):.2f}\n"
                if manual.get('total_trades', 0) > 0:
                    text += f"👆 Manual: {manual.get('winrate', 0):.1f}% WR, ${manual.get('total_pl', 0):.2f}\n"
            
            text += f"\n⏰ Generated: {report.get('generated_at', '')[:19]}"
            
            return text
            
        except Exception as e:
            logger.error(f"Error formatting report: {e}")
            return f"Error generating report: {str(e)}"
    
    def export_to_csv(self, report: Dict[str, Any]) -> str:
        """Export laporan ke format CSV
        
        Args:
            report: Data laporan
        
        Returns:
            String CSV
        """
        try:
            output = io.StringIO()
            writer = csv.writer(output)
            
            writer.writerow(['XAUUSD Trading Report'])
            writer.writerow([f"Report Type: {report.get('report_type', 'N/A')}"])
            writer.writerow([f"Generated: {report.get('generated_at', 'N/A')}"])
            writer.writerow([])
            
            writer.writerow(['Summary Statistics'])
            summary = report.get('summary', {})
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Total Trades', summary.get('total_trades', 0)])
            writer.writerow(['Wins', summary.get('wins', 0)])
            writer.writerow(['Losses', summary.get('losses', 0)])
            writer.writerow(['Win Rate (%)', summary.get('winrate', 0)])
            writer.writerow(['Total P/L ($)', summary.get('total_pl', 0)])
            writer.writerow(['Avg P/L ($)', summary.get('avg_pl', 0)])
            writer.writerow(['Largest Win ($)', summary.get('largest_win', 0)])
            writer.writerow(['Largest Loss ($)', summary.get('largest_loss', 0)])
            writer.writerow(['Profit Factor', summary.get('profit_factor', 0)])
            writer.writerow([])
            
            hourly = report.get('hourly_breakdown', {})
            if hourly:
                writer.writerow(['Hourly Performance'])
                writer.writerow(['Hour (WIB)', 'Trades', 'Wins', 'Losses', 'Win Rate (%)', 'Total P/L ($)'])
                for hour in range(24):
                    if str(hour) in hourly or hour in hourly:
                        stats = hourly.get(str(hour)) or hourly.get(hour, {})
                        writer.writerow([
                            f"{hour:02d}:00",
                            stats.get('trades', 0),
                            stats.get('wins', 0),
                            stats.get('losses', 0),
                            stats.get('winrate', 0),
                            stats.get('total_pl', 0)
                        ])
            
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return f"Error: {str(e)}"
    
    def export_to_json(self, report: Dict[str, Any]) -> str:
        """Export laporan ke format JSON
        
        Args:
            report: Data laporan
        
        Returns:
            String JSON
        """
        try:
            return json.dumps(report, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            return json.dumps({'error': str(e)})
    
    def get_performance_summary_html(self, report: Dict[str, Any]) -> str:
        """Generate HTML summary untuk web dashboard
        
        Args:
            report: Data laporan
        
        Returns:
            String HTML
        """
        try:
            summary = report.get('summary', {})
            total_pl = summary.get('total_pl', 0)
            pl_class = 'profit' if total_pl >= 0 else 'loss'
            
            html = f"""
            <div class="report-card">
                <h3>{report.get('report_type', 'Report').title()} Performance</h3>
                <div class="report-grid">
                    <div class="stat-box">
                        <span class="stat-label">Total Trades</span>
                        <span class="stat-value">{summary.get('total_trades', 0)}</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-label">Win Rate</span>
                        <span class="stat-value">{summary.get('winrate', 0):.1f}%</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-label">Total P/L</span>
                        <span class="stat-value {pl_class}">${total_pl:.2f}</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-label">Profit Factor</span>
                        <span class="stat-value">{summary.get('profit_factor', 0):.2f}</span>
                    </div>
                </div>
            </div>
            """
            
            return html
            
        except Exception as e:
            logger.error(f"Error generating HTML: {e}")
            return f"<div class='error'>Error: {str(e)}</div>"


class ScheduledReportManager:
    """Manager untuk laporan terjadwal"""
    
    def __init__(self, report_generator: ReportGenerator, alert_system=None):
        self.report_generator = report_generator
        self.alert_system = alert_system
        self.scheduled_reports: Dict[int, Dict[str, bool]] = {}
        logger.info("ScheduledReportManager initialized")
    
    def subscribe_daily_report(self, user_id: int) -> bool:
        """Subscribe user ke laporan harian"""
        if user_id not in self.scheduled_reports:
            self.scheduled_reports[user_id] = {}
        self.scheduled_reports[user_id]['daily'] = True
        logger.info(f"User {user_id} subscribed to daily report")
        return True
    
    def subscribe_weekly_report(self, user_id: int) -> bool:
        """Subscribe user ke laporan mingguan"""
        if user_id not in self.scheduled_reports:
            self.scheduled_reports[user_id] = {}
        self.scheduled_reports[user_id]['weekly'] = True
        logger.info(f"User {user_id} subscribed to weekly report")
        return True
    
    def unsubscribe_report(self, user_id: int, report_type: str) -> bool:
        """Unsubscribe user dari laporan"""
        if user_id in self.scheduled_reports:
            self.scheduled_reports[user_id][report_type] = False
            logger.info(f"User {user_id} unsubscribed from {report_type} report")
            return True
        return False
    
    def get_subscribed_users(self, report_type: str) -> List[int]:
        """Get daftar user yang subscribe ke laporan tertentu"""
        users = []
        for user_id, subs in self.scheduled_reports.items():
            if subs.get(report_type, False):
                users.append(user_id)
        return users
    
    async def send_scheduled_reports(self, report_type: str):
        """Kirim laporan terjadwal ke semua subscriber"""
        users = self.get_subscribed_users(report_type)
        
        for user_id in users:
            try:
                if report_type == 'daily':
                    report = self.report_generator.generate_daily_report(user_id)
                elif report_type == 'weekly':
                    report = self.report_generator.generate_weekly_report(user_id)
                else:
                    continue
                
                formatted = self.report_generator.format_report_text(report)
                
                if self.alert_system:
                    from bot.alert_system import Alert
                    alert = Alert(
                        alert_type=f"SCHEDULED_{report_type.upper()}_REPORT",
                        message=formatted,
                        priority='NORMAL',
                        data={'user_id': user_id, 'report_type': report_type}
                    )
                    await self.alert_system.send_alert(alert)
                
                logger.info(f"Sent {report_type} report to user {user_id}")
                
            except Exception as e:
                logger.error(f"Error sending {report_type} report to user {user_id}: {e}")
