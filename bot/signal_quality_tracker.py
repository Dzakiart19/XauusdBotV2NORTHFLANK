"""
Signal Quality Tracker untuk Bot Trading XAUUSD.

Modul ini menyediakan tracking dan analisis kualitas sinyal trading:
- Hit rate per signal type (M1_SCALP, M5_SWING, SR_REVERSION, BREAKOUT)
- Akurasi per kondisi pasar (trending vs ranging)
- Akurasi per confluence level dan jam trading
- Alert untuk penurunan kualitas sinyal
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, BigInteger, Index, text
from sqlalchemy.exc import SQLAlchemyError, OperationalError, IntegrityError
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any, Tuple
from collections import defaultdict
from enum import Enum
import threading
import pytz
import time

from bot.database import Base, DatabaseError, retry_on_db_error
from bot.logger import setup_logger

logger = setup_logger('SignalQualityTracker')


class SignalQualityTrackerError(Exception):
    """Exception untuk error pada signal quality tracker"""
    pass


class SignalResult(str, Enum):
    """Enum untuk hasil signal"""
    WIN = 'WIN'
    LOSS = 'LOSS'
    BREAKEVEN = 'BREAKEVEN'
    PENDING = 'PENDING'


class RuleType(str, Enum):
    """Enum untuk tipe signal rule"""
    M1_SCALP = 'M1_SCALP'
    M5_SWING = 'M5_SWING'
    SR_REVERSION = 'SR_REVERSION'
    BREAKOUT = 'BREAKOUT'


class MarketRegimeType(str, Enum):
    """Enum untuk tipe market regime"""
    TRENDING = 'trending'
    RANGING = 'ranging'
    HIGH_VOLATILITY = 'high_volatility'
    LOW_VOLATILITY = 'low_volatility'
    BREAKOUT = 'breakout'
    UNKNOWN = 'unknown'


class SignalQuality(Base):
    """SQLAlchemy model untuk tracking kualitas sinyal trading.
    
    Table ini menyimpan detail setiap signal yang dihasilkan untuk analisis performa.
    """
    __tablename__ = 'signal_quality'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, nullable=False, index=True)
    signal_time = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    signal_type = Column(String(10), nullable=False, index=True)
    rule_name = Column(String(30), nullable=False, index=True)
    confluence_level = Column(Integer, default=2, nullable=False, index=True)
    
    market_regime = Column(String(30), default='unknown', nullable=False, index=True)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    
    sl_pips = Column(Float, nullable=False)
    tp_pips = Column(Float, nullable=False)
    actual_pips = Column(Float, nullable=True)
    
    result = Column(String(15), default='PENDING', nullable=False, index=True)
    duration_minutes = Column(Integer, nullable=True)
    
    trading_hour = Column(Integer, nullable=True, index=True)
    close_time = Column(DateTime, nullable=True)
    
    confidence = Column(Float, default=0.0)
    reason = Column(String(255), nullable=True)
    
    __table_args__ = (
        Index('idx_signal_quality_rule_result', 'rule_name', 'result'),
        Index('idx_signal_quality_regime_result', 'market_regime', 'result'),
        Index('idx_signal_quality_hour_result', 'trading_hour', 'result'),
        Index('idx_signal_quality_confluence_result', 'confluence_level', 'result'),
        Index('idx_signal_quality_time_desc', signal_time.desc()),
        {'extend_existing': True}
    )


class QualityAlert:
    """Dataclass untuk alert kualitas sinyal"""
    
    def __init__(self, alert_type: str, message: str, severity: str = 'WARNING', data: Optional[Dict] = None):
        self.alert_type = alert_type
        self.message = message
        self.severity = severity
        self.data = data or {}
        self.timestamp = datetime.now(pytz.UTC)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_type': self.alert_type,
            'message': self.message,
            'severity': self.severity,
            'data': self.data,
            'timestamp': self.timestamp.isoformat()
        }


class SignalQualityTracker:
    """
    Tracker untuk menganalisis dan memantau kualitas sinyal trading.
    
    Fitur utama:
    1. Hit rate per signal type
    2. Average win/loss pips per type
    3. Akurasi per kondisi pasar dan confluence level
    4. Akurasi per jam trading
    5. Alert untuk penurunan kualitas
    
    Thread-safe dengan locking untuk operasi konkuren.
    """
    
    ACCURACY_WARNING_THRESHOLD = 0.45
    HOURLY_WARNING_THRESHOLD = 0.50
    MINIMUM_SIGNALS_FOR_ANALYSIS = 10
    DEFAULT_ANALYSIS_SIGNALS = 50
    
    def __init__(self, db_manager, config=None, alert_callback=None):
        """
        Inisialisasi SignalQualityTracker.
        
        Args:
            db_manager: DatabaseManager instance untuk operasi database
            config: Konfigurasi bot (opsional)
            alert_callback: Callback async untuk mengirim alert (opsional)
        """
        self.db = db_manager
        self.config = config
        self.alert_callback = alert_callback
        self._lock = threading.RLock()
        self._pending_alerts: List[QualityAlert] = []
        self._last_alert_check = datetime.now(pytz.UTC)
        self._alert_cooldown = timedelta(minutes=30)
        self._last_alerts_sent: Dict[str, datetime] = {}
        
        self._blocked_signals: Dict[int, List[Dict]] = {}
        self._blocked_signals_lock = threading.RLock()
        self._last_blocking_rate_check: Dict[int, datetime] = {}
        self.BLOCKING_RATE_ALERT_THRESHOLD = 0.80
        self.BLOCKING_RATE_WINDOW_MINUTES = 5
        
        self._ensure_table_exists()
        
        logger.info("SignalQualityTracker initialized")
    
    def _ensure_table_exists(self):
        """Pastikan tabel signal_quality ada di database."""
        try:
            if self.db.engine is not None:
                SignalQuality.__table__.create(self.db.engine, checkfirst=True)
                logger.info("✅ Signal quality table ensured")
        except SQLAlchemyError as e:
            logger.error(f"Error creating signal_quality table: {e}")
            raise DatabaseError(f"Failed to create signal_quality table: {e}")
    
    @retry_on_db_error(max_retries=3)
    def record_signal(self, signal_data: Dict[str, Any]) -> Optional[int]:
        """
        Catat signal baru ke database.
        
        Args:
            signal_data: Dict dengan data signal:
                - user_id: ID pengguna (opsional, default 0)
                - signal_type: BUY/SELL
                - rule_name: M1_SCALP/M5_SWING/SR_REVERSION/BREAKOUT
                - confluence_level: Jumlah confluence (2, 3, dst)
                - market_regime: Kondisi pasar
                - entry_price: Harga entry
                - sl_pips: Stop loss dalam pips
                - tp_pips: Take profit dalam pips
                - confidence: Tingkat keyakinan signal (0.0-1.0)
                - reason: Alasan signal (opsional)
        
        Returns:
            ID signal yang dicatat, atau None jika gagal
        """
        with self._lock:
            session = None
            try:
                session = self.db.get_session()
                if session is None:
                    logger.error("Failed to get database session for record_signal")
                    return None
                
                jakarta_tz = pytz.timezone('Asia/Jakarta')
                current_time = datetime.now(pytz.UTC)
                jakarta_time = current_time.astimezone(jakarta_tz)
                trading_hour = jakarta_time.hour
                
                signal = SignalQuality(
                    user_id=signal_data.get('user_id', 0),
                    signal_time=current_time,
                    signal_type=signal_data.get('signal_type', 'BUY'),
                    rule_name=signal_data.get('rule_name', 'UNKNOWN'),
                    confluence_level=signal_data.get('confluence_level', 2),
                    market_regime=signal_data.get('market_regime', 'unknown'),
                    entry_price=signal_data.get('entry_price', 0.0),
                    sl_pips=signal_data.get('sl_pips', 10.0),
                    tp_pips=signal_data.get('tp_pips', 20.0),
                    trading_hour=trading_hour,
                    confidence=signal_data.get('confidence', 0.0),
                    reason=signal_data.get('reason', '')[:255] if signal_data.get('reason') else None,
                    result=SignalResult.PENDING.value
                )
                
                session.add(signal)
                session.commit()
                
                session.refresh(signal)
                signal_id: Optional[int] = getattr(signal, 'id', None)
                logger.info(f"📊 Signal recorded: ID={signal_id}, Rule={signal.rule_name}, Confluence={signal.confluence_level}")
                
                self._check_quality_alerts()
                
                return signal_id
                
            except IntegrityError as e:
                if session:
                    session.rollback()
                logger.error(f"Integrity error recording signal: {e}")
                return None
            except SQLAlchemyError as e:
                if session:
                    session.rollback()
                logger.error(f"Database error recording signal: {e}")
                return None
            except Exception as e:
                if session:
                    session.rollback()
                logger.error(f"Unexpected error recording signal: {e}")
                return None
            finally:
                if session:
                    session.close()
    
    @retry_on_db_error(max_retries=3)
    def update_result(self, signal_id: int, result_data: Dict[str, Any]) -> bool:
        """
        Update hasil signal setelah trade selesai.
        
        Args:
            signal_id: ID signal yang akan diupdate
            result_data: Dict dengan data hasil:
                - exit_price: Harga exit
                - actual_pips: Hasil dalam pips (+/-)
                - result: WIN/LOSS/BREAKEVEN
                - duration_minutes: Durasi trade dalam menit
        
        Returns:
            True jika berhasil, False jika gagal
        """
        with self._lock:
            session = None
            try:
                session = self.db.get_session()
                if session is None:
                    logger.error("Failed to get database session for update_result")
                    return False
                
                signal = session.query(SignalQuality).filter(
                    SignalQuality.id == signal_id
                ).first()
                
                if signal is None:
                    logger.warning(f"Signal ID {signal_id} not found for update")
                    return False
                
                signal.exit_price = result_data.get('exit_price')
                signal.actual_pips = result_data.get('actual_pips', 0.0)
                signal.result = result_data.get('result', SignalResult.PENDING.value)
                signal.duration_minutes = result_data.get('duration_minutes')
                signal.close_time = datetime.now(pytz.UTC)
                
                session.commit()
                
                logger.info(
                    f"📊 Signal result updated: ID={signal_id}, Result={signal.result}, "
                    f"Pips={signal.actual_pips:.1f}"
                )
                
                self._check_quality_alerts()
                
                return True
                
            except SQLAlchemyError as e:
                if session:
                    session.rollback()
                logger.error(f"Database error updating signal result: {e}")
                return False
            except Exception as e:
                if session:
                    session.rollback()
                logger.error(f"Unexpected error updating signal result: {e}")
                return False
            finally:
                if session:
                    session.close()
    
    @retry_on_db_error(max_retries=3)
    def get_accuracy_by_type(self, rule_name: str, last_n_signals: int = 50) -> float:
        """
        Dapatkan akurasi untuk tipe signal tertentu.
        
        Args:
            rule_name: Nama rule (M1_SCALP, M5_SWING, SR_REVERSION, BREAKOUT)
            last_n_signals: Jumlah signal terakhir untuk analisis
        
        Returns:
            Akurasi sebagai float (0.0 - 1.0), atau -1.0 jika data tidak cukup
        """
        session = None
        try:
            session = self.db.get_session()
            if session is None:
                return -1.0
            
            signals = session.query(SignalQuality).filter(
                SignalQuality.rule_name == rule_name,
                SignalQuality.result.in_([SignalResult.WIN.value, SignalResult.LOSS.value, SignalResult.BREAKEVEN.value])
            ).order_by(SignalQuality.signal_time.desc()).limit(last_n_signals).all()
            
            if len(signals) < self.MINIMUM_SIGNALS_FOR_ANALYSIS:
                return -1.0
            
            wins = sum(1 for s in signals if s.actual_pips is not None and s.actual_pips >= 0)
            total = len(signals)
            
            accuracy = wins / total if total > 0 else 0.0
            
            logger.debug(f"Accuracy for {rule_name}: {accuracy:.2%} ({wins}/{total})")
            
            return accuracy
            
        except SQLAlchemyError as e:
            logger.error(f"Database error in get_accuracy_by_type: {e}")
            return -1.0
        finally:
            if session:
                session.close()
    
    @retry_on_db_error(max_retries=3)
    def get_accuracy_by_regime(self, regime: str, last_n_signals: int = 50) -> float:
        """
        Dapatkan akurasi untuk kondisi pasar tertentu.
        
        Args:
            regime: Market regime (trending, ranging, high_volatility, dll)
            last_n_signals: Jumlah signal terakhir untuk analisis
        
        Returns:
            Akurasi sebagai float (0.0 - 1.0), atau -1.0 jika data tidak cukup
        """
        session = None
        try:
            session = self.db.get_session()
            if session is None:
                return -1.0
            
            is_trending = regime.lower() in ['trending', 'strong_trend', 'moderate_trend', 'weak_trend']
            
            if is_trending:
                regime_filter = SignalQuality.market_regime.in_([
                    'trending', 'strong_trend', 'moderate_trend', 'weak_trend'
                ])
            else:
                regime_filter = SignalQuality.market_regime.in_([
                    'ranging', 'range_bound', regime.lower()
                ])
            
            signals = session.query(SignalQuality).filter(
                regime_filter,
                SignalQuality.result.in_([SignalResult.WIN.value, SignalResult.LOSS.value, SignalResult.BREAKEVEN.value])
            ).order_by(SignalQuality.signal_time.desc()).limit(last_n_signals).all()
            
            if len(signals) < self.MINIMUM_SIGNALS_FOR_ANALYSIS:
                return -1.0
            
            wins = sum(1 for s in signals if s.actual_pips is not None and s.actual_pips >= 0)
            total = len(signals)
            
            accuracy = wins / total if total > 0 else 0.0
            
            logger.debug(f"Accuracy for regime {regime}: {accuracy:.2%} ({wins}/{total})")
            
            return accuracy
            
        except SQLAlchemyError as e:
            logger.error(f"Database error in get_accuracy_by_regime: {e}")
            return -1.0
        finally:
            if session:
                session.close()
    
    @retry_on_db_error(max_retries=3)
    def get_accuracy_by_hour(self, hour: int, last_n_signals: int = 50) -> float:
        """
        Dapatkan akurasi untuk jam trading tertentu (WIB).
        
        Args:
            hour: Jam trading (0-23, dalam waktu Jakarta/WIB)
            last_n_signals: Jumlah signal terakhir untuk analisis
        
        Returns:
            Akurasi sebagai float (0.0 - 1.0), atau -1.0 jika data tidak cukup
        """
        session = None
        try:
            session = self.db.get_session()
            if session is None:
                return -1.0
            
            if not 0 <= hour <= 23:
                logger.warning(f"Invalid hour: {hour}")
                return -1.0
            
            signals = session.query(SignalQuality).filter(
                SignalQuality.trading_hour == hour,
                SignalQuality.result.in_([SignalResult.WIN.value, SignalResult.LOSS.value, SignalResult.BREAKEVEN.value])
            ).order_by(SignalQuality.signal_time.desc()).limit(last_n_signals).all()
            
            if len(signals) < self.MINIMUM_SIGNALS_FOR_ANALYSIS:
                return -1.0
            
            wins = sum(1 for s in signals if s.actual_pips is not None and s.actual_pips >= 0)
            total = len(signals)
            
            accuracy = wins / total if total > 0 else 0.0
            
            logger.debug(f"Accuracy for hour {hour:02d}:00 WIB: {accuracy:.2%} ({wins}/{total})")
            
            return accuracy
            
        except SQLAlchemyError as e:
            logger.error(f"Database error in get_accuracy_by_hour: {e}")
            return -1.0
        finally:
            if session:
                session.close()
    
    @retry_on_db_error(max_retries=3)
    def get_accuracy_by_confluence(self, min_confluence: int = 2, last_n_signals: int = 50) -> float:
        """
        Dapatkan akurasi untuk confluence level tertentu.
        
        Args:
            min_confluence: Minimum confluence level (2 untuk 2-confluence, 3 untuk 3+ confluence)
            last_n_signals: Jumlah signal terakhir untuk analisis
        
        Returns:
            Akurasi sebagai float (0.0 - 1.0), atau -1.0 jika data tidak cukup
        """
        session = None
        try:
            session = self.db.get_session()
            if session is None:
                return -1.0
            
            signals = session.query(SignalQuality).filter(
                SignalQuality.confluence_level >= min_confluence,
                SignalQuality.result.in_([SignalResult.WIN.value, SignalResult.LOSS.value, SignalResult.BREAKEVEN.value])
            ).order_by(SignalQuality.signal_time.desc()).limit(last_n_signals).all()
            
            if len(signals) < self.MINIMUM_SIGNALS_FOR_ANALYSIS:
                return -1.0
            
            wins = sum(1 for s in signals if s.actual_pips is not None and s.actual_pips >= 0)
            total = len(signals)
            
            accuracy = wins / total if total > 0 else 0.0
            
            logger.debug(f"Accuracy for {min_confluence}+ confluence: {accuracy:.2%} ({wins}/{total})")
            
            return accuracy
            
        except SQLAlchemyError as e:
            logger.error(f"Database error in get_accuracy_by_confluence: {e}")
            return -1.0
        finally:
            if session:
                session.close()
    
    @retry_on_db_error(max_retries=3)
    def get_overall_stats(self, days: int = 30) -> Dict[str, Any]:
        """
        Dapatkan statistik keseluruhan dari semua signal.
        
        Args:
            days: Jumlah hari untuk analisis
        
        Returns:
            Dict dengan semua metrics:
            - total_signals: Total signal yang dianalisis
            - overall_accuracy: Akurasi keseluruhan
            - accuracy_by_type: Dict akurasi per rule type
            - accuracy_by_regime: Dict akurasi per market regime
            - accuracy_by_hour: Dict akurasi per jam
            - accuracy_by_confluence: Dict akurasi per confluence level
            - avg_win_pips_by_type: Dict rata-rata win pips per type
            - avg_loss_pips_by_type: Dict rata-rata loss pips per type
            - best_performing_hour: Jam dengan akurasi terbaik
            - worst_performing_hour: Jam dengan akurasi terburuk
            - profit_factor: Total win pips / Total loss pips
        """
        session = None
        try:
            session = self.db.get_session()
            if session is None:
                return {'error': 'Failed to get database session'}
            
            cutoff_date = datetime.now(pytz.UTC) - timedelta(days=days)
            
            closed_signals = session.query(SignalQuality).filter(
                SignalQuality.signal_time >= cutoff_date,
                SignalQuality.result.in_([SignalResult.WIN.value, SignalResult.LOSS.value, SignalResult.BREAKEVEN.value])
            ).all()
            
            if not closed_signals:
                return {
                    'total_signals': 0,
                    'overall_accuracy': 0.0,
                    'period_days': days,
                    'message': 'No completed signals in the period'
                }
            
            total_signals = len(closed_signals)
            wins = sum(1 for s in closed_signals if s.actual_pips is not None and s.actual_pips >= 0)
            losses = sum(1 for s in closed_signals if s.actual_pips is not None and s.actual_pips < 0)
            breakevens = sum(1 for s in closed_signals if s.result == SignalResult.BREAKEVEN.value)
            overall_accuracy = wins / total_signals if total_signals > 0 else 0.0
            
            accuracy_by_type = {}
            avg_win_pips_by_type = {}
            avg_loss_pips_by_type = {}
            
            for rule in RuleType:
                rule_signals = [s for s in closed_signals if s.rule_name == rule.value]
                if rule_signals:
                    rule_wins = sum(1 for s in rule_signals if s.actual_pips is not None and s.actual_pips >= 0)
                    accuracy_by_type[rule.value] = {
                        'accuracy': rule_wins / len(rule_signals) if rule_signals else 0.0,
                        'total': len(rule_signals),
                        'wins': rule_wins,
                        'losses': len([s for s in rule_signals if s.actual_pips is not None and s.actual_pips < 0])
                    }
                    
                    win_pips = [s.actual_pips for s in rule_signals 
                               if s.actual_pips is not None and s.actual_pips >= 0]
                    loss_pips = [abs(s.actual_pips) for s in rule_signals 
                                if s.actual_pips is not None and s.actual_pips < 0]
                    
                    avg_win_pips_by_type[rule.value] = sum(win_pips) / len(win_pips) if win_pips else 0.0
                    avg_loss_pips_by_type[rule.value] = sum(loss_pips) / len(loss_pips) if loss_pips else 0.0
            
            accuracy_by_regime = {}
            regimes = ['trending', 'ranging', 'high_volatility', 'breakout', 'unknown']
            for regime in regimes:
                if regime == 'trending':
                    regime_signals = [s for s in closed_signals 
                                     if s.market_regime in ['trending', 'strong_trend', 'moderate_trend', 'weak_trend']]
                elif regime == 'ranging':
                    regime_signals = [s for s in closed_signals 
                                     if s.market_regime in ['ranging', 'range_bound']]
                else:
                    regime_signals = [s for s in closed_signals if s.market_regime == regime]
                
                if regime_signals:
                    regime_wins = sum(1 for s in regime_signals if s.actual_pips is not None and s.actual_pips >= 0)
                    accuracy_by_regime[regime] = {
                        'accuracy': regime_wins / len(regime_signals) if regime_signals else 0.0,
                        'total': len(regime_signals),
                        'wins': regime_wins
                    }
            
            accuracy_by_hour = {}
            best_hour = (-1, 0.0)
            worst_hour = (-1, 1.0)
            
            for hour in range(24):
                hour_signals = [s for s in closed_signals if s.trading_hour == hour]
                if hour_signals and len(hour_signals) >= 5:
                    hour_wins = sum(1 for s in hour_signals if s.actual_pips is not None and s.actual_pips >= 0)
                    hour_accuracy = hour_wins / len(hour_signals) if hour_signals else 0.0
                    accuracy_by_hour[hour] = {
                        'accuracy': hour_accuracy,
                        'total': len(hour_signals),
                        'wins': hour_wins
                    }
                    
                    if hour_accuracy > best_hour[1]:
                        best_hour = (hour, hour_accuracy)
                    if hour_accuracy < worst_hour[1]:
                        worst_hour = (hour, hour_accuracy)
            
            accuracy_by_confluence = {}
            for conf_level in [2, 3, 4, 5]:
                if conf_level == 2:
                    conf_signals = [s for s in closed_signals if s.confluence_level == 2]
                else:
                    conf_signals = [s for s in closed_signals if s.confluence_level >= conf_level]
                
                if conf_signals:
                    conf_wins = sum(1 for s in conf_signals if s.actual_pips is not None and s.actual_pips >= 0)
                    label = f"{conf_level}_confluence" if conf_level == 2 else f"{conf_level}+_confluence"
                    accuracy_by_confluence[label] = {
                        'accuracy': conf_wins / len(conf_signals) if conf_signals else 0.0,
                        'total': len(conf_signals),
                        'wins': conf_wins
                    }
            
            total_win_pips = sum(s.actual_pips for s in closed_signals 
                                if s.actual_pips is not None and s.actual_pips >= 0)
            total_loss_pips = abs(sum(s.actual_pips for s in closed_signals 
                                     if s.actual_pips is not None and s.actual_pips < 0))
            profit_factor = total_win_pips / total_loss_pips if total_loss_pips > 0 else 0.0
            
            avg_duration = sum(s.duration_minutes for s in closed_signals 
                              if s.duration_minutes is not None) / len([s for s in closed_signals 
                              if s.duration_minutes is not None]) if any(s.duration_minutes for s in closed_signals) else 0
            
            return {
                'total_signals': total_signals,
                'wins': wins,
                'losses': losses,
                'breakevens': breakevens,
                'overall_accuracy': round(overall_accuracy, 4),
                'accuracy_by_type': accuracy_by_type,
                'accuracy_by_regime': accuracy_by_regime,
                'accuracy_by_hour': accuracy_by_hour,
                'accuracy_by_confluence': accuracy_by_confluence,
                'avg_win_pips_by_type': {k: round(v, 2) for k, v in avg_win_pips_by_type.items()},
                'avg_loss_pips_by_type': {k: round(v, 2) for k, v in avg_loss_pips_by_type.items()},
                'best_performing_hour': {'hour': best_hour[0], 'accuracy': round(best_hour[1], 4)} if best_hour[0] >= 0 else None,
                'worst_performing_hour': {'hour': worst_hour[0], 'accuracy': round(worst_hour[1], 4)} if worst_hour[0] >= 0 else None,
                'profit_factor': round(profit_factor, 2),
                'total_win_pips': round(total_win_pips, 2),
                'total_loss_pips': round(total_loss_pips, 2),
                'avg_duration_minutes': round(avg_duration, 1),
                'period_days': days
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Database error in get_overall_stats: {e}")
            return {'error': str(e)}
        finally:
            if session:
                session.close()
    
    def get_performance_report(self, days: int = 30) -> str:
        """
        Generate laporan performa untuk Telegram /performa command.
        
        Args:
            days: Jumlah hari untuk analisis
        
        Returns:
            String formatted untuk Telegram (Markdown)
        """
        stats = self.get_overall_stats(days)
        
        if 'error' in stats:
            return f"❌ Error generating report: {stats['error']}"
        
        if stats.get('total_signals', 0) == 0:
            return f"📊 *Signal Quality Report ({days} hari)*\n\nBelum ada signal yang tercatat dalam periode ini."
        
        report = f"📊 *Signal Quality Report ({days} hari)*\n\n"
        
        report += "📈 *Overall Statistics*\n"
        report += f"• Total Signals: {stats['total_signals']}\n"
        report += f"• Win Rate: {stats['overall_accuracy'] * 100:.1f}%\n"
        report += f"• Wins: {stats['wins']} | Losses: {stats['losses']} | BE: {stats['breakevens']}\n"
        report += f"• Profit Factor: {stats['profit_factor']}\n"
        report += f"• Total Pips: +{stats['total_win_pips']:.1f} / -{stats['total_loss_pips']:.1f}\n\n"
        
        report += "🎯 *Accuracy by Signal Type*\n"
        for rule_type, data in stats.get('accuracy_by_type', {}).items():
            emoji = "✅" if data['accuracy'] >= 0.5 else "⚠️" if data['accuracy'] >= 0.4 else "❌"
            report += f"{emoji} {rule_type}: {data['accuracy'] * 100:.1f}% ({data['wins']}/{data['total']})\n"
        report += "\n"
        
        report += "📊 *Accuracy by Market Condition*\n"
        for regime, data in stats.get('accuracy_by_regime', {}).items():
            if data['total'] >= 5:
                emoji = "✅" if data['accuracy'] >= 0.5 else "⚠️" if data['accuracy'] >= 0.4 else "❌"
                report += f"{emoji} {regime.title()}: {data['accuracy'] * 100:.1f}% ({data['wins']}/{data['total']})\n"
        report += "\n"
        
        report += "🕐 *Accuracy by Confluence*\n"
        for conf, data in stats.get('accuracy_by_confluence', {}).items():
            emoji = "✅" if data['accuracy'] >= 0.5 else "⚠️" if data['accuracy'] >= 0.4 else "❌"
            report += f"{emoji} {conf}: {data['accuracy'] * 100:.1f}% ({data['wins']}/{data['total']})\n"
        report += "\n"
        
        if stats.get('best_performing_hour'):
            report += f"⏰ *Best Hour*: {stats['best_performing_hour']['hour']:02d}:00 WIB ({stats['best_performing_hour']['accuracy'] * 100:.1f}%)\n"
        if stats.get('worst_performing_hour'):
            report += f"⏰ *Worst Hour*: {stats['worst_performing_hour']['hour']:02d}:00 WIB ({stats['worst_performing_hour']['accuracy'] * 100:.1f}%)\n"
        
        if stats.get('avg_duration_minutes', 0) > 0:
            report += f"\n⏱️ *Avg Trade Duration*: {stats['avg_duration_minutes']:.0f} menit\n"
        
        return report
    
    def _check_quality_alerts(self):
        """Check for quality degradation and generate alerts."""
        with self._lock:
            current_time = datetime.now(pytz.UTC)
            
            if current_time - self._last_alert_check < timedelta(minutes=5):
                return
            
            self._last_alert_check = current_time
            
            try:
                for rule in RuleType:
                    accuracy = self.get_accuracy_by_type(rule.value, last_n_signals=50)
                    if accuracy >= 0 and accuracy < self.ACCURACY_WARNING_THRESHOLD:
                        alert_key = f"rule_accuracy_{rule.value}"
                        if self._can_send_alert(alert_key):
                            alert = QualityAlert(
                                alert_type="SIGNAL_QUALITY_DROP",
                                message=f"⚠️ {rule.value} accuracy dropped to {accuracy * 100:.1f}% (threshold: {self.ACCURACY_WARNING_THRESHOLD * 100}%)",
                                severity="WARNING",
                                data={'rule': rule.value, 'accuracy': accuracy}
                            )
                            self._pending_alerts.append(alert)
                            self._last_alerts_sent[alert_key] = current_time
                            logger.warning(f"Quality alert: {rule.value} accuracy at {accuracy * 100:.1f}%")
                
                for hour in range(24):
                    accuracy = self.get_accuracy_by_hour(hour, last_n_signals=30)
                    if accuracy >= 0 and accuracy < self.HOURLY_WARNING_THRESHOLD:
                        alert_key = f"hour_accuracy_{hour}"
                        if self._can_send_alert(alert_key):
                            alert = QualityAlert(
                                alert_type="HOURLY_QUALITY_WARNING",
                                message=f"⚠️ Hour {hour:02d}:00 WIB has low accuracy: {accuracy * 100:.1f}% (threshold: {self.HOURLY_WARNING_THRESHOLD * 100}%)",
                                severity="INFO",
                                data={'hour': hour, 'accuracy': accuracy}
                            )
                            self._pending_alerts.append(alert)
                            self._last_alerts_sent[alert_key] = current_time
                            logger.info(f"Hourly alert: Hour {hour:02d}:00 accuracy at {accuracy * 100:.1f}%")
                
            except Exception as e:
                logger.error(f"Error checking quality alerts: {e}")
    
    def _can_send_alert(self, alert_key: str) -> bool:
        """Check if enough time has passed since last alert of this type."""
        if alert_key not in self._last_alerts_sent:
            return True
        
        time_since_last = datetime.now(pytz.UTC) - self._last_alerts_sent[alert_key]
        return time_since_last >= self._alert_cooldown
    
    def get_pending_alerts(self) -> List[QualityAlert]:
        """Get and clear pending alerts."""
        with self._lock:
            alerts = self._pending_alerts.copy()
            self._pending_alerts.clear()
            return alerts
    
    def get_alerts_for_telegram(self) -> Optional[str]:
        """Get formatted alerts for Telegram notification."""
        alerts = self.get_pending_alerts()
        
        if not alerts:
            return None
        
        message = "🚨 *Signal Quality Alerts*\n\n"
        
        for alert in alerts:
            if alert.severity == "WARNING":
                message += f"⚠️ {alert.message}\n"
            else:
                message += f"ℹ️ {alert.message}\n"
        
        return message
    
    @retry_on_db_error(max_retries=3)
    def get_signals_count(self, days: int = 1) -> Dict[str, int]:
        """
        Dapatkan jumlah signal dalam periode tertentu.
        
        Args:
            days: Jumlah hari untuk hitung
        
        Returns:
            Dict dengan jumlah signal per status
        """
        session = None
        try:
            session = self.db.get_session()
            if session is None:
                return {'total': 0, 'pending': 0, 'wins': 0, 'losses': 0}
            
            cutoff_date = datetime.now(pytz.UTC) - timedelta(days=days)
            
            signals = session.query(SignalQuality).filter(
                SignalQuality.signal_time >= cutoff_date
            ).all()
            
            return {
                'total': len(signals),
                'pending': sum(1 for s in signals if s.result == SignalResult.PENDING.value),
                'wins': sum(1 for s in signals if s.actual_pips is not None and s.actual_pips >= 0),
                'losses': sum(1 for s in signals if s.actual_pips is not None and s.actual_pips < 0),
                'breakevens': sum(1 for s in signals if s.result == SignalResult.BREAKEVEN.value)
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Database error in get_signals_count: {e}")
            return {'total': 0, 'pending': 0, 'wins': 0, 'losses': 0}
        finally:
            if session:
                session.close()
    
    def get_bad_hours(self, threshold: float = 0.45, min_signals: int = 10) -> List[int]:
        """
        Dapatkan jam-jam dengan performa buruk.
        
        Args:
            threshold: Threshold akurasi minimum
            min_signals: Minimum jumlah signal untuk dianggap valid
        
        Returns:
            List jam (0-23) dengan akurasi di bawah threshold
        """
        bad_hours = []
        
        for hour in range(24):
            accuracy = self.get_accuracy_by_hour(hour, last_n_signals=min_signals)
            if 0 <= accuracy < threshold:
                bad_hours.append(hour)
        
        return sorted(bad_hours)
    
    def should_reduce_signals(self, rule_name: Optional[str] = None, current_hour: Optional[int] = None) -> Tuple[bool, str]:
        """
        Cek apakah sebaiknya mengurangi signal berdasarkan performa terakhir.
        
        Args:
            rule_name: Nama rule untuk dicek (opsional)
            current_hour: Jam saat ini dalam WIB (opsional)
        
        Returns:
            Tuple (should_reduce: bool, reason: str)
        """
        reasons = []
        
        if rule_name:
            accuracy = self.get_accuracy_by_type(rule_name, last_n_signals=30)
            if 0 <= accuracy < self.ACCURACY_WARNING_THRESHOLD:
                reasons.append(f"{rule_name} accuracy low ({accuracy * 100:.1f}%)")
        
        if current_hour is not None:
            accuracy = self.get_accuracy_by_hour(current_hour, last_n_signals=20)
            if 0 <= accuracy < self.HOURLY_WARNING_THRESHOLD:
                reasons.append(f"Hour {current_hour:02d}:00 accuracy low ({accuracy * 100:.1f}%)")
        
        if reasons:
            return True, "; ".join(reasons)
        
        return False, ""
    
    def track_blocked_signal(self, user_id: int, signal_data: Dict[str, Any], 
                             blocking_reason: str) -> None:
        """
        Track signal yang di-block untuk analisis.
        
        Args:
            user_id: ID user
            signal_data: Data signal (type, confidence, grade, rule_name, etc)
            blocking_reason: Alasan blocking (session_active, cooldown, etc)
        """
        with self._blocked_signals_lock:
            if user_id not in self._blocked_signals:
                self._blocked_signals[user_id] = []
            
            blocked_entry = {
                'timestamp': datetime.now(pytz.UTC),
                'signal_type': signal_data.get('signal_type', 'UNKNOWN'),
                'rule_name': signal_data.get('rule_name', 'UNKNOWN'),
                'confidence': signal_data.get('confidence', 0),
                'grade': signal_data.get('grade', 'N/A'),
                'blocking_reason': blocking_reason
            }
            
            self._blocked_signals[user_id].append(blocked_entry)
            
            if len(self._blocked_signals[user_id]) > 100:
                self._blocked_signals[user_id] = self._blocked_signals[user_id][-100:]
            
            logger.info(
                f"📊 Blocked signal tracked - User:{user_id} "
                f"Type:{signal_data.get('signal_type')} "
                f"Reason:{blocking_reason}"
            )
            
            self._check_blocking_rate_alert(user_id)
    
    def _check_blocking_rate_alert(self, user_id: int) -> None:
        """Check if blocking rate exceeds threshold and generate alert."""
        now = datetime.now(pytz.UTC)
        window_start = now - timedelta(minutes=self.BLOCKING_RATE_WINDOW_MINUTES)
        
        last_check = self._last_blocking_rate_check.get(user_id)
        if last_check and (now - last_check).total_seconds() < 60:
            return
        
        with self._blocked_signals_lock:
            user_blocked = self._blocked_signals.get(user_id, [])
            recent_blocked = [s for s in user_blocked if s['timestamp'] >= window_start]
        
        if len(recent_blocked) < 5:
            return
        
        blocking_count = len(recent_blocked)
        
        if blocking_count >= 10:
            self._last_blocking_rate_check[user_id] = now
            
            alert = QualityAlert(
                alert_type="HIGH_BLOCKING_RATE",
                message=(
                    f"⚠️ Tingkat blocking signal tinggi!\n"
                    f"User {user_id} memiliki {blocking_count} signal di-block "
                    f"dalam {self.BLOCKING_RATE_WINDOW_MINUTES} menit terakhir.\n"
                    f"Pertimbangkan untuk menutup posisi aktif."
                ),
                severity="WARNING",
                data={
                    'user_id': user_id,
                    'blocked_count': blocking_count,
                    'window_minutes': self.BLOCKING_RATE_WINDOW_MINUTES
                }
            )
            self._pending_alerts.append(alert)
            logger.warning(f"⚠️ High blocking rate alert for user {user_id}: {blocking_count} blocked")
    
    def get_blocking_stats(self, user_id: int, minutes: int = 60) -> Dict[str, Any]:
        """
        Get blocking statistics untuk user.
        
        Args:
            user_id: ID user
            minutes: Window waktu untuk analisis
        
        Returns:
            Dict dengan statistik blocking
        """
        now = datetime.now(pytz.UTC)
        window_start = now - timedelta(minutes=minutes)
        
        with self._blocked_signals_lock:
            user_blocked = self._blocked_signals.get(user_id, [])
            recent_blocked = [s for s in user_blocked if s['timestamp'] >= window_start]
        
        if not recent_blocked:
            return {
                'total_blocked': 0,
                'by_reason': {},
                'by_type': {},
                'window_minutes': minutes
            }
        
        by_reason = {}
        for s in recent_blocked:
            reason = s.get('blocking_reason', 'unknown')
            by_reason[reason] = by_reason.get(reason, 0) + 1
        
        by_type = {}
        for s in recent_blocked:
            sig_type = s.get('signal_type', 'unknown')
            by_type[sig_type] = by_type.get(sig_type, 0) + 1
        
        return {
            'total_blocked': len(recent_blocked),
            'by_reason': by_reason,
            'by_type': by_type,
            'window_minutes': minutes,
            'high_quality_blocked': sum(1 for s in recent_blocked 
                                        if s.get('grade') in ['A', 'Grade A'] or 
                                        s.get('confidence', 0) >= 0.9)
        }
    
    def clear_blocked_signals(self, user_id: int) -> int:
        """Clear blocked signals history untuk user. Returns count cleared."""
        with self._blocked_signals_lock:
            if user_id in self._blocked_signals:
                count = len(self._blocked_signals[user_id])
                del self._blocked_signals[user_id]
                logger.info(f"Cleared {count} blocked signals for user {user_id}")
                return count
            return 0
