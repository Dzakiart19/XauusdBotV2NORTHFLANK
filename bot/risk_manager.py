from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass, field
import pytz
from sqlalchemy.exc import SQLAlchemyError
from bot.logger import setup_logger

logger = setup_logger('RiskManager')


class RiskManagerError(Exception):
    """Exception untuk error pada risk management"""
    pass


@dataclass
class PartialExitLevel:
    """Data class untuk level partial exit"""
    level_name: str
    price: float
    percentage: float
    action: str
    pips_from_entry: float


@dataclass
class ExposureRecord:
    """Data class untuk tracking exposure per user"""
    user_id: int
    open_positions: List[Dict] = field(default_factory=list)
    total_risk_amount: float = 0.0
    daily_realized_pl: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(pytz.UTC))


class DynamicRiskCalculator:
    """
    Dynamic Risk Calculator untuk XAUUSD trading bot.
    
    Features:
    - Daily Exposure Control dengan max loss threshold
    - Dynamic Lot Sizing berdasarkan SL pips dan risk amount
    - Partial Exit Strategy dengan trailing stop
    - Exposure Tracking per user
    
    Attributes:
        config: Configuration object
        db: Database manager
        risk_manager: Parent RiskManager instance (optional)
    """
    
    DEFAULT_MAX_DAILY_LOSS = 10.0
    DEFAULT_MAX_DAILY_LOSS_PERCENT = 1.0
    DEFAULT_MAX_CONCURRENT_POSITIONS = 4
    DEFAULT_RISK_SAFETY_FACTOR = 0.5
    
    MIN_LOT_SIZE = 0.01
    MAX_LOT_SIZE = 0.1
    
    PARTIAL_EXIT_FIRST_TP_PERCENT = 0.40
    PARTIAL_EXIT_FIRST_TP_RR = 1.0
    PARTIAL_EXIT_SECOND_TP_PERCENT = 0.35
    PARTIAL_EXIT_SECOND_TP_RR = 1.5
    PARTIAL_EXIT_TRAILING_PERCENT = 0.25
    
    TRAILING_STOP_MOVE_PIPS = 5.0
    TRAILING_STOP_DISTANCE_PIPS = 3.0
    
    XAUUSD_PIP_VALUE = 10.0
    
    def __init__(self, config, db_manager, risk_manager=None):
        """
        Initialize DynamicRiskCalculator.
        
        Args:
            config: Configuration object dengan settings trading
            db_manager: Database manager untuk akses data
            risk_manager: Optional parent RiskManager instance
        """
        self.config = config
        self.db = db_manager
        self.risk_manager = risk_manager
        
        self._exposure_cache: Dict[int, ExposureRecord] = {}
        self._position_timestamps: Dict[int, List[datetime]] = {}
        
        self.max_daily_loss = getattr(config, 'MAX_DAILY_LOSS_AMOUNT', self.DEFAULT_MAX_DAILY_LOSS)
        self.max_daily_loss_percent = getattr(config, 'MAX_DAILY_LOSS_PERCENT', self.DEFAULT_MAX_DAILY_LOSS_PERCENT)
        self.max_concurrent_positions = getattr(config, 'MAX_CONCURRENT_POSITIONS', self.DEFAULT_MAX_CONCURRENT_POSITIONS)
        self.risk_safety_factor = getattr(config, 'RISK_SAFETY_FACTOR', self.DEFAULT_RISK_SAFETY_FACTOR)
        
        account_balance = getattr(config, 'ACCOUNT_BALANCE', 1000.0)
        threshold_from_percent = account_balance * (self.max_daily_loss_percent / 100)
        self.effective_daily_threshold = min(self.max_daily_loss, threshold_from_percent)
        
        logger.info(f"🎯 DynamicRiskCalculator initialized: "
                   f"Max Daily Loss=${self.effective_daily_threshold:.2f}, "
                   f"Max Concurrent={self.max_concurrent_positions}, "
                   f"Safety Factor={self.risk_safety_factor}")
    
    def can_open_position(self, user_id: int) -> Tuple[bool, Optional[str]]:
        """
        Check apakah user bisa membuka posisi baru.
        
        Args:
            user_id: ID user yang akan dicek
            
        Returns:
            Tuple[bool, Optional[str]]: (can_open, reason)
            - True, None jika bisa buka posisi
            - False, reason jika tidak bisa buka posisi
        """
        try:
            exposure_status = self.get_exposure_status(user_id)
            
            current_positions = exposure_status.get('open_positions_count', 0)
            if current_positions >= self.max_concurrent_positions:
                reason = (f"Max concurrent positions reached: {current_positions}/{self.max_concurrent_positions}. "
                         f"Close existing positions or wait for auto-close.")
                logger.info(f"❌ User {user_id} blocked: {reason}")
                return False, reason
            
            total_exposure = exposure_status.get('total_risk_amount', 0.0)
            daily_loss = exposure_status.get('daily_realized_loss', 0.0)
            combined_exposure = total_exposure + abs(daily_loss)
            
            if combined_exposure >= self.effective_daily_threshold:
                reason = (f"Daily exposure limit reached: ${combined_exposure:.2f} >= ${self.effective_daily_threshold:.2f}. "
                         f"Trading blocked until next day.")
                logger.info(f"❌ User {user_id} blocked: {reason}")
                return False, reason
            
            remaining_exposure = self.effective_daily_threshold - combined_exposure
            remaining_slots = self.max_concurrent_positions - current_positions
            
            if remaining_exposure <= 0 or remaining_slots <= 0:
                reason = f"No remaining exposure capacity (${remaining_exposure:.2f}) or slots ({remaining_slots})"
                logger.info(f"❌ User {user_id} blocked: {reason}")
                return False, reason
            
            logger.debug(f"✅ User {user_id} can open position: "
                        f"Positions={current_positions}/{self.max_concurrent_positions}, "
                        f"Exposure=${combined_exposure:.2f}/${self.effective_daily_threshold:.2f}")
            return True, None
            
        except SQLAlchemyError as e:
            logger.error(f"Database error in can_open_position: {e}")
            return True, None
        except (ValueError, TypeError) as e:
            logger.error(f"Value/Type error in can_open_position: {e}")
            return True, None
        except Exception as e:
            logger.error(f"Unexpected error in can_open_position: {e}")
            return True, None
    
    def calculate_risk_per_trade(self, user_id: int) -> float:
        """
        Calculate risk amount per trade berdasarkan exposure dan concurrent positions.
        
        Formula: risk_per_trade = (daily_threshold - current_exposure) / remaining_slots * safety_factor
        
        Args:
            user_id: ID user
            
        Returns:
            float: Risk amount dalam USD
        """
        try:
            exposure_status = self.get_exposure_status(user_id)
            
            current_exposure = exposure_status.get('total_risk_amount', 0.0)
            daily_loss = exposure_status.get('daily_realized_loss', 0.0)
            combined_exposure = current_exposure + abs(daily_loss)
            
            remaining_threshold = self.effective_daily_threshold - combined_exposure
            
            current_positions = exposure_status.get('open_positions_count', 0)
            remaining_slots = max(1, self.max_concurrent_positions - current_positions)
            
            risk_per_trade = (remaining_threshold / remaining_slots) * self.risk_safety_factor
            
            min_risk = 0.50
            max_risk = self.effective_daily_threshold * 0.25
            risk_per_trade = max(min_risk, min(risk_per_trade, max_risk))
            
            logger.debug(f"Risk per trade for user {user_id}: ${risk_per_trade:.2f} "
                        f"(Remaining: ${remaining_threshold:.2f}, Slots: {remaining_slots})")
            
            return round(risk_per_trade, 2)
            
        except Exception as e:
            logger.error(f"Error calculating risk per trade: {e}")
            default_risk = self.effective_daily_threshold / self.max_concurrent_positions * self.risk_safety_factor
            return round(default_risk, 2)
    
    def calculate_dynamic_lot(self, sl_pips: float, account_balance: float, 
                             user_id: Optional[int] = None) -> float:
        """
        Calculate lot size berdasarkan SL pips dan risk amount.
        
        Formula: lot_size = risk_amount / (sl_pips * pip_value)
        
        Args:
            sl_pips: Stop loss dalam pips
            account_balance: Balance akun dalam USD
            user_id: Optional user ID untuk dynamic risk calculation
            
        Returns:
            float: Lot size (min 0.01, max 0.1)
        """
        try:
            if sl_pips <= 0:
                logger.warning(f"Invalid SL pips: {sl_pips}, using minimum lot")
                return self.MIN_LOT_SIZE
            
            if account_balance <= 0:
                logger.warning(f"Invalid account balance: {account_balance}")
                return self.MIN_LOT_SIZE
            
            if user_id is not None:
                risk_amount = self.calculate_risk_per_trade(user_id)
            else:
                risk_percent = getattr(self.config, 'RISK_PER_TRADE_PERCENT', 2.0)
                risk_amount = account_balance * (risk_percent / 100)
            
            pip_value = getattr(self.config, 'XAUUSD_PIP_VALUE', self.XAUUSD_PIP_VALUE)
            
            lot_size = risk_amount / (sl_pips * pip_value)
            
            lot_size = max(self.MIN_LOT_SIZE, min(lot_size, self.MAX_LOT_SIZE))
            
            lot_size = round(lot_size, 2)
            
            logger.info(f"📊 Dynamic lot calculation: "
                       f"Risk=${risk_amount:.2f}, SL={sl_pips:.1f} pips, "
                       f"Pip Value=${pip_value:.2f}, Lot={lot_size:.2f}")
            
            return lot_size
            
        except ZeroDivisionError:
            logger.error("Division by zero in calculate_dynamic_lot")
            return self.MIN_LOT_SIZE
        except (ValueError, TypeError) as e:
            logger.error(f"Value/Type error in calculate_dynamic_lot: {e}")
            return self.MIN_LOT_SIZE
        except Exception as e:
            logger.error(f"Unexpected error in calculate_dynamic_lot: {e}")
            return self.MIN_LOT_SIZE
    
    def get_partial_exit_levels(self, entry_price: float, signal_type: str, 
                                sl_pips: float) -> Dict[str, Any]:
        """
        Calculate partial exit levels untuk Partial Exit Strategy.
        
        Strategy:
        - 40% close at 1.0x risk (first TP) -> move SL to breakeven
        - 35% close at 1.5x risk (second TP)
        - 25% keep with trailing stop
        
        Args:
            entry_price: Harga entry
            signal_type: 'BUY' atau 'SELL'
            sl_pips: Stop loss dalam pips
            
        Returns:
            Dict dengan detail partial exit levels
        """
        try:
            pip_value = getattr(self.config, 'XAUUSD_PIP_VALUE', self.XAUUSD_PIP_VALUE)
            sl_distance = sl_pips / pip_value
            
            first_tp_pips = sl_pips * self.PARTIAL_EXIT_FIRST_TP_RR
            second_tp_pips = sl_pips * self.PARTIAL_EXIT_SECOND_TP_RR
            
            first_tp_distance = first_tp_pips / pip_value
            second_tp_distance = second_tp_pips / pip_value
            
            if signal_type == 'BUY':
                first_tp_price = entry_price + first_tp_distance
                second_tp_price = entry_price + second_tp_distance
                sl_price = entry_price - sl_distance
                breakeven_sl = entry_price + (0.5 / pip_value)
            else:
                first_tp_price = entry_price - first_tp_distance
                second_tp_price = entry_price - second_tp_distance
                sl_price = entry_price + sl_distance
                breakeven_sl = entry_price - (0.5 / pip_value)
            
            levels = {
                'entry_price': round(entry_price, 2),
                'signal_type': signal_type,
                'original_sl': round(sl_price, 2),
                'sl_pips': round(sl_pips, 1),
                
                'first_tp': {
                    'price': round(first_tp_price, 2),
                    'percentage': self.PARTIAL_EXIT_FIRST_TP_PERCENT * 100,
                    'rr_ratio': self.PARTIAL_EXIT_FIRST_TP_RR,
                    'pips_from_entry': round(first_tp_pips, 1),
                    'action': 'close_partial_move_sl_breakeven',
                    'new_sl_after': round(breakeven_sl, 2)
                },
                
                'second_tp': {
                    'price': round(second_tp_price, 2),
                    'percentage': self.PARTIAL_EXIT_SECOND_TP_PERCENT * 100,
                    'rr_ratio': self.PARTIAL_EXIT_SECOND_TP_RR,
                    'pips_from_entry': round(second_tp_pips, 1),
                    'action': 'close_partial'
                },
                
                'trailing_portion': {
                    'percentage': self.PARTIAL_EXIT_TRAILING_PERCENT * 100,
                    'trailing_trigger_pips': self.TRAILING_STOP_MOVE_PIPS,
                    'trailing_distance_pips': self.TRAILING_STOP_DISTANCE_PIPS,
                    'action': 'trailing_stop'
                },
                
                'trailing_config': {
                    'move_every_pips': self.TRAILING_STOP_MOVE_PIPS,
                    'distance_pips': self.TRAILING_STOP_DISTANCE_PIPS
                }
            }
            
            logger.info(f"📈 Partial exit levels for {signal_type} @ {entry_price:.2f}: "
                       f"TP1=${first_tp_price:.2f} (40%), TP2=${second_tp_price:.2f} (35%), "
                       f"Trailing (25%) every {self.TRAILING_STOP_MOVE_PIPS} pips")
            
            return levels
            
        except (ValueError, TypeError, ZeroDivisionError) as e:
            logger.error(f"Error calculating partial exit levels: {e}")
            return {
                'error': str(e),
                'entry_price': entry_price,
                'signal_type': signal_type,
                'sl_pips': sl_pips
            }
        except Exception as e:
            logger.error(f"Unexpected error in get_partial_exit_levels: {e}")
            return {
                'error': str(e),
                'entry_price': entry_price,
                'signal_type': signal_type,
                'sl_pips': sl_pips
            }
    
    def calculate_trailing_stop(self, entry_price: float, current_price: float, 
                                signal_type: str, current_sl: float) -> Optional[float]:
        """
        Calculate new trailing stop level.
        
        Rules:
        - Move SL setiap +5 pips profit
        - Trailing distance 3 pips dari current price
        
        Args:
            entry_price: Harga entry
            current_price: Harga saat ini
            signal_type: 'BUY' atau 'SELL'
            current_sl: Stop loss saat ini
            
        Returns:
            Optional[float]: New SL price atau None jika tidak perlu update
        """
        try:
            pip_value = getattr(self.config, 'XAUUSD_PIP_VALUE', self.XAUUSD_PIP_VALUE)
            move_pips = self.TRAILING_STOP_MOVE_PIPS
            distance_pips = self.TRAILING_STOP_DISTANCE_PIPS
            
            if signal_type == 'BUY':
                profit_pips = (current_price - entry_price) * pip_value
                
                if profit_pips < move_pips:
                    return None
                
                new_sl = current_price - (distance_pips / pip_value)
                
                if new_sl > current_sl:
                    logger.debug(f"📈 Trailing stop update (BUY): "
                                f"Current SL=${current_sl:.2f} -> New SL=${new_sl:.2f}, "
                                f"Profit={profit_pips:.1f} pips")
                    return round(new_sl, 2)
                    
            else:
                profit_pips = (entry_price - current_price) * pip_value
                
                if profit_pips < move_pips:
                    return None
                
                new_sl = current_price + (distance_pips / pip_value)
                
                if new_sl < current_sl:
                    logger.debug(f"📈 Trailing stop update (SELL): "
                                f"Current SL=${current_sl:.2f} -> New SL=${new_sl:.2f}, "
                                f"Profit={profit_pips:.1f} pips")
                    return round(new_sl, 2)
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating trailing stop: {e}")
            return None
    
    def get_exposure_status(self, user_id: int) -> Dict[str, Any]:
        """
        Get exposure status untuk user tertentu.
        
        Returns:
            Dict dengan informasi exposure:
            - open_positions_count: Jumlah posisi terbuka
            - total_risk_amount: Total risk dari semua posisi terbuka
            - daily_realized_pl: Total realized P/L hari ini
            - daily_realized_loss: Total realized loss hari ini
            - remaining_daily_exposure: Sisa exposure yang tersedia
            - can_open_new: Boolean apakah bisa buka posisi baru
        """
        try:
            utc_now = datetime.now(pytz.UTC)
            jakarta_tz = pytz.timezone('Asia/Jakarta')
            jakarta_time = utc_now.astimezone(jakarta_tz)
            today_start = jakarta_time.replace(hour=0, minute=0, second=0, microsecond=0)
            today_start_utc = today_start.astimezone(pytz.UTC)
            
            cache_key = user_id
            cache_ttl = 30
            
            if cache_key in self._exposure_cache:
                cached = self._exposure_cache[cache_key]
                cache_age = (utc_now - cached.last_updated).total_seconds()
                if cache_age < cache_ttl:
                    return self._build_exposure_response(cached)
            
            session = self.db.get_session()
            if session is None:
                logger.error("Failed to get database session for exposure status")
                return self._get_default_exposure_status()
            
            try:
                from bot.database import Position, Trade
                
                open_positions = session.query(Position).filter(
                    Position.user_id == user_id,
                    Position.status == 'ACTIVE'
                ).all()
                
                position_data = []
                total_risk = 0.0
                
                for pos in open_positions:
                    sl_distance = abs(pos.entry_price - pos.stop_loss)
                    sl_pips = sl_distance * self.XAUUSD_PIP_VALUE
                    lot_size = getattr(self.config, 'LOT_SIZE', 0.01)
                    position_risk = sl_pips * lot_size * 10
                    
                    position_data.append({
                        'trade_id': pos.trade_id,
                        'entry_price': pos.entry_price,
                        'stop_loss': pos.stop_loss,
                        'risk_amount': round(position_risk, 2),
                        'opened_at': pos.opened_at
                    })
                    total_risk += position_risk
                
                daily_trades = session.query(Trade).filter(
                    Trade.user_id == user_id,
                    Trade.signal_time >= today_start_utc,
                    Trade.actual_pl.isnot(None)
                ).all()
                
                daily_realized_pl = sum(t.actual_pl for t in daily_trades if t.actual_pl)
                daily_realized_loss = sum(t.actual_pl for t in daily_trades 
                                         if t.actual_pl and t.actual_pl < 0)
                
                exposure_record = ExposureRecord(
                    user_id=user_id,
                    open_positions=position_data,
                    total_risk_amount=total_risk,
                    daily_realized_pl=daily_realized_pl,
                    last_updated=utc_now
                )
                
                self._exposure_cache[cache_key] = exposure_record
                
                return self._build_exposure_response(exposure_record, daily_realized_loss)
                
            except SQLAlchemyError as e:
                logger.error(f"Database error getting exposure status: {e}")
                return self._get_default_exposure_status()
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Unexpected error in get_exposure_status: {e}")
            return self._get_default_exposure_status()
    
    def _build_exposure_response(self, record: ExposureRecord, 
                                  daily_loss: float = 0.0) -> Dict[str, Any]:
        """Build exposure status response dari ExposureRecord."""
        combined_exposure = record.total_risk_amount + abs(daily_loss)
        remaining = max(0, self.effective_daily_threshold - combined_exposure)
        
        open_count = len(record.open_positions)
        can_open = (open_count < self.max_concurrent_positions and 
                   remaining > 0)
        
        return {
            'user_id': record.user_id,
            'open_positions_count': open_count,
            'open_positions': record.open_positions,
            'total_risk_amount': round(record.total_risk_amount, 2),
            'daily_realized_pl': round(record.daily_realized_pl, 2),
            'daily_realized_loss': round(daily_loss, 2),
            'combined_exposure': round(combined_exposure, 2),
            'daily_threshold': self.effective_daily_threshold,
            'remaining_daily_exposure': round(remaining, 2),
            'max_concurrent_positions': self.max_concurrent_positions,
            'remaining_position_slots': max(0, self.max_concurrent_positions - open_count),
            'can_open_new': can_open,
            'last_updated': record.last_updated.isoformat()
        }
    
    def _get_default_exposure_status(self) -> Dict[str, Any]:
        """Return default exposure status saat error."""
        return {
            'user_id': 0,
            'open_positions_count': 0,
            'open_positions': [],
            'total_risk_amount': 0.0,
            'daily_realized_pl': 0.0,
            'daily_realized_loss': 0.0,
            'combined_exposure': 0.0,
            'daily_threshold': self.effective_daily_threshold,
            'remaining_daily_exposure': self.effective_daily_threshold,
            'max_concurrent_positions': self.max_concurrent_positions,
            'remaining_position_slots': self.max_concurrent_positions,
            'can_open_new': True,
            'last_updated': datetime.now(pytz.UTC).isoformat(),
            'error': 'Using default values due to error'
        }
    
    def update_exposure(self, user_id: int, trade_result: Dict[str, Any]) -> None:
        """
        Update exposure tracking setelah trade selesai.
        
        Args:
            user_id: ID user
            trade_result: Dict dengan hasil trade:
                - trade_id: ID trade
                - actual_pl: Realized P/L
                - status: 'CLOSED', 'TP_HIT', 'SL_HIT', etc.
        """
        try:
            if user_id in self._exposure_cache:
                del self._exposure_cache[user_id]
                logger.debug(f"Cleared exposure cache for user {user_id}")
            
            trade_id = trade_result.get('trade_id')
            actual_pl = trade_result.get('actual_pl', 0.0)
            status = trade_result.get('status', 'CLOSED')
            
            logger.info(f"📊 Exposure updated for user {user_id}: "
                       f"Trade #{trade_id} {status}, P/L=${actual_pl:.2f}")
            
            new_status = self.get_exposure_status(user_id)
            logger.debug(f"New exposure status: {new_status}")
            
        except Exception as e:
            logger.error(f"Error updating exposure: {e}")
    
    def get_oldest_position(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Get oldest open position untuk auto-close jika max concurrent reached.
        
        Args:
            user_id: ID user
            
        Returns:
            Optional[Dict]: Position data atau None jika tidak ada posisi
        """
        try:
            exposure_status = self.get_exposure_status(user_id)
            positions = exposure_status.get('open_positions', [])
            
            if not positions:
                return None
            
            oldest = min(positions, key=lambda p: p.get('opened_at', datetime.max))
            
            logger.debug(f"Oldest position for user {user_id}: Trade #{oldest.get('trade_id')}")
            return oldest
            
        except Exception as e:
            logger.error(f"Error getting oldest position: {e}")
            return None
    
    def should_auto_close_oldest(self, user_id: int) -> Tuple[bool, Optional[Dict]]:
        """
        Check apakah perlu auto-close oldest position.
        
        Args:
            user_id: ID user
            
        Returns:
            Tuple[bool, Optional[Dict]]: (should_close, position_to_close)
        """
        try:
            exposure_status = self.get_exposure_status(user_id)
            open_count = exposure_status.get('open_positions_count', 0)
            
            if open_count >= self.max_concurrent_positions:
                oldest = self.get_oldest_position(user_id)
                if oldest:
                    logger.info(f"⚠️ Auto-close recommended for user {user_id}: "
                               f"Max concurrent ({self.max_concurrent_positions}) reached, "
                               f"closing Trade #{oldest.get('trade_id')}")
                    return True, oldest
            
            return False, None
            
        except Exception as e:
            logger.error(f"Error checking auto-close: {e}")
            return False, None
    
    def get_summary(self, user_id: int) -> Dict[str, Any]:
        """
        Get summary lengkap dari dynamic risk settings dan status.
        
        Args:
            user_id: ID user
            
        Returns:
            Dict dengan summary lengkap
        """
        try:
            exposure = self.get_exposure_status(user_id)
            risk_per_trade = self.calculate_risk_per_trade(user_id)
            can_open, reason = self.can_open_position(user_id)
            
            return {
                'settings': {
                    'max_daily_loss': self.effective_daily_threshold,
                    'max_concurrent_positions': self.max_concurrent_positions,
                    'risk_safety_factor': self.risk_safety_factor,
                    'min_lot_size': self.MIN_LOT_SIZE,
                    'max_lot_size': self.MAX_LOT_SIZE
                },
                'partial_exit': {
                    'first_tp_percent': self.PARTIAL_EXIT_FIRST_TP_PERCENT * 100,
                    'first_tp_rr': self.PARTIAL_EXIT_FIRST_TP_RR,
                    'second_tp_percent': self.PARTIAL_EXIT_SECOND_TP_PERCENT * 100,
                    'second_tp_rr': self.PARTIAL_EXIT_SECOND_TP_RR,
                    'trailing_percent': self.PARTIAL_EXIT_TRAILING_PERCENT * 100,
                    'trailing_move_pips': self.TRAILING_STOP_MOVE_PIPS,
                    'trailing_distance_pips': self.TRAILING_STOP_DISTANCE_PIPS
                },
                'current_status': {
                    'user_id': user_id,
                    'can_open_position': can_open,
                    'block_reason': reason,
                    'risk_per_trade': risk_per_trade,
                    'exposure': exposure
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting summary: {e}")
            return {'error': str(e)}

class RiskManager:
    def __init__(self, config, db_manager, alert_system=None, enable_dynamic_risk=True):
        self.config = config
        self.db = db_manager
        self.alert_system = alert_system
        self.last_signal_time = {}
        self.daily_stats = {}
        self.risk_warning_sent_today = {}
        
        self.dynamic_risk: Optional[DynamicRiskCalculator] = None
        if enable_dynamic_risk:
            try:
                self.dynamic_risk = DynamicRiskCalculator(config, db_manager, risk_manager=self)
                logger.info("✅ DynamicRiskCalculator integrated with RiskManager")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize DynamicRiskCalculator: {e}")
                self.dynamic_risk = None
    
    def get_dynamic_risk_calculator(self) -> Optional[DynamicRiskCalculator]:
        """Get the DynamicRiskCalculator instance if available."""
        return self.dynamic_risk
    
    def can_open_position_dynamic(self, user_id: int) -> Tuple[bool, Optional[str]]:
        """
        Check if user can open position using dynamic risk controls.
        Falls back to True if DynamicRiskCalculator not available.
        """
        if self.dynamic_risk:
            return self.dynamic_risk.can_open_position(user_id)
        return True, None
    
    def calculate_lot_dynamic(self, sl_pips: float, account_balance: float, 
                             user_id: Optional[int] = None) -> float:
        """
        Calculate lot size using dynamic risk calculator.
        Falls back to config LOT_SIZE if not available.
        """
        if self.dynamic_risk:
            return self.dynamic_risk.calculate_dynamic_lot(sl_pips, account_balance, user_id)
        return getattr(self.config, 'LOT_SIZE', 0.01)
    
    def get_partial_exits(self, entry_price: float, signal_type: str, 
                         sl_pips: float) -> Dict[str, Any]:
        """
        Get partial exit levels using dynamic risk calculator.
        """
        if self.dynamic_risk:
            return self.dynamic_risk.get_partial_exit_levels(entry_price, signal_type, sl_pips)
        return {}
    
    def get_exposure(self, user_id: int) -> Dict[str, Any]:
        """Get exposure status for user using dynamic risk calculator."""
        if self.dynamic_risk:
            return self.dynamic_risk.get_exposure_status(user_id)
        return {'error': 'DynamicRiskCalculator not available'}
    
    def update_exposure_after_trade(self, user_id: int, trade_result: Dict[str, Any]) -> None:
        """Update exposure after trade using dynamic risk calculator."""
        if self.dynamic_risk:
            self.dynamic_risk.update_exposure(user_id, trade_result)
        
    def can_trade(self, user_id: int, signal_type: str, spread: Optional[float] = None) -> tuple[bool, Optional[str]]:
        utc_now = datetime.now(pytz.UTC)
        jakarta_tz = pytz.timezone('Asia/Jakarta')
        jakarta_time = utc_now.astimezone(jakarta_tz)
        today_str = jakarta_time.strftime('%Y-%m-%d')
        
        time_valid, time_reason = self.check_time_filter()
        if not time_valid:
            logger.info(f"Trade blocked for user {user_id}: {time_reason}")
            return False, time_reason
        
        if spread is not None:
            spread_valid, spread_reason = self.check_spread_filter(spread)
            if not spread_valid:
                logger.info(f"Trade blocked for user {user_id}: {spread_reason}")
                return False, spread_reason
        
        # UNLIMITED - Cooldown dihapus, sinyal tanpa batas
        # Tidak ada pengecekan waktu cooldown antar sinyal
        
        cache_key = f"{user_id}_{today_str}"
        cache_ttl = 60
        
        if cache_key in self.daily_stats:
            cached_data = self.daily_stats[cache_key]
            cache_age = (utc_now - cached_data['timestamp']).total_seconds()
            if cache_age < cache_ttl:
                total_daily_pl = cached_data['total_pl']
                logger.debug(f"Using cached daily P/L for user {user_id}: ${total_daily_pl:.2f} (age: {cache_age:.1f}s)")
            else:
                total_daily_pl = None
        else:
            total_daily_pl = None
        
        if total_daily_pl is None:
            session = self.db.get_session()
            if session is None:
                logger.error("Failed to get database session")
                return False, "Error: Database session tidak tersedia"
            try:
                from bot.database import Trade
                
                today_start = jakarta_time.replace(hour=0, minute=0, second=0, microsecond=0)
                today_start_utc = today_start.astimezone(pytz.UTC)
                
                daily_pl = session.query(Trade).filter(
                    Trade.user_id == user_id,
                    Trade.signal_time >= today_start_utc,
                    Trade.actual_pl.isnot(None)
                ).with_entities(Trade.actual_pl).all()
                
                total_daily_pl = sum([pl[0] for pl in daily_pl if pl[0] is not None])
                
                self.daily_stats[cache_key] = {
                    'total_pl': total_daily_pl,
                    'timestamp': utc_now
                }
                logger.debug(f"Cached daily P/L for user {user_id}: ${total_daily_pl:.2f}")
                
            except SQLAlchemyError as e:
                logger.error(f"Database error checking trade eligibility: {e}")
                return False, f"Error database: {str(e)}"
            except ValueError as e:
                logger.error(f"Nilai tidak valid saat checking trade eligibility: {e}")
                return False, f"Error nilai: {str(e)}"
            except TypeError as e:
                logger.error(f"Tipe data salah saat checking trade eligibility: {e}")
                return False, f"Error tipe data: {str(e)}"
            except Exception as e:
                logger.error(f"Error tidak terduga checking trade eligibility: {e}")
                raise RiskManagerError(f"Error checking trade eligibility: {e}") from e
            finally:
                session.close()
        
        # UNLIMITED - Batas kerugian harian dihapus
        # Tidak ada pengecekan daily loss limit
        
        return True, None
    
    def check_spread_filter(self, spread: float) -> Tuple[bool, str]:
        try:
            max_spread = self.config.MAX_SPREAD_PIPS
            
            if spread <= 0:
                logger.warning(f"Invalid spread value: {spread}")
                return True, "Spread valid (invalid value ignored)"
            
            if spread > max_spread:
                reason = f"Spread terlalu tinggi: {spread:.1f} pips (max: {max_spread:.1f} pips)"
                logger.info(f"Spread filter blocked: {reason}")
                return False, reason
            
            logger.debug(f"Spread check passed: {spread:.1f} pips <= {max_spread:.1f} pips")
            return True, "Spread dalam batas normal"
            
        except AttributeError as e:
            logger.error(f"Config attribute error in check_spread_filter: {e}")
            return True, f"Spread check error (config): {str(e)}"
        except (ValueError, TypeError) as e:
            logger.error(f"Nilai/tipe error in check_spread_filter: {e}")
            return True, f"Spread check error (nilai): {str(e)}"
        except Exception as e:
            logger.error(f"Error tidak terduga in check_spread_filter: {e}")
            return True, f"Spread check error: {str(e)}"
    
    def check_time_filter(self) -> Tuple[bool, str]:
        try:
            if getattr(self.config, 'UNLIMITED_TRADING_HOURS', False):
                logger.debug("UNLIMITED_TRADING_HOURS aktif - time filter dilewati")
                return True, "Mode unlimited - trading 24/7"
            
            utc_now = datetime.now(pytz.UTC)
            jakarta_tz = pytz.timezone('Asia/Jakarta')
            jakarta_time = utc_now.astimezone(jakarta_tz)
            
            current_hour = jakarta_time.hour
            current_weekday = jakarta_time.weekday()
            
            if current_weekday == 4:
                friday_cutoff = self.config.FRIDAY_CUTOFF_HOUR
                if friday_cutoff < 24 and current_hour >= friday_cutoff:
                    reason = f"Trading dihentikan: Jumat setelah jam {friday_cutoff}:00 WIB"
                    logger.info(f"Friday cutoff: {reason}")
                    return False, reason
            
            if current_weekday == 5:
                return False, "Trading dihentikan: Hari Sabtu"
            
            if current_weekday == 6:
                return False, "Trading dihentikan: Hari Minggu"
            
            trading_start = self.config.TRADING_HOURS_START
            trading_end = self.config.TRADING_HOURS_END
            
            if trading_end >= 24:
                logger.debug(f"Trading 24 jam aktif")
                return True, "Trading 24 jam aktif"
            
            if current_hour < trading_start:
                reason = f"Di luar jam trading: sebelum {trading_start}:00 WIB (sekarang {current_hour}:00)"
                logger.info(f"Time filter blocked: {reason}")
                return False, reason
            
            if current_hour >= trading_end:
                reason = f"Di luar jam trading: setelah {trading_end}:00 WIB (sekarang {current_hour}:00)"
                logger.info(f"Time filter blocked: {reason}")
                return False, reason
            
            logger.debug(f"Time check passed: {current_hour}:00 WIB (trading hours: {trading_start}:00-{trading_end}:00)")
            return True, f"Dalam jam trading ({trading_start}:00-{trading_end}:00 WIB)"
            
        except AttributeError as e:
            logger.error(f"Config attribute error in check_time_filter: {e}")
            return True, f"Time check error (config): {str(e)}"
        except KeyError as e:
            logger.error(f"Key error in check_time_filter: {e}")
            return True, f"Time check error (key): {str(e)}"
        except Exception as e:
            logger.error(f"Error tidak terduga in check_time_filter: {e}")
            return True, f"Time check error: {str(e)}"
    
    def record_signal(self, user_id: int):
        self.last_signal_time[user_id] = datetime.now(pytz.UTC)
        logger.debug(f"Signal recorded for user {user_id}, cooldown timer started")
    
    def calculate_position_size(self, account_balance: float, entry_price: float, 
                               stop_loss: float, signal_type: str) -> float:
        try:
            sl_pips = abs(entry_price - stop_loss) * self.config.XAUUSD_PIP_VALUE
            
            if sl_pips <= 0:
                logger.warning(f"Invalid SL pips: {sl_pips}, using default lot size")
                return self.config.LOT_SIZE
            
            lot_size = self.calculate_lot_from_risk(account_balance, sl_pips)
            
            logger.info(f"Calculated position size: {lot_size:.2f} lots "
                       f"(Balance: {account_balance:.2f}, Risk: {self.config.RISK_PER_TRADE_PERCENT}%, "
                       f"SL: {sl_pips:.1f} pips)")
            
            return lot_size
            
        except ValueError as e:
            logger.error(f"Nilai tidak valid calculating position size: {e}")
            return self.config.LOT_SIZE
        except ZeroDivisionError as e:
            logger.error(f"Division by zero calculating position size: {e}")
            return self.config.LOT_SIZE
        except ArithmeticError as e:
            logger.error(f"Arithmetic error calculating position size: {e}")
            return self.config.LOT_SIZE
        except AttributeError as e:
            logger.error(f"Config attribute error calculating position size: {e}")
            return self.config.LOT_SIZE
        except Exception as e:
            logger.error(f"Error tidak terduga calculating position size: {e}")
            return self.config.LOT_SIZE
    
    def calculate_lot_from_risk(self, account_balance: float, sl_pips: float) -> float:
        try:
            if account_balance <= 0:
                logger.warning(f"Invalid account balance: {account_balance}")
                return self.config.LOT_SIZE
            
            if sl_pips <= 0:
                logger.warning(f"Invalid SL pips: {sl_pips}")
                return self.config.LOT_SIZE
            
            risk_percent = self.config.RISK_PER_TRADE_PERCENT
            pip_value = self.config.XAUUSD_PIP_VALUE
            
            risk_amount = account_balance * risk_percent / 100
            
            lot_size = risk_amount / (sl_pips * pip_value)
            
            lot_size = max(0.01, min(lot_size, 1.0))
            
            lot_size = round(lot_size, 2)
            
            logger.debug(f"Lot calculation: Balance={account_balance:.2f}, "
                        f"Risk%={risk_percent}, SL_pips={sl_pips:.1f}, "
                        f"Risk_amount={risk_amount:.2f}, Lot={lot_size:.2f}")
            
            return lot_size
            
        except ValueError as e:
            logger.error(f"Nilai tidak valid in calculate_lot_from_risk: {e}")
            return self.config.LOT_SIZE
        except ZeroDivisionError as e:
            logger.error(f"Division by zero in calculate_lot_from_risk: {e}")
            return self.config.LOT_SIZE
        except ArithmeticError as e:
            logger.error(f"Arithmetic error in calculate_lot_from_risk: {e}")
            return self.config.LOT_SIZE
        except AttributeError as e:
            logger.error(f"Config attribute error in calculate_lot_from_risk: {e}")
            return self.config.LOT_SIZE
        except Exception as e:
            logger.error(f"Error tidak terduga in calculate_lot_from_risk: {e}")
            return self.config.LOT_SIZE
    
    def calculate_dynamic_sl(self, entry_price: float, signal_type: str, 
                            atr: float, spread: float) -> Tuple[float, float]:
        try:
            fixed_risk = getattr(self.config, 'FIXED_RISK_AMOUNT', 2.0)
            lot_size = getattr(self.config, 'LOT_SIZE', 0.01)
            pip_value = getattr(self.config, 'XAUUSD_PIP_VALUE', 10.0)
            
            dollar_per_pip = pip_value * lot_size
            sl_pips_from_risk = fixed_risk / dollar_per_pip if dollar_per_pip > 0 else 20.0
            sl_from_risk = sl_pips_from_risk / pip_value
            
            sl_from_atr = atr * self.config.SL_ATR_MULTIPLIER
            sl_from_min = self.config.MIN_SL_PIPS / self.config.XAUUSD_PIP_VALUE
            sl_from_spread = spread * self.config.MIN_SL_SPREAD_MULTIPLIER
            
            sl_distance = min(sl_from_risk, max(sl_from_atr, sl_from_min, sl_from_spread))
            
            min_tp_ratio = getattr(self.config, 'TP_RR_RATIO', 1.5)
            max_tp_ratio = getattr(self.config, 'TP_RR_RATIO_MAX', 2.5)
            
            trend_strength = getattr(self, '_last_trend_strength', 0.5)
            dynamic_tp_ratio = min_tp_ratio + (max_tp_ratio - min_tp_ratio) * trend_strength
            
            tp_distance = sl_distance * dynamic_tp_ratio
            
            if signal_type == 'BUY':
                sl_price = entry_price - sl_distance
                tp_price = entry_price + tp_distance
            else:
                sl_price = entry_price + sl_distance
                tp_price = entry_price - tp_distance
            
            sl_pips = sl_distance * self.config.XAUUSD_PIP_VALUE
            tp_pips = tp_distance * self.config.XAUUSD_PIP_VALUE
            expected_loss = sl_pips * dollar_per_pip
            expected_profit = tp_pips * dollar_per_pip
            
            logger.info(f"💰 Fixed-Risk SL/TP for {signal_type}: "
                       f"Entry=${entry_price:.2f}, SL=${sl_price:.2f} ({sl_pips:.1f} pips), "
                       f"TP=${tp_price:.2f} ({tp_pips:.1f} pips)")
            logger.info(f"📊 Expected: Max Loss=${expected_loss:.2f}, Target Profit=${expected_profit:.2f}, R:R=1:{dynamic_tp_ratio:.1f}")
            logger.debug(f"SL components - Risk-based: {sl_from_risk:.4f}, ATR: {sl_from_atr:.4f}, "
                        f"Min: {sl_from_min:.4f}, Spread: {sl_from_spread:.4f} -> Selected: {sl_distance:.4f}")
            
            return sl_price, tp_price
            
        except ValueError as e:
            logger.error(f"Nilai tidak valid calculating dynamic SL/TP: {e}")
            default_sl_distance = self.config.DEFAULT_SL_PIPS / self.config.XAUUSD_PIP_VALUE
            default_tp_distance = self.config.DEFAULT_TP_PIPS / self.config.XAUUSD_PIP_VALUE
            if signal_type == 'BUY':
                return entry_price - default_sl_distance, entry_price + default_tp_distance
            else:
                return entry_price + default_sl_distance, entry_price - default_tp_distance
        except ZeroDivisionError as e:
            logger.error(f"Division by zero calculating dynamic SL/TP: {e}")
            default_sl_distance = self.config.DEFAULT_SL_PIPS / self.config.XAUUSD_PIP_VALUE
            default_tp_distance = self.config.DEFAULT_TP_PIPS / self.config.XAUUSD_PIP_VALUE
            if signal_type == 'BUY':
                return entry_price - default_sl_distance, entry_price + default_tp_distance
            else:
                return entry_price + default_sl_distance, entry_price - default_tp_distance
        except ArithmeticError as e:
            logger.error(f"Arithmetic error calculating dynamic SL/TP: {e}")
            default_sl_distance = self.config.DEFAULT_SL_PIPS / self.config.XAUUSD_PIP_VALUE
            default_tp_distance = self.config.DEFAULT_TP_PIPS / self.config.XAUUSD_PIP_VALUE
            if signal_type == 'BUY':
                return entry_price - default_sl_distance, entry_price + default_tp_distance
            else:
                return entry_price + default_sl_distance, entry_price - default_tp_distance
        except AttributeError as e:
            logger.error(f"Config attribute error calculating dynamic SL/TP: {e}")
            default_sl_distance = 50.0 / 10.0
            default_tp_distance = 100.0 / 10.0
            if signal_type == 'BUY':
                return entry_price - default_sl_distance, entry_price + default_tp_distance
            else:
                return entry_price + default_sl_distance, entry_price - default_tp_distance
        except Exception as e:
            logger.error(f"Error tidak terduga calculating dynamic SL/TP: {e}")
            default_sl_distance = 50.0 / 10.0
            default_tp_distance = 100.0 / 10.0
            if signal_type == 'BUY':
                return entry_price - default_sl_distance, entry_price + default_tp_distance
            else:
                return entry_price + default_sl_distance, entry_price - default_tp_distance
    
    def get_daily_stats(self, user_id: int) -> Dict:
        try:
            utc_now = datetime.now(pytz.UTC)
            jakarta_tz = pytz.timezone('Asia/Jakarta')
            jakarta_time = utc_now.astimezone(jakarta_tz)
            
            session = self.db.get_session()
            if session is None:
                logger.error("Failed to get database session for daily stats")
                return {
                    'date': jakarta_time.strftime('%Y-%m-%d'),
                    'error': 'Database session tidak tersedia',
                    'total_trades': 0,
                    'closed_trades': 0,
                    'open_trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'breakeven': 0,
                    'win_rate': 0,
                    'total_pl': 0,
                    'total_profit': 0,
                    'total_loss': 0,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'profit_factor': 'N/A',
                    'daily_loss_percent': 0,
                    'loss_limit_used': 0,
                    'can_trade': True
                }
            try:
                from bot.database import Trade
                
                today_start = jakarta_time.replace(hour=0, minute=0, second=0, microsecond=0)
                today_start_utc = today_start.astimezone(pytz.UTC)
                
                trades = session.query(Trade).filter(
                    Trade.user_id == user_id,
                    Trade.signal_time >= today_start_utc
                ).all()
                
                total_trades = len(trades)
                closed_trades = [t for t in trades if t.actual_pl is not None]
                
                wins = sum(1 for t in closed_trades if t.actual_pl > 0)
                losses = sum(1 for t in closed_trades if t.actual_pl < 0)
                breakeven = sum(1 for t in closed_trades if t.actual_pl == 0)
                
                total_pl = sum(t.actual_pl for t in closed_trades if t.actual_pl is not None)
                total_profit = sum(t.actual_pl for t in closed_trades if t.actual_pl and t.actual_pl > 0)
                total_loss = sum(t.actual_pl for t in closed_trades if t.actual_pl and t.actual_pl < 0)
                
                win_rate = (wins / len(closed_trades) * 100) if closed_trades else 0
                
                avg_win = (total_profit / wins) if wins > 0 else 0
                avg_loss = (abs(total_loss) / losses) if losses > 0 else 0
                profit_factor = (total_profit / abs(total_loss)) if total_loss != 0 else float('inf')
                
                daily_loss_percent = abs(total_pl) / self.config.ACCOUNT_BALANCE * 100 if total_pl < 0 else 0
                loss_limit_used = daily_loss_percent / self.config.DAILY_LOSS_PERCENT * 100 if self.config.DAILY_LOSS_PERCENT > 0 else 0
                
                stats = {
                    'date': jakarta_time.strftime('%Y-%m-%d'),
                    'total_trades': total_trades,
                    'closed_trades': len(closed_trades),
                    'open_trades': total_trades - len(closed_trades),
                    'wins': wins,
                    'losses': losses,
                    'breakeven': breakeven,
                    'win_rate': round(win_rate, 1),
                    'total_pl': round(total_pl, 2),
                    'total_profit': round(total_profit, 2),
                    'total_loss': round(total_loss, 2),
                    'avg_win': round(avg_win, 2),
                    'avg_loss': round(avg_loss, 2),
                    'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 'N/A',
                    'daily_loss_percent': round(daily_loss_percent, 2),
                    'loss_limit_used': round(loss_limit_used, 1),
                    'can_trade': loss_limit_used < 100
                }
                
                logger.debug(f"Daily stats for user {user_id}: {stats}")
                return stats
                
            except SQLAlchemyError as e:
                logger.error(f"Database error getting daily stats: {e}")
                return {
                    'date': jakarta_time.strftime('%Y-%m-%d'),
                    'error': f"Database error: {str(e)}",
                    'total_trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_pl': 0,
                    'can_trade': True
                }
            except ValueError as e:
                logger.error(f"Nilai tidak valid getting daily stats: {e}")
                return {
                    'date': jakarta_time.strftime('%Y-%m-%d'),
                    'error': f"Nilai error: {str(e)}",
                    'total_trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_pl': 0,
                    'can_trade': True
                }
            except TypeError as e:
                logger.error(f"Tipe data salah getting daily stats: {e}")
                return {
                    'date': jakarta_time.strftime('%Y-%m-%d'),
                    'error': f"Tipe error: {str(e)}",
                    'total_trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_pl': 0,
                    'can_trade': True
                }
            except (ZeroDivisionError, ArithmeticError) as e:
                logger.error(f"Arithmetic error getting daily stats: {e}")
                return {
                    'date': jakarta_time.strftime('%Y-%m-%d'),
                    'error': f"Kalkulasi error: {str(e)}",
                    'total_trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_pl': 0,
                    'can_trade': True
                }
            except Exception as e:
                logger.error(f"Error tidak terduga getting daily stats: {e}")
                return {
                    'date': jakarta_time.strftime('%Y-%m-%d'),
                    'error': str(e),
                    'total_trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_pl': 0,
                    'can_trade': True
                }
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Error tidak terduga in get_daily_stats (outer): {e}")
            return {
                'error': str(e),
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'total_pl': 0,
                'can_trade': True
            }
    
    def calculate_pl(self, entry_price: float, exit_price: float, 
                    signal_type: str, lot_size: Optional[float] = None) -> float:
        try:
            if lot_size is None or lot_size <= 0:
                lot_size = self.config.LOT_SIZE
                logger.debug(f"Using default lot size: {lot_size}")
            
            if signal_type == 'BUY':
                price_diff = exit_price - entry_price
            else:
                price_diff = entry_price - exit_price
            
            pips = price_diff * self.config.XAUUSD_PIP_VALUE
            
            pl = pips * lot_size * 10
            
            logger.debug(f"P/L calculation: {signal_type} entry={entry_price:.2f}, "
                        f"exit={exit_price:.2f}, lot={lot_size:.2f}, "
                        f"pips={pips:.1f}, P/L=${pl:.2f}")
            
            return round(pl, 2)
            
        except ValueError as e:
            logger.error(f"Nilai tidak valid calculating P/L: {e}")
            return 0.0
        except ZeroDivisionError as e:
            logger.error(f"Division by zero calculating P/L: {e}")
            return 0.0
        except ArithmeticError as e:
            logger.error(f"Arithmetic error calculating P/L: {e}")
            return 0.0
        except AttributeError as e:
            logger.error(f"Config attribute error calculating P/L: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Error tidak terduga calculating P/L: {e}")
            return 0.0
    
    def calculate_risk_percentage(self, entry_price: float, stop_loss: float, 
                                  signal_type: str, lot_size: Optional[float] = None) -> float:
        """Calculate risk percentage based on account balance and SL distance
        
        Risk % = (SL_pips * lot_size * pip_value) / account_balance * 100
        """
        try:
            if lot_size is None or lot_size <= 0:
                lot_size = self.config.LOT_SIZE
            
            sl_distance = abs(entry_price - stop_loss)
            sl_pips = sl_distance * self.config.XAUUSD_PIP_VALUE
            
            if sl_pips <= 0:
                return 0.0
            
            pip_value_per_lot = 10.0
            risk_amount = sl_pips * lot_size * pip_value_per_lot
            
            account_balance = self.config.ACCOUNT_BALANCE
            if account_balance <= 0:
                return 0.0
            
            risk_percent = (risk_amount / account_balance) * 100
            
            risk_percent = max(0.0, min(risk_percent, 100.0))
            
            logger.debug(f"Risk calc: SL={sl_pips:.1f} pips, Lot={lot_size:.2f}, "
                        f"Risk=${risk_amount:.2f}, Balance=${account_balance:.2f}, "
                        f"Risk%={risk_percent:.2f}%")
            
            return round(risk_percent, 2)
            
        except Exception as e:
            logger.error(f"Error calculating risk percentage: {e}")
            return 0.0
