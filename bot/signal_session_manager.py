"""
Signal Session Manager
Mengelola sesi sinyal untuk mencegah duplikasi dan konflik antara mode auto & manual
"""
import asyncio
import weakref
from datetime import datetime
from typing import Optional, Dict, Callable, List
from dataclasses import dataclass, field
import pytz
from bot.logger import setup_logger
from config import Config

logger = setup_logger('SignalSessionManager')

class SessionError(Exception):
    """Exception untuk error pada signal session management"""
    pass

@dataclass
class SignalSession:
    """Representasi sesi sinyal aktif"""
    user_id: int
    signal_id: str
    signal_source: str
    signal_type: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_id: Optional[int]
    trade_id: Optional[int]
    started_at: datetime
    chart_path: Optional[str] = None
    photo_sent: bool = False

class SignalSessionManager:
    """
    Manager untuk mengelola sesi sinyal secara global
    Mencegah sinyal duplikat dan konflik antara auto/manual mode
    
    Thread-safe dengan proper locking untuk mencegah race conditions
    """
    
    def __init__(self):
        self.active_sessions: Dict[int, SignalSession] = {}
        self._last_signal_info: Dict[int, dict] = {}
        self._session_lock = asyncio.Lock()
        self._event_lock = asyncio.Lock()
        self._event_handlers: Dict[str, List[Callable]] = {
            'on_session_start': [],
            'on_session_end': [],
            'on_session_update': []
        }
        self._pending_events: List[tuple] = []
        logger.info("Signal Session Manager diinisialisasi dengan proteksi spam sinyal")
    
    def register_event_handler(self, event: str, handler: Callable):
        """Daftarkan event handler untuk lifecycle events"""
        if event in self._event_handlers:
            self._event_handlers[event].append(handler)
            logger.debug(f"Event handler terdaftar untuk: {event}")
    
    async def _emit_event_outside_lock(self, event: str, session: SignalSession):
        """
        Emit event ke semua registered handlers.
        PENTING: Method ini harus dipanggil DILUAR session_lock untuk mencegah deadlock.
        """
        handlers = []
        async with self._event_lock:
            handlers = list(self._event_handlers.get(event, []))
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(session)
                else:
                    handler(session)
            except (SessionError, Exception) as e:
                logger.error(f"Error pada event handler untuk {event}: {e}")
    
    def _get_last_signal_info(self, user_id: int) -> Optional[dict]:
        """Ambil info sinyal terakhir untuk user"""
        return self._last_signal_info.get(user_id)
    
    def _update_last_signal_info(self, user_id: int, signal_type: str, entry_price: float):
        """Update info sinyal terakhir untuk user"""
        self._last_signal_info[user_id] = {
            'signal_type': signal_type,
            'entry_price': entry_price,
            'timestamp': datetime.now(pytz.UTC)
        }
        logger.debug(f"Info sinyal terakhir diperbarui - User:{user_id} Tipe:{signal_type} Harga:{entry_price}")
    
    async def can_create_signal(self, user_id: int, signal_source: str, 
                                signal_type: Optional[str] = None,
                                current_price: Optional[float] = None,
                                position_tracker=None) -> tuple[bool, Optional[str]]:
        """
        Cek apakah user bisa membuat sinyal baru dengan proteksi spam
        
        Args:
            user_id: ID user
            signal_source: Sumber sinyal (auto/manual)
            signal_type: Tipe sinyal (BUY/SELL)
            current_price: Harga saat ini untuk cek pergerakan minimum
            position_tracker: PositionTracker untuk validasi posisi aktif
        
        Returns:
            (can_create, rejection_reason)
        """
        async with self._session_lock:
            active_session = self.active_sessions.get(user_id)
            last_info = self._last_signal_info.get(user_id)
            
            if active_session and not Config.AUTO_SIGNAL_REPLACEMENT_ALLOWED:
                has_real_position = True
                
                if position_tracker is not None:
                    try:
                        has_real_position = await position_tracker.has_active_position_verified(user_id)
                    except Exception as e:
                        logger.warning(f"Error verifying position for user {user_id}: {e}")
                        has_real_position = True
                
                if not has_real_position:
                    logger.info(
                        f"🧹 Membersihkan sesi stale untuk user {user_id} - "
                        f"sesi ada tapi tidak ada posisi aktif di database"
                    )
                    del self.active_sessions[user_id]
                    active_session = None
                else:
                    elapsed = (datetime.now(pytz.UTC) - active_session.started_at).total_seconds()
                    reason = (
                        f"Sudah ada posisi aktif ({active_session.signal_type}). "
                        f"Tutup posisi terlebih dahulu sebelum membuat sinyal baru. "
                        f"Posisi aktif sudah berjalan {elapsed:.0f} detik."
                    )
                    logger.info(
                        f"🚫 Sinyal ditolak (posisi aktif) - User:{user_id} "
                        f"Posisi aktif:{active_session.signal_type} Berjalan:{elapsed:.0f}s | "
                        f"AUTO_SIGNAL_REPLACEMENT_ALLOWED=false"
                    )
                    return False, reason
            
            if active_session and signal_type:
                if active_session.signal_type == signal_type:
                    elapsed = (datetime.now(pytz.UTC) - active_session.started_at).total_seconds()
                    cooldown = Config.TICK_COOLDOWN_FOR_SAME_SIGNAL
                    
                    if elapsed < cooldown:
                        remaining = cooldown - elapsed
                        reason = (
                            f"Sinyal {signal_type} yang sama masih aktif. "
                            f"Tunggu {remaining:.0f} detik lagi sebelum mengirim sinyal serupa."
                        )
                        logger.info(
                            f"🚫 Sinyal ditolak (cooldown) - User:{user_id} Tipe:{signal_type} "
                            f"Sisa:{remaining:.0f}s dari {cooldown}s"
                        )
                        if signal_type and current_price is not None:
                            self._update_last_signal_info(user_id, signal_type, current_price)
                            logger.info(
                                f"📝 Signal tracking diperbarui (ditolak): {signal_type} @ ${current_price:.2f} | "
                                f"Alasan: Cooldown sesi aktif"
                            )
                        return False, reason
            
            if last_info and signal_type and current_price is not None:
                if last_info['signal_type'] == signal_type:
                    time_since_last = (datetime.now(pytz.UTC) - last_info['timestamp']).total_seconds()
                    cooldown = Config.TICK_COOLDOWN_FOR_SAME_SIGNAL
                    min_movement = Config.SIGNAL_MINIMUM_PRICE_MOVEMENT
                    
                    if cooldown <= 0 and min_movement <= 0.1:
                        logger.info(
                            f"✅ SCALPING MODE: Sinyal unlimited diizinkan - User:{user_id} "
                            f"Tipe:{signal_type} (cooldown=0, min_movement rendah)"
                        )
                    elif cooldown > 0 and time_since_last < cooldown:
                        last_price = last_info['entry_price']
                        price_movement = abs(current_price - last_price)
                        
                        if min_movement > 0 and price_movement < min_movement:
                            remaining = cooldown - time_since_last
                            reason = (
                                f"Pergerakan harga belum cukup ({price_movement:.2f} < {min_movement:.2f}). "
                                f"Tunggu {remaining:.0f}s atau pergerakan harga {min_movement:.2f}+ sebelum sinyal {signal_type} baru."
                            )
                            logger.info(
                                f"🚫 Sinyal ditolak (pergerakan harga minimal) - User:{user_id} "
                                f"Tipe:{signal_type} Pergerakan:{price_movement:.2f} < Min:{min_movement:.2f}"
                            )
                            self._update_last_signal_info(user_id, signal_type, current_price)
                            return False, reason
        
        return True, None
    
    async def create_session(self, user_id: int, signal_id: str, signal_source: str,
                            signal_type: str, entry_price: float, stop_loss: float,
                            take_profit: float) -> Optional[SignalSession]:
        """
        Buat sesi sinyal baru dengan proteksi overwrite
        
        Returns:
            SignalSession jika berhasil, None jika ditolak
        """
        async with self._session_lock:
            if user_id in self.active_sessions:
                old_session = self.active_sessions[user_id]
                elapsed = (datetime.now(pytz.UTC) - old_session.started_at).total_seconds()
                
                if not Config.AUTO_SIGNAL_REPLACEMENT_ALLOWED:
                    logger.warning(
                        f"⛔ Sesi TIDAK di-overwrite - User:{user_id} "
                        f"Posisi aktif:{old_session.signal_type} -> Baru:{signal_type} | "
                        f"AUTO_SIGNAL_REPLACEMENT_ALLOWED=false. "
                        f"Sesi aktif sudah berjalan {elapsed:.1f}s"
                    )
                    return None
                else:
                    logger.info(
                        f"🔄 Sesi akan di-overwrite (diizinkan) - User:{user_id} "
                        f"Lama:{old_session.signal_type} -> Baru:{signal_type} | "
                        f"AUTO_SIGNAL_REPLACEMENT_ALLOWED=true | Durasi sebelumnya:{elapsed:.1f}s"
                    )
            
            session = SignalSession(
                user_id=user_id,
                signal_id=signal_id,
                signal_source=signal_source,
                signal_type=signal_type,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_id=None,
                trade_id=None,
                started_at=datetime.now(pytz.UTC)
            )
            
            self.active_sessions[user_id] = session
            
            self._last_signal_info[user_id] = {
                'signal_type': signal_type,
                'entry_price': entry_price,
                'timestamp': datetime.now(pytz.UTC)
            }
            
            icon = "🤖" if signal_source == "auto" else "👤"
            logger.info(
                f"✅ Sesi sinyal dibuat - User:{user_id} {icon} {signal_source.upper()} "
                f"Tipe:{signal_type} Harga:{entry_price}"
            )
            
        await self._emit_event_outside_lock('on_session_start', session)
        
        return session
    
    async def update_session(self, user_id: int, **kwargs):
        """Update sesi yang sedang aktif"""
        async with self._session_lock:
            if user_id not in self.active_sessions:
                logger.warning(f"Mencoba update sesi yang tidak ada untuk user {user_id}")
                return False
            
            session = self.active_sessions[user_id]
            
            for key, value in kwargs.items():
                if hasattr(session, key):
                    setattr(session, key, value)
            
        await self._emit_event_outside_lock('on_session_update', session)
        
        return True
    
    async def end_session(self, user_id: int, reason: str = "closed"):
        """Akhiri sesi sinyal dan cleanup chart jika ada - thread safe"""
        session = None
        chart_path = None
        
        async with self._session_lock:
            if user_id not in self.active_sessions:
                logger.debug(f"Tidak ada sesi aktif untuk diakhiri - User:{user_id}")
                return None
            
            session = self.active_sessions[user_id]
            chart_path = session.chart_path
            
            del self.active_sessions[user_id]
            
            duration = (datetime.now(pytz.UTC) - session.started_at).total_seconds()
            icon = "🤖" if session.signal_source == "auto" else "👤"
            
            logger.info(
                f"🏁 Sesi sinyal diakhiri - User:{user_id} {icon} {session.signal_source.upper()} "
                f"Alasan:{reason} Durasi:{duration:.1f}s"
            )
        
        if chart_path:
            await self._cleanup_chart_file(chart_path)
        
        await self._emit_event_outside_lock('on_session_end', session)
        
        return session
    
    async def _cleanup_chart_file(self, chart_path: str):
        """Cleanup chart file secara async dan atomic"""
        try:
            import os
            if chart_path and os.path.exists(chart_path):
                os.remove(chart_path)
                logger.info(f"🗑️ Chart sesi dibersihkan: {chart_path}")
                
                if os.path.exists(chart_path):
                    logger.warning(f"File chart masih ada setelah dihapus: {chart_path}")
        except FileNotFoundError:
            logger.debug(f"File chart sudah dihapus: {chart_path}")
        except (SessionError, Exception) as e:
            logger.warning(f"Gagal membersihkan chart {chart_path}: {e}")
    
    async def clear_all_sessions(self, reason: str = "system_reset") -> int:
        """
        Hapus semua sesi sinyal aktif (untuk system reset) - thread safe
        
        Returns:
            int: Jumlah sesi yang dibersihkan
        """
        sessions_to_end = []
        chart_paths = []
        
        async with self._session_lock:
            session_count = len(self.active_sessions)
            
            if session_count == 0:
                logger.info("Tidak ada sesi aktif untuk dibersihkan")
                return 0
            
            logger.info(f"Membersihkan semua {session_count} sesi sinyal aktif...")
            
            sessions_to_end = list(self.active_sessions.values())
            chart_paths = [s.chart_path for s in sessions_to_end if s.chart_path]
            
            for session in sessions_to_end:
                duration = (datetime.now(pytz.UTC) - session.started_at).total_seconds()
                icon = "🤖" if session.signal_source == "auto" else "👤"
                
                logger.info(
                    f"🏁 Membersihkan sesi - User:{session.user_id} {icon} {session.signal_source.upper()} "
                    f"Tipe:{session.signal_type} Alasan:{reason} Durasi:{duration:.1f}s"
                )
            
            self.active_sessions.clear()
            self._last_signal_info.clear()
        
        for chart_path in chart_paths:
            await self._cleanup_chart_file(chart_path)
        
        for session in sessions_to_end:
            try:
                await self._emit_event_outside_lock('on_session_end', session)
            except (SessionError, Exception) as e:
                logger.error(f"Error saat emit event end untuk sesi {session.signal_id}: {e}")
        
        logger.info(f"✅ Semua {session_count} sesi sinyal berhasil dibersihkan")
        
        return session_count
    
    def get_active_session(self, user_id: int) -> Optional[SignalSession]:
        """Ambil sesi aktif untuk user (thread-safe snapshot)"""
        return self.active_sessions.get(user_id)
    
    async def get_active_session_async(self, user_id: int) -> Optional[SignalSession]:
        """Ambil sesi aktif untuk user dengan thread-safe locking"""
        async with self._session_lock:
            return self.active_sessions.get(user_id)
    
    def get_session(self, user_id: int) -> Optional[SignalSession]:
        """Alias untuk get_active_session untuk kompatibilitas"""
        return self.get_active_session(user_id)
    
    def has_active_session(self, user_id: int) -> bool:
        """Cek apakah user punya sesi aktif (thread-safe snapshot)"""
        return user_id in self.active_sessions
    
    async def has_active_session_async(self, user_id: int) -> bool:
        """Cek apakah user punya sesi aktif dengan thread-safe locking"""
        async with self._session_lock:
            return user_id in self.active_sessions
    
    def get_last_signal_info(self, user_id: int) -> Optional[dict]:
        """Ambil info sinyal terakhir untuk user (public method)"""
        return self._last_signal_info.get(user_id)
    
    def get_all_active_sessions(self) -> Dict[int, SignalSession]:
        """Ambil semua sesi yang aktif (thread-safe snapshot)"""
        return self.active_sessions.copy()
    
    async def get_all_active_sessions_async(self) -> Dict[int, SignalSession]:
        """Ambil semua sesi yang aktif dengan thread-safe locking"""
        async with self._session_lock:
            return self.active_sessions.copy()
    
    def get_session_count(self) -> int:
        """Hitung jumlah sesi aktif (thread-safe snapshot)"""
        return len(self.active_sessions)
    
    async def get_session_count_async(self) -> int:
        """Hitung jumlah sesi aktif dengan thread-safe locking"""
        async with self._session_lock:
            return len(self.active_sessions)
    
    def get_stats(self) -> dict:
        """Dapatkan statistik sesi"""
        auto_count = sum(1 for s in self.active_sessions.values() if s.signal_source == 'auto')
        manual_count = sum(1 for s in self.active_sessions.values() if s.signal_source == 'manual')
        
        return {
            'total_sessions': len(self.active_sessions),
            'auto_sessions': auto_count,
            'manual_sessions': manual_count,
            'tracked_last_signals': len(self._last_signal_info)
        }
