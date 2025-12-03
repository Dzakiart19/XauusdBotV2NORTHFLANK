"""
Signal Event Store - Modul untuk menyinkronkan sinyal antara Telegram Bot dan WebApp Dashboard

Menyediakan:
- Penyimpanan sinyal per user dengan TTL (auto-cleanup)
- Thread-safe dengan asyncio.Lock
- Akses sinyal terbaru untuk dashboard
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from bot.logger import setup_logger, mask_user_id

logger = setup_logger('SignalEventStore')


@dataclass
class SignalRecord:
    """Record sinyal dengan metadata"""
    user_id: int
    signal_type: str
    entry_price: float
    stop_loss: float
    take_profit: float
    timestamp: datetime
    trade_id: Optional[int] = None
    position_id: Optional[int] = None
    confidence: float = 0.0
    grade: str = 'B'
    timeframe: str = 'M1'
    extra_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Konversi ke dictionary untuk API response"""
        return {
            'user_id': self.user_id,
            'signal_type': self.signal_type,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'trade_id': self.trade_id,
            'position_id': self.position_id,
            'confidence': self.confidence,
            'grade': self.grade,
            'timeframe': self.timeframe,
            **self.extra_data
        }


class SignalEventStore:
    """
    Event Store untuk menyimpan dan mengakses sinyal trading.
    
    Fitur:
    - Penyimpanan sinyal per user dengan TTL (Time To Live)
    - Thread-safe dengan asyncio.Lock
    - Auto-cleanup sinyal expired
    - Maksimal 100 sinyal per user
    """
    
    DEFAULT_TTL_SECONDS = 3600  # 1 jam default TTL
    MAX_SIGNALS_PER_USER = 100
    CLEANUP_INTERVAL_SECONDS = 300  # 5 menit
    
    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS, max_signals_per_user: int = MAX_SIGNALS_PER_USER):
        self._signals: Dict[int, List[SignalRecord]] = {}
        self._global_latest: Optional[SignalRecord] = None
        self._lock = asyncio.Lock()
        self._ttl_seconds = ttl_seconds
        self._max_signals_per_user = max_signals_per_user
        self._last_cleanup: datetime = datetime.now()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._is_running: bool = False
        
        self._telemetry = {
            'total_signals_recorded': 0,
            'signals_cleaned_up': 0,
            'last_signal_time': None,
            'active_users': 0
        }
        
        logger.info(f"✅ SignalEventStore diinisialisasi - TTL: {ttl_seconds}s, Max sinyal per user: {max_signals_per_user}")
    
    async def start(self):
        """Mulai background cleanup task"""
        if self._is_running:
            return
        
        self._is_running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("🔄 SignalEventStore cleanup task dimulai")
    
    async def stop(self):
        """Hentikan background cleanup task"""
        self._is_running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        logger.info("🛑 SignalEventStore cleanup task dihentikan")
    
    async def _cleanup_loop(self):
        """Background loop untuk auto-cleanup sinyal expired"""
        while self._is_running:
            try:
                await asyncio.sleep(self.CLEANUP_INTERVAL_SECONDS)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error dalam cleanup loop: {e}")
    
    async def _cleanup_expired(self):
        """Hapus sinyal yang sudah expired berdasarkan TTL"""
        async with self._lock:
            now = datetime.now()
            cutoff = now - timedelta(seconds=self._ttl_seconds)
            cleaned_count = 0
            
            for user_id in list(self._signals.keys()):
                original_count = len(self._signals[user_id])
                self._signals[user_id] = [
                    s for s in self._signals[user_id] 
                    if s.timestamp > cutoff
                ]
                cleaned_count += original_count - len(self._signals[user_id])
                
                if not self._signals[user_id]:
                    del self._signals[user_id]
            
            if cleaned_count > 0:
                self._telemetry['signals_cleaned_up'] += cleaned_count
                logger.info(f"🧹 Auto-cleanup: {cleaned_count} sinyal expired dihapus")
            
            self._telemetry['active_users'] = len(self._signals)
            self._last_cleanup = now
    
    async def record_signal(self, user_id: int, signal_data: Dict[str, Any]) -> bool:
        """
        Simpan sinyal baru untuk user.
        
        Args:
            user_id: ID user Telegram
            signal_data: Dictionary berisi data sinyal:
                - signal_type: 'BUY' atau 'SELL'
                - entry_price: Harga entry
                - stop_loss: Harga stop loss
                - take_profit: Harga take profit
                - timestamp: Waktu sinyal (opsional, default: now)
                - trade_id: ID trade (opsional)
                - position_id: ID posisi (opsional)
                - confidence: Skor kepercayaan (opsional)
                - grade: Grade sinyal A/B/C (opsional)
                - timeframe: Timeframe (opsional)
        
        Returns:
            bool: True jika berhasil disimpan
        """
        try:
            async with self._lock:
                timestamp = signal_data.get('timestamp')
                if timestamp is None:
                    timestamp = datetime.now()
                elif isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                
                extra_data = {k: v for k, v in signal_data.items() 
                             if k not in ['user_id', 'signal_type', 'entry_price', 'stop_loss', 
                                         'take_profit', 'timestamp', 'trade_id', 'position_id',
                                         'confidence', 'grade', 'timeframe']}
                
                def safe_float(val, default=0.0):
                    """Konversi aman ke float dengan handling None"""
                    if val is None:
                        return default
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return default
                
                record = SignalRecord(
                    user_id=user_id,
                    signal_type=signal_data.get('signal_type', 'UNKNOWN'),
                    entry_price=safe_float(signal_data.get('entry_price'), 0),
                    stop_loss=safe_float(signal_data.get('stop_loss'), 0),
                    take_profit=safe_float(signal_data.get('take_profit'), 0),
                    timestamp=timestamp,
                    trade_id=signal_data.get('trade_id'),
                    position_id=signal_data.get('position_id'),
                    confidence=safe_float(signal_data.get('confidence'), 0),
                    grade=signal_data.get('grade', 'B'),
                    timeframe=signal_data.get('timeframe', 'M1'),
                    extra_data=extra_data
                )
                
                if user_id not in self._signals:
                    self._signals[user_id] = []
                
                self._signals[user_id].append(record)
                
                if len(self._signals[user_id]) > self._max_signals_per_user:
                    removed_count = len(self._signals[user_id]) - self._max_signals_per_user
                    self._signals[user_id] = self._signals[user_id][-self._max_signals_per_user:]
                    logger.debug(f"Menghapus {removed_count} sinyal lama untuk user {mask_user_id(user_id)} (melebihi limit)")
                
                self._global_latest = record
                
                self._telemetry['total_signals_recorded'] += 1
                self._telemetry['last_signal_time'] = timestamp
                self._telemetry['active_users'] = len(self._signals)
                
                logger.info(f"📝 Sinyal direkam untuk user {mask_user_id(user_id)}: {record.signal_type} @${record.entry_price:.2f}")
                return True
                
        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"Error merekam sinyal untuk user {mask_user_id(user_id)}: {e}")
            return False
    
    async def get_latest_signal(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Ambil sinyal terbaru untuk user tertentu.
        
        Args:
            user_id: ID user Telegram
            
        Returns:
            Dictionary sinyal terbaru atau None jika tidak ada
        """
        async with self._lock:
            if user_id not in self._signals or not self._signals[user_id]:
                return None
            
            latest = self._signals[user_id][-1]
            return latest.to_dict()
    
    async def get_recent_signals(self, user_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Ambil beberapa sinyal terakhir untuk user tertentu.
        
        Args:
            user_id: ID user Telegram
            limit: Jumlah maksimal sinyal yang diambil (default: 5)
            
        Returns:
            List dictionary sinyal, terurut dari terbaru ke terlama
        """
        async with self._lock:
            if user_id not in self._signals or not self._signals[user_id]:
                return []
            
            signals = self._signals[user_id][-limit:]
            return [s.to_dict() for s in reversed(signals)]
    
    async def get_global_latest_signal(self) -> Optional[Dict[str, Any]]:
        """
        Ambil sinyal terbaru dari semua user.
        
        Returns:
            Dictionary sinyal terbaru global atau None jika tidak ada
        """
        async with self._lock:
            if self._global_latest is None:
                if not self._signals:
                    return None
                
                latest = None
                for user_signals in self._signals.values():
                    if user_signals:
                        last_signal = user_signals[-1]
                        if latest is None or last_signal.timestamp > latest.timestamp:
                            latest = last_signal
                
                if latest:
                    self._global_latest = latest
                    return latest.to_dict()
                return None
            
            return self._global_latest.to_dict()
    
    async def get_all_recent_signals(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Ambil sinyal terbaru dari semua user, diurutkan berdasarkan waktu.
        
        Args:
            limit: Jumlah maksimal sinyal yang diambil (default: 10)
            
        Returns:
            List dictionary sinyal dari semua user, terurut dari terbaru
        """
        async with self._lock:
            all_signals = []
            for user_signals in self._signals.values():
                all_signals.extend(user_signals)
            
            sorted_signals = sorted(all_signals, key=lambda s: s.timestamp, reverse=True)
            return [s.to_dict() for s in sorted_signals[:limit]]
    
    async def get_user_signal_count(self, user_id: int) -> int:
        """
        Hitung jumlah sinyal yang tersimpan untuk user tertentu.
        
        Args:
            user_id: ID user Telegram
            
        Returns:
            Jumlah sinyal tersimpan
        """
        async with self._lock:
            if user_id not in self._signals:
                return 0
            return len(self._signals[user_id])
    
    async def clear_user_signals(self, user_id: int) -> int:
        """
        Hapus semua sinyal untuk user tertentu.
        
        Args:
            user_id: ID user Telegram
            
        Returns:
            Jumlah sinyal yang dihapus
        """
        async with self._lock:
            if user_id not in self._signals:
                return 0
            
            count = len(self._signals[user_id])
            del self._signals[user_id]
            
            if self._global_latest and self._global_latest.user_id == user_id:
                self._global_latest = None
            
            self._telemetry['active_users'] = len(self._signals)
            logger.info(f"🧹 Menghapus {count} sinyal untuk user {mask_user_id(user_id)}")
            return count
    
    def get_telemetry(self) -> Dict[str, Any]:
        """
        Ambil statistik telemetry store.
        
        Returns:
            Dictionary dengan statistik telemetry
        """
        return {
            **self._telemetry,
            'ttl_seconds': self._ttl_seconds,
            'max_signals_per_user': self._max_signals_per_user,
            'last_cleanup': self._last_cleanup.isoformat() if self._last_cleanup else None,
            'is_running': self._is_running
        }
    
    def get_latest_signal_sync(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Versi synchronous dari get_latest_signal untuk penggunaan non-async.
        PERINGATAN: Tidak thread-safe, gunakan hanya jika tidak ada akses concurrent.
        
        Args:
            user_id: ID user Telegram
            
        Returns:
            Dictionary sinyal terbaru atau None jika tidak ada
        """
        if user_id not in self._signals or not self._signals[user_id]:
            return None
        
        latest = self._signals[user_id][-1]
        return latest.to_dict()
    
    def get_global_latest_signal_sync(self) -> Optional[Dict[str, Any]]:
        """
        Versi synchronous dari get_global_latest_signal untuk penggunaan non-async.
        PERINGATAN: Tidak thread-safe, gunakan hanya jika tidak ada akses concurrent.
        
        Returns:
            Dictionary sinyal terbaru global atau None jika tidak ada
        """
        if self._global_latest is None:
            if not self._signals:
                return None
            
            latest = None
            for user_signals in self._signals.values():
                if user_signals:
                    last_signal = user_signals[-1]
                    if latest is None or last_signal.timestamp > latest.timestamp:
                        latest = last_signal
            
            if latest:
                return latest.to_dict()
            return None
        
        return self._global_latest.to_dict()


signal_event_store_instance: Optional[SignalEventStore] = None


def get_signal_event_store() -> Optional[SignalEventStore]:
    """Ambil instance global SignalEventStore"""
    return signal_event_store_instance


def create_signal_event_store(ttl_seconds: int = SignalEventStore.DEFAULT_TTL_SECONDS,
                               max_signals_per_user: int = SignalEventStore.MAX_SIGNALS_PER_USER) -> SignalEventStore:
    """Buat dan set instance global SignalEventStore"""
    global signal_event_store_instance
    signal_event_store_instance = SignalEventStore(ttl_seconds, max_signals_per_user)
    return signal_event_store_instance
