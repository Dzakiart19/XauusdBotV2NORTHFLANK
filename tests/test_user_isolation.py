import pytest
from datetime import datetime, timedelta
from bot.signal_event_store import SignalEventStore


@pytest.fixture
def signal_store():
    return SignalEventStore(ttl_seconds=3600, max_signals_per_user=100)


@pytest.fixture
def user_a_id():
    return 111111


@pytest.fixture
def user_b_id():
    return 222222


@pytest.fixture
def user_c_id():
    return 333333


def create_signal_data(signal_type: str, entry_price: float, timestamp: datetime | None = None):
    return {
        'signal_type': signal_type,
        'entry_price': entry_price,
        'stop_loss': entry_price - 10 if signal_type == 'BUY' else entry_price + 10,
        'take_profit': entry_price + 15 if signal_type == 'BUY' else entry_price - 15,
        'timestamp': timestamp or datetime.now(),
        'confidence': 0.85,
        'grade': 'A',
        'timeframe': 'M1'
    }


@pytest.mark.unit
class TestGetLatestSignalIsolation:
    
    async def test_get_latest_signal_returns_user_a_signal_only(self, signal_store, user_a_id, user_b_id):
        signal_a = create_signal_data('BUY', 2650.0)
        signal_b = create_signal_data('SELL', 2660.0)
        
        await signal_store.record_signal(user_a_id, signal_a)
        await signal_store.record_signal(user_b_id, signal_b)
        
        result = await signal_store.get_latest_signal(user_a_id)
        
        assert result is not None
        assert result['user_id'] == user_a_id
        assert result['signal_type'] == 'BUY'
        assert result['entry_price'] == 2650.0
    
    async def test_get_latest_signal_returns_user_b_signal_only(self, signal_store, user_a_id, user_b_id):
        signal_a = create_signal_data('BUY', 2650.0)
        signal_b = create_signal_data('SELL', 2660.0)
        
        await signal_store.record_signal(user_a_id, signal_a)
        await signal_store.record_signal(user_b_id, signal_b)
        
        result = await signal_store.get_latest_signal(user_b_id)
        
        assert result is not None
        assert result['user_id'] == user_b_id
        assert result['signal_type'] == 'SELL'
        assert result['entry_price'] == 2660.0
    
    async def test_get_latest_signal_returns_none_for_nonexistent_user(self, signal_store, user_a_id, user_b_id, user_c_id):
        signal_a = create_signal_data('BUY', 2650.0)
        signal_b = create_signal_data('SELL', 2660.0)
        
        await signal_store.record_signal(user_a_id, signal_a)
        await signal_store.record_signal(user_b_id, signal_b)
        
        result = await signal_store.get_latest_signal(user_c_id)
        
        assert result is None
    
    async def test_get_latest_signal_returns_most_recent_for_user(self, signal_store, user_a_id):
        signal_1 = create_signal_data('BUY', 2650.0, datetime.now() - timedelta(minutes=10))
        signal_2 = create_signal_data('SELL', 2670.0, datetime.now() - timedelta(minutes=5))
        signal_3 = create_signal_data('BUY', 2680.0, datetime.now())
        
        await signal_store.record_signal(user_a_id, signal_1)
        await signal_store.record_signal(user_a_id, signal_2)
        await signal_store.record_signal(user_a_id, signal_3)
        
        result = await signal_store.get_latest_signal(user_a_id)
        
        assert result is not None
        assert result['entry_price'] == 2680.0
        assert result['signal_type'] == 'BUY'


@pytest.mark.unit
class TestGetRecentSignalsIsolation:
    
    async def test_get_recent_signals_returns_only_user_a_signals(self, signal_store, user_a_id, user_b_id):
        for i in range(3):
            signal_a = create_signal_data('BUY', 2650.0 + i, datetime.now() - timedelta(minutes=10-i))
            await signal_store.record_signal(user_a_id, signal_a)
        
        for i in range(5):
            signal_b = create_signal_data('SELL', 2700.0 + i, datetime.now() - timedelta(minutes=10-i))
            await signal_store.record_signal(user_b_id, signal_b)
        
        result = await signal_store.get_recent_signals(user_a_id, 5)
        
        assert len(result) == 3
        for signal in result:
            assert signal['user_id'] == user_a_id
            assert signal['signal_type'] == 'BUY'
    
    async def test_get_recent_signals_returns_only_user_b_signals(self, signal_store, user_a_id, user_b_id):
        for i in range(3):
            signal_a = create_signal_data('BUY', 2650.0 + i, datetime.now() - timedelta(minutes=10-i))
            await signal_store.record_signal(user_a_id, signal_a)
        
        for i in range(5):
            signal_b = create_signal_data('SELL', 2700.0 + i, datetime.now() - timedelta(minutes=10-i))
            await signal_store.record_signal(user_b_id, signal_b)
        
        result = await signal_store.get_recent_signals(user_b_id, 10)
        
        assert len(result) == 5
        for signal in result:
            assert signal['user_id'] == user_b_id
            assert signal['signal_type'] == 'SELL'
    
    async def test_get_recent_signals_order_newest_to_oldest(self, signal_store, user_a_id):
        timestamps = [
            datetime.now() - timedelta(minutes=30),
            datetime.now() - timedelta(minutes=20),
            datetime.now() - timedelta(minutes=10),
            datetime.now() - timedelta(minutes=5),
            datetime.now()
        ]
        
        for i, ts in enumerate(timestamps):
            signal = create_signal_data('BUY', 2650.0 + i, ts)
            await signal_store.record_signal(user_a_id, signal)
        
        result = await signal_store.get_recent_signals(user_a_id, 5)
        
        assert len(result) == 5
        
        for i in range(len(result) - 1):
            current_ts = datetime.fromisoformat(result[i]['timestamp'])
            next_ts = datetime.fromisoformat(result[i + 1]['timestamp'])
            assert current_ts >= next_ts, "Signals should be ordered from newest to oldest"
    
    async def test_get_recent_signals_respects_limit(self, signal_store, user_a_id):
        for i in range(10):
            signal = create_signal_data('BUY', 2650.0 + i, datetime.now() - timedelta(minutes=10-i))
            await signal_store.record_signal(user_a_id, signal)
        
        result = await signal_store.get_recent_signals(user_a_id, 3)
        
        assert len(result) == 3
    
    async def test_get_recent_signals_returns_empty_for_nonexistent_user(self, signal_store, user_a_id, user_c_id):
        signal = create_signal_data('BUY', 2650.0)
        await signal_store.record_signal(user_a_id, signal)
        
        result = await signal_store.get_recent_signals(user_c_id, 5)
        
        assert result == []


@pytest.mark.unit
class TestClearUserSignalsIsolation:
    
    async def test_clear_user_signals_removes_only_target_user(self, signal_store, user_a_id, user_b_id):
        for i in range(3):
            signal_a = create_signal_data('BUY', 2650.0 + i)
            await signal_store.record_signal(user_a_id, signal_a)
        
        for i in range(5):
            signal_b = create_signal_data('SELL', 2700.0 + i)
            await signal_store.record_signal(user_b_id, signal_b)
        
        count_a_before = await signal_store.get_user_signal_count(user_a_id)
        count_b_before = await signal_store.get_user_signal_count(user_b_id)
        assert count_a_before == 3
        assert count_b_before == 5
        
        deleted_count = await signal_store.clear_user_signals(user_a_id)
        
        assert deleted_count == 3
        
        count_a_after = await signal_store.get_user_signal_count(user_a_id)
        count_b_after = await signal_store.get_user_signal_count(user_b_id)
        
        assert count_a_after == 0
        assert count_b_after == 5
    
    async def test_clear_user_signals_user_a_signals_empty(self, signal_store, user_a_id, user_b_id):
        for i in range(3):
            signal_a = create_signal_data('BUY', 2650.0 + i)
            await signal_store.record_signal(user_a_id, signal_a)
        
        for i in range(2):
            signal_b = create_signal_data('SELL', 2700.0 + i)
            await signal_store.record_signal(user_b_id, signal_b)
        
        await signal_store.clear_user_signals(user_a_id)
        
        user_a_signals = await signal_store.get_recent_signals(user_a_id, 10)
        user_a_latest = await signal_store.get_latest_signal(user_a_id)
        
        assert user_a_signals == []
        assert user_a_latest is None
    
    async def test_clear_user_signals_user_b_signals_remain(self, signal_store, user_a_id, user_b_id):
        for i in range(3):
            signal_a = create_signal_data('BUY', 2650.0 + i)
            await signal_store.record_signal(user_a_id, signal_a)
        
        for i in range(4):
            signal_b = create_signal_data('SELL', 2700.0 + i)
            await signal_store.record_signal(user_b_id, signal_b)
        
        await signal_store.clear_user_signals(user_a_id)
        
        user_b_signals = await signal_store.get_recent_signals(user_b_id, 10)
        user_b_latest = await signal_store.get_latest_signal(user_b_id)
        
        assert len(user_b_signals) == 4
        assert user_b_latest is not None
        assert user_b_latest['user_id'] == user_b_id
    
    async def test_clear_user_signals_returns_zero_for_nonexistent_user(self, signal_store, user_c_id):
        deleted_count = await signal_store.clear_user_signals(user_c_id)
        
        assert deleted_count == 0


@pytest.mark.unit
class TestGetUserSignalCountIsolation:
    
    async def test_signal_count_user_a_has_3_signals(self, signal_store, user_a_id, user_b_id):
        for i in range(3):
            signal_a = create_signal_data('BUY', 2650.0 + i)
            await signal_store.record_signal(user_a_id, signal_a)
        
        for i in range(5):
            signal_b = create_signal_data('SELL', 2700.0 + i)
            await signal_store.record_signal(user_b_id, signal_b)
        
        count_a = await signal_store.get_user_signal_count(user_a_id)
        
        assert count_a == 3
    
    async def test_signal_count_user_b_has_5_signals(self, signal_store, user_a_id, user_b_id):
        for i in range(3):
            signal_a = create_signal_data('BUY', 2650.0 + i)
            await signal_store.record_signal(user_a_id, signal_a)
        
        for i in range(5):
            signal_b = create_signal_data('SELL', 2700.0 + i)
            await signal_store.record_signal(user_b_id, signal_b)
        
        count_b = await signal_store.get_user_signal_count(user_b_id)
        
        assert count_b == 5
    
    async def test_signal_count_nonexistent_user_returns_zero(self, signal_store, user_a_id, user_c_id):
        for i in range(3):
            signal = create_signal_data('BUY', 2650.0 + i)
            await signal_store.record_signal(user_a_id, signal)
        
        count_c = await signal_store.get_user_signal_count(user_c_id)
        
        assert count_c == 0
    
    async def test_signal_count_increments_correctly(self, signal_store, user_a_id):
        assert await signal_store.get_user_signal_count(user_a_id) == 0
        
        signal_1 = create_signal_data('BUY', 2650.0)
        await signal_store.record_signal(user_a_id, signal_1)
        assert await signal_store.get_user_signal_count(user_a_id) == 1
        
        signal_2 = create_signal_data('SELL', 2660.0)
        await signal_store.record_signal(user_a_id, signal_2)
        assert await signal_store.get_user_signal_count(user_a_id) == 2
        
        signal_3 = create_signal_data('BUY', 2670.0)
        await signal_store.record_signal(user_a_id, signal_3)
        assert await signal_store.get_user_signal_count(user_a_id) == 3
    
    async def test_signal_count_independent_per_user(self, signal_store, user_a_id, user_b_id, user_c_id):
        for i in range(2):
            signal = create_signal_data('BUY', 2650.0 + i)
            await signal_store.record_signal(user_a_id, signal)
        
        for i in range(7):
            signal = create_signal_data('SELL', 2700.0 + i)
            await signal_store.record_signal(user_b_id, signal)
        
        count_a = await signal_store.get_user_signal_count(user_a_id)
        count_b = await signal_store.get_user_signal_count(user_b_id)
        count_c = await signal_store.get_user_signal_count(user_c_id)
        
        assert count_a == 2
        assert count_b == 7
        assert count_c == 0


@pytest.mark.unit
class TestMultiUserScenarios:
    
    async def test_multiple_users_complete_isolation(self, signal_store):
        users = [100001, 100002, 100003, 100004, 100005]
        
        for idx, user_id in enumerate(users):
            for i in range(idx + 1):
                signal = create_signal_data('BUY' if i % 2 == 0 else 'SELL', 2650.0 + user_id + i)
                await signal_store.record_signal(user_id, signal)
        
        for idx, user_id in enumerate(users):
            count = await signal_store.get_user_signal_count(user_id)
            assert count == idx + 1
            
            signals = await signal_store.get_recent_signals(user_id, 10)
            for signal in signals:
                assert signal['user_id'] == user_id
    
    async def test_clear_one_user_does_not_affect_others(self, signal_store):
        users = [200001, 200002, 200003]
        
        for user_id in users:
            for i in range(3):
                signal = create_signal_data('BUY', 2650.0 + i)
                await signal_store.record_signal(user_id, signal)
        
        await signal_store.clear_user_signals(users[1])
        
        assert await signal_store.get_user_signal_count(users[0]) == 3
        assert await signal_store.get_user_signal_count(users[1]) == 0
        assert await signal_store.get_user_signal_count(users[2]) == 3
