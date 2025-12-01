import pytest
from datetime import datetime, timedelta
import pytz
from unittest.mock import Mock, patch
from bot.risk_manager import RiskManager, DynamicRiskCalculator, ExposureRecord, PartialExitLevel
from bot.database import Trade, Position


@pytest.mark.unit
class TestDynamicRiskCalculator:
    """Tests for DynamicRiskCalculator class"""
    
    @pytest.fixture
    def dynamic_risk_config(self, mock_config):
        """Config with dynamic risk settings"""
        mock_config.MAX_DAILY_LOSS_AMOUNT = 10.0
        mock_config.MAX_DAILY_LOSS_PERCENT = 1.0
        mock_config.MAX_CONCURRENT_POSITIONS = 4
        mock_config.RISK_SAFETY_FACTOR = 0.5
        mock_config.ACCOUNT_BALANCE = 1000.0
        mock_config.XAUUSD_PIP_VALUE = 10.0
        mock_config.LOT_SIZE = 0.01
        return mock_config
    
    def test_initialization(self, dynamic_risk_config, mock_db_manager):
        """Test DynamicRiskCalculator initialization"""
        calc = DynamicRiskCalculator(dynamic_risk_config, mock_db_manager)
        
        assert calc.max_daily_loss == 10.0
        assert calc.max_concurrent_positions == 4
        assert calc.risk_safety_factor == 0.5
        assert calc.effective_daily_threshold == 10.0
    
    def test_initialization_with_percent_threshold(self, dynamic_risk_config, mock_db_manager):
        """Test threshold uses minimum of amount and percent"""
        dynamic_risk_config.MAX_DAILY_LOSS_AMOUNT = 50.0
        dynamic_risk_config.MAX_DAILY_LOSS_PERCENT = 0.5
        dynamic_risk_config.ACCOUNT_BALANCE = 1000.0
        
        calc = DynamicRiskCalculator(dynamic_risk_config, mock_db_manager)
        
        assert calc.effective_daily_threshold == 5.0
    
    def test_calculate_dynamic_lot_basic(self, dynamic_risk_config, mock_db_manager):
        """Test basic lot size calculation"""
        calc = DynamicRiskCalculator(dynamic_risk_config, mock_db_manager)
        
        sl_pips = 10.0
        account_balance = 1000.0
        
        lot_size = calc.calculate_dynamic_lot(sl_pips, account_balance)
        
        assert isinstance(lot_size, float)
        assert 0.01 <= lot_size <= 0.1
    
    def test_calculate_dynamic_lot_min_bound(self, dynamic_risk_config, mock_db_manager):
        """Test lot size respects minimum bound"""
        calc = DynamicRiskCalculator(dynamic_risk_config, mock_db_manager)
        
        lot_size = calc.calculate_dynamic_lot(sl_pips=100.0, account_balance=10.0)
        
        assert lot_size == 0.01
    
    def test_calculate_dynamic_lot_max_bound(self, dynamic_risk_config, mock_db_manager):
        """Test lot size respects maximum bound"""
        calc = DynamicRiskCalculator(dynamic_risk_config, mock_db_manager)
        
        lot_size = calc.calculate_dynamic_lot(sl_pips=1.0, account_balance=100000.0)
        
        assert lot_size == 0.1
    
    def test_calculate_dynamic_lot_invalid_inputs(self, dynamic_risk_config, mock_db_manager):
        """Test lot calculation with invalid inputs"""
        calc = DynamicRiskCalculator(dynamic_risk_config, mock_db_manager)
        
        assert calc.calculate_dynamic_lot(sl_pips=0, account_balance=1000.0) == 0.01
        assert calc.calculate_dynamic_lot(sl_pips=-5, account_balance=1000.0) == 0.01
        assert calc.calculate_dynamic_lot(sl_pips=10.0, account_balance=0) == 0.01
        assert calc.calculate_dynamic_lot(sl_pips=10.0, account_balance=-100) == 0.01
    
    def test_get_partial_exit_levels_buy(self, dynamic_risk_config, mock_db_manager):
        """Test partial exit levels for BUY signal"""
        calc = DynamicRiskCalculator(dynamic_risk_config, mock_db_manager)
        
        levels = calc.get_partial_exit_levels(
            entry_price=2650.0,
            signal_type='BUY',
            sl_pips=10.0
        )
        
        assert 'first_tp' in levels
        assert 'second_tp' in levels
        assert 'trailing_portion' in levels
        assert 'trailing_config' in levels
        
        assert levels['first_tp']['percentage'] == 40.0
        assert levels['second_tp']['percentage'] == 35.0
        assert levels['trailing_portion']['percentage'] == 25.0
        
        assert levels['first_tp']['price'] > levels['entry_price']
        assert levels['second_tp']['price'] > levels['first_tp']['price']
    
    def test_get_partial_exit_levels_sell(self, dynamic_risk_config, mock_db_manager):
        """Test partial exit levels for SELL signal"""
        calc = DynamicRiskCalculator(dynamic_risk_config, mock_db_manager)
        
        levels = calc.get_partial_exit_levels(
            entry_price=2650.0,
            signal_type='SELL',
            sl_pips=10.0
        )
        
        assert levels['first_tp']['price'] < levels['entry_price']
        assert levels['second_tp']['price'] < levels['first_tp']['price']
    
    def test_get_partial_exit_levels_rr_ratios(self, dynamic_risk_config, mock_db_manager):
        """Test RR ratios in partial exit levels"""
        calc = DynamicRiskCalculator(dynamic_risk_config, mock_db_manager)
        
        levels = calc.get_partial_exit_levels(
            entry_price=2650.0,
            signal_type='BUY',
            sl_pips=10.0
        )
        
        assert levels['first_tp']['rr_ratio'] == 1.0
        assert levels['second_tp']['rr_ratio'] == 1.5
    
    def test_trailing_stop_config(self, dynamic_risk_config, mock_db_manager):
        """Test trailing stop configuration"""
        calc = DynamicRiskCalculator(dynamic_risk_config, mock_db_manager)
        
        levels = calc.get_partial_exit_levels(
            entry_price=2650.0,
            signal_type='BUY',
            sl_pips=10.0
        )
        
        assert levels['trailing_config']['move_every_pips'] == 5.0
        assert levels['trailing_config']['distance_pips'] == 3.0
    
    def test_calculate_trailing_stop_buy_profit(self, dynamic_risk_config, mock_db_manager):
        """Test trailing stop calculation for BUY in profit"""
        calc = DynamicRiskCalculator(dynamic_risk_config, mock_db_manager)
        
        new_sl = calc.calculate_trailing_stop(
            entry_price=2650.0,
            current_price=2651.0,
            signal_type='BUY',
            current_sl=2649.0
        )
        
        assert new_sl is not None
        assert new_sl > 2649.0
    
    def test_calculate_trailing_stop_buy_no_update_needed(self, dynamic_risk_config, mock_db_manager):
        """Test trailing stop not updated when profit < threshold"""
        calc = DynamicRiskCalculator(dynamic_risk_config, mock_db_manager)
        
        new_sl = calc.calculate_trailing_stop(
            entry_price=2650.0,
            current_price=2650.2,
            signal_type='BUY',
            current_sl=2649.0
        )
        
        assert new_sl is None
    
    def test_calculate_trailing_stop_sell_profit(self, dynamic_risk_config, mock_db_manager):
        """Test trailing stop calculation for SELL in profit"""
        calc = DynamicRiskCalculator(dynamic_risk_config, mock_db_manager)
        
        new_sl = calc.calculate_trailing_stop(
            entry_price=2650.0,
            current_price=2649.0,
            signal_type='SELL',
            current_sl=2651.0
        )
        
        assert new_sl is not None
        assert new_sl < 2651.0
    
    def test_can_open_position_no_existing(self, dynamic_risk_config, mock_db_manager, test_db):
        """Test can open position when no existing positions"""
        mock_db_manager.get_session = Mock(return_value=test_db)
        calc = DynamicRiskCalculator(dynamic_risk_config, mock_db_manager)
        
        can_open, reason = calc.can_open_position(user_id=123456)
        
        assert can_open is True
        assert reason is None
    
    def test_get_exposure_status_no_positions(self, dynamic_risk_config, mock_db_manager, test_db):
        """Test exposure status with no positions"""
        mock_db_manager.get_session = Mock(return_value=test_db)
        calc = DynamicRiskCalculator(dynamic_risk_config, mock_db_manager)
        
        status = calc.get_exposure_status(user_id=123456)
        
        assert 'open_positions_count' in status
        assert 'total_risk_amount' in status
        assert 'remaining_daily_exposure' in status
        assert 'can_open_new' in status
        assert status['open_positions_count'] == 0
    
    def test_update_exposure_refreshes_cache(self, dynamic_risk_config, mock_db_manager, test_db):
        """Test update_exposure clears and refreshes the cache"""
        mock_db_manager.get_session = Mock(return_value=test_db)
        calc = DynamicRiskCalculator(dynamic_risk_config, mock_db_manager)
        
        old_status = calc.get_exposure_status(user_id=123456)
        assert 123456 in calc._exposure_cache
        old_timestamp = calc._exposure_cache[123456].last_updated
        
        import time
        time.sleep(0.01)
        
        calc.update_exposure(user_id=123456, trade_result={
            'trade_id': 1,
            'actual_pl': 5.0,
            'status': 'TP_HIT'
        })
        
        new_status = calc.get_exposure_status(user_id=123456)
        assert 123456 in calc._exposure_cache
    
    def test_calculate_risk_per_trade(self, dynamic_risk_config, mock_db_manager, test_db):
        """Test risk per trade calculation"""
        mock_db_manager.get_session = Mock(return_value=test_db)
        calc = DynamicRiskCalculator(dynamic_risk_config, mock_db_manager)
        
        risk = calc.calculate_risk_per_trade(user_id=123456)
        
        assert isinstance(risk, float)
        assert risk > 0
        expected_max = calc.effective_daily_threshold / calc.max_concurrent_positions * calc.risk_safety_factor
        assert risk <= expected_max * 1.1
    
    def test_get_summary(self, dynamic_risk_config, mock_db_manager, test_db):
        """Test get_summary returns all expected data"""
        mock_db_manager.get_session = Mock(return_value=test_db)
        calc = DynamicRiskCalculator(dynamic_risk_config, mock_db_manager)
        
        summary = calc.get_summary(user_id=123456)
        
        assert 'settings' in summary
        assert 'partial_exit' in summary
        assert 'current_status' in summary
        assert summary['settings']['max_daily_loss'] == calc.effective_daily_threshold
        assert summary['settings']['max_concurrent_positions'] == calc.max_concurrent_positions
    
    def test_should_auto_close_oldest_not_needed(self, dynamic_risk_config, mock_db_manager, test_db):
        """Test auto-close check when not needed"""
        mock_db_manager.get_session = Mock(return_value=test_db)
        calc = DynamicRiskCalculator(dynamic_risk_config, mock_db_manager)
        
        should_close, position = calc.should_auto_close_oldest(user_id=123456)
        
        assert should_close is False
        assert position is None


@pytest.mark.unit
class TestRiskManagerDynamicIntegration:
    """Tests for RiskManager integration with DynamicRiskCalculator"""
    
    @pytest.fixture
    def dynamic_risk_config(self, mock_config):
        """Config with dynamic risk settings"""
        mock_config.MAX_DAILY_LOSS_AMOUNT = 10.0
        mock_config.MAX_DAILY_LOSS_PERCENT = 1.0
        mock_config.MAX_CONCURRENT_POSITIONS = 4
        mock_config.RISK_SAFETY_FACTOR = 0.5
        mock_config.ACCOUNT_BALANCE = 1000.0
        return mock_config
    
    def test_risk_manager_with_dynamic_risk(self, dynamic_risk_config, mock_db_manager):
        """Test RiskManager initializes with DynamicRiskCalculator"""
        rm = RiskManager(dynamic_risk_config, mock_db_manager, enable_dynamic_risk=True)
        
        assert rm.dynamic_risk is not None
        assert isinstance(rm.dynamic_risk, DynamicRiskCalculator)
    
    def test_risk_manager_without_dynamic_risk(self, dynamic_risk_config, mock_db_manager):
        """Test RiskManager can disable DynamicRiskCalculator"""
        rm = RiskManager(dynamic_risk_config, mock_db_manager, enable_dynamic_risk=False)
        
        assert rm.dynamic_risk is None
    
    def test_get_dynamic_risk_calculator(self, dynamic_risk_config, mock_db_manager):
        """Test getter for DynamicRiskCalculator"""
        rm = RiskManager(dynamic_risk_config, mock_db_manager, enable_dynamic_risk=True)
        
        calc = rm.get_dynamic_risk_calculator()
        
        assert calc is not None
        assert isinstance(calc, DynamicRiskCalculator)
    
    def test_can_open_position_dynamic_with_calc(self, dynamic_risk_config, mock_db_manager, test_db):
        """Test can_open_position_dynamic with calculator"""
        mock_db_manager.get_session = Mock(return_value=test_db)
        rm = RiskManager(dynamic_risk_config, mock_db_manager, enable_dynamic_risk=True)
        
        can_open, reason = rm.can_open_position_dynamic(user_id=123456)
        
        assert can_open is True
        assert reason is None
    
    def test_can_open_position_dynamic_without_calc(self, dynamic_risk_config, mock_db_manager):
        """Test can_open_position_dynamic fallback when no calculator"""
        rm = RiskManager(dynamic_risk_config, mock_db_manager, enable_dynamic_risk=False)
        
        can_open, reason = rm.can_open_position_dynamic(user_id=123456)
        
        assert can_open is True
        assert reason is None
    
    def test_calculate_lot_dynamic_with_calc(self, dynamic_risk_config, mock_db_manager):
        """Test calculate_lot_dynamic with calculator"""
        rm = RiskManager(dynamic_risk_config, mock_db_manager, enable_dynamic_risk=True)
        
        lot = rm.calculate_lot_dynamic(sl_pips=10.0, account_balance=1000.0)
        
        assert isinstance(lot, float)
        assert 0.01 <= lot <= 0.1
    
    def test_calculate_lot_dynamic_without_calc(self, dynamic_risk_config, mock_db_manager):
        """Test calculate_lot_dynamic fallback"""
        rm = RiskManager(dynamic_risk_config, mock_db_manager, enable_dynamic_risk=False)
        
        lot = rm.calculate_lot_dynamic(sl_pips=10.0, account_balance=1000.0)
        
        assert lot == dynamic_risk_config.LOT_SIZE
    
    def test_get_partial_exits_with_calc(self, dynamic_risk_config, mock_db_manager):
        """Test get_partial_exits with calculator"""
        rm = RiskManager(dynamic_risk_config, mock_db_manager, enable_dynamic_risk=True)
        
        exits = rm.get_partial_exits(
            entry_price=2650.0,
            signal_type='BUY',
            sl_pips=10.0
        )
        
        assert 'first_tp' in exits
        assert 'second_tp' in exits
        assert 'trailing_portion' in exits
    
    def test_get_partial_exits_without_calc(self, dynamic_risk_config, mock_db_manager):
        """Test get_partial_exits fallback"""
        rm = RiskManager(dynamic_risk_config, mock_db_manager, enable_dynamic_risk=False)
        
        exits = rm.get_partial_exits(
            entry_price=2650.0,
            signal_type='BUY',
            sl_pips=10.0
        )
        
        assert exits == {}
    
    def test_get_exposure_with_calc(self, dynamic_risk_config, mock_db_manager, test_db):
        """Test get_exposure with calculator"""
        mock_db_manager.get_session = Mock(return_value=test_db)
        rm = RiskManager(dynamic_risk_config, mock_db_manager, enable_dynamic_risk=True)
        
        exposure = rm.get_exposure(user_id=123456)
        
        assert 'open_positions_count' in exposure
        assert 'remaining_daily_exposure' in exposure
    
    def test_get_exposure_without_calc(self, dynamic_risk_config, mock_db_manager):
        """Test get_exposure fallback"""
        rm = RiskManager(dynamic_risk_config, mock_db_manager, enable_dynamic_risk=False)
        
        exposure = rm.get_exposure(user_id=123456)
        
        assert 'error' in exposure


@pytest.mark.unit
class TestRiskManager:
    
    def test_initialization(self, mock_config, mock_db_manager):
        risk_manager = RiskManager(mock_config, mock_db_manager)
        assert risk_manager.config == mock_config
        assert risk_manager.db == mock_db_manager
        assert isinstance(risk_manager.last_signal_time, dict)
        assert isinstance(risk_manager.daily_stats, dict)
    
    def test_can_trade_no_cooldown(self, mock_config, mock_db_manager):
        risk_manager = RiskManager(mock_config, mock_db_manager)
        
        can_trade, reason = risk_manager.can_trade(123456, 'BUY')
        
        assert can_trade is True
        assert reason is None
    
    def test_can_trade_cooldown_active(self, mock_config, mock_db_manager):
        """Test that cooldown is disabled in UNLIMITED mode"""
        risk_manager = RiskManager(mock_config, mock_db_manager)
        
        risk_manager.record_signal(123456)
        can_trade, reason = risk_manager.can_trade(123456, 'BUY')
        
        assert can_trade is True
        assert reason is None
    
    def test_can_trade_cooldown_expired(self, mock_config, mock_db_manager):
        risk_manager = RiskManager(mock_config, mock_db_manager)
        
        past_time = datetime.now(pytz.UTC) - timedelta(seconds=35)
        risk_manager.last_signal_time[123456] = past_time
        
        can_trade, reason = risk_manager.can_trade(123456, 'BUY')
        
        assert can_trade is True
        assert reason is None
    
    def test_can_trade_daily_loss_limit_not_reached(self, mock_config, test_db):
        from unittest.mock import Mock
        db_manager = Mock()
        db_manager.get_session = Mock(return_value=test_db)
        
        risk_manager = RiskManager(mock_config, db_manager)
        
        utc_now = datetime.now(pytz.UTC)
        
        trade1 = Trade(
            user_id=123456,
            ticker='XAUUSD',
            signal_type='BUY',
            entry_price=2650.0,
            stop_loss=2640.0,
            take_profit=2670.0,
            actual_pl=-0.5,
            signal_time=utc_now
        )
        test_db.add(trade1)
        test_db.commit()
        
        can_trade, reason = risk_manager.can_trade(123456, 'BUY')
        
        assert can_trade is True
        assert reason is None
    
    def test_can_trade_daily_loss_limit_reached(self, mock_config, test_db):
        """Test that daily loss limit is disabled in UNLIMITED mode"""
        from unittest.mock import Mock
        db_manager = Mock()
        db_manager.get_session = Mock(return_value=test_db)
        
        risk_manager = RiskManager(mock_config, db_manager)
        
        utc_now = datetime.now(pytz.UTC)
        
        trade1 = Trade(
            user_id=123456,
            ticker='XAUUSD',
            signal_type='BUY',
            entry_price=2650.0,
            stop_loss=2640.0,
            take_profit=2670.0,
            actual_pl=-3.5,
            signal_time=utc_now
        )
        test_db.add(trade1)
        test_db.commit()
        
        can_trade, reason = risk_manager.can_trade(123456, 'BUY')
        
        assert can_trade is True
        assert reason is None
    
    def test_can_trade_different_users(self, mock_config, mock_db_manager):
        """Test that different users can trade independently in UNLIMITED mode"""
        risk_manager = RiskManager(mock_config, mock_db_manager)
        
        risk_manager.record_signal(123456)
        
        can_trade_1, _ = risk_manager.can_trade(123456, 'BUY')
        can_trade_2, _ = risk_manager.can_trade(789012, 'BUY')
        
        assert can_trade_1 is True
        assert can_trade_2 is True
    
    def test_record_signal(self, mock_config, mock_db_manager):
        risk_manager = RiskManager(mock_config, mock_db_manager)
        
        before = datetime.now(pytz.UTC)
        risk_manager.record_signal(123456)
        after = datetime.now(pytz.UTC)
        
        assert 123456 in risk_manager.last_signal_time
        recorded_time = risk_manager.last_signal_time[123456]
        assert before <= recorded_time <= after
    
    def test_calculate_position_size_basic(self, mock_config, mock_db_manager):
        risk_manager = RiskManager(mock_config, mock_db_manager)
        
        account_balance = 1000.0
        entry_price = 2650.0
        stop_loss = 2640.0
        
        lot_size = risk_manager.calculate_position_size(
            account_balance, entry_price, stop_loss, 'BUY'
        )
        
        assert isinstance(lot_size, float)
        assert 0.01 <= lot_size <= 1.0
    
    def test_calculate_position_size_buy_signal(self, mock_config, mock_db_manager):
        risk_manager = RiskManager(mock_config, mock_db_manager)
        
        lot_size = risk_manager.calculate_position_size(
            1000.0, 2650.0, 2640.0, 'BUY'
        )
        
        assert lot_size > 0
    
    def test_calculate_position_size_sell_signal(self, mock_config, mock_db_manager):
        risk_manager = RiskManager(mock_config, mock_db_manager)
        
        lot_size = risk_manager.calculate_position_size(
            1000.0, 2650.0, 2660.0, 'SELL'
        )
        
        assert lot_size > 0
    
    def test_calculate_position_size_bounds(self, mock_config, mock_db_manager):
        risk_manager = RiskManager(mock_config, mock_db_manager)
        
        very_tight_sl = risk_manager.calculate_position_size(
            1000.0, 2650.0, 2649.0, 'BUY'
        )
        assert very_tight_sl <= 1.0
        
        very_wide_sl = risk_manager.calculate_position_size(
            1000.0, 2650.0, 2500.0, 'BUY'
        )
        assert very_wide_sl >= 0.01
    
    def test_calculate_position_size_zero_risk(self, mock_config, mock_db_manager):
        risk_manager = RiskManager(mock_config, mock_db_manager)
        
        lot_size = risk_manager.calculate_position_size(
            1000.0, 2650.0, 2650.0, 'BUY'
        )
        
        assert lot_size == mock_config.LOT_SIZE
    
    def test_calculate_pl_buy_profit(self, mock_config, mock_db_manager):
        risk_manager = RiskManager(mock_config, mock_db_manager)
        
        pl = risk_manager.calculate_pl(
            entry_price=2650.0,
            exit_price=2660.0,
            signal_type='BUY',
            lot_size=0.01
        )
        
        assert pl > 0
    
    def test_calculate_pl_buy_loss(self, mock_config, mock_db_manager):
        risk_manager = RiskManager(mock_config, mock_db_manager)
        
        pl = risk_manager.calculate_pl(
            entry_price=2650.0,
            exit_price=2640.0,
            signal_type='BUY',
            lot_size=0.01
        )
        
        assert pl < 0
    
    def test_calculate_pl_sell_profit(self, mock_config, mock_db_manager):
        risk_manager = RiskManager(mock_config, mock_db_manager)
        
        pl = risk_manager.calculate_pl(
            entry_price=2650.0,
            exit_price=2640.0,
            signal_type='SELL',
            lot_size=0.01
        )
        
        assert pl > 0
    
    def test_calculate_pl_sell_loss(self, mock_config, mock_db_manager):
        risk_manager = RiskManager(mock_config, mock_db_manager)
        
        pl = risk_manager.calculate_pl(
            entry_price=2650.0,
            exit_price=2660.0,
            signal_type='SELL',
            lot_size=0.01
        )
        
        assert pl < 0
    
    def test_calculate_pl_no_movement(self, mock_config, mock_db_manager):
        risk_manager = RiskManager(mock_config, mock_db_manager)
        
        pl_buy = risk_manager.calculate_pl(
            entry_price=2650.0,
            exit_price=2650.0,
            signal_type='BUY',
            lot_size=0.01
        )
        
        pl_sell = risk_manager.calculate_pl(
            entry_price=2650.0,
            exit_price=2650.0,
            signal_type='SELL',
            lot_size=0.01
        )
        
        assert pl_buy == 0.0
        assert pl_sell == 0.0
    
    def test_calculate_pl_default_lot_size(self, mock_config, mock_db_manager):
        risk_manager = RiskManager(mock_config, mock_db_manager)
        
        pl = risk_manager.calculate_pl(
            entry_price=2650.0,
            exit_price=2660.0,
            signal_type='BUY'
        )
        
        assert pl is not None
    
    def test_calculate_pl_different_lot_sizes(self, mock_config, mock_db_manager):
        risk_manager = RiskManager(mock_config, mock_db_manager)
        
        pl_small = risk_manager.calculate_pl(
            entry_price=2650.0,
            exit_price=2660.0,
            signal_type='BUY',
            lot_size=0.01
        )
        
        pl_large = risk_manager.calculate_pl(
            entry_price=2650.0,
            exit_price=2660.0,
            signal_type='BUY',
            lot_size=0.10
        )
        
        assert pl_large > pl_small
        assert abs(pl_large - pl_small * 10) < 0.01
    
    @pytest.mark.parametrize("entry,exit,signal,expected_sign", [
        (2650.0, 2660.0, 'BUY', 1),
        (2650.0, 2640.0, 'BUY', -1),
        (2650.0, 2660.0, 'SELL', -1),
        (2650.0, 2640.0, 'SELL', 1),
    ])
    def test_calculate_pl_sign(self, mock_config, mock_db_manager, entry, exit, signal, expected_sign):
        risk_manager = RiskManager(mock_config, mock_db_manager)
        
        pl = risk_manager.calculate_pl(
            entry_price=entry,
            exit_price=exit,
            signal_type=signal,
            lot_size=0.01
        )
        
        if expected_sign > 0:
            assert pl > 0
        elif expected_sign < 0:
            assert pl < 0
        else:
            assert pl == 0
    
    def test_multiple_users_independent_cooldowns(self, mock_config, mock_db_manager):
        """Test that multiple users can trade without cooldown in UNLIMITED mode"""
        risk_manager = RiskManager(mock_config, mock_db_manager)
        
        risk_manager.record_signal(111111)
        risk_manager.record_signal(222222)
        
        assert 111111 in risk_manager.last_signal_time
        assert 222222 in risk_manager.last_signal_time
        
        can_trade_1, _ = risk_manager.can_trade(111111, 'BUY')
        can_trade_2, _ = risk_manager.can_trade(222222, 'BUY')
        
        assert can_trade_1 is True
        assert can_trade_2 is True
