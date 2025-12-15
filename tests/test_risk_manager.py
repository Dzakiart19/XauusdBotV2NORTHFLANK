"""Unit tests for risk_manager module."""
import pytest
from unittest.mock import Mock, MagicMock, patch
from bot.risk_manager import DynamicRiskCalculator


class MockConfig:
    """Mock configuration for testing."""
    MAX_DAILY_LOSS_AMOUNT = 10.0
    MAX_DAILY_LOSS_PERCENT = 1.0
    MAX_CONCURRENT_POSITIONS = 4
    RISK_SAFETY_FACTOR = 0.5
    ACCOUNT_BALANCE = 1000.0
    XAUUSD_PIP_VALUE = 10.0
    RISK_PER_TRADE_PERCENT = 2.0
    DRAWDOWN_RECOVERY_USE_LOT_REDUCTION = False


class MockDBManager:
    """Mock database manager for testing."""
    pass


class TestDynamicRiskCalculator:
    """Tests for DynamicRiskCalculator class."""
    
    @pytest.fixture
    def config(self):
        return MockConfig()
    
    @pytest.fixture
    def db_manager(self):
        return MockDBManager()
    
    @pytest.fixture
    def calculator(self, config, db_manager):
        return DynamicRiskCalculator(config, db_manager)
    
    def test_initialization(self, calculator):
        """Should initialize with correct default values."""
        assert calculator.max_daily_loss == 10.0
        assert calculator.max_concurrent_positions == 4
        assert calculator.risk_safety_factor == 0.5
        assert calculator.MIN_LOT_SIZE == 0.01
        assert calculator.MAX_LOT_SIZE == 0.1
    
    def test_calculate_dynamic_lot_normal(self, calculator):
        """Should calculate lot size correctly."""
        lot = calculator.calculate_dynamic_lot(
            sl_pips=10.0,
            account_balance=1000.0,
            user_id=None
        )
        assert lot >= calculator.MIN_LOT_SIZE
        assert lot <= calculator.MAX_LOT_SIZE
        assert isinstance(lot, float)
    
    def test_calculate_dynamic_lot_invalid_sl(self, calculator):
        """Should return minimum lot for invalid SL."""
        lot = calculator.calculate_dynamic_lot(
            sl_pips=0,
            account_balance=1000.0
        )
        assert lot == calculator.MIN_LOT_SIZE
        
        lot = calculator.calculate_dynamic_lot(
            sl_pips=-5,
            account_balance=1000.0
        )
        assert lot == calculator.MIN_LOT_SIZE
    
    def test_calculate_dynamic_lot_invalid_balance(self, calculator):
        """Should return minimum lot for invalid balance."""
        lot = calculator.calculate_dynamic_lot(
            sl_pips=10.0,
            account_balance=0
        )
        assert lot == calculator.MIN_LOT_SIZE
        
        lot = calculator.calculate_dynamic_lot(
            sl_pips=10.0,
            account_balance=-100
        )
        assert lot == calculator.MIN_LOT_SIZE
    
    def test_calculate_dynamic_sl_expansion_normal(self, calculator):
        """Should return normal expansion for low drawdown."""
        result = calculator.calculate_dynamic_sl_expansion(10.0)
        assert result['drawdown_percent'] == 10.0
        assert result['sl_multiplier'] == 1.0
        assert result['mode'] == 'NORMAL'
        assert result['lot_reduction'] == 1.0
        assert result['should_adjust'] == False
    
    def test_calculate_dynamic_sl_expansion_warning(self, calculator):
        """Should return expansion mode for medium drawdown."""
        result = calculator.calculate_dynamic_sl_expansion(25.0)
        assert result['drawdown_percent'] == 25.0
        assert result['sl_multiplier'] == 1.15
        assert result['mode'] == 'EXPANSION'
        assert result['lot_reduction'] == 1.0
        assert result['should_adjust'] == True
    
    def test_calculate_dynamic_sl_expansion_critical(self, calculator):
        """Should return recovery mode for high drawdown."""
        result = calculator.calculate_dynamic_sl_expansion(50.0)
        assert result['drawdown_percent'] == 50.0
        assert result['sl_multiplier'] == 1.30
        assert result['mode'] == 'RECOVERY'
        assert result['should_adjust'] == True
    
    def test_calculate_dynamic_sl_expansion_none_input(self, calculator):
        """Should handle None input gracefully."""
        result = calculator.calculate_dynamic_sl_expansion(None)
        assert result['sl_multiplier'] == 1.0
        assert result['should_adjust'] == False
    
    def test_calculate_dynamic_sl_expansion_negative_input(self, calculator):
        """Should handle negative input gracefully."""
        result = calculator.calculate_dynamic_sl_expansion(-10.0)
        assert result['sl_multiplier'] == 1.0
        assert result['drawdown_percent'] == 0.0


class TestGetDynamicSLAdjustment:
    """Tests for get_dynamic_sl_adjustment method."""
    
    @pytest.fixture
    def config(self):
        return MockConfig()
    
    @pytest.fixture
    def db_manager(self):
        return MockDBManager()
    
    @pytest.fixture
    def calculator(self, config, db_manager):
        calc = DynamicRiskCalculator(config, db_manager)
        calc.get_exposure_status = Mock(return_value={
            'daily_realized_loss': 0.0,
            'combined_exposure': 0.0,
            'daily_threshold': 10.0,
            'open_positions_count': 0,
            'total_risk_amount': 0.0
        })
        return calc
    
    def test_normal_drawdown(self, calculator):
        """Should return normal settings for low drawdown."""
        calculator.get_exposure_status = Mock(return_value={
            'daily_realized_loss': 1.0,
            'combined_exposure': 1.0,
            'daily_threshold': 10.0
        })
        result = calculator.get_dynamic_sl_adjustment(user_id=123)
        
        assert result['drawdown_level'] == 'NORMAL'
        assert result['sl_multiplier'] == 1.0
        assert result['lot_adjustment'] == 1.0
        assert result['should_adjust'] == False
    
    def test_warning_drawdown(self, calculator):
        """Should return warning settings for medium drawdown."""
        calculator.get_exposure_status = Mock(return_value={
            'daily_realized_loss': 3.0,
            'combined_exposure': 3.0,
            'daily_threshold': 10.0
        })
        result = calculator.get_dynamic_sl_adjustment(user_id=123)
        
        assert result['drawdown_level'] == 'WARNING'
        assert result['sl_multiplier'] == 1.15
        assert result['lot_adjustment'] == 1.0
        assert result['should_adjust'] == True
    
    def test_critical_drawdown(self, calculator):
        """Should return critical settings for high drawdown."""
        calculator.get_exposure_status = Mock(return_value={
            'daily_realized_loss': 5.0,
            'combined_exposure': 5.0,
            'daily_threshold': 10.0
        })
        result = calculator.get_dynamic_sl_adjustment(user_id=123)
        
        assert result['drawdown_level'] == 'CRITICAL'
        assert result['sl_multiplier'] == 1.30
        assert result['lot_adjustment'] == 0.5
        assert result['should_adjust'] == True


class TestCanOpenPosition:
    """Tests for can_open_position method."""
    
    @pytest.fixture
    def config(self):
        return MockConfig()
    
    @pytest.fixture
    def db_manager(self):
        return MockDBManager()
    
    @pytest.fixture
    def calculator(self, config, db_manager):
        return DynamicRiskCalculator(config, db_manager)
    
    def test_can_open_when_no_positions(self, calculator):
        """Should allow opening position when no existing positions."""
        calculator.get_exposure_status = Mock(return_value={
            'open_positions_count': 0,
            'total_risk_amount': 0.0,
            'daily_realized_loss': 0.0
        })
        
        can_open, reason = calculator.can_open_position(user_id=123)
        assert can_open == True
        assert reason is None
    
    def test_cannot_open_max_positions_reached(self, calculator):
        """Should block when max positions reached."""
        calculator.get_exposure_status = Mock(return_value={
            'open_positions_count': 4,
            'total_risk_amount': 5.0,
            'daily_realized_loss': 0.0
        })
        
        can_open, reason = calculator.can_open_position(user_id=123)
        assert can_open == False
        assert 'Max concurrent positions' in reason
    
    def test_cannot_open_daily_limit_reached(self, calculator):
        """Should block when daily exposure limit reached."""
        calculator.get_exposure_status = Mock(return_value={
            'open_positions_count': 1,
            'total_risk_amount': 5.0,
            'daily_realized_loss': 6.0
        })
        
        can_open, reason = calculator.can_open_position(user_id=123)
        assert can_open == False
        assert 'Daily exposure limit' in reason


class TestPartialExitLevels:
    """Tests for get_partial_exit_levels method."""
    
    @pytest.fixture
    def config(self):
        return MockConfig()
    
    @pytest.fixture
    def db_manager(self):
        return MockDBManager()
    
    @pytest.fixture
    def calculator(self, config, db_manager):
        return DynamicRiskCalculator(config, db_manager)
    
    def test_buy_partial_exit_levels(self, calculator):
        """Should calculate correct exit levels for BUY."""
        result = calculator.get_partial_exit_levels(
            entry_price=2650.0,
            signal_type='BUY',
            sl_pips=10.0
        )
        
        assert 'entry_price' in result
        assert 'first_tp' in result
        assert 'second_tp' in result
        assert result['entry_price'] == 2650.0
        assert result['first_tp']['price'] > result['entry_price']
        assert result['second_tp']['price'] > result['first_tp']['price']
    
    def test_sell_partial_exit_levels(self, calculator):
        """Should calculate correct exit levels for SELL."""
        result = calculator.get_partial_exit_levels(
            entry_price=2650.0,
            signal_type='SELL',
            sl_pips=10.0
        )
        
        assert result['entry_price'] == 2650.0
        assert result['first_tp']['price'] < result['entry_price']
        assert result['second_tp']['price'] < result['first_tp']['price']
