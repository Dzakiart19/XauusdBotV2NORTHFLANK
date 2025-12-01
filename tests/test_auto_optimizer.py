"""
Comprehensive tests for AutoOptimizer module.

Tests cover:
- OptimizationParameters validation and clamping
- AutoOptimizer initialization and basic operations
- Dynamic adjustment rules
- Safety guards
- Rollback functionality
- Thread safety
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import threading
import pytz

from bot.auto_optimizer import (
    AutoOptimizer,
    OptimizationParameters,
    OptimizationResult,
    Adjustment,
    AdjustmentType,
    OptimizationStatus,
    PerformanceSnapshot,
    AutoOptimizerError
)


class TestOptimizationParameters:
    """Tests for OptimizationParameters dataclass"""
    
    def test_default_values(self):
        """Test default parameter values"""
        params = OptimizationParameters()
        
        assert params.min_confluence_required == 2
        assert params.volume_threshold_multiplier == 1.1
        assert params.ema_strictness == 1.0
        assert params.signal_cooldown == 0
        assert params.enable_m5_low_adx == True
        assert params.three_confluence_weight == 1.0
        assert params.reduced_signal_hours == []
        assert params.adx_threshold_for_m5 == 15.0
    
    def test_validation_valid_params(self):
        """Test validation with valid parameters"""
        params = OptimizationParameters(
            min_confluence_required=3,
            volume_threshold_multiplier=1.2,
            ema_strictness=1.5,
            signal_cooldown=30
        )
        
        is_valid, errors = params.validate()
        assert is_valid == True
        assert len(errors) == 0
    
    def test_validation_invalid_confluence(self):
        """Test validation with invalid confluence"""
        params = OptimizationParameters(min_confluence_required=5)
        
        is_valid, errors = params.validate()
        assert is_valid == False
        assert any('min_confluence_required' in e for e in errors)
    
    def test_validation_invalid_volume_multiplier(self):
        """Test validation with invalid volume multiplier"""
        params = OptimizationParameters(volume_threshold_multiplier=2.0)
        
        is_valid, errors = params.validate()
        assert is_valid == False
        assert any('volume_threshold_multiplier' in e for e in errors)
    
    def test_validation_invalid_ema_strictness(self):
        """Test validation with invalid EMA strictness"""
        params = OptimizationParameters(ema_strictness=3.0)
        
        is_valid, errors = params.validate()
        assert is_valid == False
        assert any('ema_strictness' in e for e in errors)
    
    def test_validation_invalid_signal_cooldown(self):
        """Test validation with invalid signal cooldown"""
        params = OptimizationParameters(signal_cooldown=100)
        
        is_valid, errors = params.validate()
        assert is_valid == False
        assert any('signal_cooldown' in e for e in errors)
    
    def test_clamp_values(self):
        """Test value clamping to valid ranges"""
        params = OptimizationParameters(
            min_confluence_required=10,
            volume_threshold_multiplier=5.0,
            ema_strictness=0.1,
            signal_cooldown=-10,
            three_confluence_weight=3.0,
            reduced_signal_hours=[25, 10, -1]
        )
        
        params.clamp_values()
        
        assert params.min_confluence_required == 4
        assert params.volume_threshold_multiplier == 1.5
        assert params.ema_strictness == 0.5
        assert params.signal_cooldown == 0
        assert params.three_confluence_weight == 2.0
        assert params.reduced_signal_hours == [10]
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        params = OptimizationParameters()
        d = params.to_dict()
        
        assert isinstance(d, dict)
        assert 'min_confluence_required' in d
        assert 'volume_threshold_multiplier' in d
        assert 'ema_strictness' in d
    
    def test_from_dict(self):
        """Test creation from dictionary"""
        data = {
            'min_confluence_required': 3,
            'volume_threshold_multiplier': 1.3,
            'extra_field': 'ignored'
        }
        
        params = OptimizationParameters.from_dict(data)
        
        assert params.min_confluence_required == 3
        assert params.volume_threshold_multiplier == 1.3


class TestAdjustment:
    """Tests for Adjustment dataclass"""
    
    def test_adjustment_creation(self):
        """Test adjustment creation"""
        adj = Adjustment(
            adjustment_type=AdjustmentType.TIGHTEN_CONFLUENCE.value,
            parameter_name='min_confluence_required',
            old_value=2,
            new_value=3,
            reason='Test reason'
        )
        
        assert adj.adjustment_type == 'tighten_confluence'
        assert adj.parameter_name == 'min_confluence_required'
        assert adj.old_value == 2
        assert adj.new_value == 3
        assert adj.reason == 'Test reason'
        assert adj.timestamp is not None
    
    def test_adjustment_to_dict(self):
        """Test adjustment to dict conversion"""
        adj = Adjustment(
            adjustment_type=AdjustmentType.TIGHTEN_CONFLUENCE.value,
            parameter_name='min_confluence_required',
            old_value=2,
            new_value=3,
            reason='Test'
        )
        
        d = adj.to_dict()
        
        assert isinstance(d, dict)
        assert d['adjustment_type'] == 'tighten_confluence'
        assert 'timestamp' in d


class TestAutoOptimizer:
    """Tests for AutoOptimizer class"""
    
    @pytest.fixture
    def optimizer(self):
        """Create basic optimizer instance"""
        return AutoOptimizer()
    
    @pytest.fixture
    def mock_signal_quality_data(self):
        """Create mock signal quality data"""
        return {
            'total_signals': 50,
            'overall_accuracy': 0.55,
            'accuracy_by_type': {
                'M1_SCALP': {'accuracy': 0.40, 'total': 25, 'wins': 10},
                'M5_SWING': {'accuracy': 0.60, 'total': 25, 'wins': 15}
            },
            'accuracy_by_regime': {
                'trending': {'accuracy': 0.60, 'total': 30, 'wins': 18},
                'ranging': {'accuracy': 0.45, 'total': 20, 'wins': 9}
            },
            'accuracy_by_confluence': {
                '2_confluence': {'accuracy': 0.50, 'total': 30, 'wins': 15},
                '3+_confluence': {'accuracy': 0.75, 'total': 20, 'wins': 15}
            },
            'accuracy_by_hour': {
                8: {'accuracy': 0.40, 'total': 15, 'wins': 6},
                14: {'accuracy': 0.65, 'total': 15, 'wins': 10}
            },
            'profit_factor': 1.5
        }
    
    def test_initialization(self, optimizer):
        """Test optimizer initialization"""
        assert optimizer.is_enabled()
        assert optimizer.get_optimization_history() == []
        
        params = optimizer.get_current_parameters()
        assert params['min_confluence_required'] == 2
    
    def test_initialization_with_custom_interval(self):
        """Test optimizer initialization with custom interval"""
        optimizer = AutoOptimizer(optimization_interval_hours=12)
        assert optimizer.optimization_interval_hours == 12
    
    def test_get_current_parameters(self, optimizer):
        """Test getting current parameters"""
        params = optimizer.get_current_parameters()
        
        assert isinstance(params, dict)
        assert 'min_confluence_required' in params
        assert 'volume_threshold_multiplier' in params
    
    def test_reset_to_defaults(self, optimizer):
        """Test reset to default parameters"""
        optimizer.apply_adjustments({'min_confluence_required': 4})
        optimizer.reset_to_defaults()
        
        params = optimizer.get_current_parameters()
        assert params['min_confluence_required'] == 2
    
    def test_apply_adjustments(self, optimizer):
        """Test applying adjustments"""
        success = optimizer.apply_adjustments({
            'min_confluence_required': 3,
            'volume_threshold_multiplier': 1.3
        })
        
        assert success == True
        params = optimizer.get_current_parameters()
        assert params['min_confluence_required'] == 3
        assert params['volume_threshold_multiplier'] == 1.3
    
    def test_apply_adjustments_invalid(self, optimizer):
        """Test applying invalid adjustments"""
        success = optimizer.apply_adjustments({
            'invalid_param': 100
        })
        
        assert success == False
    
    def test_rollback_last_adjustment(self, optimizer):
        """Test rollback functionality"""
        optimizer.apply_adjustments({'min_confluence_required': 4})
        optimizer.rollback_last_adjustment()
        
        params = optimizer.get_current_parameters()
        assert params['min_confluence_required'] == 2
    
    def test_rollback_no_history(self, optimizer):
        """Test rollback with no history"""
        result = optimizer.rollback_last_adjustment()
        assert result == False
    
    def test_enable_disable(self, optimizer):
        """Test enable/disable functionality"""
        optimizer.disable()
        assert optimizer.is_enabled() == False
        
        optimizer.enable()
        assert optimizer.is_enabled() == True
    
    def test_should_run_optimization_first_run(self, optimizer):
        """Test should run optimization on first run"""
        should_run, reason = optimizer.should_run_optimization()
        
        assert should_run == True
        assert 'First' in reason
    
    def test_should_run_optimization_disabled(self, optimizer):
        """Test should not run when disabled"""
        optimizer.disable()
        should_run, reason = optimizer.should_run_optimization()
        
        assert should_run == False
        assert 'disabled' in reason
    
    def test_run_optimization_success(self, optimizer, mock_signal_quality_data):
        """Test successful optimization run"""
        result = optimizer.run_optimization(mock_signal_quality_data)
        
        assert result.status == OptimizationStatus.SUCCESS.value
        assert len(result.adjustments) > 0
        assert result.signals_analyzed == 50
    
    def test_run_optimization_insufficient_signals(self, optimizer):
        """Test optimization with insufficient signals"""
        data = {'total_signals': 5, 'overall_accuracy': 0.5}
        result = optimizer.run_optimization(data)
        
        assert result.status == OptimizationStatus.SKIPPED.value
        assert 'Insufficient' in result.error_message
    
    def test_run_optimization_disabled(self, optimizer, mock_signal_quality_data):
        """Test optimization when disabled"""
        optimizer.disable()
        result = optimizer.run_optimization(mock_signal_quality_data)
        
        assert result.status == OptimizationStatus.SKIPPED.value
        assert 'disabled' in result.error_message
    
    def test_run_optimization_no_tracker_no_data(self, optimizer):
        """Test optimization without tracker or data"""
        result = optimizer.run_optimization(None)
        
        assert result.status == OptimizationStatus.FAILED.value
    
    def test_m1_scalp_accuracy_rule(self, optimizer):
        """Test M1 scalp accuracy dynamic rule"""
        data = {
            'total_signals': 50,
            'overall_accuracy': 0.55,
            'accuracy_by_type': {
                'M1_SCALP': {'accuracy': 0.40, 'total': 30, 'wins': 12}
            },
            'accuracy_by_regime': {},
            'accuracy_by_confluence': {},
            'accuracy_by_hour': {}
        }
        
        result = optimizer.run_optimization(data)
        
        confluence_adj = [a for a in result.adjustments 
                        if a.parameter_name == 'min_confluence_required']
        assert len(confluence_adj) > 0
        assert confluence_adj[0].new_value > 2
    
    def test_three_confluence_weight_rule(self, optimizer):
        """Test 3-confluence weight dynamic rule"""
        data = {
            'total_signals': 50,
            'overall_accuracy': 0.60,
            'accuracy_by_type': {
                'M1_SCALP': {'accuracy': 0.55, 'total': 50, 'wins': 28}
            },
            'accuracy_by_regime': {},
            'accuracy_by_confluence': {
                '3+_confluence': {'accuracy': 0.80, 'total': 20, 'wins': 16}
            },
            'accuracy_by_hour': {}
        }
        
        result = optimizer.run_optimization(data)
        
        weight_adj = [a for a in result.adjustments 
                     if a.parameter_name == 'three_confluence_weight']
        assert len(weight_adj) > 0
        assert weight_adj[0].new_value > 1.0
    
    def test_hourly_accuracy_rule(self, optimizer):
        """Test hourly accuracy dynamic rule"""
        data = {
            'total_signals': 50,
            'overall_accuracy': 0.55,
            'accuracy_by_type': {},
            'accuracy_by_regime': {},
            'accuracy_by_confluence': {},
            'accuracy_by_hour': {
                8: {'accuracy': 0.35, 'total': 15, 'wins': 5},
                9: {'accuracy': 0.38, 'total': 12, 'wins': 4}
            }
        }
        
        result = optimizer.run_optimization(data)
        
        hour_adj = [a for a in result.adjustments 
                   if a.parameter_name == 'reduced_signal_hours']
        assert len(hour_adj) > 0
        assert 8 in hour_adj[0].new_value or 9 in hour_adj[0].new_value
    
    def test_get_optimization_history(self, optimizer, mock_signal_quality_data):
        """Test getting optimization history"""
        optimizer.run_optimization(mock_signal_quality_data)
        
        history = optimizer.get_optimization_history()
        
        assert isinstance(history, list)
        assert len(history) > 0
        assert 'status' in history[0]
    
    def test_get_status_report(self, optimizer):
        """Test getting status report"""
        report = optimizer.get_status_report()
        
        assert isinstance(report, str)
        assert 'Auto-Optimizer' in report
        assert 'Status' in report
    
    def test_is_market_closed(self, optimizer):
        """Test market closed detection"""
        result = optimizer.is_market_closed()
        assert isinstance(result, bool)
    
    def test_strategy_update_callback(self, mock_signal_quality_data):
        """Test strategy update callback is called"""
        callback = Mock(return_value=True)
        optimizer = AutoOptimizer(strategy_update_callback=callback)
        
        optimizer.run_optimization(mock_signal_quality_data)
        
        assert callback.called
    
    def test_set_signal_quality_tracker(self, optimizer):
        """Test setting signal quality tracker"""
        mock_tracker = Mock()
        optimizer.set_signal_quality_tracker(mock_tracker)
        
        assert optimizer.signal_quality_tracker == mock_tracker
    
    def test_set_strategy_update_callback(self, optimizer):
        """Test setting strategy update callback"""
        callback = Mock()
        optimizer.set_strategy_update_callback(callback)
        
        assert optimizer.strategy_update_callback == callback
    
    def test_check_and_rollback_if_needed(self, optimizer, mock_signal_quality_data):
        """Test check and rollback functionality"""
        optimizer.run_optimization(mock_signal_quality_data)
        
        poor_performance = {
            'total_signals': 100,
            'overall_accuracy': 0.40
        }
        
        result = optimizer.check_and_rollback_if_needed(poor_performance)
        assert isinstance(result, bool)


class TestAutoOptimizerThreadSafety:
    """Tests for AutoOptimizer thread safety"""
    
    def test_concurrent_access(self):
        """Test thread-safe concurrent access"""
        optimizer = AutoOptimizer()
        results = []
        errors = []
        
        def run_optimization():
            try:
                data = {
                    'total_signals': 50,
                    'overall_accuracy': 0.55,
                    'accuracy_by_type': {'M1_SCALP': {'accuracy': 0.40, 'total': 25, 'wins': 10}},
                    'accuracy_by_regime': {},
                    'accuracy_by_confluence': {},
                    'accuracy_by_hour': {}
                }
                result = optimizer.run_optimization(data)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        def get_params():
            try:
                for _ in range(10):
                    params = optimizer.get_current_parameters()
                    assert isinstance(params, dict)
            except Exception as e:
                errors.append(e)
        
        threads = []
        for _ in range(5):
            t1 = threading.Thread(target=run_optimization)
            t2 = threading.Thread(target=get_params)
            threads.extend([t1, t2])
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0
    
    def test_concurrent_adjustments(self):
        """Test concurrent adjustment applications"""
        optimizer = AutoOptimizer()
        errors = []
        
        def apply_adj(value):
            try:
                optimizer.apply_adjustments({'min_confluence_required': value})
            except Exception as e:
                errors.append(e)
        
        threads = []
        for i in range(2, 5):
            t = threading.Thread(target=apply_adj, args=(i,))
            threads.append(t)
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        params = optimizer.get_current_parameters()
        assert 2 <= params['min_confluence_required'] <= 4


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass"""
    
    def test_result_creation(self):
        """Test result creation"""
        result = OptimizationResult(
            status=OptimizationStatus.SUCCESS.value,
            signals_analyzed=50
        )
        
        assert result.status == 'SUCCESS'
        assert result.signals_analyzed == 50
        assert result.adjustments == []
    
    def test_result_to_dict(self):
        """Test result to dict conversion"""
        result = OptimizationResult(
            status=OptimizationStatus.SUCCESS.value,
            signals_analyzed=50
        )
        
        d = result.to_dict()
        
        assert isinstance(d, dict)
        assert d['status'] == 'SUCCESS'
        assert 'timestamp' in d


class TestPerformanceSnapshot:
    """Tests for PerformanceSnapshot dataclass"""
    
    def test_snapshot_creation(self):
        """Test snapshot creation"""
        params = OptimizationParameters()
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(pytz.UTC),
            overall_accuracy=0.60,
            parameters=params,
            signals_count=100
        )
        
        assert snapshot.overall_accuracy == 0.60
        assert snapshot.signals_count == 100
    
    def test_snapshot_to_dict(self):
        """Test snapshot to dict conversion"""
        params = OptimizationParameters()
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(pytz.UTC),
            overall_accuracy=0.60,
            parameters=params,
            signals_count=100
        )
        
        d = snapshot.to_dict()
        
        assert isinstance(d, dict)
        assert d['overall_accuracy'] == 0.60
        assert 'parameters' in d


class TestSafetyGuards:
    """Tests for safety guards functionality"""
    
    def test_minimum_signals_guard(self):
        """Test minimum signals guard"""
        optimizer = AutoOptimizer()
        
        data = {'total_signals': 10, 'overall_accuracy': 0.30}
        result = optimizer.run_optimization(data)
        
        assert result.status == OptimizationStatus.SKIPPED.value
        assert 'Insufficient' in result.error_message
    
    def test_max_adjustment_limits(self):
        """Test maximum adjustment limits"""
        optimizer = AutoOptimizer()
        
        data = {
            'total_signals': 100,
            'overall_accuracy': 0.30,
            'accuracy_by_type': {
                'M1_SCALP': {'accuracy': 0.20, 'total': 50, 'wins': 10}
            },
            'accuracy_by_regime': {},
            'accuracy_by_confluence': {},
            'accuracy_by_hour': {}
        }
        
        result = optimizer.run_optimization(data)
        
        params = optimizer.get_current_parameters()
        assert params['min_confluence_required'] <= OptimizationParameters.MAX_CONFLUENCE
    
    def test_consecutive_failures_disable(self):
        """Test auto-disable after consecutive failures"""
        optimizer = AutoOptimizer()
        optimizer._max_consecutive_failures = 2
        
        with patch.object(optimizer, '_analyze_and_generate_adjustments', side_effect=Exception("Test error")):
            optimizer.run_optimization({'total_signals': 50, 'overall_accuracy': 0.5})
            optimizer.run_optimization({'total_signals': 50, 'overall_accuracy': 0.5})
        
        assert optimizer.is_enabled() == False


class TestM5SwingRangingRule:
    """Tests for M5 swing ranging market rule"""
    
    def test_disable_m5_low_adx(self):
        """Test disabling M5 for low ADX"""
        optimizer = AutoOptimizer()
        
        data = {
            'total_signals': 50,
            'overall_accuracy': 0.50,
            'accuracy_by_type': {
                'M5_SWING': {'accuracy': 0.40, 'total': 20, 'wins': 8}
            },
            'accuracy_by_regime': {
                'ranging': {'accuracy': 0.35, 'total': 15, 'wins': 5}
            },
            'accuracy_by_confluence': {},
            'accuracy_by_hour': {}
        }
        
        result = optimizer.run_optimization(data)
        
        m5_adj = [a for a in result.adjustments 
                 if a.parameter_name == 'enable_m5_low_adx']
        if m5_adj:
            assert m5_adj[0].new_value == False
    
    def test_enable_m5_improved(self):
        """Test re-enabling M5 when performance improves"""
        optimizer = AutoOptimizer()
        optimizer.apply_adjustments({'enable_m5_low_adx': False})
        
        data = {
            'total_signals': 50,
            'overall_accuracy': 0.60,
            'accuracy_by_type': {
                'M5_SWING': {'accuracy': 0.60, 'total': 20, 'wins': 12}
            },
            'accuracy_by_regime': {
                'ranging': {'accuracy': 0.55, 'total': 15, 'wins': 8}
            },
            'accuracy_by_confluence': {},
            'accuracy_by_hour': {}
        }
        
        result = optimizer.run_optimization(data)
        
        m5_adj = [a for a in result.adjustments 
                 if a.parameter_name == 'enable_m5_low_adx']
        if m5_adj:
            assert m5_adj[0].new_value == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
