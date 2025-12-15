"""Unit tests for indicators module."""
import pytest
import pandas as pd
import numpy as np
from bot.indicators import (
    validate_series,
    safe_divide,
    safe_series_operation,
    _ensure_series,
)


class TestValidateSeries:
    """Tests for validate_series function."""
    
    def test_valid_series(self):
        """Should return valid series unchanged."""
        series = pd.Series([1.0, 2.0, 3.0])
        result = validate_series(series)
        assert len(result) == 3
        assert result.iloc[0] == 1.0
    
    def test_fills_nan_values(self):
        """Should fill NaN values with default."""
        series = pd.Series([1.0, np.nan, 3.0])
        result = validate_series(series, fill_value=0.0)
        assert not result.isna().any()
        assert result.iloc[1] == 0.0
    
    def test_fills_inf_values(self):
        """Should fill Inf values."""
        series = pd.Series([1.0, np.inf, -np.inf])
        result = validate_series(series, fill_value=0.0)
        assert not np.isinf(result).any()
    
    def test_pads_short_series(self):
        """Should pad series shorter than min_length."""
        series = pd.Series([1.0, 2.0])
        result = validate_series(series, min_length=5, fill_value=0.0)
        assert len(result) >= 5
    
    def test_converts_list_to_series(self):
        """Should convert list to Series."""
        result = validate_series([1.0, 2.0, 3.0])
        assert isinstance(result, pd.Series)
        assert len(result) == 3
    
    def test_converts_numpy_array(self):
        """Should convert numpy array to Series."""
        arr = np.array([1.0, 2.0, 3.0])
        result = validate_series(arr)
        assert isinstance(result, pd.Series)
    
    def test_raises_on_none(self):
        """Should raise ValueError on None input."""
        with pytest.raises(ValueError):
            validate_series(None)
    
    def test_handles_empty_series(self):
        """Should handle empty series."""
        series = pd.Series([])
        result = validate_series(series, min_length=1)
        assert len(result) >= 1


class TestSafeDivide:
    """Tests for safe_divide function."""
    
    def test_normal_division(self):
        """Should perform normal division."""
        num = pd.Series([10.0, 20.0, 30.0])
        denom = pd.Series([2.0, 4.0, 5.0])
        result = safe_divide(num, denom)
        assert result.iloc[0] == pytest.approx(5.0)
        assert result.iloc[1] == pytest.approx(5.0)
        assert result.iloc[2] == pytest.approx(6.0)
    
    def test_division_by_zero(self):
        """Should handle division by zero."""
        num = pd.Series([10.0, 20.0])
        denom = pd.Series([2.0, 0.0])
        result = safe_divide(num, denom, fill_value=0.0)
        assert result.iloc[0] == pytest.approx(5.0)
        assert not np.isinf(result.iloc[1])
    
    def test_handles_nan_numerator(self):
        """Should handle NaN in numerator."""
        num = pd.Series([np.nan, 20.0])
        denom = pd.Series([2.0, 4.0])
        result = safe_divide(num, denom, fill_value=0.0)
        assert not np.isnan(result.iloc[0])
    
    def test_handles_inf_values(self):
        """Should handle Inf values."""
        num = pd.Series([np.inf, 20.0])
        denom = pd.Series([2.0, 4.0])
        result = safe_divide(num, denom, fill_value=0.0)
        assert not np.isinf(result.iloc[0])
    
    def test_scalar_division(self):
        """Should handle scalar numerator."""
        result = safe_divide(10.0, pd.Series([2.0, 5.0]))
        assert result.iloc[0] == pytest.approx(5.0)
        assert result.iloc[1] == pytest.approx(2.0)


class TestSafeSeriesOperation:
    """Tests for safe_series_operation function."""
    
    def test_get_value_at_index(self):
        """Should get value at specified index."""
        series = pd.Series([1.0, 2.0, 3.0])
        result = safe_series_operation(series, 'value', index=-1)
        assert result == 3.0
    
    def test_get_mean(self):
        """Should calculate mean."""
        series = pd.Series([1.0, 2.0, 3.0])
        result = safe_series_operation(series, 'mean')
        assert result == pytest.approx(2.0)
    
    def test_get_sum(self):
        """Should calculate sum."""
        series = pd.Series([1.0, 2.0, 3.0])
        result = safe_series_operation(series, 'sum')
        assert result == pytest.approx(6.0)
    
    def test_get_min(self):
        """Should get minimum value."""
        series = pd.Series([3.0, 1.0, 2.0])
        result = safe_series_operation(series, 'min')
        assert result == pytest.approx(1.0)
    
    def test_get_max(self):
        """Should get maximum value."""
        series = pd.Series([1.0, 3.0, 2.0])
        result = safe_series_operation(series, 'max')
        assert result == pytest.approx(3.0)
    
    def test_handles_empty_series(self):
        """Should return default for empty series."""
        series = pd.Series([])
        result = safe_series_operation(series, 'mean', default=0.0)
        assert result == 0.0
    
    def test_handles_none_series(self):
        """Should return default for None series."""
        result = safe_series_operation(None, 'mean', default=-1.0)
        assert result == -1.0
    
    def test_handles_nan_in_series(self):
        """Should handle NaN values."""
        series = pd.Series([1.0, np.nan, 3.0])
        result = safe_series_operation(series, 'mean')
        assert result == pytest.approx(2.0)


class TestEnsureSeries:
    """Tests for _ensure_series function."""
    
    def test_series_passthrough(self):
        """Should return Series unchanged."""
        series = pd.Series([1.0, 2.0])
        result = _ensure_series(series)
        assert isinstance(result, pd.Series)
        pd.testing.assert_series_equal(result, series)
    
    def test_dataframe_single_column(self):
        """Should extract single column from DataFrame."""
        df = pd.DataFrame({'a': [1.0, 2.0]})
        result = _ensure_series(df)
        assert isinstance(result, pd.Series)
        assert len(result) == 2
    
    def test_numpy_array(self):
        """Should convert numpy array."""
        arr = np.array([1.0, 2.0, 3.0])
        result = _ensure_series(arr)
        assert isinstance(result, pd.Series)
        assert len(result) == 3
    
    def test_scalar_float(self):
        """Should convert scalar float."""
        result = _ensure_series(5.0)
        assert isinstance(result, pd.Series)
        assert len(result) == 1
        assert result.iloc[0] == 5.0
    
    def test_none_input(self):
        """Should handle None input."""
        result = _ensure_series(None)
        assert isinstance(result, pd.Series)
        assert result.iloc[0] == 0.0
    
    def test_nan_scalar(self):
        """Should handle NaN scalar."""
        result = _ensure_series(np.nan)
        assert isinstance(result, pd.Series)
        assert result.iloc[0] == 0.0
