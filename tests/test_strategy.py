"""Unit tests for strategy module."""
import pytest
import math
from bot.strategy import (
    PriceDataValidator,
    ValidationResult,
    validate_price_data,
    is_valid_number,
)


class TestPriceDataValidator:
    """Tests for PriceDataValidator class."""
    
    @pytest.fixture
    def validator(self):
        return PriceDataValidator()
    
    def test_validate_valid_number(self, validator):
        """Should validate normal numbers correctly."""
        result = validator.validate(100.5, "test_value")
        assert result.is_valid == True
        assert result.value == 100.5
        assert result.error is None
    
    def test_validate_none_value(self, validator):
        """Should reject None values."""
        result = validator.validate(None, "test_value")
        assert result.is_valid == False
        assert "None" in result.error
    
    def test_validate_nan_value(self, validator):
        """Should reject NaN values."""
        result = validator.validate(float('nan'), "test_value")
        assert result.is_valid == False
        assert "NaN" in result.error
    
    def test_validate_inf_value(self, validator):
        """Should reject Inf values."""
        result = validator.validate(float('inf'), "test_value")
        assert result.is_valid == False
        assert "Inf" in result.error
        
        result = validator.validate(float('-inf'), "test_value")
        assert result.is_valid == False
    
    def test_validate_negative_not_allowed(self, validator):
        """Should reject negative values when not allowed."""
        result = validator.validate(-10.0, "test_value", allow_negative=False)
        assert result.is_valid == False
        assert "negative" in result.error
    
    def test_validate_negative_allowed(self, validator):
        """Should accept negative values when allowed."""
        result = validator.validate(-10.0, "test_value", allow_negative=True)
        assert result.is_valid == True
        assert result.value == -10.0
    
    def test_validate_zero_not_allowed(self, validator):
        """Should reject zero when not allowed."""
        result = validator.validate(0, "test_value", allow_zero=False)
        assert result.is_valid == False
        assert "zero" in result.error
    
    def test_validate_zero_allowed(self, validator):
        """Should accept zero when allowed."""
        result = validator.validate(0, "test_value", allow_zero=True)
        assert result.is_valid == True
        assert result.value == 0
    
    def test_validate_min_max_range(self, validator):
        """Should clamp values to min/max range with warnings."""
        result = validator.validate(150.0, "test_value", min_val=0, max_val=100)
        assert result.is_valid == True
        assert result.value == 100
        assert len(result.warnings) > 0
        
        result = validator.validate(-10.0, "test_value", min_val=0, max_val=100, allow_negative=True)
        assert result.is_valid == True
        assert result.value == 0
    
    def test_validate_price(self, validator):
        """Should validate price values correctly."""
        result = validator.validate_price(2650.50)
        assert result.is_valid == True
        
        result = validator.validate_price(0)
        assert result.is_valid == False
        
        result = validator.validate_price(-100)
        assert result.is_valid == False
    
    def test_validate_atr(self, validator):
        """Should validate ATR values correctly."""
        result = validator.validate_atr(1.5)
        assert result.is_valid == True
        
        result = validator.validate_atr(0)
        assert result.is_valid == False
    
    def test_validate_ratio(self, validator):
        """Should validate ratio/percentage values correctly."""
        result = validator.validate_ratio(50.0)
        assert result.is_valid == True
        
        result = validator.validate_ratio(0)
        assert result.is_valid == True
        
        result = validator.validate_ratio(150.0, max_val=100.0)
        assert result.is_valid == True
        assert result.value == 100.0
    
    def test_reset_warnings(self, validator):
        """Should reset warnings correctly."""
        validator.validate(None, "test")
        assert len(validator.get_warnings()) > 0
        
        validator.reset_warnings()
        assert len(validator.get_warnings()) == 0
    
    def test_rejected_count(self, validator):
        """Should track rejected values count."""
        validator.validate(None, "test1")
        validator.validate(float('nan'), "test2")
        assert validator.get_rejected_count() == 2


class TestValidatePriceData:
    """Tests for validate_price_data function."""
    
    def test_validate_all_valid_prices(self):
        """Should validate all valid prices correctly."""
        prices = {
            'close': 2650.0,
            'open': 2645.0,
            'high': 2655.0,
            'low': 2640.0
        }
        all_valid, cleaned, warnings = validate_price_data(prices)
        
        assert all_valid == True
        assert cleaned['close'] == 2650.0
        assert len(warnings) == 0
    
    def test_validate_with_invalid_price(self):
        """Should detect invalid prices."""
        prices = {
            'close': float('nan'),
            'open': 2645.0
        }
        all_valid, cleaned, warnings = validate_price_data(prices)
        
        assert all_valid == False
        assert 'close' not in cleaned
    
    def test_validate_atr_field(self):
        """Should validate ATR fields correctly."""
        prices = {
            'atr': 1.5,
            'close': 2650.0
        }
        all_valid, cleaned, warnings = validate_price_data(prices)
        
        assert all_valid == True
        assert cleaned['atr'] == 1.5
    
    def test_validate_rsi_field(self):
        """Should validate RSI fields correctly."""
        prices = {
            'rsi': 65.5,
            'close': 2650.0
        }
        all_valid, cleaned, warnings = validate_price_data(prices)
        
        assert all_valid == True
        assert cleaned['rsi'] == 65.5
    
    def test_validate_required_fields(self):
        """Should check required fields."""
        prices = {
            'close': 2650.0,
            'optional': None
        }
        all_valid, cleaned, warnings = validate_price_data(
            prices, 
            required_fields=['close']
        )
        
        assert all_valid == True


class TestIsValidNumber:
    """Tests for is_valid_number function."""
    
    def test_valid_integer(self):
        """Should return True for valid integers."""
        assert is_valid_number(100) == True
        assert is_valid_number(-50) == True
        assert is_valid_number(0) == True
    
    def test_valid_float(self):
        """Should return True for valid floats."""
        assert is_valid_number(100.5) == True
        assert is_valid_number(-50.25) == True
        assert is_valid_number(0.0) == True
    
    def test_none_value(self):
        """Should return False for None."""
        assert is_valid_number(None) == False
    
    def test_nan_value(self):
        """Should return False for NaN."""
        assert is_valid_number(float('nan')) == False
    
    def test_inf_value(self):
        """Should return False for Inf."""
        assert is_valid_number(float('inf')) == False
        assert is_valid_number(float('-inf')) == False
    
    def test_string_value(self):
        """Should return False for strings."""
        assert is_valid_number("100") == False
        assert is_valid_number("abc") == False
    
    def test_list_value(self):
        """Should return False for lists."""
        assert is_valid_number([100]) == False


class TestValidationResult:
    """Tests for ValidationResult dataclass."""
    
    def test_create_valid_result(self):
        """Should create valid result correctly."""
        result = ValidationResult(is_valid=True, value=100.0)
        assert result.is_valid == True
        assert result.value == 100.0
        assert result.error is None
        assert len(result.warnings) == 0
    
    def test_create_invalid_result(self):
        """Should create invalid result correctly."""
        result = ValidationResult(is_valid=False, value=0.0, error="Test error")
        assert result.is_valid == False
        assert result.error == "Test error"
    
    def test_add_warning(self):
        """Should add warnings correctly."""
        result = ValidationResult(is_valid=True, value=100.0)
        result.add_warning("Test warning")
        
        assert len(result.warnings) == 1
        assert "Test warning" in result.warnings
