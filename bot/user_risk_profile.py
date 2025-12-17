"""
Per-user Risk Profile Manager - Lightweight for Koyeb free tier.
Features: Configurable risk settings per user, risk levels, adaptive limits.
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from bot.logger import setup_logger

logger = setup_logger('UserRiskProfile')


class RiskLevel(str, Enum):
    CONSERVATIVE = 'conservative'
    MODERATE = 'moderate'
    AGGRESSIVE = 'aggressive'
    CUSTOM = 'custom'


@dataclass
class RiskProfile:
    """Per-user risk profile settings"""
    user_id: int
    risk_level: RiskLevel = RiskLevel.MODERATE
    
    max_daily_loss_percent: float = 0.0  # 0.0 = unlimited (no daily loss limit)
    max_daily_loss_amount: float = 0.0   # 0.0 = unlimited (no daily loss limit)
    max_concurrent_positions: int = 1
    risk_per_trade_percent: float = 2.0
    
    lot_size: float = 0.01
    max_lot_size: float = 0.1
    min_lot_size: float = 0.01
    
    trailing_stop_enabled: bool = True
    trailing_distance_pips: float = 0.5
    breakeven_enabled: bool = True
    breakeven_threshold: float = 0.5
    
    min_confidence_threshold: float = 50.0
    min_confluence_count: int = 2
    
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'risk_level': self.risk_level.value,
            'max_daily_loss_percent': self.max_daily_loss_percent,
            'max_daily_loss_amount': self.max_daily_loss_amount,
            'max_concurrent_positions': self.max_concurrent_positions,
            'risk_per_trade_percent': self.risk_per_trade_percent,
            'lot_size': self.lot_size,
            'max_lot_size': self.max_lot_size,
            'min_lot_size': self.min_lot_size,
            'trailing_stop_enabled': self.trailing_stop_enabled,
            'trailing_distance_pips': self.trailing_distance_pips,
            'breakeven_enabled': self.breakeven_enabled,
            'breakeven_threshold': self.breakeven_threshold,
            'min_confidence_threshold': self.min_confidence_threshold,
            'min_confluence_count': self.min_confluence_count
        }


PRESET_PROFILES = {
    RiskLevel.CONSERVATIVE: {
        'max_daily_loss_percent': 0.0,  # Unlimited
        'max_daily_loss_amount': 0.0,   # Unlimited
        'max_concurrent_positions': 1,
        'risk_per_trade_percent': 1.0,
        'lot_size': 0.01,
        'max_lot_size': 0.03,
        'trailing_distance_pips': 1.0,
        'breakeven_threshold': 0.3,
        'min_confidence_threshold': 60.0,
        'min_confluence_count': 3
    },
    RiskLevel.MODERATE: {
        'max_daily_loss_percent': 0.0,  # Unlimited
        'max_daily_loss_amount': 0.0,   # Unlimited
        'max_concurrent_positions': 1,
        'risk_per_trade_percent': 2.0,
        'lot_size': 0.01,
        'max_lot_size': 0.05,
        'trailing_distance_pips': 0.5,
        'breakeven_threshold': 0.5,
        'min_confidence_threshold': 50.0,
        'min_confluence_count': 2
    },
    RiskLevel.AGGRESSIVE: {
        'max_daily_loss_percent': 0.0,  # Unlimited
        'max_daily_loss_amount': 0.0,   # Unlimited
        'max_concurrent_positions': 1,
        'risk_per_trade_percent': 3.0,
        'lot_size': 0.02,
        'max_lot_size': 0.1,
        'trailing_distance_pips': 0.3,
        'breakeven_threshold': 0.5,
        'min_confidence_threshold': 45.0,
        'min_confluence_count': 2
    }
}


class UserRiskProfileManager:
    """
    Manages per-user risk profiles with:
    - Preset risk levels (Conservative, Moderate, Aggressive)
    - Custom settings per user
    - Adaptive adjustments based on performance
    """
    
    def __init__(self, config=None, db_manager=None):
        self.config = config
        self.db_manager = db_manager
        self._profiles: Dict[int, RiskProfile] = {}
        
        logger.info("UserRiskProfileManager initialized")
    
    def get_profile(self, user_id: int) -> RiskProfile:
        """Get or create risk profile for user"""
        if user_id not in self._profiles:
            self._profiles[user_id] = RiskProfile(user_id=user_id)
            logger.info(f"Created default MODERATE risk profile for user {user_id}")
        return self._profiles[user_id]
    
    def set_risk_level(self, user_id: int, level: RiskLevel) -> RiskProfile:
        """Set risk level for user using preset"""
        profile = self.get_profile(user_id)
        preset = PRESET_PROFILES.get(level, PRESET_PROFILES[RiskLevel.MODERATE])
        
        profile.risk_level = level
        for key, value in preset.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        
        profile.updated_at = datetime.now()
        self._profiles[user_id] = profile
        
        logger.info(f"Set risk level {level.value} for user {user_id}")
        return profile
    
    def update_setting(self, user_id: int, setting: str, value: Any) -> bool:
        """Update a specific setting for user"""
        profile = self.get_profile(user_id)
        
        if not hasattr(profile, setting):
            logger.warning(f"Unknown setting: {setting}")
            return False
        
        if setting == 'max_daily_loss_amount':
            value = max(1.0, min(100.0, float(value)))
        elif setting == 'max_concurrent_positions':
            value = max(1, min(10, int(value)))
        elif setting == 'lot_size':
            value = max(0.01, min(1.0, float(value)))
        elif setting == 'trailing_distance_pips':
            value = max(0.1, min(5.0, float(value)))
        elif setting == 'min_confidence_threshold':
            value = max(30.0, min(90.0, float(value)))
        
        setattr(profile, setting, value)
        profile.risk_level = RiskLevel.CUSTOM
        profile.updated_at = datetime.now()
        
        logger.info(f"Updated {setting}={value} for user {user_id}")
        return True
    
    def get_trading_parameters(self, user_id: int) -> Dict[str, Any]:
        """Get trading parameters based on user's risk profile"""
        profile = self.get_profile(user_id)
        
        return {
            'max_daily_loss': profile.max_daily_loss_amount,
            'max_concurrent_positions': profile.max_concurrent_positions,
            'lot_size': profile.lot_size,
            'risk_per_trade': profile.risk_per_trade_percent,
            'trailing_enabled': profile.trailing_stop_enabled,
            'trailing_distance': profile.trailing_distance_pips,
            'breakeven_enabled': profile.breakeven_enabled,
            'breakeven_threshold': profile.breakeven_threshold,
            'min_confidence': profile.min_confidence_threshold,
            'min_confluence': profile.min_confluence_count
        }
    
    def should_block_signal(self, user_id: int, signal_data: Dict[str, Any]) -> tuple:
        """Check if signal should be blocked based on user's risk profile"""
        profile = self.get_profile(user_id)
        
        confidence = signal_data.get('confidence_score', signal_data.get('confidence', 0))
        if isinstance(confidence, str):
            try:
                confidence = float(confidence.replace('%', ''))
            except (ValueError, TypeError):
                confidence = 0
        
        if confidence < profile.min_confidence_threshold:
            return (True, f"Confidence {confidence:.1f}% below your minimum {profile.min_confidence_threshold:.1f}%")
        
        confluence = signal_data.get('confluence_count', 0)
        if confluence < profile.min_confluence_count:
            return (True, f"Confluence {confluence} below your minimum {profile.min_confluence_count}")
        
        return (False, None)
    
    def get_adaptive_lot_size(self, user_id: int, current_drawdown_percent: float = 0.0,
                               volatility_level: float = 0.5) -> float:
        """Get adaptive lot size based on drawdown and volatility"""
        profile = self.get_profile(user_id)
        base_lot = profile.lot_size
        
        if current_drawdown_percent > 40:
            lot_multiplier = 0.5
        elif current_drawdown_percent > 20:
            lot_multiplier = 0.75
        else:
            lot_multiplier = 1.0
        
        if volatility_level > 0.8:
            lot_multiplier *= 0.8
        elif volatility_level > 0.6:
            lot_multiplier *= 0.9
        
        adjusted_lot = base_lot * lot_multiplier
        adjusted_lot = max(profile.min_lot_size, min(adjusted_lot, profile.max_lot_size))
        
        return round(adjusted_lot, 2)
    
    def get_adaptive_trailing_distance(self, user_id: int, volatility_level: float = 0.5,
                                        atr_ratio: float = 1.0) -> float:
        """Get adaptive trailing distance based on volatility"""
        profile = self.get_profile(user_id)
        base_distance = profile.trailing_distance_pips
        
        if atr_ratio > 1.5:
            distance_multiplier = 1.5
        elif atr_ratio > 1.2:
            distance_multiplier = 1.3
        elif atr_ratio < 0.7:
            distance_multiplier = 0.8
        else:
            distance_multiplier = 1.0
        
        if volatility_level > 0.7:
            distance_multiplier *= 1.2
        
        adjusted_distance = base_distance * distance_multiplier
        adjusted_distance = max(0.2, min(adjusted_distance, 3.0))
        
        return round(adjusted_distance, 2)
    
    def get_profile_summary(self, user_id: int) -> str:
        """Get human-readable profile summary"""
        profile = self.get_profile(user_id)
        
        return f"""👤 Risk Profile for User {user_id}
━━━━━━━━━━━━━━━━━━━━━━━━
Risk Level: {profile.risk_level.value.title()}

💰 Capital Protection:
  • Max Daily Loss: ${profile.max_daily_loss_amount:.2f} ({profile.max_daily_loss_percent:.1f}%)
  • Max Concurrent: {profile.max_concurrent_positions} positions
  • Risk per Trade: {profile.risk_per_trade_percent:.1f}%

📊 Position Settings:
  • Lot Size: {profile.lot_size:.2f} (max {profile.max_lot_size:.2f})
  • Trailing Stop: {'Enabled' if profile.trailing_stop_enabled else 'Disabled'}
  • Trail Distance: {profile.trailing_distance_pips} pips
  • Breakeven: {'Enabled' if profile.breakeven_enabled else 'Disabled'} @ ${profile.breakeven_threshold}

🎯 Signal Filters:
  • Min Confidence: {profile.min_confidence_threshold:.0f}%
  • Min Confluence: {profile.min_confluence_count}"""
    
    def reset_to_default(self, user_id: int) -> RiskProfile:
        """Reset user's profile to default moderate settings"""
        return self.set_risk_level(user_id, RiskLevel.MODERATE)
    
    def get_all_profiles(self) -> Dict[int, Dict[str, Any]]:
        """Get all profiles for admin view"""
        return {uid: p.to_dict() for uid, p in self._profiles.items()}
