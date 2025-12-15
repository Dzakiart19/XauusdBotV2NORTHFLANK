"""
Backtesting Module untuk Bot Trading XAUUSD.

Modul ini menyediakan fitur backtesting untuk:
- Simulasi strategi dengan data historis
- Optimasi parameter
- Perbandingan performa strategi
- Analisis risiko
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import pytz
import random

from bot.logger import setup_logger
from bot.indicators import IndicatorEngine

logger = setup_logger('Backtesting')


class BacktestingError(Exception):
    """Exception untuk error pada backtesting"""
    pass


@dataclass
class BacktestTrade:
    """Representasi trade dalam backtest"""
    entry_time: datetime
    exit_time: Optional[datetime]
    signal_type: str
    entry_price: float
    exit_price: Optional[float]
    stop_loss: float
    take_profit: float
    result: Optional[str] = None
    pnl: float = 0.0
    pnl_pips: float = 0.0
    duration_minutes: int = 0
    exit_reason: str = ""


@dataclass
class BacktestResult:
    """Hasil backtest lengkap"""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_balance: float = 10000.0
    final_balance: float = 10000.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pips: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    avg_trade_duration: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'strategy_name': self.strategy_name,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'initial_balance': self.initial_balance,
            'final_balance': round(self.final_balance, 2),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': round(self.win_rate, 2),
            'total_pnl': round(self.total_pnl, 2),
            'total_pnl_pips': round(self.total_pnl_pips, 1),
            'max_drawdown': round(self.max_drawdown, 2),
            'max_drawdown_percent': round(self.max_drawdown_percent, 2),
            'profit_factor': round(self.profit_factor, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 2),
            'avg_trade_duration': round(self.avg_trade_duration, 1),
            'avg_win': round(self.avg_win, 2),
            'avg_loss': round(self.avg_loss, 2),
            'largest_win': round(self.largest_win, 2),
            'largest_loss': round(self.largest_loss, 2),
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'total_return_percent': round((self.final_balance - self.initial_balance) / self.initial_balance * 100, 2)
        }


class HistoricalDataProvider:
    """Provider untuk data historis"""
    
    def __init__(self, db_manager=None):
        self.db = db_manager
        self.cached_data: Dict[str, List[Dict]] = {}
        logger.info("HistoricalDataProvider initialized")
    
    def get_candles(self, timeframe: str, start_date: datetime, 
                    end_date: datetime) -> List[Dict[str, Any]]:
        """Get historical candle data
        
        Args:
            timeframe: Timeframe (M1, M5, H1, etc)
            start_date: Start datetime
            end_date: End datetime
        
        Returns:
            List of candle dictionaries
        """
        cache_key = f"{timeframe}_{start_date.date()}_{end_date.date()}"
        
        if cache_key in self.cached_data:
            return self.cached_data[cache_key]
        
        candles = self._fetch_from_database(timeframe, start_date, end_date)
        
        if not candles:
            candles = self._generate_synthetic_data(timeframe, start_date, end_date)
        
        self.cached_data[cache_key] = candles
        return candles
    
    def _fetch_from_database(self, timeframe: str, start_date: datetime,
                              end_date: datetime) -> List[Dict[str, Any]]:
        """Fetch data from database"""
        if not self.db:
            return []
        
        try:
            session = self.db.get_session()
            if not session:
                return []
            
            return []
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return []
    
    def _generate_synthetic_data(self, timeframe: str, start_date: datetime,
                                  end_date: datetime) -> List[Dict[str, Any]]:
        """Generate synthetic data for testing"""
        candles = []
        current = start_date
        
        if timeframe == 'M1':
            interval = timedelta(minutes=1)
        elif timeframe == 'M5':
            interval = timedelta(minutes=5)
        elif timeframe == 'H1':
            interval = timedelta(hours=1)
        else:
            interval = timedelta(minutes=1)
        
        base_price = 2650.00
        volatility = 0.001
        
        while current < end_date:
            change = random.gauss(0, volatility)
            open_price = base_price
            close_price = open_price * (1 + change)
            high_price = max(open_price, close_price) * (1 + abs(random.gauss(0, volatility/2)))
            low_price = min(open_price, close_price) * (1 - abs(random.gauss(0, volatility/2)))
            
            candles.append({
                'time': current,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': random.randint(100, 1000)
            })
            
            base_price = close_price
            current += interval
        
        logger.info(f"Generated {len(candles)} synthetic candles for {timeframe}")
        return candles


class BacktestEngine:
    """Engine utama untuk backtesting"""
    
    def __init__(self, config=None, indicator_engine: Optional[IndicatorEngine] = None):
        self.config = config
        self.indicator_engine = indicator_engine or IndicatorEngine(config)
        self.data_provider = HistoricalDataProvider()
        self.jakarta_tz = pytz.timezone('Asia/Jakarta')
        
        self.lot_size = 0.01
        self.pip_value = 10.0
        self.spread_pips = 2.0
        self.slippage_pips = 0.5
        
        logger.info("BacktestEngine initialized")
    
    def run_backtest(self, strategy_name: str, strategy_func: Callable,
                     start_date: datetime, end_date: datetime,
                     timeframe: str = 'M5',
                     initial_balance: float = 10000.0,
                     parameters: Optional[Dict[str, Any]] = None) -> BacktestResult:
        """Run backtest dengan strategi tertentu
        
        Args:
            strategy_name: Nama strategi
            strategy_func: Fungsi strategi yang return signal
            start_date: Tanggal mulai
            end_date: Tanggal selesai
            timeframe: Timeframe untuk analisis
            initial_balance: Modal awal
            parameters: Parameter strategi
        
        Returns:
            BacktestResult dengan statistik lengkap
        """
        try:
            logger.info(f"Starting backtest: {strategy_name} from {start_date} to {end_date}")
            
            candles = self.data_provider.get_candles(timeframe, start_date, end_date)
            
            if len(candles) < 50:
                logger.warning(f"Insufficient data for backtest: {len(candles)} candles")
                return BacktestResult(
                    strategy_name=strategy_name,
                    start_date=start_date,
                    end_date=end_date,
                    initial_balance=initial_balance
                )
            
            result = BacktestResult(
                strategy_name=strategy_name,
                start_date=start_date,
                end_date=end_date,
                initial_balance=initial_balance,
                final_balance=initial_balance
            )
            
            balance = initial_balance
            max_balance = initial_balance
            max_drawdown = 0.0
            equity_curve = [(start_date, initial_balance)]
            trades = []
            active_trade: Optional[BacktestTrade] = None
            
            for i in range(50, len(candles)):
                candle_window = candles[i-50:i+1]
                current_candle = candles[i]
                current_time = current_candle['time']
                current_price = current_candle['close']
                
                if active_trade:
                    exit_result = self._check_exit_conditions(
                        active_trade, current_candle, current_price
                    )
                    
                    if exit_result:
                        active_trade.exit_time = current_time
                        active_trade.exit_price = exit_result['exit_price']
                        active_trade.result = exit_result['result']
                        active_trade.exit_reason = exit_result['reason']
                        active_trade.pnl = exit_result['pnl']
                        active_trade.pnl_pips = exit_result['pnl_pips']
                        
                        if active_trade.entry_time:
                            duration = (current_time - active_trade.entry_time).total_seconds() / 60
                            active_trade.duration_minutes = int(duration)
                        
                        balance += active_trade.pnl
                        trades.append(active_trade)
                        active_trade = None
                        
                        equity_curve.append((current_time, balance))
                        
                        if balance > max_balance:
                            max_balance = balance
                        drawdown = max_balance - balance
                        if drawdown > max_drawdown:
                            max_drawdown = drawdown
                
                if not active_trade:
                    signal = strategy_func(candle_window, parameters)
                    
                    if signal and signal.get('signal_type') in ['BUY', 'SELL']:
                        entry_price = current_price + (self.spread_pips + self.slippage_pips) * 0.1
                        
                        active_trade = BacktestTrade(
                            entry_time=current_time,
                            exit_time=None,
                            signal_type=signal['signal_type'],
                            entry_price=entry_price,
                            exit_price=None,
                            stop_loss=signal.get('stop_loss', entry_price - 5),
                            take_profit=signal.get('take_profit', entry_price + 10)
                        )
            
            if active_trade:
                final_price = candles[-1]['close']
                pnl = self._calculate_pnl(
                    active_trade.signal_type,
                    active_trade.entry_price,
                    final_price
                )
                active_trade.exit_time = candles[-1]['time']
                active_trade.exit_price = final_price
                active_trade.result = 'WIN' if pnl > 0 else 'LOSS'
                active_trade.pnl = pnl
                active_trade.exit_reason = 'BACKTEST_END'
                balance += pnl
                trades.append(active_trade)
            
            result = self._calculate_statistics(
                result, trades, balance, max_drawdown, equity_curve
            )
            
            logger.info(f"Backtest completed: {len(trades)} trades, "
                       f"Win Rate: {result.win_rate:.1f}%, P/L: ${result.total_pnl:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Backtest error: {e}", exc_info=True)
            return BacktestResult(
                strategy_name=strategy_name,
                start_date=start_date,
                end_date=end_date,
                initial_balance=initial_balance
            )
    
    def _check_exit_conditions(self, trade: BacktestTrade, candle: Dict,
                                current_price: float) -> Optional[Dict[str, Any]]:
        """Check if trade should be closed"""
        high = candle['high']
        low = candle['low']
        
        if trade.signal_type == 'BUY':
            if low <= trade.stop_loss:
                pnl = self._calculate_pnl('BUY', trade.entry_price, trade.stop_loss)
                return {
                    'exit_price': trade.stop_loss,
                    'result': 'LOSS',
                    'reason': 'STOP_LOSS',
                    'pnl': pnl,
                    'pnl_pips': (trade.stop_loss - trade.entry_price) * 10
                }
            
            if high >= trade.take_profit:
                pnl = self._calculate_pnl('BUY', trade.entry_price, trade.take_profit)
                return {
                    'exit_price': trade.take_profit,
                    'result': 'WIN',
                    'reason': 'TAKE_PROFIT',
                    'pnl': pnl,
                    'pnl_pips': (trade.take_profit - trade.entry_price) * 10
                }
        
        else:
            if high >= trade.stop_loss:
                pnl = self._calculate_pnl('SELL', trade.entry_price, trade.stop_loss)
                return {
                    'exit_price': trade.stop_loss,
                    'result': 'LOSS',
                    'reason': 'STOP_LOSS',
                    'pnl': pnl,
                    'pnl_pips': (trade.entry_price - trade.stop_loss) * 10
                }
            
            if low <= trade.take_profit:
                pnl = self._calculate_pnl('SELL', trade.entry_price, trade.take_profit)
                return {
                    'exit_price': trade.take_profit,
                    'result': 'WIN',
                    'reason': 'TAKE_PROFIT',
                    'pnl': pnl,
                    'pnl_pips': (trade.entry_price - trade.take_profit) * 10
                }
        
        return None
    
    def _calculate_pnl(self, signal_type: str, entry: float, exit_price: float) -> float:
        """Calculate P/L in dollars"""
        if signal_type == 'BUY':
            pips = (exit_price - entry) * 10
        else:
            pips = (entry - exit_price) * 10
        
        return pips * self.lot_size * self.pip_value
    
    def _calculate_statistics(self, result: BacktestResult, trades: List[BacktestTrade],
                               final_balance: float, max_drawdown: float,
                               equity_curve: List[Tuple[datetime, float]]) -> BacktestResult:
        """Calculate comprehensive statistics"""
        result.trades = trades
        result.equity_curve = equity_curve
        result.final_balance = final_balance
        result.total_trades = len(trades)
        
        if not trades:
            return result
        
        wins = [t for t in trades if t.result == 'WIN']
        losses = [t for t in trades if t.result == 'LOSS']
        
        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        result.win_rate = (len(wins) / len(trades) * 100) if trades else 0
        
        result.total_pnl = sum(t.pnl for t in trades)
        result.total_pnl_pips = sum(t.pnl_pips for t in trades)
        
        result.max_drawdown = max_drawdown
        result.max_drawdown_percent = (max_drawdown / result.initial_balance * 100) if result.initial_balance > 0 else 0
        
        total_wins = sum(t.pnl for t in wins) if wins else 0
        total_losses = abs(sum(t.pnl for t in losses)) if losses else 0
        result.profit_factor = (total_wins / total_losses) if total_losses > 0 else 0
        
        result.avg_win = (total_wins / len(wins)) if wins else 0
        result.avg_loss = (total_losses / len(losses)) if losses else 0
        
        result.largest_win = max((t.pnl for t in wins), default=0)
        result.largest_loss = min((t.pnl for t in losses), default=0)
        
        durations = [t.duration_minutes for t in trades if t.duration_minutes > 0]
        result.avg_trade_duration = (sum(durations) / len(durations)) if durations else 0
        
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in trades:
            if trade.result == 'WIN':
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
        
        result.consecutive_wins = max_consecutive_wins
        result.consecutive_losses = max_consecutive_losses
        
        returns = []
        for i in range(1, len(equity_curve)):
            prev_balance = equity_curve[i-1][1]
            curr_balance = equity_curve[i][1]
            if prev_balance > 0:
                ret = (curr_balance - prev_balance) / prev_balance
                returns.append(ret)
        
        if returns:
            import statistics
            avg_return = statistics.mean(returns)
            std_return = statistics.stdev(returns) if len(returns) > 1 else 1
            risk_free_rate = 0.0001
            result.sharpe_ratio = ((avg_return - risk_free_rate) / std_return * (252 ** 0.5)) if std_return > 0 else 0
        
        return result


class StrategyOptimizer:
    """Optimizer untuk parameter strategi"""
    
    def __init__(self, backtest_engine: BacktestEngine):
        self.engine = backtest_engine
        logger.info("StrategyOptimizer initialized")
    
    def optimize(self, strategy_name: str, strategy_func: Callable,
                 parameter_ranges: Dict[str, List[Any]],
                 start_date: datetime, end_date: datetime,
                 optimization_target: str = 'profit_factor') -> Dict[str, Any]:
        """Optimize strategy parameters
        
        Args:
            strategy_name: Nama strategi
            strategy_func: Fungsi strategi
            parameter_ranges: Dict parameter -> list of values to test
            start_date: Tanggal mulai
            end_date: Tanggal selesai
            optimization_target: Metric untuk optimasi (profit_factor, win_rate, total_pnl)
        
        Returns:
            Dict dengan best parameters dan hasil
        """
        try:
            logger.info(f"Starting optimization for {strategy_name}")
            
            from itertools import product
            
            param_names = list(parameter_ranges.keys())
            param_values = list(parameter_ranges.values())
            combinations = list(product(*param_values))
            
            logger.info(f"Testing {len(combinations)} parameter combinations")
            
            results = []
            
            for combo in combinations:
                params = dict(zip(param_names, combo))
                
                result = self.engine.run_backtest(
                    strategy_name=f"{strategy_name}_{combo}",
                    strategy_func=strategy_func,
                    start_date=start_date,
                    end_date=end_date,
                    parameters=params
                )
                
                results.append({
                    'parameters': params,
                    'result': result.to_dict()
                })
            
            results.sort(
                key=lambda x: x['result'].get(optimization_target, 0),
                reverse=True
            )
            
            best = results[0] if results else None
            
            return {
                'strategy_name': strategy_name,
                'optimization_target': optimization_target,
                'combinations_tested': len(combinations),
                'best_parameters': best['parameters'] if best else {},
                'best_result': best['result'] if best else {},
                'all_results': results[:10]
            }
            
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return {'error': str(e)}


def default_strategy(candles: List[Dict], parameters: Optional[Dict] = None) -> Optional[Dict]:
    """Default strategy untuk testing
    
    Simple EMA crossover strategy
    """
    if len(candles) < 20:
        return None
    
    params = parameters or {}
    fast_period = params.get('fast_ema', 5)
    slow_period = params.get('slow_ema', 20)
    
    closes = [c['close'] for c in candles]
    
    def ema(data, period):
        if len(data) < period:
            return data[-1] if data else 0
        k = 2 / (period + 1)
        ema_val = sum(data[:period]) / period
        for price in data[period:]:
            ema_val = price * k + ema_val * (1 - k)
        return ema_val
    
    fast_ema = ema(closes, fast_period)
    slow_ema = ema(closes, slow_period)
    
    prev_closes = closes[:-1]
    prev_fast = ema(prev_closes, fast_period)
    prev_slow = ema(prev_closes, slow_period)
    
    current_price = closes[-1]
    
    if prev_fast <= prev_slow and fast_ema > slow_ema:
        return {
            'signal_type': 'BUY',
            'entry_price': current_price,
            'stop_loss': current_price - 3,
            'take_profit': current_price + 6
        }
    
    if prev_fast >= prev_slow and fast_ema < slow_ema:
        return {
            'signal_type': 'SELL',
            'entry_price': current_price,
            'stop_loss': current_price + 3,
            'take_profit': current_price - 6
        }
    
    return None
