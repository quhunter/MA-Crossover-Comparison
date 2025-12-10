import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
import os
warnings.filterwarnings('ignore')

# Output directory - Windows format
OUTPUT_DIR = r"C:\Users\ugurc\Desktop\MAcross"

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

class MAComparison:
    def __init__(self, short_period=8, long_period=21):
        self.short_period = short_period
        self.long_period = long_period
        
    def generate_synthetic_data(self, regime: str, timeframe: str, days: int = 365) -> pd.DataFrame:
        """Generate synthetic price data for different market regimes"""
        if timeframe == '1d':
            periods = days
        else:  # 4h
            periods = days * 6  # 6 candles per day for 4h
        
        np.random.seed(None)  # Different seed each time
        
        # Base parameters
        initial_price = 100
        
        if regime == 'bull':
            drift = 0.0008  # ~0.08% daily growth
            volatility = 0.015  # Low volatility
        elif regime == 'bear':
            drift = -0.0006  # ~-0.06% daily decline
            volatility = 0.025  # Medium volatility
        else:  # sideways
            drift = 0.0001  # Minimal drift
            volatility = 0.035  # High volatility (whipsaw)
        
        # Generate geometric brownian motion
        returns = np.random.normal(drift, volatility, periods)
        price_series = initial_price * np.exp(np.cumsum(returns))
        
        # Generate volume (VWMA için)
        volume = np.random.lognormal(mean=10, sigma=0.5, size=periods)
        
        df = pd.DataFrame({
            'close': price_series,
            'volume': volume
        })
        
        return df
    
    def calculate_sma(self, data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_wma(self, data: pd.Series, period: int) -> pd.Series:
        """Weighted Moving Average"""
        weights = np.arange(1, period + 1)
        return data.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    def calculate_vwma(self, price: pd.Series, volume: pd.Series, period: int) -> pd.Series:
        """Volume Weighted Moving Average"""
        return (price * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
    
    def backtest_strategy(self, df: pd.DataFrame, ma_type: str) -> Dict:
        """Backtest crossover strategy for given MA type"""
        df = df.copy()
        
        # Calculate MAs based on type
        if ma_type == 'SMA':
            df['ma_short'] = self.calculate_sma(df['close'], self.short_period)
            df['ma_long'] = self.calculate_sma(df['close'], self.long_period)
        elif ma_type == 'EMA':
            df['ma_short'] = self.calculate_ema(df['close'], self.short_period)
            df['ma_long'] = self.calculate_ema(df['close'], self.long_period)
        elif ma_type == 'WMA':
            df['ma_short'] = self.calculate_wma(df['close'], self.short_period)
            df['ma_long'] = self.calculate_wma(df['close'], self.long_period)
        elif ma_type == 'VWMA':
            df['ma_short'] = self.calculate_vwma(df['close'], df['volume'], self.short_period)
            df['ma_long'] = self.calculate_vwma(df['close'], df['volume'], self.long_period)
        
        # Generate signals - SADECE CROSSOVER anında pozisyon aç/kapat
        df['signal'] = 0
        df['crossover'] = 0
        
        for i in range(1, len(df)):
            if (df['ma_short'].iloc[i-1] <= df['ma_long'].iloc[i-1] and 
                df['ma_short'].iloc[i] > df['ma_long'].iloc[i]):
                df.loc[df.index[i], 'crossover'] = 1  # Bullish crossover
            elif (df['ma_short'].iloc[i-1] >= df['ma_long'].iloc[i-1] and 
                  df['ma_short'].iloc[i] < df['ma_long'].iloc[i]):
                df.loc[df.index[i], 'crossover'] = -1  # Bearish crossover
        
        # Calculate trades
        trades = []
        entry_price = None
        entry_index = None
        
        for i in range(len(df)):
            if df['crossover'].iloc[i] == 1:  # Bullish crossover - BUY
                if entry_price is None:  # Only if not already in position
                    entry_price = df['close'].iloc[i]
                    entry_index = i
            elif df['crossover'].iloc[i] == -1 and entry_price is not None:  # Bearish crossover - SELL
                exit_price = df['close'].iloc[i]
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                duration = i - entry_index
                
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'duration': duration
                })
                entry_price = None
        
        # Calculate metrics
        if len(trades) == 0:
            return {
                'win_rate': 0,
                'avg_duration': 0,
                'avg_gain': 0,
                'max_drawdown': 0,
                'total_trades': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0
            }
        
        trades_df = pd.DataFrame(trades)
        wins = trades_df[trades_df['pnl_pct'] > 0]
        losses = trades_df[trades_df['pnl_pct'] < 0]
        
        win_rate = (len(wins) / len(trades_df)) * 100 if len(trades_df) > 0 else 0
        avg_duration = trades_df['duration'].mean()
        
        # Calculate cumulative returns for drawdown
        cumulative = (1 + trades_df['pnl_pct'] / 100).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = ((cumulative - running_max) / running_max) * 100
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
        
        # Profit factor
        total_wins = wins['pnl_pct'].sum() if len(wins) > 0 else 0
        total_losses = abs(losses['pnl_pct'].sum()) if len(losses) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        return {
            'win_rate': win_rate,
            'avg_duration': avg_duration,
            'avg_gain': trades_df['pnl_pct'].mean(),
            'max_drawdown': max_drawdown,
            'total_trades': len(trades_df),
            'profit_factor': profit_factor,
            'avg_win': wins['pnl_pct'].mean() if len(wins) > 0 else 0,
            'avg_loss': losses['pnl_pct'].mean() if len(losses) > 0 else 0
        }
    
    def run_monte_carlo(self, ma_type: str, regime: str, timeframe: str, 
                       simulations: int = 1000) -> List[Dict]:
        """Run Monte Carlo simulations"""
        results = []
        
        for i in range(simulations):
            if (i + 1) % 200 == 0:
                print(f"  {ma_type} - {regime} - {timeframe}: {i+1}/{simulations}")
            
            df = self.generate_synthetic_data(regime, timeframe)
            metrics = self.backtest_strategy(df, ma_type)
            results.append(metrics)
        
        return results
    
    def aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate Monte Carlo results"""
        df = pd.DataFrame(results)
        
        return {
            'win_rate_mean': df['win_rate'].mean(),
            'win_rate_std': df['win_rate'].std(),
            'avg_duration_mean': df['avg_duration'].mean(),
            'avg_gain_mean': df['avg_gain'].mean(),
            'avg_gain_std': df['avg_gain'].std(),
            'max_drawdown_mean': df['max_drawdown'].mean(),
            'max_drawdown_std': df['max_drawdown'].std(),
            'total_trades_mean': df['total_trades'].mean(),
            'profit_factor_mean': df['profit_factor'].mean(),
            'avg_win_mean': df['avg_win'].mean(),
            'avg_loss_mean': df['avg_loss'].mean()
        }


def create_comparison_charts(all_results, ma_types, regimes, timeframes):
    """Create comprehensive comparison charts"""
    
    # Prepare data for plotting
    comparison_data = []
    for timeframe in timeframes:
        for regime in regimes:
            for ma_type in ma_types:
                key = f"{timeframe}_{regime}_{ma_type}"
                result = all_results[key]
                comparison_data.append({
                    'timeframe': timeframe,
                    'regime': regime,
                    'ma_type': ma_type,
                    'win_rate': result['win_rate_mean'],
                    'avg_gain': result['avg_gain_mean'],
                    'max_drawdown': result['max_drawdown_mean'],
                    'profit_factor': result['profit_factor_mean'],
                    'avg_duration': result['avg_duration_mean'],
                    'total_trades': result['total_trades_mean']
                })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Create subplots with better spacing
    fig = plt.figure(figsize=(22, 18))
    
    metrics = ['win_rate', 'avg_gain', 'max_drawdown', 'profit_factor', 'avg_duration', 'total_trades']
    metric_titles = ['Win Rate (%)', 'Avg Gain (%)', 'Max Drawdown (%)', 
                     'Profit Factor', 'Avg Position Duration', 'Total Trades']
    
    for idx, (metric, title) in enumerate(zip(metrics, metric_titles), 1):
        for tf_idx, timeframe in enumerate(timeframes):
            subplot_idx = (idx - 1) * 2 + tf_idx + 1
            ax = plt.subplot(6, 2, subplot_idx)
            
            df_tf = df_comparison[df_comparison['timeframe'] == timeframe]
            
            # Pivot for grouped bar chart with explicit regime order
            pivot_data = df_tf.pivot(index='regime', columns='ma_type', values=metric)
            # Reindex to ensure correct order: bull, bear, sideways
            pivot_data = pivot_data.reindex(['bull', 'bear', 'sideways'])
            
            x = np.arange(len(regimes))
            width = 0.18  # Slightly narrower bars
            
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
            
            for i, ma_type in enumerate(ma_types):
                offset = (i - 1.5) * width
                bars = ax.bar(x + offset, pivot_data[ma_type], width, 
                             label=ma_type, color=colors[i], alpha=0.85, edgecolor='white', linewidth=0.5)
            
            ax.set_xlabel('Market Regime', fontsize=10, fontweight='bold')
            ax.set_ylabel(title, fontsize=10, fontweight='bold')
            ax.set_title(f'{title} - {timeframe}', fontsize=12, fontweight='bold', pad=10)
            ax.set_xticks(x)
            ax.set_xticklabels(['Bull', 'Bear', 'Sideways'], fontsize=9)
            ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add value labels on bars with better positioning
            for i, ma_type in enumerate(ma_types):
                offset = (i - 1.5) * width
                for j, regime in enumerate(['bull', 'bear', 'sideways']):
                    value = pivot_data.loc[regime, ma_type]
                    if not np.isnan(value):
                        # Position text slightly above bar
                        y_pos = value + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
                        ax.text(j + offset, y_pos, f'{value:.1f}', 
                               ha='center', va='bottom', fontsize=6.5, fontweight='bold')
    
    plt.tight_layout(pad=2.0, h_pad=2.5, w_pad=2.0)
    output_path = os.path.join(OUTPUT_DIR, 'ma_comparison_full.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 Full comparison chart saved: {output_path}")
    
    # Create detailed regime comparison with better spacing
    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    fig.suptitle('MA Performance by Market Regime (Detailed)', fontsize=18, fontweight='bold', y=0.995)
    
    for regime_idx, regime in enumerate(regimes):
        for tf_idx, timeframe in enumerate(timeframes):
            ax = axes[regime_idx, tf_idx]
            
            df_subset = df_comparison[
                (df_comparison['regime'] == regime) & 
                (df_comparison['timeframe'] == timeframe)
            ]
            
            # Create radar-like comparison
            metrics_normalized = ['win_rate', 'avg_gain', 'profit_factor']
            ma_data = []
            
            for ma_type in ma_types:
                row = df_subset[df_subset['ma_type'] == ma_type].iloc[0]
                ma_data.append([
                    row['win_rate'],
                    row['avg_gain'] if row['avg_gain'] > 0 else 0,
                    row['profit_factor']
                ])
            
            x = np.arange(len(metrics_normalized))
            width = 0.18
            
            for i, (ma_type, data) in enumerate(zip(ma_types, ma_data)):
                offset = (i - 1.5) * width
                bars = ax.bar(x + offset, data, width, label=ma_type, color=colors[i], 
                             alpha=0.85, edgecolor='white', linewidth=0.5)
            
            ax.set_xlabel('Metrics', fontweight='bold', fontsize=10)
            ax.set_ylabel('Value', fontweight='bold', fontsize=10)
            ax.set_title(f'{regime.upper()} - {timeframe}', fontweight='bold', fontsize=13, pad=10)
            ax.set_xticks(x)
            ax.set_xticklabels(['Win Rate', 'Avg Gain', 'P.Factor'], fontsize=9)
            ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    plt.tight_layout(pad=2.0, h_pad=2.5, w_pad=2.0)
    output_path = os.path.join(OUTPUT_DIR, 'ma_regime_detailed.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Regime detailed chart saved: {output_path}")
    
    # Create summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE - Best Performing MA by Metric and Regime")
    print("="*100)
    
    for timeframe in timeframes:
        print(f"\n{'='*50}")
        print(f"TIMEFRAME: {timeframe}")
        print(f"{'='*50}")
        
        for regime in regimes:
            print(f"\n🎯 {regime.upper()} Market:")
            print("-" * 50)
            
            df_subset = df_comparison[
                (df_comparison['regime'] == regime) & 
                (df_comparison['timeframe'] == timeframe)
            ].sort_values('win_rate', ascending=False)
            
            print(f"\n{'MA Type':<10} {'Win Rate':<12} {'Avg Gain':<12} {'Max DD':<12} {'P.Factor':<12} {'Trades':<10}")
            print("-" * 70)
            
            for _, row in df_subset.iterrows():
                print(f"{row['ma_type']:<10} {row['win_rate']:>10.2f}% {row['avg_gain']:>10.2f}% "
                      f"{row['max_drawdown']:>10.2f}% {row['profit_factor']:>10.2f} {row['total_trades']:>10.0f}")
            
            # Best performers
            best_wr = df_subset.iloc[0]
            best_gain = df_subset.sort_values('avg_gain', ascending=False).iloc[0]
            best_dd = df_subset.sort_values('max_drawdown').iloc[0]
            best_pf = df_subset.sort_values('profit_factor', ascending=False).iloc[0]
            
            print(f"\n🏆 Best Win Rate: {best_wr['ma_type']} ({best_wr['win_rate']:.2f}%)")
            print(f"💰 Best Avg Gain: {best_gain['ma_type']} ({best_gain['avg_gain']:.2f}%)")
            print(f"🛡️  Lowest Drawdown: {best_dd['ma_type']} ({best_dd['max_drawdown']:.2f}%)")
            print(f"📈 Best Profit Factor: {best_pf['ma_type']} ({best_pf['profit_factor']:.2f})")
    
    # Create heatmap for overall winner identification
    print("\n" + "="*100)
    print("OVERALL WINNERS - Cross-Regime Analysis")
    print("="*100)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('MA Performance Heatmaps', fontsize=18, fontweight='bold', y=0.995)
    
    for tf_idx, timeframe in enumerate(timeframes):
        # Win Rate Heatmap
        ax = axes[tf_idx, 0]
        pivot_wr = df_comparison[df_comparison['timeframe'] == timeframe].pivot(
            index='ma_type', columns='regime', values='win_rate'
        )
        # Reorder columns: bull, bear, sideways
        pivot_wr = pivot_wr[['bull', 'bear', 'sideways']]
        pivot_wr.columns = ['Bull', 'Bear', 'Sideways']
        
        sns.heatmap(pivot_wr, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax, 
                    cbar_kws={'label': 'Win Rate (%)'}, vmin=0, vmax=100,
                    annot_kws={'fontsize': 11, 'fontweight': 'bold'},
                    linewidths=1, linecolor='white')
        ax.set_title(f'Win Rate (%) - {timeframe}', fontweight='bold', fontsize=14, pad=10)
        ax.set_xlabel('Market Regime', fontweight='bold', fontsize=11)
        ax.set_ylabel('MA Type', fontweight='bold', fontsize=11)
        ax.tick_params(labelsize=10)
        
        # Profit Factor Heatmap
        ax = axes[tf_idx, 1]
        pivot_pf = df_comparison[df_comparison['timeframe'] == timeframe].pivot(
            index='ma_type', columns='regime', values='profit_factor'
        )
        # Reorder columns: bull, bear, sideways
        pivot_pf = pivot_pf[['bull', 'bear', 'sideways']]
        pivot_pf.columns = ['Bull', 'Bear', 'Sideways']
        
        sns.heatmap(pivot_pf, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax,
                    cbar_kws={'label': 'Profit Factor'}, vmin=0,
                    annot_kws={'fontsize': 11, 'fontweight': 'bold'},
                    linewidths=1, linecolor='white')
        ax.set_title(f'Profit Factor - {timeframe}', fontweight='bold', fontsize=14, pad=10)
        ax.set_xlabel('Market Regime', fontweight='bold', fontsize=11)
        ax.set_ylabel('MA Type', fontweight='bold', fontsize=11)
        ax.tick_params(labelsize=10)
    
    plt.tight_layout(pad=2.0, h_pad=2.5, w_pad=2.0)
    output_path = os.path.join(OUTPUT_DIR, 'ma_heatmaps.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n🔥 Heatmaps saved: {output_path}")
    
    return df_comparison


def main(simulations=100):
    """Main execution function
    
    Args:
        simulations: Number of Monte Carlo simulations per regime/timeframe/MA type
                    Default: 100 for quick testing
                    Recommended: 1000 for comprehensive analysis
    """
    print("🚀 MA Comparison Monte Carlo Analysis")
    print("=" * 60)
    print("Parameters:")
    print("  - Short MA: 8 periods")
    print("  - Long MA: 21 periods")
    print(f"  - Simulations: {simulations} per regime/timeframe/MA type")
    print(f"  - Total runs: {simulations * 24:,} simulations")
    print(f"  - Output directory: {OUTPUT_DIR}")
    print("=" * 60)
    
    ma_types = ['SMA', 'EMA', 'WMA', 'VWMA']
    regimes = ['bull', 'bear', 'sideways']
    timeframes = ['1d', '4h']
    
    comparison = MAComparison(short_period=8, long_period=21)
    all_results = {}
    
    for timeframe in timeframes:
        print(f"\n📊 Timeframe: {timeframe}")
        print("-" * 60)
        
        for regime in regimes:
            print(f"\n🎯 Regime: {regime.upper()}")
            
            for ma_type in ma_types:
                mc_results = comparison.run_monte_carlo(ma_type, regime, timeframe, simulations=simulations)
                agg_results = comparison.aggregate_results(mc_results)
                
                key = f"{timeframe}_{regime}_{ma_type}"
                all_results[key] = agg_results
    
    print("\n✅ Simulations completed! Creating visualizations...")
    
    # Create all visualizations and analysis
    df_comparison = create_comparison_charts(all_results, ma_types, regimes, timeframes)
    
    # Save detailed results to CSV
    csv_path = os.path.join(OUTPUT_DIR, 'ma_comparison_results.csv')
    df_comparison.to_csv(csv_path, index=False)
    print(f"\n💾 Detailed results saved: {csv_path}")
    
    print("\n" + "="*100)
    print("✅ ANALYSIS COMPLETE!")
    print("="*100)
    print("\nGenerated files in:", OUTPUT_DIR)
    print("  1. ma_comparison_full.png - Complete metric comparison")
    print("  2. ma_regime_detailed.png - Detailed regime analysis")
    print("  3. ma_heatmaps.png - Performance heatmaps")
    print("  4. ma_comparison_results.csv - Raw data export")
    print("\n" + "="*100)


if __name__ == "__main__":
    # Quick test mode: 100 simulations (takes ~2-3 minutes)
    # Full analysis mode: 1000 simulations (takes ~20-30 minutes)
    
    # Test mode - uncomment to test quickly
    main(simulations=1000)
    
    # Full mode - uncomment for comprehensive analysis
    # main(simulations=1000)
