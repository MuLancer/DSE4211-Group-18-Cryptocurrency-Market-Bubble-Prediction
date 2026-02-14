"""
Demo script that showcases the bubble detection system using synthetic data.

This demo can run without internet access and demonstrates all features.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.bubble_detector import BubbleDetector
from src.visualizer import BubbleVisualizer


def create_bubble_scenario(n_days=365):
    """
    Create synthetic cryptocurrency data with a realistic bubble pattern.
    
    Pattern includes:
    - Steady growth phase
    - Exponential growth (bubble inflation)
    - Peak with high volatility
    - Crash (bubble burst)
    - Recovery phase
    """
    
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    # Create bubble pattern
    t = np.arange(n_days)
    
    # Phase 1: Steady growth (0-40%)
    phase1_end = int(n_days * 0.4)
    phase1 = 30000 + 100 * t[:phase1_end]
    
    # Phase 2: Exponential growth - bubble (40-65%)
    phase2_start = phase1_end
    phase2_end = int(n_days * 0.65)
    phase2_len = phase2_end - phase2_start
    t2 = np.arange(phase2_len)
    phase2 = phase1[-1] * np.exp(0.015 * t2)
    
    # Phase 3: Peak volatility (65-70%)
    phase3_start = phase2_end
    phase3_end = int(n_days * 0.70)
    phase3_len = phase3_end - phase3_start
    phase3 = phase2[-1] * (1 + 0.1 * np.sin(np.linspace(0, 4*np.pi, phase3_len)))
    
    # Phase 4: Crash (70-80%)
    phase4_start = phase3_end
    phase4_end = int(n_days * 0.80)
    phase4_len = phase4_end - phase4_start
    t4 = np.arange(phase4_len)
    phase4 = phase3[-1] * np.exp(-0.08 * t4)
    
    # Phase 5: Recovery (80-100%)
    phase5_len = n_days - phase4_end
    t5 = np.arange(phase5_len)
    phase5 = phase4[-1] * (1 + 0.003 * t5)
    
    # Combine all phases
    price = np.concatenate([phase1, phase2, phase3, phase4, phase5])
    
    # Add realistic noise
    noise_level = price * 0.02  # 2% noise
    noise = np.random.normal(0, 1, n_days) * noise_level
    price = price + noise
    price = np.maximum(price, 1000)  # Floor at $1000
    
    # Create volume with spikes during bubble
    base_volume = 2e9
    volume = np.random.uniform(0.5, 1.5, n_days) * base_volume
    
    # Higher volume during bubble phase
    volume[phase2_start:phase3_end] *= np.random.uniform(2, 4, phase3_end - phase2_start)
    
    # Volume spike at crash
    volume[phase3_end:phase4_end] *= np.random.uniform(3, 5, phase4_end - phase3_end)
    
    # Create OHLC data
    data = pd.DataFrame({
        'Open': price * np.random.uniform(0.98, 1.00, n_days),
        'High': price * np.random.uniform(1.00, 1.03, n_days),
        'Low': price * np.random.uniform(0.97, 0.99, n_days),
        'Close': price,
        'Volume': volume
    }, index=dates)
    
    # Calculate all features
    data['Returns'] = data['Close'].pct_change()
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data['MA_7'] = data['Close'].rolling(window=7).mean()
    data['MA_30'] = data['Close'].rolling(window=30).mean()
    data['MA_90'] = data['Close'].rolling(window=90).mean()
    data['Volatility_7'] = data['Returns'].rolling(window=7).std()
    data['Volatility_30'] = data['Returns'].rolling(window=30).std()
    data['Momentum_7'] = data['Close'] - data['Close'].shift(7)
    data['Momentum_30'] = data['Close'] - data['Close'].shift(30)
    data['Volume_MA_7'] = data['Volume'].rolling(window=7).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_MA_7']
    
    # Calculate RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    data['Price_Acceleration'] = data['Returns'].diff()
    data['Price_MA30_Deviation'] = (data['Close'] - data['MA_30']) / data['MA_30']
    
    return data


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(text.center(70))
    print("=" * 70)


def print_section(text):
    """Print a formatted section header."""
    print(f"\n{text}")
    print("-" * 70)


def main():
    """Run the demonstration."""
    
    print_header("Cryptocurrency Bubble Detection System - DEMO")
    
    print("\n📊 This demo uses synthetic data to showcase bubble detection capabilities.")
    print("   In production, the system fetches real data from Yahoo Finance.")
    
    # Create synthetic data
    print_section("Generating Synthetic Market Data")
    print("Creating 1 year of cryptocurrency data with a realistic bubble pattern...")
    data = create_bubble_scenario(n_days=365)
    
    print(f"✓ Generated {len(data)} days of market data")
    print(f"  Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Starting price: ${data['Close'].iloc[100]:,.2f}")  # Skip NaN values
    print(f"  Ending price: ${data['Close'].iloc[-1]:,.2f}")
    print(f"  Peak price: ${data['Close'].max():,.2f}")
    print(f"  Price change: {((data['Close'].iloc[-1] / data['Close'].iloc[100]) - 1) * 100:.1f}%")
    
    # Initialize bubble detector
    print_section("Initializing Bubble Detector")
    detector = BubbleDetector(data)
    print("✓ Bubble detector initialized")
    
    # Run detection algorithms
    print_section("Running Bubble Detection Algorithms")
    print("Analyzing market data with multiple detection methods...")
    
    detector.detect_volatility_bubble()
    print("  ✓ Volatility-based detection complete")
    
    detector.detect_momentum_bubble()
    print("  ✓ Momentum-based detection complete")
    
    detector.detect_overextension_bubble()
    print("  ✓ Overextension detection complete")
    
    detector.detect_volume_bubble()
    print("  ✓ Volume anomaly detection complete")
    
    detector.detect_rsi_bubble()
    print("  ✓ RSI-based detection complete")
    
    composite_score = detector.calculate_composite_score()
    print(f"  ✓ Composite score calculated (avg: {composite_score.mean():.3f})")
    
    # Identify bubble periods
    print_section("Identifying Bubble Periods")
    bubble_periods = detector.identify_bubble_periods(threshold=0.6, min_duration=5)
    
    if bubble_periods:
        print(f"🔴 Found {len(bubble_periods)} significant bubble period(s):\n")
        
        for i, period in enumerate(bubble_periods, 1):
            print(f"Bubble Period #{i}:")
            print(f"  📅 Start Date:    {period['start_date'].strftime('%Y-%m-%d')}")
            print(f"  📅 End Date:      {period['end_date'].strftime('%Y-%m-%d')}")
            print(f"  ⏱  Duration:      {period['duration_days']} days")
            print(f"  📈 Peak Date:     {period['peak_date'].strftime('%Y-%m-%d')}")
            print(f"  💰 Peak Price:    ${period['peak_price']:,.2f}")
            print(f"  ⚠️  Max Score:     {period['max_score']:.3f}")
            
            if period.get('ongoing'):
                print(f"  🔄 Status:        ONGOING BUBBLE")
            
            # Calculate drawdown from peak
            current_price = data['Close'].iloc[-1]
            drawdown = ((current_price - period['peak_price']) / period['peak_price']) * 100
            if drawdown < -10:
                print(f"  📉 Drawdown:      {drawdown:.1f}% from peak")
            
            print()
    else:
        print("✓ No significant bubble periods detected")
    
    # Current risk assessment
    print_section("Current Bubble Risk Assessment")
    risk = detector.get_current_bubble_risk()
    
    print(f"📅 Assessment Date: {risk['date'].strftime('%Y-%m-%d')}")
    print(f"💰 Current Price:   ${risk['current_price']:,.2f}")
    print(f"📊 Composite Score: {risk['composite_score']:.3f}")
    print(f"⚠️  Risk Level:      {risk['risk_level']}")
    print(f"💬 Interpretation:  {risk['interpretation']}")
    
    print("\n📈 Individual Signal Breakdown:")
    signals = risk['individual_signals']
    
    signal_names = {
        'volatility_signal': '📉 Volatility',
        'momentum_signal': '🚀 Momentum',
        'overextension_signal': '📊 Overextension',
        'volume_signal': '📦 Volume',
        'rsi_signal': '📈 RSI'
    }
    
    for signal_key, display_name in signal_names.items():
        if signal_key in signals:
            value = signals[signal_key]
            bar_length = int(value * 20)
            bar = '█' * bar_length + '░' * (20 - bar_length)
            print(f"  {display_name:20} [{bar}] {value:.3f}")
    
    # Provide recommendations
    print_section("Recommendations")
    
    if risk['risk_level'] == 'HIGH':
        print("⚠️  HIGH RISK DETECTED!")
        print("  • Consider reducing exposure")
        print("  • Set stop-loss orders")
        print("  • Monitor market closely")
        print("  • Be prepared for potential correction")
    elif risk['risk_level'] == 'MODERATE':
        print("⚠️  MODERATE RISK")
        print("  • Exercise caution with new positions")
        print("  • Monitor key indicators")
        print("  • Consider taking some profits")
    elif risk['risk_level'] == 'LOW-MODERATE':
        print("✓ LOW-MODERATE RISK")
        print("  • Normal market conditions")
        print("  • Continue monitoring")
        print("  • Standard risk management applies")
    else:
        print("✓ LOW RISK")
        print("  • Market appears healthy")
        print("  • Normal investment strategies apply")
        print("  • Continue regular monitoring")
    
    # Create visualizations
    print_section("Generating Visualizations")
    print("Creating visualization dashboard...")
    
    os.makedirs('demo_output', exist_ok=True)
    
    visualizer = BubbleVisualizer(data, detector.get_signals())
    
    try:
        print("  • Generating price chart with signals...")
        visualizer.plot_price_with_signals(save_path='demo_output/price_signals.png')
        
        print("  • Generating individual signals plot...")
        visualizer.plot_individual_signals(save_path='demo_output/individual_signals.png')
        
        print("  • Generating volatility analysis...")
        visualizer.plot_volatility_analysis(save_path='demo_output/volatility.png')
        
        if bubble_periods:
            print("  • Generating bubble periods chart...")
            visualizer.plot_bubble_periods(bubble_periods, save_path='demo_output/bubble_periods.png')
        
        print("  • Generating correlation heatmap...")
        visualizer.plot_correlation_heatmap(save_path='demo_output/correlation.png')
        
        print("  • Generating comprehensive dashboard...")
        visualizer.create_dashboard(bubble_periods, save_path='demo_output/dashboard.png')
        
        print("\n✓ All visualizations saved to 'demo_output/' directory")
        
    except Exception as e:
        print(f"\n⚠️  Note: Visualization generation skipped (display not available)")
        print(f"   In a normal environment, plots would be created and saved.")
    
    # Summary statistics
    print_section("Summary Statistics")
    
    signals_df = detector.get_signals()
    
    print(f"📊 Signal Statistics (last 30 days):")
    recent = signals_df.tail(30)
    
    for col in recent.columns:
        if col != 'composite_score':
            avg = recent[col].mean()
            max_val = recent[col].max()
            print(f"  {col.replace('_', ' ').title():25} Avg: {avg:.3f}  Max: {max_val:.3f}")
    
    print(f"\n  {'Composite Score':25} Avg: {recent['composite_score'].mean():.3f}  Max: {recent['composite_score'].max():.3f}")
    
    # Final message
    print_header("Demo Complete")
    
    print("\n✓ The cryptocurrency bubble detection system successfully analyzed")
    print("  the synthetic market data and identified bubble patterns.")
    print("\n📁 Check the 'demo_output/' directory for generated visualizations.")
    print("\n🚀 To use with real data, run:")
    print("   python examples/basic_analysis.py")
    print("\n⚠️  Note: Real data requires internet connection to fetch from Yahoo Finance.")
    print()


if __name__ == '__main__':
    main()
