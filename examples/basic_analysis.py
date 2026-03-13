"""
Example: Basic Bubble Detection

This script demonstrates basic usage of the cryptocurrency bubble detection system.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_fetcher import CryptoDataFetcher
from src.bubble_detector import BubbleDetector
from src.visualizer import BubbleVisualizer


def main():
    """Run basic bubble detection analysis."""
    
    print("=" * 70)
    print("Cryptocurrency Bubble Detection - Basic Example")
    print("=" * 70)
    print()
    
    # Step 1: Fetch data
    print("Step 1: Fetching cryptocurrency data...")
    fetcher = CryptoDataFetcher(symbol='BTC-USD')
    data = fetcher.fetch_data(period='2y')  # Fetch 2 years of data
    print(f"Fetched data from {data.index[0]} to {data.index[-1]}")
    print()
    
    # Step 2: Calculate features
    print("Step 2: Calculating technical indicators and features...")
    data = fetcher.calculate_features()
    print(f"Calculated {len(data.columns)} features")
    print()
    
    # Step 3: Detect bubbles
    print("Step 3: Running bubble detection algorithms...")
    detector = BubbleDetector(data)
    
    # Run individual detection methods
    detector.detect_volatility_bubble()
    detector.detect_momentum_bubble()
    detector.detect_overextension_bubble()
    detector.detect_volume_bubble()
    detector.detect_rsi_bubble()
    
    # Calculate composite score
    composite_score = detector.calculate_composite_score()
    print(f"Average bubble score: {composite_score.mean():.3f}")
    print()
    
    # Step 4: Identify bubble periods
    print("Step 4: Identifying bubble periods...")
    bubble_periods = detector.identify_bubble_periods(threshold=0.6, min_duration=5)
    
    if bubble_periods:
        print(f"Found {len(bubble_periods)} bubble period(s):")
        for i, period in enumerate(bubble_periods, 1):
            print(f"\nBubble Period {i}:")
            print(f"  Start: {period['start_date'].strftime('%Y-%m-%d')}")
            print(f"  End: {period['end_date'].strftime('%Y-%m-%d')}")
            print(f"  Duration: {period['duration_days']} days")
            print(f"  Peak Date: {period['peak_date'].strftime('%Y-%m-%d')}")
            print(f"  Peak Price: ${period['peak_price']:,.2f}")
            print(f"  Max Score: {period['max_score']:.3f}")
            if period.get('ongoing'):
                print("  Status: ONGOING")
    else:
        print("No significant bubble periods detected in the analyzed timeframe.")
    print()
    
    # Step 5: Current risk assessment
    print("Step 5: Current bubble risk assessment...")
    risk = detector.get_current_bubble_risk()
    print(f"Date: {risk['date'].strftime('%Y-%m-%d')}")
    print(f"Current Price: ${risk['current_price']:,.2f}")
    print(f"Composite Score: {risk['composite_score']:.3f}")
    print(f"Risk Level: {risk['risk_level']}")
    print(f"Interpretation: {risk['interpretation']}")
    print()
    
    print("Individual Signal Scores:")
    for signal, value in risk['individual_signals'].items():
        if signal != 'composite_score':
            print(f"  {signal.replace('_', ' ').title()}: {value:.3f}")
    print()
    
    # Step 6: Create visualizations
    print("Step 6: Creating visualizations...")
    visualizer = BubbleVisualizer(data, detector.get_signals())
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Generate plots
    print("  - Generating price chart with signals...")
    visualizer.plot_price_with_signals(save_path='output/price_with_signals.png')
    
    print("  - Generating individual signals plot...")
    visualizer.plot_individual_signals(save_path='output/individual_signals.png')
    
    print("  - Generating volatility analysis...")
    visualizer.plot_volatility_analysis(save_path='output/volatility_analysis.png')
    
    if bubble_periods:
        print("  - Generating bubble periods chart...")
        visualizer.plot_bubble_periods(bubble_periods, save_path='output/bubble_periods.png')
    
    print("  - Generating correlation heatmap...")
    visualizer.plot_correlation_heatmap(save_path='output/correlation_heatmap.png')
    
    print("  - Generating dashboard...")
    visualizer.create_dashboard(bubble_periods, save_path='output/dashboard.png')
    
    print()
    print("=" * 70)
    print("Analysis complete! Check the 'output' directory for visualizations.")
    print("=" * 70)


if __name__ == '__main__':
    main()
