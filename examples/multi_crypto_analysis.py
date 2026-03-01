"""
Example: Multi-Cryptocurrency Analysis

This script demonstrates bubble detection across multiple cryptocurrencies.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_fetcher import CryptoDataFetcher
from src.bubble_detector import BubbleDetector
import pandas as pd


def analyze_cryptocurrency(symbol, period='1y'):
    """
    Analyze a single cryptocurrency for bubble signals.
    
    Args:
        symbol (str): Cryptocurrency ticker symbol
        period (str): Time period for analysis
    
    Returns:
        dict: Analysis results
    """
    try:
        print(f"\nAnalyzing {symbol}...")
        
        # Fetch and prepare data
        fetcher = CryptoDataFetcher(symbol=symbol)
        data = fetcher.fetch_data(period=period)
        data = fetcher.calculate_features()
        
        # Detect bubbles
        detector = BubbleDetector(data)
        detector.calculate_composite_score()
        
        # Get current risk
        risk = detector.get_current_bubble_risk()
        
        # Identify bubble periods
        bubble_periods = detector.identify_bubble_periods(threshold=0.6)
        
        return {
            'symbol': symbol,
            'current_price': risk['current_price'],
            'composite_score': risk['composite_score'],
            'risk_level': risk['risk_level'],
            'num_bubbles': len(bubble_periods),
            'data_points': len(data),
            'success': True
        }
    
    except Exception as e:
        print(f"Error analyzing {symbol}: {str(e)}")
        return {
            'symbol': symbol,
            'success': False,
            'error': str(e)
        }


def main():
    """Run multi-cryptocurrency bubble analysis."""
    
    print("=" * 70)
    print("Multi-Cryptocurrency Bubble Detection Analysis")
    print("=" * 70)
    
    # List of cryptocurrencies to analyze
    cryptocurrencies = [
        'BTC-USD',   # Bitcoin
        'ETH-USD',   # Ethereum
        'BNB-USD',   # Binance Coin
        'ADA-USD',   # Cardano
        'SOL-USD',   # Solana
    ]
    
    print(f"\nAnalyzing {len(cryptocurrencies)} cryptocurrencies...")
    
    # Analyze each cryptocurrency
    results = []
    for crypto in cryptocurrencies:
        result = analyze_cryptocurrency(crypto, period='1y')
        results.append(result)
    
    # Create summary DataFrame
    successful_results = [r for r in results if r.get('success')]
    
    if successful_results:
        summary_df = pd.DataFrame(successful_results)
        
        print("\n" + "=" * 70)
        print("BUBBLE RISK SUMMARY")
        print("=" * 70)
        print()
        print(summary_df[['symbol', 'current_price', 'composite_score', 
                         'risk_level', 'num_bubbles']].to_string(index=False))
        print()
        
        # Identify highest risk cryptocurrencies
        high_risk = summary_df[summary_df['composite_score'] >= 0.6]
        if not high_risk.empty:
            print("⚠️  HIGH RISK CRYPTOCURRENCIES:")
            for _, row in high_risk.iterrows():
                print(f"  - {row['symbol']}: Score {row['composite_score']:.3f} ({row['risk_level']})")
        else:
            print("✓ No cryptocurrencies currently show high bubble risk.")
        
        print()
        
        # Calculate statistics
        avg_score = summary_df['composite_score'].mean()
        print(f"Average Bubble Score: {avg_score:.3f}")
        print(f"Highest Risk: {summary_df.loc[summary_df['composite_score'].idxmax(), 'symbol']}")
        print(f"Lowest Risk: {summary_df.loc[summary_df['composite_score'].idxmin(), 'symbol']}")
    
    # Report failures
    failed_results = [r for r in results if not r.get('success')]
    if failed_results:
        print("\n" + "=" * 70)
        print("FAILED ANALYSES:")
        for result in failed_results:
            print(f"  - {result['symbol']}: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 70)
    print("Multi-cryptocurrency analysis complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
