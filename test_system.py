"""
Simple test script to verify the bubble detection system works correctly.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from src.data_fetcher import CryptoDataFetcher
from src.bubble_detector import BubbleDetector
from src.visualizer import BubbleVisualizer
import pandas as pd
import numpy as np


def test_data_fetcher():
    """Test the data fetcher module."""
    print("Testing CryptoDataFetcher...")
    
    fetcher = CryptoDataFetcher(symbol='BTC-USD')
    assert fetcher.symbol == 'BTC-USD'
    
    # Test with a short period to speed up testing
    data = fetcher.fetch_data(period='1mo')
    assert data is not None
    assert len(data) > 0
    assert 'Close' in data.columns
    assert 'Volume' in data.columns
    
    # Test feature calculation
    data = fetcher.calculate_features()
    assert 'Returns' in data.columns
    assert 'MA_7' in data.columns
    assert 'MA_30' in data.columns
    assert 'Volatility_7' in data.columns
    assert 'RSI' in data.columns
    
    print("✓ CryptoDataFetcher tests passed")
    return data


def test_bubble_detector(data):
    """Test the bubble detector module."""
    print("\nTesting BubbleDetector...")
    
    detector = BubbleDetector(data)
    
    # Test individual signal detection
    vol_signal = detector.detect_volatility_bubble()
    assert len(vol_signal) == len(data)
    assert (vol_signal >= 0).all() and (vol_signal <= 1).all()
    
    mom_signal = detector.detect_momentum_bubble()
    assert len(mom_signal) == len(data)
    
    overext_signal = detector.detect_overextension_bubble()
    assert len(overext_signal) == len(data)
    
    vol_signal = detector.detect_volume_bubble()
    assert len(vol_signal) == len(data)
    
    rsi_signal = detector.detect_rsi_bubble()
    assert len(rsi_signal) == len(data)
    
    # Test composite score
    composite = detector.calculate_composite_score()
    assert len(composite) == len(data)
    assert (composite >= 0).all() and (composite <= 1).all()
    
    # Test bubble period identification
    periods = detector.identify_bubble_periods(threshold=0.6)
    assert isinstance(periods, list)
    
    # Test current risk assessment
    risk = detector.get_current_bubble_risk()
    assert 'composite_score' in risk
    assert 'risk_level' in risk
    assert 'interpretation' in risk
    assert risk['risk_level'] in ['LOW', 'LOW-MODERATE', 'MODERATE', 'HIGH']
    
    print("✓ BubbleDetector tests passed")
    return detector


def test_visualizer(data, detector):
    """Test the visualizer module."""
    print("\nTesting BubbleVisualizer...")
    
    signals = detector.get_signals()
    visualizer = BubbleVisualizer(data, signals)
    
    assert visualizer.data is not None
    assert visualizer.signals is not None
    
    print("✓ BubbleVisualizer tests passed")
    return visualizer


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("Running Bubble Detection System Tests")
    print("=" * 70)
    print()
    
    try:
        # Test each module
        data = test_data_fetcher()
        detector = test_bubble_detector(data)
        visualizer = test_visualizer(data, detector)
        
        print()
        print("=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        return True
        
    except Exception as e:
        print()
        print("=" * 70)
        print(f"✗ TESTS FAILED: {str(e)}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
