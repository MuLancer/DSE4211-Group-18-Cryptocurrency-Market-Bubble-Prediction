"""
Cryptocurrency Market Bubble Detection System

This package provides tools for detecting early signals of cryptocurrency market bubbles.
"""

__version__ = '0.1.0'
__author__ = 'DSE4211 Group 18'

from .data_fetcher import CryptoDataFetcher
from .bubble_detector import BubbleDetector
from .visualizer import BubbleVisualizer

__all__ = ['CryptoDataFetcher', 'BubbleDetector', 'BubbleVisualizer']
