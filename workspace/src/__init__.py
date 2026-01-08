"""
Replica Interactive Path Generator

A tool for generating high-quality RGB-D video datasets from Replica scenes.
"""

from .viewer_wrapper import KeyframeRecorder
from .generator import TrajectoryGenerator
from .renderer import DatasetRenderer

__all__ = ['KeyframeRecorder', 'TrajectoryGenerator', 'DatasetRenderer']
