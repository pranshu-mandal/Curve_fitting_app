#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration for curve fitting application
"""

# Application configuration
APP_NAME = "Advanced Curve Fitting Tool"
APP_VERSION = "1.0.0"
APP_AUTHOR = "Your Name"

# Default styling
STYLE = {
    "primary_color": "#3569b4",
    "secondary_color": "#eef0fd",
    "text_color": "#333333",
    "accent_color": "#e63946",
    "background_color": "#ffffff",
    "font_family": "Arial, sans-serif"
}

# Default plot configuration
PLOT_CONFIG = {
    "dpi": 100,
    "figsize": (8, 6),
    "data_color": "blue",
    "fit_color": "red",
    "grid": True,
    "grid_alpha": 0.3,
    "marker_size": 30
}

# Default number of points for synthetic data
DEFAULT_NUM_POINTS = 50

# Default noise level for synthetic data
DEFAULT_NOISE_LEVEL = 0.05