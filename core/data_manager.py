#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data manager for the curve fitting application
"""

import numpy as np

class DataManager:
    """
    Class to handle data generation, import, and management for curve fitting
    """
    def __init__(self):
        """Initialize the data manager"""
        self.x_data = None
        self.y_data = None
        self.true_params = None
    
    def has_data(self):
        """Check if data is loaded"""
        return self.x_data is not None and self.y_data is not None
    
    def set_data(self, x, y):
        """Set data from external source"""
        self.x_data = np.array(x)
        self.y_data = np.array(y)
        self.true_params = None  # Reset true parameters since imported data doesn't have them
    
    def generate_synthetic_data(self, function_name, function, num_points=50, noise_level=0.05):
        """
        Generate synthetic data for a given function with random noise
        
        Parameters:
        -----------
        function_name : str
            Name of the function
        function : callable
            Function to generate data from
        num_points : int
            Number of data points to generate
        noise_level : float
            Level of noise to add to the data (relative to data range)
        
        Returns:
        --------
        x : ndarray
            x values
        y : ndarray
            y values with noise
        true_params : list
            True parameters used to generate the data
        """
        # Generate x values
        x = np.linspace(0, 10, num_points)
        
        # Determine true parameters based on function type
        if function_name == "Linear":
            # y = a*x + b
            true_params = [2.5, 1.0]  # a, b
        elif function_name == "Quadratic":
            # y = a*x^2 + b*x + c
            true_params = [0.5, 2.0, 1.0]  # a, b, c
        elif function_name == "Cubic":
            # y = a*x^3 + b*x^2 + c*x + d
            true_params = [0.1, 0.5, 2.0, 1.0]  # a, b, c, d
        elif function_name == "Power Law":
            # y = a*x^b + c
            true_params = [2.0, 0.5, 1.0]  # a, b, c
        elif function_name == "Exponential":
            # y = a*exp(b*x) + c
            true_params = [2.0, 0.5, 1.0]  # a, b, c
        elif function_name == "Double Power Law":
            # y = a*x^b + c*x^d
            true_params = [2.0, 0.5, 1.0, 1.5]  # a, b, c, d
        elif function_name == "Triple Power Law":
            # y = a*x^b + c*x^d + e*x^f
            true_params = [2.0, 0.5, 1.0, 1.5, 0.5, 2.0]  # a, b, c, d, e, f
        elif function_name.startswith("Custom"):
            # For custom functions, use default parameters
            true_params = [1.0] * 3  # Assume 3 parameters for custom function
        else:
            # Default case
            true_params = [1.0] * 3
        
        # Generate clean y values
        y_clean = function(x, *true_params)
        
        # Add noise
        y_range = np.max(y_clean) - np.min(y_clean)
        noise = np.random.normal(0, noise_level * y_range, size=num_points)
        y = y_clean + noise
        
        # Store data
        self.x_data = x
        self.y_data = y
        self.true_params = true_params
        
        return x, y, true_params