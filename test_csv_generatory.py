#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV Generator for Double Power Law Function

This script generates synthetic data following a double power law function
and saves it to a CSV file with uniform absolute noise on linear scales.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def double_power_law(x, a, b, c, d):
    """
    Double power law function: a*x^b + c*x^d
    
    Parameters:
    -----------
    x : numpy.ndarray
        Independent variable
    a, b, c, d : float
        Function parameters
    
    Returns:
    --------
    numpy.ndarray
        Function values
    """
    return a * np.power(x, b) + c * np.power(x, d)

def generate_csv(filename, num_points=100, noise_level=0.05, 
                 params=(1.5, 2.3, 2.8, 0.7), x_range=(0.1, 15)):
    """
    Generate synthetic data and save to CSV
    
    Parameters:
    -----------
    filename : str
        Output CSV filename
    num_points : int
        Number of data points to generate
    noise_level : float
        Absolute noise level
    params : tuple
        (a, b, c, d) parameters for the double power law
    x_range : tuple
        (min_x, max_x) range for x values
    """
    # Generate x values (linearly spaced)
    x = np.linspace(x_range[0], x_range[1], num_points)
    
    # Calculate true y values
    y_true = double_power_law(x, *params)
    
    # Add uniform absolute noise
    noise = np.random.normal(0, noise_level, size=num_points)
    y = y_true + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': x,
        'y': y
    })
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
    
    # Create a plot for visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Data with noise')
    plt.plot(x, y_true, 'r-', label='True function')
    # Linear scales for both axes
    plt.title(f'Double Power Law: {params[0]}*x^{params[1]} + {params[2]}*x^{params[3]}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.2)
    plt.legend()
    
    # Save plot
    plot_filename = os.path.splitext(filename)[0] + '.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    
    return df

if __name__ == "__main__":
    # Parameters for the double power law
    a, b = 1.5, 2.3  # First term: a*x^b
    c, d = 2.8, 0.7  # Second term: c*x^d
    
    # Generate a single CSV file with uniform noise
    filename = "double_power_law_linear.csv"
    df = generate_csv(
        filename=filename,
        num_points=200,  # More data points for better fitting
        noise_level=10.0,  # Absolute noise level
        params=(a, b, c, d),
        x_range=(0.1, 15)
    )
    
    print("Done generating CSV file!")
    
    # Display the plot
    plt.show()