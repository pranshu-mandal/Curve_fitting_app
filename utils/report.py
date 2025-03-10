#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report generation for curve fitting results
"""

import numpy as np
from scipy.stats import linregress

def generate_report(function_name, algorithm_name, params, true_params, result, x_data, y_data, function):
    """
    Generate a report on the fitting results
    
    Parameters:
    -----------
    function_name : str
        Name of the function used
    algorithm_name : str
        Name of the algorithm used
    params : ndarray
        Fitted parameters
    true_params : ndarray or None
        True parameters (if available)
    result : object
        Result object from the optimization
    x_data : ndarray
        x data points
    y_data : ndarray
        y data points
    function : callable
        Function that was fit
    
    Returns:
    --------
    str
        HTML-formatted report
    """
    # Calculate predicted values
    y_pred = function(x_data, *params)
    
    # Calculate R^2
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    ss_res = np.sum((y_data - y_pred) ** 2)
    
    if ss_tot == 0:
        r_squared = 1  # Avoid division by zero
    else:
        r_squared = 1 - (ss_res / ss_tot)
    
    # Calculate root mean squared error
    rmse = np.sqrt(np.mean((y_data - y_pred) ** 2))
    
    # Calculate mean absolute error
    mae = np.mean(np.abs(y_data - y_pred))
    
    # Start building the report
    report = """
    <html>
    <body style="font-family: Arial, sans-serif;">
    <h3 style="color: #3569b4;">Curve Fitting Report</h3>
    
    <div style="margin-bottom: 15px;">
        <b>Function:</b> {function_name}<br>
        <b>Algorithm:</b> {algorithm_name}<br>
    </div>
    
    <div style="margin-bottom: 15px;">
        <h4 style="color: #3569b4;">Fitted Parameters</h4>
        <table style="border-collapse: collapse; width: 100%;">
            <tr style="background-color: #f2f2f2;">
                <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Parameter</th>
                <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Value</th>
    """.format(function_name=function_name, algorithm_name=algorithm_name)
    
    # Add parameters to the report
    for i, param_value in enumerate(params):
        param_name = f"p{i}" if i < len(params) else f"p{i}"
        report += f"""
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">{param_name}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{param_value:.6f}</td>
            </tr>
        """
    
    report += """
        </table>
    </div>
    """
    
    # Add true parameters if available
    if true_params is not None:
        report += """
        <div style="margin-bottom: 15px;">
            <h4 style="color: #3569b4;">True Parameters (Synthetic Data)</h4>
            <table style="border-collapse: collapse; width: 100%;">
                <tr style="background-color: #f2f2f2;">
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Parameter</th>
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">True Value</th>
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Fitted Value</th>
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Difference</th>
                </tr>
        """
        
        for i, (true_val, fitted_val) in enumerate(zip(true_params, params)):
            param_name = f"p{i}" if i < len(true_params) else f"p{i}"
            diff = fitted_val - true_val
            report += f"""
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;">{param_name}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{true_val:.6f}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{fitted_val:.6f}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{diff:.6f}</td>
                </tr>
            """
        
        report += """
            </table>
        </div>
        """
    
    # Add fit quality metrics
    report += """
    <div style="margin-bottom: 15px;">
        <h4 style="color: #3569b4;">Fit Quality Metrics</h4>
        <table style="border-collapse: collapse; width: 100%;">
            <tr style="background-color: #f2f2f2;">
                <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Metric</th>
                <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Value</th>
                <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Interpretation</th>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">R² (Coefficient of Determination)</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{r2:.6f}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{r2_interp}</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">RMSE (Root Mean Square Error)</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{rmse:.6f}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">Average error magnitude</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">MAE (Mean Absolute Error)</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{mae:.6f}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">Average absolute error</td>
            </tr>
        </table>
    </div>
    """.format(
        r2=r_squared,
        r2_interp=interpret_r_squared(r_squared),
        rmse=rmse,
        mae=mae
    )
    
    # Add algorithm details if available
    if hasattr(result, 'message'):
        report += """
        <div style="margin-bottom: 15px;">
            <h4 style="color: #3569b4;">Algorithm Details</h4>
            <p><b>Status:</b> {message}</p>
        """.format(message=result.message)
        
        if hasattr(result, 'nfev'):
            report += f"<p><b>Function evaluations:</b> {result.nfev}</p>"
        
        if hasattr(result, 'nit'):
            report += f"<p><b>Iterations:</b> {result.nit}</p>"
        
        report += """
        </div>
        """
    
    report += """
    </body>
    </html>
    """
    
    return report

def interpret_r_squared(r_squared):
    """
    Interpret the R² value
    
    Parameters:
    -----------
    r_squared : float
        R² value
    
    Returns:
    --------
    str
        Interpretation
    """
    if r_squared > 0.95:
        return "Excellent fit"
    elif r_squared > 0.9:
        return "Very good fit"
    elif r_squared > 0.8:
        return "Good fit"
    elif r_squared > 0.6:
        return "Moderate fit"
    elif r_squared > 0.3:
        return "Poor fit"
    else:
        return "Very poor fit"