#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fitting algorithms for curve fitting application
"""

import numpy as np
from scipy import optimize

class FittingAlgorithms:
    """
    Class to handle different curve fitting algorithms
    """
    def __init__(self):
        """Initialize fitting algorithms"""
        self.algorithms = {
            "Differential Evolution": self.differential_evolution,
            "Basin Hopping": self.basin_hopping,
            "SHGO": self.shgo,
            "Dual Annealing": self.dual_annealing,
            "Least Squares": self.least_squares
        }
    
    def run_fitting(self, algorithm_name, function, x_data, y_data, params=None):
        """
        Run a fitting algorithm
        
        Parameters:
        -----------
        algorithm_name : str
            Name of the algorithm to run
        function : callable
            Function to fit
        x_data : ndarray
            x data points
        y_data : ndarray
            y data points
        params : dict
            Additional parameters for the algorithm
        
        Returns:
        --------
        ndarray
            Fitted parameters
        object
            Result object from the optimization
        """
        if algorithm_name not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        if params is None:
            params = {}
        
        # Run the algorithm
        return self.algorithms[algorithm_name](function, x_data, y_data, params)
    
    def residual_func(self, function, x_data, y_data):
        """
        Create a residual function for optimization
        
        Parameters:
        -----------
        function : callable
            Function to fit
        x_data : ndarray
            x data points
        y_data : ndarray
            y data points
        
        Returns:
        --------
        callable
            Residual function that returns sum of squared residuals
        """
        def residual(params):
            y_pred = function(x_data, *params)
            return np.sum((y_data - y_pred) ** 2)
        
        return residual
    
    def residual_array(self, function, x_data, y_data):
        """
        Create a residual function that returns array of residuals
        
        Parameters:
        -----------
        function : callable
            Function to fit
        x_data : ndarray
            x data points
        y_data : ndarray
            y data points
        
        Returns:
        --------
        callable
            Residual function that returns array of residuals
        """
        def residual(params):
            y_pred = function(x_data, *params)
            return y_data - y_pred
        
        return residual
    
    # Individual fitting algorithms
    
    def differential_evolution(self, function, x_data, y_data, params):
        """
        Differential Evolution optimization
        
        Parameters:
        -----------
        function : callable
            Function to fit
        x_data : ndarray
            x data points
        y_data : ndarray
            y data points
        params : dict
            Additional parameters for the algorithm
        
        Returns:
        --------
        ndarray
            Fitted parameters
        object
            Result object from the optimization
        """
        # Parse bounds
        bounds = params.get('bounds', [(0, 10)] * 3)  # Default: 3 parameters, bounds [0, 10]
        
        # Get other parameters
        strategy = params.get('strategy', 'best1bin')
        popsize = params.get('popsize', 15)
        tol = params.get('tol', 0.01)
        mutation = params.get('mutation', 0.8)
        recombination = params.get('recombination', 0.7)
        maxiter = params.get('maxiter', 1000)
        
        # Create residual function
        residual = self.residual_func(function, x_data, y_data)
        
        # Run optimization
        result = optimize.differential_evolution(
            residual,
            bounds,
            strategy=strategy,
            popsize=popsize,
            tol=tol,
            mutation=mutation,
            recombination=recombination,
            maxiter=maxiter
        )
        
        return result.x, result
    
    def basin_hopping(self, function, x_data, y_data, params):
        """
        Basin Hopping optimization
        
        Parameters:
        -----------
        function : callable
            Function to fit
        x_data : ndarray
            x data points
        y_data : ndarray
            y data points
        params : dict
            Additional parameters for the algorithm
        
        Returns:
        --------
        ndarray
            Fitted parameters
        object
            Result object from the optimization
        """
        # Get initial guess
        x0 = params.get('x0', [1.0, 1.0, 1.0])  # Default: 3 parameters, initial guess [1, 1, 1]
        
        # Get other parameters
        niter = params.get('niter', 100)
        T = params.get('T', 1.0)
        stepsize = params.get('stepsize', 0.5)
        
        # Create residual function
        residual = self.residual_func(function, x_data, y_data)
        
        # Run optimization
        result = optimize.basinhopping(
            residual,
            x0,
            niter=niter,
            T=T,
            stepsize=stepsize
        )
        
        return result.x, result
    
    def shgo(self, function, x_data, y_data, params):
        """
        SHGO optimization
        
        Parameters:
        -----------
        function : callable
            Function to fit
        x_data : ndarray
            x data points
        y_data : ndarray
            y data points
        params : dict
            Additional parameters for the algorithm
        
        Returns:
        --------
        ndarray
            Fitted parameters
        object
            Result object from the optimization
        """
        # Parse bounds
        bounds = params.get('bounds', [(0, 10)] * 3)  # Default: 3 parameters, bounds [0, 10]
        
        # Get other parameters
        n = params.get('n', 100)
        iters = params.get('iters', 1)
        
        # Create residual function
        residual = self.residual_func(function, x_data, y_data)
        
        # Run optimization
        result = optimize.shgo(
            residual,
            bounds,
            n=n,
            iters=iters
        )
        
        return result.x, result
    
    def dual_annealing(self, function, x_data, y_data, params):
        """
        Dual Annealing optimization
        
        Parameters:
        -----------
        function : callable
            Function to fit
        x_data : ndarray
            x data points
        y_data : ndarray
            y data points
        params : dict
            Additional parameters for the algorithm
        
        Returns:
        --------
        ndarray
            Fitted parameters
        object
            Result object from the optimization
        """
        # Parse bounds
        bounds = params.get('bounds', [(0, 10)] * 3)  # Default: 3 parameters, bounds [0, 10]
        
        # Get other parameters
        maxiter = params.get('maxiter', 1000)
        initial_temp = params.get('initial_temp', 5230.0)
        restart_temp_ratio = params.get('restart_temp_ratio', 2.0e-5)
        
        # Create residual function
        residual = self.residual_func(function, x_data, y_data)
        
        # Run optimization
        result = optimize.dual_annealing(
            residual,
            bounds,
            maxiter=maxiter,
            initial_temp=initial_temp,
            restart_temp_ratio=restart_temp_ratio
        )
        
        return result.x, result
    
    def least_squares(self, function, x_data, y_data, params):
        """
        Least Squares optimization
        
        Parameters:
        -----------
        function : callable
            Function to fit
        x_data : ndarray
            x data points
        y_data : ndarray
            y data points
        params : dict
            Additional parameters for the algorithm
        
        Returns:
        --------
        ndarray
            Fitted parameters
        object
            Result object from the optimization
        """
        # Get initial guess
        x0 = params.get('x0', [1.0, 1.0, 1.0])  # Default: 3 parameters, initial guess [1, 1, 1]
        
        # Get other parameters
        method = params.get('method', 'trf')
        bounds = params.get('bounds', (-np.inf, np.inf))
        ftol = params.get('ftol', 1e-8)
        xtol = params.get('xtol', 1e-8)
        gtol = params.get('gtol', 1e-8)
        
        # Create residual function that returns array of residuals
        residual = self.residual_array(function, x_data, y_data)
        
        # Run optimization
        result = optimize.least_squares(
            residual,
            x0,
            method=method,
            bounds=bounds,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol
        )
        
        return result.x, result