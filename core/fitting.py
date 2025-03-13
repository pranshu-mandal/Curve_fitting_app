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
        
        # Get the function name from the parameters
        function_name = params.get('function_name', 'Unknown')
        
        # Import needed modules for function info
        from core.function_models import FunctionModels
        function_models = FunctionModels()
        
        # Check for custom function manager
        from core.custom_function_manager import CustomFunctionManager
        custom_function_manager = CustomFunctionManager()
        
        # Important: Get function info to determine parameter count
        function_info = function_models.get_function_info(function_name)

        # If not found in built-in functions, check custom functions
        if not function_info:
            function_info = custom_function_manager.get_function_info(function_name)

        if not function_info:
            # Don't use QMessageBox here, just raise an exception
            raise ValueError(f"Function information for '{function_name}' not found.")
        
        # Store parameter count in params for use by algorithms
        params['param_count'] = function_info['param_count']
        
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
        # Get function info to determine parameter count
        function_name = function.__name__ if hasattr(function, "__name__") else "unknown"
        
        # Import here to avoid circular imports
        from core.function_models import FunctionModels
        function_models = FunctionModels()
        
        # Direct function matching with the function object itself
        expected_param_count = None
        for name, func in function_models.functions.items():
            if func.__code__.co_code == function.__code__.co_code:  # Compare function code objects
                function_info = function_models.get_function_info(name)
                expected_param_count = function_info['param_count']
                break
        
        def residual(params):
            try:
                # Convert params to numpy array and flatten
                params_array = np.asarray(params).flatten()
                
                # Ensure we don't pass too many parameters
                if expected_param_count is not None and len(params_array) > expected_param_count:
                    # Trim to expected count
                    params_array = params_array[:expected_param_count]
                
                # Call function with correct number of parameters
                y_pred = function(x_data, *params_array)
                return np.sum((y_data - y_pred) ** 2)
            except Exception as e:
                print(f"Error in residual calculation: {e}")  # Debug info
                return np.inf
        
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
        # Get function info to determine parameter count
        function_name = function.__name__ if hasattr(function, "__name__") else "unknown"
        
        # Import here to avoid circular imports
        from core.function_models import FunctionModels
        function_models = FunctionModels()
        
        # Direct function matching with the function object itself
        expected_param_count = None
        for name, func in function_models.functions.items():
            if func.__code__.co_code == function.__code__.co_code:  # Compare function code objects
                function_info = function_models.get_function_info(name)
                expected_param_count = function_info['param_count']
                break
        
        def residual(params):
            try:
                # Convert params to numpy array and flatten
                params_array = np.asarray(params).flatten()
                
                # Ensure we don't pass too many parameters
                if expected_param_count is not None and len(params_array) > expected_param_count:
                    # Trim to expected count
                    params_array = params_array[:expected_param_count]
                
                # Call function with correct number of parameters
                y_pred = function(x_data, *params_array)
                return y_data - y_pred
            except Exception as e:
                print(f"Error in residual array calculation: {e}")  # Debug info
                return np.full_like(y_data, np.inf)
        
        return residual
    
    # Individual fitting algorithms
    
    def differential_evolution(self, function, x_data, y_data, params):
        """
        Differential Evolution optimization
        """
        # Get function name to determine parameter count
        function_name = params.get('function_name', 'Unknown')
        
        # Get parameter count info
        from core.function_models import FunctionModels
        function_models = FunctionModels()
        function_info = function_models.get_function_info(function_name)
        param_count = function_info['param_count'] if function_info else 3
        
        # Parse bounds with the correct parameter count
        bounds = params.get('bounds', [(0, 10)] * param_count)
        
        # Ensure bounds has the correct length
        if len(bounds) != param_count:
            bounds = [(0, 10)] * param_count
        
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
        """
        # Get function name to determine parameter count
        function_name = params.get('function_name', 'Unknown')
        
        # Get parameter count info
        from core.function_models import FunctionModels
        function_models = FunctionModels()
        function_info = function_models.get_function_info(function_name)
        param_count = function_info['param_count'] if function_info else 3
        
        # Get initial guess with correct parameter count
        x0 = params.get('x0', [1.0] * param_count)
        
        # Ensure x0 has the correct length
        if len(x0) != param_count:
            x0 = [1.0] * param_count
        
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
        """
        # Get function name to determine parameter count
        function_name = params.get('function_name', 'Unknown')
        
        # Get parameter count info
        from core.function_models import FunctionModels
        function_models = FunctionModels()
        function_info = function_models.get_function_info(function_name)
        param_count = function_info['param_count'] if function_info else 3
        
        # Parse bounds with the correct parameter count
        bounds = params.get('bounds', [(0, 10)] * param_count)
        
        # Ensure bounds has the correct length
        if len(bounds) != param_count:
            bounds = [(0, 10)] * param_count
        
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
        """
        # Get function name to determine parameter count
        function_name = params.get('function_name', 'Unknown')
        
        # Get parameter count info
        from core.function_models import FunctionModels
        function_models = FunctionModels()
        function_info = function_models.get_function_info(function_name)
        param_count = function_info['param_count'] if function_info else 3
        
        # Parse bounds with the correct parameter count
        bounds = params.get('bounds', [(0, 10)] * param_count)
        
        # Ensure bounds has the correct length
        if len(bounds) != param_count:
            bounds = [(0, 10)] * param_count
        
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
        # Get function name to determine parameter count
        function_name = params.get('function_name', 'Unknown')
        
        # Get initial guess
        from core.function_models import FunctionModels
        function_models = FunctionModels()
        function_info = function_models.get_function_info(function_name)
        param_count = function_info['param_count'] if function_info else 3
        
        # Use provided initial guess or ensure it has correct length
        x0 = params.get('x0', [1.0] * param_count)
        
        # Ensure x0 matches the function parameter count
        if len(x0) != param_count:
            x0 = [1.0] * param_count
        
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