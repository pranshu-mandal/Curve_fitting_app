#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function models for curve fitting
"""

import numpy as np

class FunctionModels:
    """
    Class to manage the available function models for curve fitting
    """
    def __init__(self):
        """Initialize function models"""
        self.functions = {
            "Linear": self.linear_func,
            "Quadratic": self.quadratic_func,
            "Cubic": self.cubic_func,
            "Power Law": self.power_law_func,
            "Exponential": self.exponential_func,
            "Double Power Law": self.double_power_law_func,
            "Triple Power Law": self.triple_power_law_func
        }
        
        # Function parameter counts and descriptions
        self.function_info = {
            "Linear": {
                "equation": "y = a*x + b",
                "params": ["a", "b"],
                "param_count": 2
            },
            "Quadratic": {
                "equation": "y = a*x^2 + b*x + c",
                "params": ["a", "b", "c"],
                "param_count": 3
            },
            "Cubic": {
                "equation": "y = a*x^3 + b*x^2 + c*x + d",
                "params": ["a", "b", "c", "d"],
                "param_count": 4
            },
            "Power Law": {
                "equation": "y = a*x^b + c",
                "params": ["a", "b", "c"],
                "param_count": 3
            },
            "Exponential": {
                "equation": "y = a*exp(b*x) + c",
                "params": ["a", "b", "c"],
                "param_count": 3
            },
            "Double Power Law": {
                "equation": "y = a*x^b + c*x^d",
                "params": ["a", "b", "c", "d"],
                "param_count": 4
            },
            "Triple Power Law": {
                "equation": "y = a*x^b + c*x^d + e*x^f",
                "params": ["a", "b", "c", "d", "e", "f"],
                "param_count": 6
            }
        }
    
    def get_function_names(self):
        """Get a list of available function names"""
        return list(self.functions.keys())
    
    def get_function(self, name):
        """Get a function by name"""
        return self.functions.get(name)
    
    def get_function_info(self, name):
        """Get information about a function"""
        return self.function_info.get(name)
    
    def add_custom_function(self, name, func, equation, params):
        """
        Add a custom function
        
        Parameters:
        -----------
        name : str
            Name of the function
        func : callable
            Function to add
        equation : str
            Equation representation
        params : list
            List of parameter names
        """
        self.functions[name] = func
        self.function_info[name] = {
            "equation": equation,
            "params": params,
            "param_count": len(params)
        }
    
    def parse_custom_function(self, func_str):
        """
        Parse a custom function string into a callable function
        
        Parameters:
        -----------
        func_str : str
            String representation of the function
        
        Returns:
        --------
        callable
            Parsed function
        list
            Parameter names
        str
            Equation representation
        """
        # This is a simplified version - in a real app, you'd need
        # more robust parsing and safety checks
        try:
            # Extract parameter names (assuming format like "f(x, a, b, c) = ...")
            param_str = func_str.split('(')[1].split(')')[0]
            all_params = [p.strip() for p in param_str.split(',')]
            
            # First parameter should be x, rest are fitting parameters
            x_var = all_params[0]
            params = all_params[1:]
            
            # Extract the equation part
            equation = func_str.split('=')[1].strip()
            
            # Create a function using eval (note: this is unsafe for production!)
            # In a real app, you'd want to use a safer approach
            def custom_func(x, *args):
                param_dict = {params[i]: args[i] for i in range(len(params))}
                param_dict[x_var] = x
                return eval(equation, {"__builtins__": {}}, {**param_dict, "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp, "log": np.log})
            
            return custom_func, params, func_str
            
        except Exception as e:
            raise ValueError(f"Failed to parse custom function: {str(e)}")
    
    # Built-in function definitions
    
    def linear_func(self, x, a, b):
        """Linear function: y = a*x + b"""
        return a * x + b
    
    def quadratic_func(self, x, a, b, c):
        """Quadratic function: y = a*x^2 + b*x + c"""
        return a * x**2 + b * x + c
    
    def cubic_func(self, x, a, b, c, d):
        """Cubic function: y = a*x^3 + b*x^2 + c*x + d"""
        return a * x**3 + b * x**2 + c * x + d
    
    def power_law_func(self, x, a, b, c):
        """Power law function: y = a*x^b + c"""
        return a * x**b + c
    
    def exponential_func(self, x, a, b, c):
        """Exponential function: y = a*exp(b*x) + c"""
        return a * np.exp(b * x) + c
    
    def double_power_law_func(self, x, a, b, c, d):
        """Double power law function: y = a*x^b + c*x^d"""
        return a * x**b + c * x**d
    
    def triple_power_law_func(self, x, a, b, c, d, e, f):
        """Triple power law function: y = a*x^b + c*x^d + e*x^f"""
        return a * x**b + c * x**d + e * x**f