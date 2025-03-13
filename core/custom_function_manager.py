#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manager for custom functions - handles saving, loading, and creating callable functions
"""

import os
import json
import numpy as np

class CustomFunctionManager:
    def __init__(self, save_dir="custom_functions"):
        """
        Initialize the custom function manager
        
        Parameters:
        -----------
        save_dir : str
            Directory to save custom functions
        """
        self.save_dir = save_dir
        self.custom_functions = {}
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Load existing custom functions
        self.load_functions()
    
    def load_functions(self):
        """Load all saved custom functions"""
        self.custom_functions = {}
        
        try:
            # Check if the functions JSON file exists
            functions_file = os.path.join(self.save_dir, "functions.json")
            if os.path.exists(functions_file):
                with open(functions_file, 'r') as f:
                    function_data = json.load(f)
                
                # Create callable functions from the data
                for name, data in function_data.items():
                    self.custom_functions[name] = {
                        'function': self.create_function_from_string(data['expression'], data['params']),
                        'expression': data['expression'],
                        'params': data['params'],
                        'param_count': len(data['params']),
                        'equation': data['expression']
                    }
        except Exception as e:
            print(f"Error loading custom functions: {e}")
    
    def save_functions(self):
        """Save all custom functions to disk"""
        try:
            # Prepare data for saving
            save_data = {}
            for name, data in self.custom_functions.items():
                save_data[name] = {
                    'expression': data['expression'],
                    'params': data['params']
                }
            
            # Save to JSON file
            functions_file = os.path.join(self.save_dir, "functions.json")
            with open(functions_file, 'w') as f:
                json.dump(save_data, f, indent=2)
                
            return True
        except Exception as e:
            print(f"Error saving custom functions: {e}")
            return False
    
    def add_function(self, name, expression, params):
        """
        Add a new custom function
        
        Parameters:
        -----------
        name : str
            Name of the function
        expression : str
            Mathematical expression (e.g., "a*x**b + c")
        params : list
            List of parameter dictionaries with 'name', 'init_value', 'desc'
        
        Returns:
        --------
        bool
            Success flag
        """
        try:
            # Create the function
            function = self.create_function_from_string(expression, params)
            
            # Add to dictionary
            self.custom_functions[name] = {
                'function': function,
                'expression': expression,
                'params': params,
                'param_count': len(params),
                'equation': expression
            }
            
            # Save to disk
            return self.save_functions()
        except Exception as e:
            print(f"Error adding custom function: {e}")
            return False
    
    def remove_function(self, name):
        """Remove a custom function"""
        if name in self.custom_functions:
            del self.custom_functions[name]
            return self.save_functions()
        return False
    
    def get_function(self, name):
        """Get a custom function by name"""
        if name in self.custom_functions:
            return self.custom_functions[name]['function']
        return None
    
    def get_function_info(self, name):
        """Get information about a custom function"""
        if name in self.custom_functions:
            info = self.custom_functions[name].copy()
            # Remove the function object for JSON serialization
            if 'function' in info:
                del info['function']
            return info
        return None
    
    def get_function_names(self):
        """Get names of all custom functions"""
        return list(self.custom_functions.keys())
    
    def create_function_from_string(self, expression, params):
        """
        Create a callable function from a string expression
        
        Parameters:
        -----------
        expression : str
            Mathematical expression (e.g., "a*x**b + c")
        params : list
            List of parameter dictionaries with 'name', 'init_value', 'desc'
        
        Returns:
        --------
        callable
            Function that takes x and parameters as input
        """
        param_names = [p['name'] for p in params]
        
        def custom_function(x, *args):
            # Map arguments to parameter names
            param_dict = {param_names[i]: args[i] for i in range(min(len(param_names), len(args)))}
            param_dict['x'] = x
            
            # Add numpy functions to the namespace
            namespace = {
                "np": np,
                "sin": np.sin,
                "cos": np.cos,
                "exp": np.exp,
                "log": np.log,
                **param_dict
            }
            
            # Evaluate the expression
            return eval(expression, {"__builtins__": {}}, namespace)
        
        # Set name and docstring for the function
        custom_function.__name__ = f"custom_func_{len(param_names)}_params"
        custom_function.__doc__ = f"Custom function: y = {expression}"
        
        return custom_function

    