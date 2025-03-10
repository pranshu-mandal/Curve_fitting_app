#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fitting tab module for curve fitting application
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                            QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
                            QComboBox, QGroupBox, QScrollArea)
from PyQt5.QtCore import Qt

def create_algorithm_tab(algorithm_name, algorithm_func):
    """
    Create a tab for a specific fitting algorithm
    
    Parameters:
    -----------
    algorithm_name : str
        Name of the algorithm
    algorithm_func : callable
        Reference to the algorithm function
    
    Returns:
    --------
    QWidget
        Tab widget containing parameters for the algorithm
    """
    # Create tab widget and layout
    tab = QWidget()
    main_layout = QVBoxLayout(tab)
    
    # Create scroll area for parameters
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.NoFrame)
    
    # Content widget for the scroll area
    content = QWidget()
    form_layout = QFormLayout(content)
    form_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    
    # Add parameters based on algorithm
    if algorithm_name == "Differential Evolution":
        # Bounds group
        bounds_group = QGroupBox("Parameter Bounds")
        bounds_layout = QFormLayout()
        
        bounds_label = QLabel("Format: [lower1,upper1], [lower2,upper2], ...")
        bounds_edit = QLineEdit("[0,10], [0,10], [0,10]")
        
        bounds_layout.addRow(bounds_label)
        bounds_layout.addRow("Bounds:", bounds_edit)
        bounds_group.setLayout(bounds_layout)
        form_layout.addRow(bounds_group)
        
        # Add algorithm-specific parameters
        strategy_combo = QComboBox()
        strategy_combo.addItems([
            "best1bin", "best1exp", "rand1exp", "randtobest1exp", 
            "best2exp", "rand2exp", "randtobest1bin", "best2bin", 
            "rand2bin", "rand1bin"
        ])
        form_layout.addRow("Strategy:", strategy_combo)
        
        popsize_spin = QSpinBox()
        popsize_spin.setRange(5, 100)
        popsize_spin.setValue(15)
        form_layout.addRow("Population Size:", popsize_spin)
        
        tol_spin = QDoubleSpinBox()
        tol_spin.setRange(0.0000001, 0.1)
        tol_spin.setDecimals(8)
        tol_spin.setValue(0.01)
        tol_spin.setSingleStep(0.001)
        form_layout.addRow("Tolerance:", tol_spin)
        
        mutation_spin = QDoubleSpinBox()
        mutation_spin.setRange(0.1, 2.0)
        mutation_spin.setValue(0.8)
        mutation_spin.setSingleStep(0.1)
        form_layout.addRow("Mutation:", mutation_spin)
        
        recombination_spin = QDoubleSpinBox()
        recombination_spin.setRange(0.1, 1.0)
        recombination_spin.setValue(0.7)
        recombination_spin.setSingleStep(0.1)
        form_layout.addRow("Recombination:", recombination_spin)
        
        maxiter_spin = QSpinBox()
        maxiter_spin.setRange(10, 10000)
        maxiter_spin.setValue(1000)
        maxiter_spin.setSingleStep(100)
        form_layout.addRow("Max Iterations:", maxiter_spin)
        
    elif algorithm_name == "Basin Hopping":
        # Initial guess group
        init_group = QGroupBox("Initial Guess")
        init_layout = QFormLayout()
        
        init_edit = QLineEdit("1.0, 1.0, 1.0")
        init_layout.addRow("Initial parameters:", init_edit)
        init_group.setLayout(init_layout)
        form_layout.addRow(init_group)
        
        # Algorithm parameters
        niter_spin = QSpinBox()
        niter_spin.setRange(10, 1000)
        niter_spin.setValue(100)
        niter_spin.setSingleStep(10)
        form_layout.addRow("Number of Iterations:", niter_spin)
        
        T_spin = QDoubleSpinBox()
        T_spin.setRange(0.1, 10.0)
        T_spin.setValue(1.0)
        T_spin.setSingleStep(0.1)
        form_layout.addRow("Temperature:", T_spin)
        
        stepsize_spin = QDoubleSpinBox()
        stepsize_spin.setRange(0.01, 10.0)
        stepsize_spin.setValue(0.5)
        stepsize_spin.setSingleStep(0.1)
        form_layout.addRow("Step Size:", stepsize_spin)
        
    elif algorithm_name == "SHGO":
        # Bounds group
        bounds_group = QGroupBox("Parameter Bounds")
        bounds_layout = QFormLayout()
        
        bounds_label = QLabel("Format: [lower1,upper1], [lower2,upper2], ...")
        bounds_edit = QLineEdit("[0,10], [0,10], [0,10]")
        
        bounds_layout.addRow(bounds_label)
        bounds_layout.addRow("Bounds:", bounds_edit)
        bounds_group.setLayout(bounds_layout)
        form_layout.addRow(bounds_group)
        
        # Algorithm parameters
        n_spin = QSpinBox()
        n_spin.setRange(1, 100)
        n_spin.setValue(100)
        n_spin.setSingleStep(10)
        form_layout.addRow("Sampling Points:", n_spin)
        
        iters_spin = QSpinBox()
        iters_spin.setRange(1, 10)
        iters_spin.setValue(1)
        form_layout.addRow("Iterations:", iters_spin)
        
    elif algorithm_name == "Dual Annealing":
        # Bounds group
        bounds_group = QGroupBox("Parameter Bounds")
        bounds_layout = QFormLayout()
        
        bounds_label = QLabel("Format: [lower1,upper1], [lower2,upper2], ...")
        bounds_edit = QLineEdit("[0,10], [0,10], [0,10]")
        
        bounds_layout.addRow(bounds_label)
        bounds_layout.addRow("Bounds:", bounds_edit)
        bounds_group.setLayout(bounds_layout)
        form_layout.addRow(bounds_group)
        
        # Algorithm parameters
        maxiter_spin = QSpinBox()
        maxiter_spin.setRange(10, 10000)
        maxiter_spin.setValue(1000)
        maxiter_spin.setSingleStep(100)
        form_layout.addRow("Max Iterations:", maxiter_spin)
        
        initial_temp_spin = QDoubleSpinBox()
        initial_temp_spin.setRange(1.0, 10000.0)
        initial_temp_spin.setValue(5230.0)
        initial_temp_spin.setSingleStep(100.0)
        form_layout.addRow("Initial Temperature:", initial_temp_spin)
        
        restart_temp_spin = QDoubleSpinBox()
        restart_temp_spin.setRange(0.1, 10.0)
        restart_temp_spin.setValue(2.0)
        restart_temp_spin.setSingleStep(0.1)
        form_layout.addRow("Restart Temperature Ratio:", restart_temp_spin)
        
    elif algorithm_name == "Least Squares":
        # Initial guess group
        init_group = QGroupBox("Initial Guess")
        init_layout = QFormLayout()
        
        init_edit = QLineEdit("1.0, 1.0, 1.0")
        init_layout.addRow("Initial parameters:", init_edit)
        init_group.setLayout(init_layout)
        form_layout.addRow(init_group)
        
        # Method selection
        method_combo = QComboBox()
        method_combo.addItems(["trf", "dogbox", "lm"])
        form_layout.addRow("Method:", method_combo)
        
        # Bounds group
        bounds_group = QGroupBox("Parameter Bounds (Optional)")
        bounds_layout = QFormLayout()
        
        lower_bounds = QLineEdit("-inf, -inf, -inf")
        upper_bounds = QLineEdit("inf, inf, inf")
        
        bounds_layout.addRow("Lower bounds:", lower_bounds)
        bounds_layout.addRow("Upper bounds:", upper_bounds)
        bounds_group.setLayout(bounds_layout)
        form_layout.addRow(bounds_group)
        
        # Additional parameters
        ftol_spin = QDoubleSpinBox()
        ftol_spin.setRange(1e-10, 1e-1)
        ftol_spin.setDecimals(10)
        ftol_spin.setValue(1e-8)
        ftol_spin.setSingleStep(1e-9)
        form_layout.addRow("Function Tolerance:", ftol_spin)
        
        xtol_spin = QDoubleSpinBox()
        xtol_spin.setRange(1e-10, 1e-1)
        xtol_spin.setDecimals(10)
        xtol_spin.setValue(1e-8)
        xtol_spin.setSingleStep(1e-9)
        form_layout.addRow("Parameter Tolerance:", xtol_spin)
        
        gtol_spin = QDoubleSpinBox()
        gtol_spin.setRange(1e-10, 1e-1)
        gtol_spin.setDecimals(10)
        gtol_spin.setValue(1e-8)
        gtol_spin.setSingleStep(1e-9)
        form_layout.addRow("Gradient Tolerance:", gtol_spin)
    
    # Add description about the algorithm
    description = QLabel(get_algorithm_description(algorithm_name))
    description.setWordWrap(True)
    description.setStyleSheet("font-style: italic; color: #555555;")
    form_layout.addRow(description)
    
    # Set scroll content
    scroll.setWidget(content)
    main_layout.addWidget(scroll)
    
    return tab

def get_algorithm_description(algorithm_name):
    """Return a description of the algorithm"""
    descriptions = {
        "Differential Evolution": 
            "Differential Evolution is a stochastic global optimization algorithm "
            "that is effective for finding the global minimum of a function. "
            "It's particularly good for non-differentiable, non-linear, multi-modal functions.",
            
        "Basin Hopping":
            "Basin Hopping is a global optimization algorithm that combines a global "
            "stepping algorithm with local minimization at each step. Good for functions "
            "with many local minima.",
            
        "SHGO":
            "Simplicial Homology Global Optimization (SHGO) is a global optimization "
            "method that samples points in the parameter space and then minimizes "
            "a function from the best points. Effective for expensive-to-evaluate functions.",
            
        "Dual Annealing":
            "Dual Annealing is a global optimization algorithm that combines Simulated "
            "Annealing with a local search. It's good for finding the global minimum "
            "of a function with multiple local minima.",
            
        "Least Squares":
            "Least Squares method minimizes the sum of squared residuals between the "
            "model and the data. It's the most common approach for curve fitting when "
            "the model is not too complex and a good initial guess is available."
    }
    
    return descriptions.get(algorithm_name, "No description available.")