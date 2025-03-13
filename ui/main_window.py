#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main window for the Curve Fitting Application
"""

import os
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTabWidget, QComboBox, QPushButton, QLabel, 
                             QFileDialog, QCheckBox, QTextEdit, QSplitter,
                             QGroupBox, QFormLayout, QLineEdit, QSpinBox,
                             QDoubleSpinBox, QFrame, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QPalette, QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt

from core.data_manager import DataManager
from core.function_models import FunctionModels
from core.fitting import FittingAlgorithms
from utils.report import generate_report

from ui.custom_function_dialog import CustomFunctionDialog
from core.custom_function_manager import CustomFunctionManager

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Local import to avoid circular dependency
        from ui.fitting_tab import create_algorithm_tab
        
        # Initialize core components
        self.data_manager = DataManager()
        self.function_models = FunctionModels()
        self.fitting_algorithms = FittingAlgorithms()
        self.custom_function_manager = CustomFunctionManager()
        
        # Set up the UI
        self.setWindowTitle("Advanced Curve Fitting Tool")
        self.setMinimumSize(1000, 700)
        
        # Create central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create and add the splitter for the top and bottom parts
        self.splitter = QSplitter(Qt.Vertical)
        self.main_layout.addWidget(self.splitter)
        
        # Create top widget (contains data and fitting controls)
        self.top_widget = QWidget()
        self.top_layout = QVBoxLayout(self.top_widget)
        
        # Data section
        self.data_group = QGroupBox("Data")
        self.data_layout = QVBoxLayout()
        self.data_group.setLayout(self.data_layout)
        
        # Data source controls
        self.data_source_layout = QHBoxLayout()
        self.generate_data_btn = QPushButton("Generate Synthetic Data")
        self.import_data_btn = QPushButton("Import from CSV")
        self.data_status_label = QLabel("No data loaded")
        self.data_source_layout.addWidget(self.generate_data_btn)
        self.data_source_layout.addWidget(self.import_data_btn)
        self.data_source_layout.addStretch()
        self.data_source_layout.addWidget(self.data_status_label)
        self.data_layout.addLayout(self.data_source_layout)
        
        # Function selection
        self.function_layout = QHBoxLayout()
        self.function_label = QLabel("Function Model:")
        self.function_combo = QComboBox()
        self.function_combo.addItems(self.function_models.get_function_names())
        # Add custom functions to the dropdown
        self.function_combo.addItems(self.custom_function_manager.get_function_names())
        self.add_custom_function_btn = QPushButton("Add Custom Function")
        self.function_layout.addWidget(self.function_label)
        self.function_layout.addWidget(self.function_combo)
        self.function_layout.addWidget(self.add_custom_function_btn)

        # Add function equation display
        self.function_equation_label = QLabel()
        self.function_equation_label.setStyleSheet("font-style: italic;")
        self.function_layout.addWidget(self.function_equation_label)
        self.function_layout.addStretch()
        self.data_layout.addLayout(self.function_layout)
        
        self.top_layout.addWidget(self.data_group)
        
        # Fitting algorithms tabs
        self.algorithm_group = QGroupBox("Fitting Algorithms")
        self.algorithm_layout = QVBoxLayout()
        self.algorithm_group.setLayout(self.algorithm_layout)
        
        self.algorithm_tabs = QTabWidget()
        
        # Create tabs for each algorithm using the locally imported function
        self.de_tab = create_algorithm_tab("Differential Evolution", optimize.differential_evolution)
        self.basin_tab = create_algorithm_tab("Basin Hopping", optimize.basinhopping)
        self.shgo_tab = create_algorithm_tab("SHGO", optimize.shgo)
        self.dual_annealing_tab = create_algorithm_tab("Dual Annealing", optimize.dual_annealing)
        self.least_squares_tab = create_algorithm_tab("Least Squares", optimize.least_squares)
        
        self.algorithm_tabs.addTab(self.de_tab, "Differential Evolution")
        self.algorithm_tabs.addTab(self.basin_tab, "Basin Hopping")
        self.algorithm_tabs.addTab(self.shgo_tab, "SHGO")
        self.algorithm_tabs.addTab(self.dual_annealing_tab, "Dual Annealing")
        self.algorithm_tabs.addTab(self.least_squares_tab, "Least Squares")
        
        self.algorithm_layout.addWidget(self.algorithm_tabs)
        
        # Run controls
        self.run_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run Fitting")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setEnabled(False)
        self.plot_checkbox = QCheckBox("Generate Plot")
        self.plot_checkbox.setChecked(True)
        self.run_layout.addWidget(self.run_btn)
        self.run_layout.addWidget(self.plot_checkbox)
        self.run_layout.addStretch()
        self.algorithm_layout.addLayout(self.run_layout)
        
        self.top_layout.addWidget(self.algorithm_group)
        
        # Create bottom widget (plot and report)
        self.bottom_widget = QWidget()
        self.bottom_layout = QHBoxLayout(self.bottom_widget)
        
        # Plot area
        self.plot_frame = QFrame()
        self.plot_layout = QVBoxLayout(self.plot_frame)
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.plot_layout.addWidget(self.canvas)
        
        # Report area
        self.report_group = QGroupBox("Fitting Report")
        self.report_layout = QVBoxLayout()
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_layout.addWidget(self.report_text)
        self.report_group.setLayout(self.report_layout)
        
        # Add plot and report to bottom layout
        self.bottom_layout.addWidget(self.plot_frame, 60)
        self.bottom_layout.addWidget(self.report_group, 40)
        
        # Add widgets to splitter
        self.splitter.addWidget(self.top_widget)
        self.splitter.addWidget(self.bottom_widget)
        self.splitter.setSizes([300, 400])
        
        # Connect signals and slots
        self.generate_data_btn.clicked.connect(self.generate_synthetic_data)
        self.import_data_btn.clicked.connect(self.import_csv_data)
        self.add_custom_function_btn.clicked.connect(self.add_custom_function)
        self.run_btn.clicked.connect(self.run_fitting)
        
        # Connect function selection changes to a method that updates parameter fields
        self.function_combo.currentTextChanged.connect(self.on_function_changed)

        # Apply modern styling
        self.apply_styling()
        
        # Initialize plot
        self.init_plot()

        # At the end of __init__, call the function once to set initial values
        self.on_function_changed(self.function_combo.currentText())
    
    def apply_styling(self):
        """Apply modern styling to the application"""
        # Set a blue color theme
        primary_color = QColor(53, 105, 180)
        secondary_color = QColor(234, 240, 253)
        
        # Set palette
        palette = QPalette()
        palette.setColor(QPalette.Button, primary_color)
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.Highlight, primary_color)
        palette.setColor(QPalette.HighlightedText, Qt.white)
        
        # Apply palette
        self.setPalette(palette)
        
        # Style buttons
        button_style = """
            QPushButton {
                background-color: #3569b4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #4580d1;
            }
            QPushButton:pressed {
                background-color: #2d5c9e;
            }
            QPushButton:disabled {
                background-color: #aaaaaa;
            }
        """
        
        # Apply styles
        self.generate_data_btn.setStyleSheet(button_style)
        self.import_data_btn.setStyleSheet(button_style)
        self.add_custom_function_btn.setStyleSheet(button_style)
        self.run_btn.setStyleSheet(button_style + "font-weight: bold; font-size: 14px;")
        
        # Group box style
        group_style = """
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cccccc;
                border-radius: 6px;
                margin-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 10px;
            }
        """
        self.data_group.setStyleSheet(group_style)
        self.algorithm_group.setStyleSheet(group_style)
        self.report_group.setStyleSheet(group_style)
    
    def init_plot(self):
        """Initialize the plot area"""
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Curve Fitting Results")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.figure.tight_layout()
        self.canvas.draw()
    
    @pyqtSlot()
    def generate_synthetic_data(self):
        """Generate synthetic data for fitting"""
        # Get the selected function
        function_name = self.function_combo.currentText()
        
        # Try to get the function from built-in functions first
        function = self.function_models.get_function(function_name)
        
        # If not found, check custom functions
        if function is None:
            function = self.custom_function_manager.get_function(function_name)
        
        if function is None:
            QMessageBox.critical(
                self, "Error", f"Function '{function_name}' not found."
            )
            return
        
        # Generate synthetic data with noise
        x, y, true_params = self.data_manager.generate_synthetic_data(
            function_name, function
        )
        
        # Update plot
        self.update_plot(x, y, None, None)
        
        # Update status
        self.data_status_label.setText(f"Synthetic data generated: {len(x)} points")
        self.run_btn.setEnabled(True)
        
        # Clear previous report
        self.report_text.clear()
        self.report_text.append(f"Function: {function_name}")
        self.report_text.append(f"True parameters: {true_params}")
        self.report_text.append("Generated synthetic data with random noise.")
        self.report_text.append("Select an algorithm and run fitting.")
    
    @pyqtSlot()
    def import_csv_data(self):
        """Import data from a CSV file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Import CSV Data", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if filename:
            try:
                # Read CSV file
                df = pd.read_csv(filename)
                
                # Prompt user to select columns for x and y
                columns = df.columns.tolist()
                
                # Simple dialog to select columns
                # In a full implementation, you'd want a more robust column selection UI
                if len(columns) >= 2:
                    x_col = columns[0]  # Default to first column
                    y_col = columns[1]  # Default to second column
                    
                    # Extract data
                    x = df[x_col].values
                    y = df[y_col].values
                    
                    # Store data in data manager
                    self.data_manager.set_data(x, y)
                    
                    # Update plot
                    self.update_plot(x, y, None, None)
                    
                    # Update status
                    self.data_status_label.setText(f"Imported from CSV: {len(x)} points")
                    self.run_btn.setEnabled(True)
                    
                    # Clear previous report
                    self.report_text.clear()
                    self.report_text.append(f"Data imported from: {os.path.basename(filename)}")
                    self.report_text.append(f"X column: {x_col}, Y column: {y_col}")
                    self.report_text.append("Select an algorithm and run fitting.")
                else:
                    QMessageBox.warning(
                        self, "Invalid CSV", "The CSV file must contain at least two columns."
                    )
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to import CSV data: {str(e)}"
                )
    
    @pyqtSlot()
    def add_custom_function(self):
        """Add a custom function for fitting"""
        dialog = CustomFunctionDialog(self)
        if dialog.exec_():
            # Get function details from dialog
            function_name = dialog.function_name
            function_expr = dialog.function_expr
            function_params = dialog.params
            
            # Add to custom function manager
            success = self.custom_function_manager.add_function(
                function_name, function_expr, function_params)
            
            if success:
                # Add to function dropdown
                self.function_combo.addItem(function_name)
                self.function_combo.setCurrentText(function_name)
                QMessageBox.information(
                    self, "Custom Function", 
                    f"Custom function '{function_name}' added successfully."
                )
            else:
                QMessageBox.critical(
                    self, "Error", 
                    "Failed to add custom function. Check console for details."
                )
    
    @pyqtSlot()
    def run_fitting(self):
        """Run the selected fitting algorithm"""
        if not self.data_manager.has_data():
            QMessageBox.warning(
                self, "No Data", "Please generate or import data first."
            )
            return
        
        # Get the selected function and algorithm
        function_name = self.function_combo.currentText()
        
        # Try to get the function from built-in functions first
        function = self.function_models.get_function(function_name)
        
        # If not found, check custom functions
        if function is None:
            function = self.custom_function_manager.get_function(function_name)
        
        if function is None:
            QMessageBox.critical(
                self, "Error", f"Function '{function_name}' not found."
            )
            return
        
        algorithm_index = self.algorithm_tabs.currentIndex()
        algorithm_name = self.algorithm_tabs.tabText(algorithm_index)
        
        # Get algorithm parameters from the current tab
        current_tab = self.algorithm_tabs.currentWidget()
        
        # Important: Get function info to determine parameter count
        function_info = self.function_models.get_function_info(function_name)
        
        # If not found in built-in functions, check custom functions
        if not function_info:
            function_info = self.custom_function_manager.get_function_info(function_name)
            
        if not function_info:
            QMessageBox.critical(
                self, "Error", f"Function information for '{function_name}' not found."
            )
            return
            
        param_count = function_info['param_count']
        
        # Create algorithm params dictionary with function name
        algorithm_params = {'function_name': function_name}
        
        # Extract parameters based on algorithm type
        if algorithm_name == "Differential Evolution":
            if hasattr(current_tab, 'bounds_edit'):
                # Parse bounds from UI
                bounds_text = current_tab.bounds_edit.text()
                try:
                    # Use a regular expression to extract all pairs of numbers inside brackets
                    import re
                    bounds_list = []
                    # Find all content between square brackets
                    matches = re.findall(r'\[(.*?)\]', bounds_text)
                    for match in matches:
                        # Split by comma and convert to floats
                        values = match.split(',')
                        if len(values) == 2:
                            lower, upper = float(values[0].strip()), float(values[1].strip())
                            bounds_list.append((lower, upper))
                    
                    # Make sure we have the right number of bounds
                    if len(bounds_list) != param_count:
                        # Adjust to match param_count
                        if len(bounds_list) < param_count:
                            bounds_list.extend([(0, 10)] * (param_count - len(bounds_list)))
                        else:
                            bounds_list = bounds_list[:param_count]
                    
                    algorithm_params['bounds'] = bounds_list
                except Exception as e:
                    QMessageBox.warning(self, "Parameter Error", 
                                       f"Error parsing bounds: {str(e)}\nUsing default bounds.")
                    algorithm_params['bounds'] = [(0, 10)] * param_count
        
        elif algorithm_name == "Basin Hopping" or algorithm_name == "Least Squares":
            if hasattr(current_tab, 'init_edit'):
                # Parse initial guess from UI
                init_text = current_tab.init_edit.text()
                try:
                    # Convert text like "1.0, 2.0, 3.0" to list [1.0, 2.0, 3.0]
                    init_values = [float(x.strip()) for x in init_text.split(',')]
                    # Ensure we have exactly the right number of parameters
                    if len(init_values) != param_count:
                        # Adjust to match param_count
                        if len(init_values) < param_count:
                            init_values.extend([1.0] * (param_count - len(init_values)))
                        else:
                            init_values = init_values[:param_count]
                    algorithm_params['x0'] = init_values
                except Exception as e:
                    QMessageBox.warning(self, "Parameter Error", 
                                       f"Error parsing initial guess: {str(e)}\nUsing default values.")
                    algorithm_params['x0'] = [1.0] * param_count
        
        # Fix for SHGO and Dual Annealing bounds parsing
        elif algorithm_name == "SHGO" or algorithm_name == "Dual Annealing":
            if hasattr(current_tab, 'bounds_edit'):
                # Parse bounds from UI using regex
                bounds_text = current_tab.bounds_edit.text()
                try:
                    import re
                    bounds_list = []
                    # Find all content between square brackets
                    matches = re.findall(r'\[(.*?)\]', bounds_text)
                    for match in matches:
                        # Split by comma and convert to floats
                        values = match.split(',')
                        if len(values) == 2:
                            lower, upper = float(values[0].strip()), float(values[1].strip())
                            bounds_list.append((lower, upper))
                    
                    # Make sure we have the right number of bounds
                    if len(bounds_list) != param_count:
                        # Adjust to match param_count
                        if len(bounds_list) < param_count:
                            bounds_list.extend([(0, 10)] * (param_count - len(bounds_list)))
                        else:
                            bounds_list = bounds_list[:param_count]
                    
                    algorithm_params['bounds'] = bounds_list
                except Exception as e:
                    QMessageBox.warning(self, "Parameter Error", 
                                       f"Error parsing bounds: {str(e)}\nUsing default bounds.")
                    algorithm_params['bounds'] = [(0, 10)] * param_count
        
        try:
            # Run the fitting algorithm
            params, result = self.fitting_algorithms.run_fitting(
                algorithm_name,
                function,
                self.data_manager.x_data,
                self.data_manager.y_data,
                algorithm_params
            )
            
            # Update the plot if requested
            if self.plot_checkbox.isChecked():
                x_data = self.data_manager.x_data
                y_data = self.data_manager.y_data
                
                # Generate fitted curve
                x_fit = np.linspace(min(x_data), max(x_data), 1000)
                y_fit = function(x_fit, *params)
                
                self.update_plot(x_data, y_data, x_fit, y_fit)
            
            # Generate and display report
            report = generate_report(
                function_name,
                algorithm_name,
                params,
                self.data_manager.true_params if hasattr(self.data_manager, 'true_params') else None,
                result,
                self.data_manager.x_data,
                self.data_manager.y_data,
                function
            )
            
            self.report_text.setHtml(report)
            
        except Exception as e:
            QMessageBox.critical(
                self, "Fitting Error", f"An error occurred during fitting: {str(e)}"
            )
    
    def update_plot(self, x, y, x_fit=None, y_fit=None):
        """Update the plot with data and optional fitted curve"""
        self.ax.clear()
        
        # Plot data points
        self.ax.scatter(x, y, color='blue', alpha=0.7, label='Data')
        
        # Plot fitted curve if available
        if x_fit is not None and y_fit is not None:
            self.ax.plot(x_fit, y_fit, 'r-', linewidth=2, label='Fitted Curve')
            self.ax.legend()
        
        self.ax.set_title("Curve Fitting Results")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def on_function_changed(self, function_name=None):
        """
        Update parameter UI fields when function selection changes
        """
        if function_name is None:
            function_name = self.function_combo.currentText()
        
        # Check if this is a built-in or custom function
        info = self.function_models.get_function_info(function_name)
        
        # If not found in built-in functions, check custom functions
        if not info:
            info = self.custom_function_manager.get_function_info(function_name)
        
        if not info:
            return  # safety check
        
        param_count = info['param_count']
        
        # Update the equation display
        if 'equation' in info:
            self.function_equation_label.setText(f"f(x) = {info['equation']}")
        
        # Update bounds for algorithms that use bounds
        bounds_str = ", ".join(["[0,10]"] * param_count)
        if hasattr(self.de_tab, 'bounds_edit'):
            self.de_tab.bounds_edit.setText(bounds_str)
        if hasattr(self.shgo_tab, 'bounds_edit'):
            self.shgo_tab.bounds_edit.setText(bounds_str)
        if hasattr(self.dual_annealing_tab, 'bounds_edit'):
            self.dual_annealing_tab.bounds_edit.setText(bounds_str)
        
        # Update initial values for algorithms that use them
        init_str = ", ".join(["1.0"] * param_count)
        if hasattr(self.basin_tab, 'init_edit'):
            self.basin_tab.init_edit.setText(init_str)
        if hasattr(self.least_squares_tab, 'init_edit'):
            self.least_squares_tab.init_edit.setText(init_str)
        
        # Update display to show equation and parameters
        self.report_text.clear()
        self.report_text.append(f"Selected function: {function_name}")
        self.report_text.append(f"Equation: {info['equation']}")
        
        # Fix for parameter display
        if 'params' in info:
            # Check if params contains dictionaries (custom function) or strings (built-in)
            if info['params'] and isinstance(info['params'][0], dict):
                # For custom functions: extract the name from each parameter dictionary
                param_names = [p['name'] for p in info['params']]
                self.report_text.append(f"Parameters: {', '.join(param_names)}")
            else:
                # For built-in functions: params is already a list of strings
                self.report_text.append(f"Parameters: {', '.join(info['params'])}")
        else:
            # Fallback if no params information is available
            param_list = [f'p{i}' for i in range(param_count)]
            self.report_text.append(f"Parameters: {', '.join(param_list)}")
            
        self.report_text.append(f"Required parameter count: {param_count}")