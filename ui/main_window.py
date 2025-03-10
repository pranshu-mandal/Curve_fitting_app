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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Local import to avoid circular dependency
        from ui.fitting_tab import create_algorithm_tab
        
        # Initialize core components
        self.data_manager = DataManager()
        self.function_models = FunctionModels()
        self.fitting_algorithms = FittingAlgorithms()
        
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
        self.add_custom_function_btn = QPushButton("Add Custom Function")
        self.function_layout.addWidget(self.function_label)
        self.function_layout.addWidget(self.function_combo)
        self.function_layout.addWidget(self.add_custom_function_btn)
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
        
        # Apply modern styling
        self.apply_styling()
        
        # Initialize plot
        self.init_plot()
    
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
        
        # Generate synthetic data with noise
        x, y, true_params = self.data_manager.generate_synthetic_data(
            function_name, 
            self.function_models.get_function(function_name)
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
        # In a full implementation, this would open a dialog to enter a custom function
        # For simplicity, we'll just add a placeholder custom function
        custom_function_name = "Custom Function"
        self.function_combo.addItem(custom_function_name)
        self.function_combo.setCurrentText(custom_function_name)
        
        # For a real implementation, you'd want to parse the function definition and add it to the models
        QMessageBox.information(
            self, "Custom Function", 
            "In a full implementation, this would open a dialog to define a custom function."
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
        function = self.function_models.get_function(function_name)
        
        algorithm_index = self.algorithm_tabs.currentIndex()
        algorithm_name = self.algorithm_tabs.tabText(algorithm_index)
        
        # Get algorithm parameters from the current tab
        current_tab = self.algorithm_tabs.currentWidget()
        algorithm_params = {}
        
        # In a full implementation, you'd extract parameters from the UI controls
        # For this simplified version, we'll use default parameters
        
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