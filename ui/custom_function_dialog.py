#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dialog for creating and editing custom functions
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, 
                            QLabel, QLineEdit, QPushButton, QMessageBox, 
                            QTableWidget, QTableWidgetItem, QHeaderView,
                            QSpinBox, QGroupBox)
from PyQt5.QtCore import Qt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class CustomFunctionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Define Custom Function")
        self.setMinimumSize(700, 500)
        
        # Initialize variables
        self.function_name = ""
        self.function_expr = ""
        self.params = []
        self.param_count = 0
        
        # Create layout
        main_layout = QVBoxLayout(self)
        
        # Function definition section
        definition_group = QGroupBox("Function Definition")
        form_layout = QFormLayout()
        
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g., My Custom Function")
        form_layout.addRow("Function Name:", self.name_edit)
        
        self.expr_edit = QLineEdit()
        self.expr_edit.setPlaceholderText("e.g., a*x**b + c*exp(-d*x)")
        form_layout.addRow("Expression (use x as variable):", self.expr_edit)
        
        # Parameter count
        self.param_spin = QSpinBox()
        self.param_spin.setRange(1, 10)
        self.param_spin.setValue(3)
        self.param_spin.valueChanged.connect(self.update_param_table)
        form_layout.addRow("Number of Parameters:", self.param_spin)
        
        definition_group.setLayout(form_layout)
        main_layout.addWidget(definition_group)
        
        # Parameter table
        param_group = QGroupBox("Parameters")
        param_layout = QVBoxLayout()
        
        self.param_table = QTableWidget(3, 3)
        self.param_table.setHorizontalHeaderLabels(["Name", "Initial Value", "Description"])
        self.param_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        param_layout.addWidget(self.param_table)
        
        # Initialize with default parameter names
        self.update_param_table()
        
        param_group.setLayout(param_layout)
        main_layout.addWidget(param_group)
        
        # Preview section
        preview_group = QGroupBox("Function Preview")
        preview_layout = QVBoxLayout()
        
        self.preview_figure = Figure(figsize=(5, 3), dpi=100)
        self.preview_canvas = FigureCanvas(self.preview_figure)
        preview_layout.addWidget(self.preview_canvas)
        
        self.preview_button = QPushButton("Preview Function")
        self.preview_button.clicked.connect(self.preview_function)
        preview_layout.addWidget(self.preview_button)
        
        preview_group.setLayout(preview_layout)
        main_layout.addWidget(preview_group)
        
        # Button section
        button_layout = QHBoxLayout()
        self.save_button = QPushButton("Save Function")
        self.save_button.clicked.connect(self.save_function)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        main_layout.addLayout(button_layout)
    
    def update_param_table(self):
        """Update parameter table based on parameter count"""
        self.param_count = self.param_spin.value()
        self.param_table.setRowCount(self.param_count)
        
        # Add default values for new rows
        for i in range(self.param_table.rowCount()):
            # Parameter name (a, b, c, ...)
            param_name = chr(97 + i) if i < 26 else f"p{i}"
            if self.param_table.item(i, 0) is None:
                self.param_table.setItem(i, 0, QTableWidgetItem(param_name))
            
            # Default initial value
            if self.param_table.item(i, 1) is None:
                self.param_table.setItem(i, 1, QTableWidgetItem("1.0"))
            
            # Description
            if self.param_table.item(i, 2) is None:
                self.param_table.setItem(i, 2, QTableWidgetItem(f"Parameter {param_name}"))
    
    def preview_function(self):
        """Generate and display a preview of the function"""
        try:
            # Get function expression
            expr = self.expr_edit.text()
            if not expr:
                QMessageBox.warning(self, "Error", "Please enter a function expression.")
                return
            
            # Get parameter values from table
            param_names = []
            param_values = []
            for i in range(self.param_table.rowCount()):
                param_name = self.param_table.item(i, 0).text()
                param_value = float(self.param_table.item(i, 1).text())
                param_names.append(param_name)
                param_values.append(param_value)
            
            # Create a lambda function for evaluation
            param_dict = {param_names[i]: param_values[i] for i in range(len(param_names))}
            
            # Generate x values
            x = np.linspace(0, 10, 100)
            
            # Evaluate function
            y = []
            for x_val in x:
                # Create local parameter dictionary with x value
                local_dict = param_dict.copy()
                local_dict['x'] = x_val
                # Evaluate expression
                y_val = eval(expr, {"__builtins__": {}}, {**local_dict, "np": np, 
                                                        "sin": np.sin, "cos": np.cos, 
                                                        "exp": np.exp, "log": np.log})
                y.append(y_val)
            
            # Plot the function
            self.preview_figure.clear()
            ax = self.preview_figure.add_subplot(111)
            ax.plot(x, y)
            ax.set_title("Function Preview")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(True)
            self.preview_canvas.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error previewing function: {str(e)}")
    
    def save_function(self):
        """Save the custom function"""
        try:
            # Validate inputs
            self.function_name = self.name_edit.text().strip()
            if not self.function_name:
                QMessageBox.warning(self, "Error", "Please enter a function name.")
                return
            
            self.function_expr = self.expr_edit.text().strip()
            if not self.function_expr:
                QMessageBox.warning(self, "Error", "Please enter a function expression.")
                return
            
            # Get parameter information
            self.params = []
            for i in range(self.param_table.rowCount()):
                name = self.param_table.item(i, 0).text().strip()
                init_value = self.param_table.item(i, 1).text().strip()
                desc = self.param_table.item(i, 2).text().strip()
                
                if not name:
                    QMessageBox.warning(self, "Error", f"Parameter {i+1} needs a name.")
                    return
                
                try:
                    init_value = float(init_value)
                except ValueError:
                    QMessageBox.warning(self, "Error", f"Invalid initial value for parameter {name}.")
                    return
                
                self.params.append({
                    'name': name,
                    'init_value': init_value,
                    'desc': desc
                })
            
            # Test function evaluation
            x_test = 1.0
            param_dict = {p['name']: p['init_value'] for p in self.params}
            param_dict['x'] = x_test
            
            try:
                eval(self.function_expr, {"__builtins__": {}}, {**param_dict, "np": np, 
                                                              "sin": np.sin, "cos": np.cos, 
                                                              "exp": np.exp, "log": np.log})
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Function evaluation failed: {str(e)}")
                return
            
            # If we made it here, accept the dialog
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving function: {str(e)}")