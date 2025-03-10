#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for the Curve Fitting Application
"""

import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show the main window
    window = MainWindow()
    window.show()
    
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()