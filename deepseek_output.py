import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
from scipy.optimize import (curve_fit, differential_evolution, 
                          basinhopping, shgo, dual_annealing)

class CurveFittingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Curve Fitting App")
        self.root.geometry("800x600")
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure grid for resizing
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Data storage
        self.x_data = None
        self.y_data = None
        self.is_synthetic = False
        
        # Create UI components
        self.create_top_section()
        self.create_middle_section()
        self.create_bottom_section()
        
        # Initialize default bounds
        self.set_default_bounds()

    def create_top_section(self):
        frame = ttk.LabelFrame(self.root, text="Data Input", padding=10)
        frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        frame.grid_columnconfigure(2, weight=1)

        ttk.Button(frame, text="Generate Data", command=self.generate_data,
                  style='Accent.TButton').grid(row=0, column=0, padx=5)
        ttk.Button(frame, text="Import CSV", command=self.import_csv,
                  style='Accent.TButton').grid(row=0, column=1, padx=5)
        
        self.func_var = tk.StringVar()
        self.functions = [
            "a*x**b + c*x**d",
            "a*x**b + c*x**d + e*x**f"
        ]
        self.func_dropdown = ttk.Combobox(frame, textvariable=self.func_var, 
                                        values=self.functions, state="readonly")
        self.func_dropdown.grid(row=0, column=2, padx=5, sticky="ew")
        self.func_var.set(self.functions[0])
        self.func_dropdown.bind("<<ComboboxSelected>>", self.update_parameters)

    def create_middle_section(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        
        # Create algorithm tabs
        self.algo_tabs = {
            "Differential Evolution": self.create_de_tab(self.notebook),
            "Basin Hopping": self.create_bh_tab(self.notebook),
            "SHGO": self.create_shgo_tab(self.notebook),
            "Dual Annealing": self.create_da_tab(self.notebook)
        }

        for name, frame in self.algo_tabs.items():
            self.notebook.add(frame, text=name)

    def create_de_tab(self, parent):
        frame = ttk.Frame(parent)
        self.create_common_parameters(frame)
        
        param_frame = ttk.LabelFrame(frame, text="DE Parameters")
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(param_frame, text="Strategy:").grid(row=0, column=0, padx=5)
        self.de_strategy = ttk.Combobox(param_frame, values=['best1bin', 'best1exp', 'rand1exp'])
        self.de_strategy.grid(row=0, column=1, padx=5)
        self.de_strategy.set('best1bin')
        
        ttk.Label(param_frame, text="Maxiter:").grid(row=1, column=0, padx=5)
        self.de_maxiter = ttk.Entry(param_frame)
        self.de_maxiter.grid(row=1, column=1, padx=5)
        self.de_maxiter.insert(0, '1000')
        
        return frame

    def create_bh_tab(self, parent):
        frame = ttk.Frame(parent)
        self.create_common_parameters(frame)
        
        param_frame = ttk.LabelFrame(frame, text="Basin Hopping Parameters")
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(param_frame, text="niter:").grid(row=0, column=0, padx=5)
        self.bh_niter = ttk.Entry(param_frame)
        self.bh_niter.grid(row=0, column=1, padx=5)
        self.bh_niter.insert(0, '100')
        
        ttk.Label(param_frame, text="Stepsize:").grid(row=1, column=0, padx=5)
        self.bh_stepsize = ttk.Entry(param_frame)
        self.bh_stepsize.grid(row=1, column=1, padx=5)
        self.bh_stepsize.insert(0, '0.5')
        
        return frame

    def create_shgo_tab(self, parent):
        frame = ttk.Frame(parent)
        self.create_common_parameters(frame)
        
        param_frame = ttk.LabelFrame(frame, text="SHGO Parameters")
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(param_frame, text="Sampling method:").grid(row=0, column=0, padx=5)
        self.shgo_sampling = ttk.Combobox(param_frame, values=['simplicial', 'sobol'])
        self.shgo_sampling.grid(row=0, column=1, padx=5)
        self.shgo_sampling.set('simplicial')
        
        return frame

    def create_da_tab(self, parent):
        frame = ttk.Frame(parent)
        self.create_common_parameters(frame)
        
        param_frame = ttk.LabelFrame(frame, text="Dual Annealing Parameters")
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(param_frame, text="Maxiter:").grid(row=0, column=0, padx=5)
        self.da_maxiter = ttk.Entry(param_frame)
        self.da_maxiter.grid(row=0, column=1, padx=5)
        self.da_maxiter.insert(0, '1000')
        
        return frame

    def create_common_parameters(self, parent):
        params_frame = ttk.LabelFrame(parent, text="Parameter Bounds (0-10 by default)")
        params_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.bound_entries = {}
        params = ["a", "b", "c", "d", "e", "f"]
        for i, param in enumerate(params):
            ttk.Label(params_frame, text=param).grid(row=i, column=0, padx=5, pady=2)
            lower = ttk.Entry(params_frame, width=8)
            lower.grid(row=i, column=1, padx=5, pady=2)
            upper = ttk.Entry(params_frame, width=8)
            upper.grid(row=i, column=2, padx=5, pady=2)
            self.bound_entries[param] = (lower, upper)
        
        return params_frame

    def create_bottom_section(self):
        frame = ttk.Frame(self.root)
        frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        
        ttk.Button(frame, text="Run Fitting", command=self.perform_fit,
                  style='Accent.TButton').pack(pady=5, side=tk.TOP)
        
        self.result_text = tk.Text(frame, height=10, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        self.result_text.tag_configure('success', foreground='green')
        self.result_text.tag_configure('error', foreground='red')

    def set_default_bounds(self):
        for param, (lower, upper) in self.bound_entries.items():
            lower.insert(0, '0')
            upper.insert(0, '10')

    def update_parameters(self, event=None):
        current_func = self.func_var.get()
        num_params = 4 if current_func == self.functions[0] else 6
        
        params = ["a", "b", "c", "d", "e", "f"]
        for param in params:
            row = params.index(param)
            visible = row < num_params
            self.bound_entries[param][0].master.grid_rowconfigure(row, weight=1 if visible else 0)
            self.bound_entries[param][0].master.grid_remove() if not visible else None
            self.bound_entries[param][1].master.grid_remove() if not visible else None

    def generate_data(self):
        x = np.linspace(1, 10, 100)
        current_func = self.func_var.get()
        
        if current_func == self.functions[0]:
            y = 1.5 * x**3.2 + 2.8 * x**0.8
        else:
            y = 1.5 * x**3.2 + 2.8 * x**0.8 + 3.8 * x**2.7
            
        noise = np.random.normal(0, 0.1*y.size, y.shape)
        self.x_data = x
        self.y_data = y + noise
        self.is_synthetic = True
        messagebox.showinfo("Success", "Synthetic data generated!")

    def import_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                df = pd.read_csv(file_path)
                self.x_data = df.iloc[:, 0].values
                self.y_data = df.iloc[:, 1].values
                self.is_synthetic = False
                messagebox.showinfo("Success", f"Loaded {len(self.x_data)} data points")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV: {str(e)}")

    def perform_fit(self):
        if self.x_data is None or self.y_data is None:
            messagebox.showerror("Error", "No data available!")
            return
        
        try:
            current_func = self.func_var.get()
            algo = self.notebook.tab(self.notebook.select(), "text")
            
            # Get parameter bounds
            params = ["a", "b", "c", "d", "e", "f"][:4 if current_func == self.functions[0] else 6]
            bounds = []
            for param in params:
                lower = float(self.bound_entries[param][0].get())
                upper = float(self.bound_entries[param][1].get())
                bounds.append((lower, upper))
                
            # Define function to fit
            if current_func == self.functions[0]:
                def func(x, a, b, c, d): return a*x**b + c*x**d
                initial_guess = [1.0, 1.0, 1.0, 1.0]
            else:
                def func(x, a, b, c, d, e, f): return a*x**b + c*x**d + e*x**f
                initial_guess = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

            # Perform fitting
            if algo == "Differential Evolution":
                result = differential_evolution(
                    lambda params: np.sum((func(self.x_data, *params) - self.y_data)**2),
                    bounds=bounds,
                    strategy=self.de_strategy.get(),
                    maxiter=int(self.de_maxiter.get())
                )
                popt = result.x
            elif algo == "Basin Hopping":
                result = basinhopping(
                    lambda params: np.sum((func(self.x_data, *params) - self.y_data)**2),
                    x0=initial_guess,
                    niter=int(self.bh_niter.get()),
                    stepsize=float(self.bh_stepsize.get())
                )
                popt = result.x
            elif algo == "SHGO":
                result = shgo(
                    lambda params: np.sum((func(self.x_data, *params) - self.y_data)**2),
                    bounds=bounds,
                    sampling_method=self.shgo_sampling.get()
                )
                popt = result.x
            elif algo == "Dual Annealing":
                result = dual_annealing(
                    lambda params: np.sum((func(self.x_data, *params) - self.y_data)**2),
                    bounds=bounds,
                    maxiter=int(self.da_maxiter.get())
                )
                popt = result.x
                
            # Calculate R²
            residuals = self.y_data - func(self.x_data, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((self.y_data - np.mean(self.y_data))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Display results
            output = "Fitted parameters:\n"
            for param, value in zip(params, popt):
                output += f"{param}: {value:.4f}\n"
                
            if self.is_synthetic:
                output += "\nTrue parameters:\n"
                true_params = [1.5, 3.2, 2.8, 0.8]
                if current_func == self.functions[1]:
                    true_params += [3.8, 2.7]
                for param, value in zip(params, true_params):
                    output += f"{param}: {value}\n"
                    
            output += f"\nR²: {r_squared:.4f}"
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, output)
            
        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Error: {str(e)}", 'error')

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    style.configure('TButton', font=('Arial', 10), padding=5)
    style.configure('Accent.TButton', background='#4CAF50', foreground='white')
    style.configure('TFrame', background='#f0f0f0')
    style.configure('TLabel', background='#f0f0f0')
    style.configure('TCombobox', padding=3)
    
    app = CurveFittingApp(root)
    root.mainloop()