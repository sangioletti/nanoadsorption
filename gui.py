import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import subprocess
import re
import shutil
import os
import ast
from PIL import Image, ImageTk

# ------------------ CONFIGURABLE PATHS ------------------
SOURCE_SYSTEM_VARS_FILE = 'system_variables.py'
MAIN_SCRIPT = 'main.py'
PLOT_FILE_NAME = 'adsorption.png'
DATA_FILE_NAME = 'adsorption.dat'

# ------------------ FUNCTIONS ------------------

def load_variables():
    """Parse variables and expressions from system_variables.py"""
    variables = {}
    with open(SOURCE_SYSTEM_VARS_FILE, 'r') as f:
        lines = f.readlines()

    # Match top-level assignments, excluding data_polymers
    pattern = re.compile(r'^(\w+)\s*=\s*(.+?)(\s*#.*)?$')

    for idx, line in enumerate(lines):
        match = pattern.match(line)
        if match:
            var_name = match.group(1)

            # Skip data_polymers
            if var_name == 'data_polymers':
                continue

            expression = match.group(2).strip()
            comment = match.group(3).strip() if match.group(3) else ""
            variables[var_name] = {
                'expression': expression,
                'comment': comment,
                'line_index': idx
            }

    return variables, lines

def create_experiment_folder(experiment_name):
    """Create a folder for the experiment"""
    folder = os.path.join(os.getcwd(), experiment_name)
    os.makedirs(folder, exist_ok=True)
    return folder

def save_variables_to_experiment(updated_vars, original_lines, folder):
    """Write system_variables_experiment.py inside the experiment folder"""
    pattern = re.compile(r'^(\w+)\s*=\s*(.+?)(\s*#.*)?$')
    new_lines = original_lines.copy()

    for var_name, update in updated_vars.items():
        expression = update['expression']
        comment = update['comment']
        line_index = update['line_index']
        comment_str = f' {comment}' if comment else ''
        new_line = f"{var_name} = {expression}{comment_str}\n"

        match = pattern.match(original_lines[line_index])
        if match and match.group(1) == var_name:
            new_lines[line_index] = new_line
        else:
            print(f"Warning: Skipping update for {var_name}, line mismatch.")

    experiment_vars_file = os.path.join(folder, 'system_variables_experiment.py')
    with open(experiment_vars_file, 'w') as f:
        f.writelines(new_lines)

    print(f"Saved system_variables_experiment.py to {folder}")
    return experiment_vars_file

def create_temp_main_py(experiment_folder):
    """Create temp_main.py that imports system_variables_experiment.py from experiment folder"""
    with open(MAIN_SCRIPT, 'r') as f:
        main_lines = f.readlines()

    new_main_lines = []
    for line in main_lines:
        if 'import system_variables' in line:
            new_line = line.replace('import system_variables', 'import system_variables_experiment as system_variables')
            new_main_lines.append(new_line)
        else:
            new_main_lines.append(line)

    temp_main_script = os.path.join(experiment_folder, 'temp_main.py')
    with open(temp_main_script, 'w') as f:
        f.writelines(new_main_lines)

    return temp_main_script

def validate_expressions(updated_vars):
    """Check if expressions are valid Python"""
    for var_name, update in updated_vars.items():
        expr = update['expression']
        try:
            ast.parse(expr)
        except SyntaxError as e:
            messagebox.showerror("Syntax Error", f"Invalid expression for '{var_name}':\n{expr}\n\n{str(e)}")
            return False
    return True

def run_main_py(experiment_folder):
    """Run temp_main.py and move results into experiment folder"""
    try:
        temp_main = create_temp_main_py(experiment_folder)
        result = subprocess.run(['python', temp_main], capture_output=True, text=True)

        os.remove(temp_main)  # Clean up temp_main.py

        # Move output files if they exist
        if os.path.exists(PLOT_FILE_NAME):
            shutil.move(PLOT_FILE_NAME, os.path.join(experiment_folder, PLOT_FILE_NAME))

        if os.path.exists(DATA_FILE_NAME):
            shutil.move(DATA_FILE_NAME, os.path.join(experiment_folder, DATA_FILE_NAME))

        if result.returncode == 0:
            messagebox.showinfo("Success", f"main.py executed successfully!\n\n{result.stdout}")
        else:
            messagebox.showerror("Error", f"main.py failed:\n\n{result.stderr}")

        # Show plot from experiment folder if available
        plot_path = os.path.join(experiment_folder, PLOT_FILE_NAME)
        if os.path.exists(plot_path):
            show_plot(plot_path)
        else:
            print(f"No plot found in: {plot_path}")

    except Exception as e:
        messagebox.showerror("Execution Error", str(e))

def show_plot(filepath):
    """Preview the generated plot inside a pop-up window"""
    plot_window = tk.Toplevel(root)
    plot_window.title(f"Adsorption Plot - {filepath}")

    img = Image.open(filepath)
    img = img.resize((600, 400), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(img)

    label = tk.Label(plot_window, image=photo)
    label.image = photo  # Keep a reference!
    label.pack()

def save_and_run():
    """Save updated variables, create experiment folder, and run the simulation"""
    experiment_name = experiment_name_entry.get().strip()

    if not experiment_name:
        messagebox.showerror("Missing Experiment Name", "Please enter a name for the experiment.")
        return

    updated_vars = {}
    for var_name, widgets in entries.items():
        expr_text = widgets['expression'].get()
        comment_text = widgets['comment'].get()
        updated_vars[var_name] = {
            'expression': expr_text,
            'comment': comment_text,
            'line_index': variables[var_name]['line_index']
        }

    if not validate_expressions(updated_vars):
        return

    # Create experiment folder and save variables
    folder = create_experiment_folder(experiment_name)
    save_variables_to_experiment(updated_vars, original_lines, folder)

    # Run the simulation
    run_main_py(folder)

# ------------------ GUI SETUP ------------------

root = tk.Tk()
root.title("System Variables Experiment Editor")
root.geometry("950x650")

# Experiment name input
experiment_name_frame = tk.Frame(root)
experiment_name_frame.pack(fill=tk.X, pady=10)

tk.Label(experiment_name_frame, text="Experiment Name:", font=('Arial', 12, 'bold')).pack(side=tk.LEFT, padx=5)
experiment_name_entry = tk.Entry(experiment_name_frame, width=40)
experiment_name_entry.pack(side=tk.LEFT, padx=5)

# Main frame to hold everything
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=1)

# Canvas inside the frame
canvas = tk.Canvas(main_frame)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

# Scrollbar for the canvas
scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Configure canvas scrolling
canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

# Frame inside the canvas (the scrollable area)
widget_frame = tk.Frame(canvas)
canvas.create_window((0, 0), window=widget_frame, anchor="nw")

# Load variables (excluding data_polymers)
variables, original_lines = load_variables()

entries = {}

# Headers
header_font = ('Arial', 10, 'bold')

tk.Label(widget_frame, text="Variable", font=header_font).grid(row=0, column=0, padx=5, pady=5)
tk.Label(widget_frame, text="Expression", font=header_font).grid(row=0, column=1, padx=5, pady=5)
tk.Label(widget_frame, text="Comment", font=header_font).grid(row=0, column=2, padx=5, pady=5)

# Populate variable rows
row = 1
for var_name, var_data in variables.items():
    tk.Label(widget_frame, text=var_name).grid(row=row, column=0, padx=5, pady=5, sticky='w')

    expr_entry = tk.Entry(widget_frame, width=50)
    expr_entry.insert(0, var_data['expression'])
    expr_entry.grid(row=row, column=1, padx=5, pady=5, sticky='w')

    comment_entry = tk.Entry(widget_frame, width=30)
    comment_entry.insert(0, var_data['comment'])
    comment_entry.grid(row=row, column=2, padx=5, pady=5, sticky='w')

    entries[var_name] = {
        'expression': expr_entry,
        'comment': comment_entry
    }

    row += 1

# Save & Run Button
bottom_frame = tk.Frame(root)
bottom_frame.pack(fill=tk.X, pady=10)

save_run_button = tk.Button(bottom_frame, text="Save & Run main.py", command=save_and_run)
save_run_button.pack(pady=5)

# Mouse wheel scrolling support
def _on_mousewheel(event):
    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

canvas.bind_all("<MouseWheel>", _on_mousewheel)
canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))  # Linux scroll up
canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))   # Linux scroll down

# ------------------ MAIN LOOP ------------------
root.mainloop()
