import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import subprocess
import os
import sys
import threading
import queue
import MyUtils.Util.Misc as util
from omegaconf import OmegaConf
import signal
import yaml
import wandb
import multiprocessing
import platform

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory to the script directory
os.chdir(script_dir)

class SweepAgentLauncher:
    def __init__(self, master):
        self.master = master
        master.title("Sweep Agent Launcher")

        # Load configuration
        # get the first config file in the current directory
        config_files = [f for f in os.listdir(".") if f.endswith(".yaml") and 'config' in f.lower() and 'sweep' not in f.lower()]
        if config_files:
            config_name = os.path.splitext(config_files[0])[0]
            self.cfg = util.load_and_override_config(".", config_name)
        else:
            raise ValueError("No config file found in the current directory.")

        # Create and set up widgets
        self.create_widgets()

        # Initialize agent outputs and queues
        self.agent_outputs = {}
        self.output_queues = {}
        self.agent_processes = {}
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        # Create a notebook for different tabs
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(expand=True, fill='both')

        # Create tabs
        self.agent_tab = ttk.Frame(self.notebook)
        self.sweep_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.agent_tab, text='Launch Agents')
        self.notebook.add(self.sweep_tab, text='Create Sweep')

        # Agent Tab
        self.create_agent_section()

        # Sweep Tab
        self.create_sweep_tab()

    def create_agent_section(self):
        # Project name
        ttk.Label(self.agent_tab, text="Project Name:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.project_name = ttk.Entry(self.agent_tab)
        self.project_name.insert(0, self.cfg.project_name)
        self.project_name.grid(row=0, column=1, padx=5, pady=5)

        # Username
        ttk.Label(self.agent_tab, text="Username:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.username = ttk.Entry(self.agent_tab)
        self.username.insert(0, self.cfg.username)
        self.username.grid(row=1, column=1, padx=5, pady=5)

        # Sweep ID
        ttk.Label(self.agent_tab, text="Sweep ID:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.sweep_id = ttk.Entry(self.agent_tab)
        self.sweep_id.insert(0, self.get_sweep_id())
        self.sweep_id.grid(row=2, column=1, padx=5, pady=5)

        # Add a button to create a new sweep
        self.create_sweep_button = ttk.Button(self.agent_tab, text="Create New Sweep", command=self.create_new_sweep)
        self.create_sweep_button.grid(row=2, column=2, padx=5, pady=5)

        # Number of agents
        ttk.Label(self.agent_tab, text="Number of Agents:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.num_agents = ttk.Spinbox(self.agent_tab, from_=1, to=10, width=5)
        self.num_agents.set(1)
        self.num_agents.grid(row=3, column=1, padx=5, pady=5)

        # Launch button
        self.launch_button = ttk.Button(self.agent_tab, text="Launch Agents", command=self.launch_agents)
        self.launch_button.grid(row=4, column=0, pady=10)

        # Copy command button
        self.copy_command_button = ttk.Button(self.agent_tab, text="Copy Agent Command", command=self.copy_agent_command)
        self.copy_command_button.grid(row=4, column=1, pady=10, padx=5)

        # Create a notebook for agent outputs
        self.agent_notebook = ttk.Notebook(self.agent_tab)
        self.agent_notebook.grid(row=5, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")

        # Configure row and column weights
        self.agent_tab.grid_rowconfigure(5, weight=1)
        self.agent_tab.grid_columnconfigure(0, weight=1)
        self.agent_tab.grid_columnconfigure(1, weight=1)
        self.agent_tab.grid_columnconfigure(2, weight=1)

    def create_sweep_tab(self):
        # Create two frames side by side
        config_frame = ttk.Frame(self.sweep_tab)
        sweep_frame = ttk.Frame(self.sweep_tab)
        config_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        sweep_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.sweep_tab.columnconfigure(0, weight=1)
        self.sweep_tab.columnconfigure(1, weight=1)

        # Config Frame
        ttk.Label(config_frame, text="Main Config").grid(row=0, column=0, columnspan=2, pady=5)
        
        self.config_text = scrolledtext.ScrolledText(config_frame, wrap=tk.WORD, width=40, height=20)
        self.config_text.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        self.config_text.tag_configure("bold", font=("TkDefaultFont", 10, "bold"))
        self.config_text.tag_configure("green", foreground="green")
        
        load_config_button = ttk.Button(config_frame, text="Load Config", command=self.load_config)
        load_config_button.grid(row=2, column=0, padx=5, pady=5)
        
        save_config_button = ttk.Button(config_frame, text="Save Config", command=self.save_config)
        save_config_button.grid(row=2, column=1, padx=5, pady=5)

        # Sweep Frame
        ttk.Label(sweep_frame, text="Sweep Config").grid(row=0, column=0, columnspan=2, pady=5)
        
        self.sweep_config_text = scrolledtext.ScrolledText(sweep_frame, wrap=tk.WORD, width=40, height=20)
        self.sweep_config_text.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        self.sweep_config_text.tag_configure("bold", font=("TkDefaultFont", 10, "bold"))
        self.sweep_config_text.tag_configure("green", foreground="green")
        self.sweep_config_text.tag_configure("red", foreground="red")
        
        load_sweep_button = ttk.Button(sweep_frame, text="Load Sweep", command=self.load_sweep_config)
        load_sweep_button.grid(row=2, column=0, padx=5, pady=5)
        
        save_sweep_button = ttk.Button(sweep_frame, text="Save Sweep", command=self.save_sweep_config)
        save_sweep_button.grid(row=2, column=1, padx=5, pady=5)

        # Validate and Create Sweep buttons
        validate_button = ttk.Button(self.sweep_tab, text="Validate Sweep Config", command=self.validate_sweep_config)
        validate_button.grid(row=1, column=0, padx=5, pady=5)

        create_sweep_button = ttk.Button(self.sweep_tab, text="Create Sweep", command=self.create_sweep)
        create_sweep_button.grid(row=1, column=1, padx=5, pady=5)

        # Configure weights
        config_frame.columnconfigure(0, weight=1)
        config_frame.columnconfigure(1, weight=1)
        config_frame.rowconfigure(1, weight=1)
        sweep_frame.columnconfigure(0, weight=1)
        sweep_frame.columnconfigure(1, weight=1)
        sweep_frame.rowconfigure(1, weight=1)

        # Load initial config
        self.load_initial_config()

        # Add these lines after creating the text widgets
        self.config_text.bind('<<Modified>>', self.on_config_modified)
        self.sweep_config_text.bind('<<Modified>>', self.on_sweep_config_modified)

    def load_initial_config(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load main config
        config_files = [f for f in os.listdir(script_dir) if 'config' in f.lower() and 'sweep' not in f.lower() and f.endswith('.yaml')]
        if config_files:
            config_path = os.path.join(script_dir, config_files[0])
            with open(config_path, 'r') as f:
                config_content = f.read()
            self.config_text.delete('1.0', tk.END)
            self.config_text.insert(tk.END, config_content)

        # Load sweep config
        sweep_config_files = [f for f in os.listdir(script_dir) if 'sweep' in f.lower() and 'config' in f.lower() and f.endswith('.yaml')]
        if sweep_config_files:
            sweep_config_path = os.path.join(script_dir, sweep_config_files[0])
            with open(sweep_config_path, 'r') as f:
                sweep_config_content = f.read()
            self.sweep_config_text.delete('1.0', tk.END)
            self.sweep_config_text.insert(tk.END, sweep_config_content)

    def load_config(self):
        file_path = filedialog.askopenfilename(filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")])
        if file_path:
            with open(file_path, 'r') as file:
                config_content = file.read()
                self.config_text.delete('1.0', tk.END)
                self.config_text.insert(tk.END, config_content)

    def save_config(self):
        config_content = self.config_text.get('1.0', tk.END).strip()
        try:
            # Validate the YAML content
            yaml.safe_load(config_content)
            
            file_path = filedialog.asksaveasfilename(
                defaultextension=".yaml",
                filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")],
                initialdir=os.path.dirname(os.path.abspath(__file__)),
                initialfile="config.yaml"
            )
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(config_content)
                messagebox.showinfo("Success", f"Configuration saved to: {file_path}")
        except yaml.YAMLError as e:
            messagebox.showerror("Error", f"Invalid YAML: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")

    def load_sweep_config(self):
        file_path = filedialog.askopenfilename(filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")])
        if file_path:
            with open(file_path, 'r') as file:
                config_content = file.read()
                self.sweep_config_text.delete('1.0', tk.END)
                self.sweep_config_text.insert(tk.END, config_content)

    def validate_sweep_config(self):
        config_content = self.sweep_config_text.get('1.0', tk.END)
        try:
            yaml.safe_load(config_content)
            messagebox.showinfo("Validation", "Sweep configuration is valid.")
        except yaml.YAMLError as e:
            messagebox.showerror("Validation Error", f"Invalid YAML: {str(e)}")

    def create_sweep(self):
        config_content = self.sweep_config_text.get('1.0', tk.END)
        try:
            sweep_config = yaml.safe_load(config_content)
            sweep_id = wandb.sweep(sweep_config, project=self.cfg.project_name)
            self.sweep_id.delete(0, tk.END)
            self.sweep_id.insert(0, sweep_id)
            messagebox.showinfo("Success", f"Sweep created with ID: {sweep_id}")
            
            # Save the new sweep ID to file
            self.save_sweep_id(sweep_id)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create sweep: {str(e)}")

    def save_sweep_config(self):
        config_content = self.sweep_config_text.get('1.0', tk.END).strip()
        try:
            # Validate the YAML content
            yaml.safe_load(config_content)
            
            file_path = filedialog.asksaveasfilename(
                defaultextension=".yaml",
                filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")],
                initialdir=os.path.dirname(os.path.abspath(__file__)),
                initialfile="sweep_config.yaml"
            )
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(config_content)
                messagebox.showinfo("Success", f"Sweep configuration saved to: {file_path}")
        except yaml.YAMLError as e:
            messagebox.showerror("Error", f"Invalid YAML: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save sweep configuration: {str(e)}")

    def launch_agents(self):
        project_name = self.project_name.get()
        username = self.username.get()
        sweep_id = self.sweep_id.get()
        num_agents = int(self.num_agents.get())

        if not all([project_name, username, sweep_id]):
            messagebox.showerror("Error", "Please fill in all fields.")
            return

        for i in range(num_agents):
            agent_name = f"Agent {i+1}"
            command = f"wandb agent -p {project_name} -e {username} {sweep_id}"
            self.create_agent_tab(agent_name)
            self.run_agent(command, agent_name)

        messagebox.showinfo("Success", f"Launched {num_agents} agent(s).")

    def create_agent_tab(self, agent_name):
        output_area = scrolledtext.ScrolledText(self.agent_notebook, wrap=tk.WORD, width=80, height=20)
        output_area.pack(expand=True, fill='both')
        self.agent_notebook.add(output_area, text=agent_name)
        self.agent_outputs[agent_name] = output_area
        self.output_queues[agent_name] = queue.Queue()

    def run_agent(self, command, agent_name):
        def run_process():
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, preexec_fn=os.setsid)
            self.agent_processes[agent_name] = process
            for line in iter(process.stdout.readline, ''):
                self.output_queues[agent_name].put(line)
            process.stdout.close()

        thread = threading.Thread(target=run_process)
        thread.start()

        self.update_output(agent_name)

    def update_output(self, agent_name):
        try:
            while True:
                line = self.output_queues[agent_name].get_nowait()
                self.agent_outputs[agent_name].insert(tk.END, line)
                self.agent_outputs[agent_name].see(tk.END)
        except queue.Empty:
            pass
        self.master.after(100, lambda: self.update_output(agent_name))

    def stop_all_agents(self):
        for agent_name, process in self.agent_processes.items():
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass  # Process already terminated
        self.agent_processes.clear()

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit? This will stop all running agents."):
            self.stop_all_agents()
            self.master.destroy()

    def get_sweep_id(self):
        sweep_id = util.get_or_create_sweep_id(self.cfg.project_name, None, force_create=False, allow_create=False)
        return sweep_id if sweep_id else ''

    def save_sweep_id(self, sweep_id):
        sweep_id_folder = 'sweep_ids'
        sweep_id_file = f'{self.cfg.project_name}_sweep_id.txt'
        sweep_id_file = os.path.join(sweep_id_folder, sweep_id_file)
        
        os.makedirs(sweep_id_folder, exist_ok=True)
        with open(sweep_id_file, 'w') as file:
            file.write(sweep_id)

    def on_config_modified(self, event):
        self.config_text.edit_modified(False)  # Reset the modified flag
        self.highlight_keys()

    def on_sweep_config_modified(self, event):
        self.sweep_config_text.edit_modified(False)  # Reset the modified flag
        self.highlight_keys()

    def highlight_keys(self):
        config_content = self.config_text.get('1.0', tk.END)
        sweep_content = self.sweep_config_text.get('1.0', tk.END)

        try:
            config_data = yaml.safe_load(config_content)
            sweep_data = yaml.safe_load(sweep_content)

            config_keys = set(self.get_all_keys(config_data))
            sweep_keys = set(self.get_all_keys(sweep_data, is_sweep=True))
            sweep_param_keys = set(self.get_all_keys(sweep_data.get('parameters', {})))

            # Highlight config keys
            self.highlight_text_widget(self.config_text, config_content, config_keys, sweep_param_keys)

            # Highlight sweep keys
            self.highlight_text_widget(self.sweep_config_text, sweep_content, sweep_keys, config_keys, sweep_param_keys)

        except yaml.YAMLError:
            # Ignore YAML errors during typing
            pass

    def highlight_text_widget(self, text_widget, content, keys, other_keys, sweep_param_keys=None):
        # Store the current cursor position
        cursor_pos = text_widget.index(tk.INSERT)

        # Remove existing tags
        for tag in ["bold", "green", "red"]:
            text_widget.tag_remove(tag, "1.0", tk.END)

        lines = content.split('\n')
        for line_num, line in enumerate(lines, start=1):
            stripped_line = line.strip()
            if ':' in stripped_line:
                key = stripped_line.split(':', 1)[0].strip()
                start = f"{line_num}.{line.index(key)}"
                end = f"{start}+{len(key)}c"
                text_widget.tag_add("bold", start, end)
                
                if sweep_param_keys is not None:  # This is the sweep config
                    if key in sweep_param_keys and key in other_keys:
                        text_widget.tag_add("green", start, end)
                    elif key in sweep_param_keys and key not in other_keys:
                        text_widget.tag_add("red", start, end)
                else:  # This is the main config
                    if key in other_keys:
                        text_widget.tag_add("green", start, end)

        # Restore the cursor position
        text_widget.mark_set(tk.INSERT, cursor_pos)

    def get_all_keys(self, data, prefix='', is_sweep=False):
        keys = []
        if isinstance(data, dict):
            for key, value in data.items():
                full_key = f"{prefix}{key}" if prefix else key
                keys.append(full_key)
                if is_sweep and key == 'parameters':
                    keys.extend(self.get_all_keys(value, '', is_sweep))
                else:
                    keys.extend(self.get_all_keys(value, f"{full_key}.", is_sweep))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                keys.extend(self.get_all_keys(item, f"{prefix}{i}.", is_sweep))
        return keys

    def create_new_sweep(self):
        config_content = self.sweep_config_text.get('1.0', tk.END)
        try:
            sweep_config = yaml.safe_load(config_content)
            sweep_id = util.get_or_create_sweep_id(self.cfg.project_name, sweep_config, force_create=True)
            if sweep_id:
                self.sweep_id.delete(0, tk.END)
                self.sweep_id.insert(0, sweep_id)
                messagebox.showinfo("Success", f"New sweep created with ID: {sweep_id}")
            else:
                messagebox.showerror("Error", "Failed to create new sweep.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create new sweep: {str(e)}")

    def copy_agent_command(self):
        project_name = self.project_name.get()
        username = self.username.get()
        sweep_id = self.sweep_id.get()

        if not all([project_name, username, sweep_id]):
            messagebox.showerror("Error", "Please fill in all fields.")
            return

        # Get the directory of the InitAgent.py file
        init_agent_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the command with cd
        command = f"cd {init_agent_dir} && wandb agent -p {project_name} -e {username} {sweep_id}"

        self.master.clipboard_clear()
        self.master.clipboard_append(command)
        messagebox.showinfo("Success", "Agent command copied to clipboard!")

if __name__ == "__main__":
    root = tk.Tk()
    app = SweepAgentLauncher(root)
    root.mainloop()
