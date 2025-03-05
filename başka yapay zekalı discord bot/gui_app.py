import os
import tkinter as tk
from tkinter import ttk, messagebox
import json
from utils.encryption import encrypt_data, decrypt_data

class BotConfigGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Discord Bot Control Panel")
        self.root.state('zoomed')
        self.config = self.load_config()
        self.bot_thread = None

        # Modern style configuration
        self.style = ttk.Style()
        self.style.theme_create('modern', parent='alt',
                              settings={'TNotebook': {'configure': {'tabmargins': [2, 5, 2, 0]}},
                                        'TNotebook.Tab': {'configure': {'padding': [15, 5],
                                                                      'font': ('Helvetica', 10, 'bold')}}})
        self.style.theme_use('modern')

        # Animated background
        self.canvas = tk.Canvas(root, highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.gradient_id = None
        self.animate_background()

        # Main container
        self.main_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0,0), window=self.main_frame, anchor='nw')

        # Create notebook style interface
        self.notebook = ttk.Notebook(self.main_frame)
        
        # Add modern frames
        self.create_status_frame()
        self.create_security_frame()
        self.create_features_frame()
        self.create_analytics_frame()

        # Bot control section
        self.create_bot_controls()

        # Start background tasks
        self.update_bot_status()

    def load_config(self):
        default_config = {
            "features": {
                "spam_protection": {"enabled": True},
                "link_detection": {"enabled": True},
                "heat_system": {"warning_threshold": 3}
            },
            "encrypted_settings": {
                "discord_token": "",
                "gemini_key": ""
            }
        }

        try:
            try:
                config_path = os.path.join(os.path.dirname(__file__), 'config.json')
                with open(config_path, 'r') as f:
                    config = json.load(f)
            except FileNotFoundError:
                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                config = default_config

            # Validate config structure
            if not all(key in config for key in ['features', 'encrypted_settings']):
                raise ValueError("Invalid config structure")

            # Decrypt sensitive fields
            config['encrypted_settings']['discord_token'] = decrypt_data(config['encrypted_settings'].get('discord_token', ''))
            config['encrypted_settings']['gemini_key'] = decrypt_data(config['encrypted_settings'].get('gemini_key', ''))
            return config

        except json.JSONDecodeError:
            messagebox.showerror("Error", "Invalid JSON format in config file")
            return default_config
        except Exception as e:
            messagebox.showerror("Error", f"Config error: {str(e)}")
            return default_config

    def create_security_fields(self, frame):
        ttk.Label(frame, text="Discord Bot Token:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.discord_token = ttk.Entry(frame, show='*')
        self.discord_token.insert(0, self.config.get('encrypted_settings', {}).get('discord_token', ''))
        self.discord_token.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(frame, text="Gemini API Key:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.gemini_key = ttk.Entry(frame, show='*')
        self.gemini_key.insert(0, self.config.get('encrypted_settings', {}).get('gemini_key', ''))
        self.gemini_key.grid(row=1, column=1, padx=5, pady=5)

    def create_feature_toggles(self, frame):
        # Modern toggle switches
        ttk.Label(frame, text="Bot Features:", font=('Helvetica', 12, 'bold')).grid(row=0, column=0, pady=10, sticky='w')
        
        self.feature_vars = {
            'spam': tk.BooleanVar(value=self.config.get('features', {}).get('spam_protection', {}).get('enabled', True)),
            'links': tk.BooleanVar(value=self.config.get('features', {}).get('link_detection', {}).get('enabled', True)),
            'typing_indicator': tk.BooleanVar(value=True),
            'custom_status': tk.BooleanVar(value=True)
        }

        for i, (feature, var) in enumerate(self.feature_vars.items(), start=1):
            ttk.Checkbutton(frame, text=feature.capitalize(), variable=var,
                          style='Modern.Toggle').grid(row=i, column=0, sticky='w', pady=2)

        # Status customization
        ttk.Label(frame, text="Custom Status:", font=('Helvetica', 10)).grid(row=5, column=0, pady=10, sticky='w')
        self.status_text = ttk.Entry(frame, font=('Helvetica', 10))
        self.status_text.insert(0, "Ben bir Discord botuyum")
        self.status_text.grid(row=6, column=0, sticky='ew')

        self.status_type = ttk.Combobox(frame, values=['oynuyor', 'dinliyor', 'izliyor'],
                                      font=('Helvetica', 10))
        self.status_type.current(0)
        self.status_type.grid(row=6, column=1, padx=5)

    def save_config(self):
        try:
            new_config = {
                'features': {
                    'spam_protection': {'enabled': self.spam_enabled.get()},
                    'link_detection': {'enabled': self.link_enabled.get()},
                    'heat_system': {
                        'warning_threshold': int(self.warning_threshold.get())
                    }
                },
                'encrypted_settings': {
                    'discord_token': encrypt_data(self.discord_token.get()),
                    'gemini_key': encrypt_data(self.gemini_key.get())
                }
            }
            
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')
            try:
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                with open(config_path, 'w') as f:
                    json.dump(new_config, f, indent=2)
            except PermissionError:
                raise PermissionError("Permission denied to write config file. Run as administrator?")
            except IOError as e:
                raise IOError(f"File access error: {str(e)}")
            
            messagebox.showinfo("Success", "Configuration saved successfully!")
        except PermissionError as e:
            messagebox.showerror("Permissions Error", f"Failed to save config: {str(e)}")
        except IOError as e:
            messagebox.showerror("File Error", f"Config save failed: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Save failed: {str(e)}
Check file permissions and disk space.")

if __name__ == "__main__":
    root = tk.Tk()
    app = BotConfigGUI(root)
    root.mainloop()