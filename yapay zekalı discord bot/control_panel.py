import sys
import os
import json
import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QCheckBox, QSlider, QTextEdit, QPushButton, 
                             QScrollArea, QFrame, QSplitter, QTableWidget, QTableWidgetItem,
                             QHeaderView, QComboBox, QLineEdit, QMessageBox, QGroupBox, QFormLayout)
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QRect, QSize, pyqtSignal, pyqtProperty, QProcess, QFileSystemWatcher
from PyQt5.QtGui import QColor, QPalette, QFont, QIcon, QPixmap, QPainter
import qdarkstyle
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
APP_TITLE = "Discord Bot Güvenlik Kontrol Paneli"
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
WARNING_DATA_FILE = 'warning_data.json'
LOG_FILE = 'bot_log.txt'

# Utility function to write logs
def write_log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}\n"
    
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_message)
    
    if DEBUG_MODE:
        print(log_message)

# Custom styled components
class StyledButton(QPushButton):
    def __init__(self, text, parent=None, primary=False):
        super().__init__(text, parent)
        self.primary = primary
        self.setMinimumHeight(36)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(
            f"""QPushButton {{
                background-color: {'#5865F2' if primary else '#4f545c'};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {'#4752c4' if primary else '#36393f'};
            }}
            QPushButton:pressed {{
                background-color: {'#3c45a5' if primary else '#202225'};
            }}"""
        )

class AnimatedToggle(QCheckBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(60, 30)
        self._bg_color = QColor('#777')
        self._circle_color = QColor('#DDD')
        self._active_color = QColor('#5865F2')
        self._circle_position = 3
        self.animation = QPropertyAnimation(self, b"circle_position")
        self.animation.setEasingCurve(QEasingCurve.OutBounce)
        self.animation.setDuration(500)
        self.stateChanged.connect(self._setup_animation)
        self.setCursor(Qt.PointingHandCursor)
    
    @pyqtProperty(int)
    def circle_position(self):
        return self._circle_position
    
    @circle_position.setter
    def circle_position(self, pos):
        self._circle_position = pos
        self.update()
    
    def _setup_animation(self, value):
        self.animation.stop()
        if value:
            self.animation.setStartValue(self.circle_position)
            self.animation.setEndValue(self.width() - 27)
        else:
            self.animation.setStartValue(self.circle_position)
            self.animation.setEndValue(3)
        self.animation.start()
    
    def hitButton(self, pos):
        return self.contentsRect().contains(pos)
    
    def paintEvent(self, e):
        p = QPalette()
        p.setColor(QPalette.Background, self._bg_color if not self.isChecked() else self._active_color)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        
        # Draw background
        painter.setBrush(self._bg_color if not self.isChecked() else self._active_color)
        painter.drawRoundedRect(0, 0, self.width(), self.height(), 15, 15)
        
        # Draw circle
        painter.setBrush(self._circle_color)
        painter.drawEllipse(self._circle_position, 3, 24, 24)

class LogViewer(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 9))
        self.setStyleSheet(
            """QTextEdit {
                background-color: #2f3136;
                color: #dcddde;
                border: 1px solid #202225;
                border-radius: 4px;
            }"""
        )
    
    def append_log(self, text):
        self.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {text}")
        # Auto-scroll to bottom
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

class WarningTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(["Kullanıcı ID", "Uyarı Sayısı", "Heat Puanı"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setAlternatingRowColors(True)
        self.setStyleSheet(
            """QTableWidget {
                background-color: #2f3136;
                color: #dcddde;
                border: 1px solid #202225;
                border-radius: 4px;
                gridline-color: #40444b;
            }
            QHeaderView::section {
                background-color: #202225;
                color: #dcddde;
                padding: 5px;
                border: 1px solid #40444b;
            }
            QTableWidget::item:alternate {
                background-color: #36393f;
            }"""
        )

class SettingsGroup(QGroupBox):
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setStyleSheet(
            """QGroupBox {
                font-weight: bold;
                border: 1px solid #40444b;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }"""
        )

# Main application window
class ControlPanel(QMainWindow):
    def __init__(self):
        super().__init__()
        self.bot_process = None  # Add bot process handler
        self.log_watcher = QFileSystemWatcher()  # Add file watcher
        
        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(1000, 700)
        
        # Initialize UI
        self.init_ui()
        
        # Load settings and data
        self.load_settings()
        self.load_warning_data()
        self.load_logs()
        
        # Setup auto-refresh
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_data)
        self.refresh_timer.start(5000)  # Refresh every 5 seconds
        
        # Setup log file watcher
        self.log_watcher.addPath(LOG_FILE)
        self.log_watcher.fileChanged.connect(self.handle_log_update)
        
        # Log application start
        write_log("Control Panel application started")
        
        # Start the bot process
        self.start_bot()
    
    def init_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create header
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 10, 10, 10)
        
        # Header title
        title_label = QLabel(APP_TITLE)
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        header_layout.addWidget(title_label)
        
        # Status indicator
        self.status_label = QLabel("Bot Durumu: Bağlanıyor...")
        self.status_label.setStyleSheet("color: #FAA61A;")
        header_layout.addWidget(self.status_label, alignment=Qt.AlignRight)
        
        main_layout.addWidget(header)
        
        # Add separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #40444b;")
        main_layout.addWidget(separator)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setMovable(True)
        
        # Create tabs
        self.create_dashboard_tab()
        self.create_settings_tab()
        self.create_logs_tab()
        self.create_warnings_tab()
        
        main_layout.addWidget(self.tabs)
        
        # Create footer
        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(10, 5, 10, 5)
        
        # Version info
        version_label = QLabel("v1.0.0")
        footer_layout.addWidget(version_label)
        
        # Save button
        save_button = StyledButton("Ayarları Kaydet", primary=True)
        save_button.clicked.connect(self.save_settings)
        footer_layout.addWidget(save_button, alignment=Qt.AlignRight)
        
        main_layout.addWidget(footer)
    
    def create_dashboard_tab(self):
        dashboard = QWidget()
        layout = QVBoxLayout(dashboard)
        
        # Status cards
        status_layout = QHBoxLayout()
        
        # Bot Status Card
        bot_status = QFrame()
        bot_status.setFrameShape(QFrame.StyledPanel)
        bot_status.setStyleSheet(
            """QFrame {
                background-color: #36393f;
                border-radius: 8px;
                padding: 10px;
            }"""
        )
        bot_status_layout = QVBoxLayout(bot_status)
        bot_status_layout.addWidget(QLabel("Bot Durumu"))
        self.bot_status_value = QLabel("Çevrimiçi")
        self.bot_status_value.setStyleSheet("color: #43b581; font-size: 18px; font-weight: bold;")
        bot_status_layout.addWidget(self.bot_status_value)
        status_layout.addWidget(bot_status)
        
        # Warnings Card
        warnings_status = QFrame()
        warnings_status.setFrameShape(QFrame.StyledPanel)
        warnings_status.setStyleSheet(
            """QFrame {
                background-color: #36393f;
                border-radius: 8px;
                padding: 10px;
            }"""
        )
        warnings_status_layout = QVBoxLayout(warnings_status)
        warnings_status_layout.addWidget(QLabel("Toplam Uyarı"))
        self.warnings_count = QLabel("0")
        self.warnings_count.setStyleSheet("color: #faa61a; font-size: 18px; font-weight: bold;")
        warnings_status_layout.addWidget(self.warnings_count)
        status_layout.addWidget(warnings_status)
        
        # Timeouts Card
        timeouts_status = QFrame()
        timeouts_status.setFrameShape(QFrame.StyledPanel)
        timeouts_status.setStyleSheet(
            """QFrame {
                background-color: #36393f;
                border-radius: 8px;
                padding: 10px;
            }"""
        )
        timeouts_status_layout = QVBoxLayout(timeouts_status)
        timeouts_status_layout.addWidget(QLabel("Timeout Sayısı"))
        self.timeouts_count = QLabel("0")
        self.timeouts_count.setStyleSheet("color: #f04747; font-size: 18px; font-weight: bold;")
        timeouts_status_layout.addWidget(self.timeouts_count)
        status_layout.addWidget(timeouts_status)
        
        layout.addLayout(status_layout)
        
        # Recent activity
        activity_group = SettingsGroup("Son Aktiviteler")
        activity_layout = QVBoxLayout(activity_group)
        
        self.recent_activity = LogViewer()
        activity_layout.addWidget(self.recent_activity)
        
        layout.addWidget(activity_group)
        
        # Quick actions
        actions_group = SettingsGroup("Hızlı İşlemler")
        actions_layout = QHBoxLayout(actions_group)
        
        restart_bot = StyledButton("Botu Yeniden Başlat")
        restart_bot.clicked.connect(self.restart_bot)
        actions_layout.addWidget(restart_bot)
        
        clear_warnings = StyledButton("Tüm Uyarıları Temizle")
        clear_warnings.clicked.connect(self.clear_all_warnings)
        actions_layout.addWidget(clear_warnings)
        
        refresh_data = StyledButton("Verileri Yenile")
        refresh_data.clicked.connect(self.refresh_data)
        actions_layout.addWidget(refresh_data)
        
        layout.addWidget(actions_group)
        
        self.tabs.addTab(dashboard, "Gösterge Paneli")
    
    def create_settings_tab(self):
        settings = QWidget()
        layout = QVBoxLayout(settings)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        settings_content = QWidget()
        settings_layout = QVBoxLayout(settings_content)
        
        # Security Settings
        security_group = SettingsGroup("Güvenlik Ayarları")
        security_layout = QVBoxLayout(security_group)
        
        # Create toggle switches for each security feature
        features = [
            ("spam_check", "Spam Koruması", "Spam mesajları otomatik olarak tespit et ve sil"),
            ("fake_link_check", "Sahte Link Koruması", "Sahte Discord Nitro ve benzeri dolandırıcılık linklerini engelle"),
            ("duplicate_check", "Tekrarlanan Mesaj Koruması", "Aynı mesajı tekrar tekrar gönderen kullanıcıları tespit et"),
            ("profanity_check", "Küfür Koruması", "Küfür ve hakaret içeren mesajları otomatik olarak sil"),
            ("invite_check", "Davet Linki Koruması", "Discord davet linklerini otomatik olarak engelle"),
            ("mention_check", "Aşırı Mention Koruması", "@everyone, @here veya çok sayıda kullanıcı mentionlarını engelle"),
            ("caps_check", "Büyük Harf Koruması", "Tamamen büyük harflerle yazılmış mesajları engelle"),
            ("ad_check", "Reklam Koruması", "Reklam içerikli mesajları otomatik olarak tespit et ve sil"),
            ("anomaly_check", "Anomali Tespiti", "Normal olmayan davranışları tespit et ve engelle")
        ]
        
        self.security_toggles = {}
        
        for key, title, description in features:
            feature_layout = QHBoxLayout()
            
            # Toggle switch
            toggle = AnimatedToggle()
            toggle.setChecked(True)  # Default to enabled
            self.security_toggles[key] = toggle
            
            # Feature info
            info_layout = QVBoxLayout()
            title_label = QLabel(title)
            title_label.setFont(QFont("Arial", 10, QFont.Bold))
            desc_label = QLabel(description)
            desc_label.setStyleSheet("color: #72767d;")
            
            info_layout.addWidget(title_label)
            info_layout.addWidget(desc_label)
            
            feature_layout.addLayout(info_layout)
            feature_layout.addWidget(toggle, alignment=Qt.AlignRight)
            
            security_layout.addLayout(feature_layout)
            
            # Add separator except for the last item
            if key != features[-1][0]:
                separator = QFrame()
                separator.setFrameShape(QFrame.HLine)
                separator.setFrameShadow(QFrame.Sunken)
                separator.setStyleSheet("background-color: #40444b;")
                security_layout.addWidget(separator)
        
        settings_layout.addWidget(security_group)
        
        # Warning Settings
        warning_group = SettingsGroup("Uyarı Ayarları")
        warning_layout = QFormLayout(warning_group)
        
        # Max warnings before timeout
        max_warnings_layout = QHBoxLayout()
        self.max_warnings_slider = QSlider(Qt.Horizontal)
        self.max_warnings_slider.setMinimum(1)
        self.max_warnings_slider.setMaximum(10)
        self.max_warnings_slider.setValue(3)  # Default value
        self.max_warnings_value = QLabel("3")
        self.max_warnings_slider.valueChanged.connect(lambda v: self.max_warnings_value.setText(str(v)))
        
        max_warnings_layout.addWidget(self.max_warnings_slider)
        max_warnings_layout.addWidget(self.max_warnings_value)
        
        warning_layout.addRow("Maksimum Uyarı Sayısı:", max_warnings_layout)
        
        # Warning reset days
        reset_days_layout = QHBoxLayout()
        self.reset_days_slider = QSlider(Qt.Horizontal)
        self.reset_days_slider.setMinimum(1)
        self.reset_days_slider.setMaximum(30)
        self.reset_days_slider.setValue(7)  # Default value
        self.reset_days_value = QLabel("7")
        self.reset_days_slider.valueChanged.connect(lambda v: self.reset_days_value.setText(str(v)))
        
        reset_days_layout.addWidget(self.reset_days_slider)
        reset_days_layout.addWidget(self.reset_days_value)
        
        warning_layout.addRow("Uyarı Sıfırlama Günü:", reset_days_layout)
        
        # Timeout duration
        timeout_layout = QHBoxLayout()
        self.timeout_slider = QSlider(Qt.Horizontal)
        self.timeout_slider.setMinimum(5)
        self.timeout_slider.setMaximum(60)
        self.timeout_slider.setValue(30)  # Default value
        self.timeout_value = QLabel("30")
        self.timeout_slider.valueChanged.connect(lambda v: self.timeout_value.setText(str(v)))
        
        timeout_layout.addWidget(self.timeout_slider)
        timeout_layout.addWidget(self.timeout_value)
        
        warning_layout.addRow("Timeout Süresi (dakika):", timeout_layout)
        
        settings_layout.addWidget(warning_group)
        
        # Add settings content to scroll area
        scroll.setWidget(settings_content)
        layout.addWidget(scroll)
        
        self.tabs.addTab(settings, "Ayarlar")
    
    def create_logs_tab(self):
        logs = QWidget()
        layout = QVBoxLayout(logs)
        
        # Log viewer
        self.log_viewer = LogViewer()
        layout.addWidget(self.log_viewer)
        
        # Log controls
        controls_layout = QHBoxLayout()
        
        clear_logs = StyledButton("Logları Temizle")
        clear_logs.clicked.connect(self.clear_logs)
        controls_layout.addWidget(clear_logs)
        
        export_logs = StyledButton("Logları Dışa Aktar")
        export_logs.clicked.connect(self.export_logs)
        controls_layout.addWidget(export_logs)
        
        layout.addLayout(controls_layout)
        
        self.tabs.addTab(logs, "Loglar")
    
    def create_warnings_tab(self):
        warnings = QWidget()
        layout = QVBoxLayout(warnings)
        
        # Warnings table
        self.warnings_table = WarningTable()
        layout.addWidget(self.warnings_table)
        
        # Warning controls
        controls_layout = QHBoxLayout()
        
        refresh_warnings = StyledButton("Uyarıları Yenile")
        refresh_warnings.clicked.connect(self.load_warning_data)
        controls_layout.addWidget(refresh_warnings)
        
        clear_all = StyledButton("Tüm Uyarıları Temizle")
        clear_all.clicked.connect(self.clear_all_warnings)
        controls_layout.addWidget(clear_all)
        
        remove_timeout = StyledButton("Timeout Kaldır", primary=True)
        remove_timeout.clicked.connect(self.remove_user_timeout)
        controls_layout.addWidget(remove_timeout)
        
        layout.addLayout(controls_layout)
        
        self.tabs.addTab(warnings, "Uyarılar")
    
    def load_settings(self):
        # Load settings from a JSON file if it exists
        try:
            with open('bot_settings.json', 'r') as f:
                settings = json.load(f)
                
                # Apply settings to UI elements
                for key, toggle in self.security_toggles.items():
                    if key in settings:
                        toggle.setChecked(settings[key])
                
                if 'max_warnings' in settings:
                    self.max_warnings_slider.setValue(settings['max_warnings'])
                
                if 'reset_days' in settings:
                    self.reset_days_slider.setValue(settings['reset_days'])
                
                if 'timeout_duration' in settings:
                    self.timeout_slider.setValue(settings['timeout_duration'])
                
                write_log("Settings loaded successfully")
        except FileNotFoundError:
            write_log("No settings file found, using defaults")
        except Exception as e:
            write_log(f"Error loading settings: {str(e)}")
    
    def save_settings(self):
        # Save settings to a JSON file
        settings = {}
        
        # Save security toggle states
        for key, toggle in self.security_toggles.items():
            settings[key] = toggle.isChecked()
        
        # Save other settings
        settings['max_warnings'] = self.max_warnings_slider.value()
        settings['reset_days'] = self.reset_days_slider.value()
        settings['timeout_duration'] = self.timeout_slider.value()
        
        try:
            with open('bot_settings.json', 'w') as f:
                json.dump(settings, f, indent=4)
            
            write_log("Settings saved successfully")
            QMessageBox.information(self, "Bilgi", "Ayarlar başarıyla kaydedildi.")
        except Exception as e:
            write_log(f"Error saving settings: {str(e)}")
            QMessageBox.critical(self, "Hata", f"Ayarlar kaydedilirken bir hata oluştu: {str(e)}")
    
    def load_warning_data(self):
        try:
            if os.path.exists(WARNING_DATA_FILE):
                with open(WARNING_DATA_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Clear existing table data
                    self.warnings_table.setRowCount(0)
                    
                    # Get warning data
                    user_warnings = data.get("user_warnings", {})
                    user_heat_points = data.get("user_heat_points", {})
                    
                    # Populate table with data
                    row = 0
                    for user_id, warning_count in user_warnings.items():
                        heat_points = user_heat_points.get(user_id, 0)
                        
                        self.warnings_table.insertRow(row)
                        self.warnings_table.setItem(row, 0, QTableWidgetItem(user_id))
                        self.warnings_table.setItem(row, 1, QTableWidgetItem(str(warning_count)))
                        self.warnings_table.setItem(row, 2, QTableWidgetItem(str(heat_points)))
                        
                        row += 1
                
                write_log("Warning data loaded successfully")
            else:
                write_log("No warning data file found")
        except Exception as e:
            write_log(f"Error loading warning data: {str(e)}")
    
    def load_logs(self):
        try:
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, 'r', encoding='utf-8') as f:
                    logs = f.readlines()
                    
                    # Clear existing logs in viewers
                    self.log_viewer.clear()
                    self.recent_activity.clear()
                    
                    # Show last 100 logs in log viewer
                    for log in logs[-100:]:
                        self.log_viewer.append(log.strip())
                    
                    # Show last 10 logs in recent activity
                    for log in logs[-10:]:
                        self.recent_activity.append(log.strip())
                    
                    # Update warning and timeout counts
                    warning_count = sum(1 for log in logs if "warned for" in log)
                    timeout_count = sum(1 for log in logs if "timed out" in log)
                    
                    self.warnings_count.setText(str(warning_count))
                    self.timeouts_count.setText(str(timeout_count))
            else:
                write_log("No log file found")
        except Exception as e:
            write_log(f"Error loading logs: {str(e)}")

    def apply_timeout_to_bot(self):
        if self.bot_process and self.bot_process.state() == QProcess.Running:
            timeout_minutes = int(self.timeout_value.text())
            self.bot_process.write(f"TIMEOUT {timeout_minutes}\n".encode())
    
    def remove_user_timeout(self):
        try:
            # Get the selected user from the warnings table
            selected_rows = self.warnings_table.selectedItems()
            if not selected_rows:
                QMessageBox.warning(self, "Uyarı", "Lütfen timeout'u kaldırılacak bir kullanıcı seçin.")
                return
            
            # Get the user ID from the first column of the selected row
            user_id = self.warnings_table.item(selected_rows[0].row(), 0).text()
            
            # Add a command to the bot to remove the timeout
            if self.bot_process and self.bot_process.state() == QProcess.Running:
                # Create a new command to remove timeout
                command = f"!removetimeout {user_id}"
                self.bot_process.write(f"{command}\n".encode())
                write_log(f"Sent command to remove timeout for user {user_id}")
                QMessageBox.information(self, "Bilgi", f"Kullanıcı {user_id} için timeout kaldırma komutu gönderildi.")
            else:
                QMessageBox.warning(self, "Uyarı", "Bot çalışmıyor. Lütfen botu başlatın ve tekrar deneyin.")
        except Exception as e:
            write_log(f"Error removing timeout: {str(e)}")
            QMessageBox.critical(self, "Hata", f"Timeout kaldırılırken bir hata oluştu: {str(e)}")
    
    def refresh_data(self):
        # Check bot status
        if self.bot_process and self.bot_process.state() == QProcess.Running:
            self.bot_status_value.setText("Çevrimiçi")
            self.bot_status_value.setStyleSheet("color: #43b581; font-size: 18px; font-weight: bold;")
            self.status_label.setText("Bot Durumu: Çevrimiçi")
            self.status_label.setStyleSheet("color: #43b581;")
        else:
            self.bot_status_value.setText("Çevrimdışı")
            self.bot_status_value.setStyleSheet("color: #f04747; font-size: 18px; font-weight: bold;")
            self.status_label.setText("Bot Durumu: Çevrimdışı")
            self.status_label.setStyleSheet("color: #f04747;")
        
        # Refresh warning data
        self.load_warning_data()
        
        # Refresh logs
        self.load_logs()
    
    def start_bot(self):
        # Terminate existing process if any
        if self.bot_process and self.bot_process.state() == QProcess.Running:
            self.bot_process.terminate()
        
        # Start new bot process
        self.bot_process = QProcess()
        
        # Connect signals for process state changes
        self.bot_process.started.connect(self.handle_bot_started)
        self.bot_process.finished.connect(self.handle_bot_finished)
        self.bot_process.errorOccurred.connect(self.handle_bot_error)
        self.bot_process.readyReadStandardOutput.connect(self.handle_bot_output)
        self.bot_process.readyReadStandardError.connect(self.handle_bot_error_output)
        
        # Start the bot process
        self.bot_process.start('python', ['bot.py'])
        write_log("Bot process started")
        
    def handle_bot_started(self):
        write_log("Bot process started successfully")
        self.bot_status_value.setText("Çevrimiçi")
        self.bot_status_value.setStyleSheet("color: #43b581; font-size: 18px; font-weight: bold;")

    def handle_bot_finished(self, exitCode, exitStatus):
        write_log(f"Bot process finished with exit code {exitCode}")
        self.bot_status_value.setText("Çevrimdışı")
        self.bot_status_value.setStyleSheet("color: #f04747; font-size: 18px; font-weight: bold;")

    def handle_bot_error(self, error):
        write_log(f"Bot process error: {error.name()}")
        self.bot_status_value.setText("Hata")
        self.bot_status_value.setStyleSheet("color: #faa61a; font-size: 18px; font-weight: bold;")

    def handle_bot_error_output(self):
        try:
            error_output = self.bot_process.readAllStandardError().data().decode('latin-1', errors='replace')
        except Exception as e:
            write_log(f"Error decoding bot error output: {str(e)}")
            error_output = ""
        if error_output:
            write_log(f"[Bot Error] {error_output.strip()}")

    def handle_bot_output(self):
        output = self.bot_process.readAllStandardOutput().data().decode('latin-1').strip()
        if output:
            write_log(f"[Bot Output] {output.strip()}")

    def handle_log_update(self, path):
        # Reload logs when file changes
        self.load_logs()

    def restart_bot(self):
        try:
            write_log("Restarting bot...")
            if self.bot_process:
                if self.bot_process.state() == QProcess.Running:
                    self.bot_process.terminate()
                    # Wait for process to terminate gracefully
                    if not self.bot_process.waitForFinished(5000):
                        self.bot_process.kill()  # Force kill if not terminated
                        write_log("Bot process had to be forcefully terminated")
            
            # Start the bot again
            self.start_bot()
            QMessageBox.information(self, "Bilgi", "Bot yeniden başlatıldı.")
        except Exception as e:
            write_log(f"Error restarting bot: {str(e)}")
            QMessageBox.critical(self, "Hata", f"Bot yeniden başlatılırken bir hata oluştu: {str(e)}")
    
    def clear_all_warnings(self):
        try:
            # Clear all warnings in the warning data file
            with open(WARNING_DATA_FILE, 'wb') as f:
                f.write(json.dumps({"user_warnings": {}, "user_heat_points": {}}, ensure_ascii=False).encode('utf-8'))
            
            write_log("All warnings cleared")
            self.load_warning_data()  # Refresh the warnings table
            QMessageBox.information(self, "Bilgi", "Tüm uyarılar temizlendi.")
        except Exception as e:
            write_log(f"Error clearing warnings: {str(e)}")
            QMessageBox.critical(self, "Hata", f"Uyarılar temizlenirken bir hata oluştu: {str(e)}")
    
    def clear_logs(self):
        try:
            # Clear the log file
            with open(LOG_FILE, 'w') as f:
                f.write("")
            
            # Clear log viewers
            self.log_viewer.clear()
            self.recent_activity.clear()
            
            write_log("Logs cleared")
            QMessageBox.information(self, "Bilgi", "Loglar temizlendi.")
        except Exception as e:
            write_log(f"Error clearing logs: {str(e)}")
            QMessageBox.critical(self, "Hata", f"Loglar temizlenirken bir hata oluştu: {str(e)}")
    
    def export_logs(self):
        try:
            # Export logs to a timestamped file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            export_filename = f"logs_export_{timestamp}.txt"
            
            with open(LOG_FILE, 'r', encoding='utf-8') as src, open(export_filename, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
            
            write_log(f"Logs exported to {export_filename}")
            QMessageBox.information(self, "Bilgi", f"Loglar başarıyla dışa aktarıldı: {export_filename}")
        except Exception as e:
            write_log(f"Error exporting logs: {str(e)}")
            QMessageBox.critical(self, "Hata", f"Loglar dışa aktarılırken bir hata oluştu: {str(e)}")

# Main application entry point
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Apply dark style
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    
    # Create and show the main window
    main_window = ControlPanel()
    main_window.show()
    
    # Start the application event loop
    sys.exit(app.exec_())