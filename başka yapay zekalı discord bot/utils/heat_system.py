import time
import logging
from collections import defaultdict

logger = logging.getLogger('discord_bot')

class HeatSystem:
    def __init__(self):
        # User ID -> heat level
        self.heat_levels = defaultdict(float)
        # User ID -> last actions timestamps (for decay calculation)
        self.last_action = defaultdict(lambda: time.time())
        # Heat decay rate (points per second)
        self.decay_rate = 0.01
        # Heat thresholds for different actions
        self.thresholds = {
            'warning': 30,
            'timeout': 60,
            'kick': 100,
            'ban': 150
        }
        # Timeout durations (in seconds) based on heat level
        self.timeout_durations = {
            'low': 60 * 5,  # 5 minutes
            'medium': 60 * 30,  # 30 minutes
            'high': 60 * 60 * 3,  # 3 hours
            'extreme': 60 * 60 * 24  # 24 hours
        }
        logger.info("Heat system initialized")
    
    def add_heat(self, user_id, amount):
        """Add heat to a user"""
        # Update last action time
        self.last_action[user_id] = time.time()
        
        # Apply heat
        self.heat_levels[user_id] += amount
        logger.info(f"Added {amount} heat to user {user_id}. New heat level: {self.heat_levels[user_id]}")
        
        return self.heat_levels[user_id]
    
    def get_heat(self, user_id):
        """Get current heat level for a user (with decay)"""
        # Calculate time since last action
        time_elapsed = time.time() - self.last_action.get(user_id, time.time())
        
        # Apply decay
        current_heat = self.heat_levels.get(user_id, 0)
        decayed_heat = max(0, current_heat - (self.decay_rate * time_elapsed))
        
        # Update stored heat level
        if decayed_heat != current_heat:
            self.heat_levels[user_id] = decayed_heat
            self.last_action[user_id] = time.time()
        
        return decayed_heat
    
    def reset_heat(self, user_id):
        """Reset a user's heat level to 0"""
        if user_id in self.heat_levels:
            self.heat_levels[user_id] = 0
            self.last_action[user_id] = time.time()
            logger.info(f"Reset heat for user {user_id}")
    
    def get_timeout_duration(self, heat_level):
        """Get appropriate timeout duration based on heat level"""
        if heat_level >= self.thresholds['ban']:
            return self.timeout_durations['extreme']
        elif heat_level >= self.thresholds['kick']:
            return self.timeout_durations['high']
        elif heat_level >= self.thresholds['timeout']:
            return self.timeout_durations['medium']
        else:
            return self.timeout_durations['low']
    
    def should_warn(self, user_id):
        """Check if user should be warned"""
        return self.get_heat(user_id) >= self.thresholds['warning']
    
    def should_timeout(self, user_id):
        """Check if user should be timed out"""
        return self.get_heat(user_id) >= self.thresholds['timeout']
    
    def should_kick(self, user_id):
        """Check if user should be kicked"""
        # We're not kicking users, only using timeout
        return False
    
    def should_ban(self, user_id):
        """Check if user should be banned"""
        # We're not banning users, only using timeout
        return False