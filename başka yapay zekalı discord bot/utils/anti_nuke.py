import time
import logging
import discord
from collections import defaultdict, deque

logger = logging.getLogger('discord_bot')

class AntiNuke:
    def __init__(self, heat_system):
        self.heat_system = heat_system
        
        # Track admin actions
        self.admin_actions = defaultdict(lambda: deque(maxlen=50))
        
        # Action thresholds
        self.thresholds = {
            'channel_delete': {'count': 3, 'window': 30},  # 3 channels deleted in 30 seconds
            'role_delete': {'count': 3, 'window': 30},     # 3 roles deleted in 30 seconds
            'channel_create': {'count': 5, 'window': 30},  # 5 channels created in 30 seconds
            'role_create': {'count': 5, 'window': 30},     # 5 roles created in 30 seconds
            'permission_change': {'count': 5, 'window': 30}  # 5 permission changes in 30 seconds
        }
        
        # Heat penalties
        self.nuke_attempt_penalty = 100  # Severe penalty for nuke attempts
        
        logger.info("Anti-nuke protection initialized")
    
    async def log_action(self, guild, user_id, action_type, target_id=None):
        """Log an administrative action"""
        current_time = time.time()
        action_data = {
            'time': current_time,
            'action': action_type,
            'target_id': target_id
        }
        
        # Add action to user's history
        self.admin_actions[user_id].append(action_data)
        
        # Check for suspicious activity
        if await self._check_nuke_attempt(guild, user_id, action_type):
            return True
        
        return False
    
    async def _check_nuke_attempt(self, guild, user_id, action_type):
        """Check if recent actions constitute a nuke attempt"""
        if action_type not in self.thresholds:
            return False
        
        current_time = time.time()
        threshold = self.thresholds[action_type]
        
        # Count actions of this type within the time window
        action_count = sum(
            1 for action in self.admin_actions[user_id]
            if action['action'] == action_type and current_time - action['time'] <= threshold['window']
        )
        
        if action_count >= threshold['count']:
            await self._handle_nuke_attempt(guild, user_id, action_type, action_count)
            return True
        
        return False
    
    async def _handle_nuke_attempt(self, guild, user_id, action_type, action_count):
        """Handle a potential nuke attempt"""
        # Add heat to the user
        current_heat = self.heat_system.add_heat(user_id, self.nuke_attempt_penalty)
        
        # Log the nuke attempt
        logger.warning(f"Potential nuke attempt detected in {guild.name}: User {user_id} performed {action_count} {action_type} actions")
        
        # Get the member object
        member = guild.get_member(user_id)
        if not member:
            logger.error(f"Could not find member with ID {user_id}")
            return
        
        # Find a logging channel
        log_channel = discord.utils.get(guild.text_channels, name='mod-logs')
        if not log_channel:
            log_channel = discord.utils.get(guild.text_channels, name='logs')
        if not log_channel:
            log_channel = discord.utils.get(guild.text_channels, name='general')
        
        # Create an embed for the nuke alert
        embed = discord.Embed(
            title="🚨 Anti-Nuke Alert",
            description=f"Suspicious administrative activity detected.",
            color=discord.Color.dark_red()
        )
        embed.add_field(
            name="Details", 
            value=f"User {member.mention} performed {action_count} {action_type.replace('_', ' ')} actions in a short time.", 
            inline=False
        )
        
        # Take action based on heat level
        action_taken = "User is being monitored."
        
        # If heat is high enough, remove admin roles
        if self.heat_system.should_kick(user_id) or self.heat_system.should_ban(user_id):
            # Find admin roles
            admin_roles = [role for role in member.roles if role.permissions.administrator or role.permissions.manage_guild]
            
            if admin_roles:
                try:
                    # Remove admin roles
                    for role in admin_roles:
                        await member.remove_roles(role, reason="Anti-nuke protection triggered")
                    
                    action_taken = f"Removed administrative roles: {', '.join([role.name for role in admin_roles])}"
                    logger.info(f"Removed admin roles from user {member.name} ({user_id}) due to nuke attempt")
                except Exception as e:
                    logger.error(f"Failed to remove admin roles: {e}")
                    action_taken = "Failed to remove administrative roles due to an error."
        
        embed.add_field(name="Action Taken", value=action_taken, inline=False)
        
        # Send the alert
        if log_channel:
            try:
                await log_channel.send(embed=embed)
            except Exception as e:
                logger.error(f"Failed to send nuke alert: {e}")
                
        # If this is a severe case, also send a DM to the server owner
        if self.heat_system.should_ban(user_id):
            try:
                owner = guild.owner
                if owner:
                    await owner.send(f"🚨 **URGENT SECURITY ALERT** 🚨\n\nPotential server nuke attempt detected in {guild.name}.\nUser {member.name} ({user_id}) performed suspicious administrative actions.\nPlease check your server immediately.")
            except Exception as e:
                logger.error(f"Failed to send DM to server owner: {e}")