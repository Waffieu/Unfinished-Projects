import time
import logging
import discord
from collections import defaultdict, deque

logger = logging.getLogger('discord_bot')

class AdminMonitor:
    def __init__(self, heat_system):
        self.heat_system = heat_system
        
        # Track admin actions
        self.admin_actions = defaultdict(lambda: deque(maxlen=50))
        
        # Action thresholds
        self.thresholds = {
            'ban': {'count': 3, 'window': 60},  # 3 bans in 60 seconds
            'kick': {'count': 5, 'window': 60},  # 5 kicks in 60 seconds
        }
        
        # Heat penalties
        self.admin_abuse_penalty = 80  # Severe penalty for admin abuse
        
        logger.info("Admin monitor initialized")
    
    async def check_ban(self, guild, user):
        """Log and check a ban action"""
        # Get the audit log to find who performed the ban
        try:
            async for entry in guild.audit_logs(limit=1, action=discord.AuditLogAction.ban):
                if entry.target.id == user.id:
                    await self._log_admin_action(guild, entry.user.id, 'ban', user.id)
                    break
        except Exception as e:
            logger.error(f"Failed to check audit logs for ban: {e}")
    
    async def check_kick(self, member):
        """Log and check a kick action"""
        guild = member.guild
        
        # Get the audit log to find who performed the kick
        try:
            async for entry in guild.audit_logs(limit=1, action=discord.AuditLogAction.kick):
                if entry.target.id == member.id:
                    await self._log_admin_action(guild, entry.user.id, 'kick', member.id)
                    break
        except Exception as e:
            logger.error(f"Failed to check audit logs for kick: {e}")
    
    async def _log_admin_action(self, guild, admin_id, action_type, target_id):
        """Log an administrative action"""
        current_time = time.time()
        action_data = {
            'time': current_time,
            'action': action_type,
            'target_id': target_id
        }
        
        # Add action to admin's history
        self.admin_actions[admin_id].append(action_data)
        
        # Check for suspicious activity
        if await self._check_admin_abuse(guild, admin_id, action_type):
            return True
        
        return False
    
    async def _check_admin_abuse(self, guild, admin_id, action_type):
        """Check if recent actions constitute admin abuse"""
        if action_type not in self.thresholds:
            return False
        
        current_time = time.time()
        threshold = self.thresholds[action_type]
        
        # Count actions of this type within the time window
        action_count = sum(
            1 for action in self.admin_actions[admin_id]
            if action['action'] == action_type and current_time - action['time'] <= threshold['window']
        )
        
        if action_count >= threshold['count']:
            await self._handle_admin_abuse(guild, admin_id, action_type, action_count)
            return True
        
        return False
    
    async def _handle_admin_abuse(self, guild, admin_id, action_type, action_count):
        """Handle potential admin abuse"""
        # Add heat to the admin
        current_heat = self.heat_system.add_heat(admin_id, self.admin_abuse_penalty)
        
        # Log the admin abuse
        logger.warning(f"Potential admin abuse detected in {guild.name}: Admin {admin_id} performed {action_count} {action_type} actions")
        
        # Get the admin member object
        admin = guild.get_member(admin_id)
        if not admin:
            logger.error(f"Could not find admin with ID {admin_id}")
            return
        
        # Find a logging channel
        log_channel = discord.utils.get(guild.text_channels, name='mod-logs')
        if not log_channel:
            log_channel = discord.utils.get(guild.text_channels, name='logs')
        if not log_channel:
            log_channel = discord.utils.get(guild.text_channels, name='general')
        
        # Create an embed for the admin abuse alert
        embed = discord.Embed(
            title="⚠️ Admin Action Alert",
            description=f"Unusual administrative activity detected.",
            color=discord.Color.gold()
        )
        embed.add_field(
            name="Details", 
            value=f"Admin {admin.mention} performed {action_count} {action_type} actions in a short time.", 
            inline=False
        )
        
        # Take action based on heat level
        action_taken = "Admin is being monitored."
        
        # If heat is high enough, remove admin roles
        if self.heat_system.should_kick(admin_id) or self.heat_system.should_ban(admin_id):
            # Find admin roles
            admin_roles = [role for role in admin.roles if role.permissions.administrator or role.permissions.manage_guild or role.permissions.kick_members or role.permissions.ban_members]
            
            if admin_roles:
                try:
                    # Remove admin roles
                    for role in admin_roles:
                        await admin.remove_roles(role, reason="Admin abuse protection triggered")
                    
                    action_taken = f"Removed administrative roles: {', '.join([role.name for role in admin_roles])}"
                    logger.info(f"Removed admin roles from user {admin.name} ({admin_id}) due to admin abuse")
                except Exception as e:
                    logger.error(f"Failed to remove admin roles: {e}")
                    action_taken = "Failed to remove administrative roles due to an error."
        
        embed.add_field(name="Action Taken", value=action_taken, inline=False)
        
        # Send the alert
        if log_channel:
            try:
                await log_channel.send(embed=embed)
            except Exception as e:
                logger.error(f"Failed to send admin abuse alert: {e}")
                
        # If this is a severe case, also send a DM to the server owner
        if self.heat_system.should_ban(admin_id):
            try:
                owner = guild.owner
                if owner and owner.id != admin_id:  # Don't DM the owner if they're the one being flagged
                    await owner.send(f"⚠️ **Admin Abuse Alert** ⚠️\n\nPotential admin abuse detected in {guild.name}.\nAdmin {admin.name} ({admin_id}) performed {action_count} {action_type} actions in a short time.\nPlease review their actions and permissions.")
            except Exception as e:
                logger.error(f"Failed to send DM to server owner: {e}")