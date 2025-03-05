import time
import logging
import discord
from collections import deque, defaultdict

logger = logging.getLogger('discord_bot')

class RaidProtection:
    def __init__(self, heat_system):
        self.heat_system = heat_system
        
        # Track recent joins
        self.recent_joins = deque(maxlen=100)  # Store recent join timestamps
        self.join_thresholds = {
            'warning': 5,   # 5 joins in 10 seconds
            'mild': 10,     # 10 joins in 10 seconds
            'severe': 20    # 20 joins in 10 seconds
        }
        self.time_window = 10  # Time window in seconds
        
        # Track channels with slowmode
        self.slowmode_channels = {}
        self.slowmode_duration = 60 * 30  # 30 minutes
        
        # Track users who joined during raid
        self.raid_joiners = set()
        
        logger.info("Raid protection initialized")
    
    async def check_join(self, member):
        """Check if a new member join is part of a raid"""
        current_time = time.time()
        guild = member.guild
        
        # Add join to recent joins
        self.recent_joins.append((current_time, member.id))
        
        # Count joins in the time window
        recent_count = sum(1 for t, _ in self.recent_joins if current_time - t <= self.time_window)
        
        # Check if we're experiencing a raid
        raid_level = self._get_raid_level(recent_count)
        
        if raid_level:
            # We're experiencing a raid
            await self._handle_raid(guild, member, raid_level, recent_count)
            return True
        
        return False
    
    def _get_raid_level(self, join_count):
        """Determine raid level based on join count"""
        if join_count >= self.join_thresholds['severe']:
            return 'severe'
        elif join_count >= self.join_thresholds['mild']:
            return 'mild'
        elif join_count >= self.join_thresholds['warning']:
            return 'warning'
        return None
    
    async def _handle_raid(self, guild, member, raid_level, join_count):
        """Handle a potential raid"""
        # Add user to raid joiners
        self.raid_joiners.add(member.id)
        
        # Log the raid detection
        logger.warning(f"Potential raid detected in {guild.name}: {raid_level} level ({join_count} joins in {self.time_window}s)")
        
        # Find a logging channel
        log_channel = discord.utils.get(guild.text_channels, name='mod-logs')
        if not log_channel:
            log_channel = discord.utils.get(guild.text_channels, name='logs')
        if not log_channel:
            log_channel = discord.utils.get(guild.text_channels, name='general')
        
        # Create an embed for the raid alert
        embed = discord.Embed(
            title=f"⚠️ Raid Alert: {raid_level.upper()}",
            description=f"Unusual join rate detected from {member.mention}: {join_count} users in {self.time_window} seconds.",
            color=discord.Color.red()
        )
        
        # Take action based on raid level
        if raid_level == 'severe':
            # Enable slowmode in all text channels
            await self._enable_server_slowmode(guild, 15)  # 15 seconds slowmode
            embed.add_field(
                name="Action Taken", 
                value="Enabled slowmode in all channels. New members are being monitored.", 
                inline=False
            )
            
            # Schedule slowmode removal
            self._schedule_slowmode_removal(guild)
            
        elif raid_level == 'mild':
            # Enable slowmode in general channels
            general_channels = [ch for ch in guild.text_channels if 'general' in ch.name.lower()]
            for channel in general_channels:
                await self._set_channel_slowmode(channel, 10)  # 10 seconds slowmode
            
            embed.add_field(
                name="Action Taken", 
                value="Enabled slowmode in general channels. New members are being monitored.", 
                inline=False
            )
            
        # Send the alert
        if log_channel:
            try:
                await log_channel.send(embed=embed)
            except Exception as e:
                logger.error(f"Failed to send raid alert: {e}")
    
    async def _enable_server_slowmode(self, guild, seconds):
        """Enable slowmode in all text channels"""
        for channel in guild.text_channels:
            # Skip channels that already have higher slowmode
            if channel.slowmode_delay >= seconds:
                continue
                
            await self._set_channel_slowmode(channel, seconds)
    
    async def _set_channel_slowmode(self, channel, seconds):
        """Set slowmode for a channel and track it"""
        try:
            # Store original slowmode to restore later
            self.slowmode_channels[channel.id] = {
                'original_slowmode': channel.slowmode_delay,
                'expiry': time.time() + self.slowmode_duration
            }
            
            # Set slowmode
            await channel.edit(slowmode_delay=seconds)
            logger.info(f"Enabled slowmode ({seconds}s) in #{channel.name}")
        except Exception as e:
            logger.error(f"Failed to set slowmode in #{channel.name}: {e}")
    
    def _schedule_slowmode_removal(self, guild):
        """Schedule the removal of slowmode after the duration"""
        import asyncio
        
        async def remove_slowmode():
            await asyncio.sleep(self.slowmode_duration)
            await self._remove_server_slowmode(guild)
        
        # Create task to remove slowmode later
        asyncio.create_task(remove_slowmode())
    
    async def _remove_server_slowmode(self, guild):
        """Remove slowmode from all channels where we set it"""
        current_time = time.time()
        channels_to_restore = []
        
        # Find channels with expired slowmode
        for channel_id, data in list(self.slowmode_channels.items()):
            if data['expiry'] <= current_time:
                channel = guild.get_channel(channel_id)
                if channel:
                    channels_to_restore.append((channel, data['original_slowmode']))
                del self.slowmode_channels[channel_id]
        
        # Restore original slowmode
        for channel, original_slowmode in channels_to_restore:
            try:
                await channel.edit(slowmode_delay=original_slowmode)
                logger.info(f"Removed slowmode from #{channel.name}")
            except Exception as e:
                logger.error(f"Failed to remove slowmode from #{channel.name}: {e}")
        
        # Clear raid joiners after slowmode is removed
        self.raid_joiners.clear()
        
        # Log the end of raid mode
        log_channel = discord.utils.get(guild.text_channels, name='mod-logs')
        if log_channel:
            try:
                await log_channel.send("Raid protection measures have been lifted. Server has returned to normal operation.")
            except Exception as e:
                logger.error(f"Failed to send raid end notification: {e}")