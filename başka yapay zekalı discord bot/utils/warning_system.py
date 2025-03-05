import logging
import discord
import datetime
from collections import defaultdict
from discord.utils import utcnow
from apscheduler.schedulers.asyncio import AsyncIOScheduler

logger = logging.getLogger('discord_bot')

class WarningSystem:
    def __init__(self, heat_system):
        self.heat_system = heat_system
        
        # Track warning counts per user
        self.warning_counts = defaultdict(int)
        
        # Warning threshold before punishment
        self.warning_threshold = 3
        
        # Weekly reset task
        self.scheduler = AsyncIOScheduler()
        self.scheduler.add_job(self.reset_all_warnings, 'interval', weeks=1)
        self.scheduler.start()
        
        # Heat penalty for each warning
        self.warning_penalty = 15
        
        # Default timeout duration for reaching warning threshold (in seconds)
        self.default_timeout_duration = 60 * 30  # 30 minutes
        
        # Timeout durations based on number of times user has reached threshold
        self.timeout_durations = {
            1: 60 * 30,      # 30 minutes for first punishment
            2: 60 * 60,      # 1 hour for second punishment
            3: 60 * 60 * 3,  # 3 hours for third punishment
            4: 60 * 60 * 24  # 24 hours for fourth+ punishment
        }
        
        # Track how many times a user has been punished
        self.punishment_counts = defaultdict(int)
        
        logger.info("Warning system initialized")
    
    async def add_warning(self, message, reason):
        """Add a warning to a user and take appropriate action"""
        user_id = message.author.id
        
        # Increment the warning count for this user
        self.warning_counts[user_id] += 1
        current_count = self.warning_counts[user_id]
        
        # Add heat to the user
        current_heat = self.heat_system.add_heat(user_id, self.warning_penalty)
        
        # Delete the message
        message_deleted = False
        try:
            await message.delete()
            message_deleted = True
            logger.info(f"Deleted message from {message.author.name} ({user_id}) due to warning")
        except Exception as e:
            logger.error(f"Failed to delete message: {e}")
        
        # Create a warning embed
        embed = discord.Embed(
            title="⚠️ Warning",
            description=f"{message.author.mention} has received a warning.",
            color=discord.Color.orange()
        )
        
        # Add reason information
        embed.add_field(
            name="Reason", 
            value=reason,
            inline=False
        )
        
        # Add warning count information
        embed.add_field(
            name="Warning Count", 
            value=f"{current_count}/{self.warning_threshold} (Timeout at {self.warning_threshold})",
            inline=True
        )
        
        # Add heat information
        embed.add_field(
            name="Current Heat Level", 
            value=f"{current_heat:.1f} (+{self.warning_penalty})",
            inline=True
        )
        
        # Check if user has reached the warning threshold
        should_timeout_for_warnings = current_count >= self.warning_threshold
        
        # Check if user should be timed out (either due to heat or warning count)
        if should_timeout_for_warnings or self.heat_system.should_timeout(user_id):
            # If warning threshold reached, reset warnings and increment punishment count
            if should_timeout_for_warnings:
                self.warning_counts[user_id] = 0  # Reset warnings after punishment
                self.punishment_counts[user_id] += 1  # Increment punishment count
                punishment_count = self.punishment_counts[user_id]
                
                # Determine timeout duration based on punishment count
                if punishment_count in self.timeout_durations:
                    timeout_duration = self.timeout_durations[punishment_count]
                else:
                    timeout_duration = self.timeout_durations[max(self.timeout_durations.keys())]
                    
                timeout_reason = f"Received {self.warning_threshold} warnings (Punishment #{punishment_count})"
            else:
                timeout_duration = self.heat_system.get_timeout_duration(current_heat)
                timeout_reason = "Excessive heat from warnings"
            
            try:
                # Convert seconds to datetime.timedelta for timeout
                timeout_until = utcnow() + datetime.timedelta(seconds=timeout_duration)
                
                # Apply timeout
                await message.author.timeout(timeout_until, reason=timeout_reason)
                
                # Update embed with timeout info
                embed.add_field(
                    name="Action Taken", 
                    value=f"{message.author.mention} has been timed out for {timeout_duration//60} minutes.",
                    inline=False
                )
                
                logger.info(f"Timed out user {message.author.name} ({user_id}) for {timeout_duration//60} minutes due to {timeout_reason}")
            except Exception as e:
                logger.error(f"Failed to timeout user: {e}")
        else:
            # Add timeout threshold information if not being timed out yet
            timeout_threshold = self.heat_system.thresholds['timeout']
            heat_remaining = max(0, timeout_threshold - current_heat)
            embed.add_field(
                name="Heat Warning", 
                value=f"Timeout at: {timeout_threshold} heat\nRemaining: {heat_remaining:.1f} heat",
                inline=False
            )
            
            # Just add a warning to the embed
            action_text = "Message was deleted. " if message_deleted else ""
            embed.add_field(
                name="Warning", 
                value=f"{action_text}You will be automatically timed out after {self.warning_threshold} warnings.",
                inline=False
            )
        
        # Send the warning message and set it to auto-delete after 1 minute
        try:
            warning_msg = await message.channel.send(embed=embed)
            # Schedule message to be deleted after 1 minute (60 seconds)
            await warning_msg.delete(delay=60)
        except Exception as e:
            logger.error(f"Failed to send warning message: {e}")
            
        return current_count
    
    def get_warning_count(self, user_id):
        """Get the current warning count for a user"""
        return self.warning_counts.get(user_id, 0)
    
    def reset_warnings(self, user_id):
        """Reset a user's warning count to 0"""
        if user_id in self.warning_counts:
            self.warning_counts[user_id] = 0
            logger.info(f"Reset warnings for user {user_id}")
    
    def get_punishment_count(self, user_id):
        """Get the number of times a user has been punished"""
        return self.punishment_counts.get(user_id, 0)
    
    def reset_all_warnings(self):
        """Reset all warning and punishment counts weekly"""
        self.warning_counts.clear()
        self.punishment_counts.clear()
        logger.info("Reset all warning counts weekly schedule executed")