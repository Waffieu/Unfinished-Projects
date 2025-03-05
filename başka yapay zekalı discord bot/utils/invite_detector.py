import re
import logging
import discord
from collections import defaultdict
import datetime
from discord.utils import utcnow

logger = logging.getLogger('discord_bot')

class InviteDetector:
    def __init__(self, heat_system):
        self.heat_system = heat_system
        
        # Discord invite regex pattern
        self.invite_pattern = re.compile(r'discord(?:\.gg|\.com\/invite)\/[a-zA-Z0-9-]+', re.IGNORECASE)
        
        # Heat penalty for posting invite links
        self.invite_link_penalty = 30
        
        # Track invite link count per user
        self.invite_link_counts = defaultdict(int)
        
        # Number of invite links that trigger automatic timeout
        self.invite_link_threshold = 3
        
        # Default timeout duration for repeated invite links (in seconds)
        self.default_timeout_duration = 60 * 30  # 30 minutes
        
        logger.info("Invite link detector initialized")
    
    async def check_message(self, message):
        """Check if a message contains Discord invite links"""
        # Extract invite links from the message
        invite_links = self.invite_pattern.findall(message.content)
        
        if not invite_links:
            return False  # No invite links found
        
        # Handle the invite link
        await self._handle_invite_link(message, invite_links)
        return True
    
    async def _handle_invite_link(self, message, invite_links):
        """Handle a message with Discord invite links"""
        user_id = message.author.id
        
        # Increment the invite link count for this user
        self.invite_link_counts[user_id] += 1
        current_count = self.invite_link_counts[user_id]
        
        # Add heat to the user
        current_heat = self.heat_system.add_heat(user_id, self.invite_link_penalty)
        
        # Delete the message
        try:
            await message.delete()
            logger.info(f"Deleted message with invite link from {message.author.name} ({user_id})")
        except Exception as e:
            logger.error(f"Failed to delete message with invite link: {e}")
        
        # Create a warning embed
        embed = discord.Embed(
            title="⚠️ Discord Invite Link Detected",
            description=f"A Discord invite link from {message.author.mention} was detected and removed.",
            color=discord.Color.orange()
        )
        
        # Add invite count information
        embed.add_field(
            name="Invite Link Count", 
            value=f"{current_count}/{self.invite_link_threshold} (Timeout at {self.invite_link_threshold})",
            inline=True
        )
        
        # Add heat information
        embed.add_field(
            name="Current Heat Level", 
            value=f"{current_heat:.1f} (+{self.invite_link_penalty})",
            inline=True
        )
        
        # Check if user has reached the invite link threshold
        should_timeout_for_invites = current_count >= self.invite_link_threshold
        
        # Check if user should be timed out (either due to heat or invite count)
        if should_timeout_for_invites or self.heat_system.should_timeout(user_id):
            # Determine timeout duration
            if should_timeout_for_invites:
                timeout_duration = self.default_timeout_duration
                timeout_reason = f"Sending {current_count} Discord invite links"
            else:
                timeout_duration = self.heat_system.get_timeout_duration(current_heat)
                timeout_reason = "Excessive heat from posting Discord invite links"
            
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
            embed.add_field(
                name="Warning", 
                value=f"{message.author.mention} please do not share Discord invite links. You will be automatically timed out after {self.invite_link_threshold} violations.",
                inline=False
            )
        
        # Send the warning message and set it to auto-delete after 1 minute
        try:
            warning_msg = await message.channel.send(embed=embed)
            # Schedule message to be deleted after 1 minute (60 seconds)
            await warning_msg.delete(delay=60)
        except Exception as e:
            logger.error(f"Failed to send warning message: {e}")