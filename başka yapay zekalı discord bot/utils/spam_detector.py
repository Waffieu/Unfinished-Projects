import re
import time
import logging
import discord
from collections import defaultdict

logger = logging.getLogger('discord_bot')

class SpamDetector:
    def __init__(self, heat_system):
        self.heat_system = heat_system
        
        # Track message counts per user
        self.message_counts = defaultdict(list)
        
        # Spam thresholds
        self.message_threshold = 5  # Number of messages
        self.time_threshold = 5     # Time window in seconds
        self.caps_threshold = 0.7   # Percentage of uppercase characters
        self.mention_threshold = 5  # Number of mentions in a single message
        
        # Heat penalties
        self.spam_heat_penalty = 20
        self.caps_heat_penalty = 15
        self.mention_spam_penalty = 25
        
        logger.info("Spam detector initialized")
    
    async def check_message(self, message):
        """Check if a message is spam and take appropriate action"""
        user_id = message.author.id
        current_time = time.time()
        
        # Add message to user's history
        self.message_counts[user_id].append(current_time)
        
        # Remove messages older than the time threshold
        self.message_counts[user_id] = [t for t in self.message_counts[user_id] 
                                       if current_time - t <= self.time_threshold]
        
        # Check for rapid message spam
        if len(self.message_counts[user_id]) >= self.message_threshold:
            await self._handle_spam(message, "rapid message spam")
            return True
        
        # Check for excessive caps
        if await self._check_caps_spam(message):
            await self._handle_spam(message, "excessive caps")
            return True
        
        # Check for mention spam
        if await self._check_mention_spam(message):
            await self._handle_spam(message, "mention spam")
            return True
        
        return False
    
    async def _check_caps_spam(self, message):
        """Check if a message has excessive uppercase characters"""
        content = message.content
        if len(content) < 10:  # Ignore short messages
            return False
        
        # Count uppercase letters
        uppercase_count = sum(1 for c in content if c.isupper())
        total_letters = sum(1 for c in content if c.isalpha())
        
        if total_letters > 0 and uppercase_count / total_letters >= self.caps_threshold:
            return True
        
        return False
    
    async def _check_mention_spam(self, message):
        """Check if a message has excessive mentions"""
        # Count user mentions
        mention_count = len(message.mentions)
        
        # Count role mentions
        mention_count += len(message.role_mentions)
        
        return mention_count >= self.mention_threshold
    
    async def _handle_spam(self, message, spam_type):
        """Handle a spam message"""
        user_id = message.author.id
        
        # Apply appropriate heat penalty based on spam type
        if spam_type == "rapid message spam":
            heat_penalty = self.spam_heat_penalty
        elif spam_type == "excessive caps":
            heat_penalty = self.caps_heat_penalty
        elif spam_type == "mention spam":
            heat_penalty = self.mention_spam_penalty
        else:
            heat_penalty = self.spam_heat_penalty
        
        # Add heat to the user
        current_heat = self.heat_system.add_heat(user_id, heat_penalty)
        
        # Delete the spam message
        try:
            await message.delete()
            logger.info(f"Deleted spam message from {message.author.name} ({user_id}): {spam_type}")
        except Exception as e:
            logger.error(f"Failed to delete spam message: {e}")
        
        # Create an embed for the warning
        embed = discord.Embed(
            title="⚠️ Spam Detection",
            description=f"Spam detected from {message.author.mention}: {spam_type}",
            color=discord.Color.orange()
        )
        
        # Add heat information
        embed.add_field(
            name="Current Heat Level", 
            value=f"{current_heat:.1f} (+{heat_penalty})",
            inline=True
        )
        
        # Add timeout threshold information
        timeout_threshold = self.heat_system.thresholds['timeout']
        heat_remaining = max(0, timeout_threshold - current_heat)
        embed.add_field(
            name="Timeout Warning", 
            value=f"Timeout at: {timeout_threshold} heat\nRemaining: {heat_remaining:.1f} heat",
            inline=True
        )
        
        # Check if user should be timed out
        if self.heat_system.should_timeout(user_id):
            timeout_duration = self.heat_system.get_timeout_duration(current_heat)
            
            try:
                # Convert seconds to datetime.timedelta for timeout
                import datetime
                from discord.utils import utcnow
                timeout_until = utcnow() + datetime.timedelta(seconds=timeout_duration)
                
                # Apply timeout
                await message.author.timeout(timeout_until, reason=f"Spam detection: {spam_type}")
                
                # Update embed with timeout information
                embed.add_field(
                    name="Action Taken", 
                    value=f"{message.author.mention} has been timed out for {timeout_duration//60} minutes",
                    inline=False
                )
                
                logger.info(f"Timed out user {message.author.name} ({user_id}) for {timeout_duration//60} minutes")
            except Exception as e:
                logger.error(f"Failed to timeout user: {e}")
        else:
            # Just warn the user
            embed.add_field(
                name="Warning", 
                value=f"{message.author.mention} please stop {spam_type}. Continuing may result in a timeout.",
                inline=False
            )
        
        # Send the warning message with the embed and set it to auto-delete after 1 minute
        try:
            warning_msg = await message.channel.send(embed=embed)
            # Schedule message to be deleted after 1 minute (60 seconds)
            await warning_msg.delete(delay=60)
        except Exception as e:
            logger.error(f"Failed to send warning message: {e}")