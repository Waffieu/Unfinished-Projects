import logging
import discord
from collections import defaultdict, deque

logger = logging.getLogger('discord_bot')

class AnomalyDetector:
    def __init__(self, heat_system, gemini_ai):
        self.heat_system = heat_system
        self.gemini_ai = gemini_ai
        
        # Track recent messages per user for context
        self.user_message_history = defaultdict(lambda: deque(maxlen=10))
        
        # Heat penalty for anomalous behavior
        self.anomaly_penalty = 35
        
        # Threshold for severe anomalies
        self.severe_anomaly_penalty = 70
        
        logger.info("Anomaly detector initialized")
    
    async def check_message(self, message):
        """Check if a message contains anomalous content"""
        user_id = message.author.id
        content = message.content
        
        # Skip very short messages
        if len(content) < 5:
            return False
        
        # Add message to user's history
        self.user_message_history[user_id].append(content)
        
        # Get user's recent message history
        user_history = list(self.user_message_history[user_id])[:-1]  # Exclude current message
        
        # Check for anomalies using Gemini AI
        is_anomaly, explanation = await self.gemini_ai.detect_anomaly(content, user_history)
        
        if is_anomaly:
            await self._handle_anomaly(message, explanation)
            return True
        
        return False
    
    async def _handle_anomaly(self, message, explanation):
        """Handle a message with anomalous content"""
        user_id = message.author.id
        
        # Determine severity based on explanation
        is_severe = self._is_severe_anomaly(explanation)
        heat_penalty = self.severe_anomaly_penalty if is_severe else self.anomaly_penalty
        
        # Add heat to the user
        current_heat = self.heat_system.add_heat(user_id, heat_penalty)
        
        # Delete the message regardless of heat level
        message_deleted = False
        try:
            await message.delete()
            message_deleted = True
            logger.info(f"Deleted anomalous message from {message.author.name} ({user_id})")
        except Exception as e:
            logger.error(f"Failed to delete anomalous message: {e}")
        
        # Create a warning embed
        embed = discord.Embed(
            title="⚠️ Suspicious Content Detected",
            description=f"A message with potentially problematic content from {message.author.mention} was detected and removed.",
            color=discord.Color.orange()
        )
        
        # Add a sanitized explanation
        safe_explanation = self._sanitize_explanation(explanation)
        embed.add_field(name="Analysis", value=safe_explanation[:1024], inline=False)
        
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
                await message.author.timeout(timeout_until, reason="Posting suspicious content")
                
                # Update embed with timeout info
                embed.add_field(
                    name="Action Taken", 
                    value=f"{message.author.mention} has been timed out for {timeout_duration//60} minutes for posting suspicious content.",
                    inline=False
                )
                
                logger.info(f"Timed out user {message.author.name} ({user_id}) for {timeout_duration//60} minutes due to anomalous content")
            except Exception as e:
                logger.error(f"Failed to timeout user: {e}")
        else:
            # Just add a warning to the embed
            action_text = "Message was deleted. " if message_deleted else ""
            embed.add_field(
                name="Warning", 
                value=f"{action_text}{message.author.mention} please be mindful of the content you post. Continuing to post suspicious content may result in a timeout.",
                inline=False
            )
        
        # Send the warning message and set it to auto-delete after 1 minute
        try:
            warning_msg = await message.channel.send(embed=embed)
            # Schedule message to be deleted after 1 minute (60 seconds)
            await warning_msg.delete(delay=60)
        except Exception as e:
            logger.error(f"Failed to send warning message: {e}")
    
    def _is_severe_anomaly(self, explanation):
        """Determine if an anomaly is severe based on the explanation"""
        severe_keywords = [
            "threat", "violence", "harassment", "doxx", "personal information",
            "attack", "raid", "illegal", "dangerous", "exploit", "malware",
            "phishing", "scam", "hate speech", "extremist"
        ]
        
        explanation_lower = explanation.lower()
        return any(keyword in explanation_lower for keyword in severe_keywords)
    
    def _sanitize_explanation(self, explanation):
        """Sanitize the explanation to remove any potentially harmful content"""
        # Remove any Discord formatting that could be abused
        sanitized = explanation.replace('@everyone', '`@everyone`')
        sanitized = sanitized.replace('@here', '`@here`')
        
        # Limit length
        if len(sanitized) > 1500:
            sanitized = sanitized[:1500] + "..."
            
        return sanitized