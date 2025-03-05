import logging
import discord
from collections import defaultdict, deque

logger = logging.getLogger('discord_bot')

class ProfanityDetector:
    def __init__(self, heat_system, gemini_ai):
        self.heat_system = heat_system
        self.gemini_ai = gemini_ai
        
        # Track recent messages per user for context
        self.user_message_history = defaultdict(lambda: deque(maxlen=5))
        
        # Heat penalty for profanity
        self.profanity_penalty = 30
        
        # Threshold for severe profanity
        self.severe_profanity_penalty = 60
        
        logger.info("Profanity detector initialized")
    
    async def check_message(self, message):
        """Check if a message contains profanity or insults"""
        user_id = message.author.id
        content = message.content
        
        # Skip very short messages
        if len(content) < 3:
            return False
        
        # Add message to user's history
        self.user_message_history[user_id].append(content)
        
        # Get user's recent message history
        user_history = list(self.user_message_history[user_id])[:-1]  # Exclude current message
        
        # Check for profanity using Gemini AI
        is_profane, explanation = await self._detect_profanity(content, user_history)
        
        if is_profane:
            await self._handle_profanity(message, explanation)
            return True
        
        return False
    
    async def _detect_profanity(self, content, user_history=None):
        """Detect if a message contains profanity or insults using Gemini AI"""
        try:
            # Create context from user history if available
            history_context = ""
            if user_history and len(user_history) > 0:
                history_context = "User's recent messages:\n" + "\n".join([f"- {msg}" for msg in user_history[-3:]])
            
            prompt = f"""Analyze this Discord message and determine if it contains profanity, insults, hate speech, or other inappropriate language:
            Message: {content}
            {history_context}
            
            Respond with 'PROFANE' if the message contains profanity, insults, hate speech, or other inappropriate language.
            Respond with 'CLEAN' if the message appears to be appropriate.
            Provide a brief explanation of your reasoning.
            If the message is in a language other than English, analyze it in that language."""
            
            response = await self.gemini_ai.analyze_text(prompt)
            
            if not response:
                return False, "Error analyzing message"
            
            # Check if the response indicates profanity
            is_profane = "PROFANE" in response.upper()
            explanation = response.replace("PROFANE", "").replace("CLEAN", "").strip()
            
            # Determine if this is severe profanity
            is_severe = self._is_severe_profanity(explanation)
            
            return is_profane, explanation
        except Exception as e:
            logger.error(f"Error detecting profanity with Gemini AI: {e}")
            return False, "Error analyzing message"
    
    def _is_severe_profanity(self, explanation):
        """Determine if profanity is severe based on the explanation"""
        severe_keywords = [
            "hate speech", "racial", "racist", "slur", "extremely offensive", 
            "targeted harassment", "threat", "violent", "sexual", "explicit",
            "discriminatory", "homophobic", "transphobic", "sexist"
        ]
        
        explanation_lower = explanation.lower()
        return any(keyword in explanation_lower for keyword in severe_keywords)
    
    async def _handle_profanity(self, message, explanation):
        """Handle a message with profanity"""
        user_id = message.author.id
        
        # Determine severity based on explanation
        is_severe = self._is_severe_profanity(explanation)
        heat_penalty = self.severe_profanity_penalty if is_severe else self.profanity_penalty
        
        # Add heat to the user
        current_heat = self.heat_system.add_heat(user_id, heat_penalty)
        
        # Delete the message
        message_deleted = False
        try:
            await message.delete()
            message_deleted = True
            logger.info(f"Deleted message with profanity from {message.author.name} ({user_id})")
        except Exception as e:
            logger.error(f"Failed to delete message with profanity: {e}")
        
        # Create a warning embed
        embed = discord.Embed(
            title="⚠️ Inappropriate Language Detected",
            description=f"A message with inappropriate content from {message.author.mention} was detected and removed.",
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
                await message.author.timeout(timeout_until, reason="Using inappropriate language")
                
                # Update embed with timeout info
                embed.add_field(
                    name="Action Taken", 
                    value=f"{message.author.mention} has been timed out for {timeout_duration//60} minutes for using inappropriate language.",
                    inline=False
                )
                
                logger.info(f"Timed out user {message.author.name} ({user_id}) for {timeout_duration//60} minutes due to profanity")
            except Exception as e:
                logger.error(f"Failed to timeout user: {e}")
        else:
            # Just add a warning to the embed
            action_text = "Message was deleted. " if message_deleted else ""
            embed.add_field(
                name="Warning", 
                value=f"{action_text}{message.author.mention} please be mindful of your language. Continuing to use inappropriate language may result in a timeout.",
                inline=False
            )
        
        # Send the warning message and set it to auto-delete after 1 minute
        try:
            warning_msg = await message.channel.send(embed=embed)
            # Schedule message to be deleted after 1 minute (60 seconds)
            await warning_msg.delete(delay=60)
        except Exception as e:
            logger.error(f"Failed to send warning message: {e}")
    
    def _sanitize_explanation(self, explanation):
        """Sanitize the explanation to remove any potentially harmful content"""
        # Remove any Discord formatting that could be abused
        sanitized = explanation.replace('@everyone', '`@everyone`')
        sanitized = sanitized.replace('@here', '`@here`')
        
        # Limit length
        if len(sanitized) > 1500:
            sanitized = sanitized[:1500] + "..."
            
        return sanitized