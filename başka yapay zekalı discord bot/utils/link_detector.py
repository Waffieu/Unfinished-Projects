import re
import logging
import discord

logger = logging.getLogger('discord_bot')

class LinkDetector:
    def __init__(self, heat_system, gemini_ai):
        self.heat_system = heat_system
        self.gemini_ai = gemini_ai
        
        # URL regex pattern
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        
        # Heat penalty for posting unsafe links
        self.unsafe_link_penalty = 40
        
        logger.info("Link detector initialized")
    
    async def check_message(self, message):
        """Check if a message contains potentially malicious links"""
        # Extract URLs from the message
        urls = self.url_pattern.findall(message.content)
        
        if not urls:
            return False  # No URLs found
        
        # Common safe domains that we'll skip checking
        safe_domains = [
            "youtube.com", "youtu.be", "twitter.com", "x.com",
            "facebook.com", "instagram.com", "discord.com", "discord.gg",
            "github.com", "gitlab.com", "reddit.com", "imgur.com",
            "twitch.tv", "spotify.com", "netflix.com", "amazon.com",
            "google.com", "wikipedia.org", "linkedin.com", "tiktok.com"
        ]
        
        # Check each URL
        for url in urls:
            # Skip checking for common safe domains
            if any(safe_domain in url.lower() for safe_domain in safe_domains):
                continue
                
            # Get message context (a few words around the link)
            context = self._extract_context(message.content, url)
            
            # Check if the link is safe using Gemini AI
            is_unsafe, explanation = await self.gemini_ai.check_link_safety(url, context)
            
            if is_unsafe:
                await self._handle_unsafe_link(message, url, explanation)
                return True
        
        return False
    
    def _extract_context(self, content, url):
        """Extract text context around a URL"""
        # Remove the URL from the content
        context = content.replace(url, "")
        # Remove extra whitespace
        context = re.sub(r'\s+', ' ', context).strip()
        # Limit context length
        if len(context) > 200:
            context = context[:200] + "..."
        return context
    
    async def _handle_unsafe_link(self, message, url, explanation):
        """Handle a message with an unsafe link"""
        user_id = message.author.id
        
        # Add heat to the user
        current_heat = self.heat_system.add_heat(user_id, self.unsafe_link_penalty)
        
        # Delete the message
        try:
            await message.delete()
            logger.info(f"Deleted message with unsafe link from {message.author.name} ({user_id}): {url}")
        except Exception as e:
            logger.error(f"Failed to delete message with unsafe link: {e}")
        
        # Create a warning embed
        embed = discord.Embed(
            title="⚠️ Potentially Unsafe Link Detected",
            description=f"A potentially unsafe link from {message.author.mention} was detected and removed.",
            color=discord.Color.red()
        )
        embed.add_field(name="Reason", value=explanation[:1024], inline=False)
        
        # Add heat information
        embed.add_field(
            name="Current Heat Level", 
            value=f"{current_heat:.1f} (+{self.unsafe_link_penalty})",
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
                await message.author.timeout(timeout_until, reason="Posting unsafe links")
                
                # Update embed with timeout info
                embed.add_field(
                    name="Action Taken", 
                    value=f"{message.author.mention} has been timed out for {timeout_duration//60} minutes.",
                    inline=False
                )
                
                logger.info(f"Timed out user {message.author.name} ({user_id}) for {timeout_duration//60} minutes")
            except Exception as e:
                logger.error(f"Failed to timeout user: {e}")
        else:
            # Just add a warning to the embed
            embed.add_field(
                name="Warning", 
                value=f"{message.author.mention} please be careful with the links you share. Continuing to share unsafe links may result in a timeout.",
                inline=False
            )
        
        # Send the warning message and set it to auto-delete after 1 minute
        try:
            warning_msg = await message.channel.send(embed=embed)
            # Schedule message to be deleted after 1 minute (60 seconds)
            await warning_msg.delete(delay=60)
        except Exception as e:
            logger.error(f"Failed to send warning message: {e}")