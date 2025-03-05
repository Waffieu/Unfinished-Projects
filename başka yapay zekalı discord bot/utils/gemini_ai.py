import logging
import google.generativeai as genai

logger = logging.getLogger('discord_bot')

class GeminiAI:
    def __init__(self, api_key, model="gemini-2.0-flash-lite"):
        self.api_key = api_key
        self.model_name = model
        
        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        
        try:
            # Initialize the model
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Gemini AI initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini AI: {e}")
            self.model = None
    
    async def analyze_text(self, text, prompt_context=""):
        """Analyze text content using Gemini AI"""
        if not self.model:
            logger.error("Gemini AI model not initialized")
            return None
        
        try:
            # Create a prompt that asks Gemini to analyze the text
            full_prompt = f"{prompt_context}\n\nAnalyze the following text: {text}"
            
            # Generate a response
            response = self.model.generate_content(full_prompt)
            
            return response.text
        except Exception as e:
            logger.error(f"Error analyzing text with Gemini AI: {e}")
            return None
    
    async def check_link_safety(self, link, context=""):
        """Check if a link is potentially malicious or a scam"""
        if not self.model:
            logger.error("Gemini AI model not initialized")
            return False, "AI model not available"
        
        try:
            # Whitelist common domains that should always be considered safe
            safe_domains = [
                "youtube.com", "youtu.be", "twitter.com", "x.com",
                "facebook.com", "instagram.com", "discord.com", "discord.gg",
                "github.com", "gitlab.com", "reddit.com", "imgur.com",
                "twitch.tv", "spotify.com", "netflix.com", "amazon.com",
                "google.com", "wikipedia.org", "linkedin.com", "tiktok.com"
            ]
            
            # Check if the link contains any of the safe domains
            if any(safe_domain in link.lower() for safe_domain in safe_domains):
                return False, "Link appears to be from a trusted domain"
            
            prompt = f"""Analyze this link and determine if it's potentially malicious, a scam, or fake:
            Link: {link}
            Context: {context}
            
            IMPORTANT: Only respond with 'UNSAFE' if the link is CLEARLY malicious, a scam, phishing attempt, or fake (like free Discord Nitro, Steam wallet codes, etc.).
            Regular websites, even if unfamiliar, should be considered 'SAFE' unless there are clear red flags.
            Social media links, video sharing sites, news sites, and other common websites should be considered 'SAFE'.
            
            Respond with 'UNSAFE' ONLY if you are highly confident the link is dangerous.
            Respond with 'SAFE' for all other links.
            Provide a brief explanation of your reasoning."""
            
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Check if the response indicates the link is unsafe
            is_unsafe = "UNSAFE" in response_text.upper()
            explanation = response_text.replace("UNSAFE", "").replace("SAFE", "").strip()
            
            return is_unsafe, explanation
        except Exception as e:
            logger.error(f"Error checking link safety with Gemini AI: {e}")
            # Changed to return False instead of True on error
            return False, "Error analyzing link, treating as safe by default"
    
    async def detect_anomaly(self, message_content, user_history=None):
        """Detect if a message contains anomalous or suspicious content"""
        if not self.model:
            logger.error("Gemini AI model not initialized")
            return False, "AI model not available"
        
        try:
            # Create context from user history if available
            history_context = ""
            if user_history and len(user_history) > 0:
                history_context = "User's recent messages:\n" + "\n".join([f"- {msg}" for msg in user_history[-5:]])
            
            prompt = f"""Analyze this Discord message and determine if it contains anomalous or suspicious content:
            Message: {message_content}
            {history_context}
            
            Respond with 'ANOMALY' if the message appears suspicious, contains threats, harassment, inappropriate content, or seems to be part of a coordinated attack.
            Respond with 'NORMAL' if the message appears to be regular conversation.
            Provide a brief explanation of your reasoning."""
            
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Check if the response indicates an anomaly
            is_anomaly = "ANOMALY" in response_text.upper()
            explanation = response_text.replace("ANOMALY", "").replace("NORMAL", "").strip()
            
            return is_anomaly, explanation
        except Exception as e:
            logger.error(f"Error detecting anomaly with Gemini AI: {e}")
            return False, "Error analyzing message"