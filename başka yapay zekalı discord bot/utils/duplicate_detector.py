import time
import logging
import discord
import difflib
import numpy as np
from collections import defaultdict, deque
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger('discord_bot')

class DuplicateDetector:
    def __init__(self, heat_system):
        self.heat_system = heat_system
        
        # Track recent messages per user
        self.user_messages = defaultdict(lambda: deque(maxlen=15))  # Increased history size
        self.last_reset_time = defaultdict(float)  # Track when we last reset each user's history
        
        # Duplicate thresholds
        self.duplicate_threshold = 3  # Number of identical messages
        self.time_window = 120       # Extended time window in seconds
        self.reset_interval = 120    # Reset duplicate detection after 2 minutes
        self.similarity_threshold = 0.80  # Slightly lower similarity threshold for better detection
        
        # Heat penalty for duplicate messages
        self.duplicate_penalty = 25
        self.duplicate_penalty_decay = 0.8  # Penalty decay factor for repeat offenses
        
        # Initialize TF-IDF vectorizer for better text similarity
        self.vectorizer = TfidfVectorizer(min_df=1, analyzer='char_wb', ngram_range=(2, 4))
        self.vectorizer_fitted = False
        self.message_vectors = {}  # Cache for message vectors
        
        # Track user's duplicate history for adaptive thresholds
        self.user_duplicate_history = defaultdict(lambda: deque(maxlen=5))
        
        logger.info("Enhanced duplicate message detector initialized")
    
    async def check_message(self, message):
        """Check if a message is a duplicate and take appropriate action"""
        user_id = message.author.id
        content = message.content.strip().lower()  # Normalize content
        current_time = time.time()
        
        # Remove the length restriction - all messages should be checked for duplicates
        # regardless of their length
        
        # Get current heat level to adjust thresholds
        current_heat = self.heat_system.get_heat(user_id)
        
        # Set threshold to 3 to only apply heat after 3+ duplicate messages
        adjusted_threshold = 3
        
        # Check if we should reset this user's duplicate history (after 2 minutes)
        if current_time - self.last_reset_time.get(user_id, 0) >= self.reset_interval:
            self.user_messages[user_id].clear()
            self.last_reset_time[user_id] = current_time
            logger.info(f"Reset duplicate detection for user {user_id} after {self.reset_interval} seconds")
        
        # Add message to user's history with timestamp
        self.user_messages[user_id].append({
            'content': content,
            'time': current_time,
            'vector': None  # Will be computed if needed
        })
        
        # Count identical or similar messages within time window (excluding the current message)
        duplicate_count = 0
        similar_messages = []
        
        # Get message vector using TF-IDF
        msg_vector = self._get_message_vector(content)
        
        for msg in list(self.user_messages[user_id])[:-1]:  # Exclude the most recent message (current one)
            if current_time - msg['time'] <= self.time_window:
                # Check for exact match
                if msg['content'] == content:
                    duplicate_count += 1
                    similar_messages.append((msg['content'], 1.0))
                # Check for similar messages using multiple methods
                # Remove length restriction for similarity check
                else:
                    # Use both difflib and TF-IDF for better similarity detection
                    seq_similarity = difflib.SequenceMatcher(None, msg['content'], content).ratio()
                    
                    # Get vector similarity if possible
                    vector_similarity = 0
                    if msg_vector is not None:
                        msg_vector_old = self._get_message_vector(msg['content'])
                        if msg_vector_old is not None:
                            vector_similarity = self._cosine_similarity(msg_vector, msg_vector_old)
                    
                    # Use the higher of the two similarity scores
                    similarity = max(seq_similarity, vector_similarity)
                    
                    if similarity >= self.similarity_threshold:
                        duplicate_count += 1
                        similar_messages.append((msg['content'], similarity))
                        logger.info(f"Similar message detected with ratio {similarity:.2f}: '{msg['content']}' vs '{content}'")
        
        # Update the user's duplicate history
        if duplicate_count > 0:
            self.user_duplicate_history[user_id].append({
                'time': current_time,
                'count': duplicate_count
            })
        
        # Check if duplicate threshold is reached - now we'll only delete messages after 3+ duplicates
        if duplicate_count >= 3:
            await self._handle_duplicate(message, duplicate_count, similar_messages)
            return True
        
        return False
        
    def _get_message_vector(self, content):
        """Get TF-IDF vector for a message"""
        try:
            # Check if we've already computed this vector
            if content in self.message_vectors:
                return self.message_vectors[content]
                
            # If vectorizer hasn't been fitted yet, fit it with this content
            if not self.vectorizer_fitted:
                self.vectorizer.fit([content])
                self.vectorizer_fitted = True
                
            # Transform the content to a vector
            vector = self.vectorizer.transform([content]).toarray()[0]
            
            # Cache the vector
            self.message_vectors[content] = vector
            
            # Limit cache size
            if len(self.message_vectors) > 1000:
                # Remove a random item to keep cache size manageable
                self.message_vectors.pop(next(iter(self.message_vectors)))
                
            return vector
        except Exception as e:
            logger.error(f"Error computing message vector: {e}")
            return None
            
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0
                
            return dot_product / (norm1 * norm2)
        except Exception as e:
            logger.error(f"Error computing cosine similarity: {e}")
            return 0
    
    async def _handle_duplicate(self, message, count, similar_messages=None):
        """Handle a duplicate message"""
        user_id = message.author.id
        
        # Only apply heat penalty if duplicate count is more than 3
        if count >= 3:
            # Calculate adaptive penalty based on user's duplicate history
            penalty = self._calculate_adaptive_penalty(user_id)
            
            # Add heat to the user
            current_heat = self.heat_system.add_heat(user_id, penalty)
        else:
            # Just get current heat without adding penalty
            current_heat = self.heat_system.get_heat(user_id)
            penalty = 0
        
        # Delete the duplicate message
        try:
            await message.delete()
            logger.info(f"Deleted duplicate message from {message.author.name} ({user_id})")
        except Exception as e:
            logger.error(f"Failed to delete duplicate message: {e}")
        
        # Create a more informative embed for the warning
        embed = discord.Embed(
            title="Duplicate Message Detection",
            description=f"Similar messages detected from {message.author.mention}",
            color=discord.Color.orange()
        )
        
        # Add information about similar messages if available
        if similar_messages and len(similar_messages) > 0:
            similar_examples = "\n".join([f"• {content[:50]}... ({similarity:.2f})" 
                                     for content, similarity in similar_messages[:3]])
            embed.add_field(
                name="Similar Messages Detected", 
                value=similar_examples if similar_examples else "Multiple similar messages",
                inline=False
            )
        
        # Add heat information
        embed.add_field(
            name="Current Heat Level", 
            value=f"{current_heat:.1f} {'+' + str(penalty) if penalty > 0 else ''}",
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
                await message.author.timeout(timeout_until, reason="Sending duplicate messages")
                
                # Update embed with timeout information
                embed.add_field(
                    name="Action Taken", 
                    value=f"{message.author.mention} has been timed out for {timeout_duration//60} minutes",
                    inline=True
                )
                
                logger.info(f"Timed out user {message.author.name} ({user_id}) for {timeout_duration//60} minutes")
            except Exception as e:
                logger.error(f"Failed to timeout user: {e}")
        else:
            # Just warn the user
            embed.add_field(
                name="Warning", 
                value=f"{message.author.mention} please avoid sending similar messages repeatedly",
                inline=True
            )
        
        # Send the warning message with the embed and set it to auto-delete after 1 minute
        try:
            warning_msg = await message.channel.send(embed=embed)
            # Schedule message to be deleted after 1 minute (60 seconds)
            await warning_msg.delete(delay=60)
        except Exception as e:
            logger.error(f"Failed to send warning message: {e}")
    
    def _calculate_adaptive_penalty(self, user_id):
        """Calculate an adaptive penalty based on user's duplicate history"""
        current_time = time.time()
        recent_duplicates = 0
        
        # Count recent duplicate incidents
        for incident in self.user_duplicate_history[user_id]:
            if current_time - incident['time'] <= 300:  # Last 5 minutes
                recent_duplicates += 1
        
        # Apply exponential penalty for repeat offenders, but with decay over time
        if recent_duplicates > 0:
            # Get time since last duplicate incident
            last_incident_time = self.user_duplicate_history[user_id][-1]['time'] if self.user_duplicate_history[user_id] else current_time
            time_since_last = current_time - last_incident_time
            
            # Calculate decay factor based on time since last incident
            decay_factor = max(0.5, min(1.0, self.duplicate_penalty_decay ** (time_since_last / 60)))
            
            # Calculate penalty with increasing severity for repeat offenses
            penalty = self.duplicate_penalty * (1 + (recent_duplicates * 0.2)) * decay_factor
        else:
            penalty = self.duplicate_penalty
        
        return penalty
        
        # Remove similar messages from history instead of clearing all history
        # This allows detection of other duplicate messages in the future
        content_to_check = message.content.strip().lower()
        filtered_messages = []
        
        for msg in self.user_messages[user_id]:
            # Keep messages that are not identical
            if msg['content'] != content_to_check:
                # Also filter out highly similar messages
                if len(content_to_check) > 10 and len(msg['content']) > 10:
                    similarity = difflib.SequenceMatcher(None, msg['content'], content_to_check).ratio()
                    if similarity < self.similarity_threshold:  # Keep if not too similar
                        filtered_messages.append(msg)
                else:
                    filtered_messages.append(msg)
        
        self.user_messages[user_id] = deque(filtered_messages, maxlen=15)