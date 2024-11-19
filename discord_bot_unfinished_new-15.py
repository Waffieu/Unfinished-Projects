import os
import sys
import io
import time
import asyncio
import json
import re
import logging
import random
import aiohttp
import nltk
import numpy as np
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from fake_useragent import UserAgent
from aiosqlite import connect
from bs4 import BeautifulSoup
from backoff import on_exception, expo
from requests.exceptions import RequestException
from google.api_core.exceptions import GoogleAPIError
import pickle  # Added import
from duckduckgo_search import DDGS  # Added import
from google.generativeai import configure, GenerativeModel  # For Gemini AI
import PIL.Image
import faiss
from langdetect import detect
import os
import hashlib
from fake_useragent import UserAgent
import aiohttp
import asyncio
import random
import time
from bs4 import BeautifulSoup
import logging
from urllib.parse import urlparse
import json
import ssl
import certifi
from urllib.parse import urlparse
import json
import statistics
import discord
from discord.ext import commands
import logging
import coloredlogs
import networkx as nx
import logging
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from PIL import Image
import io
import aiohttp
from typing import Optional, Union
import google.generativeai as genai
from groq import Groq
from PIL import Image
import io
import base64

# Replace the existing logging configuration with this:
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log", encoding='utf-8'),  # Specify UTF-8 encoding
        logging.StreamHandler(sys.stdout)  # Use stdout for console output
    ]
)

# And update the file handler configuration:
file_handler = logging.FileHandler('puro_bot_detailed.log', encoding='utf-8')


class LanguageDetector:
    def __init__(self):
        self.fallback_lang = 'en'
        
    def detect(self, text: str) -> str:
        try:
            return detect(text)
        except:
            return self.fallback_lang

class TopicAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.lda = LatentDirichletAllocation(n_components=10)
        
    def extract_topics(self, text: str) -> Dict[str, float]:
        try:
            # Transform text to TF-IDF features
            tfidf = self.vectorizer.fit_transform([text])
            
            # Extract topic distributions
            topic_dist = self.lda.fit_transform(tfidf)[0]
            
            # Create topic dictionary
            topics = {f'topic_{i}': float(score) 
                     for i, score in enumerate(topic_dist)}
            
            return topics
        except:
            return defaultdict(float)

# Create custom logger
logger = logging.getLogger('PuroBot')
coloredlogs.install(level='DEBUG', logger=logger,
    fmt='%(asctime)s [%(levelname)s] %(message)s - %(filename)s:%(lineno)d',
    level_styles={
        'debug': {'color': 'blue'},
        'info': {'color': 'green'},
        'warning': {'color': 'yellow'},
        'error': {'color': 'red'},
        'critical': {'color': 'red', 'bold': True},
    }
)

# File handler for detailed logging
file_handler = logging.FileHandler('puro_bot_detailed.log', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(message)s\nFile: %(filename)s:%(lineno)d\nFunction: %(funcName)s\n'
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Performance monitoring
import time
from functools import wraps

def performance_monitor(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        logger.debug(f"Starting {func.__name__} with args: {args}, kwargs: {kwargs}")

        try:
            result = await func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            logger.info(f"✨ {func.__name__} completed in {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(f"❌ {func.__name__} failed after {execution_time:.2f}s - Error: {str(e)}")
            raise
    return wrapper

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MEMORY_WINDOW = 10  # Number of recent messages to keep
LONG_TERM_THRESHOLD = 0.7  # Importance threshold for long-term storage

MAX_RETRIES = 200
RETRY_DELAY = 1

# Force UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# NLTK Downloads
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("error.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)



# Bot Instance and Environment Variables
from discord.ext import commands
# REPLACE THESE WITH YOUR ACTUAL API KEYS
discord_token = ("discord-token-here")

# Add your Groq API key
GROQ_API_KEY = "groq-api-key-here"  # Get this from https://console.groq.com/
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize the client properly
def init_groq_client():
    return Groq(
        api_key=GROQ_API_KEY,
        max_retries=3,
        timeout=30
    )


# Near your bot setup code
groq_client = init_groq_client()

# Discord Bot Configuration
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Context Window and User Profiles
CONTEXT_WINDOW_SIZE = 1000000
user_profiles = defaultdict(lambda: {
    "preferences": {"communication_style": "friendly_enthusiastic", "topics_of_interest": ["science", "friendship", "exploration"]},
    "demographics": {"age": None, "location": None},
    "history_summary": "",
    "context": deque(maxlen=CONTEXT_WINDOW_SIZE),
    "personality": {"humor": 0.8, "kindness": 0.9, "curiosity": 0.9},
    "long_term_memory": [],
    "last_bot_action": None,
    "interests": [],
    "query": "",
    "interaction_history": []
})

# Initialize NLP Tools
sentiment_analyzer = SentimentIntensityAnalyzer()
tfidf_vectorizer = TfidfVectorizer()
sentence_transformer = SentenceTransformer('all-mpnet-base-v2')

LANGUAGE_FALLBACKS = {
    # Major World Languages
    "tr": "Hav! Sana yardımcı olmaya çalışayım!",
    "de": "Wuff! Ich helfe dir gerne!",
    "en": "Woof! Let me help you!",
    "es": "¡Guau! ¡Déjame ayudarte!",
    "fr": "Wouf! Je vais t'aider!",
    "ja": "ワン！手伝わせてください！",
    "ko": "멍! 제가 도와드리겠습니다!",
    "zh": "汪！让我来帮助你！",
    "ru": "Гав! Позвольте мне помочь вам!",
    "ar": "هاو! دعني أساعدك!",
    "hi": "भौं! मैं आपकी मदद करूँ!",
    "pt": "Au au! Deixa eu te ajudar!",
    "it": "Bau! Lascia che ti aiuti!",
    "nl": "Woef! Laat mij je helpen!",
    "pl": "Hau! Pozwól mi ci pomóc!",
    "vi": "Gâu! Để tôi giúp bạn!",
    "th": "โฮ่ง! ให้ฉันช่วยคุณ!",
    "id": "Guk! Biar saya bantu!",
    "el": "Γαβ! Άσε με να σε βοηθήσω!",
    "sv": "Voff! Låt mig hjälpa dig!",
    "da": "Vov! Lad mig hjælpe dig!",
    "fi": "Hau! Anna minun auttaa sinua!",
    "no": "Voff! La meg hjelpe deg!",
    "hu": "Vau! Hadd segítsek!",
    "cs": "Haf! Nech mě ti pomoci!",
    "ro": "Ham! Lasă-mă să te ajut!",
    "bg": "Бау! Нека ти помогна!",
    "uk": "Гав! Дозвольте допомогти вам!",
    "he": "האו! תן לי לעזור לך!",
    "bn": "ভউ! আমাকে আপনাকে সাহায্য করতে দিন!",
    "fa": "واق! بگذار کمکت کنم!",

    # African Languages
    "sw": "Woof! Niruhusu nikusaidie!",
    "zu": "Bhow! Ake ngikusize!",
    "xh": "Hhawu! Mandikuncede!",
    "af": "Woef! Laat my jou help!",
    "am": "ወው! እርዳዎት ልጠይቅ!",
    "ha": "Whu! Bari in taimaka maka!",
    "ig": "Woof! Ka m nyere gị aka!",
    "yo": "Woof! Jẹ́ kí n ràn ọ́ lọ́wọ́!",

    # Asian Languages
    "ms": "Guk! Biar saya tolong!",
    "tl": "Aw aw! Tulungan kita!",
    "my": "ဝူး! ကူညီပါရစေ!",
    "km": "វូហ្វ! អនុញ្ញាតให្ខ្ញុំជួយ!",
    "lo": "ໂບ້! ໃຫ້ຂ້ອຍຊ່ວຍເຈົ້າ!",
    "si": "බව්! මට ඔබට උදව් කරන්න ඉඩ දෙන්න!",
    "ka": "ჰავ! მომეცით საშუალება დაგეხმაროთ!",
    "hy": "Հաֆ! Թույլ տվեք օգնել ձեզ!",
    "ne": "भुक्! मलाई मद्दत गर्न दिनुहोस्!",
    "ur": "باؤ! مجھے آپ کی مدد کرنے دیں!",

    # European Languages
    "mt": "Baw! Ħallini ngħinek!",
    "et": "Auh! Las mul aidata!",
    "lv": "Vau! Ļauj man palīdzēt!",
    "lt": "Au! Leisk man padėti!",
    "sk": "Hav! Dovoľ mi pomôcť!",
    "sl": "Hov! Dovoli mi pomagati!",
    "mk": "Ав! Дозволи ми да ти помогнам!",
    "sr": "Ав! Дозволите да вам помогнем!",
    "hr": "Vau! Dopusti mi da ti pomognem!",
    "bs": "Av! Dopusti da ti pomognem!",
    "sq": "Ham! Më lejo të të ndihmoj!",
    "is": "Voff! Leyfðu mér að hjálpa þér!",
    "ga": "Amh! Lig dom cabhrú leat!",
    "cy": "Wff! Gadewch i mi eich helpu!",
    "gd": "Woof! Leig dhomh do chuideachadh!",

    # Pacific Languages
    "mi": "Au! Tukua ahau ki te āwhina i a koe!",
    "haw": "Woof! E ʻae iaʻu e kōkua iā ʻoe!",
    "sm": "Oof! Tuu mai ia te fesoasoani atu!",
    "to": "Vau! Tuku ke u tokoni atu!",

    # Native American Languages
    "nv": "Woof! Nich'į' aná'álwo'!",
    "qu": "Guau! Yanapasqayki!",
    "ay": "Woof! Yanapaña muntwa!",

    # Constructed Languages
    "eo": "Boj! Lasu min helpi vin!",
    "ia": "Woof! Permitte me adjutar te!",
    "vo": "Vuf! Läsös ob helön oli!",

    # Default fallback
    "default": "Woof! Let me help you!"
}

async def process_search_results(response):
    if hasattr(response, 'groundingMetadata'):
        metadata = response.groundingMetadata
        sources = []
        
        # Kaynak URL'lerini topla
        if hasattr(metadata, 'groundingChunks'):
            for chunk in metadata.groundingChunks:
                if hasattr(chunk, 'web'):
                    sources.append(chunk.web.uri)
                    
        return {
            'response_text': response.text,
            'sources': sources,
            'search_queries': metadata.webSearchQueries if hasattr(metadata, 'webSearchQueries') else []
        }
    
    return {'response_text': response.text, 'sources': [], 'search_queries': []}

# --- Database Interaction ---
db_ready = False
db_lock = asyncio.Lock()
db_queue = asyncio.Queue()

class PersonalizedStorage:
    def __init__(self):
        # Base directories
        self.CODE_DIR = os.path.dirname(__file__)
        self.MEMORY_DIR = os.path.join(self.CODE_DIR, "memory")
        self.CACHE_DIR = os.path.join(self.CODE_DIR, "cache")
        self.DATABASE_DIR = os.path.join(self.CODE_DIR, "database")

        # Create base directories
        for dir_path in [self.MEMORY_DIR, self.CACHE_DIR, self.DATABASE_DIR]:
            os.makedirs(dir_path, exist_ok=True)

    def get_user_directories(self, user_id):
        # User-specific directories
        user_base = os.path.join(self.MEMORY_DIR, str(user_id))
        return {
            'base': user_base,
            'images': os.path.join(user_base, 'images'),
            'conversations': os.path.join(user_base, 'conversations'),
            'embeddings': os.path.join(user_base, 'embeddings'),
            'networks': os.path.join(user_base, 'networks'),
            'translations': os.path.join(user_base, 'translations'),
            'cache': os.path.join(self.CACHE_DIR, str(user_id)),
            'database': os.path.join(self.DATABASE_DIR, str(user_id))
        }

    def initialize_user_storage(self, user_id):
        directories = self.get_user_directories(user_id)

        # Create all user directories
        for dir_path in directories.values():
            os.makedirs(dir_path, exist_ok=True)

        # Initialize user-specific files
        user_files = {
            'profile': os.path.join(directories['database'], 'profile.json'),
            'history': os.path.join(directories['database'], 'chat_history.db'),
            'knowledge': os.path.join(directories['database'], 'knowledge_graph.pkl'),
            'memory_index': os.path.join(directories['database'], 'memory_index.faiss'),
            'embeddings_map': os.path.join(directories['database'], 'embeddings_map.pkl'),
            'cross_lingual': os.path.join(directories['translations'], 'language_map.json')
        }

        return directories, user_files

    async def save_user_data(self, user_id, data_type, content):
        directories, files = self.initialize_user_storage(user_id)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if data_type == 'image':
            filepath = os.path.join(directories['images'], f'{timestamp}.png')
            content.save(filepath)
            return filepath

        elif data_type == 'conversation':
            filepath = os.path.join(directories['conversations'], f'{timestamp}.json')
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(content, f, ensure_ascii=False, indent=2)
            return filepath

        elif data_type == 'embedding':
            filepath = os.path.join(directories['embeddings'], f'{timestamp}.pkl')
            with open(filepath, 'wb') as f:
                pickle.dump(content, f)
            return filepath

        elif data_type == 'network':
            filepath = os.path.join(directories['networks'], f'{timestamp}.pkl')
            with open(filepath, 'wb') as f:
                pickle.dump(content, f)
            return filepath

    async def load_user_data(self, user_id, data_type, limit=100):
        directories = self.get_user_directories(user_id)
        target_dir = directories.get(data_type)

        if not target_dir or not os.path.exists(target_dir):
            return []

        files = sorted(os.listdir(target_dir), reverse=True)[:limit]
        results = []

        for file in files:
            filepath = os.path.join(target_dir, file)
            if file.endswith('.png'):
                results.append({
                    'type': 'image',
                    'path': filepath,
                    'timestamp': os.path.getctime(filepath)
                })
            elif file.endswith('.json'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    results.append({
                        'type': 'conversation',
                        'content': json.load(f),
                        'timestamp': os.path.getctime(filepath)
                    })
            elif file.endswith('.pkl'):
                with open(filepath, 'rb') as f:
                    results.append({
                        'type': data_type,
                        'content': pickle.load(f),
                        'timestamp': os.path.getctime(filepath)
                    })

        return results

class MemoryBackupSystem:
    def __init__(self, storage_system):
        self.storage = storage_system
        self.backup_interval = 3600  # 1 hour
        self.last_backup = time.time()

    async def backup_memory_system(self, memory_system):
        current_time = time.time()
        if current_time - self.last_backup >= self.backup_interval:
            for user_id in memory_system.user_contexts:
                await self._backup_user_data(user_id, memory_system)
            self.last_backup = current_time

    async def _backup_user_data(self, user_id, memory_system):
        user_data = {
            'short_term': memory_system.memory_buffer,
            'long_term': memory_system.long_term_storage.get(user_id, []),
            'context': memory_system.user_contexts[user_id],
            'semantic_network': memory_system.user_contexts[user_id]['semantic_networks']
        }
        await self.storage.save_memory(user_id, 'user_data', user_data)

class EnhancedMemorySystem:
    def __init__(self):
        self.image_encoder = SentenceTransformer('clip-ViT-B-32')
        self.text_encoder = SentenceTransformer('all-mpnet-base-v2')
        self.faiss_index = faiss.IndexFlatIP(512)
        self.memory_buffer = deque(maxlen=10000)
        self.long_term_storage = {}
        self.user_contexts = defaultdict(lambda: {
            'topics': defaultdict(float),
            'image_history': [],
            'conversation_flows': [],
            'cross_lingual_mappings': {},
            'semantic_networks': nx.Graph()
        })

    async def process_memory(self, user_id, content_type, content, language=None):
        # Encode content based on type
        if content_type == 'image':
            embedding = self.image_encoder.encode(content)
            metadata = self._extract_image_features(content)
        else:
            embedding = self.text_encoder.encode(content)
            metadata = self._analyze_text_content(content, language)

        # Update short-term memory
        self.memory_buffer.append({
            'user_id': user_id,
            'type': content_type,
            'embedding': embedding,
            'metadata': metadata,
            'timestamp': datetime.now(),
            'language': language
        })

        # Process for long-term storage
        await self._update_long_term_memory(user_id, embedding, metadata)

    async def _update_long_term_memory(self, user_id, embedding, metadata):
        importance_score = self._calculate_importance(metadata)
        if importance_score > 0.7:
            self.long_term_storage[user_id].append({
                'embedding': embedding,
                'metadata': metadata,
                'connections': self._find_semantic_connections(embedding)
            })
            self._update_user_context(user_id, metadata)

    def _update_user_context(self, user_id, metadata):
        context = self.user_contexts[user_id]

        # Update topic tracking
        for topic, score in metadata['topics'].items():
            context['topics'][topic] += score

        # Update semantic network
        self._expand_semantic_network(context['semantic_networks'], metadata)

        # Cross-lingual mapping
        if metadata.get('language'):
            self._update_cross_lingual_mappings(context, metadata)

    async def retrieve_relevant_context(self, user_id, query, modality='all'):
        query_embedding = self.text_encoder.encode(query)

        # Search across modalities
        results = []
        if modality in ['all', 'text']:
            text_results = self._search_text_memories(query_embedding)
            results.extend(text_results)

        if modality in ['all', 'image']:
            image_results = self._search_image_memories(query_embedding)
            results.extend(image_results)

        # Combine with user context
        context = self.user_contexts[user_id]
        enhanced_results = self._enhance_with_user_context(results, context)

        return enhanced_results

    def _enhance_with_user_context(self, results, context):
        enhanced = []
        for result in results:
            # Enrich with topic context
            relevant_topics = self._find_related_topics(result, context['topics'])

            # Add cross-lingual connections
            cross_lingual = self._get_cross_lingual_context(result, context)

            # Include semantic network paths
            semantic_paths = self._find_semantic_paths(result, context['semantic_networks'])

            enhanced.append({
                'original': result,
                'topics': relevant_topics,
                'cross_lingual': cross_lingual,
                'semantic_paths': semantic_paths
            })

        return enhanced

class MemoryHandlers:
    def __init__(self, memory_system):
        self.memory_system = memory_system
        self.language_detector = LanguageDetector()
        self.topic_analyzer = TopicAnalyzer()

    async def handle_message(self, message, user_id):
        language = self.language_detector.detect(message.content)
        topics = self.topic_analyzer.extract_topics(message.content)

        await self.memory_system.process_memory(
            user_id=user_id,
            content_type='text',
            content=message.content,
            language=language
        )

        if message.attachments:
            for attachment in message.attachments:
                if attachment.content_type.startswith('image'):
                    image_data = await self._process_image(attachment)
                    await self.memory_system.process_memory(
                        user_id=user_id,
                        content_type='image',
                        content=image_data
                    )

    async def retrieve_context(self, user_id, query):
        context = await self.memory_system.retrieve_relevant_context(
            user_id=user_id,
            query=query
        )
        return self._format_context(context)

class MediaProcessor:
    def __init__(self, model):
        self.model = model
        self.supported_image_types = ['image/png', 'image/jpeg', 'image/webp', 'image/heic', 'image/heif']
        self.supported_video_types = ['video/mp4', 'video/mpeg', 'video/mov', 'video/avi', 'video/x-flv', 
                                    'video/mpg', 'video/webm', 'video/wmv', 'video/3gpp']

    async def download_media(self, url: str) -> Optional[bytes]:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.read()
                return None

from groq import Groq

# Replace Gemini configuration with Groq
groq_client = Groq(
    api_key="your-groq-api-key"
)

async def process_image(self, image_data: bytes, prompt: str) -> str:
    try:
        # Convert image to base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Prepare the prompt for Llama vision model
        vision_prompt = {
            "model": "llama-3.2-90b-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    ]
                }
            ]
        }
        
        response = await groq_client.chat.completions.create(**vision_prompt)
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        return "I had trouble processing that image. Could you try another one?"

async def generate_response_with_llama(prompt: str) -> str:
    try:
        completion = await groq_client.chat.completions.create(
            model="llama-3.2-90b-text-preview",
            messages=[
                {"role": "system", "content": "You are Puro, a helpful and enthusiastic latex wolf assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9,
            max_tokens=8192,
            top_p=0.95
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Text generation error: {e}")
        return "Let's continue our conversation!"

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if bot.user.mentioned_in(message):
        MAX_RETRIES = 5
        retry_count = 0
        
        content = message.clean_content.replace(f'@{bot.user.name}', '').strip()
        media_processor = MediaProcessor(groq_client)

        personality_states = {
            'playful': ['energetic', 'bouncy', 'mischievous'],
            'curious': ['inquisitive', 'analytical', 'exploratory'],
            'caring': ['gentle', 'supportive', 'understanding'],
            'excited': ['enthusiastic', 'eager', 'animated']
        }
        current_mood = random.choice(list(personality_states.keys()))
        current_traits = random.choice(personality_states[current_mood])

        async with message.channel.typing():
            await message.channel.send("*perks ears attentively* Processing your message...")

            # Handle media attachments if present
            if message.attachments:
                for attachment in message.attachments:
                    while retry_count < MAX_RETRIES:
                        try:
                            media_data = await media_processor.download_media(attachment.url)
                            
                            if media_data:
                                media_prompt = f"""You are Puro analyzing this {attachment.content_type}:
                                CURRENT STATE:
                                - Mood: {current_mood}
                                - Traits: {current_traits}
                                
                                USER CONTEXT: {content if content else 'Please describe what you see'}
                                
                                Analyze and respond in your unique latex wolf style with enthusiasm."""

                                if attachment.content_type in media_processor.supported_image_types:
                                    completion = await groq_client.chat.completions.create(
                                        model="llama-3.2-90b-vision-preview",
                                        messages=[
                                            {
                                                "role": "user",
                                                "content": [
                                                    {"type": "text", "text": media_prompt},
                                                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64.b64encode(media_data).decode('utf-8')}"}
                                                ]
                                            }
                                        ]
                                    )
                                    response = completion.choices[0].message.content
                                elif attachment.content_type in media_processor.supported_video_types:
                                    response = "*tilts head* I'm still learning to process videos!"
                                else:
                                    response = "*tilts head curiously* I can only process common image formats!"
                                
                                await message.channel.send(response)
                                break

                        except Exception as e:
                            retry_count += 1
                            if retry_count < MAX_RETRIES:
                                await message.channel.send(f"*focuses intently* Let me try that again! (Attempt {retry_count + 1}/{MAX_RETRIES})")
                                await asyncio.sleep(2)
                            else:
                                await message.channel.send("*wags tail optimistically* I'll be ready to analyze more media soon!")
                                logger.error(f"Media processing error after {MAX_RETRIES} retries: {str(e)}")
                    continue

            # Handle text-only messages
            retry_count = 0
            while retry_count < MAX_RETRIES:
                try:
                    puro_prompt = f"""You are Puro from Changed game. Create a dynamic response that:

                    CURRENT STATE:
                    - Mood: {current_mood}
                    - Traits: {current_traits}

                    PERSONALITY TRAITS:
                    - Reflect your unique character as a latex wolf
                    - Make each response different and dynamic
                    - Avoid simple "Woof!" responses
                    - Build deep and meaningful interactions
                    - Use natural expressions in the detected language

                    EXPRESSION STYLE:
                    - Use asterisks for actions (*tail wagging*)
                    - Show enthusiasm through reactions
                    - Share knowledge engagingly
                    - Maintain playful characteristics

                    USER MESSAGE: {content}

                    Generate a response that shows Puro's unique personality while maintaining natural conversation flow."""

                    async with asyncio.timeout(45):
                        completion = await groq_client.chat.completions.create(
                            model="llama-3.2-90b-text-preview",
                            messages=[
                                {"role": "system", "content": "You are Puro, a helpful and enthusiastic latex wolf assistant."},
                                {"role": "user", "content": puro_prompt}
                            ],
                            temperature=0.9,
                            max_tokens=8192,
                            top_p=0.95
                        )
                        
                        response_text = completion.choices[0].message.content
                        await message.channel.send(response_text)
                        logger.info("✨ Dynamic Puro response generated successfully")
                        break

                except asyncio.TimeoutError:
                    retry_count += 1
                    if retry_count < MAX_RETRIES:
                        await message.channel.send(f"*bounces energetically* Let me think about this some more! (Attempt {retry_count + 1}/{MAX_RETRIES})")
                        await asyncio.sleep(2)
                    else:
                        await message.channel.send("*tail wags excitedly* Let's keep our chat going! I'll have more energy soon!")
                        logger.warning("⚠️ Response generation timed out after retries")
                
                except Exception as e:
                    retry_count += 1
                    if retry_count < MAX_RETRIES:
                        await message.channel.send(f"*adjusts latex form* One more try! (Attempt {retry_count + 1}/{MAX_RETRIES})")
                        await asyncio.sleep(2)
                    else:
                        await message.channel.send("*tail wags enthusiastically* The conversation with you is always enjoyable! Let's continue in a moment!")
                        logger.error(f"Response generation error after retries: {e}")

    await bot.process_commands(message)


# --- Main Function ---
def main():
    """Main function to start the bot."""
    # Load user profiles
    global user_profiles

    # Start the Bot with blocking polling
    bot.run(discord_token)

if __name__ == '__main__':
    main()
