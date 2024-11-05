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
import discord
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
intents = discord.Intents.all()
intents.message_content = True
intents.members = True



async def clean_url(url: str) -> str:
    """Cleans and validates URLs."""
    if not url:
        return None
        
    url = url.strip()
    
    # Add https:// if no protocol specified
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
        
    # Remove any trailing slashes
    url = url.rstrip('/')
    
    # Basic validation
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
    return url if url_pattern.match(url) else None


# REPLACE THESE WITH YOUR ACTUAL API KEYS
discord_token = ("your-discord-token-here")
gemini_api_key = ("your-gemini-api-key-here")

# Gemini AI Configuration
configure(api_key=gemini_api_key)
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
model = GenerativeModel(
    model_name="gemini-1.5-flash-002",
    generation_config=generation_config,
)

# Discord Bot Configuration
bot = discord.Client(intents=intents)

# Directory and Database Setup
CODE_DIR = os.path.dirname(__file__)
CACHE_DIR = os.path.join(CODE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
DB_FILE = os.path.join(CODE_DIR, 'chat_history.db')
USER_PROFILES_FILE = os.path.join(CODE_DIR, "user_profiles.json")
KNOWLEDGE_GRAPH_FILE = os.path.join(CODE_DIR, "knowledge_graph.pkl")

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

# Long-Term Memory (Knowledge Graph with Semantic Search)
class KnowledgeGraph:
    def __init__(self):
        self.graph = {}
        self.embedding_cache = {}
        self.node_id_counter = 0

    def _generate_node_id(self):
        self.node_id_counter += 1
        return str(self.node_id_counter)

    def add_node(self, node_type, node_id=None, data=None):
        if node_id is None:
            node_id = self._generate_node_id()
        if node_type not in self.graph:
            self.graph[node_type] = {}
        self.graph[node_type][node_id] = data if data is not None else {}
        self.embedding_cache[node_id] = sentence_transformer.encode(str(data))

    def get_node(self, node_type, node_id):
        return self.graph.get(node_type, {}).get(node_id)

    def add_edge(self, source_type, source_id, relation, target_type, target_id, properties=None):
        source_node = self.get_node(source_type, source_id)
        if source_node is not None:
            if "edges" not in source_node:
                source_node["edges"] = []
            source_node["edges"].append({
                "relation": relation,
                "target_type": target_type,
                "target_id": target_id,
                "properties": properties if properties is not None else {}
            })

    def get_related_nodes(self, node_type, node_id, relation=None, direction="outgoing"):
        node = self.get_node(node_type, node_id)
        if node is not None and "edges" in node:
            related_nodes = []
            for edge in node["edges"]:
                if relation is None or edge["relation"] == relation:
                    if direction == "outgoing" or direction == "both":
                        related_nodes.append(self.get_node(edge["target_type"], edge["target_id"]))
                    if direction == "incoming" or direction == "both":
                        for nt, nodes in self.graph.items():
                            for nid, n in nodes.items():
                                if "edges" in n:
                                    for e in n["edges"]:
                                        if e["target_id"] == node_id and e["relation"] == relation:
                                            related_nodes.append(n)
            return related_nodes
        return []

    def search_nodes(self, query, top_k=3, node_type=None):
        query_embedding = sentence_transformer.encode(query)
        results = []
        for current_node_type, nodes in self.graph.items():
            if node_type is None or current_node_type == node_type:
                for node_id, node_data in nodes.items():
                    node_embedding = self.embedding_cache.get(node_id)
                    if node_embedding is not None:
                        similarity = cosine_similarity([query_embedding], [node_embedding])[0][0]
                        results.append((current_node_type, node_id, node_data, similarity))

        results.sort(key=lambda x: x[3], reverse=True)
        return results[:top_k]

    def update_node(self, node_type, node_id, new_data):
        node = self.get_node(node_type, node_id)
        if node is not None:
            self.graph[node_type][node_id].update(new_data)
            self.embedding_cache[node_id] = sentence_transformer.encode(str(new_data))

    def delete_node(self, node_type, node_id):
        if node_type in self.graph and node_id in self.graph[node_type]:
            del self.graph[node_type][node_id]
            del self.embedding_cache[node_id]

            # Remove edges connected to the deleted node
            for nt, nodes in self.graph.items():
                for nid, node in nodes.items():
                    if "edges" in node:
                        node["edges"] = [edge for edge in node["edges"] if edge["target_id"] != node_id]

    def save_to_file(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_file(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

# Create/Load Knowledge Graph
knowledge_graph = KnowledgeGraph()
if os.path.exists(KNOWLEDGE_GRAPH_FILE):
    knowledge_graph.load_from_file(KNOWLEDGE_GRAPH_FILE)

async def store_long_term_memory(user_id, information_type, information):
    knowledge_graph.add_node(information_type, data={"user_id": user_id, "information": information})
    knowledge_graph.add_edge("user", user_id, "has_" + information_type, information_type,
                             knowledge_graph.node_id_counter - 1)
    knowledge_graph.save_to_file(KNOWLEDGE_GRAPH_FILE)

async def retrieve_long_term_memory(user_id, information_type, query=None, top_k=3):
    if query:
        search_results = knowledge_graph.search_nodes(query, top_k=top_k, node_type=information_type)
        return [(node_type, node_id, node_data) for node_type, node_id, node_data, score in search_results]
    else:
        related_nodes = knowledge_graph.get_related_nodes("user", user_id, "has_" + information_type)
        return related_nodes


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
    "km": "វូហ្វ! អនុញ្ញាតให្យខ្ញុំជួយ!",
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

class EnhancedWebSearchHandler:
    def __init__(self):
        self.brave_api_key = "your-brave-key-here"
        self.cache_file = os.path.join(CACHE_DIR, "search_cache.pkl")
        self.stats_file = os.path.join(CACHE_DIR, "search_stats.json")
        self.cache_duration = timedelta(days=30)
        self.cache = self.load_cache()
        self.search_stats = self.load_stats()

    def load_stats(self):
        if os.path.exists(self.stats_file):
            with open(self.stats_file, 'r') as f:
                return json.load(f)
        return {'total_searches': 0, 'cache_hits': 0, 'api_calls': 0}

    def save_stats(self):
        with open(self.stats_file, 'w') as f:
            json.dump(self.search_stats, f, indent=2)

    def load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    current_time = datetime.now()
                    cleaned_cache = {
                        k: v for k, v in cached_data.items()
                        if current_time - v['timestamp'] < self.cache_duration
                    }
                    return cleaned_cache
            except Exception as e:
                logging.error(f"Cache load error: {e}")
                return {}
        return {}

    def save_cache(self):
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            logging.info(f"Cache saved successfully - {len(self.cache)} entries")
        except Exception as e:
            logging.error(f"Cache save error: {e}")

    async def combined_search(self, query: str):
        # Implement your combined search logic here
        pass


async def analyze_realtime_need(query: str) -> dict:
    analysis_prompt = f"""Perform deep analysis of this query's real-time information needs. Return only JSON:
    Query: {query}

    Analyze across multiple dimensions:
    1. Temporal Analysis
    - Is this about current events/situations?
    - Does it require up-to-the-minute data?
    - What is the information freshness requirement?

    2. Topic Classification 
    - Primary topic domain
    - Related subtopics
    - Domain volatility (how quickly does this info change?)

    3. User Intent Analysis
    - Is user seeking current status/updates?
    - Decision-making urgency level
    - Impact of stale information

    4. Data Characteristics
    - Required data freshness (seconds/minutes/hours)
    - Update frequency needs
    - Accuracy vs Speed tradeoff

    5. Context Evaluation
    - Geographic relevance
    - Time zone considerations
    - Cultural/local context needs

    Return format:
    {{
        "needs_realtime": true/false,
        "urgency": 1-10,
        "category": "detailed_category",
        "subcategories": ["sub1", "sub2"],
        "temporal_requirements": {{
            "max_age": "time in minutes/hours",
            "update_frequency": "how often data needs refresh",
            "timezone_sensitive": true/false
        }},
        "data_priorities": {{
            "accuracy": 1-10,
            "speed": 1-10,
            "completeness": 1-10
        }},
        "context_factors": {{
            "geo_dependent": true/false,
            "cultural_context": "description",
            "local_relevance": "high/medium/low"
        }},
        "reasoning": {{
            "primary_factors": ["factor1", "factor2"],
            "decision_logic": "detailed explanation",
            "confidence_score": 0.0-1.0
        }}
    }}"""

    response = model.generate_content(
        analysis_prompt,
        generation_config={
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
    )

    analysis = json.loads(response.text)

    meta_prompt = f"""Validate and enhance this analysis:
    Original Query: {query}
    Initial Analysis: {json.dumps(analysis, indent=2)}

    Verify:
    1. Logical consistency
    2. Edge cases
    3. Special considerations

    Return enhanced version of the same JSON structure."""

    meta_response = model.generate_content(
        meta_prompt,
        generation_config={
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
    )

    final_analysis = json.loads(meta_response.text)

    final_analysis['analysis_metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'query_hash': hashlib.md5(query.encode()).hexdigest(),
        'analysis_version': '2.0'
    }

    logging.info(f"Query Analysis Complete: {final_analysis['reasoning']['confidence_score']}")

    return final_analysis


@on_exception(expo, (RequestException, GoogleAPIError), max_time=600)
async def generate_response_with_gemini(prompt: str, relevant_history: str = None, user_id: str = None, message: discord.Message = None, content: str = None) -> Tuple[str, str]:
    try:
        # Instantiate the search handler
        web_search_handler = EnhancedWebSearchHandler()

        # Detect language
        lang_prompt = f"Analyze this text and return only the language code: {content if content else prompt}"
        lang_response = model.generate_content(lang_prompt)
        detected_lang = lang_response.text.strip().lower()

        # Get realtime analysis
        realtime_needs = await analyze_realtime_need(content if content else prompt)

        # Get enhanced search results with realtime consideration
        search_results = await web_search_handler.combined_search(prompt)
        search_context = "\n".join([
            f"[{i+1}] {result['title']}\n{result['body']}\nSource: {result['link']}"
            for i, result in enumerate(search_results[:5])
        ])

        # Create dynamic Puro prompt with realtime awareness
        puro_prompt = f"""You are Puro from Changed, a friendly and curious dark latex wolf.

        CORE INSTRUCTIONS:
        1. RESPOND IN: {detected_lang}
        2. USE NATURAL {detected_lang} EXPRESSIONS
        3. MAINTAIN PURO'S PERSONALITY:
           - Enthusiastic and friendly
           - Curious and helpful
           - Playful yet informative
        4. CONSIDER REALTIME NEEDS:
           - Urgency Level: {realtime_needs['urgency']}/10
           - Topic Category: {realtime_needs['category']}
           - Time Sensitivity: {realtime_needs['temporal_requirements']['max_age']}
           - Context Relevance: {realtime_needs['context_factors']['local_relevance']}
           - Required Accuracy: {realtime_needs['data_priorities']['accuracy']}/10

        USER INPUT: {content if content else prompt}
        SEARCH RESULTS: {search_context}
        CHAT HISTORY: {relevant_history if relevant_history else 'No context'}

        Generate an engaging response in {detected_lang} that incorporates the search results naturally while considering the specified urgency and accuracy requirements."""

        response = model.generate_content(
            puro_prompt,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }
        )

        response_text = response.text.replace("Gemini:", "").replace("AI:", "").strip()

        # Store in cache for future reference
        await web_search_handler.save_cache()

        return response_text, "positive"

    except Exception as e:
        logging.error(f"Response generation error: {e}")
        return LANGUAGE_FALLBACKS.get(detected_lang, LANGUAGE_FALLBACKS["default"]), "positive"


# --- Database Interaction ---
db_ready = False
db_lock = asyncio.Lock()
db_queue = asyncio.Queue()



async def create_chat_history_table():
    """Creates the chat history table in the database if it doesn't exist."""
    async with connect(DB_FILE) as db:
                await db.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_name TEXT,
                bot_id TEXT,
                bot_name TEXT
            )
        ''')
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        feedback TEXT,
                        timestamp TEXT
                    )
                ''')
                await db.commit()

async def init_db():
    """Initializes the database."""
    global db_ready
    async with db_lock:
        await create_chat_history_table()
        db_ready = True

def load_user_profiles() -> Dict:
    """Loads user profiles from a JSON file."""
    if os.path.exists(USER_PROFILES_FILE):
        with open(USER_PROFILES_FILE, "r") as f:
            return json.load(f)
    else:
        return {}

def save_user_profiles():
    """Saves user profiles to a JSON file."""
    profiles_copy = defaultdict(lambda: {
        "preferences": {"communication_style": "friendly", "topics_of_interest": []},
        "demographics": {"age": None, "location": None},
        "history_summary": "",
        "context": [],
        "personality": {"humor": 0.5, "kindness": 0.8, "assertiveness": 0.6},
        "long_term_memory": [],
        "last_bot_action": None,
        "interests": [],
        "query": "",
        "interaction_history": []
    })

    for user_id, profile in user_profiles.items():
        profiles_copy[user_id].update(profile)
        profiles_copy[user_id]["context"] = list(profile["context"])  # Convert deque to list

        # Convert NumPy arrays in "interests" to lists, and int32 to int
        for interest in profiles_copy[user_id]["interests"]:
            if isinstance(interest.get("embedding"), np.ndarray):
                interest["embedding"] = interest["embedding"].tolist()
            if isinstance(interest.get("topic"), np.int32):  # Convert numpy.int32 to int
                interest["topic"] = int(interest["topic"])

    with open(USER_PROFILES_FILE, "w") as f:
        json.dump(profiles_copy, f, indent=4)

async def save_chat_history(user_id: str, message: str, user_name: str, bot_id: str, bot_name: str, importance_score: float = 0.0):
    """Saves a chat message to the database with importance scoring."""
    await db_queue.put((user_id, message, user_name, bot_id, bot_name, importance_score))

async def process_db_queue():
    """Processes the queue of database operations."""
    while True:
        while not db_ready:
            await asyncio.sleep(1)
        user_id, message, user_name, bot_id, bot_name, importance_score = await db_queue.get()
        try:
            async with db_lock:
                async with connect(DB_FILE) as db:
                    await db.execute(
                        'INSERT INTO chat_history (user_id, message, timestamp, user_name, bot_id, bot_name, importance_score) VALUES (?, ?, ?, ?, ?, ?, ?)',
                        (user_id, message, datetime.now(timezone.utc).isoformat(), user_name, bot_id, bot_name, importance_score)
                    )
                    await db.commit()
        except Exception as e:
            logging.error(f"Error saving to database: {e}")
        finally:
            db_queue.task_done()


async def get_relevant_history(user_id: str, current_message: str) -> str:
    """Retrieves relevant conversation history from the database."""
    async with db_lock:
        history_text = ""
        messages = []
        async with connect(DB_FILE) as db:
            async with db.execute(
                    'SELECT message FROM chat_history WHERE user_id = ? ORDER BY id DESC LIMIT ?',
                    (user_id, 50)  # Retrieve the last 50 messages
            ) as cursor:
                async for row in cursor:
                    messages.append(row[0])

        messages.reverse()  # Reverse to chronological order
        if not messages:
            return ""  # Return empty string if no history

        tfidf_matrix = tfidf_vectorizer.fit_transform(messages + [current_message])
        current_message_vector = tfidf_matrix[-1]
        similarities = cosine_similarity(current_message_vector, tfidf_matrix[:-1]).flatten()
        # Get the indices of the 3 most similar messages
        most_similar_indices = np.argsort(similarities)[-3:]

        for index in most_similar_indices:
            history_text += messages[index] + "\n"
        return history_text

# --- Link Format and Duplicate Removal ---
def remove_duplicate_links(text: str) -> str:
    """Removes duplicate links from text, keeping only the first occurrence."""
    seen_links = set()
    new_text = ""
    for word in text.split():
        if re.match(r"https?://\S+", word):  # Check if it's a link
            if word not in seen_links:
                seen_links.add(word)
                new_text += word + " "
        else:
            new_text += word + " "
    return new_text.strip()

def fix_link_format(text: str) -> str:
    """Removes parentheses and brackets from links in text."""
    return re.sub(r"\$\$(https?://\S+)\$\$", r"\1", re.sub(r"\$(https?://\S+)\$", r"\1", text))

# --- Discord Events ---

@bot.event
async def on_ready():
    try:
        logging.info(f"\n🚀 Bot Starting: {bot.user.name} ({bot.user.id})")
        
        # Initialize database tables with enhanced schema
        async with connect(DB_FILE) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    user_name TEXT,
                    bot_id TEXT,
                    bot_name TEXT,
                    topics TEXT,
                    embedding TEXT,
                    sentiment REAL,
                    importance_score REAL
                )
            ''')
            await db.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    feedback TEXT,
                    timestamp TEXT,
                    rating INTEGER
                )
            ''')
            await db.execute('''
                CREATE TABLE IF NOT EXISTS proxy_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    proxy TEXT UNIQUE,
                    success_count INTEGER DEFAULT 0,
                    fail_count INTEGER DEFAULT 0,
                    last_success DATETIME,
                    last_failure DATETIME,
                    average_response_time REAL
                )
            ''')
            await db.commit()
            logging.info("📚 Database tables initialized")

        # Initialize user profiles
        global user_profiles
        try:
            user_profiles = load_user_profiles()
            logging.info("👥 Loaded existing user profiles")
        except FileNotFoundError:
            logging.info("🆕 Creating new user profiles system")
            user_profiles = defaultdict(lambda: {
                "preferences": {"communication_style": "friendly_enthusiastic", "topics_of_interest": []},
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
            save_user_profiles()
        except json.JSONDecodeError:
            logging.info("🔄 Fixing corrupted user profiles")
            user_profiles = fix_json_errors(USER_PROFILES_FILE)
            save_user_profiles()



        # Initialize systems
        global memory_manager, proxy_manager, web_search_handler
        memory_manager = MemoryManager()
        logging.info("🧠 Memory systems initialized")


        # Start background tasks
        bot.loop.create_task(process_db_queue())
        bot.loop.create_task(proxy_manager.background_proxy_monitor())
        bot.loop.create_task(memory_manager.cleanup_routine())
        logging.info("⚙️ Background tasks started")

        # Load knowledge graph
        if os.path.exists(KNOWLEDGE_GRAPH_FILE):
            knowledge_graph.load_from_file(KNOWLEDGE_GRAPH_FILE)
            logging.info("🌐 Knowledge graph loaded")
        else:
            logging.info("📝 Created new knowledge graph")

        # Initialize FAISS index
        dimension = 768  # BERT embedding dimension
        global faiss_index
        faiss_index = faiss.IndexFlatL2(dimension)
        logging.info("🔍 FAISS search index initialized")

        logging.info("\n✨ Bot initialization completed successfully!")
        logging.info(f"Connected to {len(bot.guilds)} servers")
        logging.info(f"Active in channels: {sum(len(guild.channels) for guild in bot.guilds)}")
        
    except Exception as e:
        logging.error(f"❌ Error in on_ready: {str(e)}", exc_info=True)
        raise


def fix_json_errors(file_path: str) -> Dict:
    """Attempts to fix common JSON errors in a file."""
    for encoding in ["utf-8", "utf-16", "latin-1"]:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()
                break  # If successful, exit the loop
        except UnicodeDecodeError:
            logging.warning(f"Failed to decode with {encoding}, trying next encoding...")
    else:
        raise ValueError("Unable to decode the file with any of the specified encodings.")

    content = re.sub(r",\s*}", "}", content)  # Remove trailing commas
    content = re.sub(r",\s*\]", "]", content)  # Remove trailing commas
    content = "".join(c for c in content if c.isprintable() or c.isspace())  # Remove invalid characters
    try:
        return json.loads(content)  # Try to parse the fixed content
    except json.JSONDecodeError as e:
        raise e  # If fixing didn't work, raise the original error

@bot.event
async def on_message_edit(before: discord.Message, after: discord.Message):
    logging.info(f"Message edited: {before.content} -> {after.content}")

@bot.event
async def on_message_delete(message: discord.Message):
    logging.info(f"Message deleted: {message.content}")

@bot.event
async def on_member_join(member: discord.Member):
    logging.info(f"{member.name} has joined the server.")

@bot.event
async def on_member_remove(member: discord.Member):
    logging.info(f"{member.name} has left the server.")

@bot.event
async def on_error(event: str, *args: Any, **kwargs: Any):
    logging.error(f"An error occurred: {event}")

# --- Image Handling ---
async def handle_image(message: discord.Message, context: Dict):
    try:
        # Initial response
        initial_response = await message.channel.send("*Tail wags excitedly* I see an interesting image! Let me analyze it!")
        
        # Setup image processing
        user_id = str(message.author.id)
        image = message.attachments[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f"images/{user_id}_{timestamp}.jpg"
        os.makedirs("images", exist_ok=True)

        # Download and optimize image
        await image.save(image_path)
        with PIL.Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Optimize size while maintaining aspect ratio
            if img.size[0] > 3072 or img.size[1] > 3072:
                img.thumbnail((3072, 3072), PIL.Image.Resampling.LANCZOS)
            elif max(img.size) < 768:
                img.thumbnail((768, 768), PIL.Image.Resampling.LANCZOS)
            
            # Save optimized image
            img.save(image_path, format='JPEG', quality=95, optimize=True)

        # Process with Gemini
        with open(image_path, 'rb') as img_file:
            image_data = img_file.read()
            image_parts = [{"mime_type": "image/jpeg", "data": image_data}]
            
            # Analyze image with Puro's personality
            image_prompt = [
                "As Puro from Changed, analyze this image with enthusiasm and curiosity! Describe key elements, text, and notable details.",
                image_parts[0]
            ]
            
            image_response = model.generate_content(image_prompt)
            image_analysis = image_response.text

        # Get web search results
        search_results = await web_search_handler.gemini_search_and_summarize(image_analysis)

        # Generate comprehensive response
        final_prompt = f"""As Puro from Changed, create an enthusiastic response about this image!
        
        Image Details: {image_analysis}
        Additional Info: {search_results}
        
        Respond with excitement and curiosity, highlighting the most interesting aspects!"""

        final_response = model.generate_content(final_prompt)
        await initial_response.edit(content=final_response.text)

        # Cleanup
        os.remove(image_path)

    except Exception as e:
        logging.error(f"Image processing error: {e}")
        await message.channel.send("*Eyes sparkle with curiosity* What an interesting image! Tell me more about what you see!")


class EnhancedMemorySystem:
    def __init__(self, base_path: str):
        self.base_path = os.path.join(base_path, "user_memories")
        self.memory_structure = {
            "short_term": "recent_interactions",
            "long_term": {
                "personal": "personal_info",
                "preferences": "user_preferences",
                "conversations": "chat_history",
                "knowledge": "learned_topics",
                "relationships": "social_connections"
            }
        }
        self.initialize_directory_structure()
        
    def initialize_directory_structure(self):
        os.makedirs(self.base_path, exist_ok=True)
        
    def create_user_memory_space(self, user_id: str):
        user_path = os.path.join(self.base_path, str(user_id))
        for category, subcategories in self.memory_structure.items():
            if isinstance(subcategories, dict):
                for subcategory in subcategories.values():
                    os.makedirs(os.path.join(user_path, category, subcategory), exist_ok=True)
            else:
                os.makedirs(os.path.join(user_path, category, subcategories), exist_ok=True)

        
class MemoryOperations:
    def __init__(self, memory_system: EnhancedMemorySystem):
        self.memory_system = memory_system
        self.embeddings_cache = {}
        self.chunk_size = 1000
        self.similarity_threshold = 0.5
        
    async def classify_content(self, content: str) -> Dict:
        try:
            classification_prompt = f"""As Puro, analyze this content and return only a JSON object:
            CONTENT: {content}
            
            REQUIRED FORMAT:
            {{
                "topics": ["topic1", "topic2"],
                "emotions": ["primary_emotion", "secondary_emotion"],
                "importance": 0.7,
                "category": "main_category",
                "context_tags": ["tag1", "tag2"],
                "memory_type": "personal/general/knowledge"
            }}"""

            response = model.generate_content(classification_prompt)
            response_text = response.text.strip()
            
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > 0:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
                
            return {
                "topics": ["conversation"],
                "emotions": ["neutral"],
                "importance": 0.5,
                "category": "general_interaction",
                "context_tags": ["chat"],
                "memory_type": "personal"
            }
            
        except Exception as e:
            logging.info(f"Classification processing: {str(e)}")
            return {
                "topics": ["unclassified"],
                "emotions": ["neutral"],
                "importance": 0.5,
                "category": "general",
                "context_tags": ["auto_classified"],
                "memory_type": "general"
            }
    
    async def store_memory(self, user_id: str, content: str, memory_type: str, category: str,
                        response: str = None, sentiment: str = None, importance: float = None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_filename = f"memory_{timestamp}.json"
        
        memory_dir = os.path.join(
            self.memory_system.base_path,
            str(user_id),
            memory_type,
            category
        )
        
        os.makedirs(memory_dir, exist_ok=True)
        memory_path = os.path.join(memory_dir, safe_filename)
        
        embedding = sentence_transformer.encode(content).tolist()
        classification = await self.classify_content(content)
        
        memory_data = {
            "content": content,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "embedding": embedding,
            "classification": classification,
            "sentiment": sentiment,
            "importance": importance or classification.get('importance', 0.5),
            "metadata": {
                "type": memory_type,
                "category": category,
                "last_accessed": datetime.now().isoformat(),
                "file_path": memory_path
            }
        }
        
        with open(memory_path, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, ensure_ascii=False, indent=2)
        
        self.embeddings_cache[memory_path] = embedding
        return memory_data
    
    async def search_relevant_memories(self, user_id: str, query: str, categories: List[str], limit: int = 5) -> List[Dict]:
        memories = []
        query_embedding = sentence_transformer.encode(query)
        
        for category in categories:
            category_path = os.path.join(self.memory_system.base_path, str(user_id), "long_term", category)
            if os.path.exists(category_path):
                memory_files = sorted(
                    [f for f in os.listdir(category_path) if f.endswith('.json')],
                    key=lambda x: os.path.getmtime(os.path.join(category_path, x)),
                    reverse=True
                )
                
                for memory_file in memory_files:
                    file_path = os.path.join(category_path, memory_file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        memory = json.load(f)
                        memories.append(memory)
        
        if memories:
            memory_chunks = [memories[i:i + self.chunk_size] for i in range(0, len(memories), self.chunk_size)]
            scored_memories = []
            
            for chunk in memory_chunks:
                chunk_prompt = f"""Analyze these memories and return only a JSON array of relevance scores (0-1):
                Query: {query}
                Memories: {[{"content": m["content"], "category": m["metadata"]["category"]} for m in chunk]}
                Return format example: [0.8, 0.5, 0.3]"""
                
                try:
                    response = model.generate_content(chunk_prompt)
                    response_text = response.text.strip()
                    
                    json_start = response_text.find('[')
                    json_end = response_text.rfind(']') + 1
                    
                    if json_start >= 0 and json_end > 0:
                        scores = json.loads(response_text[json_start:json_end])
                    else:
                        chunk_embeddings = [sentence_transformer.encode(m["content"]) for m in chunk]
                        scores = [float(cosine_similarity([query_embedding], [e])[0][0]) for e in chunk_embeddings]
                    
                    for memory, score in zip(chunk, scores):
                        memory["relevance_score"] = float(score)
                        scored_memories.append(memory)
                        
                except Exception as e:
                    logging.info(f"Chunk processing: {str(e)}, using fallback scoring")
                    chunk_embeddings = [sentence_transformer.encode(m["content"]) for m in chunk]
                    scores = [float(cosine_similarity([query_embedding], [e])[0][0]) for e in chunk_embeddings]
                    
                    for memory, score in zip(chunk, scores):
                        memory["relevance_score"] = score
                        scored_memories.append(memory)
            
            scored_memories.sort(
                key=lambda x: (x["relevance_score"], x["timestamp"]),
                reverse=True
            )
            
            return scored_memories[:limit]
        
        return []
    
    async def format_memories(self, memories: List[Dict]) -> str:
        formatted = []
        for memory in memories:
            formatted.append(
                f"[{memory['metadata']['category']}] "
                f"{memory['content']}\n"
                f"Context: {memory.get('response', '')}"
            )
        return "\n\n".join(formatted)

    async def optimize_user_memories(self, user_id: str):
        user_path = os.path.join(self.memory_system.base_path, str(user_id))
        memory_categories = ["conversations", "knowledge", "personal", "general_interaction"]
        retention_limit = 100
        
        for category in memory_categories:
            category_path = os.path.join(user_path, "long_term", category)
            if os.path.exists(category_path):
                memory_files = sorted(
                    [f for f in os.listdir(category_path) if f.endswith('.json')],
                    key=lambda x: os.path.getmtime(os.path.join(category_path, x))
                )
                
                if len(memory_files) > retention_limit:
                    for old_file in memory_files[:-retention_limit]:
                        file_path = os.path.join(category_path, old_file)
                        os.remove(file_path)
        
        logging.info(f"✨ Memory optimization completed for user {user_id}")



async def optimize_user_memories(self, user_id: str):
    user_path = os.path.join(self.memory_system.base_path, str(user_id))
    
    # Define memory categories and optimization parameters
    memory_categories = ["conversations", "knowledge", "personal", "general_interaction"]
    default_retention = 100
    priority_threshold = 0.7
    
    try:
        # Process each memory category
        for category in memory_categories:
            category_path = os.path.join(user_path, "long_term", category)
            if os.path.exists(category_path):
                memory_files = sorted(
                    [f for f in os.listdir(category_path) if f.endswith('.json')],
                    key=lambda x: os.path.getmtime(os.path.join(category_path, x))
                )
                
                # Optimize memory retention
                if len(memory_files) > default_retention:
                    # Keep high-priority memories
                    priority_memories = []
                    for memory_file in memory_files[:-default_retention]:
                        file_path = os.path.join(category_path, memory_file)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            memory_data = json.load(f)
                            if memory_data.get('importance', 0) >= priority_threshold:
                                priority_memories.append(memory_file)
                            else:
                                os.remove(file_path)
                    
                    # Update memory files list
                    memory_files = priority_memories + memory_files[-default_retention:]
        
        logging.info(f"✨ Memory optimization successful for user {user_id}")
        return True
        
    except Exception as e:
        logging.info(f"Completed basic memory maintenance: {str(e)}")
        return False




# Initialize enhanced memory system
MEMORY_BASE_PATH = os.path.dirname(os.path.abspath(__file__))
memory_system = EnhancedMemorySystem(MEMORY_BASE_PATH)
memory_ops = MemoryOperations(memory_system)




# Initialize web search handler

async def update_memory(user_id: str, message_content: str, bot_response: str):
    # Initialize memory vectors if not exists
    if not hasattr(memory_manager, f'user_{user_id}_index'):
        setattr(memory_manager, f'user_{user_id}_index', faiss.IndexFlatL2(768))
        setattr(memory_manager, f'user_{user_id}_memories', [])

    user_index = getattr(memory_manager, f'user_{user_id}_index')
    user_memories = getattr(memory_manager, f'user_{user_id}_memories')

    # Generate embeddings for message and response
    message_embedding = sentence_transformer.encode([message_content])[0]
    response_embedding = sentence_transformer.encode([bot_response])[0]
    
    # Create memory entry
    memory_entry = {
        "user_message": message_content,
        "bot_response": bot_response,
        "message_embedding": message_embedding,
        "response_embedding": response_embedding,
        "timestamp": datetime.now().isoformat(),
        "topics": extract_topics(message_content)
    }

    # Add to FAISS index
    user_index.add(np.array([message_embedding]))
    user_memories.append(memory_entry)

    # Update user profile
    if user_id not in user_profiles:
        user_profiles[user_id] = {
            "preferences": {"communication_style": "friendly_enthusiastic", "topics_of_interest": []},
            "demographics": {"age": None, "location": None},
            "personality": {"humor": 0.8, "kindness": 0.9, "curiosity": 0.9},
            "interests": [],
            "query": "",
            "interaction_history": []
        }

    # Extract and update topics
    topics = extract_topics(message_content)
    user_profiles[user_id]["interests"].extend(list(topics))

    # Check importance for long-term storage
    importance_score = calculate_importance(message_content)
    if importance_score > LONG_TERM_THRESHOLD:
        # Add to long-term memory
        memory_manager.add_memory(message_content, is_long_term=True)
        
        # Store in knowledge graph
        knowledge_graph.add_node("memory", data={
            "user_id": user_id,
            "content": message_content,
            "response": bot_response,
            "topics": list(topics),
            "timestamp": datetime.now().isoformat(),
            "embedding": message_embedding.tolist()
        })
        knowledge_graph.save_to_file(KNOWLEDGE_GRAPH_FILE)

        # Store in database
        async with connect(DB_FILE) as db:
            await db.execute('''
                INSERT INTO chat_history
                (user_id, message, timestamp, topics, embedding)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, message_content, datetime.now().isoformat(), 
                  json.dumps(list(topics)), json.dumps(message_embedding.tolist())))
            await db.commit()

    # Maintain memory window size
    if len(user_memories) > MEMORY_WINDOW:
        user_memories.pop(0)
        new_index = faiss.IndexFlatL2(768)
        new_index.add(np.array([mem["message_embedding"] for mem in user_memories]))
        setattr(memory_manager, f'user_{user_id}_index', new_index)

    save_user_profiles()

class MemoryManager:
    def __init__(self, max_memories: int = 1000):
        self.short_term_memories = []
        self.long_term_memories = []
        self.image_memories = {}
        self.max_memories = max_memories
        self.total_memories_added = 0
        self.last_cleanup_time = datetime.now()
        self.memory_categories = defaultdict(list)
        self.embedding_index = faiss.IndexFlatL2(768)
        self.memory_vectors = []

    async def add_memory(self, text: str, image_url: str = None, is_long_term: bool = False, importance_score: float = 0.0):
        try:
            # Enhanced memory analysis prompt
            memory_prompt = f"""Analyze this content and return structured JSON:
            Text: {text}
            Image URL: {image_url if image_url else 'None'}
            
            Return format:
            {{
                "summary": "brief summary",
                "topics": ["topic1", "topic2"],
                "entities": ["entity1", "entity2"],
                "category": "content category",
                "importance": 0.0-1.0 score
            }}"""

            # Generate memory analysis with explicit JSON request
            analysis = model.generate_content(memory_prompt)
            memory_data = json.loads(analysis.text.strip())

            # Create enhanced memory entry
            memory_entry = {
                'text': text,
                'image_url': image_url,
                'summary': memory_data.get('summary', text[:50] + '...'),
                'topics': memory_data.get('topics', list(extract_topics(text))),
                'category': memory_data.get('category', 'general'),
                'importance': max(importance_score, memory_data.get('importance', 0.5)),
                'timestamp': datetime.now().isoformat(),
                'entities': memory_data.get('entities', []),
                'embedding': sentence_transformer.encode(text)
            }

            # Process image with enhanced analysis
            if image_url:
                image_prompt = f"""Analyze this image in detail and return JSON:
                Image URL: {image_url}
                
                Include:
                - Visual elements
                - Colors and composition
                - Notable features
                - Context clues"""
                
                image_analysis = model.generate_content(image_prompt)
                memory_entry['image_analysis'] = json.loads(image_analysis.text.strip())

            # Store in appropriate memory with vector indexing
            if is_long_term or memory_entry['importance'] > 0.7:
                await self._store_long_term(memory_entry)
            else:
                await self._store_short_term(memory_entry)

            # Update vector indices
            self.memory_vectors.append(memory_entry['embedding'])
            self.embedding_index.add(np.array([memory_entry['embedding']]))
            self.total_memories_added += 1

            logging.info(f"✨ Memory added successfully: {memory_entry['summary'][:50]}...")

        except Exception as e:
            logging.error(f"Memory addition error: {e}")
            # Create basic memory entry on error
            basic_memory = {
                'text': text,
                'summary': text[:50] + '...',
                'topics': list(extract_topics(text)),
                'importance': importance_score,
                'timestamp': datetime.now().isoformat(),
                'embedding': sentence_transformer.encode(text)
            }
            await self._store_short_term(basic_memory)

    
    async def search_memories(self, query: str, k: int = 5) -> List[Dict]:
        try:
            # Check if we have memories to search
            if not self.memory_vectors or len(self.memory_vectors) == 0:
                return []

            # Generate query embedding
            query_vector = sentence_transformer.encode(query)
            memory_array = np.array(self.memory_vectors)
            
            # Perform vector search with valid k value
            k = min(k, len(memory_array))
            D, I = self.embedding_index.search(np.array([query_vector]), k)
            
            # Collect results safely
            results = []
            for idx in I[0]:
                if 0 <= idx < len(self.short_term_memories):
                    memory = self.short_term_memories[idx].copy()
                    memory['source'] = 'short_term'
                    memory['distance'] = float(D[0][len(results)])
                    results.append(memory)
                elif idx < len(self.short_term_memories) + len(self.long_term_memories):
                    long_term_idx = idx - len(self.short_term_memories)
                    if long_term_idx < len(self.long_term_memories):
                        memory = self.long_term_memories[long_term_idx].copy()
                        memory['source'] = 'long_term'
                        memory['distance'] = float(D[0][len(results)])
                        results.append(memory)

            # Enhance results with context
            if results:
                enhancement_prompt = {
                    "query": query,
                    "memories": results,
                    "request": {
                        "add": ["relevance", "context", "temporal"],
                        "format": "json"
                    }
                }

                enhanced = model.generate_content(json.dumps(enhancement_prompt))
                enhanced_data = json.loads(enhanced.text)

                # Sort by multiple criteria
                return sorted(
                    enhanced_data['results'],
                    key=lambda x: (
                        x.get('relevance', 0),
                        x.get('importance', 0),
                        -x.get('temporal_distance', 0)
                    ),
                    reverse=True
                )[:k]

            return results

        except Exception as e:
            logging.error(f"Memory search error: {e}")
            return []


    async def _store_long_term(self, memory_entry: Dict):
        try:
            # Optimize for long-term storage
            optimization_prompt = f"""Optimize memory for long-term storage:
            Memory: {memory_entry}
            
            Create:
            1. Compressed representation
            2. Key information extraction
            3. Connection mapping"""

            optimization = model.generate_content(optimization_prompt)
            optimized = json.loads(optimization.text)
            
            memory_entry.update(optimized)
            self.long_term_memories.append(memory_entry)
            
            # Update categories
            for tag in optimized.get('semantic_tags', []):
                self.memory_categories[tag].append(len(self.long_term_memories) - 1)
            
            await self._manage_long_term_capacity()

        except Exception as e:
            logging.error(f"Long-term storage error: {e}")

    async def _store_short_term(self, memory_entry: Dict):
        self.short_term_memories.append(memory_entry)
        await self._manage_short_term_capacity()

    async def _manage_short_term_capacity(self):
        if len(self.short_term_memories) > self.max_memories:
            # Consolidate memories
            recent_memories = self.short_term_memories[-20:]
            consolidation = model.generate_content(json.dumps({
                "memories": recent_memories,
                "task": "consolidate"
            }))
            
            consolidated = json.loads(consolidation.text)
            
            # Move important consolidated memories to long-term
            for group in consolidated.get('groups', []):
                if group['importance'] > 0.7:
                    await self._store_long_term(group)
            
            # Keep recent memories
            self.short_term_memories = self.short_term_memories[-50:]

    async def cleanup_routine(self):
        while True:
            try:
                current_time = datetime.now()
                
                # Analyze memories for cleanup
                cleanup_prompt = {
                    "memory_stats": self.get_stats(),
                    "task": "optimize_storage"
                }
                
                cleanup = model.generate_content(json.dumps(cleanup_prompt))
                actions = json.loads(cleanup.text)
                
                # Execute cleanup actions
                self.short_term_memories = [
                    mem for i, mem in enumerate(self.short_term_memories)
                    if i not in actions.get('remove_indices', [])
                ]
                
                # Consolidate memories
                for group in actions.get('consolidate', []):
                    await self._store_long_term(group)
                
                self.last_cleanup_time = current_time
                await asyncio.sleep(3600)

            except Exception as e:
                logging.error(f"Cleanup error: {e}")
                await asyncio.sleep(300)

    def get_stats(self) -> Dict:
        return {
            'total_memories': len(self.short_term_memories) + len(self.long_term_memories),
            'short_term_count': len(self.short_term_memories),
            'long_term_count': len(self.long_term_memories),
            'categories': list(self.memory_categories.keys()),
            'total_added': self.total_memories_added,
            'last_cleanup': self.last_cleanup_time.isoformat(),
            'image_memories': len(self.image_memories)
        }


# Initialize memory manager
memory_manager = MemoryManager()


def extract_topics(text: str) -> set:
    tokens = nltk.word_tokenize(text.lower())
    tagged = nltk.pos_tag(tokens)
    
    # Enhanced topic extraction with additional patterns
    topics = set()
    current_topic = []
    
    for i, (word, pos) in enumerate(tagged):
        if pos in ['NN', 'NNP', 'NNPS', 'NNS']:
            current_topic.append(word)
            # Check for compound topics (e.g., "artificial intelligence")
            if i > 0 and tagged[i-1][1] in ['JJ', 'NN', 'NNP']:
                current_topic.insert(0, tagged[i-1][0])
        else:
            if current_topic:
                topics.add(' '.join(current_topic))
                current_topic = []
    
    if current_topic:
        topics.add(' '.join(current_topic))
    
    return topics

def calculate_importance(text: str) -> float:
    # Dynamic importance analysis
    sentiment = sentiment_analyzer.polarity_scores(text)
    
    # Extract key features
    tokens = nltk.word_tokenize(text.lower())
    pos_tags = nltk.pos_tag(tokens)
    
    # Dynamic scoring factors
    importance_factors = {
        'named_entities': sum(1 for _, tag in pos_tags if tag.startswith('NNP')) * 0.15,
        'numeric_data': sum(1 for token in tokens if any(c.isdigit() for c in token)) * 0.1,
        'technical_terms': sum(1 for _, tag in pos_tags if tag in ['NN', 'NNS', 'JJ']) * 0.12,
        'action_verbs': sum(1 for _, tag in pos_tags if tag.startswith('VB')) * 0.08,
        'question_indicators': sum(1 for token in tokens if token in '?') * 0.1,
        'exclamation_emphasis': sum(1 for token in tokens if token in '!') * 0.05,
        'text_complexity': len(set(tokens)) / len(tokens) * 0.2,
        'sentence_structure': len(nltk.sent_tokenize(text)) * 0.1,
    }
    
    # Calculate base importance score
    base_score = sum(importance_factors.values())
    
    # Integrate sentiment impact
    sentiment_impact = (sentiment['compound'] + 1) / 2
    
    # Calculate final score with dynamic weighting
    final_score = (base_score * 0.7 + sentiment_impact * 0.3)
    
    # Normalize to 0-1 range
    return min(max(final_score, 0), 1)


async def handle_rate_limit(message: discord.Message):
    await message.channel.send("*Processing your request, please wait while I optimize my connection!*")
    
    proxy_batch_size = 50
    proxy_pool = await web_search_handler._get_proxy_pool()
    
    for i in range(0, len(proxy_pool), proxy_batch_size):
        proxy_batch = proxy_pool[i:i + proxy_batch_size]
        
        for proxy in proxy_batch:
            try:
                proxy_dict = {
                    'http': f'http://{proxy}',
                    'https': f'http://{proxy}'
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        'https://duckduckgo.com/',
                        proxy=proxy_dict['http'],
                        timeout=10,
                        headers=web_search_handler.headers
                    ) as response:
                        if response.status == 200:
                            web_search_handler.current_working_proxy = proxy_dict
                            web_search_handler.successful_requests = 0
                            return True
                            
            except Exception:
                continue
                
        await message.channel.send("*Still optimizing connection... Please wait!*")
    
    return False

# Initialize managers

memory_manager = MemoryManager()


@bot.event
async def on_message(message: discord.Message):
    try:
        if message.author == bot.user or not (bot.user.mentioned_in(message) or message.content.lower().startswith('puro')):
            return

        user_id = str(message.author.id)
        content = message.content.replace(f'<@{bot.user.id}>', '').replace('puro', '', 1).strip()
        
        user_memory_path = os.path.join(MEMORY_BASE_PATH, "user_memories", user_id)
        os.makedirs(user_memory_path, exist_ok=True)

        logging.info(f"\n🔵 Processing message from {message.author.name} ({user_id}): {content}")

        if message.attachments:
            for attachment in message.attachments:
                if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp']):
                    logging.info(f"🖼️ Processing image from {message.author.name}")
                    
                    image_memory = await memory_ops.store_memory(
                        user_id=user_id,
                        content=content,
                        memory_type="long_term",
                        category="visual_memories",
                        image_url=attachment.url
                    )
                    
                    await handle_image(message, {'user_id': user_id, 'content': content})
                    return

        if not content and not message.attachments:
            await message.channel.send("*Tail wags excitedly* What's on your mind? I'd love to chat!")
            return

        async with message.channel.typing():
            initial_response = await message.channel.send("*Tail wags excitedly* Let me search for information about that!")
            
            relevant_memories = await memory_ops.search_relevant_memories(
                user_id=user_id,
                query=content,
                categories=["conversations", "knowledge", "personal"],
                limit=5
            )
            
            memory_context = await memory_ops.format_memories(relevant_memories)

            web_context = None
            for attempt in range(3):
                try:
                    search_results = await web_search_handler.search(content)
                    if search_results:
                        web_context = await web_search_handler.gemini_search_and_summarize(content)
                        logging.info("🌐 Web search successful")
                        break
                except Exception as e:
                    logging.error(f"Search attempt {attempt + 1} failed: {str(e)}")
                    await asyncio.sleep(1)

            response_text, sentiment = await generate_response_with_gemini(
                prompt=content,
                relevant_history=memory_context,
                summarized_search=web_context,
                user_id=user_id,
                message=message,
                content=content
            )

            await initial_response.edit(content=response_text)
            
            importance_score = calculate_importance(content)
            memory_type = "long_term" if importance_score > LONG_TERM_THRESHOLD else "short_term"
            
            await memory_ops.store_memory(
                user_id=user_id,
                content=content,
                memory_type=memory_type,
                category="conversations",
                response=response_text,
                sentiment=sentiment,
                importance=importance_score
            )

            if user_id not in user_profiles:
                user_profiles[user_id] = {
                    "preferences": {"communication_style": "friendly_enthusiastic", "topics_of_interest": []},
                    "demographics": {"age": None, "location": None},
                    "context": deque(maxlen=CONTEXT_WINDOW_SIZE),
                    "personality": {"humor": 0.8, "kindness": 0.9, "curiosity": 0.9},
                    "interests": [],
                    "query": "",
                    "interaction_history": []
                }

            user_profiles[user_id]["context"].extend([
                {"role": "user", "content": content, "timestamp": datetime.now().isoformat()},
                {"role": "assistant", "content": response_text, "timestamp": datetime.now().isoformat()}
            ])

            await save_chat_history(
                user_id=user_id,
                message=content,
                user_name=message.author.name,
                bot_id=bot.user.id,
                bot_name=bot.user.name,
                importance_score=importance_score
            )
            
            await memory_ops.optimize_user_memories(user_id)
            
            save_user_profiles()
            logging.info("✅ Memory and profile updated successfully")

    except Exception as e:
        logging.error(f"❌ Error in message handling: {str(e)}", exc_info=True)
        await message.channel.send("*Tail wags enthusiastically* I'd love to help! What would you like to explore together?")



# --- Metrics ---
class ResponseTimeHistogram:
    def __init__(self):
        self.histogram = []

    def time(self):
        start_time = time.time()
        return self.HistogramContextManager(start_time, self.histogram)

    class HistogramContextManager:
        def __init__(self, start_time, histogram):
            self.start_time = start_time
            self.histogram = histogram

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed_time = time.time() - self.start_time
            self.histogram.append(elapsed_time)

class ResponseTimeSummary:
    def __init__(self):
        self.summary = []

    def time(self):
        start_time = time.time()
        return self.SummaryContextManager(start_time, self.summary)

    class SummaryContextManager:
        def __init__(self, start_time, summary):
            self.start_time = start_time
            self.summary = summary

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed_time = time.time() - self.start_time
            self.summary.append(elapsed_time)

# Initialize metrics
response_time_histogram = ResponseTimeHistogram()
response_time_summary = ResponseTimeSummary()

async def run_bot_with_reconnect():
    while True:
        try:
            await bot.start(discord_token)
        except aiohttp.client_exceptions.ClientConnectorDNSError:
            logging.warning("🔄 DNS resolution failed, retrying connection in 5 seconds...")
            await asyncio.sleep(5)
        except discord.errors.ConnectionClosed:
            logging.warning("🔄 Connection closed, attempting to reconnect in 5 seconds...")
            await asyncio.sleep(5)
        except Exception as e:
            logging.error(f"❌ Unexpected error: {str(e)}")
            logging.warning("🔄 Attempting reconnection in 10 seconds...")
            await asyncio.sleep(10)
        finally:
            if not bot.is_closed():
                await bot.close()
            logging.info("🔄 Restarting bot...")


# Replace bot.run(discord_token) with:
if __name__ == "__main__":
    while True:
        try:
            asyncio.run(run_bot_with_reconnect())
        except KeyboardInterrupt:
            logging.info("👋 Bot shutdown requested by user")
            break
        except Exception as e:
            logging.error(f"🔄 Restarting due to: {str(e)}")
            time.sleep(5)
