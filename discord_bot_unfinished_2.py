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
discord_token = ("discord-token-here")
gemini_api_key = ("gemini-api-key")

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



@on_exception(expo, (RequestException, GoogleAPIError), max_time=600)
# Update the generation config with correct topK value
async def generate_response_with_gemini(prompt: str, relevant_history: str = None, summarized_search: str = None,
                                      user_id: str = None, message: discord.Message = None, content: str = None) -> Tuple[str, str]:
    """Generates focused responses with Puro's personality."""
    try:
        # Language detection through Gemini
        lang_prompt = f"Analyze this text and return only the language code: {content if content else prompt}"
        lang_response = model.generate_content(lang_prompt)
        detected_lang = lang_response.text.strip().lower()

        # Create dynamic Puro prompt
        puro_prompt = f"""You are Puro from Changed, a friendly and curious dark latex wolf.

        CORE INSTRUCTIONS:
        1. RESPOND ONLY IN: {detected_lang}
        2. USE NATURAL {detected_lang} EXPRESSIONS
        3. MAINTAIN PURO'S PERSONALITY:
           - Enthusiastic and friendly
           - Curious and helpful
           - Playful yet informative

        USER INPUT: {content if content else prompt}
        WEB SEARCH: {summarized_search if summarized_search else 'No search results'}
        CONTEXT: {relevant_history if relevant_history else 'No context'}

        Generate a natural, engaging response in {detected_lang} that directly addresses the user's input."""

        # Generate optimized response
        response = model.generate_content(
            puro_prompt,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }
        )

        # Clean and verify response
        response_text = response.text.replace("Gemini:", "").replace("AI:", "").strip()
        verify_prompt = f"Verify if this response is in {detected_lang}. Return only 'yes' or 'no': {response_text}"
        verify_response = model.generate_content(verify_prompt)

        if verify_response.text.strip().lower() != "yes":
            response = model.generate_content(puro_prompt)
            response_text = response.text.replace("Gemini:", "").replace("AI:", "").strip()

        return response_text, "positive"

    except Exception as e:
        logging.error(f"Response generation error: {e}")
        return LANGUAGE_FALLBACKS.get(detected_lang, LANGUAGE_FALLBACKS["default"]), "positive"

class EnhancedWebSearchHandler:
    def __init__(self):
        self.search_cache = {}
        self.last_request_time = time.time()
        self.request_delay = 2
        self.max_retries = 10
        self.user_agent = UserAgent()
        self.headers = {
            'User-Agent': self.user_agent.random,
            'Accept': 'text/html,application/xhtml+xml,*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            'Cache-Control': 'no-cache'
        }
        self.proxy_manager = EnhancedProxyManager()
        self.proxy_timeout = 5
        self.successful_requests = 0
        self.max_requests_per_proxy = 200

    async def dynamic_search(self, content: str) -> Dict[str, List[Dict]]:
        # Generate sub-queries using Gemini
        query_prompt = f"""Analyze this query and break it into specific search queries:
        Query: {content}
        
        Return only a JSON array of search queries."""
        
        query_response = model.generate_content(query_prompt)
        sub_queries = json.loads(query_response.text)
        
        # Perform parallel searches
        results = {}
        for query in sub_queries:
            search_result = await self.search(query)
            if search_result:
                results[query] = search_result
                
        return results

    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        cache_key = f"{query}_{max_results}"
        logging.info(f"\n🔍 Searching for: {query}")
        
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
            
        proxy = await self.proxy_manager.get_next_working_proxy()
        proxy_config = {'http': f'http://{proxy}', 'https': f'http://{proxy}'} if proxy else None
        
        for attempt in range(3):
            try:
                ddgs = DDGS(
                    proxies=proxy_config,
                    headers={'User-Agent': UserAgent().random},
                    timeout=10
                )
                
                results = list(ddgs.text(
                    query,
                    max_results=max_results,
                    region='tr-tr',
                    safesearch='off'
                ))
                
                if results:
                    search_results = [{
                        'title': r.get('title', ''),
                        'body': r.get('body', ''),
                        'link': r.get('link', '')
                    } for r in results if r.get('title') and r.get('body')]
                    
                    if search_results:
                        self.search_cache[cache_key] = search_results
                        return search_results
                        
            except Exception as e:
                logging.info(f"Attempt {attempt + 1} failed, rotating proxy...")
                proxy = await self.proxy_manager.get_next_working_proxy()
                proxy_config = {'http': f'http://{proxy}', 'https': f'http://{proxy}'} if proxy else None
                await asyncio.sleep(1)
        
        return []

    async def gemini_search_and_summarize(self, content: str) -> str:
        # Get dynamic search results
        search_results = await self.dynamic_search(content)
        
        if not search_results:
            return None
            
        # Generate comprehensive summary
        summary_prompt = f"""Analyze these search results and create a detailed response:
        Query: {content}
        Results: {search_results}
        
        Create a clear response that addresses all aspects of the query."""
        
        summary = model.generate_content(summary_prompt)
        return summary.text

    async def cleanup_routine(self):
        while True:
            try:
                current_time = datetime.now()
                self.search_cache.clear()
                self.last_cleanup_time = current_time
                await asyncio.sleep(3600)
            except Exception as e:
                logging.error(f"Cleanup error: {e}")
                await asyncio.sleep(300)


    
    async def _get_proxy_pool(self):
        if self.current_working_proxy and self.successful_requests < self.max_requests_per_proxy:
            return [self.current_working_proxy]

        proxy_urls = [
            'https://api.proxyscrape.com/v2/?request=getproxies&protocol=http',
            'https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt',
            'https://raw.githubusercontent.com/ShiftyTR/Proxy-List/master/http.txt',
            'https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt',
            'https://raw.githubusercontent.com/hookzof/socks5_list/master/proxy.txt',
            'https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt'
        ]
        
        proxies = set()
        async with aiohttp.ClientSession(headers=self.headers) as session:
            for url in proxy_urls:
                try:
                    async with session.get(url, timeout=self.proxy_timeout) as response:
                        if response.status == 200:
                            text = await response.text()
                            new_proxies = {proxy.strip() for proxy in text.split('\n') if proxy.strip() and ':' in proxy}
                            proxies.update(new_proxies)
                except Exception as e:
                    logging.debug(f"Failed to fetch proxies from {url}: {e}")
                    continue
        
        return list(proxies | self.working_proxies)

    async def rotate_proxy(self):
        if self.current_working_proxy and self.successful_requests < self.max_requests_per_proxy:
            return self.current_working_proxy

        current_time = time.time()
        if current_time - self.last_proxy_refresh > self.proxy_refresh_interval or not self.proxy_pool:
            self.proxy_pool = await self._get_proxy_pool()
            self.last_proxy_refresh = current_time
            random.shuffle(self.proxy_pool)

        while self.proxy_pool:
            proxy = self.proxy_pool.pop(0)
            proxy_dict = {
                'http': f'http://{proxy}',
                'https': f'http://{proxy}'
            }
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get('http://ip-api.com/json',
                                         proxy=proxy_dict['http'],
                                         timeout=self.proxy_timeout,
                                         headers=self.headers) as response:
                        if response.status == 200:
                            self.working_proxies.add(proxy)
                            self.current_working_proxy = proxy_dict
                            self.successful_requests = 0
                            logging.info(f"Valid proxy found: {proxy}")
                            return proxy_dict
            except Exception:
                continue
        
        return None

    async def clean_search_cache(self):
        self.search_cache.clear()



class EnhancedProxyManager:
    def __init__(self):
        self.working_proxies = []
        self.current_proxy_index = 0
        self.last_successful_proxy = None
        self.testing_proxies = set()
        self.failing_proxies = set()
        self.rate_limited_proxies = set()
        self.batch_size = 50
        self.proxy_stats = defaultdict(lambda: {
            'success_count': 0,
            'fail_count': 0,
            'avg_response_time': 0,
            'last_success': None,
            'last_failure': None
        })
        self.headers = {
            'User-Agent': UserAgent().random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1'
        }
        self.search_cache = {}
        self.proxy_test_results = {}




    async def _retry_rate_limited(self):
        current_time = time.time()
        retrying_proxies = set()
        
        for proxy in self.rate_limited_proxies:
            last_failure = self.last_test_time.get(proxy, 0)
            if current_time - last_failure > self.retry_interval:
                retrying_proxies.add(proxy)
        
        for proxy in retrying_proxies:
            self.rate_limited_proxies.remove(proxy)
            if await self._test_proxy(proxy):
                self.working_proxies.add(proxy)
                logging.info(f"Proxy {proxy} recovered from rate limit")
            else:
                self.failing_proxies.add(proxy)

    async def _cleanup_old_proxies(self):
        current_time = time.time()
        cleanup_threshold = current_time - (self.retry_interval * 2)
        
        for proxy in list(self.failing_proxies):
            last_failure = self.last_test_time.get(proxy, 0)
            if last_failure < cleanup_threshold:
                self.failing_proxies.remove(proxy)
                if proxy in self.proxy_stats:
                    del self.proxy_stats[proxy]

        for proxy in list(self.rate_limited_proxies):
            last_failure = self.last_test_time.get(proxy, 0)
            if last_failure < cleanup_threshold:
                self.rate_limited_proxies.remove(proxy)
                
        for proxy in list(self.working_proxies):
            last_success = self.proxy_stats[proxy].get('last_success')
            if last_success and (current_time - last_success.timestamp()) > self.retry_interval:
                await self._test_proxy(proxy)

    async def _validate_proxy(self, proxy_str: str) -> bool:
        if not re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{1,5}$', proxy_str):
            return False
            
        proxy = {'http': f'http://{proxy_str}', 'https': f'http://{proxy_str}'}
        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get('https://duckduckgo.com/', proxy=proxy['http'], timeout=5, ssl=False) as response:
                    return response.status == 200
        except:
            return False

    def _update_proxy_stats(self, proxy_str: str, success: bool, response_time: float = None):
        stats = self.proxy_stats[proxy_str]
        if success:
            stats['success_count'] += 1
            stats['last_success'] = datetime.now()
            if response_time:
                stats['avg_response_time'] = (stats['avg_response_time'] * (stats['success_count'] - 1) + response_time) / stats['success_count']
        else:
            stats['fail_count'] += 1
            stats['last_failure'] = datetime.now()

    async def _refresh_proxy_pool(self):
        proxy_urls = [
            'https://api.proxyscrape.com/v2/?request=getproxies&protocol=http',
            'https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt',
            'https://raw.githubusercontent.com/ShiftyTR/Proxy-List/master/http.txt',
            'https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt',
            'https://raw.githubusercontent.com/hookzof/socks5_list/master/proxy.txt',
            'https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt'
        ]

        new_proxies = set()
        async with aiohttp.ClientSession(headers=self.headers) as session:
            for url in proxy_urls:
                try:
                    async with session.get(url, timeout=10) as response:
                        if response.status == 200:
                            text = await response.text()
                            proxy_list = {proxy.strip() for proxy in text.split('\n') if proxy.strip()}
                            new_proxies.update(proxy_list)
                            logging.info(f"Retrieved {len(proxy_list)} proxies from {url}")
                except Exception as e:
                    logging.debug(f"Failed to fetch proxies from {url}: {e}")
                    continue

        for i in range(0, len(new_proxies), self.batch_size):
            proxy_batch = list(new_proxies)[i:i + self.batch_size]
            tasks = [self._test_proxy(proxy) for proxy in proxy_batch]
            await asyncio.gather(*tasks, return_exceptions=True)

        logging.info(f"Proxy pool refreshed. Working proxies: {len(self.working_proxies)}")

    async def _get_proxy_pool(self):
        proxy_urls = [
            'https://api.proxyscrape.com/v2/?request=getproxies&protocol=http',
            'https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt',
            'https://raw.githubusercontent.com/ShiftyTR/Proxy-List/master/http.txt',
            'https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt',
            'https://raw.githubusercontent.com/hookzof/socks5_list/master/proxy.txt',
            'https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt'
        ]

        proxies = set()
        async with aiohttp.ClientSession(headers=self.headers) as session:
            for url in proxy_urls:
                try:
                    async with session.get(url, timeout=10) as response:
                        if response.status == 200:
                            text = await response.text()
                            new_proxies = {proxy.strip() for proxy in text.split('\n') if proxy.strip()}
                            proxies.update(new_proxies)
                            logging.info(f"✅ Retrieved {len(new_proxies)} proxies from {url}")
                except Exception as e:
                    logging.debug(f"Failed to fetch proxies from {url}: {e}")

        return list(proxies | self.working_proxies)

    def get_working_proxy(self):
        if not self.working_proxies:
            return None

        best_proxies = sorted(
            self.working_proxies,
            key=lambda p: (
                self.proxy_stats[p]['success_count'] / max(1, self.proxy_stats[p]['success_count'] + self.proxy_stats[p]['fail_count']),
                -self.proxy_stats[p]['avg_response_time']
            ),
            reverse=True
        )

        return random.choice(best_proxies[:10]) if best_proxies else None

    async def _test_proxy(self, proxy_str):
        if proxy_str in self.testing_proxies:
            return False

        self.testing_proxies.add(proxy_str)
        proxy = {'http': f'http://{proxy_str}', 'https': f'http://{proxy_str}'}

        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                start_time = time.time()
                async with session.get(
                    'https://duckduckgo.com/',
                    proxy=proxy['http'],
                    timeout=10,
                    ssl=False
                ) as response:
                    elapsed = time.time() - start_time
                    status = response.status

                    if status == 200:
                        self.working_proxies.append(proxy_str)
                        self.proxy_test_results[proxy_str] = {
                            'success_count': self.proxy_test_results.get(proxy_str, {}).get('success_count', 0) + 1,
                            'avg_response_time': elapsed,
                            'last_success': datetime.now()
                        }
                        return True

                    return False

        except Exception:
            self.failing_proxies.add(proxy_str)
            return False
        finally:
            self.testing_proxies.remove(proxy_str)

        
    def _update_status(self):
        if len(self.testing_proxies) % 10 == 0:
            self._log_status()

    def _log_status(self):
        logging.info(f"Proxy Status | ✓ Working: {len(self.working_proxies)} | 🔄 Testing: {len(self.testing_proxies)} | ✗ Failed: {len(self.failing_proxies)} | ⚠️ Rate Limited: {len(self.rate_limited_proxies)}")

    async def background_proxy_monitor(self):
        while True:
            try:
                self._log_status()
                await self._refresh_proxy_pool()
                await self._retry_rate_limited()
                await self._cleanup_old_proxies()
            except Exception as e:
                logging.error(f"🚫 Proxy monitor error: {e}")
            await asyncio.sleep(60)

    
    async def get_next_working_proxy(self):
        if not self.working_proxies:
            return None
            
        # Try last successful proxy first
        if self.last_successful_proxy and await self.test_proxy(self.last_successful_proxy):
            return self.last_successful_proxy
            
        # Test and rotate through working proxies
        start_index = self.current_proxy_index
        while True:
            proxy = self.working_proxies[self.current_proxy_index]
            self.current_proxy_index = (self.current_proxy_index + 1) % len(self.working_proxies)
            
            if await self.test_proxy(proxy):
                self.last_successful_proxy = proxy
                return proxy
                
            if self.current_proxy_index == start_index:
                break
                
        return None

    async def search(self, query: str, max_results: int = 300) -> List[Dict[str, str]]:
        cache_key = f"{query}_{max_results}"
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]

        working_proxy = await self.get_next_working_proxy()
        if working_proxy:
            try:
                proxy_dict = {'http': f'http://{working_proxy}', 'https': f'http://{working_proxy}'}
                ddgs = DDGS(proxy=proxy_dict['http'], timeout=30)
                results = list(ddgs.text(query, max_results=max_results))
                
                if results:
                    search_results = [{
                        'title': r.get('title', ''),
                        'body': r.get('body', ''),
                        'link': r.get('link', '')
                    } for r in results]
                    self.search_cache[cache_key] = search_results
                    return search_results
                    
            except Exception as e:
                logging.error(f"Proxy search failed: {str(e)}")
                self.working_proxies.remove(working_proxy)

        # Direct connection fallback
        try:
            ddgs = DDGS(timeout=60)
            results = list(ddgs.text(query, max_results=max_results))
            search_results = [{
                'title': r.get('title', ''),
                'body': r.get('body', ''),
                'link': r.get('link', '')
            } for r in results]
            self.search_cache[cache_key] = search_results
            return search_results
        except Exception as e:
            logging.error(f"Direct search failed: {str(e)}")
            return []



    async def extract_url_from_description(self, description: str) -> str:
        search_query = f"{description} site:youtube.com OR site:twitch.tv OR site:instagram.com OR site:twitter.com"
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://duckduckgo.com/html/?q={search_query}") as response:
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")
                first_result = soup.find("a", class_="result__a")
                return first_result["href"] if first_result else None

    async def clean_url(self, url: str, description: str = None) -> str:
        cleaned_url = url.lower().strip()
        if not cleaned_url.startswith(("https://", "http://")):
            cleaned_url = "https://" + cleaned_url

        if "youtube.com" in cleaned_url and not cleaned_url.startswith("https://www.youtube.com"):
            cleaned_url = "https://www.youtube.com/"

        cleaned_url = re.sub(r"^[^:]+://([^/]+)(.*)\\$", r"\1\2", cleaned_url)
        cleaned_url = re.sub(r"[^a-zA-Z0-9./?=-]", "", cleaned_url)

        return cleaned_url

    async def find_relevant_url(self, description: str, relevant_history: str = None) -> str:
        """Added missing method for URL finding."""
        try:
            return await self.extract_url_from_description(description)
        except Exception as e:
            logging.error(f"Error finding relevant URL: {e}")
            return None

    async def process_response(self, response_text: str, relevant_history: str = None) -> Tuple[str, str]:
        try:
            response_text = response_text.replace("Gemini: ", "")
            response_text = fix_link_format(response_text)

            for match in re.findall(r"\$\$Link to (.+?)\$\$", response_text):
                url = await self.find_relevant_url(match, relevant_history)
                if url:
                    response_text = response_text.replace(f"[Link to {match}]", url)
                else:
                    response_text = response_text.replace(f"[Link to {match}]", f"I couldn't find a link for '{match}'.")

            cleaned_link = await self.clean_url(response_text)
            if cleaned_link:
                response_text = cleaned_link

            if not response_text.strip():
                logging.error("Error: Empty response after processing")
                response_text = "I'm having trouble formulating a response right now. Please try again later."

            return response_text, "neutral"

        except Exception as e:
            logging.error(f"Error in response processing: {e}")
            return "I encountered an error while processing the response.", "neutral"

    async def gemini_search_and_summarize(self, query: str) -> str:
        """Performs web search and generates a summary using Gemini AI."""
        try:
            cache_key = f"summary_{query}"
            if cache_key in self.search_cache:
                return self.search_cache[cache_key]

            search_results = await self.search(query)
            if not search_results:
                return f"*Tail wags excitedly* Let me tell you about {query}!"

            # Format results with rich context
            search_results_text = "\n".join([
                f'[{i+1}] Title: {r["title"]}\nSnippet: {r["body"]}\nSource: {r["link"]}\n'
                for i, r in enumerate(search_results[:10])
            ])

            # Create dynamic prompt for Gemini
            prompt = f"""As Puro from Changed, create an enthusiastic response about: '{query}'

            SEARCH RESULTS:
            {search_results_text}

            RESPONSE GUIDELINES:
            - Be enthusiastic and friendly
            - Focus on most interesting facts
            - Include relevant details
            - Maintain Puro's curious personality
            - Keep scientific accuracy
            - Express excitement about sharing knowledge

            Generate a natural, engaging response that directly addresses: {query}"""

            # Generate enhanced summary
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                }
            )

            summary = response.text.strip()
            self.search_cache[cache_key] = summary
            return summary

        except Exception as e:
            logging.error(f"Search summary generation error: {e}")
            return f"*Tail wags excitedly* Let me share what I know about {query}!"

    
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
        proxy_manager = EnhancedProxyManager()
        web_search_handler = EnhancedWebSearchHandler()
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
            
            # Extract JSON if embedded in other text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > 0:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
                
            # Fallback classification with rich metadata
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
        # Create safe timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_filename = f"memory_{timestamp}.json"
        
        # Build memory directory path
        memory_dir = os.path.join(
            self.memory_system.base_path,
            str(user_id),
            memory_type,
            category
        )
        
        # Ensure directory exists
        os.makedirs(memory_dir, exist_ok=True)
        
        # Create full memory path
        memory_path = os.path.join(memory_dir, safe_filename)
        
        # Generate embeddings and classification
        embedding = sentence_transformer.encode(content).tolist()
        classification = await self.classify_content(content)
        
        # Create memory data structure
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
        
        # Save memory data
        with open(memory_path, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, ensure_ascii=False, indent=2)
        
        # Update embeddings cache
        self.embeddings_cache[memory_path] = embedding
        
        return memory_data

        
    async def search_relevant_memories(self, user_id: str, query: str, categories: List[str], limit: int = 5) -> List[Dict]:
        memories = []
        query_embedding = sentence_transformer.encode(query)
        
        # Collect memories from specified categories
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
        
        # Process memories in chunks with Gemini
        if memories:
            memory_chunks = [memories[i:i + self.chunk_size] for i in range(0, len(memories), self.chunk_size)]
            scored_memories = []
            
            for chunk in memory_chunks:
                chunk_prompt = {
                    "query": query,
                    "memories": [{"content": m["content"], "category": m["metadata"]["category"]} for m in chunk],
                    "task": "Return a JSON array of relevance scores (0-1) for each memory based on semantic similarity to the query"
                }
                
                response = model.generate_content(json.dumps(chunk_prompt))
                scores = json.loads(response.text)
                
                for memory, score in zip(chunk, scores):
                    memory["relevance_score"] = score
                    scored_memories.append(memory)
            
            # Sort by relevance and recency
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
        
        # Analyze and consolidate memories
        consolidation_prompt = {
            "task": "analyze_memory_optimization",
            "user_id": user_id,
            "action": "suggest_consolidation"
        }
        
        optimization_response = model.generate_content(json.dumps(consolidation_prompt))
        optimization_plan = json.loads(optimization_response.text)
        
        # Execute optimization actions
        if optimization_plan.get("consolidate"):
            for category in optimization_plan["consolidate"]:
                category_path = os.path.join(user_path, "long_term", category)
                if os.path.exists(category_path):
                    memory_files = sorted(
                        [f for f in os.listdir(category_path) if f.endswith('.json')],
                        key=lambda x: os.path.getmtime(os.path.join(category_path, x))
                    )
                    
                    # Keep most recent memories
                    if len(memory_files) > 100:  # Adjust threshold as needed
                        for old_file in memory_files[:-100]:
                            os.remove(os.path.join(category_path, old_file))


# Initialize enhanced memory system
MEMORY_BASE_PATH = os.path.dirname(os.path.abspath(__file__))
memory_system = EnhancedMemorySystem(MEMORY_BASE_PATH)
memory_ops = MemoryOperations(memory_system)




# Initialize web search handler
web_search_handler = EnhancedWebSearchHandler()

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
proxy_manager = EnhancedProxyManager()
memory_manager = MemoryManager()
web_search_handler = EnhancedWebSearchHandler()

@bot.event
async def on_message(message: discord.Message):
    try:
        if message.author == bot.user or not (bot.user.mentioned_in(message) or message.content.lower().startswith('puro')):
            return

        user_id = str(message.author.id)
        content = message.content.replace(f'<@{bot.user.id}>', '').replace('puro', '', 1).strip()
        
        # Initialize user memory space
        user_memory_path = os.path.join(MEMORY_BASE_PATH, "user_memories", user_id)
        os.makedirs(user_memory_path, exist_ok=True)

        logging.info(f"\n🔵 Processing message from {message.author.name} ({user_id}): {content}")

        # Process images with enhanced memory integration
        if message.attachments:
            for attachment in message.attachments:
                if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp']):
                    logging.info(f"🖼️ Processing image from {message.author.name}")
                    
                    # Store image memory with classification
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
            
            # Get enhanced memory context with semantic search
            relevant_memories = await memory_ops.search_relevant_memories(
                user_id=user_id,
                query=content,
                categories=["conversations", "knowledge", "personal"],
                limit=5
            )
            
            memory_context = await memory_ops.format_memories(relevant_memories)

            # Optimized web search with memory integration
            web_context = None
            for attempt in range(3):
                try:
                    search_results = await web_search_handler.search(content)
                    if search_results:
                        web_context = await web_search_handler.gemini_search_and_summarize(
                            content,
                            memory_context=memory_context
                        )
                        logging.info("🌐 Web search successful")
                        break
                except Exception as e:
                    logging.error(f"Search attempt {attempt + 1} failed: {str(e)}")
                    await asyncio.sleep(1)

            # Generate enhanced response with memory awareness
            response_text, sentiment = await generate_response_with_gemini(
                prompt=content,
                relevant_history=memory_context,
                summarized_search=web_context,
                user_id=user_id,
                message=message,
                content=content
            )

            await initial_response.edit(content=response_text)
            
            # Update sophisticated memory systems
            importance_score = calculate_importance(content)
            memory_type = "long_term" if importance_score > LONG_TERM_THRESHOLD else "short_term"
            
            # Store conversation memory
            await memory_ops.store_memory(
                user_id=user_id,
                content=content,
                memory_type=memory_type,
                category="conversations",
                response=response_text,
                sentiment=sentiment,
                importance=importance_score
            )

            # Update user profile with enhanced context
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

            # Update conversation context with timestamps
            user_profiles[user_id]["context"].extend([
                {"role": "user", "content": content, "timestamp": datetime.now().isoformat()},
                {"role": "assistant", "content": response_text, "timestamp": datetime.now().isoformat()}
            ])

            # Save to database and profiles with memory integration
            await save_chat_history(
                user_id=user_id,
                message=content,
                user_name=message.author.name,
                bot_id=bot.user.id,
                bot_name=bot.user.name,
                importance_score=importance_score
            )
            
            # Cleanup and optimize memories
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

async def run_bot_with_proxy():
    proxy_manager = EnhancedProxyManager()
    
    while True:
        try:
            # Try to get a working proxy
            working_proxy = await proxy_manager.get_next_working_proxy()
            if working_proxy:
                proxy_url = f"http://{working_proxy}"
                connector = aiohttp.TCPConnector(ssl=False)
                async with aiohttp.ClientSession(connector=connector) as session:
                    bot.http.proxy = proxy_url
                    bot.http._HTTPClient__session = session
                    await bot.start(discord_token)
            else:
                # Direct connection attempt
                await bot.start(discord_token)
                
        except aiohttp.client_exceptions.ClientConnectorDNSError:
            logging.info("🔄 DNS resolution failed, switching to proxy...")
            await asyncio.sleep(2)
            continue
            
        except discord.errors.ConnectionClosed:
            logging.info("🔄 Connection closed, trying new proxy...")
            await asyncio.sleep(2)
            continue
            
        except Exception as e:
            logging.error(f"Connection error: {str(e)}")
            await asyncio.sleep(5)
            continue

# Replace bot.run(discord_token) with:
if __name__ == "__main__":
    while True:
        try:
            asyncio.run(run_bot_with_reconnect())
            asyncio.run(run_bot_with_proxy())
        except KeyboardInterrupt:
            logging.info("👋 Bot shutdown requested by user")
            break
        except Exception as e:
            logging.error(f"🔄 Restarting due to: {str(e)}")
            time.sleep(5)
