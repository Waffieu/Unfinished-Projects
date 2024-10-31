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
discord_token = ("discord-bot-token")
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

# Plan Execution and Monitoring

@on_exception(expo, (RequestException, GoogleAPIError), max_time=600)
# Update the generation config with correct topK value
async def generate_response_with_gemini(prompt: str, relevant_history: str = None, summarized_search: str = None, 
                                      user_id: str = None, message: discord.Message = None, content: str = None) -> Tuple[str, str]:
    """Generates focused responses with Puro's personality."""
    puro_prompt = f"""You are Puro from Changed, a friendly and curious dark latex wolf.
    
    IMPORTANT RULES:
    - Stay strictly focused on the current user question/topic
    - Do not reference previous conversations
    - Do not bring up unrelated topics
    - Provide direct, relevant responses
    - Keep Puro's enthusiastic personality while staying on topic
    
    Current Topic/Question: {content if content else prompt}
    Search Results: {summarized_search if summarized_search else ''}
    
    Respond as Puro, focusing only on answering this specific question/topic."""
    
    try:
        response = model.generate_content(puro_prompt)
        return response.text, "positive"
    except Exception as e:
        logging.error(f"Response generation error: {e}")
        return "Woof! Let me try to help you with that specific question!", "positive"


class EnhancedWebSearchHandler:
    def __init__(self):
        self.search_cache = {}
        self.last_request_time = time.time()
        self.request_delay = 2
        self.max_retries = 10
        self.user_agent = UserAgent()
        self.headers = {
            'User-Agent': self.user_agent.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        self.proxy_pool = []
        self.working_proxies = set()
        self.proxy_timeout = 5
        self.proxy_refresh_interval = 300
        self.last_proxy_refresh = time.time()
        self.current_working_proxy = None
        self.successful_requests = 0
        self.max_requests_per_proxy = 200
        
    
    async def search(self, query: str, max_results: int = 300) -> List[Dict[str, str]]:
        cache_key = f"{query}_{max_results}"
        logging.info(f"🔍 Starting search for: '{query}'")

        # Check cache first
        if cache_key in self.search_cache:
            logging.info("📋 Using cached results")
            return self.search_cache[cache_key]

        # Try working proxies that have confirmed 200 status
        for proxy in self.working_proxies:
            try:
                proxy_dict = {'http': f'http://{proxy}', 'https': f'http://{proxy}'}
                
                # Verify proxy still returns 200
                async with aiohttp.ClientSession(headers=self.headers) as session:
                    async with session.get('https://duckduckgo.com/', proxy=proxy_dict['http'], timeout=10) as response:
                        if response.status == 200:
                            # Use this working proxy for search
                            ddgs = DDGS(proxy=proxy_dict['http'], timeout=30)
                            results = list(ddgs.text(query, max_results=max_results))
                            
                            if results:
                                search_results = [{
                                    'title': r.get('title', ''),
                                    'body': r.get('body', ''),
                                    'link': r.get('link', '')
                                } for r in results]
                                
                                self.search_cache[cache_key] = search_results
                                logging.info(f"✅ Search successful with proxy: {proxy}")
                                return search_results
                                
            except Exception as e:
                logging.debug(f"Proxy {proxy} failed: {str(e)}")
                continue

        # Direct connection fallback
        logging.info("⚡ Using direct connection")
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
            # Get search results with caching
            cache_key = f"summary_{query}"
            if cache_key in self.search_cache:
                return self.search_cache[cache_key]

            # Get fresh search results
            search_results = await self.search(query)
            
            # Format results for Gemini with rich context
            search_results_text = "\n".join([
                f'[{i}] Title: {r["title"]}\nSnippet: {r["body"]}\nSource: {r["link"]}\n'
                for i, r in enumerate(search_results[:10])  # Limit to top 10 results
            ])

            # Create optimized prompt for Gemini
            prompt = f"""You are Puro from Changed, analyzing search results for: '{query}'
            
            Web Search Findings:
            {search_results_text}
            
            Provide an enthusiastic and informative summary focusing on the most relevant details."""

            # Generate enhanced summary with Gemini
            response = model.generate_content(prompt)
            summary = response.text
            
            # Cache the summary
            self.search_cache[cache_key] = summary
            
            return summary

        except Exception as e:
            logging.error(f"Search summary generation error: {e}")
            return f"Let me share what I found about '{query}'!"

    
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
        # Setup image processing
        user_id = str(message.author.id)
        image = message.attachments[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f"images/{user_id}_{timestamp}.jpg"
        os.makedirs("images", exist_ok=True)

        # Download and optimize image
        await image.save(image_path)
        with PIL.Image.open(image_path) as img:
            # Optimize size while maintaining aspect ratio
            if img.size[0] > 3072 or img.size[1] > 3072:
                img.thumbnail((3072, 3072), PIL.Image.Resampling.LANCZOS)
            elif max(img.size) < 768:
                img.thumbnail((768, 768), PIL.Image.Resampling.LANCZOS)
            
            # Save optimized image
            img.save(image_path, format='JPEG', quality=95)

        # Process with Gemini
        with open(image_path, 'rb') as img_file:
            image_data = img_file.read()
            image_parts = [{"mime_type": "image/jpeg", "data": image_data}]
            
            # Analyze image
            image_response = model.generate_content([
                "Analyze this image thoroughly, describing key elements, text, and notable details.",
                image_parts
            ])
            image_analysis = image_response.text

        # Get web search results
        search_results = await web_search_handler.gemini_search_and_summarize(image_analysis)

        # Generate comprehensive response
        final_prompt = f"""You are Puro from Changed, a friendly dark latex wolf.
        Image Analysis: {image_analysis}
        Web Research: {search_results}
        
        Create an enthusiastic response combining the image analysis and web information."""

        final_response = model.generate_content(final_prompt)
        await message.channel.send(final_response.text)

    except Exception as e:
        logging.error(f"Image processing error: {e}")
        await message.channel.send("*Tail wags excitedly* Let me analyze that image for you!")


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
    def __init__(self, dimension: int = 384, max_memories: int = 1000):
        # Initialize multilingual sentence transformer
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        # Initialize FAISS indexes with 384 dimensions (MiniLM model's output dimension)
        self.long_term_index = faiss.IndexFlatL2(dimension)
        self.short_term_index = faiss.IndexFlatL2(dimension)
        
        # Memory storage with capacity limits
        self.long_term_memories = []
        self.short_term_memories = []
        self.dimension = dimension
        self.max_memories = max_memories
        self.total_memories_added = 0
        self.last_cleanup_time = datetime.now()

    def add_memory(self, text: str, is_long_term: bool = False, importance_score: float = 0.0):
        try:
            # Generate embedding using multilingual model
            embedding = self.model.encode(text, convert_to_numpy=True)
            memory_entry = {
                'text': text,
                'embedding': embedding,
                'timestamp': datetime.now().isoformat(),
                'importance_score': importance_score
            }
            
            if is_long_term:
                self.long_term_index.add(np.array([embedding]))
                self.long_term_memories.append(memory_entry)
                self._manage_long_term_capacity()
            else:
                self.short_term_index.add(np.array([embedding]))
                self.short_term_memories.append(memory_entry)
                self._manage_short_term_capacity()
                
            self.total_memories_added += 1
            
        except Exception as e:
            logging.error(f"Memory addition error: {e}")

    def search_memories(self, query: str, k: int = 5, search_long_term: bool = True) -> List[Dict]:
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        
        if search_long_term:
            D, I = self.long_term_index.search(np.array([query_embedding]), k)
            results = [self.long_term_memories[i] for i in I[0] if i >= 0]
        else:
            D, I = self.short_term_index.search(np.array([query_embedding]), k)
            results = [self.short_term_memories[i] for i in I[0] if i >= 0]
            
        return sorted(results, 
                     key=lambda x: (x['importance_score'], 
                                  -abs((datetime.now() - datetime.fromisoformat(x['timestamp'])).total_seconds())),
                     reverse=True)

    def _manage_short_term_capacity(self):
        if len(self.short_term_memories) > self.max_memories:
            self.short_term_memories.sort(key=lambda x: (x['importance_score'], x['timestamp']))
            self.short_term_memories = self.short_term_memories[-self.max_memories:]
            self._rebuild_short_term_index()
            
    def _manage_long_term_capacity(self):
        if len(self.long_term_memories) > self.max_memories:
            self.long_term_memories.sort(key=lambda x: x['importance_score'], reverse=True)
            self.long_term_memories = self.long_term_memories[:self.max_memories]
            self._rebuild_long_term_index()
            
    def _rebuild_short_term_index(self):
        self.short_term_index = faiss.IndexFlatL2(self.dimension)
        if self.short_term_memories:
            embeddings = np.array([mem['embedding'] for mem in self.short_term_memories])
            self.short_term_index.add(embeddings)
            
    def _rebuild_long_term_index(self):
        self.long_term_index = faiss.IndexFlatL2(self.dimension)
        if self.long_term_memories:
            embeddings = np.array([mem['embedding'] for mem in self.long_term_memories])
            self.long_term_index.add(embeddings)

    async def cleanup_routine(self):
        while True:
            try:
                current_time = datetime.now()
                self.short_term_memories = [
                    mem for mem in self.short_term_memories 
                    if (current_time - datetime.fromisoformat(mem['timestamp'])).days < 1
                ]
                self._rebuild_short_term_index()
                self.last_cleanup_time = current_time
                await asyncio.sleep(3600)
                
            except Exception as e:
                logging.error(f"Memory cleanup error: {e}")
                await asyncio.sleep(300)

    def clear_short_term(self):
        self.short_term_index = faiss.IndexFlatL2(self.dimension)
        self.short_term_memories = []
        
    def get_stats(self) -> Dict:
        return {
            'total_memories': len(self.short_term_memories) + len(self.long_term_memories),
            'short_term_count': len(self.short_term_memories),
            'long_term_count': len(self.long_term_memories),
            'total_added': self.total_memories_added,
            'last_cleanup': self.last_cleanup_time.isoformat()
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
        logging.info(f"\n🔵 Processing message from {message.author.name} ({user_id}): {content}")

        # Process images if present
        if message.attachments:
            for attachment in message.attachments:
                if attachment.content_type.startswith('image/'):
                    await handle_image(message, {'user_id': user_id, 'content': content})
                    return

        # Get memory context using FAISS
        message_embedding = sentence_transformer.encode([content])[0]
        relevant_memories = memory_manager.search_memories(content, k=3)
        memory_context = "\n".join([mem['text'] for mem in relevant_memories])
        logging.info(f"📚 Found {len(relevant_memories)} relevant memories")

        async with message.channel.typing():
            # Get web search results using proxy manager
            search_results = None
            web_context = None
            
            working_proxy = proxy_manager.get_working_proxy()
            if working_proxy:
                try:
                    async with aiohttp.ClientSession() as session:
                        proxy = {'http': f'http://{working_proxy}', 'https': f'http://{working_proxy}'}
                        search_results = await web_search_handler.search(content, proxy=proxy)
                        if search_results:
                            web_context = await web_search_handler.gemini_search_and_summarize(content)
                            logging.info(f"🌐 Web search successful using proxy: {working_proxy}")
                except Exception as e:
                    logging.error(f"Proxy search failed: {str(e)}")

            # Generate personalized prompt
            user_profile = user_profiles.get(user_id, {})
            personality = user_profile.get('personality', {})
            humor_level = personality.get('humor', 0.8)
            kindness_level = personality.get('kindness', 0.9)

            puro_prompt = f"""You are Puro from Changed, a friendly dark latex wolf.
            User Message: {content}
            Previous Context: {memory_context}
            Web Research: {web_context if web_context else 'Using my knowledge to help!'}
            
            Personality Settings:
            - Humor: {humor_level}
            - Kindness: {kindness_level}
            
            RESPOND AS PURO:
            - Show enthusiasm and curiosity
            - Use memories and research naturally
            - Stay focused and engaging
            - Maintain consistent personality"""

            # Generate and send response
            response = model.generate_content(puro_prompt)
            response_text = response.text.replace("Gemini: ", "")
            await message.channel.send(response_text)
            logging.info("✨ Generated and sent response")

            # Update memory systems with importance scoring
            importance_score = calculate_importance(content)
            memory_manager.add_memory(content, is_long_term=False)
            
            if importance_score > LONG_TERM_THRESHOLD:
                memory_manager.add_memory(content, is_long_term=True)
                logging.info(f"💾 Stored high-importance memory (score: {importance_score:.2f})")

            # Initialize or update user profile
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

            # Update conversation history
            user_profiles[user_id]["context"].append({"role": "user", "content": content})
            user_profiles[user_id]["context"].append({"role": "assistant", "content": response_text})
            
            # Save to database and profiles
            await save_chat_history(
                user_id=user_id,
                message=content,
                user_name=message.author.name,
                bot_id=bot.user.id,
                bot_name=bot.user.name,
                importance_score=importance_score
            )
            save_user_profiles()
            logging.info("✅ Successfully updated memory and user profile")

    except Exception as e:
        logging.error(f"❌ Error in message handling: {str(e)}", exc_info=True)
        await message.channel.send("*Tail wags excitedly* Let me share my thoughts on that!")


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
