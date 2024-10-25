
import logging
import json
from collections import defaultdict, deque
from datetime import datetime, timezone
import asyncio
import random
from transitions import Machine
import numpy as np
import requests
import aiohttp
from telegram import Bot, Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from typing import Optional, Tuple
import chardet
from textblob import TextBlob
import spacy
from opentelemetry import trace
from prometheus_client import Counter, Histogram
import statsd
import signal
import re
from langdetect import detect, LangDetectException
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import faiss
import base64
from groq import Groq
from telegram.constants import ParseMode
import asyncio
import random
import time
import logging
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from duckduckgo_search import AsyncDDGS
from tenacity import retry, stop_after_attempt, wait_exponential
from cachetools import TTLCache
from aiohttp import ClientSession, ClientTimeout
import structlog
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from sklearn.cluster import KMeans
from duckduckgo_search import AsyncDDGS
from cachetools import TTLCache
import signal
from loguru import logger
import structlog
import msvcrt
import traceback
import httpx
from asyncio_throttle import Throttler
from asyncio import Semaphore
from tenacity import retry, stop_after_attempt, wait_exponential
from groq import AsyncGroq
import spacy
import torch
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from concurrent.futures import ThreadPoolExecutor
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import logging
import random
from typing import List, Dict, Any
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder
import platform
import tempfile
from telegram.request import HTTPXRequest
from telegram import Bot
from telegram.ext import ApplicationBuilder
import httpx
import requests
import msvcrt
import os
import time
import asyncio
from telegram import Bot
from telegram.request import HTTPXRequest
from telegram.ext import ApplicationBuilder
import httpx
import psutil
import os
import time
from langdetect import detect as langdetect_detect
from langdetect import DetectorFactory
from sentence_transformers import SentenceTransformer
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from groq import AsyncGroq
import torch
import torch.nn as nn
import torch.nn.functional as F
import asyncio
import aiosqlite
from __main__ import __name__ as name
import time
import random
import requests
import asyncio
import structlog
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager
from tenacity import retry, stop_after_attempt, wait_exponential
from duckduckgo_search import AsyncDDGS
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from aiohttp import ClientSession
from urllib.parse import urlparse, urljoin
import tldextract
import textstat
import io
import PyPDF2
from urllib.parse import urlparse
import tracemalloc
import langdetect
from cachetools import TTLCache
from concurrent.futures import ThreadPoolExecutor
import ipaddress
from aiolimiter import AsyncLimiter
import socket
from aiohttp_retry import RetryClient, ExponentialRetry
from loguru import logger
from duckduckgo_search import AsyncDDGS
import asyncio
import random
import time
import logging
import requests
from bs4 import BeautifulSoup
from aiohttp import ClientSession, ClientTimeout
from tenacity import retry, stop_after_attempt, wait_exponential
from cachetools import TTLCache
from spacy.cli import download
import sys
import statsd


def download_spacy_model(model_name="en_core_web_trf"):
    try:d

    
        spacy.load(model_name)
        print(f"SpaCy model '{model_name}' is already downloaded.")
    except OSError:
        print(f"Downloading SpaCy model '{model_name}'...")
        download(model_name)
        print(f"SpaCy model '{model_name}' has been successfully downloaded.")

# Call this function before loading the model
download_spacy_model()

nlp = spacy.load("en_core_web_trf")  # or another appropriate model




file = getattr(sys.modules['__main__'], '__file__', None)

CODE_FOLDER = os.path.dirname(os.path.abspath(file))
DATABASE_FILE = os.path.join(CODE_FOLDER, "telegram_chat_history.db")
USER_PROFILES_FILE = os.path.join(CODE_FOLDER, "telegram_user_profiles.json")
KNOWLEDGE_GRAPH_FILE = os.path.join(CODE_FOLDER, "telegram_knowledge_graph.pkl")
IMAGE_MEMORY_FOLDER = os.path.join(CODE_FOLDER, "telegram_image_memory")
FAISS_INDEX_FILE = os.path.join(CODE_FOLDER, "telegram_faiss_index.bin")

BOT_TOKEN = "7871809935:AAFBbowW8a-BFHR5oAuOqy1g7TucKRIOsZw"
GROQ_API_KEY = "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"
API_KEY = 'AIzaSyA3MZZBARvTVM0OhY7oQTNBIH1l3YpOXZk'  
CSE_ID = '033b47c68f344483c'  



CONTEXT_WINDOW_SIZE = 8000
DEFAULT_PERSONALITY = {"humor": 0.7, "kindness": 0.9, "assertiveness": 0.3, "playfulness": 0.8}
DDGS_CACHE_SIZE = 10000
DDGS_CACHE_TTL = 3600

ERROR_COUNTER = Counter('error_count', 'Number of errors')
PROCESSING_TIME = Histogram('processing_time', 'Time taken for processing')

tracer = trace.get_tracer(__name__)

statsd_client = statsd.StatsClient()

logger.add("bot.log", rotation="1 MB", retention="10 days", level="DEBUG", enqueue=True)

tracemalloc.start()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(name)

structlog.configure(processors=[
    structlog.stdlib.filter_by_level,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
], context_class=dict, logger_factory=structlog.stdlib.LoggerFactory(),
wrapper_class=structlog.stdlib.BoundLogger, cache_logger_on_first_use=True)

try:
    download_spacy_model("en_core_web_trf")
    nlp = spacy.load("en_core_web_trf")
except Exception as e:
    structlog.get_logger().error(f"Failed to load SpaCy model: {e}", exc_info=True)
    nlp = None



def download_spacy_model(model_name="en_core_web_trf"):
    try:
        spacy.load(model_name)
    except OSError:
        download(model_name)

download_spacy_model()


try:
    embedding_model = SentenceTransformer('all-mpnet-base-v2')
except Exception as e:
    structlog.get_logger().error(f"Failed to load embedding model: {e}", exc_info=True)
    embedding_model = None

tfidf_vectorizer = TfidfVectorizer()
sentiment_analyzer = SentimentIntensityAnalyzer()
ddg_cache = TTLCache(maxsize=DDGS_CACHE_SIZE, ttl=DDGS_CACHE_TTL)

telegram_throttler = Throttler(rate_limit=25, period=60.0)
groq_throttler = Throttler(rate_limit=5, period=60.0)
ddg_throttler = Throttler(rate_limit=1, period=1)


db_ready = False
db_lock = asyncio.Lock()
db_queue = asyncio.Queue()
shutdown_event = asyncio.Event()
faiss_index = None


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
        if embedding_model:
            try:
                self.embedding_cache[node_id] = embedding_model.encode(str(data))
            except Exception as e:
                structlog.get_logger().error(f"Failed to encode data for Knowledge Graph: {e}", exc_info=True)
                self.embedding_cache[node_id] = None


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
        related_nodes = []
        if node is not None and "edges" in node:
            for edge in node["edges"]:
                if relation is None or edge["relation"] == relation:
                    if direction in ("outgoing", "both"):
                        related_node = self.get_node(edge["target_type"], edge["target_id"])
                        if related_node:
                            related_nodes.append(related_node)
                    if direction in ("incoming", "both"):
                        for nt, nodes in self.graph.items():
                            for nid, n in nodes.items():
                                if "edges" in n:
                                    for e in n["edges"]:
                                        if e["target_id"] == node_id and (relation is None or e["relation"] == relation):
                                            related_nodes.append(n)
        return related_nodes

    def search_nodes(self, query, top_k=3, node_type=None):
        if not embedding_model:
            structlog.get_logger().warning("Cannot search nodes: Embedding model not loaded.")
            return []
        try:
            query_embedding = embedding_model.encode(query)
        except Exception as e:
            structlog.get_logger().error(f"Failed to encode query for node search: {e}", exc_info=True)
            return []

        results = []
        for current_node_type, nodes in self.graph.items():
            if node_type is None or current_node_type == node_type:
                for node_id, node_data in nodes.items():
                    node_embedding = self.embedding_cache.get(node_id)
                    if node_embedding is not None:
                        try:
                            similarity = cosine_similarity([query_embedding], [node_embedding])[0][0]
                            results.append((current_node_type, node_id, node_data, similarity))
                        except Exception as e:
                            structlog.get_logger().error(f"Failed to calculate cosine similarity: {e}", exc_info=True)

        results.sort(key=lambda x: x[3], reverse=True)
        return results[:top_k]

    def update_node(self, node_type, node_id, new_data):
        node = self.get_node(node_type, node_id)
        if node is not None:
            self.graph[node_type][node_id].update(new_data)
            if embedding_model:
                try:
                    self.embedding_cache[node_id] = embedding_model.encode(str(self.graph[node_type][node_id]))
                except Exception as e:
                    structlog.get_logger().error(f"Failed to encode updated node data: {e}", exc_info=True)


    def delete_node(self, node_type, node_id):
        if node_type in self.graph and node_id in self.graph[node_type]:
            del self.graph[node_type][node_id]
            if node_id in self.embedding_cache:
                del self.embedding_cache[node_id]
            for nt, nodes in self.graph.items():
                for nid, node in nodes.items():
                    if "edges" in node:
                        node["edges"] = [edge for edge in node["edges"] if edge["target_id"] != node_id]

    def save_to_file(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_file(filename):
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
            structlog.get_logger().error(f"Failed to load knowledge graph from file: {e}", exc_info=True)
            return KnowledgeGraph()

knowledge_graph = KnowledgeGraph.load_from_file(KNOWLEDGE_GRAPH_FILE)


def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

async def analyze_image(image_url, api_key=None):
    if api_key is None:
        api_key = GROQ_API_KEY
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as resp:
                if resp.status != 200:
                    return {"success": False, "error": f"Error downloading image (status code: {resp.status})."}
                image_bytes = await resp.read()
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                try:
                    async with AsyncGroq(api_key=api_key) as client:
                        completion = await client.chat.completions.create(
                            model="llama-3.2-11b-vision-preview",
                            messages=[{
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Describe this image:"},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                                ]
                            }],
                            temperature=0.7,
                            max_tokens=1024,
                            top_p=1,
                            stream=False,
                        )
                        description = completion.choices[0].message.content
                        return {"success": True, "description": description}
                except Exception as e:
                    structlog.get_logger().error(f"Groq image analysis failed: {e}", exc_info=True)
                    return {"success": False, "error": f"Groq API error: {e}"}
    except Exception as e:
        structlog.get_logger().error(f"Image analysis failed: {e}", exc_info=True)
        return {"success": False, "error": f"An error occurred: {str(e)}"}


async def save_image_to_memory(image_url, image_description, user_id):
    os.makedirs(IMAGE_MEMORY_FOLDER, exist_ok=True)
    image_filename = os.path.join(IMAGE_MEMORY_FOLDER, f"{user_id}_{int(time.time())}.jpg")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as resp:
                if resp.status == 200:
                    image_data = await resp.read()
                    with open(image_filename, 'wb') as f:
                        f.write(image_data)
                    await store_long_term_memory(user_id, "image_memory",
                                                 {"description": image_description, "filename": image_filename})
    except Exception as e:
        structlog.get_logger().error(f"Error saving image: {str(e)}", exc_info=True)


async def store_long_term_memory(user_id, information_type, information):
    try:
        knowledge_graph.add_node(information_type, data={"user_id": user_id, "information": information})
        knowledge_graph.add_edge("user", user_id, "has_" + information_type, information_type,
                                knowledge_graph.node_id_counter - 1)
        knowledge_graph.save_to_file(KNOWLEDGE_GRAPH_FILE)
    except Exception as e:
        structlog.get_logger().error("Error storing long-term memory", exc_info=True)



async def retrieve_long_term_memory(user_id, information_type, query=None, top_k=3):
    try:
        if query:
            search_results = knowledge_graph.search_nodes(query, top_k=top_k, node_type=information_type)
            return [(node_type, node_id, node_data) for node_type, node_id, node_data, score in search_results]
        related_nodes = knowledge_graph.get_related_nodes("user", user_id, "has_" + information_type)
        return related_nodes
    except Exception as e:
        structlog.get_logger().error("Error retrieving long-term memory", exc_info=True)
        return []


async def call_groq(prompt, user_id=None, language="en", model="llama-3.2-90b-text-preview",
                    temperature=0.7, max_tokens=512, top_p=1, retry_count=3):
    for attempt in range(retry_count):
        try:
            async with groq_throttler:
                async with AsyncGroq(api_key=GROQ_API_KEY) as client:
                    completion = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        stream=False,
                    )
                    groq_response = completion.choices[0].message.content
                    cleaned_groq_response = re.sub(r'(.?)', '', groq_response)
                    cleaned_groq_response = re.sub(r"[(.?)]", r"\1", cleaned_groq_response)
                    return cleaned_groq_response
        except Exception as e:
            structlog.get_logger().error(f"Groq API error (attempt {attempt + 1}/{retry_count}):", exc_info=True, error=str(e))
            if attempt < retry_count - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    return f"Error: Failed to get a response from Groq after {retry_count} attempts."


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class AdvancedClassifier(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=100):
        super(AdvancedClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x



def load_or_create_classifier(model_path='advanced_classifier_model.pth'):
    try:
        classifier = AdvancedClassifier()
        classifier.load_state_dict(torch.load(model_path))
        return classifier
    except FileNotFoundError:
        classifier = AdvancedClassifier()
        torch.save(classifier.state_dict(), model_path)
        structlog.get_logger().warning("Classifier model not found, creating a new one.")
        return classifier
    except Exception as e:
        structlog.get_logger().error(f"Failed to load classifier: {e}", exc_info=True)
        return AdvancedClassifier()  # Return a default classifier in case of error


def load_or_create_sentiment_analyzer(model_path='advanced_sentiment_model.pth'):
    try:
        sentiment_analyzer = nn.Linear(1024, 1)
        sentiment_analyzer.load_state_dict(torch.load(model_path))
        return sentiment_analyzer
    except FileNotFoundError:
        sentiment_analyzer = nn.Linear(1024, 1)
        torch.save(sentiment_analyzer.state_dict(), model_path)
        structlog.get_logger().warning("Sentiment analyzer model not found, creating a new one.")
        return sentiment_analyzer
    except Exception as e:
        structlog.get_logger().error(f"Failed to load sentiment analyzer: {e}", exc_info=True)
        return nn.Linear(1024, 1)


async def async_load_or_create_classifier(model_path='advanced_classifier_model.pth'):
    return await asyncio.to_thread(load_or_create_classifier, model_path)


async def async_load_or_create_sentiment_analyzer(model_path='advanced_sentiment_model.pth'):
    return await asyncio.to_thread(load_or_create_sentiment_analyzer, model_path)


async def train_classifier(classifier, data, labels):
    try:
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        classifier.train()
        for epoch in range(10):
            optimizer.zero_grad()
            outputs = classifier(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        torch.save(classifier.state_dict(), 'advanced_classifier_model.pth')
        structlog.get_logger().info("Trained and saved advanced classifier model.")
    except Exception as e:
        structlog.get_logger().error(f"Failed to train classifier: {e}", exc_info=True)



def terminate_existing_instance():
    current_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'python' and len(proc.info['cmdline']) > 1:
                if 'complex chat bot for telegram v123.py' in proc.info['cmdline'][1] and proc.info['pid'] != current_pid:
                    proc.terminate()
                    proc.wait(timeout=10)
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
            structlog.get_logger().warning(f"Error checking process: {e}")
    return False

def cleanup_stale_lock(lock_file_path, age_threshold=300):
    try:
        if os.path.exists(lock_file_path):
            lock_file_age = time.time() - os.path.getmtime(lock_file_path)
            if lock_file_age > age_threshold:
                os.remove(lock_file_path)
                return True
    except OSError as e:
        structlog.get_logger().error(f"Failed to cleanup stale lock: {e}", exc_info=True)
    return False

def acquire_lock_windows():

    lock_file_path = "bot_lock_file.lock"
    max_attempts = 15
    retry_delay = 1
    lock_file_age_threshold = 300

    for attempt in range(max_attempts):

        if cleanup_stale_lock(lock_file_path, lock_file_age_threshold):
            structlog.get_logger().info("Removed stale lock file")

        try:
            lock_file = open(lock_file_path, "wb")

            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            return lock_file
        except IOError:

            if terminate_existing_instance():
                time.sleep(1)
            else:
                time.sleep(retry_delay + random.uniform(0, 0.5))
        finally:
            if 'lock_file' in locals() and not lock_file.closed:
                lock_file.close()
    return None



def release_lock_windows(lock_file):
    if lock_file and hasattr(lock_file, 'fileno'):
        try:
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        except (IOError, ValueError):
            pass
        finally:
            try:
                lock_file.close()
            except:
                pass

    try:
        if os.path.exists("bot_lock_file.lock"):
            os.remove("bot_lock_file.lock")
    except OSError:
        pass


async def signal_handler(sig, frame):
    log = structlog.get_logger()
    log.info(f"Received signal {sig}, shutting down...")
    shutdown_event.set()
    if application:
        await application.stop()
        await application.shutdown()
    log.info("Application stopped and shutdown")

if platform.system() == "Windows":
    print("Running on Windows")
else:
    import fcntl
    print("Running on Unix-like system")

try:
    encoder_layer = TransformerEncoderLayer(d_model=1024, nhead=8)
    encoder = TransformerEncoder(encoder_layer, num_layers=6)
except Exception as e:
    structlog.get_logger().error(f"Failed to create encoder: {e}", exc_info=True)
    encoder = None

class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

if embedding_model:
    projection_layer = ProjectionLayer(embedding_model.get_sentence_embedding_dimension(), 1024)
else:
    projection_layer = None

logger = logging.getLogger(__name__)
search_cache = {}
search_rate_limiter = asyncio.Semaphore(5)


async def initialize_bot():
    global user_profiles, bot, application, faiss_index, embedding_model, encoder
    try:
        user_profiles = await load_user_profiles()
        bot = Bot(token=BOT_TOKEN, request=HTTPXRequest(httpx.AsyncClient(timeout=httpx.Timeout(120.0))))
        application = (ApplicationBuilder()
                       .token(BOT_TOKEN)
                       .concurrent_updates(True)
                       .request_kwargs({'read_timeout': 120, 'connect_timeout': 20})
                       .build())
        if embedding_model is None:
            try:
                embedding_model = SentenceTransformer('all-mpnet-base-v2')
                global projection_layer
                projection_layer = ProjectionLayer(embedding_model.get_sentence_embedding_dimension(), 1024)

            except Exception as e:
                structlog.get_logger().error(f"Failed to load embedding model during init: {e}", exc_info=True)

        if encoder is None:
            try:
                encoder_layer = TransformerEncoderLayer(d_model=1024, nhead=8)
                encoder = TransformerEncoder(encoder_layer, num_layers=6)
            except Exception as e:
                structlog.get_logger().error(f"Failed to create encoder during init: {e}", exc_info=True)

        await init_db()
        await load_faiss_index()
        if faiss_index is None:
            if embedding_model:
                faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
            else:
                structlog.get_logger().error("Cannot create FAISS index: Embedding model not loaded.")
                raise RuntimeError("Embedding model required for FAISS index")
            structlog.get_logger().info("Created new FAISS index")
        structlog.get_logger().info("Bot initialization completed successfully")
    except Exception as e:
        structlog.get_logger().error("Bot initialization encountered an issue:", exc_info=True)
        raise



async def generate_self_reflection_prompt(query, relevant_history, summarized_search, user_id, message, api_key, error):
    prompt = f"""
An error occurred during the advanced reasoning process. Here are the details:

User Query: {query}
Relevant History: {relevant_history}
Summarized Search: {summarized_search}
User ID: {user_id}
Message: {message}
API Key: {api_key}
Error: {error}

As an advanced AI assistant, analyze the error and generate a detailed self-reflection prompt to fix the error. The prompt should include:
1. A summary of the error and its potential causes
2. A detailed analysis of the error, considering all possible factors
3. Potential fixes and strategies to address the error
4. A plan for retrying the advanced reasoning process with the identified fixes

Format your response entirely in the language of the user, ensuring it's natural and conversational.
"""
    try:
        self_reflection_text, _, _ = await advanced_reasoning_with_groq(prompt, [], "", user_id, message, api_key)
        return self_reflection_text
    except Exception as e:
        structlog.get_logger().error(f"Failed to generate self-reflection prompt: {e}", exc_info=True)
        return "Error: Could not generate a self-reflection prompt."


async def advanced_reasoning_with_groq(query, relevant_history=None, summarized_search=None, user_id=None, message=None, 
                                       api_key=GROQ_API_KEY, language="en", model="llama-3.2-90b-text-preview", 
                                       timeout=60, max_tokens=1024, temperature=0.7, top_p=1):
    start_time = time.time()
    response_text = ""
    error_message = None

    try:
        structlog.get_logger().info("Starting advanced reasoning process with Groq integration.")
        
        # Load classifier and sentiment analyzer asynchronously
        classifier = await async_load_or_create_classifier()
        sentiment_analyzer = await async_load_or_create_sentiment_analyzer()

        # Check and create embeddings
        if embedding_model and encoder and projection_layer:
            query_embedding = projection_layer(torch.tensor([embedding_model.encode(query)]).float().unsqueeze(0))
            query_encoding = encoder(query_embedding)
        else:
            if not embedding_model:
                structlog.get_logger().error("Embedding model not loaded, cannot encode query.")
                raise RuntimeError("Embedding model required for query encoding")
            if not encoder:
                structlog.get_logger().error("Encoder not loaded, cannot encode query.")
                raise RuntimeError("Encoder required for query encoding")

        # Encode history if available
        if relevant_history and embedding_model and encoder and projection_layer:
            history_embeddings = torch.tensor([embedding_model.encode(h) for h in relevant_history]).float()
            projected_history = projection_layer(history_embeddings)
            history_encoding = encoder(projected_history.unsqueeze(0))
        else:
            history_encoding = torch.zeros(1, 1, 1024)

        # Classification
        if classifier:
            topic_probs = F.softmax(classifier(query_encoding.mean(dim=1)), dim=1)
            current_topic = torch.argmax(topic_probs).item()
        else:
            structlog.get_logger().warning("Classifier not available, using default topic.")
            current_topic = 0

        structlog.get_logger().info(f"Current topic identified: {current_topic}")

        # Check topic continuity
        is_continuous, continuity_message = await check_topic_continuity(user_id, current_topic)
        structlog.get_logger().info(f"Topic continuity checked: {is_continuous}, {continuity_message}")

        # Retrieve related memories asynchronously
        related_memories = await get_related_memories(user_id, query, top_k=5)
        structlog.get_logger().info("Related memories retrieved.")

        # Sentiment analysis
        if sentiment_analyzer:
            sentiment_score = sentiment_analyzer(query_encoding.mean(dim=1)).item()
        else:
            structlog.get_logger().warning("Sentiment analyzer not available, using default sentiment score.")
            sentiment_score = 0.0

        # Update user personality
        update_personality(user_profiles[user_id]["personality"], sentiment_score)
        structlog.get_logger().info(f"Sentiment score calculated: {sentiment_score}")

        # Create the prompt
        prompt = f"""
        User Query: {query}
        Language: {language}
        Search Results: {summarized_search}
        Relevant History: {relevant_history}
        Related Memories: {related_memories}
        Current Topic: {current_topic}
        Topic Continuity: {continuity_message}
        User Personality: {user_profiles[user_id]["personality"]}
        Sentiment Score: {sentiment_score}

        As an advanced AI assistant, analyze the given information and generate a response in {language} that:
        1. Directly addresses the user's query with accuracy and relevance
        2. Incorporates the search results to provide up-to-date information
        3. Maintains context and topic continuity based on the conversation history
        4. Incorporates relevant historical information and memories to provide a personalized response
        5. Adapts to the user's personality and current sentiment, adjusting the tone accordingly
        6. Ensures the response is coherent, well-structured, and easy to understand
        7. Avoids biases and considers multiple perspectives when applicable
        8. Offers additional relevant information or follow-up questions to encourage engagement

        Format your response entirely in {language}, ensuring it's natural and conversational.
        """

        if not api_key:
            error_message = "API key not provided for Groq client."
            structlog.get_logger().error(error_message)
            return "", time.time() - start_time, error_message

        # Call Groq API for completion
        try:
            async with AsyncGroq(api_key=api_key) as client:
                try:
                    completion_iterator = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": f"Respond in {language}"},
                            {"role": "user", "content": prompt}
                        ],
                        stream=True,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                    )

                    async for chunk in asyncio.wait_for(completion_iterator, timeout):
                        if chunk.choices and chunk.choices[0].delta.content:
                            response_text += chunk.choices[0].delta.content
                except asyncio.TimeoutError:
                    error_message = "Request timed out."
                    structlog.get_logger().warning(error_message)
                except Exception as e:
                    error_message = f"Groq API request failed: {e}"
                    structlog.get_logger().error(error_message, exc_info=True)
        except Exception as e:
            error_message = f"Groq client or request error: {e}"
            structlog.get_logger().error(error_message, exc_info=True)

        if not error_message:
            # Store response and update user profile
            if user_id is not None and isinstance(user_profiles.get(user_id), dict):
                user_profiles[user_id]["context"].append({"role": "assistant", "content": response_text})
                user_profiles[user_id]["recent_topics"].append(current_topic)
                if len(user_profiles[user_id]["recent_topics"]) > 10:
                    user_profiles[user_id]["recent_topics"].pop(0)
                structlog.get_logger().info("User profile updated.")

                await store_long_term_memory(user_id, "interaction", {
                    "query": query,
                    "response": response_text,
                    "topic": current_topic,
                    "sentiment": sentiment_score,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

                structlog.get_logger().info("Long-term memory stored.")

                # Update persistent models
                try:
                    await update_persistent_models(query, response_text, sentiment_score, current_topic, classifier, sentiment_analyzer)
                    structlog.get_logger().info("Persistent models updated.")
                except Exception as e:
                    structlog.get_logger().error(f"Error updating persistent models: {e}", exc_info=True)

        structlog.get_logger().info("Advanced reasoning process completed.")
        return response_text, time.time() - start_time, error_message

    except Exception as e:
        error_message = f"An error occurred during advanced reasoning: {e}"
        structlog.get_logger().error("Advanced reasoning failed:", exc_info=True)
        return "", time.time() - start_time, error_message


class TranslationSystem:

    def __init__(self, model: str):
        self.model = model
        self.load_or_verify_model()

    def load_or_verify_model(self):
        logging.info(f"Using model: {self.model}")

    async def call_groq(self, prompt: str, user_id: Optional[str] = None, language: str = "en") -> str:

        try:
            translated_text = await call_groq(prompt, user_id, language)
            return translated_text
        except Exception as e:
            structlog.get_logger().error(f"Translation failed: {e}", exc_info=True)
            return "Translation Error"

    async def translate(self, text: str, target_language: str) -> str:
        logging.info(f"Starting translation for '{text}' to '{target_language}'...")
        prompt = f"Translate the following text to {target_language}: {text}"
        return await self.call_groq(prompt)


def apply_deep_neural_translation(text: str, target_lang="en") -> str:
    translation_system = TranslationSystem("llama-3.2-90b-text-preview") # Initialize within the function
    try:
        loop = asyncio.get_event_loop()
        translated_text = loop.run_until_complete(translation_system.translate(text, target_lang))
        return translated_text
    except Exception as e:
        structlog.get_logger().error(f"Deep neural translation failed: {e}", exc_info=True)
        return "Translation failed."

def fallback_byte_decoder(data: bytes) -> str:
    try:
        return data.decode('utf-8', errors='ignore')
    except Exception as e:
        structlog.get_logger().error(f"Fallback byte decoding failed: {e}", exc_info=True)
        return "Decoding failed."

def analyze_error_context(query: str, error: Exception) -> str:
    return f"Query: {query}, Error: {error}"


def alternative_string_decoder(data: str) -> str:
    logging.info(f"Attempting to decode data: {data[:50]}...")

    detected_encoding = chardet.detect(data.encode())
    encoding = detected_encoding['encoding']

    logging.info(f"Detected encoding: {encoding}")

    try:
        decoded_data = data.encode().decode(encoding, errors='replace')  # First encode to bytes, then decode
        logging.info("Data decoded successfully.")
        return decoded_data
    except Exception as e:
        logging.error(f"Decoding failed: {e}")
        return data


def apply_spelling_correction(text: str) -> str:
    if nlp:
        doc = nlp(text)
        corrected_text = " ".join([token._.correct_spelling() if token._.has_spelling_suggestions else token.text for token in doc])

        logging.info(f"Spelling corrected from '{text}' to '{corrected_text}'")
        return corrected_text
    else:
        structlog.get_logger().warning("Spelling correction failed: SpaCy model not loaded.")
        return text


def apply_syntax_correction(text: str) -> str:

    if nlp:
        doc = nlp(text)
        corrected_text = " ".join([token.text for token in doc if not token.is_punct])
        logging.info(f"Syntax corrected from '{text}' to '{corrected_text}'")
        return corrected_text
    else:
        structlog.get_logger().warning("Syntax correction failed: SpaCy model not loaded.")
        return text


def fallback_correction_system(text: str) -> str:
    logging.info("Fallback correction system invoked.")
    return text


def compute_sentiment_score(text: str) -> float:
    try:
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity
        logging.info(f"Sentiment score for '{text}': {sentiment_score}")
        return sentiment_score
    except Exception as e:
        structlog.get_logger().error(f"Sentiment analysis failed: {e}", exc_info=True)
        return 0.0


def build_dynamic_classifier() -> nn.Module:
    logging.info("Dynamic classifier built with advanced models.")
    return AdvancedClassifier()

def build_dynamic_sentiment_analyzer() -> nn.Module:
    logging.info("Dynamic sentiment analyzer built with advanced models.")
    return nn.Linear(1024, 1)


class ComplexTextProcessingApp:
    def __init__(self):
        self.translation_system = TranslationSystem("llama-3.2-90b-text-preview")

    async def process_text(self, text: str) -> None:
        try:
            logging.info(f"Processing text: {text}")
            corrected_spelling = apply_spelling_correction(text)
            corrected_syntax = apply_syntax_correction(corrected_spelling)
            sentiment_score = compute_sentiment_score(corrected_syntax)
            translated_text = await self.translation_system.translate(corrected_syntax, "en")
            
            logging.info(f"Original text: {text}")
            logging.info(f"Corrected text: {corrected_syntax}")
            logging.info(f"Sentiment score: {sentiment_score}")
            logging.info(f"Translated text: {translated_text}")
        except Exception as e:
            logging.error(f"Error processing text: {str(e)}")
            fallback_correction_system(text)



async def self_reflect_and_fix_errors(query: str, relevant_history: list, summarized_search: str,
                                       user_id: str, message: str, api_key: str) -> Tuple[str, float, Optional[str]]:

    start_time = time.time()

    structlog.get_logger().info("Initiating a deeply sophisticated error-fixing procedure.", user_id=user_id, query=query)

    max_retries = 7
    backoff_factor = 3
    initial_wait = 2


    async def retry_with_exponential_backoff(func, *args, **kwargs):
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                wait_time = initial_wait * (backoff_factor ** attempt)
                structlog.get_logger().warning(f"Retry attempt {attempt + 1} failed due to: {str(e)}. Retrying in {wait_time}s.")
                await asyncio.sleep(wait_time)
        raise Exception(f"Max retries ({max_retries}) reached. Failing with error.")


    async def translate_to_english(text: str) -> str:
        structlog.get_logger().info("Initiating complex language detection and translation.")
        try:
            non_ascii_chars = any(not char.isascii() for char in text)
            if non_ascii_chars:
                structlog.get_logger().info("Detected non-ASCII characters, launching deep language analysis.")
                language_detected = await detect_language(text)
                if language_detected != 'en':
                    structlog.get_logger().info(f"Detected language: {language_detected}. Applying custom translation model.")
                    translated_text = await apply_deep_neural_translation(text, target_lang='en')
                    return translated_text
            structlog.get_logger().info("Text appears to be in English, skipping translation.")
            return text
        except Exception as e:
            structlog.get_logger().error(f"Language detection and translation failed: {e}", exc_info=True)
            return text  # Return original text if translation fails


    async def decode_encoding(encoding: str) -> str:
        structlog.get_logger().info("Initiating complex decoding procedure.")
        if isinstance(encoding, bytes):
            try:
                structlog.get_logger().info("Detected byte-encoded content. Decoding using multi-layer strategy.")
                decoded = encoding.decode('utf-8', errors='replace')
                return decoded
            except Exception as e:
                structlog.get_logger().warning(f"Failed to decode bytes, retrying with alternative encoding strategies: {str(e)}")
                return await retry_with_exponential_backoff(fallback_byte_decoder, encoding)
        elif isinstance(encoding, str):
            try:
                structlog.get_logger().info("Detected complex string encoding. Attempting multi-pass decoding process.")
                decoded = encoding.encode('utf-8').decode('utf-8')
                return decoded
            except Exception as e:
                structlog.get_logger().warning(f"String decoding failed, switching to multi-encoding strategies: {str(e)}")
                return await retry_with_exponential_backoff(alternative_string_decoder, encoding)
        return encoding


    async def apply_general_error_correction(query: str, error: str) -> str:
        structlog.get_logger().info(f"Attempting to apply general error corrections for error: {error}")
        error_context = analyze_error_context(query, error)
        if "missing punctuation" in error_context:
            structlog.get_logger().info("Detected missing punctuation error. Correcting.")
            return query + "."
        elif "spelling mistake" in error_context:
            structlog.get_logger().info("Detected spelling mistake. Applying corrections.")
            corrected_query = apply_spelling_correction(query)  # Removed await as apply_spelling_correction is no longer async
            return corrected_query
        elif "syntax" in error_context:
            structlog.get_logger().info("Detected syntax issue. Running advanced syntax correction.")
            corrected_query = apply_syntax_correction(query)  # Removed await as apply_syntax_correction is no longer async
            return corrected_query
        else:
            structlog.get_logger().warning(f"Unrecognized error pattern: {error}. Running fallback corrections.")
            return fallback_correction_system(query)  # Removed await as fallback_correction_system is no longer async


    async def preprocess_with_classifier(query: str, classifier: nn.Module):
        structlog.get_logger().info("Applying advanced classifier preprocessing.")
        try:
            if embedding_model and encoder and projection_layer:
                query_embedding = projection_layer(torch.tensor([embedding_model.encode(query)]).float().unsqueeze(0))
                query_encoding = encoder(query_embedding)

                topic_probs = F.softmax(classifier(query_encoding.mean(dim=1)), dim=1)
                current_topic = torch.argmax(topic_probs).item()
                return f"advanced_classified_query_{current_topic}: {query}"
            else:
                raise RuntimeError("Embedding model, encoder, or projection layer not loaded.")
        except Exception as e:
            structlog.get_logger().error(f"Classifier preprocessing failed: {e}", exc_info=True)
            return query

    async def preprocess_with_sentiment(query: str, sentiment_analyzer: nn.Module):
        structlog.get_logger().info("Applying sentiment preprocessing using deep sentiment models.")
        try:
            sentiment_score = compute_sentiment_score(query)
            if sentiment_score < 0:
                structlog.get_logger().info("Negative sentiment detected, adjusting query for neutrality.")
                return f"neutralized_query: {query}"
            return query
        except Exception as e:
            structlog.get_logger().error(f"Sentiment preprocessing failed: {e}", exc_info=True)
            return query


    async def load_fallback_classifier():
        structlog.get_logger().info("Loading fallback classifier for deep classification.")
        return build_dynamic_classifier() # Removed await as build_dynamic_classifier is no longer async


    async def load_fallback_sentiment_analyzer():
        structlog.get_logger().info("Loading fallback sentiment analyzer.")
        return build_dynamic_sentiment_analyzer() # Removed await as build_dynamic_sentiment_analyzer is no longer async


    for attempt in range(max_retries):
        try:

            response_text, elapsed_time, error = await asyncio.wait_for(
                advanced_reasoning_with_groq(query, relevant_history, summarized_search, user_id, message, api_key),
                timeout=60.0
            )

            if error:
                structlog.get_logger().error(f"Error in advanced reasoning process: {error}")
                ERROR_COUNTER.labels(error_type="advanced_reasoning").inc()


                self_reflection_prompt = await generate_self_reflection_prompt(
                    query, relevant_history, summarized_search, user_id, message, api_key, error
                )

                if "language detection" in error or "translation" in error:
                    query = await translate_to_english(query) # Use the translation function for language errors
                elif "encoding" in error:
                    if embedding_model:
                        try:
                            query_encoding = embedding_model.encode(query)
                        except Exception as e:
                            structlog.get_logger().error(f"Failed to encode query: {e}", exc_info=True)
                            query_encoding = ""  # Fallback
                        query = await decode_encoding(query_encoding)
                    else:
                        structlog.get_logger().error("Cannot encode query: Embedding model not loaded.")

                elif "classifier" in error:
                    classifier = await load_fallback_classifier()
                    query = await preprocess_with_classifier(query, classifier)
                elif "sentiment_analyzer" in error:
                    sentiment_analyzer = await load_fallback_sentiment_analyzer()
                    query = await preprocess_with_sentiment(query, sentiment_analyzer)
                else:
                    query = await apply_general_error_correction(query, error)

                wait_time = initial_wait * (backoff_factor ** attempt)
                structlog.get_logger().info(f"Waiting for {wait_time} seconds before next retry.")
                await asyncio.sleep(wait_time)
            else:
                structlog.get_logger().info(f"Process completed successfully. Elapsed time: {elapsed_time}")
                PROCESSING_TIME.labels(function="advanced_reasoning").observe(elapsed_time)
                return response_text, elapsed_time, None

        except asyncio.TimeoutError:
            ERROR_COUNTER.labels(error_type="timeout").inc()
            structlog.get_logger().error(f"Timeout during reasoning attempt {attempt + 1}. Retrying.")
            error = "Timeout occurred during advanced reasoning."

        except Exception as e:
            ERROR_COUNTER.labels(error_type=type(e).__name__).inc()
            structlog.get_logger().error(f"Unexpected error in reasoning attempt {attempt + 1}. Error: {str(e)}", exc_info=True)
            error = str(e)

    total_elapsed_time = time.time() - start_time
    structlog.get_logger().error(f"Error could not be resolved after {max_retries} attempts. Error: {error}")
    PROCESSING_TIME.labels(function="self_reflect_and_fix_errors").observe(total_elapsed_time)
    statsd.increment('self_reflect_and_fix_errors.failure')
    return f"An error occurred during self-reflection and error fixing: {error}", total_elapsed_time, error



DetectorFactory.seed = 0

def detect_language(text):
    try:
        return langdetect.detect(text)
    except langdetect.lang_detect_exception.LangDetectException:
        return "en"
    
ddg_semaphore = Semaphore(1)


def calculate_readability_score(text):
    try:
        flesch_kincaid = textstat.flesch_kincaid_grade(text)
        gunning_fog = textstat.gunning_fog(text)
        smog = textstat.smog_index(text)
        dale_chall = textstat.dale_chall_readability_score(text)
        
        readability_composite = (flesch_kincaid + gunning_fog + smog + dale_chall) / 4
        
        return {
            'composite_score': readability_composite,
            'flesch_kincaid': flesch_kincaid,
            'gunning_fog': gunning_fog,
            'smog': smog,
            'dale_chall': dale_chall
        }
    except Exception as e:
        structlog.get_logger().error(f"Readability score calculation failed: {e}", exc_info=True)
        return {
            'composite_score': 0,
            'flesch_kincaid': 0,
            'gunning_fog': 0,
            'smog': 0,
            'dale_chall': 0
        }


def calculate_topic_relevance(text, query):
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        corpus = [text, query]
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return {
            'relevance_score': cosine_sim
        }
    except Exception as e:
        structlog.get_logger().error(f"Topic relevance calculation failed: {e}", exc_info=True)
        return {'relevance_score': 0}


async def estimate_content_freshness(url, text):
    days_since_modification = None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.head(url, timeout=10) as response:
                last_modified = response.headers.get('Last-Modified')
                if last_modified:
                    try:
                        last_modified_date = datetime.strptime(last_modified, '%a, %d %b %Y %H:%M:%S GMT')
                        days_since_modification = (datetime.now() - last_modified_date).days
                    except ValueError:
                        structlog.get_logger().warning(f"Invalid Last-Modified header format: {last_modified}")
    except Exception as e:
        structlog.get_logger().warning(f"Failed to fetch Last-Modified header: {e}")
    
    freshness_score = 1 / (1 + days_since_modification) if days_since_modification else 0.5
    
    return {
        'freshness_score': freshness_score,
        'days_since_modification': days_since_modification
    }


async def evaluate_source_authority(url):
    try:
        domain = tldextract.extract(url).domain + '.' + tldextract.extract(url).suffix
    except Exception as e:
        structlog.get_logger().error(f"Error extracting domain: {e}", url=url, exc_info=True)
        return {'authority_score': 0}

    https_score = 1 if url.startswith('https') else 0
    
    authority_score = https_score
    
    return {
        'authority_score': authority_score
    }


def log_response(response: requests.Response):
    logging.info(f"Response Status Code: {response.status_code}")
    logging.debug(f"Response Content: {response.text}")

def parse_search_results(results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    parsed_results = []
    
    for item in results:
        try:
            title = item.get('title')
            link = item.get('link')
            snippet = item.get('snippet')
            
            if not all([title, link, snippet]):
                raise ValueError("Missing result components")

            parsed_results.append({
                'url': link,
                'title': title,
                'snippet': snippet
            })

        except Exception as e:
            logging.warning(f"Error parsing search result: {e}, skipping this result.")

    return parsed_results


async def advanced_multi_source_search(query: str, language: str = None, num_sites: int = 10, timeout: int = 30) -> Dict[str, Any]:
    start_time = datetime.now()
    logging.info(f"Starting advanced multi-source search for query: '{query}', language: '{language}', num_sites: {num_sites}, timeout: {timeout}")

    search_results = []
    total_results = 0
    summary = ""


    async def fetch_from_google(url):
        nonlocal total_results, search_results
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url)
                log_response(response)

                if response.status_code == 200:
                    data = response.json()
                    items = data.get('items', [])
                    parsed_results = parse_search_results(items)
                    search_results.extend(parsed_results)
                    total_results += len(parsed_results)
                    logging.info(f"Retrieved {len(parsed_results)} results from API.")
                    
                    if len(parsed_results) < 10:
                        logging.info("Less than 10 results returned, stopping further requests.")
                        return True # Signal to stop
                else:
                    logging.error(f"API request failed with status code: {response.status_code}")
                    logging.debug(f"Response content: {response.text}")

        except httpx.RequestError as e:
            logging.exception(f"Request failed: {e}")
        return False # Continue to next batch


    for start_index in range(0, num_sites, 10):
        url = f"https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={CSE_ID}&q={query}&start={start_index}"
        if language:
            url += f"&hl={language}"
        
        logging.debug(f"Fetching URL: {url}")

        should_stop = await fetch_from_google(url)
        if should_stop:
            break


    if search_results:
        try:
            top_results_text = "\n".join([f"Title: {r['title']}\nSnippet: {r['snippet']}\nURL: {r['url']}\n" for r in search_results[:3]])
            summary_prompt = f"Summarize the following search results:\n\n{top_results_text}"
            summary, _, _ = await advanced_reasoning_with_groq(summary_prompt, api_key=GROQ_API_KEY, timeout=timeout)


        except Exception as e:
            structlog.get_logger().error(f"Failed to generate summary: {e}", exc_info=True)
    else:
        structlog.get_logger().warning("No search results to summarize.")



    execution_time = (datetime.now() - start_time).total_seconds()
    result = {
        'query': query,
        'total_results': total_results,
        'top_results': search_results[:num_sites],
        'execution_time': execution_time,
        'summary': summary
    }
    logging.info(f"Search completed in {execution_time:.2f} seconds. Total results: {total_results}")
    logging.debug(f"Search results: {result}")
    return result


def get_user_input():
    query = input("Enter search query: ")
    language = input("Enter language (default: en): ") or "en"
    num_sites = input("Enter number of sites to search (default: 30): ")
    num_sites = int(num_sites) if num_sites.isdigit() else 30

    timeout = input("Enter timeout in seconds (default: 30): ")
    timeout = int(timeout) if timeout.isdigit() else 30
    
    return query, language, num_sites, timeout

async def async_main():
    query, language, num_sites, timeout = get_user_input()
    await advanced_multi_source_search(query, language, num_sites, timeout)



async def check_topic_continuity(user_id, current_topic):
    recent_topics = user_profiles[user_id].get("recent_topics", [])
    if recent_topics and recent_topics[-1] == current_topic:
        return True, "Continuing the previous topic"
    elif recent_topics:
        return False, f"Switching from {recent_topics[-1]} to {current_topic}"
    return False, "Starting a new topic"

async def groq_search_and_summarize(query, language="en", num_sites=10, timeout=30):
    try:
        search_results = await advanced_multi_source_search(query, language=language, num_sites=num_sites, timeout=timeout)

        if search_results and search_results['top_results']:
            summarized_results = []
            for result in search_results['top_results'][:3]:
                summary = f"Title: {result['title']}\nURL: {result['url']}\nSummary: {result['snippet']}\n"
                summarized_results.append(summary)
            
            summarized_text = "\n".join(summarized_results)
            prompt = f"Summarize these search results and extract the most important, up-to-date information:\n\n{summarized_text}"
            
            response, _, _ = await advanced_reasoning_with_groq(prompt, api_key=GROQ_API_KEY, timeout=timeout)
            return response
        return "No relevant up-to-date information found in the search results."


    except Exception as e:
        structlog.get_logger().error("groq_search_and_summarize failed:", exc_info=True, error=str(e))
        return "An error occurred during search and summarization of up-to-date information."


async def extract_url_from_description(description):
    search_query = f"{description} site:youtube.com OR site:twitch.tv OR site:instagram.com OR site:twitter.com"
    try:
        async with ddg_throttler: # Using the throttler here
            ddg = AsyncDDGS()
            results = await ddg.json(search_query, max_results=1)
            if results and results['results']:
                return results['results'][0]['href']
    except Exception as e:
        structlog.get_logger().error(f"Failed to extract URL from description: {e}", exc_info=True)
    return None

async def clean_url(url, description=None):
    if url is None:
        return None

    cleaned_url = url.lower().strip()

    if not cleaned_url.startswith(("https://", "http://")):
        cleaned_url = "https://" + cleaned_url

    cleaned_url = re.sub(r"[^a-zA-Z0-9./?=-]", "", cleaned_url)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.head(cleaned_url, timeout=10) as response: # Using head request for faster check

                if response.status == 200:
                    return cleaned_url
                elif description:
                    better_url = await extract_url_from_description(description)
                    if better_url:
                        return better_url
    except aiohttp.ClientError as e: # Handling client errors
        structlog.get_logger().error(f"Error validating URL: {e}", cleaned_url=cleaned_url, exc_info=True)

    return None


async def complex_dialogue_manager(user_profiles, user_id, message):
    try:
        profile = user_profiles.get(user_id)
        if not profile or profile["dialogue_state"] != "planning":
            return "Dialogue is not in planning mode."
        planning_state = profile.setdefault("planning_state", {})
        stage = planning_state.get("stage")

        if stage == "initial_request":
            goal, query_type = await extract_goal(profile["query"])
            planning_state["goal"] = goal
            planning_state["query_type"] = query_type
            planning_state["stage"] = "gathering_information"
            return await ask_clarifying_questions(goal, query_type)

        elif stage == "gathering_information":
            await process_planning_information(user_id, message)
            if await has_enough_planning_information(user_id):
                planning_state["stage"] = "generating_plan"
                plan = await generate_plan(planning_state["goal"], planning_state.get("preferences", {}), user_id,
                                           message)

                is_valid, validation_result = await validate_plan(plan, user_id)
                if is_valid:
                    planning_state["plan"] = plan
                    planning_state["stage"] = "presenting_plan"
                    return await present_plan_and_ask_for_feedback(plan)

                planning_state["stage"] = "gathering_information" # Revert to information gathering if plan is invalid
                return f"The plan has some issues: {validation_result}. Please provide more information or adjust your preferences."


        elif stage == "presenting_plan":
            feedback_result = await process_plan_feedback(user_id, message.text)
            if feedback_result == "accept":
                planning_state["stage"] = "evaluating_plan"
                evaluation = await evaluate_plan(planning_state["plan"], user_id)
                planning_state["evaluation"] = evaluation
                planning_state["stage"] = "executing_plan"
                initial_execution_message = await execute_plan_step(planning_state["plan"], 0, user_id, message)
                return (await generate_response(planning_state["plan"], evaluation, {},
                                                   planning_state.get("preferences", {}))) + "\n\n" + initial_execution_message
            planning_state["stage"] = "gathering_information" # Revert if plan is not accepted
            return f"Okay, let's revise the plan. Here are some suggestions: {feedback_result}. What changes would you like to make?"


        elif stage == "executing_plan":
            execution_result = await monitor_plan_execution(planning_state["plan"], user_id, message)
            return execution_result

        else:
            return "Invalid planning stage."
    except Exception as e:
        structlog.get_logger().error("complex_dialogue_manager failed:", exc_info=True, error=str(e))
        return f"An error occurred: {e}"

async def ask_clarifying_questions(goal, query_type):
    return "To create an effective plan, I need some more details. Could you tell me:\n- What is the desired outcome of this plan?\n- What are the key steps or milestones involved?\n- Are there any constraints or limitations I should be aware of?\n- What resources or tools are available?\n- What is the timeline for completing this plan?"

async def process_planning_information(user_id, message):
    user_profiles[user_id]["planning_state"]["preferences"]["user_input"] = message.text

async def has_enough_planning_information(user_id):
    return "user_input" in user_profiles[user_id]["planning_state"]["preferences"]

async def ask_further_clarifying_questions(user_id):
    return "Please provide more details to help me create a better plan. For example, more information about steps, constraints, resources, or the time frame."

async def present_plan_and_ask_for_feedback(plan):
    plan_text = "".join([f"{i + 1}. {step['description']}\n" for i, step in enumerate(plan["steps"])])
    return f"Based on your input, here's a draft plan:\n\n{plan_text}\n\nWhat do you think? Are there any changes you would like to make? (Type 'accept' to proceed)"

async def generate_response(plan, evaluation, additional_info, preferences):
    response = f"I've created a plan for your goal: {plan['goal']}\n\n"
    response += "Steps:\n"
    response += "".join(
        [f"{i + 1}. {step['description']}" + (" (Deadline: " + step["deadline"] + ")" if "deadline" in step else "") + "\n" for i, step in enumerate(plan["steps"])])
    if evaluation:
        response += f"\nEvaluation:\n{evaluation.get('evaluation_text', '')}\n"
    if additional_info:
        response += "\nAdditional Information:\n"
        response += "".join([f"- {info_type}: {info}\n" for info_type, info in additional_info.items()])
    if preferences:
        response += "\nYour Preferences:\n"
        response += "".join(
            [f"- {preference_name}: {preference_value}\n" for preference_name, preference_value in preferences.items()])
    return response

async def extract_goal(query):
    prompt = f"You are an AI assistant capable of understanding user goals. What is the user trying to achieve with the following query? User Query: {query} Please specify the goal in a concise sentence."
    try:
        goal, _, _ = await advanced_reasoning_with_groq(prompt, api_key=GROQ_API_KEY)
        return goal.strip(), "general"
    except Exception as e:
        structlog.get_logger().error(f"Failed to extract goal: {e}", exc_info=True)
        return "Unknown goal", "general"

async def execute_plan_step(plan, step_index, user_id, message):
    try:
        step = plan["steps"][step_index]
        execution_prompt = f"You are an AI assistant helping a user carry out a plan. Here is the plan step: {step['description']} The user said: {message.text} If the user's message indicates they are ready to proceed with this step, provide a simulated response as if they completed it. If the user requests clarification or changes, accept their request and provide helpful information or guidance. Be specific and relevant to the plan step."
        execution_response, _, _ = await advanced_reasoning_with_groq(execution_prompt, user_id=user_id, api_key=GROQ_API_KEY)
        step["status"] = "in_progress"
        await store_long_term_memory(user_id, "plan_execution_result",
                                     {"step_description": step["description"], "result": "in_progress",
                                      "timestamp": datetime.now(timezone.utc).isoformat()})
        return execution_response
    except Exception as e:
        structlog.get_logger().error("execute_plan_step failed:", exc_info=True, error=str(e))
        return f"An error occurred while executing the plan step: {e}"

async def monitor_plan_execution(plan, user_id, message):
    try:
        current_step_index = next((i for i, step in enumerate(plan["steps"]) if step.get("status") == "in_progress"), None)
        if current_step_index is not None:
            if "done" in message.text.lower() or "completed" in message.text.lower() or "tamamlandı" in message.text.lower() or "bitti" in message.text.lower():
                plan["steps"][current_step_index]["status"] = "completed"
                await bot.send_message(chat_id=message.chat_id, text=f"Great! Step {current_step_index + 1} has been completed.")
                if current_step_index + 1 < len(plan["steps"]):
                    next_step_response = await execute_plan_step(plan, current_step_index + 1, user_id, message)
                    return f"Moving on to the next step: {next_step_response}"
                return "Congratulations! You have completed all the steps in the plan."

        if current_step_index is None: # Start the first step if no step is in progress.
            current_step_index = 0
        return await execute_plan_step(plan, current_step_index, user_id, message) # Changed to handle the first step as well.
    except Exception as e:
        structlog.get_logger().error("monitor_plan_execution failed:", exc_info=True, error=str(e))
        return f"An error occurred while monitoring plan execution: {e}"

async def generate_plan(goal, preferences, user_id, message):
    try:
        planning_prompt = f"You are an AI assistant specialized in planning. A user needs help with the following goal: {goal} What the user said about the plan: {preferences.get('user_input')} Based on this information, create a detailed and actionable plan by identifying key steps and considerations. Ensure the plan is: * Specific: Each step should be clearly defined. * Measurable: Add ways to track progress. * Achievable: Steps should be realistic and actionable. * Relevant: Align with the user's goal. * Time-bound: Include estimated timelines or deadlines. Analyze potential risks and dependencies for each step. Format the plan as a JSON object: json {{ 'goal': 'User's goal', 'steps': [ {{ 'description': 'Step description', 'deadline': 'Optional deadline for the step', 'dependencies': ['List of dependencies (other step descriptions)'], 'risks': ['List of potential risks'], 'status': 'waiting' }}, // ... more steps ], 'preferences': {{ // User preferences related to the plan }} }}"
        plan_text, _, _ = await advanced_reasoning_with_groq(planning_prompt, user_id=user_id, api_key=GROQ_API_KEY)
        try:
            plan = json.loads(plan_text)
            for step in plan.get("steps", []): # Ensure each step has a status
                if "status" not in step:
                    step["status"] = "waiting"
        except json.JSONDecodeError:
            structlog.get_logger().error("Invalid JSON returned from GROQ", plan_text=plan_text, exc_info=True)
            return {"goal": goal, "steps": [], "preferences": preferences}
        except Exception as e:
            structlog.get_logger().error(f"Error processing plan JSON: {e}", exc_info=True)
            return {"goal": goal, "steps": [], "preferences": preferences}


        await store_long_term_memory(user_id, "plan", plan)
        return plan
    except Exception as e:
        structlog.get_logger().error("generate_plan failed:", exc_info=True, error=str(e))
        return f"An error occurred while generating the plan: {e}"

async def evaluate_plan(plan, user_id):
    try:
        evaluation_prompt = f"You are an AI assistant tasked with evaluating a plan, including identifying potential risks and dependencies. Here is the plan: Goal: {plan['goal']} Steps: {json.dumps(plan['steps'], indent=2)} Evaluate this plan based on the following criteria: * Feasibility: Is the plan realistically achievable? * Completeness: Does the plan cover all necessary steps? * Efficiency: Is the plan optimally structured? Are there unnecessary or redundant steps? * Risks: Analyze the risks identified for each step. Are they significant? How can they be mitigated? * Dependencies: Are the dependencies between steps clear and well defined? Are there potential conflicts or bottlenecks? * Improvements: Suggest any improvements or alternative approaches considering the risks and dependencies. Provide a structured evaluation summarizing your assessment for each criterion. Be as specific as possible in your analysis."
        evaluation_text, _, _ = await advanced_reasoning_with_groq(evaluation_prompt, user_id=user_id, api_key=GROQ_API_KEY)
        await store_long_term_memory(user_id, "plan_evaluation", evaluation_text)
        return {"evaluation_text": evaluation_text}
    except Exception as e:
        structlog.get_logger().error("evaluate_plan failed:", exc_info=True, error=str(e))
        return {"evaluation_text": f"An error occurred while evaluating the plan: {e}"}

async def validate_plan(plan, user_id):
    try:
        validation_prompt = f"You are an AI assistant specialized in evaluating the feasibility and safety of plans. Carefully analyze the following plan and identify any potential issues, flaws, or missing information that could lead to failure or undesirable outcomes. Goal: {plan['goal']} Steps: {json.dumps(plan['steps'], indent=2)} Consider the following points: * Clarity and Specificity: Are the steps clear and specific enough to be actionable? * Realism and Feasibility: Are the steps realistic and achievable considering the user's context and resources? * Dependencies: Are the dependencies between steps clearly stated and logical? Are there cyclic dependencies? * Time Constraints: Are the deadlines realistic and achievable? Are there potential time conflicts? * Resource Availability: Are the necessary resources available for each step? * Risk Assessment: Are potential risks sufficiently identified and analyzed? Are there mitigation strategies? * Safety and Ethics: Does the plan comply with safety and ethical standards? Are there potential negative outcomes? Provide a detailed analysis of the plan highlighting any weaknesses or areas for improvement. Indicate if the plan is solid and well-structured, or provide specific recommendations for making it more robust and effective."
        validation_result, _, _ = await advanced_reasoning_with_groq(validation_prompt, user_id=user_id, api_key=GROQ_API_KEY)

        is_valid = "valid" in validation_result.lower() or "no issues" in validation_result.lower() # More flexible validation check.

        return is_valid, validation_result
    except Exception as e:
        structlog.get_logger().error("validate_plan failed:", exc_info=True, error=str(e))
        return False, f"An error occurred while validating the plan: {e}"


async def process_plan_feedback(user_id, message):
    try:
        feedback_prompt = f"You are an AI assistant analyzing user feedback on a plan. The user said: {message} Is the user accepting the plan? Respond with 'ACCEPT' if yes. If no, identify parts of the plan the user wants to change and suggest how the plan might be revised."
        feedback_analysis, _, _ = await advanced_reasoning_with_groq(feedback_prompt, user_id=user_id, api_key=GROQ_API_KEY)
        if "accept" in feedback_analysis.lower():
            return "accept"
        return feedback_analysis
    except Exception as e:
        structlog.get_logger().error("process_plan_feedback failed:", exc_info=True, error=str(e))
        return f"An error occurred while processing plan feedback: {e}"

user_message_buffer = defaultdict(list)


async def identify_user_interests(user_id, message):
    user_message_buffer[user_id].append(message)
    if len(user_message_buffer[user_id]) >= 5:
        messages = user_message_buffer[user_id]
        user_message_buffer[user_id] = []

        if embedding_model:
            try:
                embeddings = np.array([embedding_model.encode(message) for message in messages]).astype('float32')

                num_topics = 3
                kmeans = KMeans(n_clusters=num_topics)
                kmeans.fit(embeddings)
                clusters = defaultdict(list)
                for i, label in enumerate(kmeans.labels_):
                    clusters[label].append(messages[i])
                topics = [random.choice(cluster) for cluster in clusters.values()]
                for i, topic in enumerate(topics):
                    user_profiles[user_id]["interests"].append({"message": topic, "embedding": embeddings[i].tolist(), "topic": i})

                save_user_profiles()
            except Exception as e:
                structlog.get_logger().error(f"Failed to identify user interests: {e}", exc_info=True)

        else:
            structlog.get_logger().warning("Cannot identify user interests: Embedding model not loaded.")



async def suggest_new_topic(user_id):
    if user_profiles[user_id]["interests"]:
        interests = user_profiles[user_id]["interests"]
        topic_counts = defaultdict(int)
        for interest in interests:
            topic_counts[interest["topic"]] += 1
        most_frequent_topic = max(topic_counts, key=topic_counts.get)
        suggested_interest = random.choice([interest for interest in interests if interest["topic"] == most_frequent_topic])
        return f"Hey, maybe we could talk more about '{suggested_interest['message']}'? I'd love to hear your thoughts."
    return "I'm not sure what to talk about next. What are you interested in?"


def save_user_profiles():
    try:
        profiles_copy = defaultdict(lambda: {
            "preferences": {"communication_style": "friendly", "topics_of_interest": []},
            "demographics": {"age": None, "location": None},
            "history_summary": "",
            "context": [],
            "personality": DEFAULT_PERSONALITY.copy(),
            "dialogue_state": "greeting",
            "long_term_memory": [],
            "last_bot_action": None,
            "interests": [],
            "query": "",
            "planning_state": {},
            "interaction_history": [],
            "recent_topics": [],
            "current_mood": "neutral",
            "goals": []
        })

        for user_id, profile in user_profiles.items():
            profiles_copy[user_id].update(profile)

            profiles_copy[user_id]["context"] = list(profile["context"])  # Convert deque to list for JSON serialization

            for interest in profiles_copy[user_id]["interests"]:
                if isinstance(interest.get("embedding"), np.ndarray):
                    interest["embedding"] = interest["embedding"].tolist()
                if isinstance(interest.get("topic"), np.int32):
                    interest["topic"] = int(interest["topic"])

        with open(USER_PROFILES_FILE, "w", encoding="utf-8") as f:
            json.dump(profiles_copy, f, indent=4, ensure_ascii=False)
    except Exception as e:
        structlog.get_logger().error("save_user_profiles failed:", exc_info=True)



async def load_user_profiles():
    try:
        with open(USER_PROFILES_FILE, "r", encoding="utf-8") as f:
            loaded_profiles = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        structlog.get_logger().warning(f"Failed to load user profiles, starting with empty profiles: {e}")
        loaded_profiles = {}  # Start with an empty dictionary if loading fails

    profiles = defaultdict(lambda: {
        "preferences": {"communication_style": "friendly", "topics_of_interest": []},
        "demographics": {"age": None, "location": None},
        "history_summary": "",
        "context": deque(maxlen=CONTEXT_WINDOW_SIZE),
        "personality": DEFAULT_PERSONALITY.copy(),
        "dialogue_state": "greeting",
        "long_term_memory": [],
        "last_bot_action": None,
        "interests": [],
        "query": "",
        "planning_state": {},
        "interaction_history": [],
        "recent_topics": [],
        "current_mood": "neutral",
        "goals": []
    })
    for user_id, profile in loaded_profiles.items():
        profiles[user_id].update(profile)
        profiles[user_id]["context"] = deque(profile.get("context", []), maxlen=CONTEXT_WINDOW_SIZE)  # Load context as deque

    return profiles


user_profiles = defaultdict(lambda: {
    "preferences": {"communication_style": "friendly", "topics_of_interest": []},
    "demographics": {"age": None, "location": None},
    "history_summary": "",
    "context": deque(maxlen=CONTEXT_WINDOW_SIZE),
    "personality": DEFAULT_PERSONALITY.copy(),
    "dialogue_state": "greeting",
    "long_term_memory": [],
    "last_bot_action": None,
    "interests": [],
    "query": "",
    "planning_state": {},
    "interaction_history": [],
    "recent_topics": [],
    "current_mood": "neutral",
    "goals": []
})


DIALOGUE_STATES = ["greeting", "question_answering", "storytelling", "general_conversation", "planning", "farewell"]
BOT_ACTIONS = ["factual_response", "creative_response", "clarifying_question", "change_dialogue_state",
               "initiate_new_topic", "generate_plan", "execute_plan"]


async def create_tables():
    try:
        async with aiosqlite.connect(DATABASE_FILE) as db:
            await db.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY,
            user_id TEXT,
            message TEXT,
            timestamp TEXT
            )
            """)
            await db.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY,
            user_id TEXT,
            feedback TEXT,
            timestamp TEXT
            )
            """)
            await db.commit()
    except Exception as e:
        structlog.get_logger().error("create_tables failed:", exc_info=True)


async def init_db():
    global db_ready
    try:
        async with db_lock:
            await create_tables()
            db_ready = True
    except Exception as e:
        structlog.get_logger().error("Database initialization failed:", exc_info=True)


async def save_chat_history(user_id, message):
    try:
        await db_queue.put((user_id, message))
    except Exception as e:
        structlog.get_logger().error("Failed to save chat history:", exc_info=True)


async def process_db_queue():
    while not shutdown_event.is_set():
        while not db_ready:
            await asyncio.sleep(1)
        try:
            user_id, message = await db_queue.get()
            if faiss_index is None:
                await load_faiss_index()
            async with db_lock:
                async with aiosqlite.connect(DATABASE_FILE) as db:
                    await db.execute(
                        "INSERT INTO chat_history (user_id, message, timestamp) VALUES (?, ?, ?)",
                        (user_id, message, datetime.now(timezone.utc).isoformat()))
                    await db.commit()
                    await add_to_faiss_index(message)
            db_queue.task_done()
        except asyncio.CancelledError: # Correctly handling CancelledError

            break

        except Exception as e:
            structlog.get_logger().error("Error processing DB queue:", exc_info=True)

        finally:
            if not db_queue.empty():
                continue
            await asyncio.sleep(2)  # Introduce a delay to avoid busy-waiting


async def save_feedback_to_db(user_id, feedback):
    try:
        async with db_lock:
            async with aiosqlite.connect(DATABASE_FILE) as db:
                await db.execute("INSERT INTO feedback (user_id, feedback, timestamp) VALUES (?, ?, ?)",
                                 (user_id, feedback, datetime.now(timezone.utc).isoformat()))
                await db.commit()
    except Exception as e:
        structlog.get_logger().error("Failed to save feedback:", exc_info=True)



async def get_relevant_history(user_id, current_message):
    try:
        async with db_lock:
            async with aiosqlite.connect(DATABASE_FILE) as db:
                async with db.execute(
                    "SELECT message FROM chat_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT 50",
                    (user_id,)) as cursor:
                    history = [row[0] for row in await cursor.fetchall()]
                if history:
                    history.reverse()
                    history_text = "\n".join(history)
                    try:
                        tfidf_matrix = tfidf_vectorizer.fit_transform([history_text, current_message])
                        similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
                        if similarity > 0.1:
                            return history_text
                    except ValueError as e: # Handle empty vocabulary
                        structlog.get_logger().warning(f"TF-IDF failed: {e}")
        return ""
    except Exception as e:
        structlog.get_logger().error("get_relevant_history failed:", exc_info=True)
        return ""

async def get_recent_topics(user_id, num_topics=5):
    try:
        async with db_lock:
            async with aiosqlite.connect(DATABASE_FILE) as db:
                async with db.execute(
                    "SELECT message FROM chat_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT 50",
                    (user_id,)) as cursor:
                    recent_messages = [row[0] for row in await cursor.fetchall()]

                if embedding_model and len(recent_messages) >= num_topics:

                    embeddings = embedding_model.encode(recent_messages)
                    num_clusters = min(num_topics, len(embeddings))
                    clustering_model = KMeans(n_clusters=num_clusters)
                    clustering_model.fit(embeddings)
                    clusters = defaultdict(list)
                    for i, label in enumerate(clustering_model.labels_):
                        clusters[label].append(recent_messages[i])
                    topics = [random.choice(cluster) for cluster in clusters.values()]
                    return topics
                elif not embedding_model:
                    structlog.get_logger().error("Embedding Model not found.")

        return recent_messages # returns recent messages if embedding model is not available or not enough messages
    except Exception as e:
        structlog.get_logger().error("get_recent_topics failed:", exc_info=True)
        return []



async def load_faiss_index():
    global faiss_index
    try:
        if os.path.exists(FAISS_INDEX_FILE):
            faiss_index = faiss.read_index(FAISS_INDEX_FILE)
        else:
            if embedding_model:
                faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
            else:
                structlog.get_logger().error("Cannot create FAISS index: Embedding model not loaded.")
                raise RuntimeError("Embedding model required for FAISS index")

    except (RuntimeError, OSError) as e:
        structlog.get_logger().error("load_faiss_index failed:", exc_info=True)
        if embedding_model:
            faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
        else:
            raise RuntimeError("Embedding model required for FAISS index") from e


async def add_to_faiss_index(text):
    if not embedding_model or not faiss_index:

        if not embedding_model:
            structlog.get_logger().warning("Cannot add to FAISS index: Embedding model not loaded.")
        if not faiss_index:
            structlog.get_logger().warning("Cannot add to FAISS index: FAISS index not loaded.")
        return

    try:

        embedding = embedding_model.encode(text)

        faiss_index.add(np.array([embedding]).astype('float32'))
        faiss.write_index(faiss_index, FAISS_INDEX_FILE)
        structlog.get_logger().info("Successfully added to FAISS index")

    except Exception as e:
        structlog.get_logger().error("add_to_faiss_index failed:", exc_info=True)



async def get_related_memories(user_id, query, top_k=3):
    if not embedding_model or not faiss_index:

        if not embedding_model:
            structlog.get_logger().warning("Cannot get related memories: Embedding model not loaded.")
        if not faiss_index:
            structlog.get_logger().warning("Cannot get related memories: FAISS index not loaded.")
        return []

    try:

        query_embedding = embedding_model.encode(query)
        D, I = faiss_index.search(np.array([query_embedding]).astype('float32'), top_k)
        related_memories = []

        for i in range(top_k):
            try:
                index = I[0][i]
                memory_node = knowledge_graph.get_node("memory", str(index))
                if memory_node:
                    related_memories.append(memory_node["information"])
            except IndexError:
                break  # Handles cases where fewer than top_k results are found

        return related_memories
    except Exception as e:
        structlog.get_logger().error("get_related_memories failed:", exc_info=True)
        return []

async def analyze_sentiment(text):
    try:
        scores = sentiment_analyzer.polarity_scores(text)
        return scores['compound']
    except Exception as e:
        structlog.get_logger().error("analyze_sentiment failed:", exc_info=True)
        return 0.0


start_time = time.time()



async def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "en"

async def update_persistent_models(query, response, sentiment_score, topic, classifier, sentiment_analyzer):
    if not classifier or not sentiment_analyzer:
        structlog.get_logger().warning("Cannot update models: Classifier or sentiment analyzer not loaded.")
        return
    try:

        combined_text = f"{query} {response}"
        if embedding_model:
            embedding = embedding_model.encode(combined_text)
            embedding_tensor = torch.tensor(embedding).float().unsqueeze(0)
        else:
            structlog.get_logger().error("Cannot update models: Embedding model not loaded.")
            return


        classifier.train()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        topic_tensor = torch.tensor([topic], dtype=torch.long)
        output = classifier(embedding_tensor)
        loss = criterion(output, topic_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sentiment_analyzer.train()
        sentiment_optimizer = torch.optim.Adam(sentiment_analyzer.parameters(), lr=0.001)
        sentiment_criterion = nn.MSELoss()
        sentiment_output = sentiment_analyzer(embedding_tensor)
        sentiment_loss = sentiment_criterion(sentiment_output.squeeze(), torch.tensor([sentiment_score]).float())
        sentiment_optimizer.zero_grad()
        sentiment_loss.backward()
        sentiment_optimizer.step()

        torch.save(classifier.state_dict(), 'advanced_classifier_model.pth') # Save to correct path.
        torch.save(sentiment_analyzer.state_dict(), 'advanced_sentiment_model.pth') # Save to correct path.
        await add_to_faiss_index(combined_text)
        if faiss_index: # Write to faiss index only if available
            faiss.write_index(faiss_index, FAISS_INDEX_FILE)


        structlog.get_logger().info("Updated and saved persistent models",
                                   classifier_loss=loss.item(),
                                   sentiment_loss=sentiment_loss.item())
    except Exception as e:
        structlog.get_logger().error("Failed to update persistent models:", exc_info=True)


async def update_language_model(query, response, sentiment_score):
    try:
        combined_text = f"{query} {response}"
        await add_to_faiss_index(combined_text)

        if embedding_model:
            topic_vector = embedding_model.encode(combined_text)
            structlog.get_logger().info("Updated topic classification model", vector=topic_vector)
        else:
            structlog.get_logger().warning("Cannot update topic model: Embedding model not loaded.")

        structlog.get_logger().info("Updated sentiment analysis model", score=sentiment_score)
    except Exception as e:
        structlog.get_logger().error("Failed to update language model:", exc_info=True)


def update_personality(personality, sentiment_score):
    if sentiment_score > 0.5:
        personality["kindness"] += 0.1
    elif sentiment_score < -0.5:
        personality["assertiveness"] += 0.1
    for trait in personality:
        personality[trait] = max(0, min(1, personality[trait]))


def identify_user_goals(query):
    goals = []
    learning_keywords = ["learn", "study", "öğren", "çalış"]
    planning_keywords = ["plan", "planla"]
    if any(keyword in query.lower() for keyword in learning_keywords):
        goals.append("learning")
    if any(keyword in query.lower() for keyword in planning_keywords):
        goals.append("planning")
    return goals


async def keep_typing(chat_id):
    while True:
        await bot.send_chat_action(chat_id=chat_id, action="typing")
        await asyncio.sleep(3)





async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    start_time = time.time()
    log = structlog.get_logger()
    user_id = str(update.effective_user.id)
    message = update.message
    response_text = ""
    sentiment_score = 0.0
    current_topic = 0
    classifier = None
    sentiment_analyzer = None
    summarized_search_results = None  # Initialize to None
    typing_task = None
    
    try:
        if message is None or message.text is None:
            log.warning("Received an update without a text message.", update=update)
            return

        content = message.text.strip()

        # Start typing indicator.
        typing_task = asyncio.create_task(keep_typing(update.effective_chat.id))

        # User profile initialization or retrieval
        if user_id not in user_profiles:
            detected_language = await detect_language(content)
            user_profiles[user_id] = {
                "preferences": {"communication_style": "friendly", "topics_of_interest": []},
                "demographics": {"age": None, "location": None},
                "history_summary": "",
                "context": deque(maxlen=CONTEXT_WINDOW_SIZE),
                "personality": DEFAULT_PERSONALITY.copy(),
                "dialogue_state": "greeting",
                "long_term_memory": [],
                "last_bot_action": None,
                "interests": [],
                "query": "",
                "planning_state": {},
                "interaction_history": [],
                "recent_topics": [],
                "current_mood": "neutral",
                "goals": [],
                "preferred_language": detected_language
            }
        else:
            detected_language = user_profiles[user_id].get("preferred_language", "en")

        # Update the user context
        user_profiles[user_id]["context"].append({"role": "user", "content": content})
        user_profiles[user_id]["query"] = content

        await add_to_faiss_index(content)
        await identify_user_interests(user_id, content)
        relevant_history = await get_relevant_history(user_id, content)

        try: 
            # Attempt to search and summarize
            summarized_search_results = await advanced_multi_source_search(content, language=detected_language)
        except Exception as e:
            log.error(f"Search failed: {e}", exc_info=True)
            summarized_search_results = None  # Set to None if search fails

        summarized_search = summarized_search_results.get('summary', '') if summarized_search_results else "No internet access to get up-to-date information. Relying on internal knowledge."

        # Load classifier and sentiment analyzer
        classifier = await async_load_or_create_classifier()
        sentiment_analyzer = await async_load_or_create_sentiment_analyzer()

        # Embedding processing
        if embedding_model and encoder and projection_layer:
            query_embedding = projection_layer(torch.tensor([embedding_model.encode(content)]).float().unsqueeze(0))
            query_encoding = encoder(query_embedding)

            if classifier:
                topic_probs = F.softmax(classifier(query_encoding.mean(dim=1)), dim=1)
                current_topic = int(torch.argmax(topic_probs).item())
            else:
                log.warning("Classifier not available, using default topic.")
                current_topic = 0

            if sentiment_analyzer:
                sentiment_score = sentiment_analyzer(query_encoding.mean(dim=1)).item()
            else:
                log.warning("Sentiment analyzer not available, using default sentiment score.")
                sentiment_score = 0.0
        else:
            log.warning("Required models not loaded, skipping some features.")

        # Check for topic continuity and retrieve related memories
        is_continuous, continuity_message = await check_topic_continuity(user_id, current_topic)
        related_memories = await get_related_memories(user_id, content, top_k=5)
        update_personality(user_profiles[user_id]["personality"], sentiment_score)

        # Prompt construction for reasoning API
        prompt = f"""
        User Query: {content}
        Language: {detected_language}
        Search Results Summary: {summarized_search}
        Relevant History: {relevant_history}
        Related Memories: {related_memories}
        Current Topic: {current_topic}
        Topic Continuity: {continuity_message}
        User Personality: {user_profiles[user_id]["personality"]}
        Sentiment Score: {sentiment_score}

        As an AI assistant, analyze the given information and generate a response in {detected_language} that:
        1. Directly addresses the user's query with accuracy and relevance
        2. Incorporates the summarized search results to provide up-to-date information, or relies on internal knowledge if search fails.
        3. Maintains context and topic continuity based on the conversation history
        4. Incorporates relevant historical information and memories to provide a personalized response
        5. Adapts to the user's personality and current sentiment, adjusting the tone accordingly
        6. Ensures the response is coherent, well-structured, and easy to understand
        7. Avoids biases and considers multiple perspectives when applicable
        8. Offers additional relevant information or follow-up questions to encourage engagement

        Format your response entirely in {detected_language}, ensuring it's natural and conversational.
        """

        response_text, _, error_message = await advanced_reasoning_with_groq(prompt, api_key=GROQ_API_KEY, language=detected_language)

        if error_message:
            log.error(f"Groq API error: {error_message}", user_id=user_id, exc_info=True)
            response_text = "I encountered an issue while generating a response. Please try again later."  # Fallback response

        # Send the response
        await bot.send_message(chat_id=update.effective_chat.id, text=response_text, parse_mode="HTML", disable_web_page_preview=True)

        # Update chat history and save profile
        await save_chat_history(user_id, content)
        user_profiles[user_id]["context"].append({"role": "assistant", "content": response_text})
        save_user_profiles()

    except Exception as e:
        log.exception(f"handle_message failed", user_id=user_id, exc_info=True, error=str(e))
        try:
            await bot.send_message(chat_id=update.effective_chat.id, text="An unexpected error occurred. Please try again later.")
        except Exception as send_error:
            log.error(f"Failed to send error message: {send_error}", exc_info=True)

    finally:
        if typing_task:
            typing_task.cancel()
        try:
            await update_persistent_models(content, response_text, sentiment_score, current_topic, classifier, sentiment_analyzer)
            await update_language_model(content, response_text, sentiment_score)
        except Exception as e:
            log.error(f"Error in updating models: {e}", exc_info=True)

        elapsed_time = time.time() - start_time
        log.info(f"handle_message processed in {elapsed_time:.2f} seconds", user_id=user_id)

    return response_text


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    message = update.message

    if message and message.photo:
        try:
            file_id = message.photo[-1].file_id
            new_file = await context.bot.get_file(file_id)
            image_url = new_file.file_path
            image_analysis = await analyze_image(image_url)


            if image_analysis['success']:
                description = image_analysis['description']
                await bot.send_message(chat_id=update.effective_chat.id, text=f"I see: {description}")
                await save_image_to_memory(image_url, description, user_id)
            else:

                error_message = image_analysis.get('error', 'An unknown error occurred during image analysis.')
                await bot.send_message(chat_id=update.effective_chat.id, text=f"I couldn't analyze the image: {error_message}")


        except Exception as e:
            log.exception("Error processing image", exc_info=True, user_id=user_id, error=str(e))
            await bot.send_message(chat_id=update.effective_chat.id, text=f"An error occurred while processing the image. Please try again.")



async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    structlog.get_logger().error(f"Update caused error", update=update, error=context.error, exc_info=True)

async def start(update: Update, context):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Hello!")

async def echo(update: Update, context):
    await context.bot.send_message(chat_id=update.effective_chat.id, text=update.message.text)


async def handle_general_conversation(user_id):
    prompt = f"""You are a friendly and lovely protogen fox AI assistant engaging in general conversation."""
    try:
        response, _, _ = await advanced_reasoning_with_groq(prompt, api_key=GROQ_API_KEY)
        return response.strip()
    except Exception as e:
        structlog.get_logger().error(f"handle_general_conversation failed: {e}", exc_info=True)
        return "I'm having trouble engaging in conversation right now."


async def main():
    global application
    db_queue_task = None
    application = None  # Initialize application to None
    lock_file = acquire_lock_windows()
    if lock_file is None:
        log.info("Lock acquisition unsuccessful. This ensures only one instance runs at a time.")
        log.info("If you're certain no other instance is running, manually remove the lock file and retry.")
        return

    try:
        download_spacy_model() # Download the SpaCy model here
        await initialize_bot()


        if faiss_index is None and embedding_model:
            faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
            structlog.get_logger().info("Created new FAISS index because it was None.")

        elif faiss_index is None and not embedding_model:
            structlog.get_logger().error("Failed to create FAISS index: Embedding model not loaded.")
            raise RuntimeError("Embedding model required for FAISS index")


        db_queue_task = asyncio.create_task(process_db_queue())
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda sig=sig: asyncio.create_task(signal_handler(sig, loop)))


        application.add_handler(CommandHandler("start", start))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        application.add_handler(MessageHandler(filters.PHOTO, handle_image))
        application.add_error_handler(error_handler)

        await application.initialize()
        await application.start()


        await application.updater.start_polling(drop_pending_updates=True, timeout=120) # increased timeout
        log.info("Bot started successfully")



        await shutdown_event.wait()


    except Exception as e:
        log.exception("Critical error during bot operation", exc_info=True, error=str(e))
    finally:
        log.info("Bot operation completed")
        if db_queue_task:
            try:

                db_queue_task.cancel()
                await db_queue_task
            except asyncio.CancelledError:
                pass # This is expected.

        if application is not None:

            try:
                await application.stop()
                await application.shutdown()
            except Exception as e:
                log.error(f"Error during application shutdown: {e}", exc_info=True)

        release_lock_windows(lock_file)
        log.info("Shutdown complete")


if __name__ == '__main__':
    asyncio.run(main())