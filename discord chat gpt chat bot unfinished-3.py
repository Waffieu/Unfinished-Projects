import os
import discord
import random
import asyncio
import subprocess
from pathlib import Path
from PIL import Image
import pytesseract
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from dotenv import load_dotenv
import logging
import time
import urllib.request

load_dotenv()

DISCORD_TOKEN = "your-discord-token-here" # Store your token securely in an environment variable

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class DiscordBot:
    def __init__(self, token):
        self.client = discord.Client(intents=discord.Intents.default())
        self.token = token
        self.driver = None  # Driver starts as None
        self.first_load = True
        self.cached_response = ""

    async def on_ready(self):
        logging.info(f"Giriş yapıldı: {self.client.user}")

    async def on_message(self, message):
        if self.client.user.mentioned_in(message):
            question = message.content.replace(f"<@{self.client.user.id}>", "").strip()
            await message.channel.send("Cevap bekleniyor...")
            answer = await self.fetch_chatgpt_response(question)
            await message.channel.send(answer or "Cevabı alırken bir hata oluştu, lütfen tekrar deneyin.")

    async def fetch_chatgpt_response(self, question):
        if self.driver is None:
            self.initialize_driver()  # Initialize the driver if it is None
        
        try:
            self.driver.get("https://komo.ai/search/")
            logging.debug("Navigating to Web Seach Ai page.")

            # Refresh the page only on the first load
            if self.first_load:
                await asyncio.sleep(random.uniform(2, 5))  # Random delay to mimic human behavior
                self.driver.refresh()
                logging.debug("Page refreshed on first load.")
                self.first_load = False  # Set the flag to False after refreshing

            # Wait for the input box to be interactable
            input_box = self.wait_for_element_to_be_interactable(By.TAG_NAME, "textarea")
            if not input_box:
                logging.error("Input box not interactable.")
                return "Cevap veremiyorum, lütfen daha sonra tekrar deneyin."

            # Send the question directly
            full_message = ("cevapının başına lütfen merhaba yaz ile birlikte cevapları "
                            "interneti araştırarak ver en son bilgileri ver ve bu yazıdan sonraki dile göre cevap ver "
                            "bu yazının devamındaki dili kullanarak cevap ver " + question)
            logging.debug("Sending message: %s", full_message)
            input_box.send_keys(full_message)
            input_box.send_keys(Keys.RETURN)

            # Wait for the response to load
            response_element = WebDriverWait(self.driver, 10).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, ".text-base"))
            )
            response = response_element.text
            logging.debug("Response received: %s", response)

            # Check if the response contains the fox or wolf emoji
            if "Merhaba" in response or "Merhaba!" in response:
                self.cached_response = response  # Cache the response
                logging.debug("Valid response found, returning it.")
                return response
            else:
                logging.warning("Invalid response, no valid emojis found.")
                return "Cevap geçersiz; lütfen başka bir konuda soru sorun."

        except Exception as e:
            logging.error("Error in fetch_chatgpt_response: %s", e)
            return "Cevabı alırken bir hata oluştu, lütfen tekrar deneyin."

    def initialize_driver(self):
        """Initialize the Selenium WebDriver."""
        options = webdriver.ChromeOptions()
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        logging.info("WebDriver initialized.")

    def wait_for_element_to_be_interactable(self, by, value):
        for _ in range(5):  # Try 5 times
            try:
                element = WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable((by, value)))
                logging.debug("Element found and is interactable.")
                return element
            except Exception as e:
                logging.error("Waiting for element to be interactable: %s", e)
                time.sleep(2)
        return None

    def run(self):
        self.client.event(self.on_ready)
        self.client.event(self.on_message)
        self.client.run(self.token)

if __name__ == "__main__":
    bot = DiscordBot(DISCORD_TOKEN)
    bot.run()
