import os
import discord
import random
import asyncio
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

load_dotenv()  # Load environment variables from .env file
DISCORD_TOKEN = "your-dicord-bot-token"  # Store your token securely in an environment variable


# Configure logging to see detailed information
logging.basicConfig(level=logging.DEBUG)

# Discord and Selenium setup
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# Set up Chrome options
options = webdriver.ChromeOptions()
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Function to wait for the element to be interactable
def wait_for_element_to_be_interactable(driver, by, value):
    for _ in range(5):  # Try 5 times
        try:
            element = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((by, value)))
            logging.debug("Element found and is interactable.")
            return element
        except Exception as e:
            logging.error("Waiting for element to be interactable: %s", e)
            time.sleep(2)  # Wait before retrying
    return None

# ChatGPT response function
cached_response = None  # Global variable to cache response
first_load = True  # Flag to track the first load

async def fetch_chatgpt_response(question):
    global cached_response  # Use the global cached response variable
    global first_load  # Use the global first load flag

    driver.get("https://copilot.microsoft.com/chats")
    logging.debug("Navigating to ChatGPT page.")

    # Refresh the page only on the first load
    if first_load:
        await asyncio.sleep(random.uniform(2, 5))  # Random delay to mimic human behavior
        driver.refresh()
        logging.debug("Page refreshed on first load.")
        first_load = False  # Set the flag to False after refreshing

    # Wait for the input box to be interactable
    input_box = wait_for_element_to_be_interactable(driver, By.TAG_NAME, "textarea")
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
    try:
        response_element = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, ".text-base"))
        )
        response = response_element.text

        logging.debug("Response received: %s", response)  # Log the raw response

        # Check if the response contains the fox or wolf emoji
        if "Merhaba" in response or "Merhaba!" in response:
            cached_response = response  # Cache the response
            logging.debug("Valid response found, returning it.")
            return response  # Return the response if it contains merhaba or Merhaba!
        else:
            logging.warning("Invalid response, no valid emojis found.")
            return "Cevap geçersiz; lütfen başka bir konuda soru sorun."

    except Exception as e:
        logging.error("Error while waiting for response: %s", e)
        return "Cevabı alırken bir hata oluştu, lütfen tekrar deneyin."  # Error handling

# Discord bot events
@client.event
async def on_ready():
    logging.info(f"Giriş yapıldı: {client.user}")

@client.event
async def on_message(message):
    if client.user.mentioned_in(message):  # Responds only when mentioned
        question = message.content.replace(f"<@{client.user.id}>", "").strip()
        
        await message.channel.send("Cevap bekleniyor...")  # Notify user that a response is being fetched
        answer = await fetch_chatgpt_response(question)  # Use await for the response

        if answer:
            await message.channel.send(answer)  # Send the response to Discord
        else:
            await message.channel.send("Cevabı alırken bir hata oluştu, lütfen tekrar deneyin.")  # Error handling

        await asyncio.sleep(1)  # Rate limiting: wait before processing another message

# Ensure the WebDriver is properly closed on exit
@client.event
async def on_disconnect():
    driver.quit()  # Close the WebDriver

# Start the Discord bot
try:
    client.run(DISCORD_TOKEN)
except Exception as e:
    logging.error("Bot çalışırken bir hata oluştu: %s", e)
finally:
    driver.quit()  # Ensure driver is closed on exit