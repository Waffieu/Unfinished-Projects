
import os
import discord
import datetime
import re
import json
import asyncio
from discord.ext import commands
from dotenv import load_dotenv
import google.generativeai as genai
from collections import defaultdict, Counter

# Load environment variables
load_dotenv()

# Configure Discord bot
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

# Configure Gemini AI
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash-lite')

# Bot configuration
intents = discord.Intents.all()
bot = commands.Bot(command_prefix='!', intents=intents)

# Data structures for tracking user warnings and behavior
user_warnings = defaultdict(int)  # {user_id: warning_count}
user_last_messages = defaultdict(list)  # {user_id: [last_messages]}
user_message_timestamps = defaultdict(list)  # {user_id: [message_timestamps]}
user_heat_points = defaultdict(int)  # {user_id: heat_points}
user_duplicate_reset_tasks = {}  # {user_id: task}

# Constants
MAX_WARNINGS = 3
MAX_HEAT_POINTS = 30
WARNING_RESET_DAYS = 7
NOTIFICATION_DELETE_SECONDS = 60
DUPLICATE_RESET_SECONDS = 120
MAX_SIMILAR_MESSAGES = 3
MAX_MENTIONS = 5

# Load settings from JSON file
def load_settings():
    global MAX_WARNINGS, WARNING_RESET_DAYS, MAX_HEAT_POINTS
    try:
        with open('bot_settings.json', 'r') as f:
            settings = json.load(f)
            MAX_WARNINGS = settings.get('max_warnings', 3)
            WARNING_RESET_DAYS = settings.get('reset_days', 7)
            MAX_HEAT_POINTS = MAX_WARNINGS * 10  # Set max heat points to be 10x max warnings
            debug_log(f"Settings loaded: MAX_WARNINGS={MAX_WARNINGS}, WARNING_RESET_DAYS={WARNING_RESET_DAYS}, MAX_HEAT_POINTS={MAX_HEAT_POINTS}")
    except Exception as e:
        debug_log(f"Error loading settings: {str(e)}")

# Debug function
def debug_log(message):
    if DEBUG_MODE:
        print(f"[DEBUG] {datetime.datetime.now()}: {message}")
        # In a real implementation, you might want to log to a file as well
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('bot_log.txt', 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")

# Check if user is admin
def is_admin(member):
    return member.guild_permissions.administrator

# Save warning data to JSON file
def save_warning_data():
    data = {
        "user_warnings": {str(k): v for k, v in user_warnings.items()},
        "user_heat_points": {str(k): v for k, v in user_heat_points.items()}
    }
    with open('warning_data.json', 'w') as f:
        json.dump(data, f)
    debug_log("Warning data saved to file")

# Load warning data from JSON file
def load_warning_data():
    global user_warnings, user_heat_points
    try:
        with open('warning_data.json', 'r') as f:
            data = json.load(f)
            user_warnings = defaultdict(int, {int(k): v for k, v in data.get("user_warnings", {}).items()})
            user_heat_points = defaultdict(int, {int(k): v for k, v in data.get("user_heat_points", {}).items()})
        debug_log("Warning data loaded from file")
    except FileNotFoundError:
        debug_log("No warning data file found, starting fresh")

# Schedule warning reset task
async def reset_warnings():
    while True:
        debug_log("Checking for warnings to reset")
        # Reset warnings after 7 days
        await asyncio.sleep(86400)  # Check once per day
        # In a real implementation, you would check the timestamp of each warning
        # and only reset those older than 7 days
        user_warnings.clear()
        debug_log("All warnings have been reset")
        save_warning_data()

# Reset duplicate message tracking for a user
async def reset_duplicate_tracking(user_id):
    await asyncio.sleep(DUPLICATE_RESET_SECONDS)
    if user_id in user_last_messages:
        user_last_messages[user_id] = []
        debug_log(f"Reset duplicate message tracking for user {user_id}")

# Send warning notification
async def send_warning_notification(channel, user, reason, heat_points, timeout_applied=False):
    embed = discord.Embed(
        title="⚠️ Uyarı",
        description=f"{user.mention} kullanıcısı uyarıldı.",
        color=discord.Color.yellow()
    )
    embed.add_field(name="Sebep", value=reason, inline=False)
    embed.add_field(name="Heat Puanı", value=f"{heat_points}/{MAX_HEAT_POINTS}", inline=True)
    
    warning_count = user_warnings[user.id]
    if timeout_applied:
        embed.add_field(name="Durum", value="Timeout uygulandı!", inline=True)
    else:
        embed.add_field(
            name="Durum", 
            value=f"Uyarı {warning_count}/{MAX_WARNINGS}. {MAX_WARNINGS - warning_count} uyarı kaldı.", 
            inline=True
        )
    
    notification = await channel.send(embed=embed)
    # Delete notification after 1 minute
    await asyncio.sleep(NOTIFICATION_DELETE_SECONDS)
    try:
        await notification.delete()
    except discord.NotFound:
        pass

# Apply warning to user
async def warn_user(message, user, reason, heat_points=1):
    if is_admin(user):
        debug_log(f"Admin user {user.id} would have been warned for: {reason}, but admins are exempt")
        return
    
    user_warnings[user.id] += 1
    user_heat_points[user.id] += heat_points
    warning_count = user_warnings[user.id]
    
    debug_log(f"User {user.id} warned for: {reason}. Warning count: {warning_count}, Heat points: {user_heat_points[user.id]}")
    
    # Check if timeout should be applied
    timeout_applied = False
    if warning_count >= MAX_WARNINGS or user_heat_points[user.id] >= MAX_HEAT_POINTS:
        timeout_duration = datetime.timedelta(minutes=30)
        try:
            # Use timeout() instead of timeout_for() for compatibility
            await user.timeout(timeout_duration, reason="Aşırı uyarı")
            debug_log(f"User {user.id} timed out for {timeout_duration}")
            timeout_applied = True
        except AttributeError:
            debug_log(f"User {user.id} could not be timed out: 'Member' object has no attribute 'timeout'")
        except discord.Forbidden:
            debug_log(f"Bot doesn't have permission to timeout user {user.id}")
    
    # Send notification with correct timeout status
    await send_warning_notification(message.channel, user, reason, user_heat_points[user.id], timeout_applied)
    
    save_warning_data()

# Analyze message with Gemini AI
async def analyze_message_with_gemini(message_content, analysis_type):
    try:
        prompt = ""
        if analysis_type == "spam":
            prompt = f"Aşağıdaki mesaj spam içeriyor mu? Sadece 'evet' veya 'hayır' olarak cevap ver.\n\nMesaj: {message_content}"
        elif analysis_type == "fake_link":
            prompt = f"Aşağıdaki mesaj sahte Discord Nitro, Steam Wallet Code veya benzeri dolandırıcılık içeren bir link içeriyor mu ve phishing linkleri ve değiştirilmiş linkler içeriyormu? Sadece 'evet' veya 'hayır' olarak cevap ver.\n\nMesaj: {message_content}"
        elif analysis_type == "profanity":
            prompt = f"Aşağıdaki mesaj küfür, hakaret veya argo ifade içeriyor mu? Türkçe küfür ve hakaret içeren kelimeleri tespit et. Sadece gerçekten küfür içeren mesajlar için 'evet', diğerleri için 'hayır' olarak cevap ver.\n\nMesaj: {message_content}"
        elif analysis_type == "caps":
            prompt = f"Aşağıdaki mesaj çoğunlukla büyük harflerden mi oluşuyor? Sadece 'evet' veya 'hayır' olarak cevap ver.\n\nMesaj: {message_content}"
        elif analysis_type == "advertisement":
            prompt = f"Aşağıdaki mesaj agresif reklam içeriyor mu? Normal paylaşılan YouTube, Spotify veya diğer sosyal medya linkleri reklam olarak kabul edilmemelidir. Sadece açıkça ticari amaçlı, spam niteliğinde veya istenmeyen reklamlar için 'evet', normal link paylaşımları için 'hayır' olarak cevap ver.\n\nMesaj: {message_content}"
        elif analysis_type == "anomaly":
            prompt = f"Aşağıdaki mesaj normal bir Discord mesajından sapma gösteriyor mu? Örneğin, alışılmadık komut kullanımı, şüpheli linkler veya anormal davranış belirtileri var mı? Normal mesajlar ve normal link paylaşımları için 'hayır' cevabını ver. Sadece 'evet' veya 'hayır' olarak cevap ver.\n\nMesaj: {message_content}"
        
        if not prompt:
            return False
        
        debug_log(f"Analyzing message for {analysis_type}: {message_content[:50]}...")
        response = await model.generate_content_async(prompt)
        result = response.text.lower().strip()
        debug_log(f"Gemini analysis for {analysis_type}: {result}")
        
        return "evet" in result
    except Exception as e:
        debug_log(f"Error analyzing message with Gemini: {str(e)}")
        return False

# Check for spam messages
async def check_spam(message):
    user_id = message.author.id
    current_time = datetime.datetime.now()
    
    # Add current message timestamp
    user_message_timestamps[user_id].append(current_time)
    
    # Remove timestamps older than 5 seconds
    user_message_timestamps[user_id] = [ts for ts in user_message_timestamps[user_id] 
                                      if (current_time - ts).total_seconds() < 5]
    
    # If user sent more than 5 messages in 5 seconds
    if len(user_message_timestamps[user_id]) > 5:
        debug_log(f"Potential spam detected from user {user_id}: {len(user_message_timestamps[user_id])} messages in 5 seconds")
        
        # Use Gemini to analyze if the content is spam
        is_spam = await analyze_message_with_gemini(message.content, "spam")
        
        if is_spam:
            try:
                await message.delete()
                debug_log(f"Deleted spam message from user {user_id}")
                await warn_user(message, message.author, "Spam mesajlar gönderme", heat_points=2)
                return True
            except discord.Forbidden:
                debug_log(f"Bot doesn't have permission to delete message from user {user.id}")
    
    return False

# Check for fake links
async def check_fake_links(message):
    # Simple regex for URLs
    urls = re.findall(r'(https?://\S+)', message.content)
    
    if urls:
        debug_log(f"URLs detected in message from user {message.author.id}: {urls}")
        
        # Use Gemini to analyze if the content contains fake links
        contains_fake_links = await analyze_message_with_gemini(message.content, "fake_link")
        
        if contains_fake_links:
            try:
                await message.delete()
                debug_log(f"Deleted message with fake links from user {message.author.id}")
                await warn_user(message, message.author, "Sahte link paylaşımı", heat_points=3)
                return True
            except discord.Forbidden:
                debug_log(f"Bot doesn't have permission to delete message from user {message.author.id}")
    
    return False

# Check for duplicate messages
async def check_duplicate_messages(message):
    user_id = message.author.id
    content = message.content.lower()
    
    # Add message to user's recent messages
    user_last_messages[user_id].append(content)
    
    # Count occurrences of each message
    message_counts = Counter(user_last_messages[user_id])
    
    # Check if any message appears more than MAX_SIMILAR_MESSAGES times
    for msg, count in message_counts.items():
        if count > MAX_SIMILAR_MESSAGES:
            debug_log(f"Duplicate messages detected from user {user_id}: '{msg[:30]}...' repeated {count} times")
            
            try:
                # Delete all instances of the duplicated message
                async for old_message in message.channel.history(limit=100):
                    if (old_message.author.id == user_id and 
                        old_message.content.lower() == msg):
                        await old_message.delete()
                        debug_log(f"Deleted duplicate message from user {user_id}")
                
                await warn_user(message, message.author, "Aynı mesajı tekrar tekrar gönderme", heat_points=2)
                
                # Clear the user's message history
                user_last_messages[user_id] = []
                
                return True
            except discord.Forbidden:
                debug_log(f"Bot doesn't have permission to delete messages from user {user_id}")
    
    # Schedule reset of duplicate tracking if not already scheduled
    if user_id not in user_duplicate_reset_tasks or user_duplicate_reset_tasks[user_id].done():
        user_duplicate_reset_tasks[user_id] = asyncio.create_task(reset_duplicate_tracking(user_id))
    
    return False

# Check for profanity
async def check_profanity(message):
    # Use Gemini to analyze if the content contains profanity
    contains_profanity = await analyze_message_with_gemini(message.content, "profanity")
    
    if contains_profanity:
        debug_log(f"Profanity detected in message from user {message.author.id}")
        
        try:
            await message.delete()
            debug_log(f"Deleted message with profanity from user {message.author.id}")
            await warn_user(message, message.author, "Küfür veya hakaret içeren mesaj", heat_points=2)
            return True
        except discord.Forbidden:
            debug_log(f"Bot doesn't have permission to delete message from user {message.author.id}")
    
    return False

# Check for invite links
async def check_invite_links(message):
    # Check for Discord invite links
    if re.search(r'discord(?:.gg|.com/invite)/[a-zA-Z0-9]+', message.content):
        debug_log(f"Invite link detected in message from user {message.author.id}")
        
        # Use Gemini to verify if it's a legitimate invite or potentially harmful
        is_harmful = await analyze_message_with_gemini(message.content, "fake_link")
        
        if is_harmful:
            try:
                await message.delete()
                debug_log(f"Deleted message with harmful invite link from user {message.author.id}")
                await warn_user(message, message.author, "Zararlı davet linki paylaşımı", heat_points=2)
                return True
            except discord.Forbidden:
                debug_log(f"Bot doesn't have permission to delete message from user {message.author.id}")
        else:
            debug_log(f"Legitimate invite link detected from user {message.author.id}, no action taken")
    
    return False

# Check for excessive mentions
async def check_excessive_mentions(message):
    # Count mentions in the message
    mention_count = len(message.mentions)
    has_everyone = message.mention_everyone
    has_here = '@here' in message.content
    
    # If message has @everyone, @here, or too many mentions
    if has_everyone or has_here or mention_count > MAX_MENTIONS:
        debug_log(f"Excessive mentions detected from user {message.author.id}: {mention_count} mentions, @everyone: {has_everyone}, @here: {has_here}")
        
        try:
            await message.delete()
            debug_log(f"Deleted message with excessive mentions from user {message.author.id}")
            await warn_user(message, message.author, "Aşırı mention kullanımı", heat_points=2)
            return True
        except discord.Forbidden:
            debug_log(f"Bot doesn't have permission to delete message from user {message.author.id}")
    
    return False

# Check for all caps messages
async def check_all_caps(message):
    # Skip short messages
    if len(message.content) < 10:
        return False
    
    # Use Gemini to analyze if the message is mostly in caps
    is_caps = await analyze_message_with_gemini(message.content, "caps")
    
    if is_caps:
        debug_log(f"All caps message detected from user {message.author.id}")
        
        try:
            await message.delete()
            debug_log(f"Deleted all caps message from user {message.author.id}")
            await warn_user(message, message.author, "Tamamen büyük harflerle yazma", heat_points=1)
            return True
        except discord.Forbidden:
            debug_log(f"Bot doesn't have permission to delete message from user {message.author.id}")
    
    return False

# Check for advertisement
async def check_advertisement(message):
    # Use Gemini to analyze if the message contains advertisement
    contains_ad = await analyze_message_with_gemini(message.content, "advertisement")
    
    if contains_ad:
        debug_log(f"Advertisement detected in message from user {message.author.id}")
        
        try:
            await message.delete()
            debug_log(f"Deleted message with advertisement from user {message.author.id}")
            await warn_user(message, message.author, "Reklam içerikli mesaj", heat_points=2)
            return True
        except discord.Forbidden:
            debug_log(f"Bot doesn't have permission to delete message from user {message.author.id}")
    
    return False

# Check for anomalies
async def check_anomalies(message):
    # Use Gemini to analyze if the message contains anomalies
    contains_anomaly = await analyze_message_with_gemini(message.content, "anomaly")
    
    if contains_anomaly:
        debug_log(f"Anomaly detected in message from user {message.author.id}")
        
        try:
            await message.delete()
            debug_log(f"Deleted message with anomaly from user {message.author.id}")
            await warn_user(message, message.author, "Şüpheli davranış", heat_points=2)
            return True
        except discord.Forbidden:
            debug_log(f"Bot doesn't have permission to delete message from user {message.author.id}")
    
    return False

# Event: Bot is ready
@bot.event
async def on_ready():
    debug_log(f"{bot.user.name} is connected to Discord!")
    
    # Load settings
    load_settings()
    
    # Load warning data
    load_warning_data()
    
    # Start warning reset task
    bot.loop.create_task(reset_warnings())

# Event: Message received
@bot.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return
    
    # Skip processing for admin users
    if is_admin(message.author):
        debug_log(f"Skipping security checks for admin user {message.author.id}")
        await bot.process_commands(message)
        return
    
    # Load settings to check which features are enabled
    try:
        with open('bot_settings.json', 'r') as f:
            settings = json.load(f)
    except Exception as e:
        debug_log(f"Error loading settings: {str(e)}")
        settings = {}
    
    # Check for various violations, but only if the corresponding setting is enabled
    checks = [
        (check_spam, settings.get('spam_check', False)),
        (check_fake_links, settings.get('fake_link_check', False)),
        (check_duplicate_messages, settings.get('duplicate_check', False)),
        (check_profanity, settings.get('profanity_check', False)),
        (check_invite_links, settings.get('invite_check', False)),
        (check_excessive_mentions, settings.get('mention_check', False)),
        (check_all_caps, settings.get('caps_check', False)),
        (check_advertisement, settings.get('ad_check', False)),
        (check_anomalies, settings.get('anomaly_check', False))
    ]
    
    for check_func, is_enabled in checks:
        if is_enabled:
            try:
                result = await check_func(message)
                if result:
                    # If any check returns True, stop processing further checks
                    return
            except Exception as e:
                debug_log(f"Error in {check_func.__name__}: {str(e)}")
    
    # Process commands if no violations were found
    await bot.process_commands(message)

# Command: View user warnings
@bot.command(name="warnings")
@commands.has_permissions(administrator=True)
async def view_warnings(ctx, user: discord.Member = None):
    if user is None:
        await ctx.send("Lütfen bir kullanıcı belirtin.")
        return
    
    warning_count = user_warnings[user.id]
    heat_points = user_heat_points[user.id]
    
    embed = discord.Embed(
        title=f"{user.display_name} - Uyarı Bilgileri",
        color=discord.Color.blue()
    )
    embed.add_field(name="Uyarı Sayısı", value=f"{warning_count}/{MAX_WARNINGS}", inline=True)
    embed.add_field(name="Heat Puanı", value=f"{heat_points}/10", inline=True)
    
    await ctx.send(embed=embed)

# Command: Clear user warnings
@bot.command(name="clearwarnings")
@commands.has_permissions(administrator=True)
async def clear_warnings(ctx, user: discord.Member = None):
    if user is None:
        await ctx.send("Lütfen bir kullanıcı belirtin.")
        return
    
    user_warnings[user.id] = 0
    user_heat_points[user.id] = 0
    save_warning_data()
    
    await ctx.send(f"{user.mention} kullanıcısının tüm uyarıları temizlendi.")

# Command: Remove timeout from user
@bot.command(name="removetimeout")
@commands.has_permissions(administrator=True)
async def remove_timeout(ctx, user_id: str):
    try:
        # Convert user_id to integer
        user_id = int(user_id)
        
        # Get the user object
        user = await bot.fetch_user(user_id)
        
        # Find the guild and member
        # If command is from control panel, ctx.guild might be None
        if ctx.guild:
            guild = ctx.guild
        else:
            # Use the first available guild if ctx.guild is None
            guild = next(iter(bot.guilds), None)
            if not guild:
                debug_log(f"No guild available to remove timeout for user {user_id}")
                return
        
        # Get the member object
        try:
            member = await guild.fetch_member(user_id)
        except discord.NotFound:
            debug_log(f"User {user_id} not found in guild {guild.id}")
            if ctx.channel:
                await ctx.send("Kullanıcı sunucuda bulunamadı.")
            return
        
        # Remove timeout
        await member.timeout(None, reason="Timeout manually removed by admin")
        
        # Update warning data
        if user_id in user_warnings:
            user_warnings[user_id] = 0
        if user_id in user_heat_points:
            user_heat_points[user_id] = 0
        save_warning_data()
        
        debug_log(f"Timeout removed from user {user_id}")
        if ctx.channel:
            await ctx.send(f"{user.mention} kullanıcısının timeout'u kaldırıldı ve uyarıları sıfırlandı.")
    except ValueError:
        debug_log(f"Invalid user ID: {user_id}")
        if ctx.channel:
            await ctx.send("Geçersiz kullanıcı ID'si.")
    except discord.Forbidden:
        debug_log(f"Bot doesn't have permission to remove timeout for user {user_id}")
        if ctx.channel:
            await ctx.send("Bot'un bu işlemi gerçekleştirmek için yeterli izni yok.")
    except Exception as e:
        debug_log(f"Error removing timeout: {str(e)}")
        if ctx.channel:
            await ctx.send(f"Timeout kaldırılırken bir hata oluştu: {str(e)}")

# Function to process stdin commands
async def process_stdin_commands():
    while True:
        try:
            # Read a line from stdin using asyncio to avoid blocking
            line = await asyncio.get_event_loop().run_in_executor(None, input)
            line = line.strip()
            
            # Check if it's a command
            if line.startswith('!'):
                debug_log(f"Received command from control panel: {line}")
                # Create a mock context for the command
                # We need to find a guild and channel to execute the command in
                guild = next(iter(bot.guilds), None)
                if guild:
                    channel = next(iter(guild.text_channels), None)
                    if channel:
                        # Create a mock message
                        message = discord.Message(state=bot._connection, channel=channel, data={
                            'id': 0,
                            'content': line,
                            'author': {
                                'id': 0,
                                'username': 'ControlPanel',
                                'discriminator': '0000',
                                'bot': False
                            }
                        })
                        # Process the command
                        ctx = await bot.get_context(message)
                        await bot.invoke(ctx)
        except Exception as e:
            debug_log(f"Error processing stdin command: {str(e)}")
        await asyncio.sleep(0.1)

# Run the bot
if __name__ == "__main__":
    # Use setup_hook to initialize async tasks
    @bot.event
    async def setup_hook():
        # Start the stdin command processor in the bot's event loop
        bot.loop.create_task(process_stdin_commands())
        debug_log("Started stdin command processor")
    
    # Run the bot
    bot.run(DISCORD_TOKEN)