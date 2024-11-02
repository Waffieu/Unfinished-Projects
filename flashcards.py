import discord
from discord.ext import commands
import random
import re
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import asyncio
from fuzzywuzzy import fuzz
import os
import logging

# Loglama ayarları
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Gerekli NLTK verilerini indirin
nltk.download('stopwords')
nltk.download('punkt')

# Lemmatizer'ı başlat
lemmatizer = WordNetLemmatizer()

# Bot için izinleri tanımla
intents = discord.Intents.default()
intents.messages = True
intents.guilds = True
intents.message_content = True

# Botu başlat
bot = commands.Bot(command_prefix='!', intents=intents)

# Scriptin bulunduğu dizini al
CODE_DIR = os.path.dirname(__file__)

# Dosyayı dizin içinde bulma fonksiyonu
def find_file_in_directory(filename):
    logging.debug(f"Aranan dosya: {filename}, dizin: {CODE_DIR}")
    for root, _, files in os.walk(CODE_DIR):
        if filename in files:
            file_path = os.path.join(root, filename)
            logging.debug(f"Dosya bulundu: {file_path}")
            return file_path
    logging.error(f"{filename} dosyası dizinde bulunamadı.")
    return None

# Flash kartları yükleme
def flash_kartlari_yukle(dil):
    logging.debug(f"{dil} dili için flash kartlar yükleniyor.")
    dosya_adi = 'english.txt' if dil == 'en' else 'deutsch.txt'
    dosya_yolu = find_file_in_directory(dosya_adi)

    if not dosya_yolu:
        logging.error(f"Hata: {dosya_adi} dosyası bulunamadı.")
        return {}

    flashcards = {}
    try:
        with open(dosya_yolu, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:  # Boş satırları atla
                    continue
                # Çift tırnakları kaldır ve iki nokta ile ayır
                if line.startswith('"') and ':' in line:
                    try:
                        key_value = line.split(':', 1)  # İlk iki nokta ile ayır
                        kelime = key_value[0].strip().strip('"')  # Kelimeyi al, tırnakları kaldır
                        anlam = key_value[1].strip().strip('"')  # Anlamı al, tırnakları kaldır
                        flashcards[kelime] = anlam
                    except Exception as e:
                        logging.error(f"{dosya_adi} dosyasında {line_number}. satırı işlerken hata: {line}. Hata: {e}")
                else:
                    logging.error(f"Hata: {dosya_adi} dosyasında {line_number}. satır düzgün formatta değil: {line}")
            logging.debug(f"Flash kartlar başarıyla yüklendi: {len(flashcards)} giriş bulundu.")
            return flashcards
    except Exception as e:
        logging.error(f"{dosya_adi} dosyası okunurken hata: {e}")
        return {}

# Komut niyetlerini tanımla
niyetler_sozlugu = {
    'ogren': ['öğren', 'başla', 'başlat', 'çalış', 'öğret', 'ogren', 'başlamak'],
    'cikis': ['çık', 'durdur', 'bitir', 'kapat', 'çikis', 'çik', 'cik']
}

# Kullanıcı durumlarını ve aktivitelerini takip et
kullanici_dilleri = {}
kullanici_ogrenme_durumlari = {}
guncel_sorular = {}
kullanici_son_etkinlik = {}
etkisizlik_zaman_asimi = 300

# Emojiler için sabitler
EMOJIS = ['1️⃣', '2️⃣', '3️⃣', '4️⃣']

# Metin ön işleme
def girdi_on_isleme(metin, dil):
    logging.debug(f"Girdi ön işleniyor: {metin}, dil: {dil}")
    metin = metin.lower().strip()
    metin = re.sub(r'[^\w\s]', '', metin)
    tokenler = word_tokenize(metin, language='english' if dil == 'en' else 'german')
    durak_kelime = set(stopwords.words('english') if dil == 'en' else stopwords.words('german'))
    tokenler = [lemmatizer.lemmatize(word) for word in tokenler if word not in durak_kelime]
    logging.debug(f"İşlenmiş tokenler: {tokenler}")
    return tokenler

# En yakın komut eşleşmesini bul
def en_yakin_komut_bul(girdi):
    logging.debug(f"Girdi için en yakın komut eşleşmesi aranıyor: {girdi}")
    en_yuksek_eslesme = 0
    eslesen_komut = None
    for niyet, komutlar in niyetler_sozlugu.items():
        for komut in komutlar:
            eslesme_orani = fuzz.ratio(girdi, komut)
            logging.debug(f"{girdi} ile {komut} eşleşmesi: {eslesme_orani}%")
            if eslesme_orani > en_yuksek_eslesme:
                en_yuksek_eslesme = eslesme_orani
                eslesen_komut = niyet
    result = eslesen_komut if en_yuksek_eslesme >= 70 else None
    logging.debug(f"En iyi eşleşme: {result} ile {en_yuksek_eslesme}%")
    return result

# Kullanıcının mesajına göre komutları çalıştır
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    logging.debug(f"Kullanıcıdan gelen mesaj: {message.author}: {message.content}")
    niyet = en_yakin_komut_bul(message.content.lower())
    if niyet == 'ogren':
        await ogren(message)
    elif niyet == 'cikis':
        await cikis(message)

async def ogren(ctx):
    kullanici_id = ctx.author.id
    logging.debug(f"Kullanıcı için öğrenme oturumu başlatılıyor: {kullanici_id}")

    if kullanici_ogrenme_durumlari.get(kullanici_id) == 'dil_seciliyor':
        await ctx.channel.send("Zaten bir dil seçme sürecindesiniz.")
        return

    kullanici_ogrenme_durumlari[kullanici_id] = 'dil_seciliyor'
    await ctx.channel.send("Bugün hangi dili öğrenmek istiyorsunuz? İngilizce için '1', Almanca için '2' yazın ya da direkt 'İngilizce' veya 'Almanca' yazabilirsiniz.")

    def kontrol(mesaj):
        return mesaj.author == ctx.author and mesaj.channel == ctx.channel

    try:
        yanit = await bot.wait_for('message', timeout=60.0, check=kontrol)
        logging.debug(f"Kullanıcı yanıtı alındı: {yanit.content}")
    except asyncio.TimeoutError:
        await ctx.channel.send("Yanıt vermek için çok uzun süre beklediniz. Lütfen tekrar deneyin.")
        kullanici_dilleri.pop(kullanici_id, None)
        kullanici_ogrenme_durumlari.pop(kullanici_id, None)
        logging.warning("Zaman aşımı hatası: Kullanıcı zamanında yanıt vermedi.")
        return

    kullanici_girdisi = yanit.content.lower()
    if kullanici_girdisi in ['1', 'ingilizce']:
        dil = 'en'
    elif kullanici_girdisi in ['2', 'almanca']:
        dil = 'de'
    else:
        await ctx.channel.send("Geçersiz bir seçim yaptınız. Lütfen tekrar deneyin.")
        kullanici_ogrenme_durumlari.pop(kullanici_id, None)
        logging.warning("Geçersiz dil seçimi.")
        return

    kullanici_dilleri[kullanici_id] = dil
    kullanici_ogrenme_durumlari.pop(kullanici_id, None)
    kullanici_son_etkinlik[kullanici_id] = asyncio.get_event_loop().time()

    flash_cards = flash_kartlari_yukle(dil)
    if not flash_cards:
        await ctx.channel.send("Şu anda bu dil için flash kartlar mevcut değil.")
        logging.error("Seçilen dil için flash kart yok.")
        return

    await ctx.channel.send(f"{'İngilizce' if dil == 'en' else 'Almanca'} dersine başlıyoruz! Sorular bu kanalda gönderilecek.")
    await soru_sor(ctx.channel, ctx.author)

async def soru_sor(kanal, kullanici):
    dil = kullanici_dilleri.get(kullanici.id)
    flash_cards = flash_kartlari_yukle(dil)

    if kullanici.id in guncel_sorular and guncel_sorular[kullanici.id]:
        previously_asked = guncel_sorular[kullanici.id]
    else:
        previously_asked = []

    available_flashcards = [item for item in flash_cards.items() if item[0] not in previously_asked]

    if not available_flashcards:
        await kanal.send("Tüm soruları tamamladınız! Başka bir dil seçebilir ya da dersi kapatabilirsiniz.")
        return

    kelime, anlam = random.choice(available_flashcards)
    dogru_yanit = random.randint(1, 4)
    secenekler = random.sample(list(flash_cards.values()), 3)
    secenekler.insert(dogru_yanit - 1, anlam)

    # Güncel soruları güncelle
    guncel_sorular.setdefault(kullanici.id, []).append(kelime)

    soru_metni = f"**{kelime}** kelimesinin anlamı nedir?\n\n" + "\n".join([f"{EMOJIS[i]} {secenekler[i]}" for i in range(4)])
    await kanal.send(soru_metni)

    while True:
        try:
            yanit = await bot.wait_for('message', timeout=60.0, check=lambda m: m.author == kullanici and m.channel == kanal)
            kullanici_son_etkinlik[kullanici.id] = asyncio.get_event_loop().time()

            if yanit.content.lower() == anlam.lower():
                await kanal.send(f"Tebrikler! Doğru cevap: {anlam} 🎉")
                break  # Doğru cevap verildiğinde döngüden çık
            else:
                await kanal.send(f"Yanlış cevap! Tekrar deneyin.")

        except asyncio.TimeoutError:
            await kanal.send("Yanıt vermek için çok uzun süre beklediniz. Dersi iptal ediyorum.")
            guncel_sorular.pop(kullanici.id, None)  # Kullanıcının geçerli soru geçmişini temizle
            return

    # Doğru cevap verildiğinde yeni soru sor
    await soru_sor(kanal, kullanici)

async def cikis(ctx):
    kullanici_id = ctx.author.id
    logging.debug(f"Kullanıcı için oturumdan çıkılıyor: {kullanici_id}")
    kullanici_dilleri.pop(kullanici_id, None)
    kullanici_ogrenme_durumlari.pop(kullanici_id, None)
    guncel_sorular.pop(kullanici_id, None)  # Kullanıcının soru geçmişini temizle
    await ctx.channel.send("Ders oturumu kapatıldı.")
    logging.info(f"Kullanıcı için oturum kapatıldı: {kullanici_id}")

# Bot'u başlatma
bot.run('discord-bot-token')  # Bot tokeninizi buraya yapıştırın
