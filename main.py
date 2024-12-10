import os  # Импортируем модуль для работы с операционной системой (файлы, переменные окружения и т.д.)
import logging  # Импортируем модуль для логирования ошибок и информации
from aiogram import Bot, Dispatcher, executor, types  # Импортируем основные компоненты для работы с Telegram ботом
from dotenv import load_dotenv  # Импортируем функцию для загрузки переменных окружения из .env файла
import asyncio  # Импортируем модуль для асинхронного программирования
import soundfile as sf  # Импортируем библиотеку для работы с аудио файлами
import numpy as np  # Импортируем NumPy для работы с массивами данных (не используется в коде, но может понадобиться)
from gtts import gTTS  # Импортируем библиотеку для синтеза речи (Google Text-to-Speech)
from faster_whisper import WhisperModel  # Импортируем библиотеку для транскрипции речи с помощью модели Whisper
from mistral_client import MistralClient  # Импортируем клиент для работы с API Mistral

load_dotenv()  # Загружаем переменные окружения из .env файла

logging.basicConfig(level=logging.INFO)  # Настроили базовое логирование с уровнем INFO (информационные сообщения)


# Загружаем токен бота из переменных окружения
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:  # Проверяем, что токен бота был найден
    raise ValueError("No BOT_TOKEN found in .env file")  # Если токен не найден, выбрасываем ошибку

bot = Bot(token=BOT_TOKEN)  # Инициализируем объект бота с токеном
dp = Dispatcher(bot)  # Создаём диспетчер для обработки сообщений


# Получаем ключ API для работы с Mistral
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
mistral_client = MistralClient(api_key=MISTRAL_API_KEY)  # Инициализируем клиента для общения с API Mistral


# Инициализируем модель Whisper для транскрипции
model = WhisperModel("base", device="cpu", compute_type="int8")  # Загружаем модель Whisper (основную) для работы на CPU


@dp.message_handler(commands=['start', 'help'])  # Обработчик команд /start и /help
async def send_welcome(message: types.Message):    
    await message.reply("Привет! Отправь мне голосовое сообщение, и я отвечу тебе тоже голосом!")  # Отправляем приветственное сообщение


@dp.message_handler(content_types=[types.ContentType.VOICE])  # Обработчик голосовых сообщений
async def handle_voice(message: types.Message):
    try:
        # Получаем файл голосового сообщения
        voice = await message.voice.get_file()
        voice_path = f"temp_voice_{message.from_user.id}.ogg"  # Определяем путь для временного сохранения файла
        await voice.download(voice_path)  # Загружаем голосовое сообщение в файл
        
        await message.answer("Обрабатываю ваше сообщение...")  # Сообщаем пользователю, что обработка началась
        
        # Читаем аудиофайл и сохраняем его в формате .wav для дальнейшей работы
        audio_data, sample_rate = sf.read(voice_path)
        sf.write(f"temp_voice_{message.from_user.id}.wav", audio_data, sample_rate)  # Сохраняем в .wav формате
        
        # Транскрибируем аудио в текст с помощью модели Whisper
        segments, _ = model.transcribe(f"temp_voice_{message.from_user.id}.wav", beam_size=5)
        transcribed_text = " ".join([segment.text for segment in segments])  # Собираем текст из транскрибированных сегментов
        
        # Формируем запрос для API Mistral, передавая транскрибированный текст
        messages = [
            {"role": "system", "content": "Тебя зовут Mistral Ai. Ты - полезный голосовой помощник. Отвечайте лаконично и естественно."},
            {"role": "user", "content": transcribed_text}
        ]
        
        # Получаем ответ от модели Mistral
        ai_response = await mistral_client.chat_completion(
            messages=messages,  # Отправляем сообщения на сервер
            model="mistral-small-latest"  # Указываем модель для генерации ответа
        )
        
        # Конвертируем ответ в аудиофайл с помощью gTTS
        tts = gTTS(text=ai_response, lang='ru')  # Генерируем речь на русском языке
        response_audio_path = f"temp_response_{message.from_user.id}.mp3"  # Указываем путь для сохранения аудио
        tts.save(response_audio_path)  # Сохраняем аудио файл с ответом
        
        # Отправляем текстовый ответ пользователю
        await message.reply(f"Вы сказали: {transcribed_text}\n\nМой ответ: {ai_response}")
        
        # Отправляем аудиофайл с голосовым ответом
        with open(response_audio_path, 'rb') as audio:
            await message.reply_voice(audio)  # Отправляем голосовое сообщение
        
        # Очищаем временные файлы
        cleanup_files = [
            voice_path,  # Исходный голосовой файл
            f"temp_voice_{message.from_user.id}.wav",  # Временный .wav файл
            response_audio_path  # Аудиофайл с ответом
        ]
        for file in cleanup_files:
            if os.path.exists(file):  # Проверяем, существует ли файл
                os.remove(file)  # Удаляем временные файлы
                
    except Exception as e:
        logging.error(f"Error processing voice message: {e}")  # Логируем ошибку
        await message.reply("Извините, произошла ошибка при обработке голосового сообщения. Попробуйте еще раз.")  # Сообщаем пользователю об ошибке

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)  # Запускаем бота, пропуская старые обновления
