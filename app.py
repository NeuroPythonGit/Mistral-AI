# Импортируем необходимые библиотеки
import sounddevice as sd  # Для работы с аудио-устройствами
import numpy as np  # Для работы с массивами данных
import wave  # Для работы с WAV файлами
import time  # Для работы с временем
import os  # Для работы с операционной системой (например, переменные окружения)
from scipy.io import wavfile  # Для записи и чтения WAV файлов
from faster_whisper import WhisperModel  # Для использования модели Whisper для транскрипции аудио
from gtts import gTTS  # Для генерации речи с помощью Google Text-to-Speech
import gradio as gr  # Для создания пользовательского интерфейса
import asyncio  # Для работы с асинхронными задачами
from concurrent.futures import ThreadPoolExecutor  # Для многозадачности в отдельном потоке
from mistral_client import MistralClient  # Для работы с Mistral AI (для генерации ответов)
from dotenv import load_dotenv  # Для загрузки переменных окружения из .env файла

# Загрузка переменных окружения
load_dotenv()

# Инициализация клиента для работы с Mistral API, используя API-ключ из переменных окружения
mistral_client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))

# Инициализация модели Whisper для транскрипции аудио
model = WhisperModel("base", device="cpu", compute_type="int8")

# Определение класса для голосового ассистента
class VoiceAssistant:
    def __init__(self):
        self.recording = False  # Флаг для отслеживания состояния записи
        self.audio_data = []  # Список для хранения аудио данных
        self.sample_rate = 44100  # Частота дискретизации аудио
        self.executor = ThreadPoolExecutor(max_workers=1)  # Используем один поток для выполнения задач

    # Метод для проверки доступности микрофона
    def check_microphone(self):
        """Проверка наличия работающего микрофона"""
        try:
            devices = sd.query_devices()  # Запрос списка всех устройств
            input_devices = [d for d in devices if d['max_input_channels'] > 0]  # Выбираем устройства, поддерживающие ввод
            if not input_devices:
                return "No input devices found!"  # Если нет устройств для ввода, выводим сообщение
            
            device_list = "\nAvailable input devices:\n"  # Строка для вывода доступных устройств
            for d in input_devices:
                device_list += f"- {d['name']}\n"  # Добавляем устройство в список
            default_device = sd.query_devices(kind='input')  # Получаем информацию об устройстве по умолчанию
            device_list += f"\nUsing default input device: {default_device['name']}"  # Добавляем информацию о дефолтном устройстве
            return device_list
        except Exception as e:
            return f"Error checking microphone: {e}"  # В случае ошибки выводим сообщение об ошибке

    # Асинхронный метод для обработки голосового ввода
    async def process_voice(self, audio_data, sample_rate):
        """Обработка голосового ввода из Gradio"""
        try:
            # Сохраняем аудио в временный файл
            wavfile.write("temp.wav", sample_rate, audio_data)
            
            # Транскрибируем аудио в текст
            text = self.transcribe_audio()
            if not text:
                return "Failed to transcribe audio", None  # Если не удалось транскрибировать, возвращаем ошибку
                
            # Получаем ответ от ИИ
            response = await self.get_ai_response(text)
            if not response:
                return "Failed to get AI response", None  # Если не удалось получить ответ, возвращаем ошибку
                
            # Генерируем речь с помощью gTTS
            try:
                print("Generating audio response...")
                tts = gTTS(text=response, lang='ru')  # Генерация речи на русском языке
                tts.save("temp_response.wav")  # Сохраняем аудио-ответ в файл
                
                return f"You: {text}\nAssistant: {response}", "temp_response.wav"  # Возвращаем текст и путь к аудио
            except Exception as e:
                return f"Error in speech generation: {e}", None  # Ошибка генерации речи
                
        except Exception as e:
            return f"Error processing voice: {e}", None  # Ошибка обработки голоса

    # Метод для транскрипции аудио в текст с использованием модели Whisper
    def transcribe_audio(self, filename="temp.wav"):
        """Транскрипция аудио в текст с помощью Whisper"""
        try:
            segments, info = model.transcribe(filename, beam_size=5)  # Транскрибируем аудио
            transcribed_text = " ".join([segment.text for segment in segments])  # Собираем текст из сегментов
            return transcribed_text
        except Exception as e:
            print(f"Error during transcription: {e}")  # В случае ошибки выводим сообщение
            return None

    # Асинхронный метод для получения ответа от ИИ с использованием Mistral
    async def get_ai_response(self, text):
        """Получение ответа от ИИ с помощью Mistral AI"""
        try:
            # Формируем сообщения для общения с ИИ
            messages = [
                {"role": "system", "content": "You are a helpful voice assistant. Keep your responses concise and natural."},
                {"role": "user", "content": text}
            ]
            
            # Отправляем запрос к Mistral API
            response = await mistral_client.chat_completion(
                messages=messages,
                model="mistral-small-latest"
            )
            return response  # Возвращаем ответ ИИ
        except Exception as e:
            print(f"Error getting AI response: {e}")  # В случае ошибки выводим сообщение
            return None

# Функция для создания интерфейса с помощью Gradio
def create_gradio_interface():
    assistant = VoiceAssistant()  # Создаем экземпляр голосового ассистента
    
    with gr.Blocks(title="Voice Assistant") as interface:  # Создаем интерфейс с помощью Gradio
        gr.Markdown("# 🎙️ Voice Assistant with gTTS")  # Добавляем заголовок
        
        with gr.Row():  # Создаем строку
            with gr.Column():  # Создаем колонку
                mic_info = gr.Textbox(  # Текстовое поле для отображения информации о микрофоне
                    value=assistant.check_microphone(),
                    label="Microphone Status",
                    interactive=False  # Только для чтения
                )
                
        with gr.Row():  # Создаем строку
            audio_input = gr.Audio(  # Поле для ввода аудио с микрофона
                sources=["microphone"],
                type="numpy",  # Тип данных для ввода
                label="Speak here"
            )
            
        with gr.Row():  # Создаем строку
            output_text = gr.Textbox(label="Conversation")  # Текстовое поле для отображения текста
            output_audio = gr.Audio(label="Assistant's Response")  # Поле для воспроизведения аудио-ответа
            
        # Обработчик изменения аудио-входа
        audio_input.change(
            fn=lambda x: asyncio.run(assistant.process_voice(x[1], x[0])),  # Асинхронно обрабатываем голос
            inputs=[audio_input],  # Входные данные — аудио
            outputs=[output_text, output_audio]  # Выходные данные — текст и аудио
        )
        
    return interface  # Возвращаем интерфейс

# Запуск интерфейса при старте программы
if __name__ == "__main__":  
    interface = create_gradio_interface()  # Создаем интерфейс
    interface.launch(share=True)  # Запускаем интерфейс с возможностью общего доступа
