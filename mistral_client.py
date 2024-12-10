import aiohttp  # Импортируем библиотеку aiohttp для асинхронных HTTP-запросов
import json  # Импортируем json для работы с JSON данными (хотя в этом коде json не используется напрямую)

class MistralClient:  # Определяем класс для работы с Mistral API
    def __init__(self, api_key: str):  # Конструктор класса, принимает API-ключ
        self.api_key = api_key  # Сохраняем API-ключ как атрибут объекта
        self.base_url = "https://api.mistral.ai/v1"  # Базовый URL для доступа к API Mistral
        self.headers = {  # Заголовки для HTTP-запросов
            "Authorization": f"Bearer {self.api_key}",  # Заголовок авторизации с использованием Bearer токена
            "Content-Type": "application/json"  # Указываем, что мы отправляем и ожидаем данные в формате JSON
        }
    
    # Метод для получения ответа от модели (чат-комплит)
    async def chat_completion(self, messages: list, model: str = "mistral-small-latest"):  # Асинхронный метод для запроса к модели
        async with aiohttp.ClientSession() as session:  # Создаём асинхронную сессию для HTTP-запросов
            async with session.post(  # Отправляем POST-запрос к API
                f"{self.base_url}/chat/completions",  # URL для получения ответов от модели
                headers=self.headers,  # Заголовки запроса
                json={  # Данные для запроса, отправляем в формате JSON
                    "model": model,  # Указываем модель, которую будем использовать (по умолчанию "mistral-small-latest")
                    "messages": messages  # Список сообщений (входные данные для модели)
                }
            ) as response:  # Ожидаем получения ответа от API
                if response.status == 200:  # Проверяем, успешен ли запрос (статус 200 - OK)
                    result = await response.json()  # Извлекаем JSON-ответ
                    return result["choices"][0]["message"]["content"]  # Возвращаем содержимое ответа модели
                else:
                    error_text = await response.text()  # Если статус не 200, получаем текст ошибки
                    raise Exception(f"Error {response.status}: {error_text}")  # Генерируем исключение с сообщением об ошибке
    
    # Метод для получения доступных моделей
    async def get_available_models(self):  # Асинхронный метод для получения списка доступных моделей
        async with aiohttp.ClientSession() as session:  # Создаём асинхронную сессию для HTTP-запросов
            async with session.get(  # Отправляем GET-запрос для получения списка моделей
                f"{self.base_url}/models",  # URL для получения доступных моделей
                headers=self.headers  # Заголовки запроса
            ) as response:  # Ожидаем получения ответа от API
                if response.status == 200:  # Проверяем, успешен ли запрос
                    result = await response.json()  # Извлекаем JSON-ответ
                    return [model["id"] for model in result["data"]]  # Возвращаем список идентификаторов моделей
                else:
                    error_text = await response.text()  # Если статус не 200, получаем текст ошибки
                    raise Exception(f"Error {response.status}: {error_text}")  # Генерируем исключение с сообщением об ошибке
