import json
import time
import logging
from typing import List, Dict, Any
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

class SemanticDeduplicator:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        # Расширенный пул моделей: перегрузка одной не остановит пайплайн
        # Актуальные рабочие модели (по таблице пользователя)
        self.models_pool = [
            "gemini-3.6-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.7-flash"
        ]

    def select_unique_and_best_posts(
        self,
        candidates: List[Dict[str, Any]],
        past_topics_3d: List[str]
    ) -> Dict[str, Any]:
        if not candidates:
            return {
                "selected_posts": [],
                "rejected_as_3d_duplicates": [],
                "rejected_as_semantic_duplicates": [],
                "gemini_ok": True
            }

        items_payload = [
            {"id": item["id"], "text": item["text"][:1500]}
            for item in candidates
        ]

        system_instruction = (
            "Ты — строгий редактор-арбитр новостей. Твоя задача — анализировать списки Telegram-постов.\n"
            "Критически важное правило: ОДИНАКОВАЯ ТЕМА ≠ ОДИНАКОВОЕ СОБЫТИЕ.\n"
            "ПРАВИЛА (строго соблюдать):\n"
            "1. НИКОГДА не переписывай, не сокращай и не сочиняй новости.\n"
            "2. Перед тобой список 'УЖЕ ОСВЕЩЁННЫЕ СОБЫТИЯ ЗА 3 ДНЯ' (recent_3d_events).\n"
            "   Если пост описывает ТО ЖЕ САМОЕ КОНКРЕТНОЕ СОБЫТИЕ (а не просто похожую тему) — добавь его ID в rejected_as_3d_duplicates.\n"
            "3. Оставшиеся посты сгруппируй по КОНКРЕТНЫМ СОБЫТИЯМ.\n"
            "   Объединяй ТОЛЬКО если несколько каналов сообщают ОДНО И ТО ЖЕ СОБЫТИЕ своими словами.\n"
            "   Пример объединения: 'Apple представила iPhone 18' и 'Apple показала новый iPhone 18'.\n"
            "   Пример НЕ объединять: 'Apple представила iPhone 18' и 'iPhone 18 поступил в продажу в России' — это разные события.\n"
            "4. В каждой группе выбери ровно ОДИН пост (самый полный, информативный, с фактами, меньше рекламы/эмоций).\n"
            "   Добавь выбранный пост в selected_posts с ключами: selected_post_id (ID поста) и topic_summary (краткая суть события, до 12 слов, конкретное событие, не общая тема).\n"
            "5. Если есть сомнение, что два поста — одно событие — НЕ объединяй их. Лучше оставить две похожие новости, чем потерять отдельный инфоповод.\n"
            "6. Для каждого удалённого смыслового дубля добавь его пост-ID в rejected_as_semantic_duplicates (только дубли, не выбранные посты).\n"
            "7. Верни строго валидный JSON с тремя массивами: selected_posts, rejected_as_3d_duplicates, rejected_as_semantic_duplicates.\n"
            "8. Формулировка topic_summary должна отвечать на вопрос: 'Какое конкретное событие?' (например, 'Apple представила iPhone 18'), а не 'О чём тема?' (не 'Apple и технологии')."
        )

        prompt = {
            "recent_3d_events": past_topics_3d,
            "incoming_posts": items_payload
        }

        schema = {
            "type": "OBJECT",
            "properties": {
                "selected_posts": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "selected_post_id": {"type": "STRING"},
                            "topic_summary": {"type": "STRING"}
                        },
                        "required": ["selected_post_id", "topic_summary"]
                    }
                },
                "rejected_as_3d_duplicates": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"}
                },
                "rejected_as_semantic_duplicates": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"}
                }
            },
            "required": ["selected_posts", "rejected_as_3d_duplicates", "rejected_as_semantic_duplicates"]
        }

        safety_settings = [
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        ]

        # Каскадный обход моделей с возрастающими паузами (5с и 10с)
        for model_name in self.models_pool:
            for attempt in range(1, 3):
                try:
                    logger.info(f"Запрос к {model_name} (попытка {attempt})...")
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=json.dumps(prompt, ensure_ascii=False),
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            response_mime_type="application/json",
                            response_schema=schema,
                            safety_settings=safety_settings,
                            temperature=0.1
                        )
                    )
                    data = json.loads(response.text)

                    selected_posts = data.get("selected_posts", [])
                    rejected_3d = data.get("rejected_as_3d_duplicates", [])
                    rejected_semantic = data.get("rejected_as_semantic_duplicates", [])

                    selected_ids = [item["selected_post_id"] for item in selected_posts]
                    new_topics = [item["topic_summary"] for item in selected_posts]

                    return {
                        "selected_ids": selected_ids,
                        "new_topics": new_topics,
                        "filtered_past_count": len(rejected_3d),
                        "filtered_semantic_count": len(rejected_semantic),
                        "gemini_ok": True,
                        # Также сохраняем явные списки для надёжности
                        "rejected_semantic_ids": rejected_semantic,
                        "rejected_3d_ids": rejected_3d,
                    }
                except Exception as e:
                    delay = attempt * 5  # 5 секунд в первой попытке, 10 секунд во второй
                    logger.warning(f"Сбой модели {model_name} (попытка {attempt}): {e}. Пауза {delay}с...")
                    time.sleep(delay)

        logger.error("Все попытки обращения к пулу моделей Gemini исчерпаны.")
        # При полном отказе: не удаляем ничего, выбираем всё (безопасный режим)
        return {
            "selected_ids": [c["id"] for c in candidates],
            "new_topics": [f"Событие из поста {c['id']}" for c in candidates],
            "filtered_past_count": 0,
            "filtered_semantic_count": 0,
            "gemini_ok": False,
            "rejected_semantic_ids": [],
            "rejected_3d_ids": [],
        }
