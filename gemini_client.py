import json
import logging
from typing import List, Dict, Any
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

class SemanticDeduplicator:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"

    def select_unique_and_best_posts(
        self, 
        candidates: List[Dict[str, Any]], 
        past_topics_3d: List[str]
    ) -> Dict[str, Any]:
        """
        ИИ выступает исключительно арбитром:
        1. Исключает сюжеты, уже встречавшиеся за 3 дня.
        2. Группирует смысловые дубликаты.
        3. Выбирает ID самой полной оригинальной версии.
        """
        if not candidates:
            return {
                "selected_ids": [],
                "new_topics": [],
                "filtered_past_count": 0,
                "filtered_semantic_count": 0,
                "gemini_ok": True
            }

        items_payload = [
            {"id": item["id"], "text": item["text"][:1500]}
            for item in candidates
        ]

        system_instruction = (
            "Ты — строгий редактор-арбитр новостей. Твоя задача — анализировать списки постов.\n"
            "ПРАВИЛА:\n"
            "1. НИКОГДА не переписывай, не сокращай и не сочиняй новости.\n"
            "2. Сверь посты со списком 'УЖЕ ОСВЕЩЕННЫЕ ТЕМЫ ЗА 3 ДНЯ'. Если пост описывает то же самое событие — отбрось его.\n"
            "3. Оставшиеся посты сгруппируй по смысловым событиям.\n"
            "4. В каждой группе выбери ровно ОДИН пост (самый полный, информативный, с фактами).\n"
            "5. Для каждого выбранного поста сформулируй краткую суть события (до 10 слов) для пополнения памяти.\n"
            "6. Верни строго валидный JSON."
        )

        prompt = {
            "recent_3d_topics": past_topics_3d,
            "incoming_posts": items_payload
        }

        schema = {
            "type": "OBJECT",
            "properties": {
                "rejected_as_3d_dupes_count": {"type": "INTEGER"},
                "semantic_groups_merged_count": {"type": "INTEGER"},
                "results": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "selected_post_id": {"type": "STRING"},
                            "topic_summary": {"type": "STRING"}
                        },
                        "required": ["selected_post_id", "topic_summary"]
                    }
                }
            },
            "required": ["rejected_as_3d_dupes_count", "semantic_groups_merged_count", "results"]
        }

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=json.dumps(prompt, ensure_ascii=False),
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.1
                )
            )
            data = json.loads(response.text)
            selected_ids = [item["selected_post_id"] for item in data.get("results", [])]
            new_topics = [item["topic_summary"] for item in data.get("results", [])]
            
            return {
                "selected_ids": selected_ids,
                "new_topics": new_topics,
                "filtered_past_count": data.get("rejected_as_3d_dupes_count", 0),
                "filtered_semantic_count": data.get("semantic_groups_merged_count", 0),
                "gemini_ok": True
            }
        except Exception as e:
            logger.error(f"Ошибка вызова Gemini API: {e}")
            # Fallback: при сбое возвращаем посты без смыслового объединения
            return {
                "selected_ids": [c["id"] for c in candidates],
                "new_topics": [],
                "filtered_past_count": 0,
                "filtered_semantic_count": 0,
                "gemini_ok": False
            }
