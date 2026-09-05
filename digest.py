import os
import re
import json
import time
import html
import logging
import asyncio
import difflib
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

import requests
from telethon import TelegramClient
from telethon.sessions import StringSession

from gemini_client import SemanticDeduplicator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STATE_FILE = "state.json"
CHANNELS_FILE = "channels.txt"
MEMORY_RETENTION_DAYS = 3

# Часовой пояс Перми (UTC+5)
PERM_TZ = timezone(timedelta(hours=5))

# Фильтр рекламы по закону РФ, промокодам и мусорным заглушкам
AD_REGEXES = [
    re.compile(r"erid:\s*[A-Za-z0-9]+", re.IGNORECASE),
    re.compile(r"\bтокен\s+erid\b", re.IGNORECASE),
    re.compile(r"\bинн\s*\d{10,12}\b", re.IGNORECASE),
    re.compile(r"\bогрн\s*\d{13,15}\b", re.IGNORECASE),
    re.compile(r"#реклама\b", re.IGNORECASE),
    re.compile(r"\b(на правах рекламы|партн[её]рский материал|спонсорский пост)\b", re.IGNORECASE),
    re.compile(r"\bпромокод:?\s*[A-Z0-9_-]+\b", re.IGNORECASE),
]

def is_ad_or_junk(text: str) -> bool:
    clean_text = text.strip()
    if len(clean_text) < 45:
        return True
    if ("t.me/+" in clean_text or "t.me/joinchat/" in clean_text) and len(clean_text) < 250:
        return True
    for pattern in AD_REGEXES:
        if pattern.search(clean_text):
            return True
    return False

is_ad = is_ad_or_junk

def send_telegram_messages(bot_token: str, chat_id: str, messages: List[str]):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    safe_chunks = []
    for msg in messages:
        while len(msg) > 4000:
            split_idx = msg.rfind("\n\n", 0, 4000)
            if split_idx == -1:
                split_idx = msg.rfind("\n", 0, 4000)
            if split_idx == -1:
                split_idx = 4000
            safe_chunks.append(msg[:split_idx])
            msg = msg[split_idx:].lstrip()
        safe_chunks.append(msg)

    for chunk in safe_chunks:
        if not chunk.strip():
            continue
        resp = requests.post(url, json={
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }, timeout=20)
        resp.raise_for_status()
        time.sleep(1)

def alert_failure(bot_token: str, chat_id: str, error_msg: str):
    if bot_token and chat_id:
        try:
            safe_err = html.escape(str(error_msg)[:500])
            msg = f"🚨 <b>Сбой в работе 'Чистого дайджеста'</b>\n\nПроцесс завершился с ошибкой:\n<code>{safe_err}</code>"
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}, timeout=10)
        except Exception as e:
            logger.error(f"Не удалось отправить алерт об ошибке: {e}")

def load_state() -> Dict[str, Any]:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"channels": {}, "history_3d": []}

def save_state(state: Dict[str, Any]):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def filter_exact_and_near_duplicates(posts: List[Dict[str, Any]], threshold: float = 0.85) -> (List[Dict[str, Any]], int):
    unique_posts = []
    filtered_count = 0
    for post in posts:
        norm_text = " ".join(post["text"].lower().split())
        is_duplicate = False
        for idx, u_post in enumerate(unique_posts):
            target_text = " ".join(u_post["text"].lower().split())
            similarity = difflib.SequenceMatcher(None, norm_text, target_text).quick_ratio()
            if similarity >= threshold:
                is_duplicate = True
                # Если новая копия текста полнее и длиннее, заменяем её точечно по индексу
                if len(post["text"]) > len(u_post["text"]):
                    unique_posts[idx] = post
                break
        if is_duplicate:
            filtered_count += 1
        else:
            unique_posts.append(post)
    return unique_posts, filtered_count

async def collect_posts(client: TelegramClient, channels: List[str], state: Dict[str, Any]) -> (List[Dict[str, Any]], Dict[str, int]):
    collected = []
    updated_state_channels = dict(state.get("channels", {}))

    for channel in channels:
        channel = channel.strip()
        if not channel or channel.startswith("#"):
            continue

        last_id = updated_state_channels.get(channel, 0)
        try:
            entity = await client.get_entity(channel)
            channel_title = getattr(entity, "title", channel)
            username = getattr(entity, "username", None)

            # Лимит увеличен до 100 сообщений для активных каналов
            messages = await client.get_messages(entity, min_id=last_id, limit=100)
            if not messages:
                continue

            max_id = last_id
            for msg in reversed(messages):
                if msg.id > max_id:
                    max_id = msg.id
                if not msg.text:
                    continue

                post_link = f"https://t.me/{username}/{msg.id}" if username else f"https://t.me/c/{entity.id}/{msg.id}"
                
                collected.append({
                    "id": f"{channel}_{msg.id}",
                    "channel": channel_title,
                    "username": username,
                    "date": msg.date.astimezone(timezone.utc),
                    "text": msg.text.strip(),
                    "link": post_link
                })

            updated_state_channels[channel] = max_id
        except Exception as e:
            logger.error(f"Ошибка сбора из канала {channel}: {e}")

    return collected, updated_state_channels

def prune_3d_memory(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=MEMORY_RETENTION_DAYS)
    valid_records = []
    for item in history:
        try:
            item_date = datetime.fromisoformat(item["date"])
            if item_date >= cutoff:
                valid_records.append(item)
        except Exception:
            continue
    return valid_records

def build_grouped_messages(stats_header: str, posts: List[Dict[str, Any]], max_chars: int = 3500) -> List[str]:
    separator = "\n\n────────────────────\n\n"
    messages = []
    current_message = stats_header

    for p in posts:
        post_date = p["date"].astimezone(PERM_TZ)
        date_str = post_date.strftime("%d.%m %H:%M")
        raw_tag = f"@{p['username']}" if p.get("username") else p["channel"]
        channel_tag = html.escape(raw_tag)

        escaped_text = html.escape(p["text"])
        post_block = (
            f"🕒 {date_str} · {channel_tag}\n"
            f"{escaped_text}\n\n"
            f"Источник: {p['link']}"
        )

        addition = separator + post_block
        if len(current_message) + len(addition) <= max_chars:
            current_message += addition
        else:
            messages.append(current_message)
            current_message = post_block

    if current_message:
        messages.append(current_message)

    return messages

async def main():
    api_id = int(os.environ["TG_API_ID"])
    api_hash = os.environ["TG_API_HASH"]
    session_string = os.environ["TG_SESSION_STRING"]
    bot_token = os.environ["TG_BOT_TOKEN"]
    chat_id = os.environ["TG_CHAT_ID"]
    gemini_key = os.environ["GEMINI_API_KEY"]

    state = load_state()
    
    with open(CHANNELS_FILE, "r", encoding="utf-8") as f:
        channel_names = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    client = TelegramClient(StringSession(session_string), api_id, api_hash)
    await client.connect()
    
    logger.info("Сбор новых постов...")
    raw_posts, updated_channels = await collect_posts(client, channel_names, state)
    await client.disconnect()

    total_collected = len(raw_posts)
    logger.info(f"Собрано {total_collected} постов.")

    if total_collected == 0:
        logger.info("Новых постов нет. Завершение работы.")
        return

    # Фильтр рекламы и спама
    clean_from_ads = []
    ads_count = 0
    for post in raw_posts:
        if is_ad_or_junk(post["text"]):
            ads_count += 1
        else:
            clean_from_ads.append(post)

    # Фильтр точных дублей
    after_exact, exact_dupes_count = filter_exact_and_near_duplicates(clean_from_ads)

    # Фильтр смысловых повторов
    active_history = prune_3d_memory(state.get("history_3d", []))
    past_topics = [h["topic"] for h in active_history]

    deduplicator = SemanticDeduplicator(api_key=gemini_key)
    semantic_result = deduplicator.select_unique_and_best_posts(after_exact, past_topics)

    selected_ids = set(semantic_result["selected_ids"])
    filtered_past_count = semantic_result["filtered_past_count"]
    filtered_semantic_count = semantic_result["filtered_semantic_count"]

    # Формирование итогового пула постов
    final_posts = [p for p in after_exact if p["id"] in selected_ids]
    final_posts.sort(key=lambda p: p["date"])

    # Проверка статуса Gemini
    gemini_status = "Gemini ON✅" if semantic_result.get("gemini_ok", True) else "Gemini OFF❌"

    # Статистика воронки
    stats_header = (
        "❗️❗️❗️❗️❗️❗️❗️❗️\n"
        "📊 <b>Сводка обработки новостей</b>\n"
        f"• Всего постов собрано: <b>{total_collected}</b>\n"
        f"• Отсеяно рекламы и спама: <b>{ads_count}</b>\n"
        f"• Точных повторов: <b>{exact_dupes_count}</b>\n"
        f"• Смысловых дубликатов: <b>{filtered_semantic_count}</b>\n"
        f"• Повторов сюжетов за 3 дня: <b>{filtered_past_count}</b>\n"
        f"• <b>Осталось уникальных новостей: {len(final_posts)}</b>\n"
        f"• <b>{gemini_status}</b>"
    )

    # Отправка сообщений в Telegram
    messages_to_send = build_grouped_messages(stats_header, final_posts)
    send_telegram_messages(bot_token, chat_id, messages_to_send)
    logger.info("Дайджест успешно отправлен.")

    # Обновление и сохранение состояния СТРОГО ПОСЛЕ успешной отправки
    now_iso = datetime.now(timezone.utc).isoformat()
    for topic in semantic_result["new_topics"]:
        active_history.append({"date": now_iso, "topic": topic})

    state["channels"] = updated_channels
    state["history_3d"] = active_history
    save_state(state)

if __name__ == "__main__":
    bot_token = os.getenv("TG_BOT_TOKEN")
    chat_id = os.getenv("TG_CHAT_ID")
    try:
        asyncio.run(main())
    except Exception as e:
        logger.exception("Критическая ошибка пайплайна")
        alert_failure(bot_token, chat_id, str(e))
        raise
        
