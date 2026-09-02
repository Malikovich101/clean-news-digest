from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import requests
from google import genai
from telethon.sessions import StringSession
from telethon.sync import TelegramClient

UTC = timezone.utc
PERM_TIMEZONE = timezone(timedelta(hours=5))
TELEGRAM_LIMIT = 3900
MEMORY_HOURS = 72
INITIAL_LOOKBACK_HOURS = 6
RECOVERY_LOOKBACK_HOURS = 9
MIN_TEXT_LENGTH = 20
MAX_AI_POST_CHARS = 1600
MAX_AI_INPUT_CHARS = 48000
AI_BATCH_SIZE = 30
STATE_VERSION = 1
AD_MARKERS = (
    "#реклама", "erid", "на правах рекламы", "рекламная интеграция",
    "партнёрский материал", "партнерский материал",
)
PROMO_PATTERNS = (
    r"\bпромокод\b", r"\bскидк\w*\b", r"\bкуп\w*\b",
    r"\bзаказ\w*\b", r"\bрегистр\w*\b", r"\bрозыгрыш\w*\b",
)
URL_RE = re.compile(r"https?://\S+|t\.me/\S+", re.IGNORECASE)
SLOT_MINUTES = {"morning": 8 * 60 + 5, "afternoon": 14 * 60 + 5, "evening": 20 * 60 + 5}


@dataclass(frozen=True)
class Post:
    id: str
    channel: str
    date: str
    text: str
    url: str


@dataclass
class Stats:
    source_posts: int = 0
    ads: int = 0
    exact_duplicates: int = 0
    semantic_duplicates: int = 0
    original_posts: int = 0


class StateStore:
    def __init__(self, path: str = "state.json") -> None:
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": STATE_VERSION, "watermarks": {}, "completed_slots": {}, "delivered_news": []}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("state.json is unreadable") from exc
        if not isinstance(data, dict):
            raise RuntimeError("state.json must contain an object")
        data.setdefault("version", STATE_VERSION)
        data.setdefault("watermarks", {})
        data.setdefault("completed_slots", {})
        data.setdefault("delivered_news", [])
        return data

    def save(self, data: dict[str, Any]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalized_duplicate_text(text: str) -> str:
    value = normalize_space(text).lower()
    value = URL_RE.sub(" ", value)
    value = re.sub(r"@[a-z0-9_]{5,}", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"#[\wа-яё]+", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def is_explicit_ad(text: str) -> bool:
    value = normalize_space(text).lower()
    return any(marker in value for marker in AD_MARKERS)


def is_suspicious_ad(text: str) -> bool:
    value = normalize_space(text).lower()
    if is_explicit_ad(value):
        return False
    return any(re.search(pattern, value) for pattern in PROMO_PATTERNS)


def source_url(channel: str, message_id: int) -> str:
    return f"https://t.me/{channel.lstrip('@')}/{message_id}"


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def slot_id(local_dt: datetime, name: str) -> str:
    return f"{local_dt.date().isoformat()}:{name}"


def expected_slots(now_local: datetime) -> list[tuple[str, datetime]]:
    midnight = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    return [(name, midnight + timedelta(minutes=minutes)) for name, minutes in SLOT_MINUTES.items()]


def next_pending_slot(now_local: datetime, completed: dict[str, Any]) -> tuple[str, str] | None:
    candidates = []
    for name, dt in expected_slots(now_local):
        if dt <= now_local:
            key = slot_id(now_local, name)
            if key not in completed:
                candidates.append((dt, name, key))
    if not candidates:
        return None
    candidates.sort()
    _, name, key = candidates[0]
    return name, key


def trim_memory(items: Iterable[dict[str, Any]], now: datetime) -> list[dict[str, Any]]:
    cutoff = now - timedelta(hours=MEMORY_HOURS)
    result = []
    for item in items:
        delivered_at = parse_dt(item.get("delivered_at")) if isinstance(item, dict) else None
        if delivered_at is None or delivered_at < cutoff or not item.get("id") or not item.get("text"):
            continue
        result.append({"id": str(item["id"]), "text": normalize_space(str(item["text"]))[:700], "delivered_at": delivered_at.isoformat()})
    result.sort(key=lambda x: x["delivered_at"], reverse=True)
    return result[:500]


def filter_local(posts: list[Post]) -> tuple[list[Post], Stats, list[Post]]:
    stats = Stats(source_posts=len(posts))
    kept, suspicious = [], []
    seen = set()
    for post in posts:
        text = normalize_space(post.text)
        if len(URL_RE.sub("", text)) < MIN_TEXT_LENGTH:
            continue
        if is_explicit_ad(text):
            stats.ads += 1
            continue
        if is_suspicious_ad(text):
            suspicious.append(post)
        key = normalized_duplicate_text(text)
        if key in seen:
            stats.exact_duplicates += 1
            continue
        seen.add(key)
        kept.append(post)
    return kept, stats, suspicious


def make_batches(posts: list[Post]) -> list[list[Post]]:
    batches, current = [], []
    size = 0
    for post in posts:
        item_size = len(json.dumps({"id": post.id, "text": normalize_space(post.text)[:MAX_AI_POST_CHARS]}, ensure_ascii=False))
        if current and (size + item_size > MAX_AI_INPUT_CHARS or len(current) >= AI_BATCH_SIZE):
            batches.append(current)
            current, size = [], 0
        current.append(post)
        size += item_size
    if current:
        batches.append(current)
    return batches


def parse_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            raise
        data = json.loads(match.group())
    if not isinstance(data, dict):
        raise ValueError("Gemini returned non-object JSON")
    return data


def gemini_json(client: genai.Client, prompt: str) -> dict[str, Any]:
    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    return parse_json_object(response.text or "")


def review_ads(client: genai.Client | None, posts: list[Post]) -> tuple[set[str], int]:
    if client is None:
        return set(), 0
    dropped = set()
    calls = 0
    for batch in make_batches(posts):
        if not batch:
            continue
        payload = [{"id": p.id, "text": normalize_space(p.text)[:MAX_AI_POST_CHARS]} for p in batch]
        prompt = (
            "Определи только рекламу в Telegram-публикациях. Рекламой считай публикацию, "
            "основная цель которой — продать или продвинуть товар, услугу, бренд, мероприятие, "
            "промокод или коммерческое предложение. При сомнении НЕ помечай как рекламу. "
            'Верни только JSON формата {"ads":["id"]}.\n' + json.dumps(payload, ensure_ascii=False)
        )
        result = gemini_json(client, prompt)
        calls += 1
        ids = result.get("ads", [])
        allowed = {p.id for p in batch}
        if isinstance(ids, list):
            dropped.update(str(x) for x in ids if str(x) in allowed)
    return dropped, calls


def semantic_deduplicate(client: genai.Client | None, posts: list[Post]) -> tuple[set[str], int]:
    if client is None or len(posts) < 2:
        return set(), 0
    dropped, calls = set(), 0
    for batch in make_batches(posts):
        if len(batch) < 2:
            continue
        payload = [{"id": p.id, "text": normalize_space(p.text)[:MAX_AI_POST_CHARS]} for p in batch]
        prompt = (
            "Найди только смысловые дубли публикаций об одном и том же конкретном событии. "
            "Не объединяй новые развития одной истории. При сомнении оставь обе. Для каждой "
            "группы выбери наиболее полную публикацию. "
            'Верни только JSON формата {"groups":[{"keep":"id","duplicates":["id"]}]}.\n'
            + json.dumps(payload, ensure_ascii=False)
        )
        result = gemini_json(client, prompt)
        calls += 1
        groups = result.get("groups", [])
        allowed = {p.id for p in batch}
        if not isinstance(groups, list):
            continue
        for group in groups:
            if not isinstance(group, dict):
                continue
            keep = str(group.get("keep", ""))
            duplicates = group.get("duplicates", [])
            if keep not in allowed or not isinstance(duplicates, list):
                continue
            for item in duplicates:
                item_id = str(item)
                if item_id in allowed and item_id != keep:
                    dropped.add(item_id)
    return dropped, calls


def filter_memory(posts: list[Post], memory: list[dict[str, Any]]) -> tuple[list[Post], int]:
    known = {normalized_duplicate_text(str(item["text"])) for item in memory if item.get("text")}
    result = [p for p in posts if normalized_duplicate_text(p.text) not in known]
    return result, len(posts) - len(result)


def format_post(post: Post) -> str:
    dt = parse_dt(post.date)
    stamp = dt.astimezone(PERM_TIMEZONE).strftime("%d.%m %H:%M") if dt else post.date
    return f"────────────\n🕒 {stamp} · {post.channel}\n{post.text}\nИсточник: {post.url}"


def split_telegram(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    chunks = []
    while len(text) > limit:
        cut = text.rfind("\n────────────", 0, limit)
        if cut < limit // 2:
            cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        chunks.append(text)
    return chunks


def render_digest(posts: list[Post], stats: Stats, memory_dropped: int) -> str:
    lines = [
        "🗞 ЧИСТЫЙ ДАЙДЖЕСТ",
        f"📊 Исходных постов: {stats.source_posts}; реклама: {stats.ads}; точные повторы: {stats.exact_duplicates}; "
        f"смысловые повторы: {stats.semantic_duplicates}; повторы за 3 дня: {memory_dropped}; уникальных новостей: {len(posts)}",
        "Каждый текст ниже — исходная публикация канала без пересказа и сокращения.",
    ]
    lines.extend(format_post(p) for p in sorted(posts, key=lambda x: x.date))
    return "\n".join(lines).strip()


def require_env() -> None:
    required = ("TG_API_ID", "TG_API_HASH", "TG_SESSION_STRING", "TG_BOT_TOKEN", "TG_CHAT_ID")
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError("Не заданы обязательные secrets: " + ", ".join(missing))
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("Не задан обязательный secret: GEMINI_API_KEY")


def load_channels(path: str = "channels.txt") -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    channels = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    if not channels:
        raise RuntimeError("channels.txt пуст")
    return channels


def telegram_send(text: str) -> None:
    token = os.environ["TG_BOT_TOKEN"]
    chat_id = os.environ["TG_CHAT_ID"]
    for chunk in split_telegram(text):
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": chunk, "disable_web_page_preview": True},
            timeout=30,
        )
        response.raise_for_status()
        if not response.json().get("ok"):
            raise RuntimeError("Telegram Bot API rejected message")


def fetch_posts(client: TelegramClient, channels: list[str], state: dict[str, Any], since: datetime) -> list[Post]:
    posts = []
    for channel in channels:
        entity = client.get_entity(channel)
        watermark = int(state.get("watermarks", {}).get(channel, 0))
        channel_posts = []
        for message in client.iter_messages(entity, min_id=watermark, reverse=True):
            if not message.date:
                continue
            date = message.date.astimezone(UTC)
            if date < since:
                continue
            text = message.message or ""
            if not text.strip():
                continue
            channel_posts.append(
                Post(str(f"{channel}:{message.id}"), channel, date.isoformat(), text, source_url(channel, int(message.id)))
            )
        posts.extend(channel_posts)
        if channel_posts:
            state.setdefault("watermarks", {})[channel] = max(int(p.id.rsplit(":", 1)[1]) for p in channel_posts)
    return posts


def main() -> None:
    require_env()
    now_utc = datetime.now(UTC)
    now_local = now_utc.astimezone(PERM_TIMEZONE)
    store = StateStore()
    state = store.load()
    state["delivered_news"] = trim_memory(state.get("delivered_news", []), now_utc)

    pending = next_pending_slot(now_local, state["completed_slots"])
    if pending is None:
        print("Нет незавершённого ожидаемого слота.")
        return
    slot_name, slot_key = pending

    channels = load_channels()
    has_watermarks = bool(state.get("watermarks"))
    lookback_hours = RECOVERY_LOOKBACK_HOURS if has_watermarks else INITIAL_LOOKBACK_HOURS
    since = now_utc - timedelta(hours=lookback_hours)

    telegram = TelegramClient(
        StringSession(os.environ["TG_SESSION_STRING"]),
        int(os.environ["TG_API_ID"]),
        os.environ["TG_API_HASH"],
    )
    telegram.connect()
    try:
        posts = fetch_posts(telegram, channels, state, since)
    finally:
        telegram.disconnect()

    kept, stats, suspicious = filter_local(posts)
    ai_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    ad_ids, _ = review_ads(ai_client, suspicious)
    if ad_ids:
        kept = [p for p in kept if p.id not in ad_ids]
        stats.ads += len(ad_ids)

    semantic_ids, _ = semantic_deduplicate(ai_client, kept)
    if semantic_ids:
        kept = [p for p in kept if p.id not in semantic_ids]
        stats.semantic_duplicates = len(semantic_ids)

    kept, memory_dropped = filter_memory(kept, state["delivered_news"])
    stats.original_posts = len(kept)

    # A slot is completed only after Telegram accepts the whole digest.
    telegram_send(render_digest(kept, stats, memory_dropped))

    state["completed_slots"][slot_key] = now_utc.isoformat()
    state["delivered_news"].extend({"id": p.id, "text": p.text, "delivered_at": now_utc.isoformat()} for p in kept)
    state["delivered_news"] = trim_memory(state["delivered_news"], now_utc)
    state["last_successful_run"] = now_utc.isoformat()
    store.save(state)
    print(f"Слот {slot_name} завершён: {len(kept)} новостей.")


if __name__ == "__main__":
    main()
