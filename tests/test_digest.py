import json
from datetime import datetime, timedelta, timezone

from digest import (
    Post,
    StateStore,
    filter_local,
    next_pending_slot,
    normalized_duplicate_text,
    split_telegram,
    trim_memory,
)


def test_normalized_duplicate_text_removes_urls_and_markup():
    a = "Новости: тест! https://example.com #tag"
    b = "Новости тест"
    assert normalized_duplicate_text(a) == normalized_duplicate_text(b)


def test_filter_local_removes_explicit_and_exact_duplicates():
    posts = [
        Post("a:1", "@a", "2026-09-02T05:00:00+00:00", "Одна важная новость сегодня", "https://t.me/a/1"),
        Post("b:2", "@b", "2026-09-02T05:01:00+00:00", "Одна важная новость сегодня!!!", "https://t.me/b/2"),
        Post("a:3", "@a", "2026-09-02T05:02:00+00:00", "#реклама Купите курс прямо сейчас", "https://t.me/a/3"),
    ]
    kept, stats, suspicious = filter_local(posts)
    assert len(kept) == 1
    assert stats.exact_duplicates == 1
    assert stats.ads == 1


def test_pending_slots_returns_earliest_uncompleted():
    tz = timezone(timedelta(hours=5))
    now = datetime(2026, 9, 2, 16, 0, tzinfo=tz)
    pending = next_pending_slot(now, {})
    assert pending == ("morning", "2026-09-02:morning")


def test_trim_memory_keeps_only_last_72_hours():
    now = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)
    items = [
        {"id": "new", "text": "new", "delivered_at": (now - timedelta(hours=2)).isoformat()},
        {"id": "old", "text": "old", "delivered_at": (now - timedelta(hours=80)).isoformat()},
    ]
    result = trim_memory(items, now)
    assert [x["id"] for x in result] == ["new"]


def test_split_telegram_respects_limit():
    text = "x" * 8000
    chunks = split_telegram(text, 3900)
    assert len(chunks) == 3
    assert all(len(chunk) <= 3900 for chunk in chunks)
