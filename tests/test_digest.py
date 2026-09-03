from datetime import datetime, timezone, timedelta
from digest import is_ad, filter_exact_and_near_duplicates, prune_3d_memory

def test_ad_detection():
    assert is_ad("Скидка 50% по промокоду: SALE2024 только сегодня!") is True
    assert is_ad("Компания объявляет набор. Реклама. ИНН 7701234567") is True
    assert is_ad("Законопроект принят в третьем чтении erid: 2VtzquvP4") is True
    assert is_ad("Обычная политическая новость без каких-либо интеграций.") is False

def test_exact_and_near_duplicates():
    posts = [
        {"id": "1", "text": "Центральный банк повысил ключевую ставку до 21% годовых."},
        {"id": "2", "text": "Центральный банк повысил ключевую ставку до 21% годовых."},
        {"id": "3", "text": "В Москве открылась новая станция метро."},
    ]
    unique, dupes_count = filter_exact_and_near_duplicates(posts)
    assert len(unique) == 2
    assert dupes_count == 1

def test_prune_3d_memory():
    now = datetime.now(timezone.utc)
    old_record = {"date": (now - timedelta(days=4)).isoformat(), "topic": "Старая новость"}
    fresh_record = {"date": (now - timedelta(days=1)).isoformat(), "topic": "Свежая новость"}
    
    pruned = prune_3d_memory([old_record, fresh_record])
    assert len(pruned) == 1
    assert pruned[0]["topic"] == "Свежая новость"
