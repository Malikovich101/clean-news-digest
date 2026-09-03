from datetime import datetime, timezone, timedelta
from digest import is_ad_or_junk, is_ad, filter_exact_and_near_duplicates, prune_3d_memory

def test_ad_and_junk_detection():
    assert is_ad_or_junk("Скидка 50% по промокоду: SALE2026 только сегодня для всех подписчиков!") is True
    assert is_ad_or_junk("Компания открывает новый филиал. Реклама. ИНН 7701234567") is True
    assert is_ad_or_junk("Законопроект принят в третьем чтении erid: 2VtzquvP4") is True
    assert is_ad_or_junk("Короткий пост") is True
    assert is_ad_or_junk("Заходи в наш канал: https://t.me/+q6SAUXqW4fYzMzhi") is True
    assert is_ad_or_junk("Мировые сборы нового фильма превысили 110 миллионов долларов за первый уикенд проката.") is False
    assert is_ad("Мировые сборы нового фильма превысили 110 миллионов долларов за первый уикенд проката.") is False

def test_exact_and_near_duplicates():
    posts = [
        {"id": "1", "text": "Центральный банк повысил ключевую ставку до 21% годовых на заседании."},
        {"id": "2", "text": "Центральный банк повысил ключевую ставку до 21% годовых на заседании."},
        {"id": "3", "text": "В Москве открылась новая станция метро в северном округе столицы."},
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
