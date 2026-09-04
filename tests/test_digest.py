import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Добавляем корень репозитория в sys.path для корректного импорта в runner
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import digest


# ==========================================
# 1. ТЕСТЫ ФИЛЬТРАЦИИ РЕКЛАМЫ И МУСОРА
# ==========================================

def test_short_posts_filtered():
    """Посты короче 45 символов отсекаются как неинформативный мусор."""
    short_text = "Короткий текст новости."
    assert len(short_text) < 45
    check_fn = getattr(digest, "is_ad_or_junk", getattr(digest, "is_ad", None))
    assert check_fn is not None, "Функция фильтрации рекламы не найдена в digest.py"
    assert check_fn(short_text) is True


def test_invite_links_filtered():
    """Инвайт-заглушки (t.me/+ и t.me/joinchat/) блокируются."""
    invite_plus = "Эксклюзивные инсайды в канале: https://t.me/+AbCdEf12345 заходите!"
    invite_join = "Подписывайтесь на закрытый чат https://t.me/joinchat/XYZ123 прямо сейчас!"

    check_fn = getattr(digest, "is_ad_or_junk", getattr(digest, "is_ad", None))
    assert check_fn is not None
    assert check_fn(invite_plus) is True
    assert check_fn(invite_join) is True


def test_clean_news_allowed():
    """Полноценная новость без рекламы успешно проходит фильтр."""
    clean_text = (
        "Правительство утвердило новые параметры бюджета на следующий квартал. "
        "Основное внимание уделено инфраструктурным проектам и поддержке региональных программ."
    )
    check_fn = getattr(digest, "is_ad_or_junk", getattr(digest, "is_ad", None))
    assert check_fn is not None
    assert check_fn(clean_text) is False


# ==========================================
# 2. ТЕСТЫ ДЕДУПЛИКАЦИИ
# ==========================================

def test_exact_and_near_duplicates():
    """Проверка фильтрации текстовых дубликатов."""
    posts = [
        {"id": "1", "text": "Центральный банк повысил ключевую ставку до 21% годовых на заседании."},
        {"id": "2", "text": "Центральный банк повысил ключевую ставку до 21% годовых на заседании."},
        {"id": "3", "text": "В Москве открылась новая станция метро в северном округе столицы."},
    ]
    unique, dupes_count = digest.filter_exact_and_near_duplicates(posts)
    assert len(unique) == 2
    assert dupes_count == 1


# ==========================================
# 3. ТЕСТЫ РОТАЦИИ ПАМЯТИ (3 ДНЯ)
# ==========================================

def test_prune_3d_memory():
    """Удаление записей старше 3-х дней из скользящего окна."""
    now = datetime.now(timezone.utc)
    old_time = (now - timedelta(days=4)).isoformat()
    fresh_time = (now - timedelta(days=1)).isoformat()

    sample_history = [
        {"topic": "Старая новость 4-дневной давности", "date": old_time},
        {"topic": "Свежая вчерашняя новость", "date": fresh_time},
    ]

    result = digest.prune_3d_memory(sample_history)
    assert len(result) == 1
    assert result[0]["topic"] == "Свежая вчерашняя новость"
    
