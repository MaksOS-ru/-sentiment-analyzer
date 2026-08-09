"""
Базовые тесты для preprocess.py.

Зачем тесты нужны: чтобы убедиться, что после любых
изменений в коде функция чистки текста всё ещё работает
правильно — не приходится проверять руками каждый раз.

Запуск: pytest (из корня проекта)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocess import clean_text


def test_lowercase():
    assert clean_text("ОТЛИЧНЫЙ Товар") == "отличный товар"


def test_removes_punctuation():
    assert clean_text("Супер!!! Рекомендую :)") == "супер рекомендую"


def test_removes_extra_spaces():
    assert clean_text("слишком    много   пробелов") == "слишком много пробелов"


def test_removes_digits():
    assert clean_text("купил 5 штук за 100 рублей") == "купил штук за рублей"
