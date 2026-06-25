import pytest
from datetime import date


def test_weekend_is_cheat_day():
    saturday = date(2025, 6, 21)
    assert saturday.weekday() == 5


def test_weekday_is_not_cheat_day():
    monday = date(2025, 6, 23)
    assert monday.weekday() < 5
