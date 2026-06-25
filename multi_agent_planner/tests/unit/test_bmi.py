import pytest
from health.bmi import calculate_bmi, WeightGoal


def test_normal_bmi():
    r = calculate_bmi(70, 175)
    assert 18.5 <= r.bmi < 25
    assert r.goal == WeightGoal.MAINTAIN
    assert r.category == "Normal weight"


def test_underweight():
    r = calculate_bmi(45, 175)
    assert r.bmi < 18.5
    assert r.goal == WeightGoal.GAIN


def test_overweight():
    r = calculate_bmi(95, 175)
    assert 25 <= r.bmi < 30
    assert r.goal == WeightGoal.LOSE


def test_obese():
    r = calculate_bmi(120, 175)
    assert r.bmi >= 30
    assert r.goal == WeightGoal.LOSE


def test_ideal_weight_within_normal():
    r = calculate_bmi(70, 175)
    bmi_check = r.ideal_weight_kg / (1.75 ** 2)
    assert 18.5 <= bmi_check < 25
