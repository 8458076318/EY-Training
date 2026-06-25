"""
BMI calculator + AI-powered weight goal planner.
"""
import json
import logging
from dataclasses import dataclass
from enum import Enum
from orchestrator.graph import AgentOrchestrator

logger = logging.getLogger(__name__)


class WeightGoal(str, Enum):
    LOSE     = "lose"
    GAIN     = "gain"
    MAINTAIN = "maintain"


@dataclass
class BMIResult:
    bmi: float
    category: str
    goal: WeightGoal
    ideal_weight_kg: float
    kg_to_change: float


def calculate_bmi(weight_kg: float, height_cm: float) -> BMIResult:
    h_m = height_cm / 100
    bmi = round(weight_kg / (h_m ** 2), 1)
    if bmi < 18.5:
        category, goal = "Underweight", WeightGoal.GAIN
    elif bmi < 25.0:
        category, goal = "Normal weight", WeightGoal.MAINTAIN
    elif bmi < 30.0:
        category, goal = "Overweight", WeightGoal.LOSE
    else:
        category, goal = "Obese", WeightGoal.LOSE
    ideal_weight_kg = round(21.75 * (h_m ** 2), 1)
    kg_to_change = round(abs(weight_kg - ideal_weight_kg), 1)
    return BMIResult(bmi=bmi, category=category, goal=goal,
                     ideal_weight_kg=ideal_weight_kg, kg_to_change=kg_to_change)


WEIGHT_PLAN_PROMPT = """
You are a certified Indian fitness coach and dietitian. Profile:
Name: {name}, Age: {age}, Gender: {gender}
Height: {height_cm} cm, Weight: {weight_kg} kg
BMI: {bmi} ({category}), Goal: {goal} weight
Target change: {kg_to_change} kg, Activity: {activity_level}

Create a 4-week personalised plan. Return JSON:
{{
  "summary": "2-sentence summary",
  "weekly_calorie_target": int,
  "protein_g": int, "carbs_g": int, "fats_g": int,
  "weekly_workout_plan": [{{"day":"Monday","workout":"...","duration_mins":45,"intensity":"moderate"}}],
  "dietary_tips": ["tip1","tip2","tip3"],
  "indian_foods_to_favour": ["dal","oats upma"],
  "indian_foods_to_avoid": ["maida","fried snacks"],
  "milestone_weeks": [{{"week":1,"expected_change_kg":0.5,"focus":"..."}}],
  "motivational_note": "personalised note"
}}
Keep all food suggestions Indian. Be realistic and safe.
"""


async def generate_weight_plan(name, age, gender, weight_kg, height_cm, activity_level="moderately_active"):
    bmi_result = calculate_bmi(weight_kg, height_cm)
    orchestrator = AgentOrchestrator()
    prompt = WEIGHT_PLAN_PROMPT.format(
        name=name, age=age, gender=gender, height_cm=height_cm,
        weight_kg=weight_kg, bmi=bmi_result.bmi, category=bmi_result.category,
        goal=bmi_result.goal.value, kg_to_change=bmi_result.kg_to_change,
        activity_level=activity_level,
    )
    raw = await orchestrator.invoke("generate_schedule", prompt)
    try:
        plan = json.loads(raw["result"])
    except Exception:
        plan = {"summary": raw.get("result", ""), "error": "parse_failed"}
    return {
        "bmi": bmi_result.bmi, "category": bmi_result.category,
        "goal": bmi_result.goal.value, "ideal_weight_kg": bmi_result.ideal_weight_kg,
        "kg_to_change": bmi_result.kg_to_change, "plan": plan,
    }
