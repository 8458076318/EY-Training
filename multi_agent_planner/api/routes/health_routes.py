from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from health.bmi import generate_weight_plan, calculate_bmi

router = APIRouter(prefix="/health", tags=["Health & BMI"])


class BMIRequest(BaseModel):
    name: str
    age: int = Field(ge=10, le=100)
    gender: Literal["male", "female", "other"]
    weight_kg: float = Field(ge=20, le=300)
    height_cm: float = Field(ge=100, le=250)
    activity_level: Literal["sedentary","lightly_active","moderately_active","very_active"] = "moderately_active"
    user_id: str = "default"


@router.get("/bmi")
async def bmi_quick(weight_kg: float, height_cm: float):
    r = calculate_bmi(weight_kg, height_cm)
    return {"bmi": r.bmi, "category": r.category, "goal": r.goal.value,
            "ideal_weight_kg": r.ideal_weight_kg, "kg_to_change": r.kg_to_change}


@router.post("/weight-plan")
async def weight_plan(req: BMIRequest):
    try:
        return await generate_weight_plan(
            name=req.name, age=req.age, gender=req.gender,
            weight_kg=req.weight_kg, height_cm=req.height_cm,
            activity_level=req.activity_level,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
