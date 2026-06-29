"""Pydantic schemas for user auth and health profile."""
from uuid import UUID
from pydantic import BaseModel, EmailStr, Field


class HealthProfile(BaseModel):
    height_cm: float = Field(..., gt=50, lt=300)
    weight_kg: float = Field(..., gt=20, lt=500)
    age: int = Field(..., ge=1, le=120)
    gender: str = Field(..., pattern="^(male|female|other)$")
    profession: str = Field(..., max_length=100)
    diseases: list[str] = Field(default_factory=list)
    disabilities: list[str] = Field(default_factory=list)


class UserSignup(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=2, max_length=100)
    phone_number: str | None = None
    health_profile: HealthProfile


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class UserResponse(BaseModel):
    id: UUID
    email: str
    full_name: str
    is_active: bool
    is_verified: bool

    model_config = {"from_attributes": True}
