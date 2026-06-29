"""ORM models for users, plans, feedback, and notifications."""
import uuid
from datetime import date, time
from sqlalchemy import Boolean, Date, Float, ForeignKey, Integer, String, Text, Time, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.db.base import Base, TimestampMixin


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)

    # FCM token for push notifications (updated by the frontend on login)
    fcm_token: Mapped[str | None] = mapped_column(String(512))

    # Health profile (collected at signup)
    height_cm: Mapped[float | None] = mapped_column(Float)
    weight_kg: Mapped[float | None] = mapped_column(Float)
    age: Mapped[int | None] = mapped_column(Integer)
    gender: Mapped[str | None] = mapped_column(String(20))
    profession: Mapped[str | None] = mapped_column(String(100))
    diseases: Mapped[list | None] = mapped_column(JSON)        # ["diabetes", "hypertension"]
    disabilities: Mapped[list | None] = mapped_column(JSON)    # ["BP", "Sugar", "Heart"]

    plans: Mapped[list["WeeklyPlan"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    feedbacks: Mapped[list["UserFeedback"]] = relationship(back_populates="user")
    notifications: Mapped[list["NotificationLog"]] = relationship(back_populates="user")


class WeeklyPlan(Base, TimestampMixin):
    __tablename__ = "weekly_plans"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    week_start: Mapped[date] = mapped_column(Date)
    preferences: Mapped[dict] = mapped_column(JSON)
    plan_data: Mapped[dict] = mapped_column(JSON)
    llm_provider: Mapped[str] = mapped_column(String(50))
    agent_version: Mapped[str] = mapped_column(String(20))

    user: Mapped["User"] = relationship(back_populates="plans")
    days: Mapped[list["DayPlan"]] = relationship(back_populates="weekly_plan", cascade="all, delete-orphan")


class DayPlan(Base, TimestampMixin):
    __tablename__ = "day_plans"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    weekly_plan_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("weekly_plans.id"))
    day_name: Mapped[str] = mapped_column(String(10))
    plan_date: Mapped[date] = mapped_column(Date)
    wake_time: Mapped[time | None] = mapped_column(Time)
    sleep_time: Mapped[time | None] = mapped_column(Time)
    breakfast: Mapped[str | None] = mapped_column(Text)
    lunch: Mapped[str | None] = mapped_column(Text)
    dinner: Mapped[str | None] = mapped_column(Text)
    workout: Mapped[str | None] = mapped_column(Text)
    meditation: Mapped[str | None] = mapped_column(Text)
    book_recommendation: Mapped[str | None] = mapped_column(Text)
    gym_time_minutes: Mapped[int | None] = mapped_column(Integer)
    reminders_sent: Mapped[bool] = mapped_column(Boolean, default=False)

    weekly_plan: Mapped["WeeklyPlan"] = relationship(back_populates="days")


class UserFeedback(Base, TimestampMixin):
    __tablename__ = "user_feedbacks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    day_plan_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("day_plans.id"))
    rating: Mapped[int] = mapped_column(Integer)
    feedback_text: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(50))
    processed: Mapped[bool] = mapped_column(Boolean, default=False)
    embedding_id: Mapped[str | None] = mapped_column(String(255))

    user: Mapped["User"] = relationship(back_populates="feedbacks")


class NotificationLog(Base, TimestampMixin):
    __tablename__ = "notification_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    day_plan_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("day_plans.id"))
    channel: Mapped[str] = mapped_column(String(20), default="push")  # always "push" now
    title: Mapped[str] = mapped_column(String(255))
    message: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="delivered")  # delivered / failed / read
    is_read: Mapped[bool] = mapped_column(Boolean, default=False)
    sent_at: Mapped[str | None] = mapped_column(String(50))
    extra_data: Mapped[dict | None] = mapped_column(JSON)  # day_plan_id, event type, etc.

    user: Mapped["User"] = relationship(back_populates="notifications")
