"""Pydantic request/response schemas for the API."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, EmailStr


# ─── Auth ──────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: int
    email: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ─── Conversations ─────────────────────────────────────────────────────────────

class ConversationCreate(BaseModel):
    title: str = "New conversation"
    backend: str = "groq"


class ConversationOut(BaseModel):
    id: int
    title: str
    backend: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    sources: Optional[list[Any]] = None
    created_at: datetime

    model_config = {"from_attributes": True}


# ─── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    conversation_id: Optional[int] = None
    backend: str = "groq"
