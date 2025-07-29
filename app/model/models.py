import uuid
import enum
from datetime import datetime
from typing import Optional

from sqlalchemy import func

from sqlmodel import Field, SQLModel

class SentimentEnum(str, enum.Enum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"
    no_analyzed = "no_analyzed"

class FeedbackBase(SQLModel):
    text: str

class FeedbackCreate(FeedbackBase):
    pass

class Feedback(FeedbackBase, table=True):
    id: uuid.UUID = Field(
        default_factory=uuid.uuid4, 
        primary_key=True
    )
    text: str = Field(
        default= 'Nothing', 
        max_length=255
    )
    sentiment: SentimentEnum | None = Field(
        default=SentimentEnum.no_analyzed
    )
    score_compound: float | None = Field(default=0.0)

    created_at: Optional[datetime] = Field(
        default=None,
        sa_column_kwargs={"server_default": func.now()}
    )

class Word_Frequency(SQLModel, table=True):
    id: uuid.UUID = Field(
        default_factory=uuid.uuid4, 
        primary_key=True
    )
    word: str = Field(
        default= 'Nothing', 
        max_length=50
    )
    sentiment: SentimentEnum = Field(
        default=SentimentEnum.neutral
    )
    frequency: int = Field(
        default=0
    )

    created_at: Optional[datetime] = Field(
        default=None,
        sa_column_kwargs={"server_default": func.now()}
    )