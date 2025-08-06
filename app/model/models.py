import uuid
import enum
from datetime import datetime, date
from typing import Optional
from decimal import Decimal

from sqlalchemy import func, Column, DECIMAL

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

# Modelos para os dados financeiros do analytics_db
class Profit(SQLModel, table=True):
    id: str = Field(primary_key=True)
    period_start: date = Field()
    period_end: date = Field()
    total_revenue: float = Field()
    total_expenses: float = Field()
    net_profit: float = Field()
    profit_margin: float = Field()
    created_at: datetime = Field()
    
    class Config:
        table = True

class ProfitForecast(SQLModel, table=True):
    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True
    )
    forecast_date: date = Field(index=True)  # Data da previsão (ds no Prophet)
    predicted_net_profit: float = Field()  # yhat
    lower_bound: float = Field()  # yhat_lower
    upper_bound: float = Field()  # yhat_upper
    model_version: str = Field(default="v1.0")  # Para rastreabilidade
    created_at: datetime = Field(
        default_factory=datetime.now,
        sa_column_kwargs={"server_default": func.now()}
    )

# DTOs para resposta da API
class ForecastResponse(SQLModel):
    forecast_date: date
    predicted_net_profit: float
    lower_bound: float
    upper_bound: float
    confidence_interval: float  

class ForecastSummary(SQLModel):
    total_forecasts: int
    forecast_period_start: date
    forecast_period_end: date
    avg_predicted_profit: float
    model_version: str
    created_at: datetime