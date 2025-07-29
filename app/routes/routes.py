from typing import Sequence
from fastapi import APIRouter
from database.db_connect import SessionDep
from model.models import Feedback, FeedbackCreate
from services.analyzer import (
    create_feedback_service,
    get_all_feedbacks_service,
    get_most_common_words_by_sentiment_service
)
from tests.run_tests import validate_sentiment_analysis

router = APIRouter(prefix="/feedbacks", tags=["feedbacks"])

@router.post("/")
def create_feedback(feedback: FeedbackCreate, session: SessionDep) -> Feedback:
    return create_feedback_service(feedback, session)

@router.get("/")
def get_all_feedbacks(session: SessionDep) -> Sequence[Feedback]:
    return get_all_feedbacks_service(session)

@router.get("/word-frequency")
def most_common_words_by_sentiment(session: SessionDep):
    return get_most_common_words_by_sentiment_service(session)

@router.post("/test/validate-system")
def validate_system(session: SessionDep):
    """Endpoint para validar sistema em produção"""
    return validate_sentiment_analysis()