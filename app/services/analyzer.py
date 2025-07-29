from sqlmodel import Session, select
from sqlalchemy import desc
from model.models import Feedback, FeedbackCreate, Word_Frequency
from database.db_connect import SessionDep
from typing import Sequence

def create_feedback_service(feedback_data: FeedbackCreate, session: SessionDep) -> Feedback:
    feedback_obj = Feedback(text=feedback_data.text) # sentiment fica None
    session.add(feedback_obj)
    session.commit()
    session.refresh(feedback_obj)
    return feedback_obj

def get_all_feedbacks_service(session: Session) -> Sequence[Feedback]:
    return session.exec(select(Feedback)).all()

def get_most_common_words_by_sentiment_service(session: Session) -> dict:
    """Retorna as palavras mais frequentes já calculadas pelo scheduler"""
    word_frequencies = session.exec(
        select(Word_Frequency).order_by(desc(Word_Frequency.sentiment))
    ).all()

    result = {
        "positive": [],
        "negative": [],
        "neutral": []
    }

    for wf in word_frequencies:
        if wf.sentiment.value in result:
            result[wf.sentiment.value].append({
                "word": wf.word,
                "frequency": wf.frequency
            })
    
    return result

def populating_word_frequency_service(session: Session):
    """
    Função de compatibilidade para testes.
    Executa o job do scheduler uma única vez para popular a tabela.
    """
    from services.scheduler import get_scheduler
    
    print("Executando análise ML para popular tabela de frequência...")
    scheduler = get_scheduler()
    scheduler.run_analysis_job()
    print("Análise concluída.")