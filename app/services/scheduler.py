import joblib
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import time
import re
from collections import Counter
from typing import List, Tuple
import nltk
from nltk.corpus import stopwords

from sqlmodel import Session, select, delete
from database.db_connect import engine
from model.models import Feedback, Word_Frequency, SentimentEnum

import traceback

# Download stopwords if not already downloaded
try:
    stopwords.words("portuguese")
except LookupError:
    nltk.download('stopwords')

# --- CONFIGURAÇÕES ---
MODEL_PATH = 'ai_model/sentiment_model.joblib'

class MLSentimentScheduler:
    def __init__(self):
        """Inicializa o scheduler e carrega o modelo uma única vez."""
        print(f"Carregando modelo de '{MODEL_PATH}'...")
        try:
            self.sentiment_model = joblib.load(MODEL_PATH)
            print("Modelo carregado com sucesso.")
        except FileNotFoundError:
            print(f"ERRO: Modelo não encontrado em '{MODEL_PATH}'")
            raise
        
        self.stopwords_pt = set(stopwords.words("portuguese"))
        self.stopwords_pt.update(['pra', 'pro', 'aqui', 'né', 'tá', 'vc', 'voce'])

        # Inicializa o scheduler
        self.scheduler = AsyncIOScheduler()

    def get_most_common_words(self, texts: List[str], top_n: int = 10) -> List[Tuple[str, int]]:
        """Extrai as words mais comuns usando Bag-of-Words."""
        if not texts:
            return []
        
        full_text = ' '.join(texts).lower()
        words = re.findall(r'\b[a-záàâãéèêíìîóòôõúùûç]{3,}\b', full_text)
        filtered_words = [word for word in words if word not in self.stopwords_pt]
        
        word_counts = Counter(filtered_words)
        return word_counts.most_common(top_n)

    def classify_feedbacks(self, session: Session) -> int:
        """Classifica feedbacks pendentes usando o modelo ML."""
        statement = select(Feedback).where(Feedback.sentiment == SentimentEnum.no_analyzed)
        feedbacks_para_processar = session.exec(statement).all()

        if not feedbacks_para_processar:
            print("Nenhum feedback novo para classificar.")
            return 0

        print(f"Classificando {len(feedbacks_para_processar)} novos feedbacks...")
        
        # Processa em lotes para melhor performance
        texts = [feedback.text for feedback in feedbacks_para_processar]
        predictions = self.sentiment_model.predict(texts)
        
        for feedback, prediction in zip(feedbacks_para_processar, predictions):
            # Mapeia predições do modelo (remove aspas se houver)
            prediction_str = str(prediction).strip('"').lower()
            
            if prediction_str == 'positive':
                feedback.sentiment = SentimentEnum.positive
            elif prediction_str == 'negative':
                feedback.sentiment = SentimentEnum.negative
            elif prediction_str == 'neutral':
                feedback.sentiment = SentimentEnum.neutral
            else:
                # Fallback
                feedback.sentiment = SentimentEnum.neutral
                print(f"Predição desconhecida: {prediction} -> usando neutral")
            
            session.add(feedback)
        
        session.commit()
        print("Classificação concluída e salva no banco.")
        return len(feedbacks_para_processar)

    def update_word_frequency(self, session: Session):
        """Atualiza a tabela de frequência de words."""
        
        print("Recalculando frequência de words...")
        
        # Limpa dados antigos
        session.execute(delete(Word_Frequency))
        
        for sentiment in SentimentEnum:
            # Busca textos do sentiment específico
            statement = select(Feedback.text).where(Feedback.sentiment == sentiment)
            textos = [text for text in session.exec(statement).all() if text]
            
            if not textos:
                print(f"  - Nenhum texto encontrado para {sentiment.value}")
                continue
            
            # Calcula words mais comuns
            common_words = self.get_most_common_words(textos, top_n=10)
            
            if common_words:
                print(f"  - {sentiment.value}: {len(common_words)} words encontradas")
                
                # Insere na tabela
                for word, frequency in common_words:
                    wf = Word_Frequency(
                        word=word,
                        sentiment=sentiment,
                        frequency=frequency
                    )
                    session.add(wf)
        
        session.commit()
        print("Tabela de frequência atualizada.")

    def run_analysis_job(self):
        """Executa o job completo de análise."""
        print(f"\n[{time.ctime()}] INICIANDO JOB ML: Análise de Sentiments...")
        
        try:
            with Session(engine) as session:
                # Classifica novos feedbacks
                processed_count = self.classify_feedbacks(session)
                
                # Atualiza frequência de words se novos feedbacks chegaram
                if processed_count > 0:
                    self.update_word_frequency(session)
                else:
                    print("Nenhum feedback novo processado. Frequência de palavras mantida.")
                
                print(f"[{time.ctime()}] JOB CONCLUÍDO! {processed_count} feedbacks processados.")
                
        except Exception as e:
            print(f"ERRO no job: {e}")
            traceback.print_exc() # mostra stack trace completo

    def start_scheduler(self):
        """Inicia o scheduler."""
        # Executa imediatamente na inicialização
        self.run_analysis_job()
        
        # Agenda para rodar a cada 5 min
        self.scheduler.add_job(
            func=self.run_analysis_job,
            trigger=IntervalTrigger(minutes=5), # Periodo curto para demonstração
            id='sentiment_analysis_job',
            name='Análise de Sentiments ML',
            replace_existing=True
        )
        
        self.scheduler.start()
        print("\nML Scheduler ativo com APScheduler.")
        
    def stop_scheduler(self):
        """Para o scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown()
            print("Scheduler parado.")

# Instância global
scheduler_instance = None

def get_scheduler() -> MLSentimentScheduler:
    """Retorna a instância singleton do scheduler."""
    global scheduler_instance
    if scheduler_instance is None:       # Só carrega o modelo na inicializaçção
        scheduler_instance = MLSentimentScheduler()
    return scheduler_instance

def start_ml_scheduler():
    """Inicia o scheduler ML."""
    scheduler = get_scheduler()
    scheduler.start_scheduler()
    return scheduler

def stop_ml_scheduler():
    """Para o scheduler ML."""
    global scheduler_instance
    if scheduler_instance:
        scheduler_instance.stop_scheduler()

# --- EXECUÇÃO ---
if __name__ == "__main__":
    try:
        scheduler = MLSentimentScheduler()
        scheduler.start_scheduler()
        
        # Mantém o programa rodando
        print("Pressione Ctrl+C para parar...")
        async def keep_alive():
            while True:
                await asyncio.sleep(1)
        
        asyncio.run(keep_alive())
        
    except KeyboardInterrupt:
        print("\nParando scheduler...")
        scheduler.stop_scheduler()
        print("Scheduler finalizado.")