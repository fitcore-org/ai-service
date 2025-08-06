import joblib
import asyncio
import re
import unicodedata
import logging
from datetime import datetime
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
import os

# Download stopwords if not already downloaded
try:
    stopwords.words("portuguese")
except LookupError:
    nltk.download('stopwords')

# --- CONFIGURAÇÕES ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../ai_model/sentiment_model.joblib")
MODEL_PATH = os.path.abspath(MODEL_PATH)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MLSentimentScheduler:
    def __init__(self):
        """Inicializa o scheduler e carrega o modelo uma única vez."""
        logger.info(f"Carregando modelo de '{MODEL_PATH}'...")
        try:
            self.sentiment_model = joblib.load(MODEL_PATH)
            logger.info("Modelo carregado com sucesso.")
        except FileNotFoundError:
            logger.error(f"ERRO: Modelo não encontrado em '{MODEL_PATH}'")
            raise
        
        self.stopwords_pt = set(stopwords.words("portuguese"))
        self.stopwords_pt.update(['pra', 'pro', 'aqui', 'né', 'tá', 'vc', 'voce'])

        # Stopwords específicas para o domínio de academia
        self.domain_stopwords = {
            # Temporal e lugar
            'hoje', 'ontem', 'amanhã', 'agora', 'sempre', 'antes', 'depois',
            'aqui', 'lá', 'ali', 'cá', 'toda', 'todo', 'dias', 'vezes',

            # Intensidade / quantificadores
            'muito', 'pouco', 'mais', 'menos', 'bem', 'bastante', 'tão', 'tudo', 'nada', 'algum', 'alguma', 'alguns', 'algumas',

            # Opinião vazia / pouco útil
            'bom', 'boa', 'ótimo', 'ótima', 'legal', 'ruim', 'mediana', 'excelente', 'péssimo', 'ok', 'normal',
            'adoro', 'adorável', 'gosto', 'gostei', 'amo', 'incrível', 'amei', 'amei!', 'massa', 'top', 'show', 'perfeito',

            # Genéricos e pronomes
            'algo', 'alguém', 'coisa', 'tipo', 'gente', 'ninguém', 'nada', 'tudo', 'isso', 'aquilo', 'esse', 'essa', 'isso', 'aquela', 'aquele',

            # Verbos comuns (que sozinhos não indicam sentimento)
            'ser', 'estar', 'ter', 'foi', 'vai', 'tá', 'era', 'fica', 'ficar', 'parece', 'tem', 'deu', 'dá', 'vai', 'estava', 'está',

            # Interjeições e expressões informais
            'ufa', 'kkk', 'rs', 'haha', 'eh', 'ah', 'ai', 'eita', 'ixi', 'aff', 'hum', 'nossa', 'vish',

            # Palavras de conexão e estrutura
            'pra', 'pro', 'por', 'com', 'sem', 'mas', 'porque', 'que', 'se', 'em', 'no', 'na', 'nos', 'nas', 'ao', 'aos', 'às', 'de', 'do', 'da', 'dos', 'das', 'e', 'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas',

            # Vocativos e termos de atendimento
            'vc', 'você', 'voce', 'obrigado', 'obrigada', 'valeu', 'atendimento', 'cliente', 'pessoal',
        }


        # Inicializa o scheduler
        self.scheduler = AsyncIOScheduler()

    def normalize_text(self, text: str) -> str:
        """
        Normaliza texto para predição (mesma função usada no treinamento).
        """
        # Converte para minúsculas
        text = text.lower()
        
        # Remove acentos mantendo caracteres especiais do português
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
        
        # Padroniza gírias e expressões comuns
        girias_map = {
            r'\bmassa\b': 'bom',
            r'\btop\b': 'bom', 
            r'\bshow\b': 'bom',
            r'\barretado\b': 'bom',
            r'\baff\b': 'ruim',
            r'\bvish\b': 'ruim',
            r'\brs\b': '',  # Remove risos
            r'\bkkk+\b': '',  # Remove risos
            r'\bhaha+\b': '',  # Remove risos
            r'\bvc\b': 'voce',
            r'\bpra\b': 'para',
            r'\bpro\b': 'para',
            r'\bne\b': 'nao e',
            r'\bta\b': 'esta'
        }
        
        for padrao, substituicao in girias_map.items():
            text = re.sub(padrao, substituicao, text)
        
        # Remove pontuação excessiva (mantém apenas . , ! ?)
        text = re.sub(r'[^\w\s.!?,-]', ' ', text)
        
        # Normaliza pontuação repetida
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Remove números isolados
        text = re.sub(r'\b\d+\b(?!\w)', '', text)
        
        # Normaliza espaços múltiplos
        text = re.sub(r'\s+', ' ', text)
        
        # Remove espaços no início e fim
        text = text.strip()
        
        return text

    def simple_stem(self, word: str) -> str:
        # Corrigir plurais comuns com 'es' no final
        if word.endswith('ões'):
            return word[:-3] + 'ão'  # avaliações -> avaliação
        elif word.endswith('ães'):
            return word[:-3] + 'ão'  # refrigerações -> refrigeração
        elif word.endswith('es') and len(word) > 4:
            return word[:-2]         # Fallback simples que remove 'es' mas cuidado com casos irregulares
        elif word.endswith('s') and len(word) > 4:
            return word[:-1]        # instrutoras -> instrutora, equipamentos -> equipamento
        elif word.endswith('mente'):
            return word[:-5]
        elif word.endswith('ções'):
            return word[:-4] + 'ção'
        return word

    def get_most_common_words(self, texts: List[str], top_n: int = 10) -> List[Tuple[str, int]]:
        """Extrai as palavras mais comuns usando Bag-of-Words com normalização consistente."""
        if not texts:
            return []
        
        # Normaliza todos os textos usando a mesma função do modelo
        normalized_texts = [self.normalize_text(text) for text in texts]
        full_text = ' '.join(normalized_texts)
        
        # Extrai palavras de pelo menos 3 caracteres (já normalizadas)
        words = re.findall(r'\b[a-z]{3,}\b', full_text)
        
        # Filtra stopwords e aplica stemming
        filtered_words = [
            self.simple_stem(word) for word in words
            if word not in self.stopwords_pt and word not in self.domain_stopwords
        ]
        
        word_counts = Counter(filtered_words)
        return word_counts.most_common(top_n)

    def classify_feedbacks(self, session: Session) -> int:
        """Classifica feedbacks pendentes usando o modelo ML."""
        statement = select(Feedback).where(Feedback.sentiment == SentimentEnum.no_analyzed)
        feedbacks_para_processar = session.exec(statement).all()

        if not feedbacks_para_processar:
            logger.info("Nenhum feedback novo para classificar.")
            return 0

        logger.info(f"Classificando {len(feedbacks_para_processar)} novos feedbacks...")
        
        # Normaliza os textos antes da predição
        texts = [self.normalize_text(feedback.text) for feedback in feedbacks_para_processar]
        
        # Processa em lotes para melhor performance
        predictions = self.sentiment_model.predict(texts)
        
        # Obter probabilidades para logging de confiança
        prediction_probabilities = self.sentiment_model.predict_proba(texts)
        
        # Contar predições por classe para logging
        prediction_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        confidence_scores = []
        
        for feedback, prediction, probabilities in zip(feedbacks_para_processar, predictions, prediction_probabilities):
            # Mapeia predições do modelo (remove aspas se houver)
            prediction_str = str(prediction).strip('"').lower()
            
            # Calcula a confiança (probabilidade máxima)
            max_confidence = max(probabilities)
            confidence_scores.append(max_confidence)
            
            if prediction_str == 'positive':
                feedback.sentiment = SentimentEnum.positive
                prediction_counts['positive'] += 1
            elif prediction_str == 'negative':
                feedback.sentiment = SentimentEnum.negative
                prediction_counts['negative'] += 1
            elif prediction_str == 'neutral':
                feedback.sentiment = SentimentEnum.neutral
                prediction_counts['neutral'] += 1
            else:
                # Fallback
                feedback.sentiment = SentimentEnum.neutral
                prediction_counts['neutral'] += 1
                logger.warning(f"Predição desconhecida: {prediction} -> usando neutral")
            
            session.add(feedback)
        
        session.commit()
        
        # Log estatísticas resumidas
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        logger.info(f"Classificacao: {prediction_counts['positive']} positivos, {prediction_counts['negative']} negativos, {prediction_counts['neutral']} neutros")
        logger.info(f"Confianca media: {avg_confidence:.3f}")
        
        logger.info("Classificacao concluida e salva no banco.")
        return len(feedbacks_para_processar)

    def update_word_frequency(self, session: Session):
        """Atualiza a tabela de frequência de palavras."""
        
        # Limpa dados antigos
        session.execute(delete(Word_Frequency))
        
        total_words_inserted = 0
        
        for sentiment in SentimentEnum:
            # Busca textos do sentiment específico
            statement = select(Feedback.text).where(Feedback.sentiment == sentiment)
            textos = [text for text in session.exec(statement).all() if text]
            
            if not textos:
                continue
            
            # Calcula palavras mais comuns
            common_words = self.get_most_common_words(textos, top_n=10)
            
            if common_words:
                
                # Insere na tabela
                for word, frequency in common_words:
                    wf = Word_Frequency(
                        word=word,
                        sentiment=sentiment,
                        frequency=frequency
                    )
                    session.add(wf)
                    total_words_inserted += 1
        
        session.commit()
        logger.info(f"Palavras atualizadas: {total_words_inserted} inseridas.")

    def run_analysis_job(self):
        """Executa o job completo de análise."""
        start_time = datetime.now()
        logger.info(f"Executando analise de sentimentos...")
        
        try:
            with Session(engine) as session:
                # Classifica novos feedbacks
                processed_count = self.classify_feedbacks(session)
                
                # Atualiza frequência de palavras se novos feedbacks chegaram
                if processed_count > 0:
                    self.update_word_frequency(session)
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                logger.info(f"Job concluido: {processed_count} feedbacks em {duration:.2f}s")
                
        except Exception as e:
            logger.error(f"ERRO no job: {e}")
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
            name='Analise de Sentimentos ML',
            replace_existing=True
        )
        
        self.scheduler.start()
        logger.info("ML Scheduler ativo.")
        
    def stop_scheduler(self):
        """Para o scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Scheduler parado.")

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
