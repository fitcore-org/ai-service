from fastapi import FastAPI
from contextlib import asynccontextmanager
from database.db_connect import create_db_and_tables
from routes.routes import router as feedbacks_router
from services.scheduler import start_ml_scheduler, stop_ml_scheduler
from services.forecast_scheduler import start_forecast_scheduler, stop_forecast_scheduler
from tests.run_tests import validate_sentiment_analysis 
from database.seed_data import clear_test_data, create_test_feedbacks
import os
import logging


# Configurar logging para o startup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando FitCore AI Service...")
    
    # 1. Criar tabelas do banco de dados
    create_db_and_tables()
    logger.info("Banco de dados inicializado")
    
    # 2. Popular com dados de teste (seed_data)
    logger.info("Executando seed_data...")
    clear_test_data()
    create_test_feedbacks()
    logger.info("Dados de teste criados com sucesso!")
    
    # 3. Executar validação se solicitado via variável de ambiente
    if os.getenv("RUN_VALIDATE_ON_STARTUP", "false").lower() == "true":
        logger.info("Executando validacao de sentimentos...")
        validate_sentiment_analysis()
        logger.info("Validacao concluida")
    
    # 4. Iniciar agendadores (o ML scheduler vai processar os feedbacks criados)
    logger.info("Iniciando agendadores...")
    start_ml_scheduler()
    start_forecast_scheduler()
    logger.info("Agendadores ativos")
    
    logger.info("FitCore AI Service iniciado com sucesso!")
    
    yield
    
    # Para agendadores
    logger.info("Parando agendadores...")
    stop_ml_scheduler()
    stop_forecast_scheduler()
    logger.info("Aplicacao finalizada")

app = FastAPI(
    title="FitCore AI Service",
    description="Servico de analise de sentimentos e previsoes financeiras",
    version="1.0.0",
    lifespan=lifespan
) # uvicorn main:app --reload

@app.get("/")
async def root():
    return {
        "message": "FitCore AI Service esta funcionando!",
        "version": "1.0.0",
        "features": [
            "Dados de teste carregados automaticamente",
            "Analise de sentimentos com ML",
            "Previsoes financeiras automaticas",
            "Agendadores ativos"
        ],
        "status": "Todos os servicos operacionais"
    }

app.include_router(feedbacks_router)