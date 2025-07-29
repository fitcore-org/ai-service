from fastapi import FastAPI
from contextlib import asynccontextmanager
from database.db_connect import create_db_and_tables
from routes.routes import router as feedbacks_router
from services.scheduler import start_ml_scheduler, stop_ml_scheduler
from tests.run_tests import validate_sentiment_analysis 
import os


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    if os.getenv("RUN_VALIDATE_ON_STARTUP", "false").lower() == "true": # Popular as tables com as classificações das frases
        validate_sentiment_analysis()
    start_ml_scheduler()
    yield
    stop_ml_scheduler()

app = FastAPI(lifespan=lifespan) # uvicorn main:app --reload

app.include_router(feedbacks_router)