from fastapi import FastAPI
from contextlib import asynccontextmanager
from database.db_connect import create_db_and_tables
from routes.routes import router as feedbacks_router
from services.scheduler import start_ml_scheduler, stop_ml_scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    start_ml_scheduler()
    yield
    stop_ml_scheduler()

app = FastAPI(lifespan=lifespan) # uvicorn main:app --reload

app.include_router(feedbacks_router)