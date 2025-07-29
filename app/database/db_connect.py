import os
from sqlmodel import create_engine, Session, SQLModel
from typing import Annotated
from fastapi import Depends

db_url = os.getenv("ANALYTIC_DATABASE_URL", "postgresql+psycopg2://analytics_user:analytics_pass@localhost:5433/analytics_db")
connect_args = {"check_same_thread": False}
engine = create_engine(db_url, echo=False)

def get_session():
    with Session(engine) as session:
        yield session

def create_db_and_tables():  # Creating and populating the tables in inicialization
    SQLModel.metadata.create_all(engine)

SessionDep = Annotated[Session, Depends(get_session)]