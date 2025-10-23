"""
This module sets up the database connection and session management
for the application using SQLAlchemy.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()
TARGET_DB = os.getenv("TARGET_DB")

if TARGET_DB is None:
    raise ValueError("TARGET_DB is not set in the environment or .env file")

DATABASE_URL = TARGET_DB
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
