from sqlalchemy import create_engine, Column, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Get database URL from environment variable, with a fallback for local development
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./campus.db")

# Handle special case for Postgres URLs from some hosting providers
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

is_prod = DATABASE_URL.startswith("postgresql://")

engine_args = {
    "pool_recycle": 1800,  # Recycle connections after 30 mins
}

if is_prod:
    engine_args.update({
        "pool_size": 5,      # Default number of connections
        "max_overflow": 10,  # Allow up to 10 connections beyond pool_size
        "pool_timeout": 30   # Seconds to wait for available connection
    })

engine = create_engine(DATABASE_URL, **engine_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class BuildingDetails(Base):
    __tablename__ = "building_details"
    
    building_id = Column(String, primary_key=True)
    full_research_question = Column(String)
    full_building_type = Column(String)
    full_donor_name = Column(String)
    website = Column(String)
    donation_amount = Column(Float)
    creation_date = Column(DateTime)

def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 