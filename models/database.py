from sqlalchemy import create_engine, Column, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Get database URL from environment variable, with a fallback for local development
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./campus.db")

# Handle special case for Postgres URLs from some hosting providers
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Configure SQLAlchemy
engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,  # Recycle connections after 30 minutes
)
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