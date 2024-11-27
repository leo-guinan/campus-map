from sqlalchemy import create_engine, Column, String, Float, DateTime, Boolean, ForeignKey, Table
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import os
from datetime import datetime
from dotenv import load_dotenv
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

def parse_db_url(url: str) -> str:
    """Parse and validate database URL."""
    if not url:
        raise ValueError("Database URL is not set")
        
    # Convert postgres:// to postgresql://
    if url.startswith('postgres://'):
        url = url.replace('postgres://', 'postgresql://', 1)
    
    # Validate URL format
    try:
        result = urlparse(url)
        if not all([result.scheme, result.hostname, result.username, result.password]):
            raise ValueError("Invalid database URL format")
    except Exception as e:
        raise ValueError(f"Failed to parse database URL: {str(e)}")
        
    return url

# Get and parse database URL
DATABASE_URL = parse_db_url(os.getenv("DATABASE_URL"))

# Configure engine arguments
engine_args = {
    "pool_recycle": 1800,  # Recycle connections after 30 mins
    "pool_size": 5,        # Default number of connections
    "max_overflow": 10,    # Allow up to 10 connections beyond pool_size
    "pool_timeout": 30     # Seconds to wait for available connection
}

# Create engine
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
    
    # Add relationship to regions
    regions = relationship("Region", secondary="building_regions", back_populates="buildings")

class Region(Base):
    __tablename__ = "regions"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    color = Column(String, nullable=False)
    center = Column(ARRAY(Float), nullable=False)
    is_hardcoded = Column(Boolean, nullable=False, default=False)
    description = Column(String)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Add relationship to buildings
    buildings = relationship("BuildingDetails", secondary="building_regions", back_populates="regions")

# Junction table for many-to-many relationship
building_regions = Table('building_regions',
    Base.metadata,
    Column('building_id', String, ForeignKey('building_details.building_id'), primary_key=True),
    Column('region_id', String, ForeignKey('regions.id'), primary_key=True),
    Column('created_at', DateTime, nullable=False, default=datetime.utcnow)
)

def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 