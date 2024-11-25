from typing import List, Dict
import chromadb
from datetime import datetime
import numpy as np
from faker import Faker
from fastapi import FastAPI, Request, HTTPException
import stripe
import os
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
import colorsys
import asyncio
from fastapi import BackgroundTasks
from collections import defaultdict
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import json
import logging
import sys
from fastapi.logger import logger
from contextlib import asynccontextmanager
from models.database import get_db, BuildingDetails
from sqlalchemy.orm import Session
from fastapi import Depends
from sklearn.decomposition import PCA
from utils.coordinates import generate_2d_coordinates, update_metadata_with_coordinates

load_dotenv()  # Load environment variables from .env file

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup/shutdown events."""
    try:
        logger.info("Starting application...")
        # Test ChromaDB connection
        logger.info("Testing ChromaDB connection...")
        try:
            donation_processor.client.heartbeat()
            logger.info("ChromaDB connection successful")
        except Exception as e:
            logger.error(f"ChromaDB connection failed: {str(e)}")
            raise
        
        # Test Stripe configuration
        logger.info("Testing Stripe configuration...")
        if not stripe.api_key:
            raise ValueError("Stripe API key not configured")
        logger.info("Stripe configuration verified")
        
        logger.info("Application startup complete")
        yield
    finally:
        logger.info("Application shutting down...")

app = FastAPI(lifespan=lifespan)
fake = Faker()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # Ensure logs go to stdout for CloudWatch
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

class DonationProcessor:
    def __init__(self):
        # Create a persistent client with a local directory
        self.client = chromadb.HttpClient(
            ssl=True,
            host='api.trychroma.com',
            tenant='44bcbb14-87f0-4601-9e2f-3bf64104d7c4',
            database='buildinpublicuniversitycampusmap',
            headers={
                'x-chroma-token': os.getenv("CHROMA_API_KEY")
            }
        ) 
        self.buildings = self.client.get_or_create_collection("buildings")
        
        self.last_cluster_update = datetime.now()
        self.cluster_manager = None  # Will be set after initialization
        self.MAX_METADATA_VALUE_SIZE = 32  # ChromaDB's limit in bytes
        
    def _truncate_metadata_value(self, value: str, max_bytes: int = 32) -> str:
        """Truncate a string to fit within byte limit."""
        if not value:
            return ""
        encoded = value.encode('utf-8')
        if len(encoded) <= max_bytes:
            return value
        # Truncate and add ellipsis while staying under byte limit
        return value.encode('utf-8')[:max_bytes-3].decode('utf-8', errors='ignore') + '...'
    
    def process_donation(self, amount: float, donor_id: str, metadata: Dict = None, db: Session = None) -> Dict:
        """Process a new donation and create a building."""
        building_name = self.generate_building_name()
        building_id = f"building_{donor_id}_{int(datetime.now().timestamp())}"
        
        metadata = metadata or {}
        
        # Prepare building metadata with truncated values for ChromaDB
        building_metadata = {
            "name": self._truncate_metadata_value(building_name),
            "donation_amount": amount,
            "donor_id": self._truncate_metadata_value(donor_id),
            "creation_date": datetime.now().isoformat()[:19],
            "region_id": "",
            "donor_name": self._truncate_metadata_value(metadata.get("donor_name", "Anonymous")),
            "building_type": self._truncate_metadata_value(metadata.get("building_type", "Building")),
            "research_question": self._truncate_metadata_value(metadata.get("research_question", "")),
            "website": self._truncate_metadata_value(metadata.get("website", ""))
        }
        
        # Store in ChromaDB
        self.buildings.add(
            documents=[f"Building: {building_name}\nDonor: {donor_id}\nAmount: ${amount:.2f}\nResearch Question: {metadata.get('research_question', '')}\nBuilding Type: {metadata.get('building_type', 'Building')}\nDonor Name: {metadata.get('donor_name', 'Anonymous')}\nWebsite: {metadata.get('website', '')}"],
            metadatas=[building_metadata],
            ids=[building_id]
        )
        
        # Store full text in SQLite
        if db:
            building_details = BuildingDetails(
                building_id=building_id,
                full_research_question=metadata.get("research_question", ""),
                full_building_type=metadata.get("building_type", "Building"),
                full_donor_name=metadata.get("donor_name", "Anonymous"),
                website=metadata.get("website", ""),
                donation_amount=amount,
                creation_date=datetime.now()
            )
            db.add(building_details)
            db.commit()
        
        # Return combined data
        return {
            **building_metadata,
            "full_research_question": metadata.get("research_question", ""),
            "full_building_type": metadata.get("building_type", "Building"),
            "full_donor_name": metadata.get("donor_name", "Anonymous")
        }
    
    def generate_building_name(self) -> str:
        """Generate a unique building name."""
        return f"{fake.last_name()} {fake.random_element(['Hall', 'Center', 'Building', 'Laboratory'])}"
    

    async def process_stripe_event(self, event: stripe.Event, db: Session) -> Dict:
        """Process a Stripe webhook event."""
        print(f"Processing event type: {event.type}")  # Debug logging
        
        if event.type == "checkout.session.completed":
            session = event.data.object
            print(f"Processing completed session: {session.id}")  # Debug logging
            
            # Extract donation details from session
            amount = session.amount_total / 100  # Convert cents to dollars
            donor_id = session.customer or "anonymous"  # Handle case where customer is None
            
            # Extract metadata from the session
            metadata = {
                "donor_name": session.metadata.get("donor_name", "Anonymous"),
                "building_type": session.metadata.get("building_type", "Building"),
                "research_question": session.metadata.get("research_question", ""),
                "website": session.metadata.get("website", "")
            }
            
            return self.process_donation(amount, donor_id, metadata, db)
        elif event.type == "payment_intent.succeeded":
            payment_intent = event.data.object
            print(f"Processing succeeded payment: {payment_intent.id}")  # Debug logging
            
            amount = payment_intent.amount / 100
            donor_id = payment_intent.customer or "anonymous"
            
            return self.process_donation(amount, donor_id, db=db)
            
        return {"status": "ignored", "event_type": event.type}


class Region:
    def __init__(self, name: str, color: str, center: List[float]):
        self.name = name
        self.color = color
        self.center = center
        self.buildings = []
        self.total_donations = 0.0

class ClusterManager:
    def __init__(self, donation_processor: DonationProcessor):
        self.dp = donation_processor
        self.regions = {}
        self.max_clusters = 5
        self.region_names = [
            "Innovation District",
            "Scholar's Quarter",
            "Research Park",
            "Arts Colony",
            "Technology Hub"
        ]
    
    def generate_colors(self, n: int) -> List[str]:
        """Generate visually distinct colors."""
        colors = []
        for i in range(n):
            hue = i / n
            saturation = 0.7
            value = 0.9
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(rgb[0] * 255),
                int(rgb[1] * 255),
                int(rgb[2] * 255)
            )
            colors.append(hex_color)
        return colors

    async def update_clusters(self):
        """Update region assignments for all buildings."""
        # Get all buildings
        buildings = self.dp.buildings.get(include=["embeddings"])
        if not buildings["ids"]:
            return
        
        # Generate coordinates
        coordinates = generate_2d_coordinates(buildings["embeddings"])
        if not coordinates:
            return
        
        # Convert to numpy array for clustering
        coordinates_array = np.array(coordinates)
        
        # Determine number of clusters based on number of buildings
        n_clusters = min(self.max_clusters, len(coordinates_array))
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(coordinates_array)
        
        # Generate colors for regions
        colors = self.generate_colors(n_clusters)
        
        # Create regions
        self.regions = {}
        for i in range(n_clusters):
            self.regions[str(i)] = Region(
                name=self.region_names[i],
                color=colors[i],
                center=kmeans.cluster_centers_[i].tolist()
            )
        
        # Assign buildings to regions and update metadata
        for building_id, meta, label, coords in zip(
            buildings["ids"],
            buildings["metadatas"],
            cluster_labels,
            coordinates
        ):
            region_id = str(label)
            region = self.regions[region_id]
            region.buildings.append(building_id)
            region.total_donations += meta["donation_amount"]
            
            # Update building metadata with region and coordinates
            self.dp.buildings.update(
                ids=[building_id],
                metadatas=[{
                    **meta, 
                    "region_id": region_id,
                    "coordinates": json.dumps(coords)
                }]
            )
        
        return self.get_region_data()
    
    def get_region_data(self) -> Dict:
        """Get current region data."""
        return {
            region_id: {
                "name": region.name,
                "color": region.color,
                "center": region.center,
                "building_count": len(region.buildings),
                "total_donations": region.total_donations
            }
            for region_id, region in self.regions.items()
        }


class DonationRequest(BaseModel):
    amount: float = Field(..., gt=0)
    currency: str = Field(default="usd")
    donor_email: str
    success_url: str
    cancel_url: str
    donor_info: Dict = Field(default_factory=dict)  # Add donor info field

# Initialize processor
donation_processor = DonationProcessor()
cluster_manager = ClusterManager(donation_processor)
donation_processor.cluster_manager = cluster_manager

@app.post("/webhook/stripe")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    """Handle Stripe webhook events."""
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, webhook_secret
        )
        print(f"Received webhook event: {event.type}")  # Debug logging
    except ValueError as e:
        print(f"Invalid payload: {e}")  # Debug logging
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        print(f"Invalid signature: {e}")  # Debug logging
        raise HTTPException(status_code=400, detail="Invalid signature")
        
    try:
        result = await donation_processor.process_stripe_event(event, db)
        print(f"Processed webhook result: {result}")  # Debug logging
        return result
    except Exception as e:
        print(f"Error processing webhook: {e}")  # Debug logging
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create-checkout-session")
async def create_checkout_session(request: DonationRequest):
    """Create a Stripe Checkout session for donation."""
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': request.currency,
                    'product_data': {
                        'name': f'Campus Building Donation - {request.donor_info.get("building_type", "Building")}',
                        'description': 'Your donation will create a new building in our virtual campus',
                    },
                    'unit_amount': int(request.amount * 100),
                },
                'quantity': 1,
            }],
            mode='payment',
            customer_email=request.donor_email,
            success_url=request.success_url,
            cancel_url=request.cancel_url,
            metadata={
                'donation_type': 'building',
                'donor_name': request.donor_info.get('full_name', ''),
                'building_type': request.donor_info.get('building_type', ''),
                'research_question': request.donor_info.get('research_question', '')
            }
        )
        return {"session_id": session.id, "url": session.url}
        
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/regions")
async def get_regions():
    """Get current region data."""
    return cluster_manager.get_region_data()

@app.post("/regions/update")
async def force_update_regions(background_tasks: BackgroundTasks):
    """Force update of region clustering."""
    background_tasks.add_task(cluster_manager.update_clusters)
    return {"status": "update scheduled"}

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint."""
    health_status = {
        "status": "ok",
        "checks": {
            "chromadb": "ok",
            "stripe": "ok"
        }
    }
    
    try:
        donation_processor.client.heartbeat()
    except Exception as e:
        health_status["checks"]["chromadb"] = f"error: {str(e)}"
        health_status["status"] = "error"
    
    if not stripe.api_key:
        health_status["checks"]["stripe"] = "error: API key not configured"
        health_status["status"] = "error"
    
    return health_status

@app.get("/buildings")
async def get_buildings():
    """Get all buildings data."""
    try:
        buildings = donation_processor.buildings.get(include=["embeddings"])
        return update_metadata_with_coordinates(buildings)
    except Exception as e:
        logger.error(f"Error getting buildings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/buildings/{building_id}/details")
async def get_building_details(building_id: str, db: Session = Depends(get_db)):
    """Get full building details including non-truncated text."""
    details = db.query(BuildingDetails).filter(BuildingDetails.building_id == building_id).first()
    if not details:
        raise HTTPException(status_code=404, detail="Building not found")
    return details

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", 8008))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)