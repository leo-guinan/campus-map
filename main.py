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

load_dotenv()  # Load environment variables from .env file

app = FastAPI()
fake = Faker()

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
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        self.buildings = self.client.get_or_create_collection("buildings")
        
        self.last_cluster_update = datetime.now()
        self.cluster_manager = None  # Will be set after initialization
        
    def process_donation(self, amount: float, donor_id: str, metadata: Dict = None) -> Dict:
        """Process a new donation and create a building."""
        building_name = self.generate_building_name()
        coordinates = self.generate_coordinates()
        
        metadata = metadata or {}
        building_metadata = {
            "name": building_name,
            "donation_amount": amount,
            "donor_id": donor_id,
            "creation_date": datetime.now().isoformat(),
            "coordinates": json.dumps(coordinates),
            "region_id": "",  # Will be assigned during clustering
            "donor_name": metadata.get("donor_name", "Anonymous"),
            "building_type": metadata.get("building_type", "Building"),
            "research_question": metadata.get("research_question", ""),
            "website": metadata.get("website", "")
        }
        
        # Generate a simple embedding based on coordinates for now
        embedding = coordinates
        
        self.buildings.add(
            embeddings=[embedding],
            metadatas=[building_metadata],
            ids=[f"building_{donor_id}_{int(datetime.now().timestamp())}"]
        )
        
        # Check if clustering update is needed
        if (datetime.now() - self.last_cluster_update).seconds > 300:  # 5 minutes
            asyncio.create_task(self.cluster_manager.update_clusters())
            self.last_cluster_update = datetime.now()
        
        return building_metadata
    
    def generate_building_name(self) -> str:
        """Generate a unique building name."""
        return f"{fake.last_name()} {fake.random_element(['Hall', 'Center', 'Building', 'Laboratory'])}"
    
    def generate_coordinates(self) -> List[float]:
        """Generate random 2D coordinates for initial placement."""
        return [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
    
    async def process_stripe_event(self, event: stripe.Event) -> Dict:
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
            
            return self.process_donation(amount, donor_id, metadata)
        elif event.type == "payment_intent.succeeded":
            payment_intent = event.data.object
            print(f"Processing succeeded payment: {payment_intent.id}")  # Debug logging
            
            amount = payment_intent.amount / 100
            donor_id = payment_intent.customer or "anonymous"
            
            return self.process_donation(amount, donor_id)
            
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
        buildings = self.dp.buildings.get()
        if not buildings["ids"]:
            return
        
        # Extract coordinates for clustering
        coordinates = np.array([
            json.loads(meta["coordinates"]) for meta in buildings["metadatas"]
        ])
        
        # Determine number of clusters based on number of buildings
        n_clusters = min(self.max_clusters, len(coordinates))
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(coordinates)
        
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
        
        # Assign buildings to regions
        for building_id, meta, label in zip(
            buildings["ids"],
            buildings["metadatas"],
            cluster_labels
        ):
            region_id = str(label)
            region = self.regions[region_id]
            region.buildings.append(building_id)
            region.total_donations += meta["donation_amount"]
            
            # Update building metadata with region
            self.dp.buildings.update(
                ids=[building_id],
                metadatas=[{**meta, "region_id": region_id}]
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
async def stripe_webhook(request: Request):
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
        result = await donation_processor.process_stripe_event(event)
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
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)