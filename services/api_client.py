import requests
import os
from typing import Dict, Optional

class APIClient:
    def __init__(self):
        self.base_url = os.getenv("API_URL", "http://localhost:8008")
        
    def get_buildings(self) -> Dict:
        """Get all buildings from the API."""
        response = requests.get(f"{self.base_url}/buildings")
        response.raise_for_status()
        return response.json()
    
    def get_regions(self) -> Dict:
        """Get all regions from the API."""
        response = requests.get(f"{self.base_url}/regions")
        response.raise_for_status()
        return response.json()
    
    def update_regions(self) -> Dict:
        """Force update of regions."""
        response = requests.post(f"{self.base_url}/regions/update")
        response.raise_for_status()
        return response.json()
    
    def create_checkout_session(self, amount: float, email: str, donor_info: Dict) -> Optional[Dict]:
        """Create Stripe checkout session via API."""
        try:
            response = requests.post(
                f"{self.base_url}/create-checkout-session",
                json={
                    "amount": amount,
                    "currency": "usd",
                    "donor_email": email,
                    "success_url": f"{os.getenv('STREAMLIT_URL', 'http://localhost:8501')}/success",
                    "cancel_url": f"{os.getenv('STREAMLIT_URL', 'http://localhost:8501')}/donation_page",
                    "donor_info": donor_info
                },
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return None 