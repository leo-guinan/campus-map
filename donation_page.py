import streamlit as st
import stripe
import os
from typing import Dict
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
STREAMLIT_URL = os.getenv("STREAMLIT_URL")
API_URL = os.getenv("API_URL")
# Configure Stripe
PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY")
if not PUBLISHABLE_KEY:
    st.error("Stripe publishable key not found. Make sure your .env file is properly configured.")

st.set_page_config(
    page_title="Donate to Virtual Campus",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def create_checkout_session(amount: float, email: str, donor_info: Dict) -> Dict:
    """Create Stripe checkout session via API."""
    try:
        response = requests.post(
            f"{API_URL}/create-checkout-session",
            json={
                "amount": amount,
                "currency": "usd",
                "donor_email": email,
                "success_url": f"{STREAMLIT_URL}/success",
                "cancel_url": f"{STREAMLIT_URL}/donation_page",
                "donor_info": donor_info
            },
            timeout=5
        )
        
        if not response.ok:
            st.error(f"Server response: {response.text}")
            return None
            
        return response.json()
        
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the server. Is the FastAPI server running?")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

def main():
    st.title("🏛️ Build Your Legacy")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Leave Your Mark on Our Virtual Campus
        
        Your donation will create a permanent building in our virtual campus. 
        The size and prominence of your building will reflect your contribution, 
        and it will be automatically placed in one of our themed districts.
        
        **Benefits:**
        - Permanent virtual building named in your honor
        - Part of an interactive virtual campus
        - Real-time visualization of your impact
        - Join a community of donors
        """)
        
        # Donation form
        with st.form("donation_form"):
            # Personal Information
            st.subheader("Your Information")
            full_name = st.text_input(
                "Full Name",
                help="Name to be associated with your building"
            )
            
            email = st.text_input(
                "Email",
                help="We'll send your building details to this address"
            )
            
            website = st.text_input(
                "Website (optional)",
                help="A link to your personal or project website"
            )
            
            # Building Preferences
            st.subheader("Building Details")
            building_type = st.selectbox(
                "What kind of building do you want to name?",
                options=[
                    "Research Laboratory",
                    "Academic Hall",
                    "Innovation Center",
                    "Library",
                    "Student Center",
                    "Technology Hub",
                    "Arts Center",
                    "Athletic Facility",
                    "Other"
                ]
            )
            
            # Show custom building type input if "Other" is selected
            custom_building_type = None
            if building_type == "Other":
                custom_building_type = st.text_input(
                    "Custom Building Type",
                    help="Enter your custom building type"
                )
                # Use custom type if provided, otherwise keep "Other"
                building_type = custom_building_type if custom_building_type else building_type
            
            # Research Question
            st.subheader("Research Interest")
            research_question = st.text_area(
                "What's a question you wish someone would help you answer?",
                help="Your question will help guide future research in your building"
            )
            
            # Donation Amount
            st.subheader("Donation Amount")
            amount = st.number_input(
                "Amount ($)",
                min_value=5.0,
                max_value=1000000.0,
                value=100.0,
                step=5.0,
                help="Choose your donation amount. Larger donations create more prominent buildings!"
            )
            
            submitted = st.form_submit_button("Donate Now")
            
            if submitted:
                if not all([full_name, email, research_question]):
                    st.error("Please fill in all required fields")
                elif building_type == "Other" and not custom_building_type:
                    st.error("Please specify your custom building type")
                else:
                    donor_info = {
                        "full_name": full_name,
                        "building_type": building_type,
                        "research_question": research_question,
                        "website": website
                    }
                    session = create_checkout_session(amount, email, donor_info)
                    if session and 'url' in session:
                        # Use Streamlit's native redirect
                        st.link_button("Proceed to Payment", session['url'])
                        # Or use JavaScript redirect as fallback
                        st.markdown(
                            f"""
                            <script>
                                window.top.location.href = '{session['url']}';
                            </script>
                            """,
                            unsafe_allow_html=True
                        )
    
    with col2:
        st.markdown("""
        ### Building Tiers
        
        🏛️ **Landmark** ($1000+)
        - Largest building size
        - Premium placement
        - Special recognition
        
        🏢 **Monument** ($500-999)
        - Large building size
        - Prominent placement
        
        🏫 **Institute** ($100-499)
        - Medium building size
        - Standard placement
        
        🏠 **Foundation** ($5-99)
        - Basic building size
        - Community placement
        """)
        
        # Sample building preview based on amount
        st.markdown("### Building Preview")
        if 'amount' not in locals():
            amount = 100.0
            
        preview_size = min(400, max(200, 200 + (amount / 1000) * 200))
        st.image(
            f"https://placehold.co/{int(preview_size)}x{int(preview_size)}/darkblue/white?text=Building+Preview%0A${amount:,.2f}",
            caption=f"Example building for ${amount:,.2f} donation"
        )

if __name__ == "__main__":
    main() 