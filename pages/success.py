import streamlit as st
from services.api_client import APIClient

st.set_page_config(
    page_title="Donation Successful!",
    layout="wide",
    initial_sidebar_state="collapsed"
)

api_client = APIClient()

def main():
    st.title("🎉 Thank You for Your Donation!")
    
    st.markdown("""
    ### Your Building is Being Created!
    
    Your virtual building is now being added to our campus. In a few moments, 
    it will be assigned to one of our themed districts based on its characteristics.
    
    #### What's Next?
    - Your building will appear on the campus map within minutes
    - You'll be able to see it in its assigned district
    - The size and prominence will reflect your generous donation
    """)
    
    # Add buttons to view the campus or make another donation
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("View Campus Map"):
            st.switch_page("pages/map.py")
    
    with col2:
        if st.button("Make Another Donation"):
            st.switch_page("donation_page.py")
    
    # Show recent campus stats
    st.markdown("### Campus Statistics")
    try:
        buildings = api_client.get_buildings()
        regions = api_client.get_regions()
        
        total_donations = sum(b["donation_amount"] for b in buildings["metadata"])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Donations", f"${total_donations:,.2f}")
        with col2:
            st.metric("Total Buildings", len(buildings["ids"]))
        with col3:
            st.metric("Active Districts", len(regions) if regions else 0)
            
    except Exception as e:
        st.warning("Campus statistics temporarily unavailable. Please check back later.")
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 