import streamlit as st
from main import donation_processor, cluster_manager

st.set_page_config(
    page_title="Donation Successful!",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
        buildings = donation_processor.buildings.get()
        regions = cluster_manager.get_region_data()
        
        total_donations = sum(b["donation_amount"] for b in buildings["metadatas"])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Donations", f"${total_donations:,.2f}")
        with col2:
            st.metric("Total Buildings", len(buildings["ids"]))
        with col3:
            st.metric("Active Districts", len(regions))
            
    except Exception as e:
        st.warning("Campus statistics temporarily unavailable. Please check back later.")

if __name__ == "__main__":
    main() 