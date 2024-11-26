import streamlit as st
import plotly.graph_objects as go
import numpy as np
from services.api_client import APIClient
import asyncio
from typing import Dict, List
import json

st.set_page_config(
    page_title="Virtual Campus Map",
    layout="wide",
    initial_sidebar_state="expanded"
)

api_client = APIClient()

# Add session state for view management
if 'view_state' not in st.session_state:
    st.session_state.view_state = {
        'zoomed': False,
        'focused_region': None,
        'original_ranges': {'x': [-1.2, 1.2], 'y': [-1.2, 1.2]}
    }

def calculate_region_bounds(region_center, radius=0.3):
    """Calculate the bounds for a zoomed region view."""
    padding = radius * 0.2  # Add 20% padding
    return {
        'x': [region_center[0] - radius - padding, region_center[0] + radius + padding],
        'y': [region_center[1] - radius - padding, region_center[1] + radius + padding]
    }

def handle_click_event():
    """Handle click events on the plot."""
    clicked_data = st.session_state.get('plotly_click', None)
    if clicked_data and clicked_data.get('event', '') == 'plotly_doubleclick':
        point_data = clicked_data.get('points', [{}])[0]
        curve_name = point_data.get('data', {}).get('name', '')
        
        # Check if we clicked on a region
        regions = api_client.get_regions()
        for region_id, region in regions.items():
            if region['name'] == curve_name:
                if st.session_state.view_state['zoomed'] and st.session_state.view_state['focused_region'] == region_id:
                    # Reset view if clicking on same region
                    st.session_state.view_state['zoomed'] = False
                    st.session_state.view_state['focused_region'] = None
                else:
                    # Zoom to region
                    st.session_state.view_state['zoomed'] = True
                    st.session_state.view_state['focused_region'] = region_id
                st.experimental_rerun()
                break

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_buildings_data():
    """Get all buildings data from the API."""
    return api_client.get_buildings()

def create_campus_map():
    """Create interactive campus map visualization."""
    try:
        buildings = get_buildings_data()
        regions = api_client.get_regions()
        
        fig = go.Figure()
        
        # Add regions as filled shapes if they exist
        if regions:
            for region_id, region in regions.items():
                # Skip regions that aren't focused when zoomed
                if (st.session_state.view_state['zoomed'] and 
                    st.session_state.view_state['focused_region'] != region_id):
                    continue
                    
                center = region["center"]
                color = region["color"]
                
                # Create a circle for each region
                theta = np.linspace(0, 2*np.pi, 100)
                radius = 0.3
                x = center[0] + radius * np.cos(theta)
                y = center[1] + radius * np.sin(theta)
                
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    fill="toself",
                    fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}",
                    line=dict(color=color),
                    name=region["name"],
                    hoverinfo="text",
                    text=f"{region['name']}<br>Buildings: {region['building_count']}<br>Total Donations: ${region['total_donations']:,.2f}"
                ))
        
        # Add buildings as scatter points
        if buildings["metadata"]:
            for meta in buildings["metadata"]:
                # Skip buildings not in focused region when zoomed
                if (st.session_state.view_state['zoomed'] and 
                    meta.get('region_id') != st.session_state.view_state['focused_region']):
                    continue
                    
                coords = json.loads(meta["coordinates"])
                region_id = meta.get("region_id", "")
                region_color = regions.get(region_id, {}).get("color", "#888888") if region_id else "#888888"
                
                hover_text = [
                    f"<b>{meta['name']}</b>",
                    f"Type: {meta.get('building_type', 'Building')}",
                    f"Donor: {meta.get('donor_name', 'Anonymous')}",
                    f"Donation: ${meta['donation_amount']:,.2f}",
                    f"Date: {meta['creation_date'][:10]}"
                ]
                
                if meta.get('research_question'):
                    hover_text.append(f"<br><b>Research Question:</b><br>{meta['research_question']}")
                
                if meta.get('website'):
                    hover_text.append(f"<a href='{meta['website']}' target='_blank'>Visit Website</a>")
                
                fig.add_trace(go.Scatter(
                    x=[coords[0]], 
                    y=[coords[1]],
                    mode="markers",
                    marker=dict(
                        size=10 + np.log(meta["donation_amount"]),
                        color=region_color,
                        line=dict(color="white", width=1)
                    ),
                    name=meta["name"],
                    hovertext="<br>".join(hover_text),
                    hoverinfo="text"
                ))
        
        # Set axis ranges based on view state
        if st.session_state.view_state['zoomed'] and st.session_state.view_state['focused_region']:
            region = regions[st.session_state.view_state['focused_region']]
            bounds = calculate_region_bounds(region['center'])
            xrange = bounds['x']
            yrange = bounds['y']
        else:
            xrange = st.session_state.view_state['original_ranges']['x']
            yrange = st.session_state.view_state['original_ranges']['y']
        
        # Update layout
        fig.update_layout(
            title="Virtual Campus Map",
            showlegend=True,
            hovermode='closest',
            plot_bgcolor='rgba(240, 240, 240, 0.8)',
            width=1000,
            height=800,
            xaxis=dict(
                range=xrange,
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            yaxis=dict(
                range=yrange,
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            dragmode='pan'  # Enable panning
        )
        
        return fig
    except Exception as e:
        st.error(f"Error loading map data: {str(e)}")
        return None

def main():
    st.title("Virtual Campus Visualization")
    
    try:
        # Sidebar stats
        st.sidebar.header("Campus Statistics")
        buildings = get_buildings_data()
        regions = api_client.get_regions()
        
        total_donations = sum(b["donation_amount"] for b in buildings["metadata"])
        st.sidebar.metric("Total Donations", f"${total_donations:,.2f}")
        st.sidebar.metric("Total Buildings", len(buildings["ids"]))
        
        # Region breakdown
        if regions:
            st.sidebar.header("Regions")
            for region_id, region in regions.items():
                st.sidebar.markdown(
                    f"""
                    <div style='padding: 10px; border-left: 4px solid {region["color"]}'>
                        <h4>{region["name"]}</h4>
                        Buildings: {region["building_count"]}<br>
                        Donations: ${region["total_donations"]:,.2f}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        # Main map
        fig = create_campus_map()
        if fig:
            # Add click event handling
            st.plotly_chart(fig, use_container_width=True, key='campus_map')
            handle_click_event()
        
        # Update button and view controls
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Force Region Update"):
                with st.spinner("Updating regions..."):
                    api_client.update_regions()
                    st.success("Regions updated!")
                    st.rerun()
            
            if st.session_state.view_state['zoomed']:
                if st.button("Reset View"):
                    st.session_state.view_state['zoomed'] = False
                    st.session_state.view_state['focused_region'] = None
                    st.rerun()
                    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please make sure the API server is running and accessible.")

if __name__ == "__main__":
    main() 