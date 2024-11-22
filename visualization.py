import streamlit as st
import plotly.graph_objects as go
import numpy as np
from main import donation_processor, cluster_manager
import asyncio
from typing import Dict, List

st.set_page_config(
    page_title="Virtual Campus Map",
    layout="wide"
)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_buildings_data():
    """Get all buildings data from ChromaDB."""
    buildings = donation_processor.buildings.get()
    return {
        "ids": buildings["ids"],
        "metadata": buildings["metadatas"]
    }

def create_campus_map():
    """Create interactive campus map visualization."""
    buildings = get_buildings_data()
    regions = cluster_manager.get_region_data()
    
    fig = go.Figure()
    
    # Add regions as filled shapes
    for region_id, region in regions.items():
        center = region["center"]
        color = region["color"]
        
        # Create a circle for each region
        theta = np.linspace(0, 2*np.pi, 100)
        radius = 0.3  # Adjust based on your needs
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
    for meta in buildings["metadata"]:
        coords = meta["coordinates"]
        region_color = regions[meta["region_id"]]["color"] if meta["region_id"] else "#888888"
        
        fig.add_trace(go.Scatter(
            x=[coords[0]], 
            y=[coords[1]],
            mode="markers",
            marker=dict(
                size=10 + np.log(meta["donation_amount"]),  # Size based on donation
                color=region_color,
                line=dict(color="white", width=1)
            ),
            name=meta["name"],
            hovertext=f"{meta['name']}<br>Donation: ${meta['donation_amount']:,.2f}<br>Date: {meta['creation_date'][:10]}",
            hoverinfo="text"
        ))
    
    # Update layout
    fig.update_layout(
        title="Virtual Campus Map",
        showlegend=True,
        hovermode='closest',
        plot_bgcolor='rgba(240, 240, 240, 0.8)',
        width=1000,
        height=800,
        xaxis=dict(
            range=[-1.2, 1.2],
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            range=[-1.2, 1.2],
            showgrid=False,
            zeroline=False,
            showticklabels=False
        )
    )
    
    return fig

def main():
    st.title("Virtual Campus Visualization")
    
    # Sidebar stats
    st.sidebar.header("Campus Statistics")
    buildings = get_buildings_data()
    regions = cluster_manager.get_region_data()
    
    total_donations = sum(b["donation_amount"] for b in buildings["metadata"])
    st.sidebar.metric("Total Donations", f"${total_donations:,.2f}")
    st.sidebar.metric("Total Buildings", len(buildings["ids"]))
    
    # Region breakdown
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
    st.plotly_chart(fig, use_container_width=True)
    
    # Update button
    if st.button("Force Region Update"):
        asyncio.run(cluster_manager.update_clusters())
        st.experimental_rerun()

if __name__ == "__main__":
    main() 