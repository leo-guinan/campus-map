# Priority Features:
- Core Data Pipeline
- Stripe webhook endpoint for donation processing
- Building name generation from donation metadata
- ChromaDB integration for storing embeddings
- Basic 2D projection visualization
Clustering & Regions
- K-means clustering implementation (k=5)
- Region name generation
- Color generation via LLM
- Region metadata storage
Interactive Visualization
- Streamlit map view with plotly/networkx
- Node sizing based on donations
- Region coloring
- Hover tooltips with building/region info
Payment Processing
Stripe integration
Donation form
- Custom fields for name generation
- Webhook handling
Advanced Features
- Real-time updates
Region rebalancing
- Search functionality
- Donation history
- Admin dashboard
Suggested Additional Features:
- Time-based visualization showing campus growth
- Virtual tours between connected buildings
- Donor recognition wall
Region-based statistics
- Path finding between buildings
- Mobile-responsive design

Implementation will start with local ChromaDB for dev and migrate to HTTP client for prod. Each phase builds on the previous, ensuring core functionality before adding complexity.

Sources:
- ChromaDB docs: https://docs.trychroma.com/
- Streamlit: https://docs.streamlit.io/
Stripe API: https://stripe.com/docs/api