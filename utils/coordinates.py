import numpy as np
from sklearn.decomposition import PCA
from typing import List, Dict, Any
import json

def generate_2d_coordinates(embeddings: List[Any]) -> List[List[float]]:
    """Generate 2D coordinates from embeddings using PCA."""
    if not embeddings.all():
        return []
        
    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings)
    
    # Handle different shapes of embeddings
    if len(embeddings_array.shape) == 1:
        embeddings_array = embeddings_array.reshape(1, -1)
    elif len(embeddings_array.shape) == 3:
        embeddings_array = embeddings_array.reshape(embeddings_array.shape[0], -1)
    
    # Perform PCA
    pca = PCA(n_components=2)
    coordinates_2d = pca.fit_transform(embeddings_array)
    
    return coordinates_2d.tolist()

def update_metadata_with_coordinates(buildings_data: Dict) -> Dict:
    """Update building metadata with 2D coordinates."""
    if not buildings_data["ids"]:
        return {"ids": [], "metadata": []}
    
    # Generate 2D coordinates
    coordinates = generate_2d_coordinates(buildings_data["embeddings"])
    
    # Update metadata with coordinates
    for i, coords in enumerate(coordinates):
        buildings_data["metadatas"][i]["coordinates"] = json.dumps(coords)
    
    return {
        "ids": buildings_data["ids"],
        "metadata": buildings_data["metadatas"]
    } 