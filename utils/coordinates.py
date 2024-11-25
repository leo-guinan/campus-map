import numpy as np
from sklearn.decomposition import PCA
from typing import List, Dict, Any
import json
import logging

logger = logging.getLogger(__name__)

def generate_2d_coordinates(embeddings: List[Any]) -> List[List[float]]:
    """Generate 2D coordinates from embeddings using PCA."""
    if embeddings is None or not isinstance(embeddings, (list, np.ndarray)) or len(embeddings) == 0:
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
    if not buildings_data or not isinstance(buildings_data, dict):
        return {"ids": [], "metadata": []}
        
    if "ids" not in buildings_data or not buildings_data["ids"]:
        return {"ids": [], "metadata": []}
    
    if "embeddings" not in buildings_data:
        logger.warning("No embeddings found in buildings data")
        return {
            "ids": buildings_data["ids"],
            "metadata": buildings_data["metadatas"]
        }
    
    # Generate 2D coordinates
    coordinates = generate_2d_coordinates(buildings_data["embeddings"])
    
    # Update metadata with coordinates
    for i, coords in enumerate(coordinates):
        if i < len(buildings_data["metadatas"]):
            buildings_data["metadatas"][i]["coordinates"] = json.dumps(coords)
    
    return {
        "ids": buildings_data["ids"],
        "metadata": buildings_data["metadatas"]
    } 