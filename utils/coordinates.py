import numpy as np
from sklearn.decomposition import PCA
from typing import List, Dict, Any
import json
import logging

logger = logging.getLogger(__name__)

def generate_2d_coordinates(embeddings: List[Any]) -> List[List[float]]:
    """Generate 2D coordinates from embeddings using PCA."""
    try:
        if embeddings is None:
            logger.warning("Embeddings is None")
            return []
            
        if not isinstance(embeddings, (list, np.ndarray)):
            logger.warning(f"Embeddings is not a list or ndarray, got {type(embeddings)}")
            return []
            
        if len(embeddings) == 0:
            logger.warning("Embeddings is empty")
            return []
            
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings)
        
        if embeddings_array.size == 0:
            logger.warning("Embeddings array is empty after conversion")
            return []
        
        # Handle different shapes of embeddings
        if len(embeddings_array.shape) == 1:
            embeddings_array = embeddings_array.reshape(1, -1)
        elif len(embeddings_array.shape) == 3:
            embeddings_array = embeddings_array.reshape(embeddings_array.shape[0], -1)
        
        # Perform PCA
        pca = PCA(n_components=2)
        coordinates_2d = pca.fit_transform(embeddings_array)
        
        return coordinates_2d.tolist()
    except Exception as e:
        logger.error(f"Error generating coordinates: {str(e)}")
        return []

def update_metadata_with_coordinates(buildings_data: Dict) -> Dict:
    """Update building metadata with 2D coordinates."""
    try:
        if not buildings_data:
            logger.warning("Buildings data is None or empty")
            return {"ids": [], "metadata": []}
            
        if not isinstance(buildings_data, dict):
            logger.warning(f"Buildings data is not a dict, got {type(buildings_data)}")
            return {"ids": [], "metadata": []}
            
        if "ids" not in buildings_data:
            logger.warning("No ids in buildings data")
            return {"ids": [], "metadata": []}
            
        if not buildings_data["ids"]:
            logger.warning("Empty ids list in buildings data")
            return {"ids": [], "metadata": []}
        
        if "embeddings" not in buildings_data:
            logger.warning("No embeddings found in buildings data")
            return {
                "ids": buildings_data["ids"],
                "metadata": buildings_data.get("metadatas", [])
            }
        
        if "metadatas" not in buildings_data:
            logger.warning("No metadatas found in buildings data")
            return {
                "ids": buildings_data["ids"],
                "metadata": []
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
    except Exception as e:
        logger.error(f"Error updating metadata with coordinates: {str(e)}")
        return {"ids": [], "metadata": []} 