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
            
        # Convert embeddings to numpy array and ensure it's 2D
        embeddings_array = np.array(embeddings)
        if embeddings_array.size == 0:
            logger.warning("Embeddings array is empty after conversion")
            return []
        
        # Reshape based on input shape
        if len(embeddings_array.shape) == 1:
            embeddings_array = embeddings_array.reshape(1, -1)
        elif len(embeddings_array.shape) == 3:
            embeddings_array = embeddings_array.reshape(embeddings_array.shape[0], -1)
        
        # Ensure we have valid data for PCA
        if embeddings_array.shape[0] < 1 or embeddings_array.shape[1] < 1:
            logger.warning(f"Invalid embeddings shape: {embeddings_array.shape}")
            return []
            
        # Perform PCA
        n_components = min(2, embeddings_array.shape[1])
        pca = PCA(n_components=n_components)
        coordinates_2d = pca.fit_transform(embeddings_array)
        
        # If we only got 1 component, add a zero column
        if n_components == 1:
            coordinates_2d = np.column_stack([coordinates_2d, np.zeros(len(coordinates_2d))])
        
        return coordinates_2d.tolist()
    except Exception as e:
        logger.error(f"Error generating coordinates: {str(e)}")
        return []

def generate_coordinates(buildings_data: Dict) -> Dict:
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
        
        if "embeddings" not in buildings_data or buildings_data["embeddings"] is None:
            logger.warning("No embeddings found in buildings data")
            # Generate random coordinates if no embeddings
            coordinates = [[np.random.uniform(-1, 1), np.random.uniform(-1, 1)] 
                         for _ in range(len(buildings_data["ids"]))]
        else:
            coordinates = generate_2d_coordinates(buildings_data["embeddings"])
            if not coordinates:
                # Fallback to random coordinates
                coordinates = [[np.random.uniform(-1, 1), np.random.uniform(-1, 1)] 
                             for _ in range(len(buildings_data["ids"]))]
        
        # Update metadata with coordinates
        result_metadata = []
        for i, meta in enumerate(buildings_data.get("metadatas", [])):
            if i < len(coordinates):
                meta_copy = meta.copy() if meta else {}
                meta_copy["coordinates"] = json.dumps(coordinates[i])
                result_metadata.append(meta_copy)
        
        return {
            "ids": buildings_data["ids"],
            "metadata": result_metadata
        }
    except Exception as e:
        logger.error(f"Error updating metadata with coordinates: {str(e)}")
        return {"ids": [], "metadata": []} 