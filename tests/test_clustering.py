import pytest
from unittest.mock import MagicMock, patch
from main import ClusterManager, DonationProcessor

@pytest.fixture
def mock_buildings_data():
    return {
        "ids": ["b1", "b2", "b3"],
        "metadatas": [
            {"coordinates": [0.1, 0.1], "donation_amount": 100},
            {"coordinates": [0.2, 0.2], "donation_amount": 200},
            {"coordinates": [-0.1, -0.1], "donation_amount": 300},
        ]
    }

@pytest.fixture
def cluster_manager():
    dp = DonationProcessor()
    return ClusterManager(dp)

def test_color_generation(cluster_manager):
    """Test generation of distinct colors."""
    colors = cluster_manager.generate_colors(5)
    assert len(colors) == 5
    assert len(set(colors)) == 5  # All colors should be unique
    assert all(c.startswith('#') for c in colors)

@patch('main.DonationProcessor')
async def test_cluster_update(mock_dp, cluster_manager, mock_buildings_data):
    """Test cluster updating."""
    mock_dp.buildings.get.return_value = mock_buildings_data
    
    result = await cluster_manager.update_clusters()
    
    assert len(result) == 5  # Number of regions
    for region_data in result.values():
        assert "name" in region_data
        assert "color" in region_data
        assert "center" in region_data
        assert "building_count" in region_data
        assert "total_donations" in region_data

def test_get_region_data(cluster_manager):
    """Test region data retrieval."""
    # Setup some test regions
    cluster_manager.regions = {
        "0": MagicMock(
            name="Test Region",
            color="#ff0000",
            center=[0, 0],
            buildings=["b1", "b2"],
            total_donations=300.0
        )
    }
    
    data = cluster_manager.get_region_data()
    assert "0" in data
    assert data["0"]["name"] == "Test Region"
    assert data["0"]["building_count"] == 2
    assert data["0"]["total_donations"] == 300.0 