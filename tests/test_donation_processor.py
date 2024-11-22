import pytest
from main import DonationProcessor

def test_process_donation():
    processor = DonationProcessor()
    
    # Process a test donation
    result = processor.process_donation(100.0, "test_donor_1")
    
    # Verify the structure of the result
    assert isinstance(result, dict)
    assert all(key in result for key in [
        "name", "donation_amount", "donor_id", "creation_date",
        "coordinates", "region_id"
    ])
    
    # Verify data types
    assert isinstance(result["name"], str)
    assert isinstance(result["donation_amount"], float)
    assert isinstance(result["coordinates"], list)
    assert len(result["coordinates"]) == 2
    assert all(isinstance(x, float) for x in result["coordinates"])

def test_building_name_generation():
    processor = DonationProcessor()
    
    # Generate multiple names and check they're unique
    names = [processor.generate_building_name() for _ in range(10)]
    assert len(set(names)) == len(names)  # All names should be unique
    
    # Verify name format
    name = processor.generate_building_name()
    assert any(suffix in name for suffix in ['Hall', 'Center', 'Building', 'Laboratory'])

def test_coordinate_generation():
    processor = DonationProcessor()
    
    # Generate coordinates and verify bounds
    coords = processor.generate_coordinates()
    assert len(coords) == 2
    assert all(-1 <= x <= 1 for x in coords) 