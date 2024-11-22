import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from main import app

client = TestClient(app)

@patch('stripe.checkout.Session.create')
def test_create_checkout_session(mock_create_session):
    """Test checkout session creation."""
    # Mock Stripe session response
    mock_session = MagicMock()
    mock_session.id = "cs_test_123"
    mock_session.url = "https://checkout.stripe.com/test"
    mock_create_session.return_value = mock_session
    
    response = client.post(
        "/create-checkout-session",
        json={
            "amount": 100.0,
            "donor_email": "test@example.com",
            "success_url": "https://example.com/success",
            "cancel_url": "https://example.com/cancel"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert "url" in data
    assert data["session_id"] == "cs_test_123"
    assert data["url"] == "https://checkout.stripe.com/test"

def test_create_checkout_session_invalid_amount():
    """Test checkout session with invalid amount."""
    response = client.post(
        "/create-checkout-session",
        json={
            "amount": -100.0,  # Invalid negative amount
            "donor_email": "test@example.com",
            "success_url": "https://example.com/success",
            "cancel_url": "https://example.com/cancel"
        }
    )
    
    assert response.status_code == 422  # Validation error

@patch('stripe.checkout.Session.create')
def test_create_checkout_session_stripe_error(mock_create_session):
    """Test handling of Stripe errors."""
    mock_create_session.side_effect = stripe.error.StripeError("Test error")
    
    response = client.post(
        "/create-checkout-session",
        json={
            "amount": 100.0,
            "donor_email": "test@example.com",
            "success_url": "https://example.com/success",
            "cancel_url": "https://example.com/cancel"
        }
    )
    
    assert response.status_code == 400
    assert "Test error" in response.json()["detail"] 