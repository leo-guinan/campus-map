import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import stripe
from main import app, DonationProcessor

client = TestClient(app)

def test_stripe_webhook_invalid_signature():
    """Test webhook with invalid signature."""
    response = client.post(
        "/webhook/stripe",
        json={"type": "checkout.session.completed"},
        headers={"stripe-signature": "invalid"}
    )
    assert response.status_code == 400

@patch('stripe.Webhook.construct_event')
def test_stripe_webhook_success(mock_construct_event):
    """Test successful webhook processing."""
    # Mock Stripe event
    mock_session = MagicMock()
    mock_session.amount_total = 10000  # $100.00 in cents
    mock_session.customer = "cus_123"
    
    mock_event = MagicMock()
    mock_event.type = "checkout.session.completed"
    mock_event.data.object = mock_session
    
    mock_construct_event.return_value = mock_event
    
    response = client.post(
        "/webhook/stripe",
        json={"type": "checkout.session.completed"},
        headers={"stripe-signature": "valid"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["donation_amount"] == 100.0
    assert "name" in data
    assert "coordinates" in data

@patch('stripe.Webhook.construct_event')
def test_stripe_webhook_ignored_event(mock_construct_event):
    """Test handling of non-donation events."""
    mock_event = MagicMock()
    mock_event.type = "other.event"
    mock_construct_event.return_value = mock_event
    
    response = client.post(
        "/webhook/stripe",
        json={"type": "other.event"},
        headers={"stripe-signature": "valid"}
    )
    
    assert response.status_code == 200
    assert response.json() == {"status": "ignored"} 