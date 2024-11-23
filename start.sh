#!/bin/bash
set -e  # Exit on error

echo "Starting application setup..."

# Print environment for debugging (excluding sensitive values)
echo "Environment variables:"
env | grep -v "KEY\|SECRET\|TOKEN\|PASSWORD"

# Print Python path and version
echo "Python configuration:"
which python
python --version
echo "PYTHONPATH: $PYTHONPATH"

# Print installed packages
echo "Installed packages:"
pip list

# Verify required environment variables
required_vars=("STRIPE_SECRET_KEY" "STRIPE_WEBHOOK_SECRET" "CHROMA_API_KEY")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "Error: Required environment variable $var is not set"
        exit 1
    fi
done

echo "All required environment variables are set"

# Create necessary directories
mkdir -p /app/logs

# Start the server with detailed logging
echo "Starting server..."
exec python -m uvicorn main:app \
    --host 0.0.0.0 \
    --port ${PORT:-8008} \
    --log-level debug \
    2>&1 | tee -a /app/logs/server.log 