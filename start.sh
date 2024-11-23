#!/bin/bash
set -e  # Exit on error

# Print environment for debugging
echo "Environment variables:"
env | grep -v "KEY\|SECRET\|TOKEN"  # Don't print sensitive values

# Print Python path
echo "Python path:"
which python
python --version

# Print installed packages
echo "Installed packages:"
pip list

# Start the server
echo "Starting server..."
exec python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8008} 