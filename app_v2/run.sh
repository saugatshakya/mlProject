#!/bin/bash

# ML2025 Multi-Model App - Quick Start Script

echo "============================================================"
echo "ML2025 MULTI-MODEL WEB APPLICATION"
echo "============================================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p uploads models static/css static/js templates

echo ""
echo "============================================================"
echo "SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "Starting server..."
echo ""
echo "Access the app at: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo "============================================================"
echo ""

# Run the app
python app.py
