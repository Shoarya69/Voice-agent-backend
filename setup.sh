#!/bin/bash

# Voice Bot Echo Server Setup Script
# ===================================
# Automated setup for Exotel voice streaming echo server

set -e  # Exit on any error

echo "🚀 Setting up Voice Bot Echo Server for Exotel..."

# Check Python version
python3 --version || {
    echo "❌ Python 3 is required but not found"
    exit 1
}

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create logs directory
echo "📁 Creating logs directory..."
mkdir -p logs

# Make scripts executable
chmod +x start.sh 2>/dev/null || true

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎯 To start the server:"
echo "   ./start.sh"
echo ""
echo "📡 Or manually:"
echo "   source venv/bin/activate"
echo "   python3 simple_server.py"
echo ""
echo "🌐 Server will run at: ws://0.0.0.0:5000"
echo "📝 Logs will be saved to: logs/"
echo "" 