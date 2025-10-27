#!/bin/bash
# GridWorld Q-Learning Quick Start Script

echo "======================================"
echo "GridWorld Q-Learning Quick Start"
echo "======================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # macOS/Linux
    source venv/bin/activate
fi
echo "✅ Virtual environment activated"
echo ""

# Install requirements
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Dependencies installed"
echo ""

# Create output directories
echo "📁 Creating output directories..."
mkdir -p output/models
mkdir -p output/plots
mkdir -p output/gifs
mkdir -p output/reports
touch output/.gitkeep
touch output/models/.gitkeep
touch output/plots/.gitkeep
touch output/gifs/.gitkeep
touch output/reports/.gitkeep
echo "✅ Output directories created"
echo ""

# Launch Streamlit app
echo "======================================"
echo "🚀 Launching Streamlit app..."
echo "======================================"
echo ""
echo "The app will open in your browser at: http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""
streamlit run app.py