#!/bin/bash

# Auto-format Script for RAG Chatbot
# Automatically formats code with Black and isort

echo "🎨 Auto-formatting code..."
echo ""

# Change to project root directory
cd "$(dirname "$0")"

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# 1. Format with Black
echo "1️⃣  Formatting with Black..."
uv run black backend/
echo ""

# 2. Sort imports with isort
echo "2️⃣  Sorting imports with isort..."
uv run isort backend/
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✨ Code formatting complete! ✨${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
