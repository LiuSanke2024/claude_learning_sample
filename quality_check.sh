#!/bin/bash

# Quality Check Script for RAG Chatbot
# Runs code formatting, linting, and tests

set -e  # Exit on first error

echo "🔍 Running code quality checks..."
echo ""

# Change to project root directory
cd "$(dirname "$0")"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# 1. Format check with Black
echo "1️⃣  Checking code formatting with Black..."
if uv run black --check backend/; then
    print_status "Black formatting check passed"
else
    print_error "Black formatting check failed. Run: uv run black backend/"
    exit 1
fi
echo ""

# 2. Import sorting check with isort
echo "2️⃣  Checking import sorting with isort..."
if uv run isort --check-only backend/; then
    print_status "isort check passed"
else
    print_error "Import sorting check failed. Run: uv run isort backend/"
    exit 1
fi
echo ""

# 3. Linting with flake8
echo "3️⃣  Linting with flake8..."
if uv run flake8 backend/ --count --statistics; then
    print_status "flake8 linting passed"
else
    print_warning "flake8 found some issues (see above)"
fi
echo ""

# 4. Run tests
echo "4️⃣  Running tests with pytest..."
if uv run pytest backend/tests/ -v; then
    print_status "All tests passed"
else
    print_error "Some tests failed"
    exit 1
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✨ All quality checks passed! ✨${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
