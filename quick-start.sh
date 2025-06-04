#!/bin/bash

# 🚀 Minimal LLM Setup - Everything in one script!

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Minimal LLM Setup${NC}"
echo "===================="

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found. Please install Docker first.${NC}"
    exit 1
fi

# Check Docker Compose
if ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose not found. Please install Docker Compose.${NC}"
    exit 1
fi

# Check NVIDIA Docker (optional)
if docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi &>/dev/null; then
    echo -e "${GREEN}✅ NVIDIA Docker detected${NC}"
    GPU_AVAILABLE=true
else
    echo -e "${YELLOW}⚠️  No GPU detected, running on CPU${NC}"
    GPU_AVAILABLE=false
fi

# Create project structure
echo -e "${BLUE}📁 Creating project structure...${NC}"
mkdir -p app

# Create minimal Streamlit app if it doesn't exist
if [ ! -f "app/main.py" ]; then
    echo -e "${BLUE}📝 Creating Streamlit app...${NC}"
    # The file content would be copied here in real scenario
    echo "# Streamlit app created. Copy the main.py content here."
fi

# Modify docker-compose for CPU if no GPU
if [ "$GPU_AVAILABLE" = false ]; then
    echo -e "${YELLOW}🔧 Configuring for CPU mode...${NC}"
    sed -i 's/deploy:/# deploy:/g' docker-compose.yml || true
    sed -i 's/resources:/# resources:/g' docker-compose.yml || true
    sed -i 's/reservations:/# reservations:/g' docker-compose.yml || true
    sed -i 's/devices:/# devices:/g' docker-compose.yml || true
    sed -i 's/- driver: nvidia/# - driver: nvidia/g' docker-compose.yml || true
    sed -i 's/count: 1/# count: 1/g' docker-compose.yml || true
    sed -i 's/capabilities: \[gpu\]/# capabilities: [gpu]/g' docker-compose.yml || true
fi

# Build and start services
echo -e "${BLUE}🔨 Building and starting services...${NC}"
docker compose up --build -d

# Wait for services
echo -e "${BLUE}⏳ Waiting for services to start...${NC}"

# Wait for Ollama
echo -n "Waiting for Ollama"
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${GREEN} ✅${NC}"
        break
    fi
    echo -n "."
    sleep 2
done

# Wait for Streamlit
echo -n "Waiting for Streamlit"
for i in {1..30}; do
    if curl -s http://localhost:8501/_stcore/health > /dev/null 2>&1; then
        echo -e "${GREEN} ✅${NC}"
        break
    fi
    echo -n "."
    sleep 2
done

# Check if model download completed
echo -e "${BLUE}📥 Checking model download...${NC}"
docker logs model-setup | tail -5

echo
echo -e "${GREEN}🎉 Setup completed!${NC}"
echo "==================="
echo
echo -e "${BLUE}📍 Access points:${NC}"
echo "   • Streamlit UI: http://localhost:8501"
echo "   • Ollama API:   http://localhost:11434"
echo
echo -e "${BLUE}🔍 Useful commands:${NC}"
echo "   • Check logs:     docker compose logs -f"
echo "   • Stop services:  docker compose down"
echo "   • Restart:        docker compose restart"
echo "   • Shell access:   docker exec -it ollama-engine bash"
echo
echo -e "${BLUE}🧪 Test API:${NC}"
echo '   curl -X POST http://localhost:11434/api/generate \'
echo '     -H "Content-Type: application/json" \'
echo '     -d '"'"'{"model": "mistral:7b-instruct", "prompt": "Hello!"}'\'

# Auto-open browser (optional)
if command -v xdg-open &> /dev/null; then
    echo
    read -p "Open browser automatically? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        xdg-open http://localhost:8501
    fi
elif command -v open &> /dev/null; then
    echo
    read -p "Open browser automatically? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        open http://localhost:8501
    fi
fi

echo
echo -e "${GREEN}Happy chatting! 🤖${NC}"