#!/bin/bash

# 🔄 Convert fine-tuned model to GGUF format for Ollama
# This script converts your custom fine-tuned model to GGUF format

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔄 Converting Model to GGUF Format${NC}"
echo "====================================="

# Configuration
MODEL_DIR="./fine_tuned_model"
OUTPUT_FILE="my_custom_model.gguf"
LLAMA_CPP_DIR="./llama.cpp"

# Check if fine-tuned model exists
if [ ! -d "$MODEL_DIR" ]; then
    echo -e "${RED}❌ Fine-tuned model not found at: $MODEL_DIR${NC}"
    echo "Run fine-tuning first: python create_custom_model.py (option 2)"
    exit 1
fi

echo -e "${GREEN}✅ Found fine-tuned model at: $MODEL_DIR${NC}"

# Check if llama.cpp exists, if not clone it
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo -e "${YELLOW}📥 Cloning llama.cpp...${NC}"
    git clone https://github.com/ggerganov/llama.cpp.git

    echo -e "${YELLOW}🔨 Building llama.cpp...${NC}"
    cd llama.cpp

    # Build with CUDA support if available
    if command -v nvcc &> /dev/null; then
        echo -e "${GREEN}🚀 NVIDIA CUDA detected, building with GPU support${NC}"
        make LLAMA_CUBLAS=1 -j$(nproc)
    else
        echo -e "${YELLOW}⚠️  No CUDA detected, building CPU-only version${NC}"
        make -j$(nproc)
    fi

    cd ..
else
    echo -e "${GREEN}✅ llama.cpp already exists${NC}"
fi

# Check required Python dependencies
echo -e "${BLUE}📦 Checking Python dependencies...${NC}"
python3 -c "import torch, transformers, sentencepiece" 2>/dev/null || {
    echo -e "${YELLOW}⚠️  Installing missing dependencies...${NC}"
    pip install torch transformers sentencepiece protobuf
}

# Convert model to GGUF
echo -e "${BLUE}🔄 Converting to GGUF format...${NC}"
echo "This may take several minutes..."

# Method 1: Direct conversion (recommended)
if [ -f "$LLAMA_CPP_DIR/convert.py" ]; then
    echo -e "${GREEN}Using convert.py${NC}"
    python3 "$LLAMA_CPP_DIR/convert.py" \
        "$MODEL_DIR" \
        --outtype f16 \
        --outfile "$OUTPUT_FILE"
else
    # Method 2: Convert via HF format (fallback)
    echo -e "${YELLOW}Using alternative conversion method${NC}"
    python3 -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

print('Loading model...')
model = AutoModelForCausalLM.from_pretrained('$MODEL_DIR', torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained('$MODEL_DIR')

print('Saving in HF format...')
model.save_pretrained('./temp_hf_model', safe_serialization=True)
tokenizer.save_pretrained('./temp_hf_model')
print('Conversion to HF format complete')
"

    # Then convert HF to GGUF
    if [ -d "./temp_hf_model" ]; then
        python3 "$LLAMA_CPP_DIR/convert.py" \
            "./temp_hf_model" \
            --outtype f16 \
            --outfile "$OUTPUT_FILE"
        rm -rf ./temp_hf_model
    fi
fi

# Verify conversion
if [ -f "$OUTPUT_FILE" ]; then
    FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    echo
    echo -e "${GREEN}🎉 Conversion successful!${NC}"
    echo -e "${BLUE}📄 Output file: $OUTPUT_FILE${NC}"
    echo -e "${BLUE}📊 File size: $FILE_SIZE${NC}"

    # Optional: Quantize to smaller sizes
    echo
    echo -e "${YELLOW}💡 Optional: Create quantized versions?${NC}"
    read -p "Create Q4_K_M quantized version? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}🔄 Creating Q4_K_M quantized version...${NC}"
        "$LLAMA_CPP_DIR/quantize" "$OUTPUT_FILE" "${OUTPUT_FILE%.gguf}_q4_k_m.gguf" Q4_K_M

        if [ -f "${OUTPUT_FILE%.gguf}_q4_k_m.gguf" ]; then
            QUANT_SIZE=$(du -h "${OUTPUT_FILE%.gguf}_q4_k_m.gguf" | cut -f1)
            echo -e "${GREEN}✅ Quantized version created: ${OUTPUT_FILE%.gguf}_q4_k_m.gguf ($QUANT_SIZE)${NC}"
        fi
    fi

    # Test the converted model
    echo
    echo -e "${YELLOW}🧪 Test the converted model?${NC}"
    read -p "Run a quick test? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}🧪 Testing model...${NC}"
        echo "Prompt: 'Hello, how are you?'"
        echo "Response:"
        "$LLAMA_CPP_DIR/main" -m "$OUTPUT_FILE" -p "Hello, how are you?" -n 50 --temp 0.7
    fi

else
    echo -e "${RED}❌ Conversion failed!${NC}"
    echo "Check the error messages above."
    exit 1
fi

# Instructions for next steps
echo
echo -e "${GREEN}🎯 Next Steps:${NC}"
echo "1. Create Ollama Modelfile:"
echo "   python create_custom_model.py  # option 4"
echo
echo "2. Import to Ollama:"
echo "   ollama create my-custom-model -f Modelfile"
echo
echo "3. Test in Ollama:"
echo "   ollama run my-custom-model \"Hello!\""
echo
echo "4. Push to Ollama Library:"
echo "   ollama push my-custom-model"
echo
echo -e "${BLUE}📚 Files created:${NC}"
echo "   • $OUTPUT_FILE (F16 version)"
if [ -f "${OUTPUT_FILE%.gguf}_q4_k_m.gguf" ]; then
    echo "   • ${OUTPUT_FILE%.gguf}_q4_k_m.gguf (Quantized version)"
fi

echo
echo -e "${GREEN}🎉 GGUF conversion completed successfully!${NC}"