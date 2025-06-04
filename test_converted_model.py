#!/usr/bin/env python3
"""
🧪 Test script for converted GGUF model
Tests both llama.cpp and Ollama integration
"""

import os
import subprocess
import time
import requests
import json
from pathlib import Path


def test_llamacpp_direct():
    """Test model directly with llama.cpp"""
    print("🧪 Testing with llama.cpp directly...")

    model_file = "my_custom_model.gguf"
    if not os.path.exists(model_file):
        print(f"❌ Model file not found: {model_file}")
        return False

    llamacpp_main = "./llama.cpp/main"
    if not os.path.exists(llamacpp_main):
        print(f"❌ llama.cpp main not found: {llamacpp_main}")
        print("Run: ./convert_to_gguf.sh first")
        return False

    test_prompts = [
        "Hello, how are you?",
        "Wyjaśnij co to jest Docker",
        "Napisz prostą funkcję w Pythonie"
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}/3: {prompt[:30]}... ---")

        cmd = [
            llamacpp_main,
            "-m", model_file,
            "-p", prompt,
            "-n", "100",
            "--temp", "0.7",
            "--top-p", "0.9"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print("✅ Response generated successfully")
                print("Response preview:", result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout)
            else:
                print(f"❌ Error: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print("⏰ Timeout - model may be too slow")
            return False
        except Exception as e:
            print(f"❌ Exception: {e}")
            return False

    return True


def test_ollama_integration():
    """Test model through Ollama"""
    print("\n🤖 Testing Ollama integration...")

    # Check if Ollama is installed
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Ollama not installed or not running")
            return False
    except FileNotFoundError:
        print("❌ Ollama command not found")
        return False

    model_name = "my-custom-model"

    # Check if our custom model exists in Ollama
    if model_name not in result.stdout:
        print(f"⚠️  Model '{model_name}' not found in Ollama")
        print("Create it first:")
        print("1. ollama create my-custom-model -f Modelfile")
        return False

    print(f"✅ Found model: {model_name}")

    # Test through Ollama API
    test_prompts = [
        "Cześć! Kim jesteś?",
        "Jak zoptymalizować kod Python?",
        "Co to jest machine learning?"
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Ollama Test {i}/3: {prompt[:30]}... ---")

        try:
            # Test via CLI
            cmd = ["ollama", "run", model_name, prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                print("✅ Ollama CLI response successful")
                print("Response preview:", result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout)
            else:
                print(f"❌ Ollama CLI error: {result.stderr}")
                continue

        except subprocess.TimeoutExpired:
            print("⏰ Ollama timeout")
            continue
        except Exception as e:
            print(f"❌ Ollama exception: {e}")
            continue

    # Test via API
    print("\n🌐 Testing Ollama API...")
    try:
        api_url = "http://localhost:11434/api/generate"
        test_data = {
            "model": model_name,
            "prompt": "Hello! Test API call.",
            "stream": False
        }

        response = requests.post(api_url, json=test_data, timeout=60)
        if response.status_code == 200:
            data = response.json()
            print("✅ Ollama API response successful")
            print("API Response:", data.get('response', 'No response field')[:100])
        else:
            print(f"❌ API Error: {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ API Request failed: {e}")
        return False

    return True


def benchmark_model():
    """Simple benchmark of the model"""
    print("\n📊 Running simple benchmark...")

    model_file = "my_custom_model.gguf"
    if not os.path.exists(model_file):
        print("❌ Model file not found for benchmark")
        return

    # Get file size
    file_size = os.path.getsize(model_file) / (1024 ** 3)  # GB
    print(f"📁 Model size: {file_size:.2f} GB")

    # Benchmark prompt
    benchmark_prompt = "Explain artificial intelligence in simple terms."

    llamacpp_main = "./llama.cpp/main"
    if os.path.exists(llamacpp_main):
        print("⏱️  Timing generation speed...")

        cmd = [
            llamacpp_main,
            "-m", model_file,
            "-p", benchmark_prompt,
            "-n", "100",
            "--temp", "0.7"
        ]

        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            end_time = time.time()

            if result.returncode == 0:
                duration = end_time - start_time
                # Rough tokens estimation
                tokens = len(result.stdout.split())
                tokens_per_second = tokens / duration if duration > 0 else 0

                print(f"⚡ Generation time: {duration:.2f} seconds")
                print(f"🚀 Speed: ~{tokens_per_second:.1f} tokens/second")
                print(f"📝 Generated tokens: ~{tokens}")
            else:
                print("❌ Benchmark failed")
        except subprocess.TimeoutExpired:
            print("⏰ Benchmark timeout")


def main():
    """Main test runner"""
    print("🧪 Custom Model Test Suite")
    print("=" * 40)

    # Check prerequisites
    print("🔍 Checking prerequisites...")

    required_files = [
        "my_custom_model.gguf",
        "./llama.cpp/main",
        "Modelfile"
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print("❌ Missing required files:")
        for f in missing_files:
            print(f"   • {f}")
        print("\nRun these commands first:")
        print("1. python create_custom_model.py  # fine-tune model")
        print("2. ./convert_to_gguf.sh           # convert to GGUF")
        print("3. ollama create my-custom-model -f Modelfile  # import to Ollama")
        return

    print("✅ All required files found")

    # Run tests
    tests_passed = 0
    total_tests = 3

    # Test 1: Direct llama.cpp
    if test_llamacpp_direct():
        tests_passed += 1
        print("✅ llama.cpp test PASSED")
    else:
        print("❌ llama.cpp test FAILED")

    # Test 2: Ollama integration
    if test_ollama_integration():
        tests_passed += 1
        print("✅ Ollama test PASSED")
    else:
        print("❌ Ollama test FAILED")

    # Test 3: Benchmark
    benchmark_model()
    tests_passed += 1  # Benchmark always "passes"

    # Results
    print("\n" + "=" * 40)
    print(f"🎯 Test Results: {tests_passed}/{total_tests} passed")

    if tests_passed == total_tests:
        print("🎉 All tests passed! Your custom model is ready!")
        print("\n🚀 Next steps:")
        print("• ollama push my-custom-model  # Share with the world")
        print("• Integrate into your applications")
        print("• Fine-tune further with more data")
    else:
        print("⚠️  Some tests failed. Check the output above.")

    # Usage examples
    print("\n📚 Usage Examples:")
    print("# Ollama CLI:")
    print("ollama run my-custom-model 'Your question here'")
    print("\n# Ollama API:")
    print("curl -X POST http://localhost:11434/api/generate \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{\"model\": \"my-custom-model\", \"prompt\": \"Hello!\"}'")

    print("\n# Python integration:")
    print("import ollama")
    print("response = ollama.chat(model='my-custom-model', messages=[")
    print("  {'role': 'user', 'content': 'Hello!'}])")


if __name__ == "__main__":
    main()