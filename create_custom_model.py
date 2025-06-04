#!/usr/bin/env python3
"""
🎯 Tworzenie własnego modelu LLM - od fine-tuningu do publikacji
Przykład z Mistral 7B + własne dane
"""

import os
import json
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
import torch

# === 1. PRZYGOTOWANIE DANYCH ===

def create_sample_dataset():
    """Tworzy przykładowy dataset do fine-tuningu"""
    
    # Przykładowe dane - zamień na swoje!
    sample_data = [
        {
            "instruction": "Jak nazywa się stolica Polski?",
            "input": "",
            "output": "Stolica Polski to Warszawa."
        },
        {
            "instruction": "Wyjaśnij czym jest sztuczna inteligencja",
            "input": "",
            "output": "Sztuczna inteligencja (AI) to dziedzina informatyki zajmująca się tworzeniem systemów zdolnych do wykonywania zadań wymagających inteligencji."
        },
        {
            "instruction": "Napisz krótką funkcję w Pythonie",
            "input": "funkcja do obliczania silni",
            "output": "```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n```"
        },
        {
            "instruction": "Jak zoptymalizować kod Python?",
            "input": "",
            "output": "Główne sposoby optymalizacji kodu Python:\n1. Używaj wbudowanych funkcji\n2. Unikaj pętli, preferuj list comprehensions\n3. Używaj numpy dla operacji numerycznych\n4. Profiluj kod przed optymalizacją"
        },
        {
            "instruction": "Co to jest Docker?",
            "input": "",
            "output": "Docker to platforma konteneryzacji umożliwiająca pakowanie aplikacji wraz z zależnościami w lekkie, przenośne kontenery."
        }
    ]
    
    # Zapisz dataset
    os.makedirs("data", exist_ok=True)
    with open("data/training_data.json", "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print("✅ Sample dataset created in data/training_data.json")
    return sample_data

def format_training_data(examples):
    """Formatuje dane dla Mistral Instruct"""
    formatted_texts = []
    
    for example in examples:
        if example.get("input"):
            prompt = f"<s>[INST] {example['instruction']}\n{example['input']} [/INST] {example['output']}</s>"
        else:
            prompt = f"<s>[INST] {example['instruction']} [/INST] {example['output']}</s>"
        formatted_texts.append(prompt)
    
    return {"text": formatted_texts}

# === 2. FINE-TUNING Z LORA ===

def setup_model_and_tokenizer(model_name="mistralai/Mistral-7B-Instruct-v0.1"):
    """Ładuje model i tokenizer"""
    print(f"📥 Loading model: {model_name}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Model z quantization dla RTX 3050
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True,  # 4-bit quantization
        trust_remote_code=True
    )
    
    return model, tokenizer

def setup_lora_config():
    """Konfiguracja LoRA dla efficient fine-tuning"""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # LoRA rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Mistral attention modules
    )

def fine_tune_model():
    """Główna funkcja fine-tuningu"""
    
    # 1. Przygotuj dane
    print("🔄 Preparing training data...")
    sample_data = create_sample_dataset()
    
    # 2. Ładuj model
    model, tokenizer = setup_model_and_tokenizer()
    
    # 3. Setup LoRA
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    
    print(f"📊 Trainable parameters: {model.print_trainable_parameters()}")
    
    # 4. Przygotuj dataset
    dataset = Dataset.from_list(sample_data)
    formatted_dataset = dataset.map(
        lambda x: format_training_data([x]),
        remove_columns=dataset.column_names
    )
    
    # Tokenizacja
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
    
    tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True)
    
    # 5. Training arguments - optymalizowane dla RTX 3050
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Mały batch size dla RTX 3050
        gradient_accumulation_steps=4,
        warmup_steps=10,
        learning_rate=2e-4,
        fp16=True,  # Mixed precision
        logging_steps=1,
        save_strategy="epoch",
        evaluation_strategy="no",
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        remove_unused_columns=False,
    )
    
    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    
    # 7. Train!
    print("🚀 Starting fine-tuning...")
    trainer.train()
    
    # 8. Save model
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    
    print("✅ Fine-tuning completed! Model saved to ./fine_tuned_model")
    
    return model, tokenizer

# === 3. KONWERSJA DO GGUF ===

def convert_to_gguf():
    """Konwertuje model do formatu GGUF dla Ollama"""
    
    print("🔄 Converting to GGUF format...")
    
    # Ten skrypt wymaga llama.cpp
    conversion_script = """
#!/bin/bash

# Pobierz llama.cpp jeśli nie masz
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp
    make -j
    cd ..
fi

# Konwertuj model
python llama.cpp/convert.py ./fine_tuned_model --outtype f16 --outfile my_custom_model.gguf

echo "✅ GGUF conversion completed: my_custom_model.gguf"
"""
    
    with open("convert_to_gguf.sh", "w") as f:
        f.write(conversion_script)
    
    os.chmod("convert_to_gguf.sh", 0o755)
    
    print("📝 Created convert_to_gguf.sh script")
    print("Run: ./convert_to_gguf.sh")

# === 4. TWORZENIE MODELFILE DLA OLLAMA ===

def create_ollama_modelfile():
    """Tworzy Modelfile dla Ollama"""
    
    modelfile_content = '''FROM ./my_custom_model.gguf

# Model metadata
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 2048

# System prompt
SYSTEM """
Jesteś pomocnym asystentem AI stworzonym specjalnie dla polskich użytkowników.
Odpowiadasz w języku polskim, jesteś precyzyjny i pomocny.
Specjalizujesz się w programowaniu, technologii i sztucznej inteligencji.
"""

# Chat template dla Mistral
TEMPLATE """<s>[INST] {{ if .System }}{{ .System }}

{{ end }}{{ .Prompt }} [/INST] {{ .Response }}</s>"""

# Metadata
PARAMETER num_predict 256
PARAMETER stop "<s>"
PARAMETER stop "[INST]"
PARAMETER stop "[/INST]"
'''
    
    with open("Modelfile", "w", encoding="utf-8") as f:
        f.write(modelfile_content)
    print("✅ Utworzono Modelfile dla Ollama")
    print("✅ Created Modelfile for Ollama")

# === 5. PUBLIKACJA MODELU ===

def create_model_in_ollama():
    """Tworzy model w Ollama"""
    
    ollama_commands = """
# 1. Utwórz model w Ollama
ollama create wronai -f Modelfile

# 2. Test modelu
ollama run wronai "Cześć! Kim jesteś?"

# 3. Push do Ollama Library (wymaga konta)
ollama push wronai

# 4. Alternatywnie - export do pliku
ollama save wronai wronai-model.tar
"""
    
    with open("ollama_commands.sh", "w") as f:
        f.write(ollama_commands)
    
    print("✅ Created ollama_commands.sh")

# === 6. PUBLIKACJA NA HUGGING FACE ===

def create_hf_publish_script():
    """Skrypt do publikacji na Hugging Face"""
    
    hf_script = '''#!/usr/bin/env python3
"""
Publikacja modelu na Hugging Face Hub
"""

from huggingface_hub import HfApi, create_repo
import os

def publish_to_hf():
    # Konfiguracja
    model_name = "your-username/my-custom-mistral-7b"
    
    # Login (wymagany HF token)
    # huggingface-cli login
    
    # Utwórz repo
    api = HfApi()
    
    try:
        create_repo(
            repo_id=model_name,
            repo_type="model",
            private=False  # Ustaw True dla prywatnego
        )
        print(f"✅ Repository created: {model_name}")
    except Exception as e:
        print(f"Repository may already exist: {e}")
    
    # Upload plików
    api.upload_folder(
        folder_path="./fine_tuned_model",
        repo_id=model_name,
        commit_message="Initial model upload"
    )
    
    # Upload GGUF (jeśli istnieje)
    if os.path.exists("my_custom_model.gguf"):
        api.upload_file(
            path_or_fileobj="my_custom_model.gguf",
            path_in_repo="my_custom_model.gguf",
            repo_id=model_name,
            commit_message="Add GGUF version"
        )
    
    print(f"🎉 Model published: https://huggingface.co/{model_name}")

if __name__ == "__main__":
    publish_to_hf()
'''
    
    with open("publish_to_hf.py", "w") as f:
        f.write(hf_script)
    
    print("✅ Created publish_to_hf.py")

# === GŁÓWNA FUNKCJA ===

def main():
    """Pełny pipeline tworzenia własnego modelu"""
    
    print("🎯 Custom LLM Creation Pipeline")
    print("===============================")
    
    choice = input("""
Wybierz opcję:
1. Stwórz sample dataset
2. Fine-tune model (wymaga GPU)
3. Konwertuj do GGUF
4. Utwórz Modelfile dla Ollama
5. Przygotuj skrypty publikacji
6. Pełny pipeline (1-5)

Wybór (1-6): """).strip()
    
    if choice == "1":
        create_sample_dataset()
    elif choice == "2":
        fine_tune_model()
    elif choice == "3":
        convert_to_gguf()
    elif choice == "4":
        create_ollama_modelfile()
    elif choice == "5":
        create_hf_publish_script()
    elif choice == "6":
        print("🚀 Running full pipeline...")
        create_sample_dataset()
        
        if input("Continue with fine-tuning? (y/N): ").lower() == 'y':
            fine_tune_model()
            convert_to_gguf()
        
        create_ollama_modelfile()
        create_model_in_ollama()
        create_hf_publish_script()
        
        print("✅ Full pipeline completed!")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()