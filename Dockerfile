FROM python:3.11-slim

# Minimalne zależności
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (tylko 3 pakiety!)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Aplikacja
WORKDIR /app
COPY app/ .

# Port Streamlit
EXPOSE 8501

# Uruchomienie
CMD ["streamlit", "run", "main.py", "--server.address", "0.0.0.0", "--server.port", "8501"]