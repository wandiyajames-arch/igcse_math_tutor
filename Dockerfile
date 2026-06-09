# 1. Use an official lightweight Python image
FROM python:3.12-slim

# 2. Install background tools (curl) to download Ollama
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# 3. Install the Ollama Engine
RUN curl -fsSL https://ollama.com/install.sh | sh

# 4. Set up a secure user environment (Hugging Face requires this)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# 5. Copy your requirements and install the Python packages
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy your entire project folder (including your code and the Chroma database)
COPY --chown=user . .

# 7. Open the Gradio port
EXPOSE 7860

# 8. The Boot Sequence:
# Start Ollama in the background, wait 10 seconds for it to wake up, 
# download the Llama 3.2 model, and FINALLY start your web interface!
CMD ollama serve & sleep 10 && ollama pull llama3.2 && python3 -m src.agent.ui