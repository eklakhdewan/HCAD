#!/bin/bash
# Boots Ollama + SSH; models/data persist on the mounted volume (/data)
mkdir -p /data/ollama
curl -fsSL https://ollama.com/install.sh | sh
nohup ollama serve > /data/ollama.log 2>&1 &
sleep 5
# 8-bit tags (same as src/config.LLM_MODEL and judge registry index 0 at q8_0)
ollama pull llama3.1:8b-instruct-q8_0 || echo "model pull failed — check volume"
ollama pull qwen3:14b-q8_0 || true
service ssh start || /usr/sbin/sshd
tail -f /data/ollama.log
