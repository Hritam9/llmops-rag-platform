#!/usr/bin/env bash
# One-time local setup: venv, deps, .env file.
set -e

echo "Creating virtual environment..."
python -m venv venv
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f .env ]; then
  echo "Creating .env from template..."
  cp .env.example .env
  echo "IMPORTANT: edit .env and add your free Groq API key from https://console.groq.com/keys"
fi

echo "Setup complete. Activate with: source venv/bin/activate"
