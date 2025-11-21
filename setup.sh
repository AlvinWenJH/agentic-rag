#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

cp -f .env.example .env

read -p "Enter GEMINI API KEY: " GEMINI_API_KEY
if [ -z "$GEMINI_API_KEY" ]; then
  echo "GEMINI API KEY cannot be empty"
  exit 1
fi

read -s -p "Enter DEFAULT ADMIN PASSWORD: " DEFAULT_ADMIN_PASSWORD
echo
if [ -z "$DEFAULT_ADMIN_PASSWORD" ]; then
  echo "DEFAULT ADMIN PASSWORD cannot be empty"
  exit 1
fi

ESCAPED_GEMINI=$(printf '%s' "$GEMINI_API_KEY" | sed -e 's/[\\\/&]/\\&/g')
ESCAPED_ADMIN=$(printf '%s' "$DEFAULT_ADMIN_PASSWORD" | sed -e 's/[\\\/&]/\\&/g')

sed -i.bak \
  -e "s/^GEMINI_API_KEY=.*/GEMINI_API_KEY=$ESCAPED_GEMINI/" \
  -e "s/^DEFAULT_ADMIN_PASSWORD=.*/DEFAULT_ADMIN_PASSWORD=$ESCAPED_ADMIN/" \
  .env
rm -f .env.bak

if docker compose version >/dev/null 2>&1; then
  docker compose up -d
elif docker-compose --version >/dev/null 2>&1; then
  docker-compose up -d
else
  echo "Docker Compose not found"
  exit 1
fi

echo "Setup complete."