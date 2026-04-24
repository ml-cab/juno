#!/bin/bash
# =============================================================
#  launcher.sh — Run any AWS script as juno-bot
#  Usage: ./launcher.sh some_shell.sh [args...]
# =============================================================

# ── JUNO-BOT CREDENTIALS — change these once per environment ─
export AWS_ACCESS_KEY_ID= # NB !!! Please insert value from setup-IAM.sh  !!!
export AWS_SECRET_ACCESS_KEY= # NB !!! Please insert value from setup-IAM.sh  !!!
export AWS_DEFAULT_REGION="eu-north-1" #"us-east-1"
# ─────────────────────────────────────────────────────────────

# Validate argument
if [[ -z "$1" ]]; then
  echo "❌ Usage: ./launcher.sh <script.sh> [args...]"
  exit 1
fi

if [[ ! -f "$1" ]]; then
  echo "❌ File not found: $1"
  exit 1
fi

TARGET_SCRIPT="$1"
shift  # remaining args get passed to the target script

# Confirm identity before running
echo "🤖 Running as: $(aws sts get-caller-identity --query Arn --output text)"
echo "📄 Script    : $TARGET_SCRIPT"
echo "🌍 Region    : $AWS_DEFAULT_REGION"
echo ""

chmod +x "$TARGET_SCRIPT"
bash "$TARGET_SCRIPT" "$@"
