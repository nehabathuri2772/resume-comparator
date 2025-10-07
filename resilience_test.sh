#!/usr/bin/env bash
set -euo pipefail

SERVICE="resume-comparator"
APP_USER="student-admin"
APP_DIR="/home/${APP_USER}/resume-comparator"
KEYS_DIR="/tmp/team_keys"                          # team *.pub files
DEFAULT_KEY_FRAGMENT="rcpaffenroth@paffenroth-23"  # key to remove each run
PORT=8015

say(){ echo -e "\n[resilience] $*"; }

# --- Wipe ---
say "Stopping & removing unit..."
sudo systemctl stop "$SERVICE" || true
sudo systemctl disable "$SERVICE" || true
sudo rm -f "/etc/systemd/system/${SERVICE}.service"
sudo systemctl daemon-reload

say "Deleting app dir & venv..."
rm -rf "$APP_DIR"

# --- Recover ---
say "Re-deploying with deploy.sh..."
sudo ~/deploy.sh "$KEYS_DIR" "$DEFAULT_KEY_FRAGMENT"

# --- Verify ---
say "Service enabled/active?"
systemctl is-enabled "$SERVICE" | grep -q enabled && echo "  enabled ✅"
systemctl is-active "$SERVICE"  | grep -q active  && echo "  active  ✅"

say "ExecStart uses venv?"
grep -q "${APP_DIR}/.venv/bin/python ${APP_DIR}/app.py" "/etc/systemd/system/${SERVICE}.service" && echo "  ExecStart OK ✅"

say "Port & health?"
ss -ltnp | grep -q ":${PORT} " && echo "  port ${PORT} listening ✅"
curl -fsSI "http://127.0.0.1:${PORT}" | grep -q "200 OK" && echo "  HTTP 200 ✅"

say "Default/admin key removed?"
grep -qi "${DEFAULT_KEY_FRAGMENT}" "/home/${APP_USER}/.ssh/authorized_keys" && { echo "  still present ❌"; exit 1; } || echo "  removed ✅"

say "Rebooting to prove persistence..."
sudo reboot
