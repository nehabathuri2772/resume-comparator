#!/bin/bash
# -------------------------------
# Automated Deployment Script
# Case Study 2 - Resume Comparator
# -------------------------------

echo "===== Starting Automated Deployment ====="

# 1️⃣ System setup
sudo apt update -y
sudo apt install -y python3-venv git

# 2️⃣ Clone or update repo
cd /home/student-admin
if [ ! -d "resume-comparator" ]; then
    echo "Cloning repository..."
    git clone https://github.com/nehabathuri2772/resume-comparator.git
else
    echo "Updating repository..."
    cd resume-comparator
    git pull
    cd ..
fi

# 3️⃣ Virtual environment setup
cd /home/student-admin/resume-comparator
if [ ! -d ".venv" ]; then
    echo "Creating new virtual environment..."
    python3 -m venv .venv
fi

echo "Activating environment and installing dependencies..."
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# 4️⃣ Create or restart systemd service
SERVICE_PATH="/etc/systemd/system/resume-comparator.service"

if [ ! -f "$SERVICE_PATH" ]; then
    echo "Creating systemd service..."
    sudo bash -c "cat > $SERVICE_PATH" <<EOF
[Unit]
Description=Resume Comparator (Gradio) on 8015
After=network.target

[Service]
User=student-admin
WorkingDirectory=/home/student-admin/resume-comparator
ExecStart=/home/student-admin/resume-comparator/.venv/bin/python app.py
Restart=always
RestartSec=5
Environment="PYTHONUNBUFFERED=1"
Environment="HF_TOKEN=${HF_TOKEN}"

[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl daemon-reload
    sudo systemctl enable resume-comparator
fi

echo "Starting service..."
sudo systemctl restart resume-comparator
sudo systemctl status resume-comparator --no-pager

echo "===== Deployment Complete ====="