FROM python:3.10-slim

WORKDIR /opt/app
COPY . .
RUN pip install --no-cache-dir -r /opt/app/requirements.txt

# Install packages that we need. vim is for helping with debugging
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get upgrade -yq ca-certificates && \
    apt-get install -yq --no-install-recommends \
    prometheus-node-exporter && \
    rm -rf /var/lib/apt/lists/*

EXPOSE 7860
EXPOSE 8000
EXPOSE 9100

# Gradio listens on all interfaces
ENV GRADIO_SERVER_NAME=0.0.0.0

# Start node exporter and the app; JSON (exec) form prevents signal issues
CMD ["sh","-lc","prometheus-node-exporter --web.listen-address=':9100' & exec python /opt/app/app.py"]
