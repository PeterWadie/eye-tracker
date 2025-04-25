# Dockerfile
FROM python:3.10-slim

# system deps for OpenCV & Dlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libjpeg-dev \
    libtiff-dev \
    libpng-dev && \
    rm -rf /var/lib/apt/lists/*

# create app directory
WORKDIR /app

# copy requirements
COPY requirements.txt .

# install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# copy source
COPY . .

# expose camera permission if needed
# ENTRYPOINT for Mac/Linux might need --device /dev/video0
CMD ["python", "main.py"]
