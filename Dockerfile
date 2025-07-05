# Dockerfile สำหรับ InsightFace (แก้ปัญหกกกา timeout)
FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive

# ติดตั้ง dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgl1-mesa-glx \
    build-essential \
    cmake \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy requirements และติดตั้ง
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy โค้ดและข้อมูล
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV PYTHONDONTWRITEBYTECODE=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# สร้างโฟลเดอร์
RUN mkdir -p /app/captures

EXPOSE 8080

# แก้ไข Gunicorn settings สำหรับ InsightFace
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8080", \
     "--workers", "1", \
     "--worker-class", "sync", \
     "--timeout", "0", \
     "--keep-alive", "65", \
     "--max-requests", "10", \
     "--max-requests-jitter", "5", \
     "--worker-tmp-dir", "/dev/shm", \
     "--log-level", "info", \
     "app:app"]