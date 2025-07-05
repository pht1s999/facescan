# Dockerfile สำหรับ Face Scanner API (Fixed)
FROM python:3.9-slim

# ป้องกัน interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# ติดตั้ง system dependencies สำหรับ OpenCV และ InsightFace
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
    libglib2.0-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ตั้งค่า working directory
WORKDIR /app

# อัปเกรด pip
RUN pip install --upgrade pip setuptools wheel

# Copy requirements.txt และติดตั้ง Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy โค้ดและข้อมูลทั้งหมด
COPY . .

# ตั้งค่า environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV PYTHONDONTWRITEBYTECODE=1

# สร้างโฟลเดอร์ที่จำเป็น
RUN mkdir -p /app/captures && mkdir -p /app/logs

# ตั้งค่าสิทธิ์ไฟล์
RUN chmod -R 755 /app

# เปิด port 8080
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# รัน application ด้วย gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "300", "--keep-alive", "2", "app:app"]