# app.py - Simplified Face Scanner API for Cloud Run
import os
import cv2
import pickle
import numpy as np
import csv
from datetime import datetime, time as dtime
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
from insightface.app import FaceAnalysis
import logging

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ---- Configuration ----
MIN_FACE_W = 150
BLUR_VAR_TH = 50.0
SIM_THR = 0.75
WORK_START = dtime(8, 0, 0)
WORK_END = dtime(17, 0, 0)
COOLDOWN_SEC = 60

# ---- Global Variables ----
face_app = None
known_encs = None
known_ids = None
metadata = {}
last_scan = {}

def init_face_recognition():
    """เริ่มต้นระบบรู้จำใบหน้า"""
    global face_app, known_encs, known_ids, metadata
    
    try:
        # โหลด InsightFace
        face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        face_app.prepare(ctx_id=0)
        logger.info("✅ InsightFace loaded successfully")
        
        # โหลดข้อมูลใบหน้า
        if os.path.exists('encodings_insightface.pkl'):
            with open('encodings_insightface.pkl', 'rb') as f:
                data = pickle.load(f)
            known_encs = np.asarray(data['embeddings'], dtype=np.float32)
            known_encs /= np.linalg.norm(known_encs, axis=1, keepdims=True) + 1e-6
            known_ids = data['ids']
            logger.info(f"✅ Loaded {len(known_ids)} face encodings")
        else:
            logger.error("❌ encodings_insightface.pkl not found")
            return False
            
        # โหลดข้อมูลพนักงาน
        if os.path.exists('metadata.csv'):
            with open('metadata.csv', 'r', encoding='utf-8') as f:
                metadata = {row['id']: row for row in csv.DictReader(f)}
            logger.info(f"✅ Loaded metadata for {len(metadata)} employees")
        else:
            logger.warning("⚠️ metadata.csv not found")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        return False

def base64_to_opencv(base64_string):
    """แปลง base64 เป็น OpenCV image"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        image_bytes = base64.b64decode(base64_string)
        pil_image = Image.open(io.BytesIO(image_bytes))
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return opencv_image
    except Exception as e:
        logger.error(f"Error converting base64: {e}")
        return None

def get_today_events(emp_id):
    """ดึงข้อมูลการเข้าออกวันนี้ (mock data - ปรับใช้ database จริง)"""
    # ในการใช้งานจริง ควรเชื่อมต่อกับ database
    return [], []  # ins, outs

def detect_faces(frame):
    """ตรวจจับและรู้จำใบหน้า - แบบง่าย"""
    global last_scan
    
    if frame is None:
        return []
        
    # ตรวจสอบความคมชัด
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if blur_score < BLUR_VAR_TH:
        return [{'error': 'Image too blurry', 'blur_score': blur_score}]
    
    try:
        faces = face_app.get(frame)
        results = []
        h, w = frame.shape[:2]
        now = datetime.now().timestamp()
        
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            face_width = x2 - x1
            
            # ตรวจสอบขนาดใบหน้า
            if face_width < MIN_FACE_W:
                results.append({
                    'bbox': [x1, y1, x2, y2],
                    'label': 'Too Far',
                    'status': 'too_far'
                })
                continue
            
            # เปรียบเทียบใบหน้า
            emb = face.normed_embedding
            distances = np.linalg.norm(known_encs - emb, axis=1)
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]
            
            if min_dist < SIM_THR:
                emp_id = known_ids[min_idx]
                confidence = (SIM_THR - min_dist) / SIM_THR
                
                # ตรวจสอบ cooldown
                if emp_id in last_scan and now - last_scan[emp_id] < COOLDOWN_SEC:
                    results.append({
                        'bbox': [x1, y1, x2, y2],
                        'label': f'{emp_id} (Wait)',
                        'status': 'cooldown',
                        'employee_id': emp_id
                    })
                    continue
                
                # ตรวจสอบการเข้าออก
                ins, outs = get_today_events(emp_id)
                event = 'IN' if not ins else ('OUT' if not outs else None)
                
                if event:
                    last_scan[emp_id] = now
                    
                    # คำนวณ tags
                    current_time = datetime.now().time()
                    late = current_time > WORK_START if event == 'IN' else False
                    early = current_time < WORK_END if event == 'OUT' else False
                    
                    results.append({
                        'bbox': [x1, y1, x2, y2],
                        'label': f'{emp_id} {event}',
                        'status': 'success',
                        'employee_id': emp_id,
                        'event': event,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'confidence': confidence,
                        'late': late,
                        'early': early,
                        'employee_name': metadata.get(emp_id, {}).get('full_name', '')
                    })
                else:
                    results.append({
                        'bbox': [x1, y1, x2, y2],
                        'label': f'{emp_id} (Complete)',
                        'status': 'already_scanned',
                        'employee_id': emp_id
                    })
            else:
                results.append({
                    'bbox': [x1, y1, x2, y2],
                    'label': 'Unknown',
                    'status': 'unknown'
                })
        
        return results
        
    except Exception as e:
        logger.error(f"Error in face detection: {e}")
        return [{'error': str(e)}]

# ---- API Routes ----

@app.route('/health', methods=['GET'])
def health_check():
    """ตรวจสอบสถานะ API"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'face_recognition_ready': face_app is not None,
        'encodings_loaded': len(known_ids) if known_ids else 0,
        'version': 'simplified'
    })

@app.route('/detect_face', methods=['POST'])
def detect_face_api():
    """API สำหรับตรวจจับใบหน้า - แบบง่าย"""
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # แปลงรูปภาพ
        frame = base64_to_opencv(data['image'])
        if frame is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # ตรวจจับใบหน้า
        faces = detect_faces(frame)
        
        # ส่งผลลัพธ์
        response = {
            'success': True,
            'faces': faces,
            'timestamp': datetime.now().isoformat(),
            'total_faces': len([f for f in faces if 'error' not in f])
        }
        
        # ถ้ามีการสแกนสำเร็จ ส่งข้อมูลเพิ่มเติม
        successful_scans = [f for f in faces if f.get('status') == 'success']
        if successful_scans:
            response['successful_scans'] = successful_scans
            logger.info(f"✅ Successful scans: {[s['employee_id'] for s in successful_scans]}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"API Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/employees', methods=['GET'])
def get_employees():
    """ดึงรายชื่อพนักงาน"""
    return jsonify({
        'employees': list(metadata.values()),
        'total': len(metadata)
    })

if __name__ == '__main__':
    logger.info("🚀 Starting Simplified Face Scanner API...")
    
    if not init_face_recognition():
        logger.error("❌ Failed to initialize")
        exit(1)
    
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)