# app.py - Optimized InsightFace for Cloud Run
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
import logging
import traceback
import gc
import threading

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
init_lock = threading.Lock()
is_initialized = False

def init_face_recognition():
    """เริ่มต้นระบบรู้จำใบหน้า - แบบ lazy loading"""
    global face_app, known_encs, known_ids, metadata, is_initialized
    
    with init_lock:
        if is_initialized:
            return True
            
        try:
            logger.info("🚀 Starting InsightFace initialization...")
            
            # ตรวจสอบไฟล์
            pkl_file = 'encodings_insightface.pkl'
            if not os.path.exists(pkl_file):
                logger.error(f"❌ File not found: {pkl_file}")
                return False
                
            logger.info(f"📄 Loading {pkl_file} ({os.path.getsize(pkl_file)} bytes)")
            
            # โหลดข้อมูลใบหน้าก่อน (ใช้ memory น้อย)
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            if 'embeddings' not in data or 'ids' not in data:
                logger.error("❌ Invalid data format")
                return False
            
            # ประมวลผล embeddings
            embeddings_raw = data['embeddings']
            known_encs = np.asarray(embeddings_raw, dtype=np.float32)
            known_encs = known_encs / (np.linalg.norm(known_encs, axis=1, keepdims=True) + 1e-6)
            known_ids = data['ids']
            
            logger.info(f"✅ Loaded {len(known_ids)} face encodings")
            
            # Clear data from memory
            del data, embeddings_raw
            gc.collect()
            
            # โหลด metadata
            if os.path.exists('metadata.csv'):
                with open('metadata.csv', 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    metadata = {row['id']: row for row in reader}
                logger.info(f"✅ Loaded metadata for {len(metadata)} employees")
            
            # โหลด InsightFace ทีหลัง (เมื่อต้องใช้จริง)
            logger.info("✅ Basic initialization completed")
            is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            logger.error(traceback.format_exc())
            return False

def get_face_app():
    """โหลด InsightFace แบบ lazy loading"""
    global face_app
    
    if face_app is None:
        try:
            logger.info("🔄 Loading InsightFace model...")
            
            # ใช้ providers ที่เหมาะสม
            from insightface.app import FaceAnalysis
            face_app = FaceAnalysis(
                name='buffalo_l', 
                providers=['CPUExecutionProvider'],
                allowed_modules=['detection', 'recognition']
            )
            face_app.prepare(ctx_id=0, det_size=(640, 640))
            
            logger.info("✅ InsightFace model loaded")
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            logger.error(f"❌ Failed to load InsightFace: {e}")
            face_app = None
    
    return face_app

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

def detect_faces(frame):
    """ตรวจจับและรู้จำใบหน้า"""
    global last_scan, known_encs, known_ids
    
    if frame is None:
        return [{'error': 'No frame provided'}]
        
    if known_encs is None or known_ids is None:
        return [{'error': 'No face encodings loaded'}]
    
    try:
        # ตรวจสอบความคมชัด
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if blur_score < BLUR_VAR_TH:
            return [{'error': 'Image too blurry', 'blur_score': blur_score}]
        
        # โหลด face_app แบบ lazy
        app = get_face_app()
        if app is None:
            return [{'error': 'Face recognition model not available'}]
        
        # ตรวจจับใบหน้า
        faces = app.get(frame)
        results = []
        now = datetime.now().timestamp()
        h, w = frame.shape[:2]
        
        logger.info(f"🔍 Detected {len(faces)} faces")
        
        for i, face in enumerate(faces):
            try:
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
                    
                    logger.info(f"✅ Recognized: {emp_id} (dist: {min_dist:.3f})")
                    
                    # ตรวจสอบ cooldown
                    if emp_id in last_scan and now - last_scan[emp_id] < COOLDOWN_SEC:
                        results.append({
                            'bbox': [x1, y1, x2, y2],
                            'label': f'{emp_id} (Wait)',
                            'status': 'cooldown',
                            'employee_id': emp_id
                        })
                        continue
                    
                    # Mock การเข้าออก
                    ins, outs = [], []
                    event = 'IN' if not ins else ('OUT' if not outs else None)
                    
                    if event:
                        last_scan[emp_id] = now
                        
                        results.append({
                            'bbox': [x1, y1, x2, y2],
                            'label': f'{emp_id} {event}',
                            'status': 'success',
                            'employee_id': emp_id,
                            'event': event,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'confidence': confidence,
                            'employee_name': metadata.get(emp_id, {}).get('full_name', '')
                        })
                        
                        logger.info(f"🎯 Event: {emp_id} {event}")
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
                    
            except Exception as e:
                logger.error(f"❌ Error processing face {i+1}: {e}")
        
        # Force cleanup
        gc.collect()
        return results
        
    except Exception as e:
        logger.error(f"❌ Detection error: {e}")
        return [{'error': str(e)}]

# ---- API Routes ----

@app.route('/health', methods=['GET'])
def health_check():
    """ตรวจสอบสถานะ API"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'face_recognition_ready': is_initialized,
        'encodings_loaded': len(known_ids) if known_ids else 0,
        'employee_ids': known_ids if known_ids else [],
        'metadata_loaded': len(metadata),
        'insightface_loaded': face_app is not None,
        'version': 'optimized_insightface'
    })

@app.route('/detect_face', methods=['POST'])
def detect_face_api():
    """API สำหรับตรวจจับใบหน้า"""
    try:
        if not is_initialized:
            return jsonify({'error': 'System not initialized'}), 503
            
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        logger.info("🔄 Processing detection request...")
        
        frame = base64_to_opencv(data['image'])
        if frame is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        faces = detect_faces(frame)
        
        response = {
            'success': True,
            'faces': faces,
            'timestamp': datetime.now().isoformat(),
            'total_faces': len([f for f in faces if 'error' not in f])
        }
        
        successful_scans = [f for f in faces if f.get('status') == 'success']
        if successful_scans:
            response['successful_scans'] = successful_scans
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ API Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/debug', methods=['GET'])
def debug_info():
    """ข้อมูล debug"""
    return jsonify({
        'files_in_directory': os.listdir('.'),
        'is_initialized': is_initialized,
        'face_app_loaded': face_app is not None,
        'known_encs_shape': known_encs.shape if known_encs is not None else None,
        'known_ids': known_ids if known_ids else [],
        'metadata_keys': list(metadata.keys()),
        'library': 'insightface_optimized'
    })

if __name__ == '__main__':
    logger.info("🚀 Starting Optimized InsightFace API...")
    
    # Lazy initialization - จะโหลดตอนต้องใช้
    if not init_face_recognition():
        logger.error("❌ Failed to initialize")
        exit(1)
    
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)