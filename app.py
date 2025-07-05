# app.py - Complete Face Scanner API for Cloud Runnnn
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
import traceback

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
        logger.info("🚀 Starting Face Recognition Initialization...")
        
        # โหลด InsightFace
        logger.info("🔄 Loading InsightFace model...")
        face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        face_app.prepare(ctx_id=0)
        logger.info("✅ InsightFace loaded successfully")
        
        # ตรวจสอบไฟล์ encodings
        pkl_file = 'encodings_insightface.pkl'
        if not os.path.exists(pkl_file):
            logger.error(f"❌ File not found: {pkl_file}")
            logger.info(f"📁 Current directory files: {os.listdir('.')}")
            return False
            
        logger.info(f"📄 Found {pkl_file}, size: {os.path.getsize(pkl_file)} bytes")
        
        # โหลดข้อมูลใบหน้า
        logger.info("🔄 Loading face encodings...")
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"✅ Pickle file loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load pickle file: {e}")
            return False
        
        # ตรวจสอบ structure ของข้อมูล
        logger.info(f"📋 Data keys: {list(data.keys())}")
        
        if 'embeddings' not in data:
            logger.error("❌ No 'embeddings' key in data")
            return False
            
        if 'ids' not in data:
            logger.error("❌ No 'ids' key in data")
            return False
        
        # ประมวลผล embeddings
        try:
            embeddings_raw = data['embeddings']
            logger.info(f"📊 Raw embeddings type: {type(embeddings_raw)}")
            logger.info(f"📊 Raw embeddings shape: {getattr(embeddings_raw, 'shape', 'No shape attribute')}")
            
            known_encs = np.asarray(embeddings_raw, dtype=np.float32)
            logger.info(f"📊 Converted embeddings shape: {known_encs.shape}")
            
            # Normalize embeddings
            norms = np.linalg.norm(known_encs, axis=1, keepdims=True) + 1e-6
            known_encs = known_encs / norms
            logger.info(f"✅ Embeddings normalized")
            
        except Exception as e:
            logger.error(f"❌ Failed to process embeddings: {e}")
            logger.error(f"❌ Embeddings data: {type(data['embeddings'])}")
            return False
        
        # ประมวลผล IDs
        try:
            known_ids = data['ids']
            logger.info(f"📋 Employee IDs: {known_ids}")
            logger.info(f"✅ Loaded {len(known_ids)} face encodings")
            
        except Exception as e:
            logger.error(f"❌ Failed to process IDs: {e}")
            return False
            
        # โหลดข้อมูลพนักงาน
        metadata_file = 'metadata.csv'
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    metadata = {row['id']: row for row in reader}
                logger.info(f"✅ Loaded metadata for {len(metadata)} employees")
                logger.info(f"📋 Metadata IDs: {list(metadata.keys())}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load metadata: {e}")
                metadata = {}
        else:
            logger.warning(f"⚠️ Metadata file not found: {metadata_file}")
            metadata = {}
            
        logger.info("🎉 Face Recognition initialization completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Critical error in initialization: {e}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
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
    """ตรวจจับและรู้จำใบหน้า"""
    global last_scan, face_app, known_encs, known_ids
    
    if frame is None:
        return [{'error': 'No frame provided'}]
        
    if face_app is None:
        return [{'error': 'Face recognition not initialized'}]
        
    if known_encs is None or known_ids is None:
        return [{'error': 'No face encodings loaded'}]
    
    try:
        # ตรวจสอบความคมชัด
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if blur_score < BLUR_VAR_TH:
            return [{'error': 'Image too blurry', 'blur_score': blur_score}]
        
        # ตรวจจับใบหน้า
        faces = face_app.get(frame)
        results = []
        h, w = frame.shape[:2]
        now = datetime.now().timestamp()
        
        logger.info(f"🔍 Detected {len(faces)} faces in frame")
        
        for i, face in enumerate(faces):
            try:
                x1, y1, x2, y2 = map(int, face.bbox)
                face_width = x2 - x1
                
                logger.info(f"👤 Face {i+1}: bbox=({x1},{y1},{x2},{y2}), width={face_width}")
                
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
                logger.info(f"🧠 Face embedding shape: {emb.shape}")
                
                distances = np.linalg.norm(known_encs - emb, axis=1)
                min_idx = np.argmin(distances)
                min_dist = distances[min_idx]
                
                logger.info(f"📏 Min distance: {min_dist:.4f}, threshold: {SIM_THR}")
                
                if min_dist < SIM_THR:
                    emp_id = known_ids[min_idx]
                    confidence = (SIM_THR - min_dist) / SIM_THR
                    
                    logger.info(f"✅ Recognized: {emp_id} (confidence: {confidence:.3f})")
                    
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
                        
                        logger.info(f"🎯 Event recorded: {emp_id} {event}")
                    else:
                        results.append({
                            'bbox': [x1, y1, x2, y2],
                            'label': f'{emp_id} (Complete)',
                            'status': 'already_scanned',
                            'employee_id': emp_id
                        })
                else:
                    logger.info(f"❓ Unknown face (distance: {min_dist:.4f})")
                    results.append({
                        'bbox': [x1, y1, x2, y2],
                        'label': 'Unknown',
                        'status': 'unknown'
                    })
                    
            except Exception as e:
                logger.error(f"❌ Error processing face {i+1}: {e}")
                results.append({
                    'bbox': [0, 0, 100, 100],
                    'label': 'Error',
                    'status': 'error',
                    'error': str(e)
                })
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error in face detection: {e}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return [{'error': str(e)}]

# ---- API Routes ----

@app.route('/health', methods=['GET'])
def health_check():
    """ตรวจสอบสถานะ API"""
    try:
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'face_recognition_ready': face_app is not None,
            'encodings_loaded': len(known_ids) if known_ids else 0,
            'employee_ids': known_ids if known_ids else [],
            'metadata_loaded': len(metadata),
            'version': 'simplified_with_logging'
        })
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/detect_face', methods=['POST'])
def detect_face_api():
    """API สำหรับตรวจจับใบหน้า"""
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        logger.info("🔄 Processing face detection request...")
        
        # แปลงรูปภาพ
        frame = base64_to_opencv(data['image'])
        if frame is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        logger.info(f"📷 Image converted: {frame.shape}")
        
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
        logger.error(f"❌ API Error: {e}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/employees', methods=['GET'])
def get_employees():
    """ดึงรายชื่อพนักงาน"""
    try:
        return jsonify({
            'employees': list(metadata.values()),
            'total': len(metadata),
            'employee_ids': known_ids if known_ids else []
        })
    except Exception as e:
        logger.error(f"❌ Employees API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/debug', methods=['GET'])
def debug_info():
    """ข้อมูล debug สำหรับตรวจสอบปัญหา"""
    try:
        return jsonify({
            'files_in_directory': os.listdir('.'),
            'face_app_loaded': face_app is not None,
            'known_encs_shape': known_encs.shape if known_encs is not None else None,
            'known_ids': known_ids if known_ids else [],
            'metadata_keys': list(metadata.keys()),
            'environment_variables': dict(os.environ),
            'working_directory': os.getcwd()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Face Scanner API...")
    
    if not init_face_recognition():
        logger.error("❌ Failed to initialize face recognition system")
        exit(1)
    
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
