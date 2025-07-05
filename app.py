# app.py - Face Scanner API with Improved Logic
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

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# ---- Mock Database (ในการใช้งานจริงใช้ database) ----
attendance_records = []  # [{'employee_id': 'emp001', 'event': 'IN', 'timestamp': '2025-07-06 08:30:00', 'image_url': '...'}]

# ---- Immediate Initialization ----
logger.info("🚀 STARTING FACE RECOGNITION INITIALIZATION...")

try:
    # โหลดข้อมูลใบหน้า
    pkl_file = 'encodings_insightface.pkl'
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        if 'embeddings' in data and 'ids' in data:
            embeddings_raw = data['embeddings']
            known_encs = np.asarray(embeddings_raw, dtype=np.float32)
            known_encs = known_encs / (np.linalg.norm(known_encs, axis=1, keepdims=True) + 1e-6)
            known_ids = data['ids']
            logger.info(f"✅ Loaded {len(known_ids)} face encodings: {known_ids}")
            del data, embeddings_raw
            gc.collect()
    
    # โหลด metadata
    if os.path.exists('metadata.csv'):
        with open('metadata.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            metadata = {row['id']: row for row in reader}
        logger.info(f"✅ Loaded metadata for {len(metadata)} employees")
        
except Exception as e:
    logger.error(f"❌ Initialization error: {e}")

# ---- Create Flask App ----
app = Flask(__name__)
CORS(app)

def get_face_app():
    """โหลด InsightFace แบบ lazy loading"""
    global face_app
    
    if face_app is None:
        try:
            from insightface.app import FaceAnalysis
            face_app = FaceAnalysis(
                name='buffalo_l', 
                providers=['CPUExecutionProvider'],
                allowed_modules=['detection', 'recognition']
            )
            face_app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("✅ InsightFace model loaded")
            gc.collect()
        except Exception as e:
            logger.error(f"❌ Failed to load InsightFace: {e}")
            face_app = None
    
    return face_app

def get_today_events(emp_id):
    """ดึงข้อมูลการเข้าออกวันนี้"""
    today = datetime.now().strftime('%Y-%m-%d')
    ins, outs = [], []
    
    for record in attendance_records:
        if record['employee_id'] == emp_id and record['timestamp'].startswith(today):
            dt = datetime.strptime(record['timestamp'], '%Y-%m-%d %H:%M:%S')
            if record['event'] == 'IN':
                ins.append(dt)
            elif record['event'] == 'OUT':
                outs.append(dt)
    
    return ins, outs

def calculate_work_stats(emp_id, ins, outs):
    """คำนวณสถิติการทำงาน"""
    stats = {
        'late_minutes': 0,
        'early_minutes': 0,
        'work_hours': 0,
        'work_minutes': 0,
        'ot_minutes': 0,
        'late_tag': '',
        'early_tag': '',
        'ot_tag': '',
        'total_work': ''
    }
    
    if ins:
        # คำนวณการมาสาย
        first_in = min(ins)
        work_start_dt = datetime.combine(first_in.date(), WORK_START)
        if first_in > work_start_dt:
            stats['late_minutes'] = int((first_in - work_start_dt).total_seconds() // 60)
            stats['late_tag'] = f"(มาสาย {stats['late_minutes']} นาที)"
    
    if outs and ins:
        # คำนวณการออกก่อนเวลา
        last_out = max(outs)
        work_end_dt = datetime.combine(last_out.date(), WORK_END)
        if last_out < work_end_dt:
            stats['early_minutes'] = int((work_end_dt - last_out).total_seconds() // 60)
            stats['early_tag'] = f"(ออกก่อนเวลา {stats['early_minutes']} นาที)"
        
        # คำนวณเวลาทำงาน
        first_in = min(ins)
        actual_start = max(first_in, datetime.combine(first_in.date(), WORK_START))
        actual_end = min(last_out, datetime.combine(last_out.date(), WORK_END))
        
        if actual_end > actual_start:
            work_duration = actual_end - actual_start
            total_minutes = int(work_duration.total_seconds() // 60)
            stats['work_hours'] = total_minutes // 60
            stats['work_minutes'] = total_minutes % 60
            stats['total_work'] = f"{stats['work_hours']}ชม {stats['work_minutes']}นาที"
        
        # คำนวณ OT
        if last_out.time() > WORK_END:
            ot_duration = last_out - datetime.combine(last_out.date(), WORK_END)
            stats['ot_minutes'] = int(ot_duration.total_seconds() // 60)
            stats['ot_tag'] = f"(โอที {stats['ot_minutes']} นาที)"
    
    return stats

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
        return [{'error': 'Face encodings not loaded'}]
    
    try:
        # ตรวจสอบความคมชัด
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if blur_score < BLUR_VAR_TH:
            return [{'error': 'Image too blurry', 'blur_score': blur_score}]
        
        # โหลด face_app
        app = get_face_app()
        if app is None:
            return [{'error': 'InsightFace model not available'}]
        
        # ตรวจจับใบหน้า
        faces = app.get(frame)
        results = []
        now = datetime.now().timestamp()
        
        logger.info(f"🔍 Detected {len(faces)} faces")
        
        for i, face in enumerate(faces):
            try:
                x1, y1, x2, y2 = map(int, face.bbox)
                face_width = x2 - x1
                
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
                    
                    logger.info(f"✅ Recognized: {emp_id} (distance: {min_dist:.3f})")
                    
                    # ตรวจสอบ cooldown
                    if emp_id in last_scan and now - last_scan[emp_id] < COOLDOWN_SEC:
                        remaining = int(COOLDOWN_SEC - (now - last_scan[emp_id]))
                        results.append({
                            'bbox': [x1, y1, x2, y2],
                            'label': f'{emp_id} (รอ {remaining}วิ)',
                            'status': 'cooldown',
                            'employee_id': emp_id,
                            'remaining_seconds': remaining
                        })
                        continue
                    
                    # ตรวจสอบการเข้าออก (ตรรกะเดียวกับโค้ดต้นฉบับ)
                    ins, outs = get_today_events(emp_id)
                    event = 'IN' if not ins else ('OUT' if not outs else None)
                    
                    if event:
                        # บันทึกการเข้าออก
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        last_scan[emp_id] = now
                        
                        # เพิ่มข้อมูลใน mock database
                        attendance_records.append({
                            'employee_id': emp_id,
                            'event': event,
                            'timestamp': timestamp,
                            'image_url': f'mock_image_{emp_id}_{event}_{now}.jpg'
                        })
                        
                        # คำนวณสถิติ (อัปเดตหลังจากบันทึกแล้ว)
                        updated_ins, updated_outs = get_today_events(emp_id)
                        stats = calculate_work_stats(emp_id, updated_ins, updated_outs)
                        
                        results.append({
                            'bbox': [x1, y1, x2, y2],
                            'label': f'{emp_id} {event} {stats["late_tag"]}{stats["early_tag"]}{stats["ot_tag"]}',
                            'status': 'success',
                            'employee_id': emp_id,
                            'event': event,
                            'timestamp': timestamp,
                            'confidence': confidence,
                            'employee_name': metadata.get(emp_id, {}).get('full_name', ''),
                            'work_stats': stats
                        })
                        
                        logger.info(f"🎯 EVENT RECORDED: {emp_id} {event} {stats['late_tag']}{stats['early_tag']}{stats['ot_tag']}")
                    else:
                        # เข้าและออกครบแล้ววันนี้
                        results.append({
                            'bbox': [x1, y1, x2, y2],
                            'label': f'{emp_id} (วันนี้ครบแล้ว)',
                            'status': 'completed',
                            'employee_id': emp_id
                        })
                else:
                    results.append({
                        'bbox': [x1, y1, x2, y2],
                        'label': 'ไม่รู้จัก',
                        'status': 'unknown'
                    })
                    
            except Exception as e:
                logger.error(f"❌ Error processing face {i+1}: {e}")
        
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
        'face_recognition_ready': known_encs is not None and known_ids is not None,
        'encodings_loaded': len(known_ids) if known_ids else 0,
        'employee_ids': known_ids if known_ids else [],
        'metadata_loaded': len(metadata),
        'insightface_loaded': face_app is not None,
        'total_records': len(attendance_records),
        'version': 'improved_logic'
    })

@app.route('/detect_face', methods=['POST'])
def detect_face_api():
    """API สำหรับตรวจจับใบหน้า"""
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
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

@app.route('/attendance_today', methods=['GET'])
def get_attendance_today():
    """ดึงข้อมูลการเข้าออกวันนี้"""
    today = datetime.now().strftime('%Y-%m-%d')
    today_records = []
    
    for emp_id in known_ids:
        ins, outs = get_today_events(emp_id)
        if ins or outs:
            stats = calculate_work_stats(emp_id, ins, outs)
            today_records.append({
                'employee_id': emp_id,
                'employee_name': metadata.get(emp_id, {}).get('full_name', ''),
                'in_time': min(ins).strftime('%H:%M:%S') if ins else None,
                'out_time': max(outs).strftime('%H:%M:%S') if outs else None,
                'status': 'OUT' if outs else 'IN',
                'work_stats': stats
            })
    
    return jsonify({
        'date': today,
        'total_employees': len(known_ids),
        'present_count': len(today_records),
        'records': today_records
    })

@app.route('/attendance_history', methods=['GET'])
def get_attendance_history():
    """ดึงประวัติการเข้าออกทั้งหมด"""
    return jsonify({
        'total_records': len(attendance_records),
        'records': attendance_records
    })

if __name__ == '__main__':
    logger.info("🚀 Starting Face Scanner API...")
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
