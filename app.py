from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from datetime import datetime
from models.cnn_model import CNNEnhancer
from database.connection import DatabaseManager
import cv2
import numpy as np
from ultralytics import YOLO
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ENHANCED_FOLDER = 'static/enhanced'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ENHANCED_FOLDER, exist_ok=True)

# Initialize models
cnn_enhancer = CNNEnhancer()

try:
    from models.gan_model import GANEnhancer
    gan_enhancer = GANEnhancer()
    GAN_AVAILABLE = True
except Exception as e:
    print(f"GAN unavailable: {e}")
    gan_enhancer = None
    GAN_AVAILABLE = False
    
# Load YOLO model (fallback to ultralytics default if custom weights missing)
yolo_weights = os.getenv('YOLO_WEIGHTS', 'yolov8/best.pt')
if not os.path.isfile(yolo_weights):
    print(f"WARNING: YOLO weights not found at {yolo_weights}. Falling back to 'yolov8n.pt' (will download).")
    yolo_weights = 'yolov8n.pt'
yolo_model = YOLO(yolo_weights)

db_manager = DatabaseManager()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_image():
    """Handle image upload and processing pipeline"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Save uploaded image
        unique_id = str(uuid.uuid4())
        original_filename = f"{unique_id}_original.jpg"
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        file.save(original_path)
        
        # Read image
        image = cv2.imread(original_path)
        if image is None:
            return jsonify({'error': 'Failed to read image'}), 400
        
        # Enhance with CNN
        cnn_enhanced = cnn_enhancer.enhance(image)

        if GAN_AVAILABLE:
            gan_enhanced = gan_enhancer.enhance(cnn_enhanced)
        else:
            gan_enhanced = cnn_enhanced

        # Save enhanced image
        enhanced_filename = f"{unique_id}_enhanced.jpg"
        enhanced_path = os.path.join(app.config['ENHANCED_FOLDER'], enhanced_filename)
        cv2.imwrite(enhanced_path, gan_enhanced)
        
        results = yolo_model(gan_enhanced)
        detections = []
        unknown_objects = []
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = result.names[class_id]
                
                # Check if object is unknown
                if confidence < 0.5:
                    unknown_objects.append({
                        'confidence': confidence,
                        'box': box.xyxy[0].tolist()
                    })
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'box': box.xyxy[0].tolist()
                })
        
        # Generate alert if unknown objects detected
        alert = None
        if unknown_objects:
            alert = f"Warning: {len(unknown_objects)} unknown objects detected!"
        
        # Store in database
        db_record = {
            'unique_id': unique_id,
            'original_filename': original_filename,
            'enhanced_filename': enhanced_filename,
            'detections': detections,
            'detection_count': len(detections),
            'unknown_objects_count': len(unknown_objects),
            'alert': alert,
            'timestamp': datetime.now(),
            'file_uploaded': file.filename
        }
        
        db_manager.insert_record(db_record)
        
        return jsonify({
            'success': True,
            'unique_id': unique_id,
            'original_image': f'/static/uploads/{original_filename}',
            'enhanced_image': f'/static/enhanced/{enhanced_filename}',
            'detections': detections,
            'alert': alert,
            'detection_count': len(detections),
            'unknown_objects_count': len(unknown_objects)
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get analysis history"""
    try:
        history = db_manager.get_all_records()
        return jsonify(history), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/result/<unique_id>', methods=['GET'])
def get_result(unique_id):
    """Get specific analysis result"""
    try:
        record = db_manager.get_record(unique_id)
        if not record:
            return jsonify({'error': 'Record not found'}), 404
        return jsonify(record), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete/<unique_id>', methods=['DELETE'])
def delete_record(unique_id):
    """Delete analysis record"""
    try:
        db_manager.delete_record(unique_id)
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large'}), 413

if __name__ == '__main__':
    app.run(debug=True, port=5000)
