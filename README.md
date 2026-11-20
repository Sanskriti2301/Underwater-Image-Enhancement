# Underwater AI Project

Advanced underwater image enhancement and object detection system using CNN, GAN, and YOLOv8.

## Features

- **CNN Enhancement**: Noise removal and clarity improvement using deep learning
- **GAN Enhancement**: Color correction and contrast restoration with generative models
- **YOLOv8 Detection**: Real-time underwater object and species detection
- **MongoDB Storage**: Persistent storage of analysis results and detection history
- **Web Interface**: Professional Flask-based UI for image upload and result visualization
- **Alert System**: Automatic detection of unknown/anomalous objects

## Installation

1. Clone the repository
2. Install dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. Set up MongoDB:
   - Ensure MongoDB is running locally or set `MONGODB_URI` environment variable
   \`\`\`bash
   export MONGODB_URI=mongodb://your-connection-string
   \`\`\`

4. Download YOLOv8 weights:
   - Place trained weights in `yolov8/best.pt`
   - Or train your own using Roboflow/Kaggle dataset

## Running the Application

\`\`\`bash
python app.py
\`\`\`

Visit `http://localhost:5000` in your browser.

## Project Structure

- `app.py`: Main Flask application and API routes
- `models/cnn_model.py`: CNN-based image enhancement
- `models/gan_model.py`: GAN-based color correction
- `database/connection.py`: MongoDB connection manager
- `templates/index.html`: Web interface
- `static/`: Uploaded and enhanced images
- `dataset/`: Training data folders (train, val)
- `yolov8/best.pt`: Trained YOLOv8 weights

## API Endpoints

- `POST /api/upload`: Upload and process image
- `GET /api/history`: Get analysis history
- `GET /api/result/<unique_id>`: Get specific result
- `DELETE /api/delete/<unique_id>`: Delete record

## Model Training

### CNN Enhancement
\`\`\`python
from models.cnn_model import CNNEnhancer

enhancer = CNNEnhancer()
# Train on your dataset
# enhancer.model.fit(train_data, val_data, epochs=50)
\`\`\`

### YOLOv8 Detection
\`\`\`bash
pip install ultralytics
yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=100 imgsz=640
\`\`\`

## Technology Stack

- **Frontend**: HTML5, CSS3, Bootstrap 5, JavaScript
- **Backend**: Flask, Python 3.10+
- **AI Models**: TensorFlow/Keras (CNN), PyTorch (GAN), Ultralytics (YOLOv8)
- **Database**: MongoDB
- **Image Processing**: OpenCV

## Environment Variables

\`\`\`
MONGODB_URI=mongodb://localhost:27017
FLASK_ENV=development
\`\`\`

## Notes

- Ensure all models are properly trained before deployment
- YOLOv8 weights should be obtained from Roboflow or trained on your dataset
- CNN and GAN models are initialized with default architectures
- For production, use appropriate model checkpoints

## License

MIT License
