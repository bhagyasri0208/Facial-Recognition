from flask import Flask, render_template, Response, jsonify
import cv2
import os
import time
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# Directory to save images
SAVE_DIR = "captured_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# Initialize camera with optimized settings
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)  # Set FPS to 30

def save_image(frame_data):
    frame, index = frame_data
    file_path = os.path.join(SAVE_DIR, f"image_{index + 1}.jpg")
    
    # Convert to grayscale and resize
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.resize(gray_frame, (640, 480))
    
    # Save with initial quality
    cv2.imwrite(file_path, gray_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    
    return file_path

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_images', methods=['POST'])
def capture_images():
    frames = []
    count = 0
    start_time = time.time()
    target_time = 10.0  # 10 seconds total
    interval = target_time / 100  # Time between each capture
    
    # Capture 100 frames over 10 seconds
    while count < 100:
        frame_start = time.time()
        
        success, frame = camera.read()
        if not success:
            return jsonify({"status": "error", "message": "Camera read failed"}), 500
        
        frames.append((frame.copy(), count))
        count += 1
        
        # Calculate sleep time to maintain consistent interval
        elapsed = time.time() - frame_start
        sleep_time = max(0, interval - elapsed)
        time.sleep(sleep_time)
    
    # Use ThreadPoolExecutor to save images in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        saved_files = list(executor.map(save_image, frames))
    
    elapsed_time = time.time() - start_time
    return jsonify({
        "status": "success", 
        "message": f"Captured {len(saved_files)} images in {elapsed_time:.2f} seconds",
        "save_directory": SAVE_DIR
    })

@app.route('/')
def index():
    return render_template('index.html')

if __name__=='__main__':
    app.run(debug=True)