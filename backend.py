from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os, cv2, numpy as np, torch, base64, json, logging, signal
from io import BytesIO
from pathlib import Path
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Flask app
app = Flask(__name__)
CORS(app)

# Config
IMAGES_FOLDER = os.getenv("IMAGES_FOLDER", "dataset_images")
LABELS_FOLDER = "labels"
SAM2_CHECKPOINT = os.getenv("SAM2_CHECKPOINT", "sam2_hiera_large.pt")
SAM2_CONFIG = os.getenv("SAM2_CONFIG", "sam2_hiera_l")

# Globals
sam2_predictor = None
SAM2_AVAILABLE = False

# Try importing SAM2
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    logging.warning("SAM2 not available, running in mock mode.")

# ------------------ SAM2 Init ------------------
def initialize_sam2():
    """Initialize SAM2 predictor"""
    global sam2_predictor
    if not SAM2_AVAILABLE:
        return False
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Initializing SAM2 on {device}...")
        sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
        sam2_predictor = SAM2ImagePredictor(sam2_model)
        logging.info("SAM2 initialized successfully.")
        return True
    except Exception as e:
        logging.error(f"Error initializing SAM2: {e}")
        sam2_predictor = None
        return False

# Init on startup
initialize_sam2()

# ------------------ Helpers ------------------
def get_images_from_folder():
    if not os.path.exists(IMAGES_FOLDER):
        return []
    supported = {'.jpg','.jpeg','.png','.bmp','.tiff','.tif'}
    images = []
    for f in os.listdir(IMAGES_FOLDER):
        if Path(f).suffix.lower() in supported:
            images.append({
                "id": len(images)+1,
                "name": Path(f).stem,
                "filename": f,
                "url": f"/api/image/{f}"
            })
    return images

def create_mock_mask_base64():
    mask = np.zeros((200,200), dtype=np.uint8)
    cv2.circle(mask, (100,100), 50, 255, -1)
    pil = Image.fromarray(mask, "L")
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ------------------ API Routes ------------------
@app.route("/api/upload_images", methods=["POST"])
def upload_images():
    """Upload single/multiple images"""
    if "files" not in request.files:
        return jsonify({"success":False,"error":"No files"}),400
    files = request.files.getlist("files")
    os.makedirs(IMAGES_FOLDER, exist_ok=True)
    count=0
    for f in files:
        if f.filename:
            f.save(os.path.join(IMAGES_FOLDER,f.filename))
            count+=1
    return jsonify({"success":True,"message":f"Uploaded {count} images"})

@app.route("/api/images")
def list_images():
    imgs = get_images_from_folder()
    return jsonify({"success":True,"images":imgs,"count":len(imgs)})

@app.route("/api/image/<filename>")
def serve_image(filename):
    path = os.path.join(IMAGES_FOLDER, filename)
    if not os.path.exists(path):
        return jsonify({"error":"not found"}),404
    return send_file(path)

@app.route("/api/segment", methods=["POST"])
def segment_image():
    data = request.json
    filename = data.get("filename")
    points = data.get("points",[])
    if not filename: return jsonify({"error":"Filename required"}),400
    path = os.path.join(IMAGES_FOLDER, filename)
    if not os.path.exists(path): return jsonify({"error":"Image not found"}),404

    if not SAM2_AVAILABLE or sam2_predictor is None:
        return jsonify({"success":True,"mock":True,"results":[{
            "id":1,"name":"Mock","confidence":0.85,"mask_base64":create_mock_mask_base64()
        }]})

    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sam2_predictor.set_image(img_rgb)

    results=[]
    for i,p in enumerate(points):
        try:
            coords = np.array([[p["x"],p["y"]]])
            labels = np.array([1])
            masks,scores,_ = sam2_predictor.predict(
                point_coords=coords, point_labels=labels, multimask_output=True
            )
            idx=np.argmax(scores)
            mask=masks[idx]
            conf=float(scores[idx])
            pil=Image.fromarray((mask*255).astype(np.uint8),"L")
            buf=BytesIO(); pil.save(buf,format="PNG")
            mask_b64=base64.b64encode(buf.getvalue()).decode()
            results.append({"id":i+1,"name":f"Seg {i+1}","confidence":conf,"mask_base64":mask_b64,"point":p})
        except Exception as e:
            logging.error(f"Segmentation error {p}: {e}")
    return jsonify({"success":True,"results":results})

@app.route("/api/save_labels", methods=["POST"])
def save_labels():
    data = request.json
    fn = data.get("filename")
    labels = data.get("labels",[])
    if not fn: return jsonify({"error":"Filename required"}),400
    os.makedirs(LABELS_FOLDER, exist_ok=True)
    path=os.path.join(LABELS_FOLDER, Path(fn).stem+"_labels.json")
    with open(path,"w") as f:
        json.dump({"image":fn,"labels":labels}, f, indent=2)
    return jsonify({"success":True,"message":"Labels saved"})

@app.route("/api/status")
def status():
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    return jsonify({
        "status":"running",
        "sam2_available":sam2_predictor is not None,
        "images_count":len(get_images_from_folder()),
        "device":device
    })

@app.route("/api/restart", methods=["POST"])
def restart_backend():
    """Restart the whole backend (kill container -> Docker restarts it)"""
    os.kill(os.getpid(), signal.SIGTERM)
    return jsonify({"success": True, "message": "Restarting backend..."})

# ------------------ Run ------------------
if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
