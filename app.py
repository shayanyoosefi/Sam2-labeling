# Backend server code integrated into Colab
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import cv2
import numpy as np
import torch
import base64
from io import BytesIO
from PIL import Image
import json
from pathlib import Path
import threading
import time

# Import SAM2 components
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
    print("✅ SAM2 imported successfully")
except ImportError as e:
    print(f"❌ SAM2 not available: {e}")
    SAM2_AVAILABLE = False

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
IMAGES_FOLDER = "dataset_images"
SAM2_CHECKPOINT = "sam2_hiera_large.pt"
SAM2_CONFIG = "sam2_hiera_l"

# Global variables
sam2_predictor = None

def initialize_sam2():
    """Initialize SAM2 model"""
    global sam2_predictor
    if not SAM2_AVAILABLE:
        return False

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Initializing SAM2 on {device}...")

        # Build SAM2 model
        sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
        sam2_predictor = SAM2ImagePredictor(sam2_model)
        print("✅ SAM2 model initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Error initializing SAM2: {e}")
        return False

def get_images_from_folder():
    """Get all images from the specified folder"""
    if not os.path.exists(IMAGES_FOLDER):
        return []

    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    images = []

    for filename in os.listdir(IMAGES_FOLDER):
        if Path(filename).suffix.lower() in supported_formats:
            images.append({
                'id': len(images) + 1,
                'name': Path(filename).stem,
                'filename': filename,
                'url': f'/api/image/{filename}'
            })

    return images

@app.route('/api/images', methods=['GET'])
def get_images():
    """Get list of all images in the dataset folder"""
    try:
        images = get_images_from_folder()
        return jsonify({
            'success': True,
            'images': images,
            'count': len(images)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/image/<filename>', methods=['GET'])
def serve_image(filename):
    """Serve individual images"""
    try:
        image_path = os.path.join(IMAGES_FOLDER, filename)
        if not os.path.exists(image_path):
            return jsonify({'error': 'Image not found'}), 404

        return send_file(image_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/segment', methods=['POST'])
def segment_image():
    """Perform SAM2 segmentation on an image"""
    try:
        data = request.json
        filename = data.get('filename')
        click_points = data.get('points', [])

        if not filename:
            return jsonify({'error': 'Filename is required'}), 400

        image_path = os.path.join(IMAGES_FOLDER, filename)
        if not os.path.exists(image_path):
            return jsonify({'error': 'Image not found'}), 404

        if not SAM2_AVAILABLE or sam2_predictor is None:
            return jsonify({
                'success': True,
                'mock': True,
                'message': 'SAM2 not available, returning mock results',
                'results': [{
                    'id': 1,
                    'name': 'Mock Segmentation',
                    'confidence': 0.85,
                    'mask_base64': create_mock_mask_base64()
                }]
            })

        # Load and process image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Set image for SAM2 predictor
        sam2_predictor.set_image(image_rgb)

        results = []

        # Process each click point
        for i, point in enumerate(click_points):
            try:
                # Convert point coordinates
                input_point = np.array([[point['x'], point['y']]])
                input_label = np.array([1])  # Positive point

                # Predict mask
                masks, scores, logits = sam2_predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,
                )

                # Get the best mask
                best_mask_idx = np.argmax(scores)
                mask = masks[best_mask_idx]
                confidence = float(scores[best_mask_idx])

                # Convert mask to base64 image
                mask_image = (mask * 255).astype(np.uint8)
                mask_pil = Image.fromarray(mask_image, mode='L')

                # Save mask as base64
                buffer = BytesIO()
                mask_pil.save(buffer, format='PNG')
                mask_base64 = base64.b64encode(buffer.getvalue()).decode()

                results.append({
                    'id': i + 1,
                    'name': f'Segment {i + 1}',
                    'confidence': confidence,
                    'mask_base64': mask_base64,
                    'point': point
                })

            except Exception as e:
                print(f"Error processing point {point}: {e}")
                continue

        return jsonify({
            'success': True,
            'results': results,
            'total_segments': len(results)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def create_mock_mask_base64():
    """Create a mock segmentation mask"""
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 50, 255, -1)
    mask_pil = Image.fromarray(mask, mode='L')
    buffer = BytesIO()
    mask_pil.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

@app.route('/api/save_labels', methods=['POST'])
def save_labels():
    """Save segmentation labels"""
    try:
        data = request.json
        filename = data.get('filename')
        labels = data.get('labels', [])

        if not filename:
            return jsonify({'error': 'Filename is required'}), 400

        # Create labels directory if it doesn't exist
        labels_dir = "labels"
        os.makedirs(labels_dir, exist_ok=True)

        # Save labels as JSON
        label_filename = Path(filename).stem + '_labels.json'
        label_path = os.path.join(labels_dir, label_filename)

        with open(label_path, 'w') as f:
            json.dump({
                'image': filename,
                'labels': labels,
                'timestamp': str(np.datetime64('now'))
            }, f, indent=2)

        return jsonify({
            'success': True,
            'message': f'Labels saved to {label_path}'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/labels/<filename>', methods=['GET'])
def get_labels(filename):
    """Get existing labels for an image"""
    try:
        labels_dir = "labels"
        label_filename = Path(filename).stem + '_labels.json'
        label_path = os.path.join(labels_dir, label_filename)

        if not os.path.exists(label_path):
            return jsonify({
                'success': True,
                'labels': [],
                'message': 'No existing labels found'
            })

        with open(label_path, 'r') as f:
            data = json.load(f)

        return jsonify({
            'success': True,
            'labels': data.get('labels', []),
            'timestamp': data.get('timestamp', '')
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/labels/<filename>', methods=['DELETE'])
def delete_labels(filename):
    """Delete labels for an image"""
    try:
        labels_dir = "labels"
        label_filename = Path(filename).stem + '_labels.json'
        label_path = os.path.join(labels_dir, label_filename)

        if not os.path.exists(label_path):
            return jsonify({
                'success': False,
                'error': 'Label file not found'
            }), 404

        os.remove(label_path)

        return jsonify({
            'success': True,
            'message': f'Labels deleted for {filename}'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/labels', methods=['GET'])
def list_all_labels():
    """List all existing label files"""
    try:
        labels_dir = "labels"
        if not os.path.exists(labels_dir):
            return jsonify({
                'success': True,
                'labels': []
            })

        label_files = []
        for filename in os.listdir(labels_dir):
            if filename.endswith('_labels.json'):
                image_name = filename.replace('_labels.json', '')
                label_path = os.path.join(labels_dir, filename)

                # Get file modification time
                mod_time = os.path.getmtime(label_path)

                # Get label count
                try:
                    with open(label_path, 'r') as f:
                        data = json.load(f)
                        label_count = len(data.get('labels', []))
                except:
                    label_count = 0

                label_files.append({
                    'image_name': image_name,
                    'filename': filename,
                    'modified': mod_time,
                    'label_count': label_count
                })

        # Sort by modification time (newest first)
        label_files.sort(key=lambda x: x['modified'], reverse=True)

        return jsonify({
            'success': True,
            'labels': label_files
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get server status"""
    return jsonify({
        'status': 'running',
        'sam2_available': SAM2_AVAILABLE and sam2_predictor is not None,
        'images_folder': IMAGES_FOLDER,
        'images_count': len(get_images_from_folder()),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })

# Initialize SAM2
print("🚀 Initializing SAM2 Dataset Viewer Backend...")
print(f"📁 Images folder: {IMAGES_FOLDER}")
print(f"🔧 Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

sam2_initialized = initialize_sam2()
if sam2_initialized:
    print("✅ SAM2 model loaded successfully")
else:
    print("⚠️ SAM2 not available - using mock segmentation")

print("✅ Backend setup complete!")

import gradio as gr
import requests
import numpy as np
from PIL import Image, ImageDraw
import base64
from io import BytesIO
import json
import zipfile
from pathlib import Path

# Global variables for the interface
current_image = None
current_filename = None
click_points = []
segmentation_results = []
backend_running = False

def get_available_images():
    """Get list of available images from backend"""
    try:
        if not backend_running:
            return []
        response = requests.get("http://localhost:5000/api/images")
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                return [(img['name'], img['filename']) for img in data['images']]
    except:
        pass
    return []

def load_image(filename):
    """Load an image from the backend"""
    global current_image, current_filename, click_points, segmentation_results

    if not filename:
        return None, "No image selected", "", [], "", []

    try:
        # Load the image
        response = requests.get(f"http://localhost:5000/api/image/{filename}")
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            current_image = image
            current_filename = filename
            click_points = []
            segmentation_results = []

            # Check for existing labels
            existing_labels_info = check_existing_labels(filename)

            return image, f"Loaded: {filename}", "Click on objects in the image to add segmentation points", [], existing_labels_info, []
        else:
            return None, f"Failed to load image: {filename}", "", [], "", []
    except Exception as e:
        return None, f"Error loading image: {str(e)}", "", [], "", []

def check_existing_labels(filename):
    """Check if image has existing labels"""
    try:
        response = requests.get(f"http://localhost:5000/api/labels/{filename}")
        if response.status_code == 200:
            data = response.json()
            if data['success'] and data['labels']:
                return f"⚠️ Existing labels found: {len(data['labels'])} segments (Created: {data.geet('timestamp', 'Unknown')})"
            else:
                return "✅ No existing labels"
        else:
            return "❓ Could not check existing labels"
    except:
        return "❓ Could not check existing labels"

def add_click_point(image, evt: gr.SelectData):
    """Add a click point to the image"""
    global click_points, current_image

    if current_image is None:
        return image, "No image loaded", []

    # Add click point
    x, y = evt.index[0], evt.index[1]
    click_points.append({"x": x, "y": y})

    # Draw points on image
    img_with_points = current_image.copy()
    draw = ImageDraw.Draw(img_with_points)

    for i, point in enumerate(click_points):
        # Draw circle for each point
        radius = 8
        draw.ellipse([
            point["x"] - radius, point["y"] - radius,
            point["x"] + radius, point["y"] + radius
        ], fill='red', outline='white', width=2)
        # Add number
        draw.text((point["x"] + 10, point["y"] - 10), str(i + 1), fill='white')

    status = f"Added point {len(click_points)} at ({x}, {y}). Total points: {len(click_points)}"    

    return img_with_points, status, create_points_display()

def create_points_display():
    """Create a display of current click points"""
    if not click_points:
        return []

    points_info = []
    for i, point in enumerate(click_points):
        points_info.append([f"Point {i+1}", f"({point['x']}, {point['y']})"])

    return points_info

def perform_segmentation():
    """Perform SAM2 segmentation"""
    global segmentation_results

    if not current_filename:
        return [], "No image selected", []

    if not click_points:
        return [], "No click points added. Click on objects in the image first.", []

    try:
        # Call backend segmentation API
        response = requests.post("http://localhost:5000/api/segment", json={
            "filename": current_filename,
            "points": click_points
        })

        if response.status_code == 200:
            data = response.json()
            if data['success']:
                segmentation_results = data['results']

                # Create mask images for display
                mask_images = []
                for result in segmentation_results:
                    if 'mask_base64' in result:
                        # Decode base64 mask
                        mask_data = base64.b64decode(result['mask_base64'])
                        mask_img = Image.open(BytesIO(mask_data))
                        mask_images.append(mask_img)

                status = f"Segmentation complete! Found {len(segmentation_results)} segments"       
                if data.get('mock'):
                    status += " (Mock results - SAM2 not available)"

                return mask_images, status, create_results_display()
            else:
                return [], f"Segmentation failed: {data.get('error', 'Unknown error')}", []
        else:
            return [], f"Backend error: {response.status_code}", []

    except Exception as e:
        return [], f"Error during segmentation: {str(e)}", []

def create_results_display():
    """Create a display of segmentation results"""
    if not segmentation_results:
        return []

    results_info = []
    for result in segmentation_results:
        confidence = result.get('confidence', 0) * 100
        point = result.get('point', {})
        results_info.append([
            result.get('name', 'Unknown'),
            f"{confidence:.1f}%",
            f"({point.get('x', 0)}, {point.get('y', 0)})"
        ])

    return results_info

def reset_points():
    """Reset all click points"""
    global click_points, segmentation_results

    click_points = []
    segmentation_results = []

    if current_image:
        existing_info = check_existing_labels(current_filename) if current_filename else ""
        return current_image, "Points reset", [], [], [], existing_info
    else:
        return None, "Points reset", [], [], [], ""

def delete_existing_labels():
    """Delete existing labels for current image"""
    if not current_filename:
        return "No image selected"

    try:
        response = requests.delete(f"http://localhost:5000/api/labels/{current_filename}")
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                return f"✅ Deleted labels for {current_filename}"
            else:
                return f"❌ Failed to delete: {data.get('error', 'Unknown error')}"
        else:
            return f"❌ Server error: {response.status_code}"
    except Exception as e:
        return f"❌ Error deleting labels: {str(e)}"

def get_all_labeled_images():
    """Get list of all images with labels"""
    try:
        response = requests.get("http://localhost:5000/api/labels")
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                # Format for display
                labeled_images = []
                for label_info in data['labels']:
                    import datetime
                    mod_time = datetime.datetime.fromtimestamp(label_info['modified']).strftime("%Y-%m-%d %H:%M")
                    labeled_images.append([
                        label_info['image_name'],
                        str(label_info['label_count']),
                        mod_time
                    ])
                return labeled_images
            else:
                return []
    except:
        return []
    return []

def delete_selected_labels(selected_image):
    """Delete labels for selected image from the list"""
    if not selected_image:
        return "No image selected", get_all_labeled_images()

    try:
        # Extract filename from the selected row
        image_name = selected_image
        response = requests.delete(f"http://localhost:5000/api/labels/{image_name}")
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                return f"✅ Deleted labels for {image_name}", get_all_labeled_images()
            else:
                return f"❌ Failed to delete: {data.get('error', 'Unknown error')}", get_all_labeled_images()
        else:
            return f"❌ Server error: {response.status_code}", get_all_labeled_images()
    except Exception as e:
        return f"❌ Error: {str(e)}", get_all_labeled_images()

def save_labels():
    """Save segmentation labels"""
    if not current_filename or not segmentation_results:
        return "No segmentation results to save"

    try:
        response = requests.post("http://localhost:5000/api/save_labels", json={
            "filename": current_filename,
            "labels": segmentation_results
        })

        if response.status_code == 200:
            data = response.json()
            if data['success']:
                return f"✅ Labels saved successfully!"
            else:
                return f"❌ Failed to save: {data.get('error', 'Unknown error')}"
        else:
            return f"❌ Backend error: {response.status_code}"

    except Exception as e:
        return f"❌ Error saving labels: {str(e)}"

def download_results():
    """Create and download results zip file"""
    try:
        # Create a zip file with all results
        zip_buffer = BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add label files
            labels_path = Path('labels')
            if labels_path.exists():
                for label_file in labels_path.glob('*.json'):
                    zip_file.write(label_file, f'labels/{label_file.name}')

            # Add sample info file
            info = {
                "total_images": len(get_available_images()),
                "total_labels": len(list(Path('labels').glob('*.json'))) if Path('labels').exists() else 0,
                "generated_by": "SAM2 Dataset Viewer - Google Colab"
            }
            zip_file.writestr('info.json', json.dumps(info, indent=2))

        zip_buffer.seek(0)
        return zip_buffer.getvalue()

    except Exception as e:
        print(f"Error creating download: {e}")
        return None

def check_backend_status():
    """Check if backend is running"""
    global backend_running
    try:
        response = requests.get("http://localhost:5000/api/status", timeout=2)
        if response.status_code == 200:
            backend_running = True
            data = response.json()
            device = data.get('device', 'unknown')
            sam2_status = "✅ Available" if data.get('sam2_available') else "❌ Not available (mock mode)"
            images_count = data.get('images_count', 0)
            return f"🟢 Backend Running | Device: {device.upper()} | SAM2: {sam2_status} | Images: {images_count}"
        else:
            backend_running = False
            return "🔴 Backend not responding"
    except:
        backend_running = False
        return "🔴 Backend not running - Please start the backend first!"

print("✅ Gradio frontend functions created!")


import threading
import time

# Start Flask backend in background
def run_backend():
    app.run(host='0.0.0.0', port=5000, debug=False)

# Start backend server
print("🚀 Starting Flask backend server...")
backend_thread = threading.Thread(target=run_backend, daemon=True)
backend_thread.start()

# Wait for backend to start
print("⏳ Waiting for backend to initialize...")
time.sleep(3)

# Create Gradio interface
with gr.Blocks(title="🎯 SAM2 Dataset Viewer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎯 SAM2 Dataset Viewer
    Interactive image segmentation with SAM2 in Google Colab

    **Instructions:**
    1. Select an image from the dropdown
    2. Click on objects in the image to add segmentation points
    3. Click "Apply SAM2 Segmentation" to generate masks
    4. Save your labels when finished
    5. Use the Label Management tab to delete wrong labels
    """)

    # Status section
    with gr.Row():
        status_display = gr.Textbox(
            value=check_backend_status(),
            label="🔧 System Status",
            interactive=False
        )
        refresh_status_btn = gr.Button("🔄 Refresh Status", size="sm")

    # Main interface with tabs
    with gr.Tabs():
        # Main segmentation tab
        with gr.TabItem("🎯 Segmentation", id="main_tab"):
            with gr.Row():
                # Left column - Image selection and viewing
                with gr.Column(scale=2):
                    gr.Markdown("### 📁 Image Selection")

                    image_dropdown = gr.Dropdown(
                        choices=get_available_images(),
                        label="Select Image",
                        info="Choose an image from your uploaded dataset"
                    )

                    refresh_images_btn = gr.Button("🔄 Refresh Images", size="sm")

                    # Existing labels info
                    existing_labels_display = gr.Textbox(
                        label="📋 Existing Labels Status",
                        interactive=False,
                        value=""
                    )

                    # Delete existing labels button
                    delete_current_labels_btn = gr.Button("🗑️ Delete Current Image Labels", variant=="secondary", size="sm")

                    gr.Markdown("### 🖼️ Image Viewer")
                    image_display = gr.Image(
                        label="Click on objects to add segmentation points",
                        type="pil",
                        interactive=True
                    )

                    # Control buttons
                    with gr.Row():
                        segment_btn = gr.Button("🎯 Apply SAM2 Segmentation", variant="primary")    
                        reset_btn = gr.Button("🔄 Reset Points", variant="secondary")
                        save_btn = gr.Button("💾 Save Labels", variant="stop")

                # Right column - Points and results
                with gr.Column(scale=1):
                    gr.Markdown("### 📍 Click Points")
                    points_table = gr.Dataframe(
                        headers=["Point", "Coordinates"],
                        datatype=["str", "str"],
                        label="Current Points",
                        interactive=False
                    )

                    gr.Markdown("### 🎯 Segmentation Results")
                    results_table = gr.Dataframe(
                        headers=["Segment", "Confidence", "Point"],
                        datatype=["str", "str", "str"],
                        label="Results",
                        interactive=False
                    )

                    gr.Markdown("### 🖼️ Segmentation Masks")
                    mask_gallery = gr.Gallery(
                        label="Generated Masks",
                        show_label=True,
                        elem_id="gallery",
                        columns=2,
                        rows=2,
                        height="auto"
                    )

        # Label management tab
        with gr.TabItem("🗂️ Label Management", id="labels_tab"):
            gr.Markdown("### 📋 Manage Existing Labels")
            gr.Markdown("View and delete existing segmentation labels. Use this to remove wrong or outdated labels.")

            with gr.Row():
                refresh_labels_btn = gr.Button("🔄 Refresh Label List", variant="primary")

            labeled_images_table = gr.Dataframe(
                headers=["Image Name", "Segments", "Created"],
                datatype=["str", "str", "str"],
                label="Images with Labels",
                interactive=False,
                value=get_all_labeled_images()
            )

            with gr.Row():
                selected_image_input = gr.Textbox(
                    label="Image name to delete labels for",
                    placeholder="Enter image name (without extension)",
                    info="Type the exact image name from the table above"
                )
                delete_selected_btn = gr.Button("🗑️ Delete Selected Labels", variant="stop")        

            label_management_status = gr.Textbox(
                label="📊 Label Management Status",
                interactive=False,
                value="Ready to manage labels"
            )

    # Status and download section
    with gr.Row():
        operation_status = gr.Textbox(
            label="📊 Operation Status",
            interactive=False,
            value="Ready to process images"
        )

        download_btn = gr.Button("📥 Download Results", size="sm")

    # Event handlers
    def refresh_image_list():
        choices = get_available_images()
        return gr.Dropdown.update(choices=choices)

    def refresh_status():
        return check_backend_status()

    # Connect events
    refresh_images_btn.click(
        refresh_image_list,
        outputs=[image_dropdown]
    )

    refresh_status_btn.click(
        refresh_status,
        outputs=[status_display]
    )

    image_dropdown.change(
        load_image,
        inputs=[image_dropdown],
        outputs=[image_display, operation_status, operation_status, points_table, existing_labels_display, results_table]
    )

    image_display.select(
        add_click_point,
        inputs=[image_display],
        outputs=[image_display, operation_status, points_table]
    )

    segment_btn.click(
        perform_segmentation,
        outputs=[mask_gallery, operation_status, results_table]
    )

    reset_btn.click(
        reset_points,
        outputs=[image_display, operation_status, points_table, mask_gallery, results_table, existing_labels_display]
    )

    save_btn.click(
        save_labels,
        outputs=[operation_status]
    )

    # Delete current image labels
    delete_current_labels_btn.click(
        delete_existing_labels,
        outputs=[operation_status]
    ).then(
        lambda: check_existing_labels(current_filename) if current_filename else "",
        outputs=[existing_labels_display]
    )

    # Label management tab events
    refresh_labels_btn.click(
        get_all_labeled_images,
        outputs=[labeled_images_table]
    )

    delete_selected_btn.click(
        delete_selected_labels,
        inputs=[selected_image_input],
        outputs=[label_management_status, labeled_images_table]
    )

    download_btn.click(
        download_results,
        outputs=[gr.File()]
    )

# Launch the interface
print("🌟 Launching Gradio interface...")
demo.launch(
    share=True,  # Creates public link
    debug=False,
    show_error=True,
    server_name="0.0.0.0",
    server_port=7263
)
