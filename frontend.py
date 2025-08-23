import gradio as gr, requests, os
from PIL import Image, ImageDraw
from io import BytesIO
import base64

BACKEND="http://localhost:5000"

current_filename=None
click_points=[]

# -------- Helpers --------
def get_images():
    try:
        r=requests.get(f"{BACKEND}/api/images")
        if r.status_code==200:
            d=r.json()
            if d["success"]:
                return [(img["name"], img["filename"]) for img in d["images"]]
    except: pass
    return []

def load_image(filename):
    global current_filename, click_points
    current_filename=filename; click_points=[]
    r=requests.get(f"{BACKEND}/api/image/{filename}")
    if r.status_code!=200: return None,"Error loading"
    img=Image.open(BytesIO(r.content))
    return img,f"Loaded {filename}"

def add_point(img, evt:gr.SelectData):
    global click_points
    click_points.append({"x":evt.index[0],"y":evt.index[1]})
    draw=ImageDraw.Draw(img)
    r=6
    draw.ellipse([evt.index[0]-r,evt.index[1]-r,evt.index[0]+r,evt.index[1]+r],fill="red")
    return img,f"Added point {len(click_points)}"

def segment():
    if not current_filename: return [],"No image"
    if not click_points: return [],"No points"
    r=requests.post(f"{BACKEND}/api/segment",json={"filename":current_filename,"points":click_points})
    if r.status_code!=200: return [],"Backend error"
    d=r.json()
    masks=[]
    for res in d["results"]:
        mask=Image.open(BytesIO(base64.b64decode(res["mask_base64"])))
        masks.append(mask)
    return masks, f"Segmentation complete ({len(masks)} segments)"

def reset():
    global click_points
    click_points=[]
    return "Points reset"

def save():
    r=requests.post(f"{BACKEND}/api/save_labels",json={"filename":current_filename,"labels":click_points})
    return r.json().get("message","Error")

def upload_files(files):
    if not files: return "No files"
    data=[("files",(os.path.basename(f.name),open(f.name,"rb"),"image/*")) for f in files]
    try:
        r=requests.post(f"{BACKEND}/api/upload_images",files=data)
    finally:
        for _,f,_ in data: f.close()
    if r.status_code==200: return r.json().get("message")
    return "Upload failed"

def restart_backend():
    try:
        requests.post(f"{BACKEND}/api/restart", timeout=1)
    except:
        pass
    return "🔄 Backend restarting... please wait a few seconds."

# -------- Gradio UI --------
with gr.Blocks(title="SAM2 Labeler") as demo:
    gr.Markdown("# 🎯 SAM2 Dataset Viewer")

    restart_btn=gr.Button("🔄 Restart Backend")
    restart_status=gr.Textbox(label="Backend Status")

    with gr.Tab("Segmentation"):
        img_dropdown=gr.Dropdown(choices=get_images(),label="Select Image")
        img_display=gr.Image(type="pil",interactive=True)
        seg_btn=gr.Button("Apply Segmentation")
        masks_out=gr.Gallery(label="Masks")
        msg_out=gr.Textbox(label="Status")
        reset_btn=gr.Button("Reset Points")
        save_btn=gr.Button("Save Labels")

    with gr.Tab("Upload"):
        upload_input=gr.File(file_types=[".jpg",".png",".jpeg"],file_count="multiple")
        upload_btn=gr.Button("Upload")
        upload_status=gr.Textbox(label="Upload Status")

    # --- Events ---
    img_dropdown.change(load_image,inputs=[img_dropdown],outputs=[img_display,msg_out])
    img_display.select(add_point,inputs=[img_display],outputs=[img_display,msg_out])
    seg_btn.click(segment,outputs=[masks_out,msg_out])
    reset_btn.click(reset,outputs=[msg_out])
    save_btn.click(save,outputs=[msg_out])
    upload_btn.click(upload_files,inputs=[upload_input],outputs=[upload_status]).then(
        lambda:gr.Dropdown.update(choices=get_images()),outputs=[img_dropdown]
    )
    restart_btn.click(restart_backend,outputs=[restart_status])

demo.launch(server_name="0.0.0.0",server_port=7263,share=True)
