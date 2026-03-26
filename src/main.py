import asyncio
import sys

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import cv2
import torch
import gradio as gr
import numpy as np
from pathlib import Path
from PIL import Image

from ultralytics import YOLO    
import onnxruntime as ort
import config.par as PAR_CFG
import config.var as VAR_CFG

## Define Global Variables
MODEL_ROOT  : Path = Path(__file__).parent.parent / "models"
DETECTOR                            = None
VAR_MODEL   : ort.InferenceSession  = None
PAR_MODEL   : ort.InferenceSession  = None

def load_detector():
    global DETECTOR
    if DETECTOR is None:
        DETECTOR = YOLO(MODEL_ROOT / "detect" / "yolo26l.pt")

def load_var():
    global VAR_MODEL
    if VAR_MODEL is None:
        path = MODEL_ROOT / "var" / VAR_CFG.DEMO_MODELS[0]
        if not path.exists():
            print(f"Model not found at {path}!!")
        VAR_MODEL = ort.InferenceSession(MODEL_ROOT / "var" / VAR_CFG.DEMO_MODELS[0])

def load_par():
    global PAR_MODEL
    if PAR_MODEL is None:
        path = MODEL_ROOT / "par" / PAR_CFG.DEMO_MODELS[0]
        if not path.exists():
            print(f"Model not found at {path}!!")
        PAR_MODEL = ort.InferenceSession(MODEL_ROOT / "par" / PAR_CFG.DEMO_MODELS[0])

def run_var_inference(pil: Image, return_dict=True):

    load_var()

    rgb   = cv2.resize(np.array(pil), (224, 224))
    rgb   = rgb.astype(np.float32)
    rgb   = rgb.transpose(2, 0, 1)[None]
    probs = VAR_MODEL.run(None, {'input': rgb})[0][0]
    if return_dict:
        return dict(zip(VAR_CFG.ATTRIBUTES, probs))
    return probs

def run_par_inference(pil: Image, return_dict=True):

    load_par()

    rgb   = cv2.resize(np.array(pil), (128, 256))
    rgb   = rgb.transpose(2, 0, 1)[None]
    probs = PAR_MODEL.run(None, {'input': rgb})[0][0]
    if return_dict:
        return dict(zip(PAR_CFG.ATTRIBUTES, probs))
    return probs

@torch.no_grad()
def run_video_inference(video_path: str):
    if video_path is None:
        return None

    global DETECTOR, PAR_MODEL, VAR_MODEL

    load_detector()
    load_var()
    load_par()

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    N = 15 # predict every N frame
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # mp4 編碼
    out_path = "temp/processed_output.mp4"
    Path(out_path).parent.mkdir(exist_ok=True)
    out = cv2.VideoWriter(out_path, fourcc, fps // N, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % N == 0:
            results = DETECTOR(frame, classes=[0, 2, 5, 7], conf=0.3)
            clss  = results[0].boxes.cls.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for box, cls in zip(boxes, clss):
                x1, y1, x2, y2 = box.astype(int)
                crop = Image.fromarray(frame[y1:y2, x1:x2, ::-1])

                if cls in [0]: # Human
                    probs = run_par_inference(crop, return_dict=False)
                    age_index = np.argmax(probs[:3])
                    upper_color = np.argmax(probs[ 5:15])
                    lower_color = np.argmax(probs[16:26])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color=PAR_CFG.AGE_COLOR[age_index], thickness=2)
                    cv2.rectangle(frame, (x2+3, y1), (x2+7, (y1+y2)//2), color=PAR_CFG.DRESS_COLOR[upper_color], thickness=3)
                    cv2.rectangle(frame, (x2+3, (y1+y2)//2), (x2+7, y2), color=PAR_CFG.DRESS_COLOR[lower_color], thickness=3)

                elif cls in [2, 5, 7]: # Car, Bus, Truck
                    probs = run_var_inference(crop, return_dict=False)
                    color_index = np.argmax(probs[-10:])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
                    cv2.rectangle(frame, (x2+3, y1), (x2+7, y2), color=VAR_CFG.VEHICLE_COLOR[color_index], thickness=3)

            out.write(frame)
        
        frame_idx += 1

    cap.release()
    out.release()
    
    return out_path

def on_par_gallery_select(evt: gr.SelectData):
    return Image.open(PAR_CFG.DEMO_IMGS[evt.index]).convert('RGB')

def on_var_gallery_select(evt: gr.SelectData):
    return Image.open(VAR_CFG.DEMO_IMGS[evt.index]).convert('RGB')

def on_model_select(model_choice):
    return None

# ===== UI =====
with gr.Blocks() as demo:
    gr.Markdown("## PAR Classification Demo")
    gr.Markdown("### Usage\n- 上傳圖片或從左下方選取\n- 按下左下角Predict按鈕")

    state = gr.State()
    par_model_input = gr.Dropdown(choices=PAR_CFG.DEMO_MODELS, label="選擇PAR模型", interactive=True)
    var_model_input = gr.Dropdown(choices=VAR_CFG.DEMO_MODELS, label="選擇VAR模型", interactive=True)

    with gr.Tabs():

        # Image mode
        with gr.Tab("Mode-PAR-Image"):

            with gr.Row():
                
                # PAR (Left)
                with gr.Column():
                    par_image_input = gr.Image(type="pil", label="Input Image", height=400)

                    # Create Gallery Option
                    par_gallery = gr.Gallery(value=PAR_CFG.DEMO_IMGS, columns=3, height=200)
                    par_gallery.select(on_par_gallery_select, inputs=None, outputs=par_image_input)

                    # Create Predict Button
                    par_btn = gr.Button("Predict")
                    par_btn.click(run_par_inference, inputs=par_image_input, outputs=state)

                # PAR (Right)
                with gr.Tabs():
                    for group_name, class_list in PAR_CFG.GRUOPS.items():
                        with gr.Tab(label=group_name):

                            par_output = gr.Label(num_top_classes=20)

                            def filter_group(preds, class_list=class_list):
                                if preds is None:
                                    return {}
                                return {k: v for k, v in preds.items() if k in class_list}

                            par_model_input.select(on_model_select, inputs=[par_model_input], outputs=par_output)
                            par_btn.click(fn=filter_group, inputs=state, outputs=par_output)

        with gr.Tab("Mode-VAR-Image"):

            with gr.Row():

                # VAR (left)
                with gr.Column():
                    var_image_input = gr.Image(type="pil", label="Input Image", height=400)

                    # Create Gallery Option
                    var_gallery = gr.Gallery(value=VAR_CFG.DEMO_IMGS, columns=3, height=200)
                    var_gallery.select(on_var_gallery_select, inputs=None, outputs=var_image_input)

                    # Create Predict Button
                    var_btn = gr.Button("Predict")
                    var_btn.click(run_var_inference, inputs=var_image_input, outputs=state)

                # VAR (Right)
                with gr.Tabs():
                    for group_name, class_list in VAR_CFG.GRUOPS.items():
                        with gr.Tab(label=group_name):

                            var_output = gr.Label(num_top_classes=20)

                            def filter_group(preds, class_list=class_list):
                                if preds is None:
                                    return {}
                                return {k: v for k, v in preds.items() if k in class_list}

                            var_model_input.select(on_model_select, inputs=[var_model_input], outputs=var_output)
                            var_btn.click(fn=filter_group, inputs=state, outputs=var_output)

        # Video mode
        with gr.Tab("Mode-Video"):
            with gr.Column():
                video_input = gr.Video()
                video_btn   = gr.Button("Process")
                video_output = gr.Video(format="mp4")

                video_btn.click(run_video_inference, inputs=video_input, outputs=video_output)

demo.launch(share=False, debug=True, server_port=7860, server_name="0.0.0.0")