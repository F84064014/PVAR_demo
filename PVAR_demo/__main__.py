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
from functools import partial

from ultralytics import YOLO    
import onnxruntime as ort
import PVAR_demo.config.par as PAR_CFG
import PVAR_demo.config.var as VAR_CFG
import PVAR_demo.plot as VPAR_plot

## Define Global Variables
MODEL_ROOT  : Path = Path(__file__).parent.parent / "models"
DETECTOR                            = None
VAR_MODEL   : ort.InferenceSession  = None
PAR_MODEL   : ort.InferenceSession  = None
PAR_GALLERY_ITEMS : list            = PAR_CFG.DEMO_IMGS
VAR_GALLERY_ITEMS : list            = VAR_CFG.DEMO_IMGS

def load_detector():
    global DETECTOR
    if DETECTOR is None:
        DETECTOR = YOLO(MODEL_ROOT / "detect" / "yolo26l.pt")
        DETECTOR = DETECTOR.to("cuda" if torch.cuda.is_available() else "cpu")

def load_var(force=False):
    global VAR_MODEL
    if force or VAR_MODEL is None:
        path = MODEL_ROOT / "var" / VAR_CFG.DEMO_MODEL
        if not path.exists():
            print(f"Model not found at {path}!!")
        VAR_MODEL = ort.InferenceSession(MODEL_ROOT / "var" / VAR_CFG.DEMO_MODEL,
                                         providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        print(f"[INFO] load model {path}")

def load_par(force=False):
    global PAR_MODEL
    if force or PAR_MODEL is None:
        path = MODEL_ROOT / "par" / PAR_CFG.DEMO_MODEL
        if not path.exists():
            print(f"Model not found at {path}!!")
        PAR_MODEL = ort.InferenceSession(MODEL_ROOT / "par" / PAR_CFG.DEMO_MODEL,
                                         providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        print(f"[INFO] load model {path}")

def run_var_inference(pil: Image, return_dict=True):

    load_var()

    input_size = VAR_MODEL.get_inputs()[0].shape[-2:][::-1]
    rgb   = cv2.resize(np.array(pil), input_size)
    rgb   = rgb.astype(np.float32)
    rgb   = rgb.transpose(2, 0, 1)[None]
    probs = VAR_MODEL.run(None, {'input': rgb})[0][0]
    if return_dict:
        return dict(zip(VAR_CFG.ATTRIBUTES, probs))
    return probs

def run_par_inference(pil: Image, return_dict=True):

    load_par()

    input_size = PAR_MODEL.get_inputs()[0].shape[-2:][::-1]
    rgb   = cv2.resize(np.array(pil), input_size)
    rgb   = rgb.transpose(2, 0, 1)[None]
    probs = PAR_MODEL.run(None, {'input': rgb})[0][0]
    if return_dict:
        return dict(zip(PAR_CFG.ATTRIBUTES, probs))
    return probs

@torch.no_grad()
def run_video_inference(video_path: str, conf: float, rate: int, par: bool, var: bool):
    if video_path is None:
        return None

    global DETECTOR, PAR_MODEL, VAR_MODEL, PAR_GALLERY_ITEMS, VAR_GALLERY_ITEMS

    load_detector()
    load_var()
    load_par()

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    N = rate # predict every N frame
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # mp4 編碼
    out_path = f"temp/Video.PAR={PAR_CFG.DEMO_MODEL}.VAR={VAR_CFG.DEMO_MODEL}.mp4"
    Path(out_path).parent.mkdir(exist_ok=True)
    out = cv2.VideoWriter(out_path, fourcc, fps // N, (width, height))

    par_crops = [] # collect crop image for gallery
    var_crops = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # For display and show (frame MUST be read only)
        frame_view = frame.copy()

        if frame_idx % N == 0:
            results = DETECTOR(frame, classes=[0, 2, 5, 7], conf=conf)
            clss  = results[0].boxes.cls.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for box, cls in zip(boxes, clss):
                x1, y1, x2, y2 = box.astype(int)
                crop = Image.fromarray(frame[y1:y2, x1:x2, ::-1])

                if par and cls in [0]: # Human
                    par_crops.append(crop)
                    probs = run_par_inference(crop, return_dict=False)
                    age_index = np.argmax(probs[:3])
                    upper_color = np.argmax(probs[ 5:15])
                    lower_color = np.argmax(probs[16:26])
                    gender   = 'F' if probs[3]>probs[4] else 'M'
                    backpage = 'B' if probs[26]>0.7 else 'X'
                    bag      = 'B' if probs[27]>0.7 else 'X'
                    hat      = 'H' if probs[28]>0.7 else 'X'
                    label    = gender + backpage + bag + hat
                    VPAR_plot.draw_box(  frame_view, (x1, y1), (x2, y2), PAR_CFG.AGE_COLOR[age_index])
                    VPAR_plot.draw_color(frame_view, (x1, y1), (x2, y2), PAR_CFG.DRESS_COLOR[upper_color], 'top')
                    VPAR_plot.draw_color(frame_view, (x1, y1), (x2, y2), PAR_CFG.DRESS_COLOR[lower_color], 'bot')
                    VPAR_plot.draw_label(frame_view, (x1, y1), (x2, y2), text=label)

                elif var and cls in [2, 5, 7]: # Car, Bus, Truck
                    var_crops.append(crop)
                    probs = run_var_inference(crop, return_dict=False)
                    model_index = np.argmax(probs[len(VAR_CFG.VEHICLE_MAKE):len(VAR_CFG.VEHICLE_MODEL)])
                    color_index = np.argmax(probs[-len(VAR_CFG.VEHICLE_COLOR):])
                    VPAR_plot.draw_box(  frame_view, (x1, y1), (x2, y2), color=(255, 0, 0))
                    VPAR_plot.draw_color(frame_view, (x1, y1), (x2, y2), color=VAR_CFG.VEHICLE_COLOR[color_index])
                    VPAR_plot.draw_label(frame_view, (x1, y1), (x2, y2), text=VAR_CFG.VEHICLE_MODEL[model_index])
            out.write(frame_view)
        
        frame_idx += 1

    cap.release()
    out.release()
    
    PAR_GALLERY_ITEMS = par_crops
    VAR_GALLERY_ITEMS = var_crops
    return out_path, par_crops, var_crops, par_crops+var_crops

def on_par_gallery_select(evt: gr.SelectData):
    image = PAR_GALLERY_ITEMS[evt.index]
    if isinstance(image, Image.Image):
        return image
    elif isinstance(image, str):
        return Image.open(image).convert('RGB')
    return image

def on_var_gallery_select(evt: gr.SelectData):
    image = VAR_GALLERY_ITEMS[evt.index]
    if isinstance(image, Image.Image):
        return image
    elif isinstance(image, str):
        return Image.open(image).convert('RGB')
    return image

def on_par_model_select(model_choice: str):
    print(f"[DEBUG] On model select = {model_choice}")
    PAR_CFG.DEMO_MODEL = model_choice
    load_par(force=True)
    return None

def on_var_model_select(model_choice: str):
    print(f"[DEBUG] On model select = {model_choice}")
    VAR_CFG.DEMO_MODEL = model_choice
    load_var(force=True)
    return None

def filter_group(preds, class_list):
    if preds is None:
        return {}
    out = {k: v for k, v in preds.items() if k in class_list}
    tot = sum(out.values())
    return {k: v / tot for k, v in out.items()}

# ===== UI =====
with gr.Blocks() as demo:
    gr.Markdown("## PVAR Classification Demo")
    gr.Markdown("### Usage\n- 上傳圖片或從Gallery選取\n- 更換圖片就會更新預測結果(Video點擊process按鈕)\n- Video Inference 後會將crop直接更新到 PAR/VAR Gallery\n- ultralytics預設會下載 torch-cpu 有需要請更新 torch-gpu")

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
                    par_image_input.change(run_par_inference, inputs=par_image_input, outputs=state)

                # PAR (Right)
                with gr.Tabs():
                    for group_name, class_list in PAR_CFG.GRUOPS.items():
                        with gr.Tab(label=group_name):

                            par_output = gr.Label(num_top_classes=20)
                            par_model_input.change(on_par_model_select, inputs=[par_model_input], outputs=par_output)
                            par_image_input.change(fn=partial(filter_group, class_list=class_list), inputs=state, outputs=par_output)

        with gr.Tab("Mode-VAR-Image"):

            with gr.Row():

                # VAR (left)
                with gr.Column():
                    var_image_input = gr.Image(type="pil", label="Input Image", height=400)

                    # Create Gallery Option
                    var_gallery = gr.Gallery(value=VAR_CFG.DEMO_IMGS, columns=3, height=200)
                    var_gallery.select(on_var_gallery_select, inputs=None, outputs=var_image_input)
                    var_image_input.change(run_var_inference, inputs=var_image_input, outputs=state)

                # VAR (Right)
                with gr.Tabs():
                    for group_name, class_list in VAR_CFG.GRUOPS.items():
                        with gr.Tab(label=group_name):

                            var_output = gr.Label(num_top_classes=20)
                            var_model_input.change(on_var_model_select, inputs=[var_model_input], outputs=var_output)
                            var_image_input.change(fn=partial(filter_group, class_list=class_list), inputs=state, outputs=var_output)

        # Video mode
        with gr.Tab("Mode-Video"):

            # Video (option)
            with gr.Row():
                detect_conf = gr.Number(value=0.3, label="Detect Conf", precision=2)
                detect_rate = gr.Number(value=  5, label="Detect every N frames", precision=0)
                par_checkbox = gr.Checkbox(value=True, label="Enable PAR")
                var_checkbox = gr.Checkbox(value=True, label="Enable VAR")

            with gr.Row():

                # Video (left, input / output / button)
                with gr.Column():
                    video_input = gr.Video()
                    video_btn   = gr.Button("Process")
                    video_output = gr.Video(format="mp4")

                # Video (right, input / output / button)
                with gr.Column():
                    crops_output = gr.Gallery(columns=6, height=350)

                video_inference_inputs = [video_input, detect_conf, detect_rate, par_checkbox, var_checkbox]
                video_btn.click(run_video_inference, inputs=video_inference_inputs, outputs=[video_output, par_gallery, var_gallery, crops_output])

if __name__=="__main__":
    # demo.launch(share=False, debug=True, server_port=7860, server_name="0.0.0.0")
    demo.launch(share=False)