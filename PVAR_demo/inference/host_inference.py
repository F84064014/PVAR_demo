import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort

def run_host_inference(
        pil: Image,
        model: ort.InferenceSession,
        dtype = np.uint8,
    ) -> np.ndarray:

    """
    This function simple return prob = model(pil)
    """
    input_size = model.get_inputs()[0].shape[-2:][::-1]
    rgb = cv2.resize(np.array(pil), input_size)
    rgb = rgb.astype(dtype)
    rgb = rgb.transpose(2, 0, 1)[None]
    probs = model.run(None, {'input': rgb})[0][0]
    return probs