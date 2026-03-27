from setuptools import setup, find_packages

setup(
    name="PVAR_demo",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "ultralytics",
        "onnxruntime-gpu",
        "gradio",
        "Pillow",
        "numpy"
    ],
)