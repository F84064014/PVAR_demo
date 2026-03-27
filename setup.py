from setuptools import setup, find_packages

setup(
    name="PVAR_demo",
    version="0.1.1",
    packages=find_packages(),
    package_data={
        "PVAR_demo": ["samples/*/*.png", "models/*/*.onnx", "models/*/*.pt"]
    },
    include_package_data=True,
    install_requires=[
        "opencv-python",
        "ultralytics",
        "onnxruntime-gpu",
        "gradio",
        "Pillow",
        "numpy"
    ],
)