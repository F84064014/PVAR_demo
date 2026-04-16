import io
import requests
from PIL import Image
from requests.auth import HTTPDigestAuth

# URL = "http://192.168.0.250/cgi-bin/admin/{}"
URL = "http://{}/cgi-bin/admin/{}"
AUTH = HTTPDigestAuth('Admin', '1234')

def run_device_inference(img: Image, cgi_name: str, DEVICE_CFG: object):
    AUTH = HTTPDigestAuth(DEVICE_CFG.DEVICE_USERNAME, DEVICE_CFG.DEVICE_PASSWORD)

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    prob = []

    try:
        response = requests.post(
            URL.format(DEVICE_CFG.DEVICE_IP, cgi_name), data=img_bytes,
            headers={"Content-Type": "application/octet-stream"},
            timeout=5, auth=AUTH
            )
        if response.status_code == 200:
            print("Status Code = 200")
            prob = response.json()['prob']
        else:
            print(f"Status Code: {response.status_code}")
            print(f"Raw Response: {response.text}")
    except Exception as e:
        print(f"Failed to connect to device; Error Message:\n {e}")

    print(len(prob))
    return prob