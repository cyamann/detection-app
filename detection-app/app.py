from flask import Flask, render_template, request
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import base64

app = Flask(__name__)

# Load the model and class labels
config_file = "detection/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model = "detection/frozen_inference_graph.pb"
model = cv2.dnn_DetectionModel(frozen_model, config_file)
classLables = []
with open("C:\\Users\\Ceren.CEREN\\detection\\detection-app\\detection\\Lables.txt", 'rt') as fpt:
    classLables = fpt.read().rstrip('\n').split('\n')
model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5)

            for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
                cv2.rectangle(img, boxes, (255, 0, 0), 2)
                cv2.putText(img, classLables[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)

            _, img_encoded = cv2.imencode('.jpg', img)
            img_base64 = base64.b64encode(img_encoded).decode()

            return render_template('index.html', image=img_base64)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
