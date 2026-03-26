# Automated Helmet Violation Detection and Number Plate Recognition

from flask import Flask, request, render_template_string
import cv2
import pytesseract
import os
from ultralytics import YOLO

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Change this path if needed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load YOLO model
model = YOLO("yolov8n.pt")

# ---------------- FLASK ----------------
app = Flask(__name__)

# ---------------- HTML ----------------
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Helmet Violation System</title>
</head>
<body>

<h2>Helmet Violation Detection System 🚦</h2>

<form method="POST" action="/upload" enctype="multipart/form-data">
    <input type="file" name="image" required>
    <button type="submit">Upload Image</button>
</form>

{% if image %}
    <br>
    <img src="{{ image }}" width="400"><br><br>

    <form method="POST" action="/detect">
        <input type="hidden" name="path" value="{{ image }}">
        <button type="submit">Detect</button>
    </form>
{% endif %}

{% if result %}
    <h3>Result:</h3>
    <pre>{{ result }}</pre>
{% endif %}

</body>
</html>
"""

# ---------------- DETECTION ----------------
def detect_objects(image_path):
    img = cv2.imread(image_path)
    results = model(img)[0]

    bike = False
    person = False
    helmet = False
    plate_img = None

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label in ["motorcycle", "motorbike"]:
            bike = True

        if label == "person":
            person = True

        if label == "helmet":
            helmet = True

        # Simple plate crop (basic)
        if bike:
            plate_img = img[y1:y2, x1:x2]

    return bike, person, helmet, plate_img


# ---------------- OCR ----------------
def extract_number(img):
    if img is None:
        return "Not Detected"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config="--psm 8")
    return text.strip()


# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["image"]
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)
    return render_template_string(HTML, image=path)


@app.route("/detect", methods=["POST"])
def detect():
    path = request.form["path"]

    bike, person, helmet, plate_img = detect_objects(path)

    result = ""

    if bike and person:
        result += "Bike & Person Detected ✅\n"

        if helmet:
            result += "Helmet Detected ✅\n"
        else:
            result += "No Helmet ❌\n"
            number = extract_number(plate_img)
            result += f"Number Plate: {number}\n"
    else:
        result = "No Bike Found ❌"

    return render_template_string(HTML, image=path, result=result)


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)