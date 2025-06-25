from cursor import CursorAPI
from inference import detect_text

app = CursorAPI()

@app.route("/extract", methods=["POST"])
def extract_text(req):
    image_path = req.json["image_path"]
    return {"text": detect_text(image_path)}

app.run(port=8000)
