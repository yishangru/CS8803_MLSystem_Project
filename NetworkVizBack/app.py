import os
from flask import Flask, request, jsonify, render_template, url_for
from StyleTransfer import image_style_transfer_main
from model_generation.generator import generateAPI

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    if not os.path.exists("./static/API"):
        os.mkdir("./static/API")
    if not os.path.exists("./static/API/PyTorch"):
        os.mkdir("./static/API/PyTorch")
    generatePath = "./static/API/PyTorch/VizAPI.json"
    if os.path.exists(generatePath):
        os.remove(generatePath)
    generateAPI(API="PyTorch", GeneratePath=generatePath)
    return render_template("index.html", title="VizML")

@app.route("/ModelGeneration", methods=['POST'])
def model_generation():
    if request.method == "POST":
        data = request.json
        print(data)
        return jsonify({'msg': "test"})

@app.route("/transfer", methods=['POST'])
def style_transfer():
    if request.method == "POST":
        style_image_path = request.json['style_img']
        content_image_path = request.json['content_img']
        status, msg = image_style_transfer_main(style_image_path, content_image_path)
        return jsonify({'style_img': style_image_path, 'content_img': content_image_path, 'status': status, 'msg': msg})

if __name__ == "__main__":
    app.run(debug=True)