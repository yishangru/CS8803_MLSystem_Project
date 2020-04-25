import os
import json
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
        model_save_prefix = "./static/model/generateModel"
        model = open(os.path.join(model_save_prefix, "generate.json"), mode="w", encoding="utf-8")
        data = request.json
        json.dump(data, fp=model, indent=2)
        model.close()
        # call method for generation
        return jsonify({'msg': "success"})

if __name__ == "__main__":
    app.run(debug=True)