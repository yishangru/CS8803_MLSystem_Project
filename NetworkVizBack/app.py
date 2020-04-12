from flask import Flask, request, jsonify, render_template, url_for
from StyleTransfer import image_style_transfer_main

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html", title="Style Transfer")

@app.route("/uploadStyle", methods=['POST'])
def upload_style_image():
    pass

@app.route("/uploadContent", methods=['POST'])
def upload_content_image():
    pass

@app.route("/transfer", methods=['POST'])
def style_transfer():
    if request.method == "POST":
        style_image_path = request.json['style_img']
        content_image_path = request.json['content_img']
        status, msg = image_style_transfer_main(style_image_path, content_image_path)
        return jsonify({'style_img': style_image_path, 'content_img': content_image_path, 'status': status, 'msg': msg})

if __name__ == "__main__":
    app.run(debug=True)