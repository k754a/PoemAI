#this will handle our server and the requests to it

from flask import Flask, request, jsonify, render_template
from AI.chat import generate_poem

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/generate', methods = ["POST"])
def generate():
    data = request.get_json()

    word = data["word"]

    #run the ai
    output = generate_poem(word)

    #send it back
    return jsonify ({ "output" : output })

if __name__ == '__main__':
    app.run("127.0.0.1", port=6767)