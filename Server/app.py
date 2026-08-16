#this will handle our server and the requests to it

from flask import Flask, request, jsonify, render_template
from AI.chat import generate_poem

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/generate', methods = ["POST"])
def generate():
    data = request.get_json(silent=True) 

    if not data or "word" not in data:
        return jsonify({"error": "No word provided"}), 400
    
    word = data["word"].strip()

    if not word:
        return jsonify({"error": "Empty word provided"}), 400
    
    #run the ai - send it for generation
    output = generate_poem(word)

    #send it back
    return jsonify ({ "output" : output })

if __name__ == '__main__':
    app.run("127.0.0.1", port=6767)