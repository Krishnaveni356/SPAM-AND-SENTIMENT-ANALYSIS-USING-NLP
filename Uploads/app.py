from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def index():
    navn = "Sandvika"
    return render_template("index.html", navn=navn)

