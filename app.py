from data_figures import build_figures
from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import json
import threading
import time
import webbrowser
from threading import Timer
import signal

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
TIMEOUT_SECONDS = 5 * 60  # 5 minute timeout
active_windows = 0  # Track number of open browser windows
shutdown_flag = False

# Track last activity time
last_activity_time = time.time()


def graceful_shutdown():
    print("grace")
    global shutdown_flag
    shutdown_flag = True
    time.sleep(1.1)  # Small delay to ensure response is sent
    if shutdown_flag:
        shutdown_flag = False  # ensure that there is only one kill call
        print("Server shutting down...")
        os.kill(os.getpid(), signal.SIGINT)
    else:
        return


@app.route("/shutdown")
def shutdown():
    print("/shutdown")
    graceful_shutdown()
    return jsonify(success=True)


@app.route("/window-opened", methods=["POST"])
def update_activity():
    global active_windows, shutdown_flag, last_activity_time
    shutdown_flag = False
    active_windows += 1
    print("active wind++", active_windows)
    last_activity_time = time.time()
    return jsonify(success=True)


@app.route("/window-closed", methods=["POST"])
def window_closed():
    global active_windows, shutdown_flag
    active_windows -= 1
    print("active wind--", active_windows)
    if active_windows <= 0 and not shutdown_flag:
        threading.Thread(target=graceful_shutdown).start()
    return jsonify(success=True)


def get_remaining_time():
    return max(0, TIMEOUT_SECONDS - (time.time() - last_activity_time))


@app.route("/remaining-time")
def remaining_time():
    if shutdown_flag:
        out = 0
    else:
        out = get_remaining_time()
    return jsonify({"remaining_time": out})


def check_inactivity():
    while True:
        time.sleep(1)
        if shutdown_flag:
            break
        if get_remaining_time() <= 0:
            print("Shutting down due to inactivity...")
            graceful_shutdown()
            break


@app.route("/", methods=["GET", "POST"])
def index():
    print("index")
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file and file.filename.endswith(".json"):
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            return redirect(url_for("dashboard", filename=file.filename))
    return render_template("index.html", remaining_time=get_remaining_time())


@app.route("/dashboard/<filename>")
def dashboard(filename):
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    with open(filepath, "r") as f:
        data = json.load(f)

    figures = build_figures(data)
    # TODO error handing

    return render_template(
        "dashboard.html", figures=figures, remaining_time=get_remaining_time()
    )


def open_browser():
    webbrowser.open_new("http://localhost:5000/")


if __name__ == "__main__":
    # Start inactivity checker thread
    inactivity_thread = threading.Thread(target=check_inactivity, daemon=True)
    inactivity_thread.start()
    Timer(1, open_browser).start()
    app.run(debug=False)
