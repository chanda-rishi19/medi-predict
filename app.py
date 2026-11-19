from flask import Flask, render_template, request, jsonify, send_from_directory
from pathlib import Path
import json
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
RESULT_DIR = ROOT / "results"
SCRIPT_PATH = ROOT / "gradcam_v5_fixed.py"
CHECKPOINT_PATH = ROOT / "models" / "best_model_v5.pth"
ANALYSIS_PREFIX = "MEDIPREDICT_ANALYSIS_JSON="

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)


def extract_analysis(stdout):
    """Extract the structured explanation while preserving the original CMD output."""
    for line in stdout.splitlines():
        if line.startswith(ANALYSIS_PREFIX):
            try:
                return json.loads(line[len(ANALYSIS_PREFIX):])
            except json.JSONDecodeError:
                return None
    return None


def display_output(stdout):
    """Keep the human-readable CMD output; hide only the internal JSON transport line."""
    return "\n".join(
        line for line in stdout.splitlines()
        if not line.startswith(ANALYSIS_PREFIX)
    ).strip()


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/predict")
def predict():
    file = request.files.get("image")
    if not file or not file.filename:
        return jsonify({"error": "Please choose an X-ray image."}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        return jsonify({"error": "Unsupported image type."}), 400

    job_id = uuid.uuid4().hex
    input_path = UPLOAD_DIR / f"{job_id}{ext}"
    job_dir = RESULT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    output_base = job_dir / "gradcam"
    file.save(input_path)

    command = [
        sys.executable, str(SCRIPT_PATH), str(input_path.resolve()),
        "--checkpoint", str(CHECKPOINT_PATH),
        "--output", str(output_base.resolve())
    ]

    try:
        completed = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False
        )
    except Exception as exc:
        return jsonify({"error": f"Could not start prediction: {exc}"}), 500

    if completed.returncode != 0:
        return jsonify({
            "error": "Prediction process failed.",
            "output": completed.stdout,
            "stderr": completed.stderr,
            "returncode": completed.returncode
        }), 500

    heatmap_name = "gradcam_heatmap.png"
    overlay_name = "gradcam_overlay.png"
    heatmap_path = job_dir / heatmap_name
    overlay_path = job_dir / overlay_name

    return jsonify({
        "output": display_output(completed.stdout),
        "analysis": extract_analysis(completed.stdout),
        "heatmap_url": f"/results/{job_id}/{heatmap_name}" if heatmap_path.exists() else None,
        "overlay_url": f"/results/{job_id}/{overlay_name}" if overlay_path.exists() else None
    })


@app.get("/results/<job_id>/<filename>")
def result_image(job_id, filename):
    # Limit served files to the two generated visualization images.
    if filename not in {"gradcam_heatmap.png", "gradcam_overlay.png"}:
        return jsonify({"error": "File not found."}), 404
    return send_from_directory(RESULT_DIR / job_id, filename)

if __name__ == "__main__":
    import os
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
