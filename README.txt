MediPredict frontend bridge setup

Place this app.py, templates/, static/, uploads/, results/ into your project ROOT.
Your root project layout should be:

project-root/
  app.py
  gradcam.py                 (your existing script, root)
  model/
    best_model_v5.pth        (checkpoint)
  templates/
    index.html
  static/
    script.js
  uploads/
  results/

Note: The uploaded script was named gradcam_v5_fixed(2).py in this chat.
Rename/copy it to gradcam.py in your project root (or edit SCRIPT_PATH in app.py).
The script should remain unchanged otherwise.

In CMD from project-root:
  py -m pip install -r requirements.txt
  py app.py

Open:
  http://127.0.0.1:5000

The backend executes your script using subprocess and returns stdout as-is to the browser.
Checkpoint path is configured as model/best_model_v5.pth.
