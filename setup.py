import os
import gdown

# ============================================================
#   STEP 1 — Make each file in Google Drive public:
#            Right-click file → Share → Anyone with the link
#
#   STEP 2 — Get the File ID from the share link:
#            https://drive.google.com/file/d/FILE_ID_HERE/view
#                                             ^^^^^^^^^^^^
#            Copy only that FILE_ID part and paste below
# ============================================================

FILES = {
    "models/AuralSentimentEngine_best.keras": "PASTE_KERAS_FILE_ID_HERE",
    "models/scaler.joblib":                   "PASTE_SCALER_FILE_ID_HERE",
    "models/encoder.joblib":                  "PASTE_ENCODER_FILE_ID_HERE",
    "visuals/confusion_matrix.png":        "PASTE_CONFUSION_MATRIX_FILE_ID_HERE",
    "visuals/roc_curve.png":               "PASTE_ROC_CURVE_FILE_ID_HERE",
}

def download_assets():
    os.makedirs("models", exist_ok=True)
    os.makedirs("visuals", exist_ok=True)

    for dest, file_id in FILES.items():
        if os.path.exists(dest):
            continue  # already downloaded, skip
        if "PASTE" in file_id:
            print(f"[SKIP] {dest} — no File ID set yet")
            continue
        print(f"Downloading {dest} ...")
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            dest,
            quiet=False,
            fuzzy=True
        )
        print(f"✓  {dest} ready")