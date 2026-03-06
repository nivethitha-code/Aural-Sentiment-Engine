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
    "models/AuralSentimentEngine_best.keras": "1t_P4NuHhQGN8g0sOmc_3GJlgPGOBij8y",
    "models/scaler.joblib":                   "1ixjVWu6Lv1Zuy6EQYY5APpNGRyfCBmJm",
    "models/encoder.joblib":                  "1W-qtY6h1MxX3N63OYtsb8yYZraP8EN91",
    "visuals/confusion_matrix.png":        "13NLJT4ZQ-yVsIHL3i_SyDDJCEBY6lMg2",
    "visuals/roc_curve.png":               "13NLJT4ZQ-yVsIHL3i_SyDDJCEBY6lMg2",
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