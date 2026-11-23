"""Download data kompetisi ke ./data/ (train/, test/, sample_submission.csv).

Pemakaian:
    export KAGGLE_KEY="KGAT_xxx"      # token Kaggle milikmu sendiri (JANGAN di-commit)
    python data.py
"""
import os
import shutil

import kagglehub
import kagglehub.clients
import requests

TOKEN = os.environ.get("KAGGLE_KEY")
if not TOKEN:
    raise SystemExit("Set dulu environment variable KAGGLE_KEY (token Kaggle kamu).")


# Patch kagglehub supaya mendukung Kaggle Bearer Token (KGAT_...)
class BearerAuth(requests.auth.AuthBase):
    def __init__(self, token):
        self.token = token

    def __call__(self, r):
        r.headers["Authorization"] = f"Bearer {self.token}"
        return r


kagglehub.clients.KaggleApiV1Client._get_auth = lambda self: BearerAuth(TOKEN)
kagglehub.clients.KaggleApiV1Client.has_credentials = lambda self: True

competition_name = "task-1-datavidia-11-0"
target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(target_dir, exist_ok=True)

print(f"Downloading competition '{competition_name}' via kagglehub...")
path = kagglehub.competition_download(competition_name)
print(f"Downloaded to cache: {path}")

for item in os.listdir(path):
    s, d = os.path.join(path, item), os.path.join(target_dir, item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)
    print(f" - Copied {item}")
# Kalau file terbungkus subfolder (mis. data/Task-1-Datavidia-11.0/train), ratakan ke data/train
if not os.path.isdir(os.path.join(target_dir, "train")):
    for root, dirs, _ in os.walk(target_dir):
        if "train" in dirs and "test" in dirs:
            for item in os.listdir(root):
                shutil.move(os.path.join(root, item), os.path.join(target_dir, item))
            break
assert os.path.isdir(os.path.join(target_dir, "train")), "folder train/ tidak ditemukan"
print(f"Done! Data ada di {target_dir}")
