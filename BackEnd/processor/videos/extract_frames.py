from pathlib import Path
import os
from kaggle_secrets import UserSecretsClient
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from video_utils import (
    DATA_ROOT, WORK_DIR, discover_videos, extract_keyframes_for_video,
    list_video_dirs, process_video
)
import load_dotenv


# ==== Extract keyframes ====
videos = discover_videos(DATA_ROOT)
print(f"Found {len(videos)} videos")
for i, vpath in enumerate(videos, 1):
    print(f"[{i}/{len(videos)}] {Path(vpath).name}")
    extract_keyframes_for_video(Path(vpath), Path(WORK_DIR))

# ==== ====
load_dotenv()
USE_GEMINI = bool(os.getenv("GOOGLE_API_KEY"))
flash = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0) if USE_GEMINI else None
pro   = ChatGoogleGenerativeAI(model="gemini-2.5-pro",   temperature=0) if USE_GEMINI else None

# ==== Describe frames ====
video_dirs = list_video_dirs(WORK_DIR)
for vd in video_dirs:
    process_video(vd, overwrite=False, USE_GEMINI=USE_GEMINI, pro=pro, flash=flash)
print("Done.")
