import os, json, cv2, math, shutil, subprocess, base64, re
from pathlib import Path
import webvtt
import whisper
from tqdm import tqdm
import torch
from moviepy.editor import VideoFileClip
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# ==== CONFIG ====
DATA_ROOT = "./shared_data/data/video"
WORK_DIR  = "./shared_data/video_keyframes"
WHISPER_MODEL = "small"
LANGUAGE = "vi"
RESIZE_HEIGHT = 720
WINDOW_MS = 500
STEP_MS   = 100
MIN_CHARS = 3
AGGREGATE_METADATA = False

Path(WORK_DIR).mkdir(parents=True, exist_ok=True)

# ==== Whisper Model ====
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model(WHISPER_MODEL, device=device)

# ==== UTILS ====
def str2time(s: str) -> float:
    """
    Converts a time string in the format "HH:MM:SS.sss" to milliseconds.
    Args:
        s (str): Time string in the format "HH:MM:SS.sss".
    Returns:
        float: The time represented in milliseconds.
    Raises:
        ValueError: If the input string is not in the correct format.
    """
    h, m, rest = s.split(":")
    return (int(h)*3600 + int(m)*60 + float(rest)) * 1000.0

def maintain_aspect_ratio_resize(img, height=720):
    """
    Resizes an image while maintaining its aspect ratio.
    Parameters:
        img (numpy.ndarray): The input image to be resized.
        height (int, optional): The desired height of the resized image. Default is 720.
    Returns:
        numpy.ndarray: The resized image with the specified height, maintaining the original aspect ratio.
    """

    h, w = img.shape[:2]
    if h == 0: return img
    scale = height / h
    return cv2.resize(img, (int(w * scale), height), interpolation=cv2.INTER_AREA)

def vtt_from_whisper_segments(segments):
    """
    Converts a list of Whisper transcription segments into a WebVTT subtitle format string.
    Args:
        segments (list of dict): A list of dictionaries where each dictionary represents a 
            transcription segment with the following keys:
            - "start" (float): The start time of the segment in seconds.
            - "end" (float): The end time of the segment in seconds.
            - "text" (str): The transcribed text for the segment.
    Returns:
        str: A string in WebVTT format representing the transcription segments.
    Example:
        segments = [
            {"start": 0.0, "end": 2.5, "text": "Hello world."},
            {"start": 3.0, "end": 5.0, "text": "This is a test."}
        ]
        vtt_content = vtt_from_whisper_segments(segments)
        print(vtt_content)
        # Output:
        # WEBVTT
        #
        # 00:00:00.000 --> 00:00:02.500
        # Hello world.
        #
        # 00:00:03.000 --> 00:00:05.000
        # This is a test.
    """

    lines = ["WEBVTT"]
    for seg in segments:
        s, e = float(seg["start"]), float(seg["end"])
        text = (seg.get("text") or "").strip()
        def fmt(t):
            hh = int(t//3600); mm = int((t%3600)//60); ss = t%60
            return f"{hh:02d}:{mm:02d}:{ss:06.3f}"
        lines.append(f"{fmt(s)} --> {fmt(e)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def discover_videos(root: str):
    """
    Discover and return a list of all .mp4 video files within the specified root directory 
    and its subdirectories for the years 2022, 2023, 2024, and 2025.
    Args:
        root (str): The root directory to search for video files.
    Returns:
        list: A sorted list of file paths (as Path objects) to .mp4 video files found 
              in the specified directories.
    """
    vids = []
    for y in ["2022","2023","2024","2025"]:
        p = Path(root)/y
        if p.exists():
            vids += sorted(p.rglob("*.mp4"))
    return vids

def ensure_vtt(video_path: Path, out_root: Path) -> Path:

    """
    Ensures the existence of a VTT (Web Video Text Tracks) file for a given video file.
    This function checks if a corresponding VTT file exists for the provided video file.
    If it exists, it copies the VTT file to the specified output directory. If it does not
    exist, the function extracts the audio from the video, transcribes it using a Whisper
    model, and generates a VTT file.
    Args:
        video_path (Path): The path to the input video file.
        out_root (Path): The root directory where the output VTT file will be stored.
    Returns:
        Path: The path to the generated or copied VTT file.
    Raises:
        subprocess.CalledProcessError: If the `ffmpeg` command fails during audio extraction.
        Exception: If any error occurs during video processing or transcription.
    Notes:
        - The function uses `moviepy` for audio extraction. If `moviepy` fails, it falls back
          to using `ffmpeg`.
        - The transcription is performed using a Whisper model, and the resulting segments
          are converted into VTT format.
        - Temporary audio files are cleaned up after processing.
    """
    vtt_in = video_path.with_suffix(".vtt")
    vid_dir = Path(out_root) / video_path.stem
    vid_dir.mkdir(parents=True, exist_ok=True)
    vtt_out = vid_dir / f"{video_path.stem}.vtt"
    if vtt_out.exists(): return vtt_out
    if vtt_in.exists():
        shutil.copy(vtt_in, vtt_out)
        return vtt_out

    wav = vid_dir / f"{video_path.stem}.wav"
    try:
        clip = VideoFileClip(str(video_path))
        clip.audio.write_audiofile(str(wav), verbose=False, logger=None)
        clip.close()
    except Exception:
        subprocess.run(
            ["ffmpeg","-y","-i",str(video_path),"-vn","-ac","1","-ar","16000",str(wav)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    tx = whisper_model.transcribe(str(wav), language=LANGUAGE, verbose=False)
    vtt_out.write_text(vtt_from_whisper_segments(tx["segments"]), encoding="utf-8")
    try: wav.unlink()
    except: pass
    return vtt_out

def _sharpness(frame):
    return cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

def best_frame_near(cap: cv2.VideoCapture, t_ms: int, window_ms=500, step_ms=100):
    """
    Finds the best (sharpest) frame near a specified timestamp in a video.
    Args:
        cap (cv2.VideoCapture): OpenCV VideoCapture object for the video.
        t_ms (int): Target timestamp in milliseconds to search around.
        window_ms (int, optional): Time window in milliseconds to search before 
            and after the target timestamp. Defaults to 500 ms.
        step_ms (int, optional): Step size in milliseconds for iterating through 
            frames within the time window. Defaults to 100 ms.
    Returns:
        tuple: A tuple containing:
            - best_t (int or None): Timestamp in milliseconds of the sharpest frame found, 
              or None if no valid frame was found.
            - best_fr (numpy.ndarray or None): The sharpest frame as an image array, 
              or None if no valid frame was found.
    """
    best_t, best_fr, best_s = None, None, -1.0
    for cur in range(max(0, t_ms-window_ms), t_ms+window_ms+1, step_ms):
        cap.set(cv2.CAP_PROP_POS_MSEC, cur)
        ok, fr = cap.read()
        if not ok: continue
        s = _sharpness(fr)
        if s > best_s:
            best_t, best_fr, best_s = cur, fr, s
    return best_t, best_fr

def extract_keyframes_for_video(video_path: Path, out_root: Path):
    """
    Extracts keyframes from a video file and saves them along with metadata.
    This function processes a video file to extract keyframes based on metadata 
    from a corresponding WebVTT file. The extracted frames and metadata are saved 
    in a structured directory under the specified output root.
    Args:
        video_path (Path): The path to the input video file.
        out_root (Path): The root directory where the extracted frames and metadata 
                         will be saved.
    Directory Structure:
        - <out_root>/<video_id>/frames: Contains the extracted keyframe images.
        - <out_root>/<video_id>/meta_segments: Contains metadata JSON files for 
          each segment (if AGGREGATE_METADATA is False).
        - <out_root>/<video_id>_metadata.json: Contains aggregated metadata for 
          all segments (if AGGREGATE_METADATA is True).
    Metadata:
        Each metadata entry includes:
        - video_id (str): The ID of the video (derived from the video filename).
        - segment_id (int): The segment index.
        - start_time (float): The start time of the segment in seconds.
        - end_time (float): The end time of the segment in seconds.
        - capture_time (float): The timestamp of the extracted frame in seconds.
        - frame_file (str): The filename of the extracted frame.
        - extracted_frame_path (str): The full path to the extracted frame.
        - transcript (str): The transcript text for the segment.
    Notes:
        - The function ensures that the output directories exist.
        - Frames are resized to maintain aspect ratio with a fixed height (RESIZE_HEIGHT).
        - Metadata can be saved either as individual JSON files or as a single aggregated 
          JSON file, depending on the value of AGGREGATE_METADATA.
        - The function skips segments with invalid timestamps or transcripts shorter 
          than MIN_CHARS.
    Dependencies:
        - OpenCV (cv2) for video processing and frame extraction.
        - WebVTT for parsing subtitle files.
        - JSON for saving metadata.
    """
    video_id = video_path.stem
    out_dir = Path(out_root) / video_id
    frames_dir = out_dir / "frames"
    meta_dir   = out_dir / "meta_segments"
    frames_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    vtt_path = ensure_vtt(video_path, out_root=out_root)
    v = cv2.VideoCapture(str(video_path))
    metadatas = []

    for idx, entry in enumerate(webvtt.read(str(vtt_path))):
        start_ms, end_ms = str2time(entry.start), str2time(entry.end)
        if end_ms <= start_ms: continue
        text = entry.text.replace("\n"," ").strip()
        if len(text) < MIN_CHARS: continue

        mid_ms = int((start_ms + end_ms) / 2)
        best_ms, fr = best_frame_near(v, mid_ms, window_ms=WINDOW_MS, step_ms=STEP_MS)
        if fr is None: continue

        fr = maintain_aspect_ratio_resize(fr, height=RESIZE_HEIGHT)
        fname = f"seg{idx:04d}_{(best_ms or mid_ms):010d}.jpg"
        fpath = frames_dir / fname
        cv2.imwrite(str(fpath), fr)

        meta = {
            "video_id": video_id,
            "segment_id": idx,
            "start_time": round(start_ms/1000.0, 3),
            "end_time": round(end_ms/1000.0, 3),
            "capture_time": round((best_ms or mid_ms)/1000.0, 3),
            "frame_file": fpath.name,
            "extracted_frame_path": str(fpath),
            "transcript": text
        }
        if AGGREGATE_METADATA:
            metadatas.append(meta)
        else:
            (meta_dir / f"{Path(fname).stem}.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    v.release()
    if AGGREGATE_METADATA:
        big_json = out_dir / f"{video_id}_metadata.json"
        big_json.write_text(json.dumps(metadatas, ensure_ascii=False, indent=2), encoding="utf-8")

# ==== GEMINI ====
SCHEMA_INSTR = """<giữ nguyên như code gốc>"""
def clean_json_text(txt: str) -> str:
    txt = txt.strip()
    txt = re.sub(r"^```(?:json)?\s*", "", txt)
    txt = re.sub(r"\s*```$", "", txt)
    return txt

def safe_parse_json(txt: str):
    try: return json.loads(clean_json_text(txt))
    except: return None

def load_b64(path: str):
    try: return base64.b64encode(Path(path).read_bytes()).decode("utf-8")
    except: return None

def pick_model_for(transcript: str, pro, flash, USE_GEMINI):
    if not USE_GEMINI: return None
    t = transcript or ""
    return pro if (len(t) > 400 or sum(ch.isdigit() for ch in t) > 8) else flash

def build_prompt(img_b64: str) -> HumanMessage:
    return HumanMessage(content=[
        {"type":"text","text": SCHEMA_INSTR},
        {"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{img_b64}"}}
    ])

def describe_keyframe_and_merge(meta_path: Path, overwrite=False, USE_GEMINI=False, pro=None, flash=None):
    """
    Processes metadata for a video keyframe, optionally generating image captions and merging them with existing metadata.
    Args:
        meta_path (Path): The path to the metadata JSON file.
        overwrite (bool, optional): If True, overwrite existing metadata even if it already contains "image_struct" 
            and "combined_text". Defaults to False.
        USE_GEMINI (bool, optional): If True, use the GEMINI model to generate image captions and metadata. 
            Defaults to False.
        pro (optional): Optional parameter for selecting a specific model configuration. Defaults to None.
        flash (optional): Optional parameter for selecting a specific model configuration. Defaults to None.
    Returns:
        dict: The updated metadata dictionary.
    Behavior:
        - If `overwrite` is False and the metadata already contains "image_struct" and "combined_text", 
          the function returns the metadata without modification.
        - If `USE_GEMINI` is False, the function skips image caption generation and updates the metadata 
          with only the transcript.
        - If `USE_GEMINI` is True, the function:
            - Loads the base64-encoded image from the specified frame path.
            - Uses a model to generate image captions and metadata.
            - Handles errors gracefully, adding error messages to the metadata if issues occur.
        - The function combines the transcript, image captions, tags, and OCR text into a single "combined_text" field.
        - The updated metadata is written back to the `meta_path` file.
    """

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (not overwrite) and meta.get("image_struct") and meta.get("combined_text"):
        return meta

    frame_path = meta.get("extracted_frame_path") or str((meta_path.parent.parent / "frames" / meta["frame_file"]).resolve())

    if not USE_GEMINI:
        meta["image_struct"], meta["image_caption"] = None, None
        meta["combined_text"] = meta.get("transcript","")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta

    b64 = load_b64(frame_path)
    if not b64:
        meta["image_struct"], meta["image_caption"] = None, "<ERROR: cannot read image>"
        meta["combined_text"] = f"{meta.get('transcript','')}\n\n[IMAGE_CAPTION]\n{meta['image_caption']}".strip()
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta

    model = pick_model_for(meta.get("transcript",""), pro, flash, USE_GEMINI)
    prompt = build_prompt(b64)
    try:
        resp = model.invoke([prompt])
        raw = resp.content
        parsed = safe_parse_json(raw) or {
            "caption_short": raw.strip()[:200],
            "subjects": [], "scene": {"setting":"unknown","environment":[]},
            "objects": [], "ocr_text": "", "quality_flags": {"blur": False, "low_light": False}
        }
        meta["image_struct"], meta["image_caption"] = parsed, parsed.get("caption_short","").strip()
    except Exception as e:
        meta["image_struct"], meta["image_caption"] = None, f"<ERROR> {e}"

    parts = [meta.get("transcript","")]
    if meta.get("image_caption"):
        parts.append("[IMAGE_CAPTION]\n" + meta["image_caption"])
    if isinstance(meta.get("image_struct"), dict):
        ocr = meta["image_struct"].get("ocr_text","")
        tags = []
        for s in meta["image_struct"].get("subjects", []):
            if s.get("type"): tags.append(s["type"])
            tags.extend(s.get("actions",[])[:3])
        if tags: parts.append("[TAGS]\n" + ", ".join(sorted(set(tags))))
        if ocr: parts.append("[OCR]\n" + ocr)
    meta["combined_text"] = "\n\n".join([p for p in parts if p]).strip()
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta

def list_video_dirs(work_dir: str):
    return [p for p in Path(work_dir).iterdir() if p.is_dir() and (p/"meta_segments").exists()]

def process_video(video_dir: Path, overwrite=False, USE_GEMINI=False, pro=None, flash=None):
    """
    Processes video metadata by enriching it with additional information and saves the result to a JSON file.
    Args:
        video_dir (Path): The directory containing the video and its associated metadata.
        overwrite (bool, optional): Whether to overwrite existing metadata. Defaults to False.
        USE_GEMINI (bool, optional): Flag to indicate whether to use the GEMINI processing pipeline. Defaults to False.
        pro (optional): Additional processing parameter (specific to the implementation). Defaults to None.
        flash (optional): Additional processing parameter (specific to the implementation). Defaults to None.
    Returns:
        Path: The path to the enriched metadata JSON file.
    """

    metas = sorted((video_dir/"meta_segments").glob("*.json"))
    enriched = [describe_keyframe_and_merge(mp, overwrite, USE_GEMINI, pro, flash) for mp in tqdm(metas, desc=f"Describe {video_dir.name}", leave=False)]
    out = video_dir / f"{video_dir.name}_metadata_enriched.json"
    out.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
