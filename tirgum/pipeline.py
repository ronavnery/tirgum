#!/usr/bin/env python3
"""Download a YouTube video and produce English subtitles from Hebrew.

Pipeline:
  1. Download the video with yt-dlp (plus Hebrew subtitles if the uploader provided them).
  2. If there are no Hebrew subtitles, transcribe the audio with a Hebrew-tuned Whisper model.
  3. Translate the Hebrew subtitles to English with Claude (or Whisper's built-in translation
     as an offline fallback).
  4. Write .he.srt / .en.srt files and a copy of the video with English subtitles
     (soft track by default, or burned into the picture with --burn).
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("TIRGUM_HOME", PACKAGE_DIR.parent)).resolve()
DOWNLOADS = DATA_DIR / "downloads"

DEFAULT_WHISPER_MODEL = "ivrit-ai/whisper-large-v3-turbo-ct2"  # fine-tuned for Hebrew
WHISPER_TRANSLATE_MODEL = "large-v3"  # turbo models were not trained for translation
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
BATCH_SIZE = 60  # subtitle segments per translation request


# ---------------------------------------------------------------------- progress
# Stages and their share of the overall progress bar (roughly their share of the run time).
STAGES = {"download": 0.10, "detect": 0.03, "transcribe": 0.22, "scan": 0.15,
          "translate": 0.25, "render": 0.25}
ON_PROGRESS = None  # set by the web server: called with (stage, fraction 0..1, message)


def report(stage: str, fraction: float, message: str = "") -> None:
    if ON_PROGRESS:
        ON_PROGRESS(stage, max(0.0, min(1.0, fraction)), message)


# --------------------------------------------------------------------------- SRT


@dataclass
class Segment:
    start: float  # seconds
    end: float
    text: str


def _fmt_ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def _parse_ts(ts: str) -> float:
    h, m, rest = ts.strip().replace(".", ",").split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def write_srt(segments: list[Segment], path: Path) -> None:
    blocks = [
        f"{i}\n{_fmt_ts(seg.start)} --> {_fmt_ts(seg.end)}\n{seg.text.strip()}\n"
        for i, seg in enumerate(segments, 1)
    ]
    path.write_text("\n".join(blocks), encoding="utf-8")


def read_srt(path: Path) -> list[Segment]:
    segments = []
    content = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = block.strip().split("\n")
        idx = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if idx is None:
            continue
        start, end = lines[idx].split("-->")
        text = "\n".join(lines[idx + 1 :])
        text = re.sub(r"<[^>]+>", "", text).strip()  # strip inline tags like <i>
        if text:
            segments.append(Segment(_parse_ts(start), _parse_ts(end.split()[0]), text))
    return segments


# ---------------------------------------------------------------------- download


def download(url: str, out_root: Path, want_subs: bool) -> tuple[Path, Path | None, dict]:
    """Download the video (and any manual Hebrew subtitles). Returns (video, he_srt|None)."""
    import yt_dlp

    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    title = re.sub(r'[\\/:*?"<>|]+', "_", info.get("title") or "video").strip()[:80]
    work_dir = out_root / f"{title} [{info['id']}]"
    work_dir.mkdir(parents=True, exist_ok=True)

    opts = {
        "outtmpl": str(work_dir / "video.%(ext)s"),
        # Highest resolution, then highest bitrate. Prefer H.264 so the result plays
        # everywhere (QuickTime, phones, TVs); many players can't handle AV1/VP9.
        "format": "bv*[vcodec^=avc1]+ba[ext=m4a]/bv*+ba/b",
        "format_sort": ["res", "tbr"],
        "merge_output_format": "mp4",
        "noplaylist": True,
        "progress_hooks": [lambda d: report(
            "download", (d.get("downloaded_bytes") or 0) / (d.get("total_bytes") or d.get("total_bytes_estimate") or 1),
            "Downloading video") if d.get("status") == "downloading" else None],
    }
    if want_subs:
        # Only uploader-provided subtitles. NEVER YouTube's auto-generated captions or
        # auto-translations: they are low quality, and the Hebrew Whisper model does far better.
        opts.update(
            writesubtitles=True,
            writeautomaticsub=False,
            subtitleslangs=["he", "iw", "he-IL", "iw-IL"],
            subtitlesformat="srt/vtt/best",
            postprocessors=[{"key": "FFmpegSubtitlesConvertor", "format": "srt"}],
        )

    print(f"⬇  Downloading: {info.get('title')}")
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])

    video = work_dir / "video.mp4"
    if not video.exists():
        candidates = [p for p in work_dir.glob("video.*") if p.suffix not in (".srt", ".vtt")]
        if not candidates:
            sys.exit("Download failed: no video file produced.")
        video = candidates[0]

    he_srt = next(iter(sorted(work_dir.glob("video.*.srt"))), None) if want_subs else None
    if he_srt:
        target = work_dir / "subtitles.he.srt"
        he_srt.replace(target)
        he_srt = target
        print("✓ Found Hebrew subtitles on YouTube")
    report("download", 1, "Downloaded")
    return video, he_srt, info


# ------------------------------------------------------------------ transcription


def extract_audio(video: Path) -> Path:
    audio = video.with_name("audio.wav")
    if not audio.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
             "-vn", "-ac", "1", "-ar", "16000", str(audio)],
            check=True,
        )
    return audio


MAX_SUB_SECONDS = 6.0
MAX_SUB_CHARS = 80


def split_segment(seg) -> list[Segment]:
    """Split a long Whisper segment into subtitle-sized pieces using word timestamps,
    preferring to break after punctuation."""
    words = seg.words or []
    if not words or (seg.end - seg.start <= MAX_SUB_SECONDS and len(seg.text) <= MAX_SUB_CHARS):
        return [Segment(seg.start, seg.end, seg.text.strip())]

    pieces: list[Segment] = []
    current: list = []
    for i, w in enumerate(words):
        current.append(w)
        text = "".join(x.word for x in current).strip()
        duration = w.end - current[0].start
        at_punct = w.word.rstrip()[-1:] in ".,?!:;"
        is_last = i == len(words) - 1
        too_long = duration >= MAX_SUB_SECONDS or len(text) >= MAX_SUB_CHARS
        if is_last or too_long or (at_punct and duration >= MAX_SUB_SECONDS / 2):
            pieces.append(Segment(current[0].start, w.end, text))
            current = []
    return pieces


def run_whisper(audio: Path, model_name: str, task: str) -> list[Segment]:
    from faster_whisper import WhisperModel

    print(f"🎙  Whisper ({task}) with {model_name} — first run downloads the model…")
    model = WhisperModel(model_name, device="auto", compute_type="auto")
    seg_iter, info = model.transcribe(
        str(audio), language="he", task=task, vad_filter=True, beam_size=5,
        word_timestamps=True,
    )
    segments = []
    for seg in seg_iter:
        segments.extend(split_segment(seg))
        pct = min(100, seg.end / info.duration * 100) if info.duration else 0
        print(f"\r   {pct:5.1f}%  {_fmt_ts(seg.end)}", end="", flush=True)
        report("transcribe", pct / 100, "Transcribing on this computer")
    print()
    return [s for s in segments if s.text]


# ------------------------------------------------------------------------ RunPod
# Transcription on a RunPod serverless GPU using ivrit-ai's ready-made worker image.

RUNPOD_IMAGE = "yairlifshitz/whisper-runpod-serverless:latest"
RUNPOD_REST = "https://rest.runpod.io/v1"
RUNPOD_GPUS = ["NVIDIA GeForce RTX 4090", "NVIDIA L40S", "NVIDIA RTX A6000", "NVIDIA A100 80GB PCIe"]
RUNPOD_CHUNK_SECONDS = 20 * 60  # keeps each base64 request under RunPod's 10 MB limit


def runpod_configured() -> bool:
    return bool(os.environ.get("RUNPOD_API_KEY") and os.environ.get("RUNPOD_ENDPOINT_ID"))


def run_runpod(video: Path, model_name: str) -> list[Segment]:
    import concurrent.futures
    import tempfile

    import ivrit

    model = ivrit.load_model(
        engine="runpod", model=model_name,
        api_key=os.environ["RUNPOD_API_KEY"], endpoint_id=os.environ["RUNPOD_ENDPOINT_ID"],
    )
    tmp = Path(tempfile.mkdtemp(prefix="yt-translate-"))
    # Compressed mono speech audio, cut into chunks: ~4.8 MB per 20 minutes.
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "libopus", "-b:a", "32k", "-f", "segment",
         "-segment_time", str(RUNPOD_CHUNK_SECONDS), str(tmp / "chunk%03d.ogg")],
        check=True,
    )
    chunks = sorted(tmp.glob("chunk*.ogg"))
    print(f"☁️  Transcribing on RunPod GPU ({len(chunks)} chunk{'s' * (len(chunks) > 1)}, {model_name})…")

    done: list[int] = []
    report("transcribe", 0.02, "Waiting for a GPU worker")

    def transcribe(idx_chunk: tuple[int, Path]) -> list[Segment]:
        idx, chunk = idx_chunk
        offset = idx * RUNPOD_CHUNK_SECONDS
        result = model.transcribe(path=str(chunk), language="he")
        pieces = [p for seg in result["segments"] for p in split_segment(seg)]
        print(f"   ✓ chunk {idx + 1}/{len(chunks)}")
        done.append(idx)
        report("transcribe", len(done) / len(chunks), f"Transcribing on GPU ({len(done)}/{len(chunks)})")
        return [Segment(p.start + offset, p.end + offset, p.text) for p in pieces if p.text]

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(chunks))) as pool:
            results = list(pool.map(transcribe, enumerate(chunks)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return [s for chunk_segments in results for s in chunk_segments]


def setup_runpod() -> None:
    """Create the RunPod template + serverless endpoint and save its id to .env."""
    import requests

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        sys.exit("Set RUNPOD_API_KEY in .env first (RunPod console → Settings → API Keys).")
    headers = {"Authorization": f"Bearer {api_key}"}

    def call(method: str, path: str, body: dict | None = None):
        r = requests.request(method, f"{RUNPOD_REST}{path}", headers=headers, json=body, timeout=60)
        if r.status_code >= 400:
            sys.exit(f"RunPod API error {r.status_code} on {path}: {r.text}")
        return r.json()

    templates = [t for t in call("GET", "/templates") if t.get("name") == "yt-translate-whisper"]
    template = templates[0] if templates else call("POST", "/templates", {
        "name": "yt-translate-whisper",
        "imageName": RUNPOD_IMAGE,
        "isServerless": True,
        "containerDiskInGb": 30,
        "volumeInGb": 0,
        "ports": [],
    })
    endpoint = call("POST", "/endpoints", {
        "name": "yt-translate-whisper",
        "templateId": template["id"],
        "gpuTypeIds": RUNPOD_GPUS,
        "workersMin": 0,  # scale to zero: no cost while idle
        "workersMax": 4,
        "idleTimeout": 60,
        "flashboot": True,
        "executionTimeoutMs": 30 * 60 * 1000,
    })
    save_env("RUNPOD_ENDPOINT_ID", endpoint["id"])
    print(f"✓ Created RunPod endpoint {endpoint['id']} (saved to .env). "
          "It scales to zero, so you only pay while transcribing.")


# ------------------------------------------------------------------ .env config

ENV_FILE = DATA_DIR / ".env"


def load_env() -> None:
    """.env is this tool's source of truth: its values override the shell's. When it supplies
    an Anthropic key, credentials or endpoints inherited from the shell (e.g. another tool's
    proxy settings) are dropped so requests go straight to Anthropic with that key."""
    if not ENV_FILE.exists():
        return
    values = {}
    for line in ENV_FILE.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.strip().startswith("#") and value.strip():
            values[key.strip()] = value.strip()
    if "ANTHROPIC_API_KEY" in values:
        for inherited in ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
            if inherited not in values:
                os.environ.pop(inherited, None)
    os.environ.update(values)


def check_claude_key() -> None:
    """Fail fast (free request) before spending anything on download or RunPod."""
    import anthropic

    try:
        anthropic.Anthropic().models.list(limit=1)
    except anthropic.AuthenticationError:
        sys.exit("Anthropic rejected the API key. Check ANTHROPIC_API_KEY in .env.")
    except anthropic.APIConnectionError:
        sys.exit("Couldn't reach the Anthropic API. Check your internet connection.")


def save_env(key: str, value: str) -> None:
    lines = ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else []
    lines = [l for l in lines if not l.startswith(f"{key}=")] + [f"{key}={value}"]
    ENV_FILE.write_text("\n".join(lines) + "\n")
    ENV_FILE.chmod(0o600)
    os.environ[key] = value


# ---------------------------------------------------------------------- feedback
# Your review clicks (yt-translate --review DIR) are saved under feedback/ and fed back into
# every future translation: approved/corrected lines become examples, rules.md becomes rules.

FEEDBACK_DIR = DATA_DIR / "feedback"
MAX_EXAMPLES = 40


def load_feedback() -> str:
    """Prompt section built from your saved rules and reviewed lines ('' if none yet)."""
    parts = []
    rules = FEEDBACK_DIR / "rules.md"
    if rules.exists() and rules.read_text().strip():
        parts.append("Translation rules from the user (always follow these):\n" + rules.read_text().strip())
    examples = []
    for f in sorted(FEEDBACK_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime):
        for label in json.loads(f.read_text()).get("labels", {}).values():
            approved = label.get("corrected") or next(
                (label["candidates"][k] for k in ("final", "voice", "image")
                 if k in label.get("accurate", []) and label["candidates"].get(k)), None)
            # Most useful: fixes, and lines where the user picked something other than our output.
            informative = label.get("corrected") or "final" not in label.get("accurate", [])
            if approved and informative:
                examples.append((label.get("updated", ""), label, approved))
    examples.sort(key=lambda e: e[0])
    if examples:
        lines = []
        for _, label, approved in examples[-MAX_EXAMPLES:]:
            src = f"speech: {label.get('he_speech', '')}"
            if label.get("he_caption"):
                src += f" | on-screen caption: {label['he_caption']}"
            lines.append(f"- {src}\n  -> {approved}")
        parts.append("Lines the user reviewed and approved or corrected. Match their choices, "
                     "style and name spellings:\n" + "\n".join(lines))
    return "\n\n".join(parts)


def serve_review(work_dir: Path, port: int = 8765) -> None:
    """Local review page: mark which translation is accurate, fix lines, write rules."""
    import http.server
    import threading
    import webbrowser
    from datetime import datetime, timezone

    work_dir = work_dir.resolve()
    m = re.search(r"\[([\w-]{6,})\]$", work_dir.name)
    video_id = m.group(1) if m else work_dir.name
    FEEDBACK_DIR.mkdir(exist_ok=True)
    store = FEEDBACK_DIR / f"{video_id}.json"
    rules = FEEDBACK_DIR / "rules.md"
    page = (PACKAGE_DIR / "review.html").read_bytes()
    lock = threading.Lock()

    video = work_dir / "video.mp4"  # the original, without our subtitles
    final = read_srt(work_dir / "subtitles.en.srt") if (work_dir / "subtitles.en.srt").exists() else []
    captions = work_dir / "captions.json"
    rows = []
    if captions.exists():  # one row per caption, exactly as the English is timed
        for i, u in enumerate(json.loads(captions.read_text())["units"]):
            rows.append({"t": _fmt_ts(u["start"])[:8], "start": u["start"], "end": u["end"],
                         "he_speech": u["speech"], "he_caption": "\n".join(u.get("caption", [])),
                         "he_caption_img": f"/caption/{(u.get('images') or [u.get('image')])[0]}" if (u.get("images") or u.get("image")) else "",
                         "verdict": "", "note": "",
                         "candidates": {"final": final[i].text} if i < len(final) else {}})
    else:
        he = read_srt(work_dir / "subtitles.he.srt")
        compare = json.loads((work_dir / "compare.json").read_text()) if (work_dir / "compare.json").exists() else []
        for i, seg in enumerate(he):
            c = compare[i] if i < len(compare) else {}
            cands = {"final": final[i].text if i < len(final) else "",
                     "voice": c.get("voice", ""), "image": c.get("image", "")}
            rows.append({"t": _fmt_ts(seg.start)[:8], "start": seg.start, "end": seg.end,
                         "he_speech": seg.text, "he_caption": c.get("he_caption", ""),
                         "verdict": c.get("verdict", ""), "note": c.get("note", ""),
                         "candidates": {k: v for k, v in cands.items() if v}})
    for i, r in enumerate(rows):
        r["i"] = i
        r["key"] = f"{r['start']:.1f}"  # labels survive re-translation (row order may change)

    def read_store() -> dict:
        return json.loads(store.read_text()) if store.exists() else {"video": video_id, "labels": {}}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, body: bytes, ctype: str = "application/json", code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self.send(page, "text/html; charset=utf-8")
            elif self.path.startswith("/media"):
                self.send_media()
            elif m := re.fullmatch(r"/caption/(\d{4}(?:-\d)?\.jpg)", self.path):
                img = work_dir / "caption_images" / m.group(1)
                self.send(img.read_bytes() if img.exists() else b"", "image/jpeg", 200 if img.exists() else 404)
            elif self.path == "/api/data":
                saved = read_store()["labels"]
                by_row = {r["i"]: saved[r["key"]] for r in rows if r["key"] in saved}
                self.send(json.dumps({"title": work_dir.name, "rows": rows,
                                      "labels": by_row,
                                      "rules": rules.read_text() if rules.exists() else ""},
                                     ensure_ascii=False).encode())
            else:
                self.send(b"{}", code=404)

        def send_media(self):
            """Stream the original video with HTTP Range support, so the page can seek."""
            size = video.stat().st_size
            start, end = 0, size - 1
            rng = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            if rng:
                if rng.group(1):
                    start = int(rng.group(1))
                    end = int(rng.group(2)) if rng.group(2) else end
                else:  # suffix range: last N bytes
                    start = size - int(rng.group(2))
            end = min(end, size - 1, start + 8 * 1024 * 1024 - 1)  # at most 8 MB per response
            self.send_response(206 if rng else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with open(video, "rb") as f:
                f.seek(start)
                remaining = end - start + 1
                try:
                    while remaining > 0:
                        chunk = f.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the browser cancelled the request after seeking elsewhere

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            with lock:
                if self.path == "/api/label":
                    data = read_store()
                    i = int(body["i"])
                    key = rows[i]["key"]
                    if not body.get("accurate") and not body.get("corrected"):
                        data["labels"].pop(key, None)
                    else:
                        r = rows[i]
                        data["labels"][key] = {
                            "accurate": body.get("accurate", []), "corrected": body.get("corrected", ""),
                            "he_speech": r["he_speech"], "he_caption": r["he_caption"],
                            "candidates": r["candidates"], "t": r["t"],
                            "updated": datetime.now(timezone.utc).isoformat()}
                    tmp = store.with_suffix(".tmp")
                    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
                    tmp.replace(store)
                elif self.path == "/api/rules":
                    rules.write_text(body.get("rules", ""))
                else:
                    return self.send(b"{}", code=404)
            self.send(b'{"ok":true}')

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://localhost:{port}/"
    print(f"📝 Review page: {url}  (labels save to {store.relative_to(FEEDBACK_DIR.parent)}; Ctrl+C to stop)")
    if "--no-open" not in sys.argv:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


# ------------------------------------------------------------------- translation

SYSTEM_PROMPT = """You translate Hebrew video subtitles into natural, fluent English subtitles.

- Translate each numbered segment and return exactly one translation per input id, in order.
- Segments are fragments of continuous speech: use the surrounding segments for context, \
but keep each translation aligned to its own segment's meaning so timing stays correct.
- Translate the meaning, not the words. Hebrew slang, idioms and expressions (e.g. "wallah", \
"on my mother", "sababa", "yalla", "achi") become natural English with the same tone and register \
("I swear", "no way", "cool", "come on", "bro") - never a literal word-for-word rendering.
- Keep lines concise and readable as subtitles, the way a professional subtitler would. \
Transliterate Hebrew names and places the way they are usually written in English.
- If a segment is noise, music, or already English, return it cleaned up rather than dropping it."""


CAPTION_CROP = 0.32  # bottom fraction of the frame where burned-in captions sit
CAPTION_WIDTH = 0.8  # centred fraction of the width (captions are centred)


def video_duration(video: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(video)], capture_output=True, text=True).stdout
    return float(out.strip() or 0)


def caption_frame(video: Path, t: float) -> str:
    """Base64 JPEG of the bottom-centre (caption) area of the frame at time t."""
    import base64

    x = (1 - CAPTION_WIDTH) / 2
    jpg = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-ss", f"{max(0.0, t):.2f}", "-i", str(video),
         "-frames:v", "1",
         "-vf", f"crop=iw*{CAPTION_WIDTH}:ih*{CAPTION_CROP}:iw*{x}:ih*{1 - CAPTION_CROP},scale=720:-2",
         "-q:v", "4", "-f", "image2pipe", "-vcodec", "mjpeg", "-"],
        capture_output=True, check=True).stdout
    return base64.b64encode(jpg).decode()


def image_block(b64: str) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}


def has_burned_captions(video: Path, model: str) -> bool:
    """Ask Claude whether sample frames show burned-in Hebrew captions."""
    import anthropic
    from pydantic import BaseModel

    class Answer(BaseModel):
        has_hebrew_captions: bool

    duration = video_duration(video)
    content: list[dict] = [image_block(caption_frame(video, duration * f / 10)) for f in range(1, 10)]
    content.append({"type": "text", "text":
                    "These are crops of the bottom of frames from one video. Does the video have "
                    "Hebrew subtitles/captions of the spoken dialogue burned into the picture? "
                    "Ignore logos, channel names, news tickers and signs in the scene."})
    response = anthropic.Anthropic().messages.parse(
        model=model, max_tokens=2000, output_config={"effort": "low"},
        messages=[{"role": "user", "content": content}], output_format=Answer,
    )
    track_usage(model, response.usage)
    return bool(response.parsed_output and response.parsed_output.has_hebrew_captions)


# ------------------------------------------------------- burned-in caption timing
# English subtitles follow the broadcaster's Hebrew captions exactly: same appearance and
# disappearance times and the same number of lines. A 10 fps pixel scan of the caption area
# (local, free) finds when each caption appears, changes and disappears, how many lines it has
# and how wide it is. Claude then sees one image of each caption while translating, so there is
# no separate (and costly) step of transcribing the Hebrew.

SCAN_FPS = 10
STRIP_EVERY = 2  # scan frames between saved caption images (5 per second)
SCAN_W, SCAN_H = 480, 108

# $ per million tokens (input, output), to report what each run cost.
PRICES = {"claude-haiku-4-5": (1, 5), "claude-sonnet-4-6": (3, 15), "claude-sonnet-5": (2, 10),
          "claude-opus-5": (5, 25), "claude-opus-4-8": (5, 25),
          # Gemini 3.8 Flash launch price until 2026-12-31; doubles on 2027-01-01.
          "gemini-3.8-flash": (0.75, 3.75), "gemini-3.5-flash-lite": (0.30, 2.50)}
USAGE: dict[str, list[int]] = {}


def no_thinking(model: str) -> dict:
    """Request options that keep a model from spending output tokens on reasoning: Sonnet 5 and
    Opus 5 think unless told not to; older models only think when asked."""
    return {"thinking": {"type": "disabled"}} if model in ("claude-sonnet-5", "claude-opus-5") else {}


def track_usage(model: str, usage) -> None:
    tokens_in = (usage.input_tokens + (usage.cache_creation_input_tokens or 0)
                 + (usage.cache_read_input_tokens or 0))
    totals = USAGE.setdefault(model, [0, 0])
    totals[0] += tokens_in
    totals[1] += usage.output_tokens


def print_cost() -> None:
    total = 0.0
    for model, (tokens_in, tokens_out) in USAGE.items():
        p_in, p_out = PRICES.get(model, (0, 0))
        cost = tokens_in / 1e6 * p_in + tokens_out / 1e6 * p_out
        total += cost
        print(f"   {model}: {tokens_in:,} in + {tokens_out:,} out = ${cost:.3f}")
    if USAGE:
        print(f"💲 AI cost this run: ${total:.3f}")


@dataclass
class CaptionScan:
    has_text: "np.ndarray"   # per scan frame: caption-like text present
    change: "np.ndarray"     # per scan frame: how much the stable text changed here (0..~2)
    band: tuple[int, int]    # caption rows within the scan crop (0..SCAN_H)
    strips: list[str]        # base64 JPEG caption-band images, one per STRIP_EVERY scan frames
    height: int              # video height
    cols: "np.ndarray"       # per scan frame: [upper line, lower line] columns containing text
    packed: list             # per scan frame: packed text mask of the caption band

    def bottom(self) -> float:
        """Bottom of the caption band as a fraction of the video height."""
        return 1 - CAPTION_CROP + CAPTION_CROP * self.band[1] / SCAN_H

    def mask(self, t: int) -> "np.ndarray":
        import numpy as np

        rows = self.band[1] - self.band[0]
        return np.unpackbits(self.packed[t])[: rows * SCAN_W].reshape(rows, SCAN_W).astype(bool)


def scan_captions(video: Path, strips: bool = True) -> CaptionScan:
    """One ffmpeg pass: a low-res greyscale caption-area stream for the pixel analysis, plus
    5 fps crops that are cut down to the caption band for Claude to look at."""
    import base64
    import tempfile

    import numpy as np

    print("🔎 Scanning the burned-in captions' timing…")
    tmp = Path(tempfile.mkdtemp(prefix="yt-translate-caps-"))
    x = (1 - CAPTION_WIDTH) / 2
    crop = f"crop=iw*{CAPTION_WIDTH}:ih*{CAPTION_CROP}:iw*{x}:ih*{1 - CAPTION_CROP}"
    graph = f"[0:v]{crop},fps={SCAN_FPS},scale={SCAN_W}:{SCAN_H},format=gray[g]"
    outputs = ["-map", "[g]", "-f", "rawvideo", "pipe:1"]
    if strips:
        graph = (f"[0:v]{crop},split=2[a][b];[a]fps={SCAN_FPS},scale={SCAN_W}:{SCAN_H},format=gray[g];"
                 f"[b]fps={SCAN_FPS / STRIP_EVERY},scale=600:-2[j]")
        outputs += ["-map", "[j]", "-q:v", "4", str(tmp / "s%06d.jpg")]
    proc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", str(video),
                             "-filter_complex", graph, *outputs], stdout=subprocess.PIPE)

    packed, row_activity = [], np.zeros(SCAN_H)
    size = SCAN_W * SCAN_H
    expected = max(1, video_duration(video) * SCAN_FPS)
    while (buf := proc.stdout.read(size)) and len(buf) == size:
        g = np.frombuffer(buf, np.uint8).reshape(SCAN_H, SCAN_W)
        # Caption text = bright pixels right next to dark ones (the outline/shadow).
        dark = g < 90
        near_dark = dark.copy()
        for dy in (-2, -1, 1, 2):
            near_dark |= np.roll(dark, dy, axis=0)
        for dx in (-2, -1, 1, 2):
            near_dark |= np.roll(near_dark, dx, axis=1)
        mask = (g > 190) & near_dark
        row_activity += mask.sum(1)
        packed.append(np.packbits(mask))
        if len(packed) % 200 == 0:
            report("scan", 0.9 * len(packed) / expected, "Finding when each caption appears")
    proc.wait()

    # The caption band: the rows where text-like pixels show up most across the video.
    active = np.where(row_activity > 0.25 * row_activity.max())[0]
    r0, r1 = max(0, int(active.min()) - 3), min(SCAN_H, int(active.max()) + 4)
    band = [np.unpackbits(p)[:size].reshape(SCAN_H, SCAN_W)[r0:r1].astype(bool) for p in packed]
    n = len(band)
    px = np.array([m.sum() for m in band])
    raw = (px > 25).astype(int)
    has_text = raw.astype(bool)
    has_text[1:-1] = (raw[:-2] + raw[1:-1] + raw[2:]) >= 2  # ignore single-frame flicker
    change = np.zeros(n)
    k = 3
    for t in range(k, n - k):
        before = np.logical_and.reduce(band[t - k : t])  # text stable over the last 0.3 s
        after = np.logical_and.reduce(band[t : t + k])
        change[t] = (before ^ after).sum() / max(1, before.sum(), after.sum())
    mid = (r1 - r0) // 2  # captions are bottom-aligned: 1-line ones use the lower half
    cols = np.array([[m[:mid].any(0), m[mid:].any(0)] for m in band])

    # Cut the crops down to the caption band (with a margin): small, cheap, legible images.
    jpgs = sorted(tmp.glob("s*.jpg"))
    images: list[str] = []
    if jpgs:
        h = int(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=height",
                                "-of", "csv=p=0", str(jpgs[0])], capture_output=True,
                               text=True).stdout.strip())
        margin = round(h * 0.08)  # keep the tops/bottoms of letters inside the crop
        y0 = max(0, int(h * r0 / SCAN_H) - margin)
        y1 = min(h, int(h * r1 / SCAN_H) + margin)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(tmp / "s%06d.jpg"),
                        "-vf", f"crop=iw:{y1 - y0}:0:{y0}", "-q:v", "4", str(tmp / "b%06d.jpg")],
                       check=True)
        images = [base64.b64encode(p.read_bytes()).decode() for p in sorted(tmp.glob("b*.jpg"))]
    shutil.rmtree(tmp, ignore_errors=True)
    return CaptionScan(has_text, change, (r0, r1), images, video_size(video)[1], cols,
                       [np.packbits(m) for m in band])


def caption_segments(scan: CaptionScan) -> list[tuple[int, int, int]]:
    """(first frame, end frame, line count) of each caption, from the pixel scan alone."""
    import numpy as np

    n, has = len(scan.has_text), scan.has_text
    bounds = {0, n} | set((np.flatnonzero(has[1:] != has[:-1]) + 1).tolist())
    for t in range(3, n - 3):  # the caption was replaced by another
        if has[t] and scan.change[t] > 0.5 and scan.change[t] == scan.change[t - 3 : t + 4].max():
            bounds.add(t)
    b = sorted(bounds)
    segs = [[a, z] for a, z in zip(b, b[1:]) if z - a >= 3 and has[a:z].mean() > 0.5]

    def text_of(a: int, z: int):
        """Pixels that are text in at least 80% of the frames: the caption, not the background."""
        return np.mean([scan.mask(t) for t in range(a, z)], axis=0) >= 0.8

    # Rejoin pieces of one caption, split by a dropout, flicker or background change: adjacent
    # pieces whose persistent text is essentially the same.
    merged: list[list[int]] = []
    for a, z in segs:
        if merged and a - merged[-1][1] <= 5:
            left, right = text_of(*merged[-1]), text_of(a, z)
            if (left ^ right).sum() / max(1, left.sum(), right.sum()) < 0.4:
                merged[-1][1] = z
                continue
        merged.append([a, z])
    # Close tiny gaps so the Hebrew never flashes through between two English subtitles.
    for j in range(len(merged) - 1):
        if merged[j + 1][0] - merged[j][1] < 5:
            merged[j][1] = merged[j + 1][0]
    out = []
    for a, z in merged:
        upper = (scan.cols[a:z, 0].mean(0) >= 0.5).sum()  # stable text in the upper line
        out.append((a, z, 2 if upper > 8 else 1))
    return out


def caption_extents(scan: CaptionScan, segs: list[tuple[int, int, int]]) -> list[list[list[float]]]:
    """Horizontal extent [left, right] (fractions of the video width) of each Hebrew caption
    line, top to bottom, so each English box can be made wide enough to cover it."""
    import numpy as np

    x0 = (1 - CAPTION_WIDTH) / 2
    out = []
    for a, z, n_lines in segs:
        extents = []
        for h in ([1] if n_lines < 2 else [0, 1]):
            # Columns with text in at least half the caption's frames (ignores moving background),
            # then the contiguous run of words around the centre.
            stable = np.flatnonzero(scan.cols[a:z, h].mean(0) >= 0.5)
            if len(stable) == 0:
                extents.append([])
                continue
            i = int(np.argmin(np.abs(stable - SCAN_W // 2)))
            lo = hi = i
            while lo > 0 and stable[lo] - stable[lo - 1] <= 25:
                lo -= 1
            while hi < len(stable) - 1 and stable[hi + 1] - stable[hi] <= 25:
                hi += 1
            extents.append([round(x0 + CAPTION_WIDTH * stable[lo] / SCAN_W, 4),
                            round(x0 + CAPTION_WIDTH * (stable[hi] + 1) / SCAN_W, 4)])
        out.append(extents)
    return out


@dataclass
class Unit:
    """One English subtitle to produce: timed like a Hebrew caption (or like speech where the
    video has no caption), with the sources Claude translates from."""
    start: float
    end: float
    lines: int          # line count of the on-screen Hebrew caption (0 = no caption: speech only)
    speech: str         # speech transcript heard during it
    images: list = field(default_factory=list)  # base64 JPEG of each caption it covers
    extent: list = field(default_factory=list)  # [left, right] per caption line, width fractions


def build_units(scan: CaptionScan, speech: list[Segment]) -> list[Unit]:
    segs = caption_segments(scan)
    extents = caption_extents(scan, segs)

    def overlap(a0, a1, b0, b1):
        return max(0.0, min(a1, b1) - max(a0, b0))

    units = []
    for (a, z, n_lines), extent in zip(segs, extents):
        start, end = a / SCAN_FPS, z / SCAN_FPS
        heard = [s.text for s in speech if overlap(start, end, s.start, s.end) > 0.25 * (s.end - s.start)
                 or overlap(start, end, s.start, s.end) > 0.5 * (end - start)]
        image = scan.strips[min(len(scan.strips) - 1, round((a + z) / 2 / STRIP_EVERY))] if scan.strips else ""
        units.append(Unit(start, end, n_lines, " ".join(heard), [image] if image else [], extent))
    # Speech the captions don't cover (e.g. uncaptioned background talk) keeps speech timing.
    for s in speech:
        covered = sum(overlap(s.start, s.end, a / SCAN_FPS, z / SCAN_FPS) for a, z, _ in segs)
        if covered < 0.3 * (s.end - s.start):
            units.append(Unit(s.start, s.end, 0, s.text))
    units.sort(key=lambda u: u.start)
    return units


MIN_SECONDS = 1.2  # shortest time a subtitle stays on screen


def group_for_reading(units: list[Unit]) -> list[Unit]:
    """Join captions that follow each other too quickly to read into one English subtitle
    (at most 2 lines, 6 seconds), the way professional subtitlers handle fast dialogue."""
    out: list[Unit] = []
    for u in units:
        prev = out[-1] if out else None
        if (prev and prev.lines and u.lines and u.start - prev.end < 0.3
                and prev.lines + u.lines <= 2 and u.end - prev.start <= 6
                and (prev.end - prev.start < MIN_SECONDS or u.end - u.start < MIN_SECONDS)):
            out[-1] = Unit(prev.start, u.end, prev.lines + u.lines,
                           " ".join(x for x in (prev.speech, u.speech) if x),
                           prev.images + u.images, prev.extent + u.extent)
        else:
            out.append(u)
    return out


UNITS_PROMPT = """

The video has Hebrew captions burned into the picture by the broadcaster, and your English \
subtitles will be drawn over them with exactly the same timing. Each item comes with an image \
of the on-screen Hebrew caption at that moment, its line count, and the speech transcript \
heard during it.
- Read the Hebrew caption in the image: it is professionally edited and the main source for \
meaning, wording and nuance. The speech transcript may contain recognition errors; use it only \
for context or what the caption leaves out. Captions often split a sentence across items: \
translate each item so it reads naturally in sequence.
- Return `en` with EXACTLY the given number of lines, each at most 42 characters, and within \
the item's character limit (set by how long it stays on screen, so viewers can read it). \
Condense like a professional subtitler: keep the meaning and tone, drop filler and repetition.
- Some items show two consecutive captions (two images): translate them together as one \
subtitle.
- Every item whose image shows a Hebrew caption MUST get its own translation of that caption: \
never leave it empty or fold it into a neighbouring item. If consecutive items show the same \
caption, return the same translation for each.
- Only if an image shows no dialogue caption at all (just scenery, credits or a logo), return \
an empty list.
- Items without an image are speech the broadcaster didn't caption: translate the speech in \
at most 2 lines."""


FALLBACK_MODEL = "claude-sonnet-4-6"
GEMINI_MODEL = "gemini-3.8-flash"


def ask_claude(model: str, system: str, content: list[dict], schema, ids: list[int]) -> dict[int, list[str]]:
    import anthropic

    client = anthropic.Anthropic()
    got: dict[int, list[str]] = {}
    for _attempt in range(3):
        r = client.messages.parse(model=model, max_tokens=8000, system=system,
                                  messages=[{"role": "user", "content": content}],
                                  output_format=schema, **no_thinking(model))
        track_usage(model, r.usage)
        if r.stop_reason == "refusal":
            sys.exit("Claude declined to translate this content.")
        if r.parsed_output:
            got = {x.id: x.en for x in r.parsed_output.t}
            if all(i in got for i in ids):
                break
        print("   ↻ incomplete batch, retrying")
    return got


def ask_gemini(model: str, system: str, parts: list[tuple[str, str]], schema) -> dict[int, list[str]]:
    """Gemini with low image resolution: each caption image costs ~240 tokens instead of ~1,000,
    with no measurable loss in translation quality in our tests."""
    import base64
    import time

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=types.HttpOptions(
        timeout=180_000, retry_options=types.HttpRetryOptions(attempts=1)))
    contents = [types.Part.from_text(text=t) if kind == "text" else
                types.Part.from_bytes(data=base64.b64decode(t), mime_type="image/jpeg")
                for kind, t in parts]
    config = types.GenerateContentConfig(
        system_instruction=system, response_mime_type="application/json", response_schema=schema,
        thinking_config=types.ThinkingConfig(thinking_level="low"),
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
    for attempt in range(3):
        try:
            r = client.models.generate_content(model=model, contents=contents, config=config)
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(15 * (attempt + 1))  # usually "overloaded": wait and retry
    um = r.usage_metadata
    totals = USAGE.setdefault(model, [0, 0])
    totals[0] += um.prompt_token_count or 0
    totals[1] += (um.candidates_token_count or 0) + (um.thoughts_token_count or 0)
    if not r.parsed:
        raise ValueError("no parsable output")
    return {x.id: [l for l in x.en if l.strip()] for x in r.parsed.t}


def translate_units(units: list[Unit], model: str, context: str = "",
                    reading_speed: float = 17) -> tuple[list[Segment], list[Unit]]:
    """Translate caption units. Returns the English segments (units' timing, line breaks) and
    the units they belong to; units Claude found to have no caption are dropped from both."""
    import concurrent.futures

    from pydantic import BaseModel

    class Item(BaseModel):
        id: int
        en: list[str]

    class Batch(BaseModel):
        t: list[Item]

    system = SYSTEM_PROMPT + UNITS_PROMPT
    if feedback := load_feedback():
        system += "\n\n" + feedback
    starts = list(range(0, len(units), BATCH_SIZE))

    def fit(lines: list[str], n: int) -> list[str]:
        """Rewrap to at most n lines if the model returned more."""
        lines = [l.strip() for l in lines if l.strip()]
        if len(lines) <= n:
            return lines
        words = " ".join(lines).split()
        per = -(-len(" ".join(words)) // n)
        out, cur = [], ""
        for w in words:
            if cur and len(cur) + 1 + len(w) > per and len(out) < n - 1:
                out.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        return out + [cur]

    finished: list[int] = []

    def translate_batch(start: int) -> list[tuple[Unit, Segment | None]]:
        batch = units[start : start + BATCH_SIZE]
        intro = f"Video context: {context}\n\n" if context else ""
        before = [u.speech for u in units[max(0, start - 6) : start] if u.speech]
        if before:
            intro += "Speech just before these items (context only):\n" + "\n".join(before) + "\n\n"
        item_parts: list[list[tuple[str, str]]] = []
        for i, u in enumerate(batch):
            seconds = u.end - u.start
            budget = max(14, round(seconds * reading_speed))
            label = (f"Item {i} ({u.lines or 'at most 2'} line{'s' * (u.lines != 1)}, on screen "
                     f"{seconds:.1f}s, at most {budget} characters)")
            if u.images:
                item_parts.append([("text", f"{label} - speech: {u.speech or '-'}")]
                                  + [("image", img) for img in u.images])
            else:
                item_parts.append([("text", f"{label}, no caption - speech: {u.speech}")])

        def parts_for(ids: list[int]) -> list[tuple[str, str]]:
            return [("text", intro + "Translate these items:")] + [p for i in ids for p in item_parts[i]]

        got: dict[int, list[str]] = {}
        if model.startswith("gemini"):
            try:
                got = ask_gemini(model, system, parts_for(list(range(len(batch)))), Batch)
            except Exception as e:  # overloaded, timed out, bad output: Claude takes the batch
                print(f"   ↻ {model} failed ({str(e)[:60]}); using {FALLBACK_MODEL}")
        # Claude handles the whole batch, or only the items Gemini didn't return at all (an empty
        # answer is deliberate: no caption in that image). It is sent only those items.
        missing = [i for i in range(len(batch)) if i not in got]
        if missing:
            claude_model = model if model.startswith("claude") else FALLBACK_MODEL
            content = [{"type": "text", "text": t} if kind == "text" else image_block(t)
                       for kind, t in parts_for(missing)]
            got.update(ask_claude(claude_model, system, content, Batch, missing))
        print(f"   ✓ translated {start + 1}-{start + len(batch)}")
        finished.append(start)
        report("translate", len(finished) / len(starts), f"Translating ({len(finished)}/{len(starts)} batches)")
        out = []
        for i, u in enumerate(batch):
            lines = fit(got.get(i, ["[untranslated] " + u.speech]), u.lines or 2)
            out.append((u, Segment(u.start, u.end, "\n".join(lines)) if lines else None))
        return out

    print(f"🌐 Translating {len(units)} captions with {model} ({len(starts)} batches in parallel)")
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        pairs = [p for part in pool.map(translate_batch, starts) for p in part if p[1]]
    # Neighbouring pieces of one caption come back with the same English: show it once.
    joined: list[tuple[Unit, Segment]] = []
    for u, seg in pairs:
        if (joined and u.lines and seg.text == joined[-1][1].text
                and u.start - joined[-1][1].end < 0.5):
            prev_u, prev = joined[-1]
            prev_u.end = u.end
            joined[-1] = (prev_u, Segment(prev.start, u.end, prev.text))
        else:
            joined.append((u, seg))
    # Give each subtitle enough time to be read, using the gap before the next one.
    for k, (u, seg) in enumerate(joined):
        need = max(MIN_SECONDS, len(seg.text.replace("\n", " ")) / reading_speed)
        limit = joined[k + 1][1].start if k + 1 < len(joined) else seg.end + need
        end = min(limit, max(seg.end, seg.start + need))
        if end > seg.end:
            u.end = end
            joined[k] = (u, Segment(seg.start, end, seg.text))
    return [seg for _, seg in joined], [u for u, _ in joined]


def save_captions(work_dir: Path, units: list[Unit], caption_bottom: float) -> None:
    """captions.json (timing, line counts, widths, speech) plus the caption images, used by
    re-renders and the review page."""
    import base64

    images = work_dir / "caption_images"
    shutil.rmtree(images, ignore_errors=True)
    images.mkdir()
    rows = []
    for k, u in enumerate(units):
        names = []
        for j, img in enumerate(u.images):
            names.append(f"{k:04d}-{j}.jpg")
            (images / names[-1]).write_bytes(base64.b64decode(img))
        rows.append({"start": u.start, "end": u.end, "lines": u.lines, "speech": u.speech,
                     "extent": u.extent, "images": names})
    (work_dir / "captions.json").write_text(json.dumps(
        {"caption_bottom": caption_bottom, "units": rows}, ensure_ascii=False, indent=1))


def translate_with_claude(segments: list[Segment], model: str, context: str = "") -> list[Segment]:
    """Translate Hebrew speech segments (used when the video has no burned-in captions)."""
    import anthropic
    from pydantic import BaseModel

    class Line(BaseModel):
        id: int
        english: str

    class Batch(BaseModel):
        lines: list[Line]

    import concurrent.futures

    client = anthropic.Anthropic()
    starts = list(range(0, len(segments), BATCH_SIZE))

    system = SYSTEM_PROMPT
    if feedback := load_feedback():
        system += "\n\n" + feedback

    def translate_batch(start: int) -> list[Segment]:
        batch = segments[start : start + BATCH_SIZE]
        items = [{"id": i, "hebrew": s.text} for i, s in enumerate(batch)]
        prompt = ""
        if context:
            prompt += f"Video context: {context}\n\n"
        before = segments[max(0, start - 10) : start]
        if before:
            prompt += "Preceding lines (context only, do not translate):\n"
            prompt += "\n".join(s.text for s in before) + "\n\n"
        prompt += "Translate these segments:\n" + json.dumps(items, ensure_ascii=False, indent=1)
        content = [{"type": "text", "text": prompt}]

        translations: dict[int, str] = {}
        for _attempt in range(3):
            response = client.messages.parse(
                model=model,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": content}],
                output_format=Batch,
            )
            track_usage(model, response.usage)
            if response.stop_reason == "refusal":
                sys.exit("Claude declined to translate this content.")
            if response.parsed_output:
                translations = {l.id: l.english for l in response.parsed_output.lines}
                if all(i in translations for i in range(len(batch))):
                    break
            print("   ↻ incomplete batch, retrying")
        print(f"   ✓ translated segments {start + 1}-{start + len(batch)}")
        return [Segment(seg.start, seg.end, translations.get(i, f"[untranslated] {seg.text}"))
                for i, seg in enumerate(batch)]

    print(f"🌐 Translating {len(segments)} segments with {model} ({len(starts)} batches in parallel)")
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(translate_batch, starts))
    return [s for batch in results for s in batch]


def claude_available() -> bool:
    # The SDK doesn't validate credentials at construction, so check the sources it reads.
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return (Path.home() / ".config" / "anthropic").is_dir()  # `ant auth login` profile


# ------------------------------------------------------------------------ output


def video_size(video: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(video)], capture_output=True, text=True, check=True).stdout
    w, h = out.strip().split(",")[:2]
    return int(w), int(h)


# Arial Bold on macOS; Liberation Sans Bold (metric-compatible with Arial) on Linux/Docker.
FONT_FILE = next((f for f in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf")
                  if Path(f).exists()), "")


BOX_COLORS = {  # box colour -> (box, text) in ASS BGR hex
    "white": ("FFFFFF", "1A1A1A"),
    "black": ("000000", "FFFFFF"),
}


def write_ass(segments: list[Segment], path: Path, width: int, height: int,
              bottom: float | None, opacity: float, extents: list | None = None,
              box_color: str = "white", captioned: list | None = None) -> None:
    """Write burn-in subtitles in the standard broadcast/streaming style: one fixed font size,
    each line on its own semi-transparent box with a little padding. `bottom` is where the
    lowest line's box ends, as a fraction of the height (the Hebrew caption area's bottom, so
    the English sits over it); None uses the usual position near the bottom of the frame.
    `extents` (per segment, [left, right] width fractions per Hebrew line) widens a line's box
    just enough to cover the Hebrew line under it."""
    font = round(height * 0.048)
    pad_x, pad_y = round(font * 0.35), round(font * 0.12)  # box padding around the text
    box_h = round(font * 1.15) + 2 * pad_y  # one line's box; lines stack without gaps
    bottom_px = round(height * (bottom if bottom is not None else 0.93)) + pad_y
    max_w = round(width * 0.9)
    alpha = f"{round((1 - opacity) * 255):02X}"  # ASS alpha: 00 = opaque, FF = transparent
    box_rgb, text_rgb = BOX_COLORS[box_color]
    try:
        from PIL import ImageFont

        measure = ImageFont.truetype(FONT_FILE, font).getlength
    except (ImportError, OSError):
        measure = lambda text: len(text) * font * 0.55  # rough Arial Bold average

    def ts(t: float) -> str:
        cs = int(round(t * 100))
        h, cs = divmod(cs, 360000)
        m, cs = divmod(cs, 6000)
        sec, cs = divmod(cs, 100)
        return f"{h}:{m:02}:{sec:02}.{cs:02}"

    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {width}", f"PlayResY: {height}",
        "WrapStyle: 2", "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Text,Arial,{font},&H00{text_rgb},&H00{text_rgb},&H00000000,&H00000000,-1,0,0,0,"
        f"100,100,0,0,1,0,0,5,0,0,0,1",
        f"Style: Box,Arial,{font},&H{alpha}{box_rgb},&H{alpha}{box_rgb},&H00000000,&H00000000,0,0,0,0,"
        f"100,100,0,0,1,0,0,7,0,0,0,1",
        "", "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    def rows_of(seg: Segment) -> list[str]:
        return [r.strip() for r in seg.text.replace("{", "(").replace("}", ")").split("\n") if r.strip()]

    def box(start: str, end: str, half: float, n_rows: int, j0: int = 0) -> None:
        w = min(max_w, round(2 * half))
        h = box_h * n_rows
        x, y = (width - w) // 2, bottom_px - j0 * box_h - h
        lines.append(f"Dialogue: 0,{start},{end},Box,,0,0,0,,"
                     f"{{\\pos({x},{y})\\p1}}m 0 0 l {w} 0 {w} {h} 0 {h}{{\\p0}}")

    def text(start: str, end: str, rows: list[str]) -> None:
        for j, row in enumerate(reversed(rows)):  # bottom line first
            y = bottom_px - (j + 1) * box_h
            lines.append(f"Dialogue: 1,{start},{end},Text,,0,0,0,,"
                         f"{{\\pos({width // 2},{y + box_h // 2})}}{row}")

    def hebrew_half(k: int) -> float:
        """Half-width (from the centre) needed to cover segment k's Hebrew caption lines."""
        return max((max(width / 2 - e[0] * width, e[1] * width - width / 2) + font * 0.15
                    for e in (extents[k] if extents and k < len(extents) else []) if e), default=0)

    # Over the burned-in Hebrew captions: one steady band per stretch of dialogue, bridging short
    # gaps, sized once for everything shown in that stretch, so the Hebrew never flashes through.
    flags = captioned if captioned and len(captioned) == len(segments) else [False] * len(segments)
    covered: set[int] = set()
    k = 0
    while k < len(segments):
        if not flags[k]:
            k += 1
            continue
        members = [k]
        while (members[-1] + 1 < len(segments) and flags[members[-1] + 1]
               and segments[members[-1] + 1].start - segments[members[-1]].end < 1.5):
            members.append(members[-1] + 1)
        half = max(max((measure(r) / 2 + pad_x for m in members for r in rows_of(segments[m])), default=0),
                   max(hebrew_half(m) for m in members))
        n_rows = min(2, max(max(len(rows_of(segments[m])), len(extents[m]) if extents else 1)
                            for m in members))
        box(ts(segments[members[0]].start), ts(segments[members[-1]].end), half, n_rows)
        for m in members:
            text(ts(segments[m].start), ts(segments[m].end), rows_of(segments[m]))
        covered.update(members)
        k = members[-1] + 1

    # Elsewhere (speech without a Hebrew caption): a box per line, sized to its text.
    for k, seg in enumerate(segments):
        if k in covered:
            continue
        start, end = ts(seg.start), ts(seg.end)
        rows = rows_of(seg)
        for j, row in enumerate(reversed(rows)):
            box(start, end, measure(row) / 2 + pad_x, 1, j)
        text(start, end, rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ffmpeg(cmd: list[str], video: Path) -> bool:
    """Run an ffmpeg render, reporting progress from its -progress output."""
    total = video_duration(video) or 1
    cmd = [c for c in cmd if c != "-stats"]
    proc = subprocess.Popen(cmd[:1] + ["-progress", "pipe:1", "-nostats"] + cmd[1:],
                            stdout=subprocess.PIPE, text=True)
    for line in proc.stdout:
        if line.startswith("out_time_us=") and line.strip()[12:].isdigit():
            report("render", int(line.strip()[12:]) / 1e6 / total, "Burning in subtitles")
    return proc.wait() == 0


def add_subtitles(video: Path, en_srt: Path, soft: bool, bottom: float | None,
                  opacity: float, box_color: str = "white") -> Path:
    """Burn in (or add as a track) the English subtitles. Uses the Hebrew caption widths saved
    in captions.json, when present, to size each line's box."""
    out = video.with_name(f"{video.stem}.en.mp4")
    if soft:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-i", str(en_srt),
               "-map", "0", "-map", "1", "-c", "copy", "-c:s", "mov_text",
               "-metadata:s:s:0", "language=eng", "-metadata:s:s:0", "title=English",
               "-disposition:s:0", "default", str(out)]
    else:
        ass = en_srt.with_suffix(".ass")
        segments = read_srt(en_srt)
        captions = en_srt.parent / "captions.json"
        extents = captioned = None
        if captions.exists():
            units = json.loads(captions.read_text())["units"]
            if len(units) == len(segments):
                extents = [u.get("extent", []) for u in units]
                captioned = [bool(u.get("lines", len(u.get("caption", [])))) for u in units]
        write_ass(segments, ass, *video_size(video), bottom, opacity, extents, box_color, captioned)
        # the subtitles filter needs ':', '\\' and quotes in the path escaped
        esc = str(ass).replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")
        # The hardware encoder is less efficient than YouTube's, so give it ~2.5x the source
        # bitrate: visually identical to the original without a needlessly huge file.
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=bit_rate",
                                "-of", "csv=p=0", str(video)], capture_output=True, text=True)
        src_kbps = int(probe.stdout.strip() or 4_000_000) // 1000
        bitrate = f"{max(4000, round(src_kbps * 2.5))}k"
        base = ["ffmpeg", "-y", "-loglevel", "error", "-stats", "-i", str(video),
                "-vf", f"subtitles='{esc}'", "-c:a", "copy", "-movflags", "+faststart"]
        # Hardware encoder (fast on Apple Silicon) at a high bitrate to keep source quality.
        hw = base + ["-c:v", "h264_videotoolbox", "-b:v", bitrate, "-profile:v", "high", str(out)]
        sw = base + ["-c:v", "libx264", "-preset", "fast", "-crf", "17", str(out)]
        print("🎬 Burning in subtitles…")
        if not run_ffmpeg(hw, video):
            if not run_ffmpeg(sw, video):
                raise RuntimeError("ffmpeg failed to render the video")
        return out
    print("🎬 Adding subtitles to video…")
    subprocess.run(cmd, check=True)
    return out


# -------------------------------------------------------------------------- main


# ------------------------------------------------------------------ runs & estimates


@dataclass
class Options:
    translator: str = "auto"          # auto | claude | whisper
    claude_model: str = DEFAULT_CLAUDE_MODEL
    translate_model: str | None = None  # default: Gemini when GEMINI_API_KEY is set
    whisper_model: str = DEFAULT_WHISPER_MODEL
    local: bool = False               # transcribe on this machine even if RunPod is configured
    force_transcribe: bool = False
    captions: str = "auto"            # auto | on | off
    reading_speed: float = 17
    context: str = ""
    soft: bool = False
    position: float | None = None
    box_color: str = "white"
    box_opacity: float = 1.0
    no_video: bool = False


# Per-item token use and speeds measured on real runs (24-minute episodes), used to estimate.
ITEMS_PER_MINUTE = 30          # English subtitles per minute of video after joining fast captions
ITEM_TOKENS = {"gemini": (320, 34), "claude": (140, 20)}
DETECT_TOKENS = (2000, 20)     # caption check: 9 small images
RUNPOD_PER_MINUTE = 0.001      # $ per minute of audio (estimated)


def translate_model_for(opts: Options) -> str:
    return opts.translate_model or (GEMINI_MODEL if os.environ.get("GEMINI_API_KEY") else opts.claude_model)


@functools.lru_cache(maxsize=64)
def video_info(url: str) -> dict:
    import yt_dlp

    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "noplaylist": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    return {"id": info["id"], "title": info.get("title") or "video", "url": url,
            "duration": info.get("duration") or 0, "thumbnail": info.get("thumbnail") or "",
            "channel": info.get("channel") or info.get("uploader") or ""}


def estimate(url: str, opts: Options | None = None) -> dict:
    """Expected tokens, cost and time for a run, from the video's length (no download)."""
    opts = opts or Options()
    info = video_info(url)
    minutes = info["duration"] / 60
    model = translate_model_for(opts)
    items = round(minutes * ITEMS_PER_MINUTE)
    per_in, per_out = ITEM_TOKENS["gemini" if model.startswith("gemini") else "claude"]
    tokens = {model: [items * per_in, items * per_out]}
    detect = tokens.setdefault(opts.claude_model, [0, 0])
    detect[0] += DETECT_TOKENS[0]
    detect[1] += DETECT_TOKENS[1]
    ai_cost = sum(t_in / 1e6 * PRICES.get(m, (0, 0))[0] + t_out / 1e6 * PRICES.get(m, (0, 0))[1]
                  for m, (t_in, t_out) in tokens.items())
    gpu = runpod_configured() and not opts.local
    gpu_cost = minutes * RUNPOD_PER_MINUTE if gpu else 0
    render_speed = 8 if sys.platform == "darwin" else 4  # × real time
    stage_minutes = {"download": 0.5 + minutes * 0.02,
                     "transcribe": (1 + minutes * 0.08) if gpu else minutes,
                     "scan": minutes * 0.1, "translate": 1 + minutes * 0.05,
                     "render": 0 if opts.no_video else minutes / render_speed}
    return {**info, "translate_model": model, "items": items, "tokens": tokens,
            "ai_cost": round(ai_cost, 3), "gpu_cost": round(gpu_cost, 3),
            "cost": round(ai_cost + gpu_cost, 3), "gpu": gpu,
            "minutes": round(sum(stage_minutes.values()), 1)}


def usage_cost() -> tuple[dict, float]:
    usage = {m: list(v) for m, v in USAGE.items()}
    cost = sum(t_in / 1e6 * PRICES.get(m, (0, 0))[0] + t_out / 1e6 * PRICES.get(m, (0, 0))[1]
               for m, (t_in, t_out) in usage.items())
    return usage, round(cost, 4)


def write_record(work_dir: Path, record: dict) -> None:
    tmp = work_dir / "run.json.tmp"
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1))
    tmp.replace(work_dir / "run.json")


def run(url: str, opts: Options | None = None, est: dict | None = None) -> dict:
    """Download, transcribe, translate and subtitle one video. Returns the run record, which is
    also saved as run.json in the video's folder (the library reads those)."""
    import time
    from dataclasses import asdict
    from datetime import datetime, timezone

    opts = opts or Options()
    USAGE.clear()
    started = time.time()
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required (brew install ffmpeg).")

    translator = opts.translator
    if translator == "auto":
        translator = "claude" if claude_available() else "whisper"
        if translator == "whisper":
            print("ℹ  No Anthropic credentials found — using Whisper's built-in translation.")
    elif translator == "claude" and not claude_available():
        raise RuntimeError("Claude translation needs an Anthropic API key (Settings).")
    if translator == "claude":
        check_claude_key()
    use_runpod = runpod_configured() and not opts.local

    video, he_srt, info = download(url, DOWNLOADS, want_subs=not opts.force_transcribe)
    work_dir = video.parent
    record = {"id": info["id"], "url": url, "title": info.get("title"), "folder": work_dir.name,
              "duration": info.get("duration"), "thumbnail": info.get("thumbnail"),
              "channel": info.get("channel") or info.get("uploader"),
              "started": datetime.now(timezone.utc).isoformat(), "status": "running",
              "options": asdict(opts), "estimate": est}
    write_record(work_dir, record)
    en_srt = work_dir / "subtitles.en.srt"
    try:
        caption_bottom = None  # set when English follows burned-in Hebrew captions
        if translator == "claude":
            # Check the picture for burned-in captions first (seconds, 9 small images): it decides
            # the rest of the run, and transcription can then overlap with the caption scan.
            report("detect", 0.3, "Checking for Hebrew captions")
            use_captions = opts.captions == "on" or (
                opts.captions == "auto" and has_burned_captions(video, opts.claude_model))
            report("detect", 1, "Captions found" if use_captions else "No captions: translating speech")
            if opts.captions == "auto":
                print("✓ Burned-in Hebrew captions detected — using them for the translation"
                      if use_captions else "ℹ  No burned-in Hebrew captions — translating from speech")

            def transcribe() -> list[Segment]:
                if he_srt:  # Hebrew subtitles uploaded to YouTube: no transcription needed
                    return read_srt(he_srt)
                if use_runpod:
                    return run_runpod(video, opts.whisper_model)
                return run_whisper(extract_audio(video), opts.whisper_model, "transcribe")

            scan = None
            if use_captions and use_runpod and not he_srt:
                # GPU transcription (remote) and the caption scan (local CPU) run side by side.
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    speech = pool.submit(transcribe)
                    scan = scan_captions(video)
                    he_segments = speech.result()
            else:
                he_segments = transcribe()
            if not he_srt:
                he_srt = work_dir / "subtitles.he.srt"
                write_srt(he_segments, he_srt)
            report("transcribe", 1, "Transcribed")
            if not he_segments:
                raise RuntimeError("No speech found in the video.")
            print(f"✓ Hebrew subtitles: {he_srt}")
            record["captions"] = use_captions
            if use_captions:
                scan = scan or scan_captions(video)
                report("scan", 0.95, "Grouping captions for reading speed")
                units = group_for_reading(build_units(scan, he_segments))
                caption_bottom = scan.bottom()
                captioned = sum(1 for u in units if u.lines)
                print(f"✓ {captioned} Hebrew captions timed; {len(units) - captioned} uncaptioned speech lines")
                report("scan", 1, f"{captioned} captions timed")
                model = translate_model_for(opts)
                record["translate_model"] = model
                en_segments, units = translate_units(units, model, opts.context, opts.reading_speed)
                save_captions(work_dir, units, caption_bottom)
            else:
                record["translate_model"] = opts.claude_model
                en_segments = translate_with_claude(he_segments, opts.claude_model, opts.context)
        else:
            # The RunPod worker only runs the Hebrew models, which can't translate.
            en_segments = run_whisper(extract_audio(video), WHISPER_TRANSLATE_MODEL, "translate")
            if not en_segments:
                raise RuntimeError("No speech found in the video.")
        report("translate", 1, "Translated")
        write_srt(en_segments, en_srt)
        print(f"✓ English subtitles: {en_srt}")
        record["subtitles"] = len(en_segments)

        if not opts.no_video:
            out = add_subtitles(video, en_srt, opts.soft,
                                opts.position if opts.position is not None else caption_bottom,
                                opts.box_opacity, opts.box_color)
            report("render", 1, "Done")
            print(f"✓ Video with English subtitles: {out}")
        record["status"] = "done"
    except BaseException as e:
        record["status"] = "failed"
        record["error"] = str(e) or type(e).__name__
        raise
    finally:
        (work_dir / "audio.wav").unlink(missing_ok=True)
        record["usage"], ai_cost = usage_cost()
        gpu_cost = (record.get("duration") or 0) / 60 * RUNPOD_PER_MINUTE if use_runpod else 0
        record["cost"] = {"ai": ai_cost, "gpu_estimated": round(gpu_cost, 3), "total": round(ai_cost + gpu_cost, 4)}
        record["finished"] = datetime.now(timezone.utc).isoformat()
        record["seconds"] = round(time.time() - started)
        write_record(work_dir, record)
        print_cost()
    return record


def cli_options(argv: list[str]) -> tuple[str, Options]:
    p = argparse.ArgumentParser(prog="tirgum",
        description="Download a YouTube video and create English subtitles from Hebrew. "
                    "Run 'tirgum serve' for the web app.")
    p.add_argument("url", help="YouTube video URL")
    p.add_argument("--translator", choices=["auto", "claude", "whisper"], default="auto",
                   help="claude = best quality (needs ANTHROPIC_API_KEY); whisper = offline, "
                        "lower quality; auto = claude if credentials exist (default)")
    p.add_argument("--claude-model", default=DEFAULT_CLAUDE_MODEL,
                   help="Claude model for caption detection and as the translation fallback")
    p.add_argument("--translate-model", default=None,
                   help=f"model that translates the captions (default: {GEMINI_MODEL} when "
                        f"GEMINI_API_KEY is set, else the Claude model)")
    p.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL,
                   help=f"model for Hebrew transcription (default: {DEFAULT_WHISPER_MODEL})")
    p.add_argument("--local", action="store_true",
                   help="transcribe on this machine even if RunPod is configured")
    p.add_argument("--force-transcribe", action="store_true",
                   help="ignore YouTube's Hebrew subtitles and transcribe the audio")
    p.add_argument("--captions", choices=["auto", "on", "off"], default="auto",
                   help="use Hebrew captions burned into the video as the main translation "
                        "source (auto = detect them; default)")
    p.add_argument("--reading-speed", type=float, default=17,
                   help="max characters per second viewers have to read (default 17); fast "
                        "captions are joined and translations condensed to stay under it")
    p.add_argument("--context", default="",
                   help="optional hint for the translator, e.g. 'cooking show, casual tone'")
    p.add_argument("--soft", action="store_true",
                   help="add a toggleable subtitle track instead of burning subtitles in")
    p.add_argument("--position", type=float, default=None,
                   help="bottom edge of the subtitles as a fraction of the height (default: over "
                        "the burned-in Hebrew captions if found, else 0.93)")
    p.add_argument("--box-color", choices=sorted(BOX_COLORS), default="white",
                   help="box behind the subtitles: white with dark text (default) or black")
    p.add_argument("--box-opacity", type=float, default=1.0,
                   help="opacity of the box, 0-1 (default 1 = solid, hides the Hebrew underneath)")
    p.add_argument("--no-video", action="store_true",
                   help="only write .srt files, don't create a subtitled video")
    a = p.parse_args(argv)
    return a.url, Options(a.translator, a.claude_model, a.translate_model, a.whisper_model, a.local,
                          a.force_transcribe, a.captions, a.reading_speed, a.context, a.soft,
                          a.position, a.box_color, a.box_opacity, a.no_video)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    load_env()
    if "--setup-runpod" in argv:
        setup_runpod()
        return
    if "--review" in argv:
        rest = [a for a in argv if a not in ("--review", "--no-open")]
        if not rest:
            sys.exit("usage: tirgum --review downloads/<video folder>")
        serve_review(Path(rest[0]))
        return
    url, opts = cli_options(argv)
    try:
        run(url, opts)
    except RuntimeError as e:
        sys.exit(str(e))
