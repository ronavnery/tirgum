"""Tirgum web app: a local API over the pipeline plus a single-page UI (tirgum/static).

Runs on this Mac (`tirgum serve`) or in Docker on a server, where it's meant to sit behind a
reverse proxy that handles login (e.g. Authelia). Only without such a proxy, set TIRGUM_PASSWORD
to make the app ask for a password itself (HTTP basic auth, any username).
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from dataclasses import asdict, fields
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import pipeline as p

STATIC = Path(__file__).resolve().parent / "static"
KEYS = {  # settings field -> .env variable
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "runpod": "RUNPOD_API_KEY",
    "runpod_endpoint": "RUNPOD_ENDPOINT_ID",
}
SERVED_FILES = {"video.en.mp4", "video.mp4", "subtitles.en.srt", "subtitles.he.srt"}

app = FastAPI(title="Tirgum")


@app.middleware("http")
async def password(request: Request, call_next):
    expected = os.environ.get("TIRGUM_PASSWORD")
    if expected and not request.url.path.startswith("/s/"):
        header = request.headers.get("authorization", "")
        try:
            given = base64.b64decode(header.removeprefix("Basic ")).decode().split(":", 1)[1]
        except Exception:
            given = ""
        if not secrets.compare_digest(given, expected):
            return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="Tirgum"'})
    response = await call_next(request)
    if not request.url.path.startswith(("/api/", "/s/")):
        # The UI files change with updates: make browsers check for a new version every time.
        response.headers["Cache-Control"] = "no-cache"
    return response


# ------------------------------------------------------------------------- jobs
# One job runs at a time (the pipeline uses the GPU endpoint, the CPU and shared usage
# counters); others wait in the queue.

ORDER = list(p.STAGES)
jobs: dict[str, dict] = {}
queue: list[str] = []
lock = threading.Lock()
wake = threading.Event()


def options_from(data: dict | None) -> p.Options:
    names = {f.name for f in fields(p.Options)}
    return p.Options(**{k: v for k, v in (data or {}).items() if k in names and v not in (None, "")})


def on_progress(job: dict):
    def update(stage: str, fraction: float, message: str) -> None:
        idx = ORDER.index(stage) if stage in ORDER else 0
        done = sum(p.STAGES[s] for s in ORDER[:idx])  # earlier (or skipped) stages count as done
        overall = (done + p.STAGES.get(stage, 0) * fraction) / sum(p.STAGES.values())
        usage, cost = p.usage_cost()
        job.update(stage=stage, stage_fraction=fraction, message=message,
                   progress=max(job.get("progress", 0), overall), usage=usage, cost=cost)
    return update


def worker() -> None:
    while True:
        wake.wait()
        with lock:
            job_id = queue.pop(0) if queue else None
            if not queue:
                wake.clear()
        if not job_id:
            continue
        job = jobs[job_id]
        job.update(status="running", started=time.time(), progress=0.0, stage="download",
                   message="Starting")
        p.ON_PROGRESS = on_progress(job)
        try:
            p.load_env()
            record = p.run(job["url"], options_from(job["options"]), job.get("estimate"))
            job.update(status="done", progress=1.0, record=record, message="Done")
        except BaseException as e:  # SystemExit from the pipeline included
            job.update(status="failed", error=str(e) or type(e).__name__, message="Failed")
            traceback.print_exc()
        finally:
            p.ON_PROGRESS = None
            job["finished"] = time.time()


threading.Thread(target=worker, daemon=True).start()


@app.post("/api/jobs")
def start_job(body: dict):
    url = (body.get("url") or "").strip()
    if not url:
        raise HTTPException(400, "Missing YouTube URL")
    job_id = uuid.uuid4().hex[:10]
    jobs[job_id] = {"id": job_id, "url": url, "options": body.get("options") or {},
                    "estimate": body.get("estimate"), "status": "queued", "progress": 0.0,
                    "created": time.time(), "message": "Waiting in queue"}
    with lock:
        queue.append(job_id)
        wake.set()
    return jobs[job_id]


@app.get("/api/jobs")
def list_jobs():
    return sorted(jobs.values(), key=lambda j: j["created"], reverse=True)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Unknown job")
    return jobs[job_id]


# --------------------------------------------------------------------- estimate


@app.post("/api/estimate")
def estimate(body: dict):
    p.load_env()
    try:
        est = p.estimate(body.get("url", ""), options_from(body.get("options")))
    except Exception as e:
        raise HTTPException(400, f"Couldn't read that video: {str(e)[:200]}")
    existing = folder_for(est["id"])
    est["existing"] = existing is not None and (existing / "video.en.mp4").exists()
    return est


# ---------------------------------------------------------------------- library


def folder_for(video_id: str) -> Path | None:
    return next((d for d in p.DOWNLOADS.glob(f"*[[]{video_id}[]]") if d.is_dir()), None)


def record_for(folder: Path) -> dict:
    record_file = folder / "run.json"
    if record_file.exists():
        record = json.loads(record_file.read_text())
    else:  # runs from before run.json existed
        title, _, rest = folder.name.rpartition(" [")
        record = {"id": rest.rstrip("]"), "title": title or folder.name, "status": "done",
                  "legacy": True}
        started = (folder / "video.mp4").stat().st_mtime if (folder / "video.mp4").exists() else 0
        record["started"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started))
    record["folder"] = folder.name
    record["files"] = sorted(f for f in SERVED_FILES if (folder / f).exists())
    if record.get("status") == "done" and "video.en.mp4" not in record["files"] and "subtitles.en.srt" not in record["files"]:
        record["status"] = "incomplete"
    return record


@app.get("/api/library")
def library():
    if not p.DOWNLOADS.exists():
        return []
    records = [record_for(d) for d in p.DOWNLOADS.iterdir() if d.is_dir()]
    return sorted(records, key=lambda r: r.get("started") or "", reverse=True)


def library_file(video_id: str, name: str) -> Path:
    folder = folder_for(video_id)
    if not folder:
        raise HTTPException(404, "Not in the library")
    return folder / name


@app.get("/api/library/{video_id}/thumb.jpg")
def thumbnail(video_id: str):
    thumb = library_file(video_id, "thumb.jpg")
    if not thumb.exists():
        source = thumb.with_name("video.mp4")
        if not source.exists():
            raise HTTPException(404, "No video")
        at = max(1.0, p.video_duration(source) * 0.1)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{at:.1f}", "-i", str(source),
                        "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "4", str(thumb)])
    if not thumb.exists():
        raise HTTPException(404, "No thumbnail")
    return FileResponse(thumb, media_type="image/jpeg")


@app.get("/api/library/{video_id}/{name}")
def library_download(video_id: str, name: str, download: bool = False):
    if name not in SERVED_FILES:
        raise HTTPException(404, "Unknown file")
    path = library_file(video_id, name)
    if not path.exists():
        raise HTTPException(404, "Missing file")
    media = "video/mp4" if name.endswith(".mp4") else "application/x-subrip"
    record = record_for(path.parent)
    filename = f"{record.get('title') or video_id} - {name}" if download else None
    return FileResponse(path, media_type=media, filename=filename)


@app.delete("/api/library/{video_id}")
def delete_video(video_id: str):
    """Remove a video's folder from disk (original, subtitles, render) and revoke its shares."""
    folder = folder_for(video_id)
    if not folder:
        raise HTTPException(404, "Not in the library")
    busy = any(j["status"] in ("queued", "running") and video_id in j["url"] for j in jobs.values())
    if busy:
        raise HTTPException(409, "This video is being translated right now")
    shutil.rmtree(folder)
    shares = load_shares()
    save_shares({t: s for t, s in shares.items() if s["id"] != video_id})
    return {"deleted": video_id}


# ----------------------------------------------------------------------- shares
# A share is an unguessable link (/s/<token>) to one video's watch page; nothing else in the app
# is reachable through it. They expire, and can be revoked.

SHARES_FILE = p.DATA_DIR / "shares.json"


def load_shares() -> dict:
    shares = json.loads(SHARES_FILE.read_text()) if SHARES_FILE.exists() else {}
    now = time.time()
    return {t: s for t, s in shares.items() if s.get("expires", now + 1) > now}


def save_shares(shares: dict) -> None:
    SHARES_FILE.write_text(json.dumps(shares, indent=1))


def shared(token: str) -> tuple[dict, Path]:
    share = load_shares().get(token)
    folder = folder_for(share["id"]) if share else None
    if not share or not folder or not (folder / "video.en.mp4").exists():
        raise HTTPException(404, "This link has expired or was removed")
    return share, folder


@app.post("/api/library/{video_id}/share")
def create_share(video_id: str, body: dict | None = None):
    if not library_file(video_id, "video.en.mp4").exists():
        raise HTTPException(400, "Only videos with burned-in subtitles can be shared")
    days = max(1, min(90, int((body or {}).get("days") or 7)))
    token = secrets.token_urlsafe(16)
    shares = load_shares()
    shares[token] = {"id": video_id, "created": time.time(), "expires": time.time() + days * 86400}
    save_shares(shares)
    return {"token": token, "path": f"/s/{token}", **shares[token]}


@app.get("/api/library/{video_id}/shares")
def list_shares(video_id: str):
    return [{"token": t, "path": f"/s/{t}", **s} for t, s in load_shares().items() if s["id"] == video_id]


@app.delete("/api/shares/{token}")
def revoke_share(token: str):
    shares = load_shares()
    shares.pop(token, None)
    save_shares(shares)
    return {"revoked": token}


@app.get("/s/{token}", response_class=HTMLResponse)
def share_page(token: str):
    share, folder = shared(token)
    title = record_for(folder).get("title") or "Video"
    page = (STATIC / "share.html").read_text()
    return page.replace("{{TITLE}}", title.replace("&", "&amp;").replace("<", "&lt;")).replace("{{TOKEN}}", token)


@app.get("/s/{token}/video.mp4")
def share_video(token: str):
    _, folder = shared(token)
    return FileResponse(folder / "video.en.mp4", media_type="video/mp4")


# ----------------------------------------------------------------------- browse
# Pick a video to translate from a YouTube channel (Kan 11 by default) or one of its playlists.

CHANNELS = {"kan11": ("Kan 11", "https://www.youtube.com/@kan11")}
_browse_cache: dict[str, tuple[float, list]] = {}


def flat_list(url: str, limit: int = 60) -> list[dict]:
    cached = _browse_cache.get(url)
    if cached and time.time() - cached[0] < 600:
        return cached[1]
    import yt_dlp

    opts = {"quiet": True, "no_warnings": True, "extract_flat": True, "playlistend": limit}
    with yt_dlp.YoutubeDL(opts) as ydl:
        entries = ydl.extract_info(url, download=False).get("entries") or []
    _browse_cache[url] = (time.time(), entries)
    return entries


@app.get("/api/browse")
def browse(channel: str = "kan11", kind: str = "videos", playlist: str = ""):
    name, base = CHANNELS.get(channel, CHANNELS["kan11"])
    done = {r["id"] for r in library() if r.get("status") == "done"}
    try:
        if kind == "playlists":
            items = [{"playlist": e.get("id"), "title": e.get("title"),
                      "thumbnail": (e.get("thumbnails") or [{}])[-1].get("url", ""),
                      "count": e.get("playlist_count")} for e in flat_list(f"{base}/playlists", 100)]
            return {"channel": name, "items": [i for i in items if i["playlist"]]}
        url = f"https://www.youtube.com/playlist?list={playlist}" if playlist else f"{base}/videos"
        items = [{"id": e.get("id"), "title": e.get("title"), "duration": e.get("duration"),
                  "url": f"https://www.youtube.com/watch?v={e.get('id')}",
                  "thumbnail": f"https://i.ytimg.com/vi/{e.get('id')}/mqdefault.jpg",
                  "in_library": e.get("id") in done} for e in flat_list(url)]
        return {"channel": name, "items": [i for i in items if i["id"]]}
    except Exception as e:
        raise HTTPException(502, f"Couldn't load from YouTube: {str(e)[:160]}")


# --------------------------------------------------------------------- settings


def masked(value: str) -> dict:
    return {"set": bool(value), "hint": f"…{value[-4:]}" if len(value) > 8 else ("set" if value else "")}


@app.get("/api/settings")
def get_settings():
    p.load_env()
    return {name: masked(os.environ.get(var, "")) for name, var in KEYS.items()} | {
        "gemini_model": p.GEMINI_MODEL, "claude_model": p.DEFAULT_CLAUDE_MODEL,
        "fallback_model": p.FALLBACK_MODEL, "prices": p.PRICES}


@app.post("/api/settings")
def save_settings(body: dict):
    for name, var in KEYS.items():
        value = (body.get(name) or "").strip()
        if value:
            p.save_env(var, value)
    return get_settings()


@app.post("/api/settings/test")
def test_settings():
    """Free requests that check each saved key works."""
    p.load_env()
    results = {}
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic

            anthropic.Anthropic().models.list(limit=1)
            results["anthropic"] = "ok"
        except Exception as e:
            results["anthropic"] = f"error: {str(e)[:120]}"
    if os.environ.get("GEMINI_API_KEY"):
        try:
            from google import genai

            # Keep the client referenced: the model list is fetched lazily, after this line.
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            next(iter(client.models.list()))
            results["gemini"] = "ok"
        except Exception as e:
            results["gemini"] = f"error: {str(e)[:120]}"
    if os.environ.get("RUNPOD_API_KEY") and os.environ.get("RUNPOD_ENDPOINT_ID"):
        import requests

        r = requests.get(f"https://api.runpod.ai/v2/{os.environ['RUNPOD_ENDPOINT_ID']}/health",
                         headers={"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"}, timeout=15)
        results["runpod"] = "ok" if r.ok else f"error: HTTP {r.status_code}"
    return results


@app.get("/api/health")
def health():
    return {"ok": True, "ffmpeg": bool(p.shutil.which("ffmpeg")), "data": str(p.DATA_DIR)}


app.mount("/", StaticFiles(directory=STATIC, html=True), name="ui")


def serve(host: str = "127.0.0.1", port: int = 8420, open_browser: bool = True) -> None:
    import uvicorn
    import webbrowser

    p.load_env()
    p.DOWNLOADS.mkdir(parents=True, exist_ok=True)
    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}/"
    print(f"Tirgum is running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
