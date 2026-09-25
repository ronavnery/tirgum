# Tirgum (תרגום)

Download a YouTube video and create **English subtitles from Hebrew** speech.

1. Downloads the video (H.264 MP4) with `yt-dlp`, plus any Hebrew subtitles the uploader provided.
2. If there are none, transcribes the audio with [ivrit-ai's Hebrew-tuned Whisper](https://huggingface.co/ivrit-ai/whisper-large-v3-turbo-ct2).
3. Translates to English with Claude. If the video has Hebrew captions burned into the picture,
   Claude reads them from the frames and uses them as the primary source (they're edited and
   capture nuance better than speech recognition), with the speech transcript as a fallback.
4. Writes `subtitles.he.srt`, `subtitles.en.srt` and `video.en.mp4` with English subtitles burned in
   on a solid white box per line that covers any Hebrew captions already in the video. Output keeps the source
   resolution (highest-bitrate stream YouTube offers, up to 1080p without Premium).

## Web app

```bash
uv run tirgum serve
```

Opens http://localhost:8420: paste a YouTube link to see the estimated tokens, cost and time,
start the translation with a live progress bar, and browse the library of finished videos (watch,
download the video or subtitles, compare estimated vs. actual cost). API keys can be entered under
Settings; they're saved to `.env`.

### On a server (ronbox)

```bash
docker compose up -d --build
```

The container publishes no ports: it joins the `seedbox-stack_default` network, where Nginx Proxy
Manager reaches it as `http://tirgum:8420`. Add a proxy host for `tirgum.ronbox.me` pointing
there, with the same Authelia configuration as the other protected hosts, so login is handled by
Authelia. (`TIRGUM_PASSWORD` adds the app's own password prompt, only for setups without a proxy.)
Videos, keys and feedback are kept in `./data`. Transcription uses RunPod (the server has no GPU);
rendering uses the CPU.

## Setup

Requires `ffmpeg` and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

Keys go in `.env` (git-ignored):

```
ANTHROPIC_API_KEY=sk-ant-...
RUNPOD_API_KEY=...        # optional: transcribe on a RunPod GPU
RUNPOD_ENDPOINT_ID=...    # filled in by --setup-runpod
```

### RunPod (fast GPU transcription)

With `RUNPOD_API_KEY` in `.env`, run once:

```bash
uv run tirgum --setup-runpod
```

This creates a serverless endpoint running ivrit-ai's Hebrew Whisper worker
(`yairlifshitz/whisper-runpod-serverless`). It scales to zero, so you only pay while it transcribes.
After that, transcription runs on RunPod automatically (`--local` forces local). The video is still
downloaded on your machine (YouTube blocks most datacenter IPs); only compressed audio is sent,
in 20-minute chunks processed in parallel.

## Review & teach the translator

```bash
uv run tirgum --review "downloads/<video folder>"
```

Opens a local page (http://localhost:8765) with every line: Hebrew speech, Hebrew on-screen caption,
and the English candidates. Play each line's clip, click the translation(s) that are accurate, or
write the correct one. You can also write general translation rules. Everything is saved in
`feedback/` and included in every future translation: your corrections become examples to follow,
and your rules become instructions.

Keys: `J`/`K` next/previous line (plays it), `Space` replay, `1`/`2`/`3` mark final/voice/image
accurate, `E` write a correction.

## Usage

```bash
uv run tirgum "https://www.youtube.com/watch?v=VIDEO_ID"
```

Output goes to `downloads/<title> [<id>]/`.

| Option | Effect |
|---|---|
| `--captions auto/on/off` | Use burned-in Hebrew captions for the translation (default: auto-detect) |
| `--soft` | Toggleable subtitle track instead of burned-in (player styles it, no box) |
| `--position F` | Bottom edge of the subtitles as a fraction of height (default: over the Hebrew captions, else 0.93) |
| `--box-color white/black` | Box behind each line: white with dark text (default) or black with white text |
| `--box-opacity F` | Box opacity 0-1 (default 1 = solid, hides the Hebrew underneath) |
| `--local` | Transcribe locally even if RunPod is configured |
| `--no-video` | Only produce the `.srt` files |
| `--context "..."` | Hint for the translator, e.g. `"cooking show, casual tone"` |
| `--force-transcribe` | Ignore YouTube's Hebrew subtitles and transcribe the audio |
| `--translator whisper` | Offline translation with Whisper large-v3 (no API key, lower quality) |
| `--whisper-model` / `--claude-model` | Override models (defaults: `ivrit-ai/whisper-large-v3-turbo-ct2`, `claude-opus-5`) |
| `-o DIR` | Output folder (default `downloads`) |

Without an API key, `--translator auto` (the default) falls back to Whisper translation.

Notes: the first run downloads the Whisper model (~1.6 GB; ~3 GB for large-v3 in whisper mode).
Transcription runs on CPU at roughly real time on an M3 Pro.
