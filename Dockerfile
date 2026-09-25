# Tirgum web app for a Linux server (e.g. ronbox). Data (videos, .env keys, feedback) lives in /data.
FROM python:3.12-slim

# ffmpeg renders and scans video; Liberation Sans is metric-compatible with Arial for subtitles.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg fonts-liberation fontconfig ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev --python /usr/local/bin/python3
COPY tirgum ./tirgum
RUN uv sync --frozen --no-dev --python /usr/local/bin/python3

ENV TIRGUM_HOME=/data PYTHONUNBUFFERED=1
VOLUME /data
EXPOSE 8420
CMD ["uv", "run", "--no-sync", "tirgum", "serve", "--host", "0.0.0.0", "--no-open"]
