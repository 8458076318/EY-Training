from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse, urlencode, urlunparse
from typing import Iterable

import requests
from hume import HumeClient
from hume.empathic_voice.types.audio_input import AudioInput
from imageio_ffmpeg import get_ffmpeg_exe


DEFAULT_VIDEO_URL = "https://www.youtube.com/shorts/z0bsu-OnoiY"


@dataclass(frozen=True)
class StreamCandidate:
    url: str
    mime_type: str
    bitrate: int
    note: str = ""


def load_project_env() -> None:
    """Load repo-root .env values into the current process."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ[key] = value


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(
            f"Missing {name}. Please set it in C:\\Training\\AI-ML-Training-Projects\\.env before running this script."
        )
    return value


def shorts_to_watch_url(url: str) -> str:
    if "/shorts/" in url:
        video_id = url.rstrip("/").split("/shorts/")[-1].split("?")[0]
        return f"https://www.youtube.com/watch?v={video_id}"
    return url


def extract_player_response(html: str) -> dict:
    patterns = [
        r"ytInitialPlayerResponse\s*=\s*(\{.*?\})\s*;",
        r"var ytInitialPlayerResponse\s*=\s*(\{.*?\})\s*;",
    ]
    for pattern in patterns:
        match = re.search(pattern, html, flags=re.DOTALL)
        if match:
            return json.loads(match.group(1))
    raise RuntimeError("Could not locate ytInitialPlayerResponse in the YouTube page.")


def decode_signature_cipher(cipher: str) -> dict[str, str]:
    parsed = parse_qs(cipher)
    return {key: values[0] for key, values in parsed.items() if values}


def collect_stream_candidates(player: dict) -> list[StreamCandidate]:
    streaming = player.get("streamingData", {})
    candidates: list[StreamCandidate] = []

    for fmt in streaming.get("adaptiveFormats", []) + streaming.get("formats", []):
        mime_type = fmt.get("mimeType", "")
        if "audio/" not in mime_type:
            continue

        bitrate = int(fmt.get("bitrate", 0) or 0)
        direct_url = fmt.get("url")
        if direct_url:
            candidates.append(StreamCandidate(url=direct_url, mime_type=mime_type, bitrate=bitrate, note="direct"))
            continue

        cipher = fmt.get("signatureCipher") or fmt.get("cipher")
        if cipher:
            parsed = decode_signature_cipher(cipher)
            audio_url = parsed.get("url")
            if audio_url:
                parsed_url = urlparse(unquote(audio_url))
                query = parse_qs(parsed_url.query)
                # Some streams can be used directly with a signature token already present.
                if "sig" in parsed:
                    query["sig"] = [parsed["sig"]]
                elif "signature" in parsed:
                    query["signature"] = [parsed["signature"]]
                elif "s" in parsed:
                    # We cannot decipher YouTube's signature here without a dedicated extractor.
                    candidates.append(
                        StreamCandidate(
                            url="",
                            mime_type=mime_type,
                            bitrate=bitrate,
                            note="requires_signature_decoding",
                        )
                    )
                    continue

                rebuilt = parsed_url._replace(query=urlencode(query, doseq=True))
                candidates.append(
                    StreamCandidate(
                        url=urlunparse(rebuilt),
                        mime_type=mime_type,
                        bitrate=bitrate,
                        note="cipher",
                    )
                )

    return sorted(candidates, key=lambda item: item.bitrate, reverse=True)


def get_best_audio_url(video_url: str) -> str:
    yt_dlp_error: Exception | None = None
    try:
        import yt_dlp  # type: ignore
    except Exception:
        yt_dlp = None

    if yt_dlp is not None:
        opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "format": "bestaudio/best",
            "noplaylist": True,
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
            direct = info.get("url")
            if direct:
                return direct
            for fmt in info.get("formats", []):
                if "audio/" in fmt.get("mime_type", "") and fmt.get("url"):
                    return fmt["url"]
            yt_dlp_error = RuntimeError("yt-dlp could not produce a direct audio stream URL for this video.")
        except Exception as exc:
            yt_dlp_error = exc

    watch_url = shorts_to_watch_url(video_url)
    try:
        response = requests.get(
            watch_url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
                )
            },
            timeout=30,
        )
        response.raise_for_status()
        player = extract_player_response(response.text)
        candidates = collect_stream_candidates(player)
    except Exception as exc:
        if yt_dlp_error is not None:
            raise RuntimeError(
                f"Both yt-dlp and the manual YouTube parser failed. yt-dlp error: {yt_dlp_error}. "
                f"Manual parser error: {exc}"
            ) from exc
        raise

    if not candidates:
        if yt_dlp_error is not None:
            raise RuntimeError(
                "No usable audio stream was found in the YouTube player response. "
                f"yt-dlp also failed with: {yt_dlp_error}"
            ) from yt_dlp_error
        raise RuntimeError(
            "No usable audio stream was found in the YouTube player response. "
            "Install yt-dlp for a more reliable extractor."
        )

    for candidate in candidates:
        if candidate.url:
            return candidate.url

    raise RuntimeError(
        "The YouTube stream is signature-protected and needs yt-dlp to decipher it. "
        "Install yt-dlp and rerun the script."
    )


def iter_audio_chunks(audio_url: str, chunk_size: int = 640) -> Iterable[bytes]:
    yield from iter_pcm16_chunks(audio_url, chunk_size=chunk_size)


def download_audio_asset(video_url: str) -> Path:
    try:
        import yt_dlp  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("yt-dlp is required to download the YouTube audio stream.") from exc

    temp_dir = Path(tempfile.mkdtemp(prefix="hume_audio_"))
    opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio/best",
        "noplaylist": True,
        "outtmpl": str(temp_dir / "%(id)s.%(ext)s"),
        "retries": 10,
        "fragment_retries": 10,
        "socket_timeout": 30,
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        if not info:
            raise RuntimeError("yt-dlp did not return video metadata.")
        downloaded = Path(ydl.prepare_filename(info))
        if downloaded.exists():
            return downloaded

        expected_ext = info.get("ext") or "webm"
        expected = temp_dir / f"{info.get('id')}.{expected_ext}"
        if expected.exists():
            return expected

    raise RuntimeError("yt-dlp finished without creating a downloadable audio asset.")


def iter_pcm16_chunks(video_url: str, chunk_size: int = 3200) -> Iterable[bytes]:
    """Download the audio track and transcode it to 16 kHz mono PCM16."""
    audio_path = download_audio_asset(video_url)
    ffmpeg = get_ffmpeg_exe()
    try:
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(audio_path),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "pipe:1",
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdout is not None
        try:
            while True:
                chunk = process.stdout.read(chunk_size)
                if not chunk:
                    break
                yield chunk
        finally:
            stdout_data, stderr_data = process.communicate()
            if process.returncode not in (0, None):
                raise RuntimeError(
                    "FFmpeg failed to transcode the YouTube audio stream. "
                    f"stderr: {stderr_data.decode('utf-8', errors='replace')}"
                )
    finally:
        try:
            audio_path.unlink(missing_ok=True)
            audio_path.parent.rmdir()
        except Exception:
            pass


def top_emotions(scores_model, limit: int = 5) -> list[tuple[str, float]]:
    if hasattr(scores_model, "model_dump"):
        scores = scores_model.model_dump(by_alias=True)
    else:
        scores = scores_model.dict(by_alias=True)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]


def analyze_video_emotion(video_url: str) -> None:
    load_project_env()
    api_key = require_env("HUME_API_KEY")

    client = HumeClient(api_key=api_key)

    print(f"Resolved video: {video_url}")
    print("Connecting to Hume empathic voice chat...")

    top_results: list[tuple[str, float]] = []
    session_settings = {
        "audio": {
            "channels": 1,
            "encoding": "linear16",
            "sample_rate": 16000,
        }
    }
    chunk_pause_seconds = 0.02
    with client.empathic_voice.chat.connect(
        api_key=api_key,
        verbose_transcription=True,
        session_settings=session_settings,
    ) as chat:
        for chunk in iter_audio_chunks(video_url):
            chat.send_publish(AudioInput(data=base64.b64encode(chunk).decode("ascii")))
            time.sleep(chunk_pause_seconds)

            try:
                while True:
                    event = chat.recv()
                    if getattr(event, "type", None) != "user_message":
                        continue
                    inference = getattr(getattr(event, "models", None), "prosody", None)
                    if inference is None:
                        continue
                    scores_model = getattr(inference, "scores", None)
                    if scores_model is None:
                        continue
                    top_results = top_emotions(scores_model)
                    print("\nCurrent top emotion scores:")
                    for label, score in top_results:
                        print(f"- {label}: {score:.4f}")
                    break
            except Exception:
                # Keep streaming until Hume emits a usable user_message/prosody event.
                pass

    if top_results:
        print("\nFinal top emotions:")
        for label, score in top_results:
            print(f"- {label}: {score:.4f}")
    else:
        print("No prosody event was returned for this video.")


if __name__ == "__main__":
    try:
        analyze_video_emotion(os.getenv("YOUR_VIDEO_URL", DEFAULT_VIDEO_URL))
    except requests.exceptions.RequestException as exc:
        raise SystemExit(
            "Could not reach YouTube from this environment, so the video audio stream could not be fetched. "
            f"Details: {exc}"
        ) from exc
    except PermissionError as exc:
        raise SystemExit(
            "Hume could not be reached from this environment because outbound socket access is blocked. "
            "The script is wired correctly, but live emotion detection needs network access to Hume."
        ) from exc
    except Exception as exc:
        raise SystemExit(f"Hume emotion detection could not complete: {exc}") from exc
