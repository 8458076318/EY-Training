from __future__ import annotations

import contextlib
import io
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, unquote, urlencode, urlparse, urlunparse

import requests
from elevenlabs import ElevenLabs


DEFAULT_VIDEO_URL = "https://www.youtube.com/shorts/z0bsu-OnoiY"
DEFAULT_ELEVENLABS_MODEL = "scribe_v2"
DEFAULT_DEMO_TRANSCRIPT = (
    "I was really worried at first, but now I feel much better. Thanks for your help."
)
DEFAULT_VAPI_CHAT_URLS = [
    "https://api.vapi.ai/openai/v1/chat/completions",
    "https://api.vapi.ai/openai/chat/completions",
    "https://api.vapi.ai/chat",
    "https://api.vapi.ai/chat/completions",
    "https://api.vapi.ai/v1/chat/completions",
    "https://api.vapi.ai/v1/chat",
    "https://api.vapi.ai/v1/assistant/chat",
]
DEFAULT_VAPI_MODEL = os.getenv("VAPI_MODEL", "gpt-4o-mini")


@dataclass(frozen=True)
class StreamCandidate:
    url: str
    mime_type: str
    bitrate: int
    note: str = ""


@dataclass(frozen=True)
class EmotionResult:
    label: str
    confidence: float
    scores: dict[str, float]
    evidence: list[str]
    empathic_voice_style: str


def load_project_env() -> None:
    """Load repo-root .env values into the current process."""
    repo_root = Path(__file__).resolve().parents[1]
    candidate_envs = [
        repo_root / ".env",
        repo_root / "multi_agent_planner" / ".env",
        repo_root / "date_19_06" / ".env",
    ]

    for env_path in candidate_envs:
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
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
                if "sig" in parsed:
                    query["sig"] = [parsed["sig"]]
                elif "signature" in parsed:
                    query["signature"] = [parsed["signature"]]
                elif "s" in parsed:
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


def download_audio_asset(video_url: str) -> Path:
    try:
        import yt_dlp  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("yt-dlp is required to download the YouTube audio stream.") from exc

    temp_dir = Path(tempfile.mkdtemp(prefix="elevenlabs_audio_"))
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

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
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


def extract_transcript_text(result: object) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        for key in ("text", "transcript", "transcription"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for attr in ("text", "transcript", "transcription"):
        value = getattr(result, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if hasattr(result, "model_dump"):
        payload = result.model_dump()
        for key in ("text", "transcript", "transcription"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return str(result).strip()


def transcribe_with_elevenlabs(audio_path: Path) -> tuple[str, object]:
    api_key = require_env("ELEVENLABS_API_KEY")
    client = ElevenLabs(api_key=api_key)
    with audio_path.open("rb") as fh:
        result = client.speech_to_text.convert(
            model_id=DEFAULT_ELEVENLABS_MODEL,
            file=fh,
            enable_logging=False,
        )
    transcript = extract_transcript_text(result)
    if not transcript:
        raise RuntimeError("ElevenLabs returned an empty transcript.")
    return transcript, result


def resolve_transcript(video_url: str) -> tuple[str, str]:
    live_fetch_enabled = os.getenv("LIVE_YOUTUBE_FETCH", "1").strip().lower() not in {"0", "false", "no"}
    if live_fetch_enabled:
        try:
            print("Attempting live ElevenLabs transcription from the YouTube audio...")
            audio_path = download_audio_asset(video_url)
            print(f"Downloaded audio asset: {audio_path}")
            transcript, _ = transcribe_with_elevenlabs(audio_path)
            try:
                audio_path.unlink(missing_ok=True)
                audio_path.parent.rmdir()
            except Exception:
                pass
            return transcript, "live"
        except Exception as exc:
            if os.getenv("VERBOSE_ERRORS", "").strip().lower() in {"1", "true", "yes"}:
                print(f"Live YouTube -> ElevenLabs path failed: {exc}")
            else:
                print(
                    "Live YouTube -> ElevenLabs path failed because this environment blocked the YouTube request."
                )

    transcript_override = os.getenv("TRANSCRIPT_TEXT", "").strip()
    if transcript_override:
        print("Using TRANSCRIPT_TEXT fallback from the environment.")
        return transcript_override, "override"

    print("Using built-in demo transcript fallback because the live fetch path is unavailable here.")
    return DEFAULT_DEMO_TRANSCRIPT, "demo"


def infer_local_emotion(transcript: str) -> EmotionResult:
    text = transcript.lower()
    tokens = re.findall(r"[a-z']+", text)
    token_counts = Counter(tokens)

    keyword_groups: dict[str, list[str]] = {
        "joy": ["great", "amazing", "awesome", "happy", "love", "excited", "thanks", "thank", "good", "wonderful"],
        "sadness": ["sad", "sorry", "bad", "upset", "hurt", "tired", "lonely", "miss", "cry", "pain"],
        "anger": ["angry", "annoyed", "hate", "furious", "frustrated", "mad", "irritated", "worst"],
        "fear": ["afraid", "scared", "worry", "worried", "anxious", "panic", "nervous", "concerned"],
        "surprise": ["wow", "unexpected", "surprised", "shocked", "sudden", "amazing"],
        "neutral": ["okay", "fine", "maybe", "just", "simply", "normal"],
    }

    scores: dict[str, float] = {label: 0.0 for label in keyword_groups}
    evidence: list[str] = []

    for label, keywords in keyword_groups.items():
        for keyword in keywords:
            count = token_counts.get(keyword, 0)
            if count:
                scores[label] += float(count)
                if len(evidence) < 5:
                    evidence.append(keyword)

    punctuation_boost = 0.0
    if "!" in transcript:
        punctuation_boost += 0.5
        scores["joy"] += 0.25
        scores["surprise"] += 0.25
    if "?" in transcript:
        scores["fear"] += 0.2
        scores["surprise"] += 0.2
    if re.search(r"\b(?:i'm|i am|i feel|feeling)\b", text):
        scores["sadness"] += 0.2
        scores["fear"] += 0.1
    if re.search(r"\b(?:never|nothing|nobody|no one|cannot|can't|won't)\b", text):
        scores["sadness"] += 0.1
        scores["anger"] += 0.1

    word_count = max(len(tokens), 1)
    intensity = min(1.0, (sum(scores.values()) + punctuation_boost) / (word_count / 6.0 + 1.0))

    if all(score == 0 for score in scores.values()):
        label = "neutral"
        confidence = 0.35
    else:
        label, raw_score = max(scores.items(), key=lambda item: item[1])
        total = sum(scores.values()) or 1.0
        confidence = min(0.99, max(0.35, raw_score / total + intensity * 0.15))

    empathic_voice_style = {
        "joy": "bright, warm, upbeat",
        "sadness": "gentle, calm, reassuring",
        "anger": "steady, respectful, de-escalating",
        "fear": "soft, clear, grounding",
        "surprise": "clear, lively, attentive",
        "neutral": "balanced, friendly, professional",
    }.get(label, "balanced, friendly, professional")

    return EmotionResult(
        label=label,
        confidence=round(confidence, 3),
        scores=dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)),
        evidence=evidence[:5],
        empathic_voice_style=empathic_voice_style,
    )


def infer_emotion_with_optional_transformer(transcript: str) -> EmotionResult:
    text = transcript.strip()
    if not text:
        return infer_local_emotion(transcript)

    if os.getenv("USE_TRANSFORMERS_EMOTION_MODEL", "0").strip().lower() not in {"1", "true", "yes"}:
        return infer_local_emotion(transcript)

    try:
        from transformers import pipeline  # type: ignore
    except Exception:
        return infer_local_emotion(transcript)

    try:
        classifier = pipeline(
            task="text-classification",
            model=os.getenv("TRANSFORMERS_EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base"),
            top_k=None,
            truncation=True,
        )
        outputs = classifier(text)
    except Exception:
        return infer_local_emotion(transcript)

    if isinstance(outputs, list) and outputs and isinstance(outputs[0], list):
        outputs = outputs[0]
    if not isinstance(outputs, list):
        return infer_local_emotion(transcript)

    scores = {str(item["label"]).lower(): float(item["score"]) for item in outputs if isinstance(item, dict) and "label" in item and "score" in item}
    if not scores:
        return infer_local_emotion(transcript)

    label, confidence = max(scores.items(), key=lambda item: item[1])
    evidence = sorted(scores, key=scores.get, reverse=True)[:5]

    empathic_voice_style = {
        "joy": "bright, warm, upbeat",
        "happiness": "bright, warm, upbeat",
        "sadness": "gentle, calm, reassuring",
        "anger": "steady, respectful, de-escalating",
        "fear": "soft, clear, grounding",
        "surprise": "clear, lively, attentive",
        "neutral": "balanced, friendly, professional",
    }.get(label, "balanced, friendly, professional")

    return EmotionResult(
        label=label,
        confidence=round(confidence, 3),
        scores=dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)),
        evidence=evidence,
        empathic_voice_style=empathic_voice_style,
    )


def build_vapi_handoff(transcript: str, emotion: EmotionResult) -> dict[str, str]:
    return {
        "summary": (
            f"Emotion detection: {emotion.label} at {emotion.confidence:.2f} confidence. "
            f"Recommended voice style: {emotion.empathic_voice_style}."
        ),
        "transcript": transcript,
        "emotion": emotion.label,
        "confidence": f"{emotion.confidence:.2f}",
        "voice_style": emotion.empathic_voice_style,
    }


def invoke_vapi_voice_agent(transcript: str, emotion: EmotionResult) -> str | None:
    vapi_key = os.getenv("VAPI_API_KEY", "").strip()
    if not vapi_key:
        print("VAPI_API_KEY is not set, so the Vapi voice-agent step is skipped.")
        return None

    handoff = build_vapi_handoff(transcript, emotion)
    prompt = (
        "You are a calm, empathic voice agent.\n"
        "Respond to the transcript in a short, helpful way using the requested voice style.\n"
        f"Voice style: {emotion.empathic_voice_style}\n"
        f"Detected emotion: {emotion.label} ({emotion.confidence:.2f})\n"
        f"Transcript: {transcript}\n"
        "Keep the response concise, empathetic, and natural for speech."
    )

    payload_variants = []
    assistant_id = os.getenv("VAPI_ASSISTANT_ID", "").strip()
    if assistant_id:
        payload_variants.append(
            {
                "assistantId": assistant_id,
                "model": DEFAULT_VAPI_MODEL,
                "messages": [{"role": "user", "content": prompt}],
            }
        )
    payload_variants.append(
        {
            "model": DEFAULT_VAPI_MODEL,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
        }
    )

    base_url = os.getenv("VAPI_CHAT_URL", "").strip()
    candidate_urls = [base_url] if base_url else DEFAULT_VAPI_CHAT_URLS
    auth_headers = []
    if vapi_key.lower().startswith("bearer "):
        auth_headers.append({"Authorization": vapi_key})
    else:
        auth_headers.extend(
            [
                {"Authorization": f"Bearer {vapi_key}"},
                {"Authorization": vapi_key},
                {"x-api-key": vapi_key},
                {"api-key": vapi_key},
            ]
        )

    last_error: Exception | None = None
    last_status_code: int | None = None
    for url in candidate_urls:
        if not url:
            continue
        for auth_headers_variant in auth_headers:
            headers = {
                **auth_headers_variant,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            for payload in payload_variants:
                try:
                    response = requests.post(url, headers=headers, json=payload, timeout=60)
                    response.raise_for_status()
                    data = response.json()
                    if isinstance(data, dict):
                        for key in ("message", "output", "text", "response"):
                            value = data.get(key)
                            if isinstance(value, str) and value.strip():
                                return value.strip()
                        choices = data.get("choices")
                        if isinstance(choices, list) and choices:
                            first = choices[0]
                            if isinstance(first, dict):
                                message = first.get("message")
                                if isinstance(message, dict):
                                    content = message.get("content")
                                    if isinstance(content, str) and content.strip():
                                        return content.strip()
                                content = first.get("content")
                                if isinstance(content, str) and content.strip():
                                    return content.strip()
                        if data:
                            return json.dumps(data, indent=2)
                except requests.HTTPError as exc:
                    last_error = exc
                    response = exc.response
                    if response is not None:
                        last_status_code = response.status_code
                        if response.status_code in (401, 404):
                            continue
                    continue
                except Exception as exc:
                    last_error = exc
                    continue

    print("VAPI_API_KEY loaded.")
    print("Prepared Vapi handoff context:")
    print(json.dumps(handoff, indent=2))
    if last_error is not None:
        if last_status_code == 404:
            print(
                "Vapi request could not complete because the default chat endpoint is not correct for this account. "
                "Set VAPI_CHAT_URL to the exact Vapi chat/OpenAI-compatibility URL from your dashboard."
            )
        else:
            print(f"Vapi request could not complete: {last_error}")
    return None


def analyze_video_emotion(video_url: str) -> None:
    load_project_env()
    elevenlabs_key = require_env("ELEVENLABS_API_KEY")

    print(f"Resolved video: {shorts_to_watch_url(video_url)}")
    print("Using ELEVENLABS_API_KEY from C:\\Training\\AI-ML-Training-Projects\\.env")
    transcript, transcript_mode = resolve_transcript(video_url)
    print(f"Transcript source: {transcript_mode}")

    emotion = infer_emotion_with_optional_transformer(transcript)

    print("\nTranscript:")
    print(transcript)
    print("\nLocal emotion analysis:")
    print(f"- label: {emotion.label}")
    print(f"- confidence: {emotion.confidence:.3f}")
    print(f"- empathic voice style: {emotion.empathic_voice_style}")
    if emotion.evidence:
        print(f"- evidence: {', '.join(emotion.evidence)}")
    print("Emotion scores:")
    for label, score in emotion.scores.items():
        print(f"- {label}: {score:.3f}")

    vapi_output = invoke_vapi_voice_agent(transcript, emotion)
    if vapi_output:
        print("\nVapi voice-agent output:")
        print(vapi_output)

    _ = elevenlabs_key  # keep the loaded key explicit in the execution path


if __name__ == "__main__":
    try:
        analyze_video_emotion(os.getenv("YOUR_VIDEO_URL", DEFAULT_VIDEO_URL))
    except requests.exceptions.RequestException as exc:
        raise SystemExit(
            "Could not reach YouTube from this environment, so the video audio stream could not be fetched. "
            f"Details: {exc}"
        ) from exc
    except Exception as exc:
        raise SystemExit(f"ElevenLabs emotion analysis could not complete: {exc}") from exc
