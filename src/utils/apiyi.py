"""
APIYI backend adapter.

All model traffic in this project goes through APIYI's OpenAI-compatible endpoint
(https://api.apiyi.com/v1), so that the same code runs locally and on the GPU
server, which can reach APIYI but not Google/Aliyun/Volcengine directly.

Two things are worth knowing before editing this file.

1. Image generation comes back through /chat/completions, not /images/generations.
   The generated image is embedded in the assistant message as a markdown data URI:
       ![image](data:image/png;base64,....)
   We therefore parse the message content rather than reading an `images` field.
   `extract_images_from_message` also accepts bare data URIs and http(s) URLs, since
   proxies differ in how they wrap the payload.

2. The OpenAI-compatible schema has no `aspect_ratio` parameter, which the native
   Gemini SDK did have. We pass the target ratio in the prompt instead, and
   `generate_image` reports the ratio it actually received so callers can detect
   drift. Do not silently resize the result: a wrong aspect ratio is a real
   difference from the native-SDK results reported in the paper, and it should be
   visible rather than papered over.

Env: APIYI_KEY, BASE_URL (both in code/.env)
"""
from __future__ import annotations

import base64
import io
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

DEFAULT_BASE_URL = "https://api.apiyi.com/v1"

# Models verified present on APIYI (GET /v1/models) and used by this project.
MODELS = {
    "understanding":      "gemini-2.5-flash",
    "understanding_pro":  "gemini-2.5-pro",
    "generation":         "gemini-2.5-flash-image",
    "generation_pro":     "gemini-3-pro-image",
    "doubao":             "seedream-4-0-250828",
    "qwen_understanding": "qwen3-vl-plus",
    "flux":               "flux-kontext-pro",
}

_DATA_URI_RE = re.compile(r"data:image/(\w+);base64,([A-Za-z0-9+/=\s]+)")
_MD_URL_RE = re.compile(r"!\[[^\]]*\]\((https?://[^)\s]+)\)")


def is_transient(exc) -> bool:
    """True for upstream conditions that clear on their own.

    APIYI returns 429 "upstream group saturated" under load. It is not a
    capability failure and not a bad request; it just needs a longer wait.
    """
    m = str(exc).lower()
    return any(k in m for k in
               ("429", "rate limit", "ratelimit", "\u9971\u548c", "overload", "saturat",
                "502", "503", "504", "timeout", "timed out", "temporarily"))


def _backoff(attempt: int, exc) -> float:
    return (2 ** attempt) * (3.0 if is_transient(exc) else 1.0)


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name)
    return v if v else default


def load_env(dotenv_path: Optional[str] = None) -> None:
    """Load code/.env if python-dotenv is available. Safe to call repeatedly."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    if dotenv_path is None:
        dotenv_path = str(Path(__file__).resolve().parents[2] / ".env")
    load_dotenv(dotenv_path)


def get_client():
    """OpenAI SDK client pointed at APIYI."""
    from openai import OpenAI
    load_env()
    key = _env("APIYI_KEY") or _env("API_KEY") or _env("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("APIYI_KEY is not set. Put it in code/.env")
    return OpenAI(api_key=key, base_url=_env("BASE_URL", DEFAULT_BASE_URL))


def encode_image(path: str) -> str:
    """Local image path -> data URI. Pass-through for existing URIs/URLs."""
    if isinstance(path, str) and path.startswith(("http://", "https://", "data:")):
        return path
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image path not found: {path}")
    mime, _ = mimetypes.guess_type(path)
    if not mime or not mime.startswith("image/"):
        mime = "image/png"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def extract_images_from_message(message) -> List[Image.Image]:
    """Pull every generated image out of an assistant message.

    Handles, in order: a markdown/bare data URI in the text content; a markdown
    http(s) link (downloaded); and a non-standard `images` attribute that some
    proxies attach. Returns [] if the message carries no image.
    """
    out: List[Image.Image] = []

    content = getattr(message, "content", None)
    if isinstance(content, list):  # some proxies return content parts
        content = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    text = content or ""

    for _fmt, b64 in _DATA_URI_RE.findall(text):
        try:
            out.append(Image.open(io.BytesIO(base64.b64decode(re.sub(r"\s+", "", b64)))))
        except Exception:
            continue

    if not out:
        for url in _MD_URL_RE.findall(text):
            try:
                import requests
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                out.append(Image.open(io.BytesIO(r.content)))
            except Exception:
                continue

    if not out:
        for attr in ("images", "multi_mod_content"):
            payload = getattr(message, attr, None)
            if not payload:
                continue
            for item in payload if isinstance(payload, list) else [payload]:
                blob = item.get("image_url", {}).get("url") if isinstance(item, dict) else str(item)
                if not blob:
                    continue
                m = _DATA_URI_RE.search(blob)
                if m:
                    try:
                        out.append(Image.open(io.BytesIO(base64.b64decode(re.sub(r"\s+", "", m.group(2))))))
                    except Exception:
                        continue
    return out


def strip_images(text: str) -> str:
    """Remove embedded image payloads so logs stay readable."""
    if not text:
        return ""
    return re.sub(r"!\[[^\]]*\]\(data:image/[^)]+\)", "[IMAGE]", text).strip()


def _ratio_hint(target_ratio: Optional[float]) -> str:
    """Nearest nameable aspect ratio; the OpenAI schema has no ratio parameter,
    so this is conveyed through the prompt."""
    if not target_ratio:
        return ""
    named = {"1:1": 1.0, "3:2": 1.5, "2:3": 2 / 3, "3:4": 0.75, "4:3": 4 / 3,
             "4:5": 0.8, "5:4": 1.25, "9:16": 9 / 16, "16:9": 16 / 9, "21:9": 21 / 9}
    closest = min(named, key=lambda r: abs(named[r] - target_ratio))
    return f"\n\nOutput the image with an aspect ratio of {closest}."


_TAG_RE = re.compile(r"`([^`]+)`")


def _framing_index(image_paths: List[str], target_ratio: Optional[float],
                   tol: float = 0.12) -> Optional[int]:
    """Index of the input image that should be shown last to fix the output shape.

    Returns None when the last image already has the target shape, or when no
    input matches it -- moving an image with the wrong ratio to the end would make
    the framing worse rather than better.
    """
    if not target_ratio or len(image_paths) < 2:
        return None
    try:
        from PIL import Image as _Image

        def ratio(p):
            with _Image.open(p) as im:
                return im.size[0] / im.size[1]

        rs = [ratio(p) for p in image_paths]
        if abs(rs[-1] - target_ratio) <= tol:
            return None
        best = min(range(len(rs)), key=lambda i: abs(rs[i] - target_ratio))
        return best if abs(rs[best] - target_ratio) <= tol else None
    except Exception:
        return None


def _renumber(label: str, i: int) -> str:
    """Rewrite 'image 1 is `content_image`, ' for a new position, keeping the tag.

    Reordering without renumbering would leave the text and the position
    contradicting each other; the tag is what stage prompts refer to, so it is
    preserved and only the index changes.
    """
    m = _TAG_RE.search(str(label or ""))
    return f"image {i + 1} is `{m.group(1)}`, " if m else str(label or "")


def chat(messages, model: str = None, temperature: float = None,
         max_retries: int = 4, **kwargs) -> str:
    """Text/vision completion. Returns the assistant text with images stripped."""
    client = get_client()
    model = model or MODELS["understanding"]
    params = {"model": model, "messages": messages}
    if temperature is not None:
        params["temperature"] = temperature
    params.update(kwargs)

    last = None
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(**params)
            return strip_images(r.choices[0].message.content or "")
        except Exception as e:
            last = e
            if attempt == max_retries - 1:
                break
            time.sleep(_backoff(attempt, e))
    raise RuntimeError(f"APIYI chat failed after {max_retries} attempts: {last}")


def generate_image(text_prompt: str,
                   image_paths: List[str] = None,
                   image_clare_prompt: List[str] = None,
                   target_ratio: float = None,
                   model: str = None,
                   max_retries: int = 4,
                   return_usage: bool = False):
    """Generate one image, optionally conditioned on reference images.

    Images and their labels are interleaved in the order given, matching how the
    native-SDK path built its `contents` list, so stage prompts that say
    "image 1 is the content image" keep their meaning.
    """
    client = get_client()
    model = model or MODELS["generation"]
    image_paths = image_paths or []
    labels = list(image_clare_prompt or [])

    # Framing control is OFF by default, and the reported results were produced
    # with it off. Read this before turning it on.
    #
    # The generator takes the output's aspect ratio from the LAST image it is
    # shown, and ignores every documented way of asking for one: the
    # OpenAI-compatible schema has no aspect_ratio field, this proxy drops
    # extra_body, and the model does not act on a textual request. Measured on a
    # portrait content image with a landscape style reference, all of those
    # returned 1248x832 -- the style image's shape -- while showing the content
    # image last returned 896x1152. Re-showing the content image at the end as an
    # extra input was tried first and is unreliable: with a square style reference
    # the duplicate still yielded 1024x1024.
    #
    # Moving the reference to the end does work, but it is not a neutral fix. The
    # model is demonstrably position-sensitive -- that sensitivity is the very
    # mechanism being exploited -- so reordering changes the conditioning, and it
    # does so once per stage, which affects a multi-stage pipeline more than a
    # single call. Enabling it therefore alters what is being compared, and it is
    # not used for any number reported in the paper.
    #
    # Set AGENTICST_PRESERVE_FRAMING=1 to enable. When on, reordering is sound in
    # the narrow sense that stage prompts refer to images by tag (`content_image`)
    # rather than by position -- verified across all 37 stage prompts in the
    # evaluation runs -- and the labels are renumbered to match.
    idx = (_framing_index(image_paths, target_ratio)
           if os.environ.get("AGENTICST_PRESERVE_FRAMING", "").strip() in ("1", "true", "True")
           else None)
    if idx is not None:
        image_paths = image_paths[:idx] + image_paths[idx + 1:] + [image_paths[idx]]
        if len(labels) == len(image_paths):
            labels = labels[:idx] + labels[idx + 1:] + [labels[idx]]
            labels = [_renumber(l, i) for i, l in enumerate(labels)]

    parts = []
    for i, p in enumerate(image_paths):
        label = labels[i] if i < len(labels) else f"Image {i + 1}:"
        if label:
            parts.append({"type": "text", "text": str(label)})
        parts.append({"type": "image_url", "image_url": {"url": encode_image(p)}})
    parts.append({"type": "text", "text": text_prompt + _ratio_hint(target_ratio)})

    last = None
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(model=model,
                                               messages=[{"role": "user", "content": parts}])
            imgs = extract_images_from_message(r.choices[0].message)
            if not imgs:
                raise RuntimeError(
                    "response contained no image; text was: "
                    + strip_images(r.choices[0].message.content or "")[:300])
            img = imgs[0].convert("RGBA")
            if return_usage:
                u = r.usage
                usage = {"input_tokens": getattr(u, "prompt_tokens", None),
                         "output_tokens": getattr(u, "completion_tokens", None),
                         "total_tokens": getattr(u, "total_tokens", None),
                         "model": model,
                         "returned_ratio": round(img.size[0] / img.size[1], 4),
                         "target_ratio": round(target_ratio, 4) if target_ratio else None}
                return img, usage
            return img
        except Exception as e:
            last = e
            if attempt == max_retries - 1:
                break
            time.sleep(_backoff(attempt, e))
    raise RuntimeError(f"APIYI image generation failed after {max_retries} attempts: {last}")


def resolve_generation_model(name: str) -> str:
    """Map the project's short backend names onto APIYI model ids.

    Raises rather than substituting when a requested backend is unavailable: the
    open-source-generality experiment depends on actually using the model it
    claims to use, so a silent fallback would corrupt the result.
    """
    if not name:
        return MODELS["generation"]
    n = name.lower()
    if n in {m.lower() for m in MODELS.values()}:
        return name
    if "qwen" in n:
        raise ValueError(
            "APIYI does not serve a Qwen image-editing model (only qwen3-vl-* for "
            "understanding). Run the Qwen generation ablation against Aliyun/DashScope "
            "directly, or drop it -- do not silently substitute another generator.")
    if "seedream" in n or "doubao" in n:
        return MODELS["doubao"]
    if "flux" in n:
        return MODELS["flux"]
    if "pro" in n and "gemini" in n:
        return MODELS["generation_pro"]
    if "gemini" in n or "genai" in n or n == "default":
        return MODELS["generation"]
    return name  # assume the caller passed a literal APIYI model id


def langchain_model_kwargs(model: str = None, temperature: float = None) -> dict:
    """kwargs for langchain's init_chat_model so the agent graph also uses APIYI."""
    load_env()
    kw = {"model_provider": "openai",
          "model": model or MODELS["understanding"],
          "api_key": _env("APIYI_KEY"),
          "base_url": _env("BASE_URL", DEFAULT_BASE_URL)}
    if temperature is not None:
        kw["temperature"] = temperature
    return kw
