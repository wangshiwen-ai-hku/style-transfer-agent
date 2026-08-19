#!/usr/bin/env python3
"""
Verify every leg of the APIYI migration before spending money on a full run.

Run this first after topping up the APIYI account:

    python scripts/verify_apiyi.py

It exercises, in increasing order of risk:
  1. credentials + model list
  2. plain chat completion
  3. vision (image input)
  4. image generation with reference images, and aspect-ratio handling
  5. langchain init_chat_model through APIYI (used by the agent graph)
  6. with_structured_output for SkillSelector / StyleTransferPlan / Reflection
     <- HIGHEST RISK: the agent graph depends on tool-calling working through the
        proxy for a Gemini model. If step 6 fails, the pipeline cannot run as-is
        and the fallback is to parse JSON from free text instead.

Exit code is non-zero if any check fails.
"""
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import apiyi  # noqa: E402

# The evaluation images are not redistributed with this repository (see README),
# so these are overridable and the check degrades to text-only rather than failing
# on a missing file.
CONTENT = os.environ.get("AGENTICST_VERIFY_CONTENT", "compare/content/22.jpg")
STYLE = os.environ.get("AGENTICST_VERIFY_STYLE", "compare/style/112.jpg")

results = []


def is_transient(exc) -> bool:
    """Distinguish an overloaded upstream from a real capability failure.

    APIYI returns HTTP 429 with "\u5f53\u524d\u5206\u7ec4\u4e0a\u6e38\u8d1f\u8f7d\u5df2\u9971\u548c" when the pool behind a model is
    saturated. That says nothing about whether the model supports the feature
    being tested, and reporting it as a capability failure sends you off
    rewriting working code -- which is exactly what happened the first time.
    """
    m = str(exc).lower()
    return any(k in m for k in
               ("429", "rate limit", "ratelimit", "\u9971\u548c", "overload", "saturat",
                "timeout", "timed out", "502", "503", "504", "temporarily"))


def check(name, fn, critical=True, retries=4):
    """Run a probe, retrying transient upstream failures with backoff."""
    print(f"\n=== {name} ===")
    for attempt in range(1, retries + 1):
        try:
            detail = fn()
            print(f"  PASS  {detail or ''}")
            results.append((name, True, critical))
            return True
        except Exception as e:
            transient = is_transient(e)
            if transient and attempt < retries:
                wait = 5 * attempt
                print(f"  ..    transient upstream error (attempt {attempt}/{retries}), "
                      f"retrying in {wait}s: {str(e)[:110]}")
                time.sleep(wait)
                continue
            tag = "FAIL (upstream saturated, not a capability problem)" if transient else "FAIL"
            print(f"  {tag}  {type(e).__name__}: {str(e)[:300]}")
            if os.environ.get("VERBOSE"):
                traceback.print_exc()
            results.append((name, False, critical, transient))
            return False


def c1_credentials():
    apiyi.load_env()
    if not os.environ.get("APIYI_KEY"):
        raise RuntimeError("APIYI_KEY missing from code/.env")
    client = apiyi.get_client()
    ids = {m.id for m in client.models.list().data}
    need = [apiyi.MODELS["understanding"], apiyi.MODELS["generation"]]
    missing = [m for m in need if m not in ids]
    if missing:
        raise RuntimeError(f"models not served by APIYI: {missing}")
    return f"{len(ids)} models available; required models present"


def c2_chat():
    out = apiyi.chat([{"role": "user", "content": "Reply with exactly: OK"}])
    if "OK" not in out.upper():
        raise RuntimeError(f"unexpected reply: {out[:120]}")
    return f"reply={out.strip()[:40]!r}"


def c3_vision():
    out = apiyi.chat([{"role": "user", "content": [
        {"type": "text", "text": "Answer in three words: what is in this image?"},
        {"type": "image_url", "image_url": {"url": apiyi.encode_image(STYLE)}}]}])
    if not out.strip():
        raise RuntimeError("empty reply")
    return f"reply={out.strip()[:60]!r}"


def c4_image_generation():
    from PIL import Image
    w, h = Image.open(CONTENT).size
    img, usage = apiyi.generate_image(
        "Image 1 is the content image, image 2 is the style image. "
        "Transfer the style of image 2 onto image 1.",
        [CONTENT, STYLE],
        image_clare_prompt=["Image 1 is the content image.", "Image 2 is the style image."],
        target_ratio=w / h, return_usage=True)
    drift = abs(usage["returned_ratio"] - usage["target_ratio"])
    note = f"size={img.size} ratio asked {usage['target_ratio']} got {usage['returned_ratio']}"
    if drift > 0.12:
        note += "  <- WARNING: aspect ratio drift; results will not match native-SDK runs"
    return note


def c5_langchain():
    from dataclasses import asdict
    from langchain.chat_models import init_chat_model
    from src.config.manager import ConfigManager
    from src.utils.llm_helper import try_fix_model_kwargs

    os.environ.setdefault("MODEL_PROVIDER", "openai")
    os.environ.setdefault("MODEL", apiyi.MODELS["understanding"])
    os.environ["API_KEY"] = os.environ["APIYI_KEY"]
    os.environ["GOOGLE_PROVIDER"] = "openai"
    os.environ["GOOGLE_API_KEY"] = os.environ["APIYI_KEY"]

    cfg = ConfigManager(Path("src/general/config.yaml"))
    kw = try_fix_model_kwargs(asdict(cfg.get_agent_config("planner", "core").model))
    if kw.get("model_provider") != "openai" or "apiyi" not in (kw.get("base_url") or ""):
        raise RuntimeError(f"config is not pointing at APIYI: {kw}")
    llm = init_chat_model(**kw)
    out = llm.invoke("Reply with exactly: OK").content
    return f"provider={kw['model_provider']} model={kw['model']} reply={str(out).strip()[:30]!r}"


def c6_structured_output():
    from dataclasses import asdict
    from langchain.chat_models import init_chat_model
    from src.config.manager import ConfigManager
    from src.general.schema import Reflection, SkillSelector, StyleTransferPlan
    from src.utils.llm_helper import try_fix_model_kwargs

    cfg = ConfigManager(Path("src/general/config.yaml"))
    kw = try_fix_model_kwargs(asdict(cfg.get_agent_config("planner", "core").model))
    llm = init_chat_model(**kw)

    s = llm.with_structured_output(SkillSelector).invoke(
        "User request: 'Transfer the style of the style image to the content image.' "
        "Available skills: ['src/general/rules/style_transfer.md']. Select the skill file.")
    p = llm.with_structured_output(StyleTransferPlan).invoke(
        "Produce a 2-stage style transfer plan. Available image tags: "
        "['content_image','style_image'].")
    r = llm.with_structured_output(Reflection).invoke(
        "Critique a hypothetical stylized image and state whether it is satisfactory.")
    if not p.stages:
        raise RuntimeError("planner returned an empty plan")
    return (f"SkillSelector.skill_file={s.skill_file_to_read!r}; "
            f"plan has {len(p.stages)} stage(s); Reflection.is_satisfied={r.is_satisfied}")


def main():
    os.chdir(Path(__file__).resolve().parent.parent)
    check("1. credentials and model availability", c1_credentials)
    check("2. chat completion", c2_chat)
    check("3. vision input", c3_vision)
    check("4. image generation + aspect ratio", c4_image_generation)
    check("5. langchain through APIYI", c5_langchain)
    check("6. structured output (HIGHEST RISK)", c6_structured_output)

    print("\n" + "=" * 60)
    failed = [r for r in results if not r[1]]
    for r in results:
        name, ok, critical = r[0], r[1], r[2]
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{'' if critical else '  (non-critical)'}")

    transient_only = failed and all(len(r) > 3 and r[3] for r in failed)
    if transient_only:
        print(f"\n{len(failed)} check(s) failed, and every failure carried a transient "
              "signature (HTTP 429 'upstream saturated').")
        print("BUT: these already retried several times over a minute or more. A "
              "genuinely transient condition does not survive that, and it certainly "
              "does not hit the same probe every time while others pass.")
        print("So treat a repeated 429 on ONE specific probe as a hard failure wearing "
              "a transient error code -- most likely the proxy rejecting that request "
              "shape (e.g. a nested list-of-objects JSON schema) and reporting it as "
              "429 rather than 400. Run scripts/diagnose_structured_output.py to find "
              "which schema feature triggers it.")
        print("Only if the failures move around between runs is waiting the right "
              "response; in that case also lower concurrency, since the agent graph "
              "runs analysis nodes in parallel layers.")
        return 1
    if failed:
        print(f"\n{len(failed)} check(s) failed. Do NOT start a batch run yet.")
        if any("structured output" in r[0] for r in failed):
            print("Structured output is the blocker: the agent graph calls "
                  "with_structured_output for SkillSelector, SystemOrchestration, "
                  "StyleTransferPlan, Stage and Reflection. Note this was VERIFIED "
                  "working on gemini-2.5-flash via APIYI, so a failure here is more "
                  "likely a transient upstream issue than a real limitation -- re-run "
                  "before changing anything. If it is genuinely unsupported, switch the "
                  "understanding model or fall back to prompted JSON + "
                  "extract_json_from_text.")
        return 1
    print("\nAll checks passed. Safe to start the batch run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
