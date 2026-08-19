from langchain_core.messages import BaseMessage
from langchain.chat_models import BaseChatModel
from PIL import Image
from typing import Union, List
from pathlib import Path
import time
import asyncio
from pydantic import BaseModel
import base64
import json
import os
import io


def try_fix_model_kwargs(model_kwargs: dict) -> dict:
    """Try to fix the model kwargs if the model provider is google_vertexai"""
    if model_kwargs.get('model_provider') == "azure_openai":
        model_kwargs['azure_endpoint'] = "https://routinetask.cognitiveservices.azure.com/"
        model_kwargs['api_version'] = os.environ.get('API_VERSION', '2024-12-01-preview')
        model_kwargs['temperature'] = 1.0
    # if model_kwargs.get('model_provider') == "openai":
    #     model_kwargs['base_url'] = "https://ark.cn-beijing.volces.com/api/v3"
    # if model_kwargs.get('model_provider') == "openai":
    #     model_kwargs['base_url'] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if model_kwargs.get('model_provider') != "openai":
        model_kwargs.pop('base_url', None)
    if model_kwargs.get('model_provider') == "google_genai":
        # Set transport to 'rest' to avoid location/network issues with google_genai
        # This helps bypass "User location is not supported for the API use" errors
        model_kwargs['transport'] = "rest"
    
    return model_kwargs

def _is_transient_upstream(exc) -> bool:
    """True for errors that clear on their own (rate limits, saturation, 5xx).

    Kept separate from the generic retry so the wait can be longer: a saturated
    upstream typically needs tens of seconds, not one.
    """
    m = str(exc).lower()
    return any(k in m for k in
               ("429", "rate limit", "ratelimit", "\u9971\u548c", "overload", "saturat",
                "502", "503", "504", "timeout", "timed out", "temporarily"))


def estimate_text_tokens(text: str) -> int:
    """Estimate text tokens using approximate word-based counting for Gemini models."""
    if not text:
        return 0
    # Approximate: ~4 characters per token for English text
    return max(1, len(text) // 4)

def estimate_image_tokens(image_path: str = None, base64_data: str = None) -> tuple[int, int]:
    """
    Estimate image tokens for Gemini models.
    Returns (image_count, estimated_tokens)
    Gemini charges based on image size:
    - Images under 512x512: 85 tokens
    - 512x512 to 1024x1024: 170 tokens
    - Over 1024x1024: 340 tokens
    """
    if not image_path and not base64_data:
        return 0, 0

    try:
        if image_path:
            with Image.open(image_path) as img:
                width, height = img.size
        elif base64_data:
            # Decode base64 and get image size
            image_data = base64.b64decode(base64_data)
            with Image.open(io.BytesIO(image_data)) as img:
                width, height = img.size
        else:
            return 1, 85  # Default fallback

        # Gemini token estimation based on image size
        if width <= 512 and height <= 512:
            return 1, 85
        elif width <= 1024 and height <= 1024:
            return 1, 170
        else:
            return 1, 340
    except Exception:
        # Fallback for any image processing errors
        return 1, 85

async def llm_call_and_report(llm: BaseChatModel, messages: List[BaseMessage] | BaseMessage, file: str | Path, max_retries: int = 5, retry_delay: float = 2.0) -> BaseMessage:
    """
    Call the LLM and log the token consumption and response tokens to file
    Args:
        llm: BaseChatModel
        messages: List[BaseMessage] | BaseMessage
        file: str | Path - path to log file
        max_retries: int - maximum number of retry attempts (default: 3)
        retry_delay: float - delay between retries in seconds (default: 1.0)
    Returns:
        BaseMessage: The LLM response
    """
    start_time = time.time()

    # Ensure messages is a list
    if not isinstance(messages, list):
        messages = [messages]

    # Analyze input tokens
    input_text_tokens = 0
    input_images_count = 0
    input_image_tokens = 0

    for message in messages:
        if hasattr(message, 'content'):
            content = message.content
            if isinstance(content, str):
                input_text_tokens += estimate_text_tokens(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get('type') == 'text':
                            text = item.get('text', '')
                            input_text_tokens += estimate_text_tokens(text)
                        elif item.get('type') == 'image_url':
                            image_url = item.get('image_url', {})
                            if isinstance(image_url, dict) and 'url' in image_url:
                                url = image_url['url']
                                if url.startswith('data:image'):
                                    # Base64 encoded image
                                    img_count, img_tokens = estimate_image_tokens(base64_data=url.split(',')[1] if ',' in url else None)
                                    input_images_count += img_count
                                    input_image_tokens += img_tokens
                                else:
                                    # File path
                                    input_images_count += 1
                                    input_image_tokens += 85  # Default estimate

    # Call the LLM with retry mechanism
    last_exception = None
    retry_count = 0
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            response = await llm.ainvoke(messages)
            break  # Success, exit retry loop
        except Exception as e:
            last_exception = e
            if attempt < max_retries:  # Don't wait after the last attempt
                retry_count += 1
                # APIYI answers 429 "upstream group saturated" when the pool behind a
                # model is busy. That clears on its own but can take tens of seconds,
                # so back off harder for it than for an ordinary error -- otherwise a
                # long batch run dies partway through on a condition that would have
                # resolved by itself.
                wait = retry_delay * 3 if _is_transient_upstream(e) else retry_delay
                print(f"LLM call attempt {attempt + 1} failed: {e}. Retrying in {wait:.0f}s...")
                await asyncio.sleep(wait)
                # Exponential backoff: double the delay for next retry
                retry_delay *= 2
            else:
                print(f"LLM call failed after {max_retries + 1} attempts. Last error: {e}")
                raise e  # Re-raise the last exception

    end_time = time.time()
    duration = end_time - start_time

    # Analyze output tokens
    output_text_tokens = 0
    output_images_count = 0
    output_image_tokens = 0

    # Handle structured output (Pydantic models)
    if hasattr(response, 'model_dump') or hasattr(response, '__dict__'):
        try:
            # Try to serialize structured output to JSON for token estimation
            if hasattr(response, 'model_dump'):
                # Pydantic model
                response_json = json.dumps(response.model_dump(), ensure_ascii=False)
            else:
                # Other structured objects
                response_json = json.dumps(response.__dict__, ensure_ascii=False)
            output_text_tokens = estimate_text_tokens(response_json)
        except Exception:
            # Fallback: convert to string
            response_str = str(response)
            output_text_tokens = estimate_text_tokens(response_str)
    # Handle content-based responses (like regular LLM responses)
    elif hasattr(response, 'content'):
        content = response.content
        if isinstance(content, str):
            output_text_tokens = estimate_text_tokens(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if item.get('type') == 'text':
                        text = item.get('text', '')
                        output_text_tokens += estimate_text_tokens(text)
                    elif item.get('type') == 'image_url':
                        output_images_count += 1
                        output_image_tokens += 85  # Default estimate

    # Calculate totals
    total_input_tokens = input_text_tokens + input_image_tokens
    total_output_tokens = output_text_tokens + output_image_tokens
    total_tokens = total_input_tokens + total_output_tokens

    # Log to file
    log_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": round(duration, 3),
        "retry_count": retry_count,
        "input_tokens": {
            "text": input_text_tokens,
            "images_count": input_images_count,
            "images": input_image_tokens,
            "total": total_input_tokens
        },
        "output_tokens": {
            "text": output_text_tokens,
            "images_count": output_images_count,
            "images": output_image_tokens,
            "total": total_output_tokens
        },
        "total_tokens": total_tokens,
        "model": str(llm) if hasattr(llm, '__str__') else "unknown"
    }

    # Ensure directory exists
    log_file_path = Path(file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Write log
    with open(log_file_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    return response

def calculate_token_consumption(log_dir: str | Path) -> None:
    """
    Calculate the token consumption and time consumption for all LLM calls in the log directory
    Args:
        log_dir: str | Path - directory containing JSON log files
    Returns:
        Creates a CSV file with detailed statistics and prints summary
    """
    import csv
    from pathlib import Path

    log_path = Path(log_dir)
    if not log_path.exists() or not log_path.is_dir():
        print(f"Log directory {log_dir} does not exist or is not a directory")
        return

    # Find all JSON log files
    json_files = list(log_path.glob("*.json"))
    if not json_files:
        print(f"No JSON log files found in {log_dir}")
        return

    # Collect data from all log files
    all_calls = []
    totals = {
        "num_llm_calls": 0,
        "total_duration": 0.0,
        "total_retry_count": 0,
        "total_input_text_tokens": 0,
        "total_input_images_count": 0,
        "total_input_image_tokens": 0,
        "total_output_text_tokens": 0,
        "total_output_images_count": 0,
        "total_output_image_tokens": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0
    }

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract call data
            call_data = {
                "llm_call": json_file.stem,  # filename without extension
                "timestamp": data.get("timestamp", ""),
                "duration": data.get("duration_seconds", 0),
                "retry_count": data.get("retry_count", 0),
                "model": data.get("model", ""),
                "input_text_tokens": data.get("input_tokens", {}).get("text", 0),
                "input_images_count": data.get("input_tokens", {}).get("images_count", 0),
                "input_image_tokens": data.get("input_tokens", {}).get("images", 0),
                "input_total_tokens": data.get("input_tokens", {}).get("total", 0),
                "output_text_tokens": data.get("output_tokens", {}).get("text", 0),
                "output_images_count": data.get("output_tokens", {}).get("images_count", 0),
                "output_image_tokens": data.get("output_tokens", {}).get("images", 0),
                "output_total_tokens": data.get("output_tokens", {}).get("total", 0),
                "total_tokens": data.get("total_tokens", 0)
            }

            all_calls.append(call_data)

            # Update totals
            totals["num_llm_calls"] += 1
            totals["total_duration"] += call_data["duration"]
            totals["total_retry_count"] += call_data["retry_count"]
            totals["total_input_text_tokens"] += call_data["input_text_tokens"]
            totals["total_input_images_count"] += call_data["input_images_count"]
            totals["total_input_image_tokens"] += call_data["input_image_tokens"]
            totals["total_input_tokens"] += call_data["input_total_tokens"]
            totals["total_output_text_tokens"] += call_data["output_text_tokens"]
            totals["total_output_images_count"] += call_data["output_images_count"]
            totals["total_output_image_tokens"] += call_data["output_image_tokens"]
            totals["total_output_tokens"] += call_data["output_total_tokens"]
            totals["total_tokens"] += call_data["total_tokens"]

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    if not all_calls:
        print("No valid log data found")
        return

    # Sort by timestamp
    all_calls.sort(key=lambda x: x["timestamp"])

    # Write detailed CSV
    csv_path = log_path / "llm_token_consumption_report.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            "llm_call", "timestamp", "duration", "retry_count", "model",
            "input_text_tokens", "input_images_count", "input_image_tokens", "input_total_tokens",
            "output_text_tokens", "output_images_count", "output_image_tokens", "output_total_tokens",
            "total_tokens"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_calls)

    # Write summary CSV
    summary_path = log_path / "llm_token_consumption_summary.csv"
    with open(summary_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            "metric", "value"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for key, value in totals.items():
            writer.writerow({"metric": key, "value": value})

    # Print summary
    print("=== LLM Token Consumption Summary ===")
    print(f"Total LLM calls: {totals['num_llm_calls']}")
    print(".2f")
    print(f"Total retry attempts: {totals['total_retry_count']}")
    print(f"Average retries per call: {totals['total_retry_count'] / totals['num_llm_calls']:.2f}" if totals['num_llm_calls'] > 0 else "Average retries per call: 0.00")
    print(f"Total input tokens: {totals['total_input_tokens']} (text: {totals['total_input_text_tokens']}, images: {totals['total_input_image_tokens']})")
    print(f"Total output tokens: {totals['total_output_tokens']} (text: {totals['total_output_text_tokens']}, images: {totals['total_output_image_tokens']})")
    print(f"Total tokens (input + output): {totals['total_tokens']}")
    print(f"Input images processed: {totals['total_input_images_count']}")
    print(f"Output images generated: {totals['total_output_images_count']}")
    print(f"\nDetailed report saved to: {csv_path}")
    print(f"Summary report saved to: {summary_path}")


# ---------------------------------------------------------------------------
# Structured output with a prompted-JSON fallback.
#
# Why this exists: the APIYI proxy rejects tool/function schemas that contain a
# nested array of objects. It answers HTTP 429 "upstream group saturated", which
# reads like load but is fully deterministic -- verified by probing:
#
#     flat schema (str, List[str], Optional[float])   -> OK
#     Stage (flat)                                    -> OK
#     List[Stage] / List[FunctionAgentConfig]         -> always 429
#     method="json_schema"                            -> always 429
#     method="json_mode"                              -> request OK, but the model
#                                                        invents its own field names
#
# Two schemas in this project have that shape and both sit on the critical path:
# SystemOrchestration.agent_graph and StyleTransferPlan.stages. So we try the
# native structured path first (it still works for the flat schemas, and on
# providers without this limitation) and fall back to asking for JSON and
# validating it ourselves. The fallback is not a downgrade in rigour: the output
# is still validated against the same pydantic model, and this is already how
# every dynamically-created analysis agent in the graph returns its results.
# ---------------------------------------------------------------------------

def _schema_instruction(schema) -> str:
    """Render a pydantic model as an instruction block for the fallback path."""
    return (
        "\n\nReturn ONLY a single JSON object, with no prose and no code fence, "
        "conforming exactly to this JSON Schema. Use exactly these field names; "
        "do not add, rename, or nest fields differently:\n"
        + json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
    )


def _append_instruction(messages, text: str):
    """Append instruction text to the last human turn, whatever the input shape."""
    from langchain_core.messages import HumanMessage

    if isinstance(messages, str):
        return messages + text
    msgs = list(messages) if isinstance(messages, list) else [messages]
    last = msgs[-1]
    content = getattr(last, "content", None)
    if isinstance(content, str):
        msgs[-1] = type(last)(content=content + text)
    elif isinstance(content, list):
        msgs[-1] = type(last)(content=content + [{"type": "text", "text": text}])
    else:
        msgs.append(HumanMessage(content=text))
    return msgs


def _parse_json_object(text: str):
    """Extract the first complete JSON object/array from a model reply.

    Prefers src.utils.json_utils (which can repair malformed JSON via
    json_repair) but must not depend on it: the fallback path below is what
    keeps the pipeline running when the proxy rejects a tool schema, so it
    cannot itself be gated on an optional package being installed.
    """
    try:
        from src.utils.json_utils import extract_json_from_text as _repairing
        return _repairing(text)
    except ImportError:
        pass

    if not isinstance(text, str):
        raise ValueError("model reply is not a string")
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    start = next((i for i, c in enumerate(text) if c in "{["), None)
    if start is None:
        raise ValueError("no JSON object or array found in the reply")
    stack, in_str, esc = [], False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in "{[":
            stack.append(c)
        elif c in "}]":
            if not stack:
                raise ValueError("unbalanced brackets in the reply")
            stack.pop()
            if not stack:
                return json.loads(text[start:i + 1])
    raise ValueError("no complete JSON object found in the reply")


def _rejects_nested_schema(exc) -> bool:
    """The proxy's way of saying 'I cannot handle this tool schema'."""
    m = str(exc).lower()
    return ("429" in m or "\u9971\u548c" in m or "rate limit" in m
            or "400" in m or "invalid" in m or "schema" in m)


async def llm_structured_and_report(llm, schema, messages, file, max_retries: int = 3):
    """Invoke `llm` for a structured `schema`, falling back to prompted JSON.

    `llm` must be the UNBOUND model (do not call .with_structured_output yourself);
    this function decides which path to use.
    """
    try:
        return await llm_call_and_report(llm.with_structured_output(schema), messages,
                                         file=file, max_retries=1)
    except Exception as e:
        if not _rejects_nested_schema(e):
            raise
        print(f"[structured] native path rejected for {schema.__name__} "
              f"({str(e)[:100]}); falling back to prompted JSON.")

    last_err = None
    prompt = _append_instruction(messages, _schema_instruction(schema))
    for attempt in range(max_retries):
        try:
            resp = await llm_call_and_report(llm, prompt, file=file, max_retries=2)
            text = resp.content if hasattr(resp, "content") else str(resp)
            if isinstance(text, list):
                text = " ".join(p.get("text", "") if isinstance(p, dict) else str(p)
                                for p in text)
            return schema.model_validate(_parse_json_object(text))
        except Exception as e:
            last_err = e
            print(f"[structured] fallback attempt {attempt + 1}/{max_retries} for "
                  f"{schema.__name__} failed: {str(e)[:160]}")
            # Show the validator's complaint to the model on the next attempt.
            prompt = _append_instruction(
                prompt, f"\n\nYour previous reply was rejected: {str(e)[:400]}\n"
                        "Return corrected JSON matching the schema exactly.")
    raise RuntimeError(
        f"Structured output for {schema.__name__} failed on both the native and the "
        f"prompted-JSON path. Last error: {last_err}")
