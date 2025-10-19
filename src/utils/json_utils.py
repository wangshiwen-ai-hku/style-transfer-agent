import logging
import json
import json_repair

logger = logging.getLogger(__name__)

def extract_json_from_text(text: str):
    """
    Extract the first complete JSON object or array from a string and return the
    parsed Python object. This function is robust to surrounding chatty text and
    handles nested structures and string escapes.

    Raises ValueError if no complete JSON object/array is found or if parsing fails.
    """
    if not isinstance(text, str):
        raise ValueError("Response content is not a string")

    text = text.strip()

    # Fast path: the whole text is valid JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # Find the first opening brace/bracket
    start = None
    for idx, ch in enumerate(text):
        if ch == '{' or ch == '[':
            start = idx
            break

    if start is None:
        raise ValueError("No JSON object/array start found in text")

    stack = []
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == '{' or ch == '[':
            stack.append(ch)
            continue

        if ch == '}' or ch == ']':
            if not stack:
                raise ValueError("Unexpected closing bracket in text")
            opening = stack.pop()
            if (opening == '{' and ch != '}') or (opening == '[' and ch != ']'):
                raise ValueError("Mismatched brackets in text")

            # If stack is empty, we've closed the outermost JSON structure
            if not stack:
                candidate = text[start:i+1]
                try:
                    return json.loads(candidate)
                except Exception as e:
                    raise ValueError(f"Found JSON-like span but failed to parse: {e}")

    # No complete JSON structure found
    raise ValueError("No complete JSON object/array found in text")

def repair_json_output(content: str) -> str:
    """
    Repair and normalize JSON output.

    Args:
        content (str): String content that may contain JSON

    Returns:
        str: Repaired JSON string, or original content if not JSON
    """
    content = content.strip()
    if content.startswith(("{", "[")) or "```json" in content or "```ts" in content:
        try:
            # If content is wrapped in ```json code block, extract the JSON part
            if content.startswith("```json"):
                content = content.removeprefix("```json")

            if content.startswith("```ts"):
                content = content.removeprefix("```ts")

            if content.endswith("```"):
                content = content.removesuffix("```")

            # Try to repair and parse JSON
            repaired_content = json_repair.loads(content)
            return json.dumps(repaired_content, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"JSON repair failed: {e}")
    return content
