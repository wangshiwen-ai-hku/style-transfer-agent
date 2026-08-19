"""
"""
from typing import List, Callable, TypeVar, Any
from PIL import Image
import functools
# from dotenv import load_dotenv
# load_dotenv()


T = TypeVar('T')

def image_generation_tool(text_prompt: str, image_paths: List[str], image_clare_prompt: List[str]=None, target_ratio: float=1.0, model: str = "gemini-2.5-flash-image") -> Image.Image:
    """
    """
    import os
    import time
    import base64
    import mimetypes
    import tempfile
    import io

    from pathlib import Path
    # logging helpers (use project's colored logger when available)
    try:
        from src.utils.colored_logger import log_tool, log_debug, log_success, log_error, log_warning
    except Exception:
        # fallback no-op loggers if project logger isn't importable in this context
        def log_tool(*args, **kwargs):
            return None
        def log_debug(*args, **kwargs):
            return None
        def log_success(*args, **kwargs):
            return None
        def log_error(*args, **kwargs):
            return None
        def log_warning(*args, **kwargs):
            return None

    def _is_retryable_error(error: Exception) -> bool:
        """判断错误是否可重试"""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # 检查HTTP状态码（从错误消息中提取）
        import re
        status_code_match = re.search(r'status[_\s]*[=:]?\s*(\d{3})', error_str)
        if status_code_match:
            status_code = int(status_code_match.group(1))
            # 5xx和429（rate limit）可重试
            if status_code >= 500 or status_code == 429:
                return True
            # 4xx（除了429）通常不可重试
            if 400 <= status_code < 500:
                return False
        
        # 不可重试的错误类型
        non_retryable_keywords = [
            'authentication', 'auth', 'unauthorized', 'forbidden', '403', '401',
            'invalid', 'validation', 'bad request', '400', '404', 'not found',
            'unsupported', 'valueerror', 'file not found', 'filenotfounderror'
        ]
        
        # 可重试的错误类型
        retryable_keywords = [
            'timeout', 'connection', 'network', '500', '502', '503', '504',
            'rate limit', 'too many', '429', 'service unavailable',
            'internal server', 'temporary', 'retry', '503', '502', '504'
        ]
        
        # 检查不可重试的错误
        for keyword in non_retryable_keywords:
            if keyword in error_str or keyword in error_type:
                return False
        
        # 检查可重试的错误
        for keyword in retryable_keywords:
            if keyword in error_str or keyword in error_type:
                return True
        
        # 默认情况下，网络相关错误可重试，其他错误不可重试
        return 'requests' in error_type or 'connection' in error_type or 'timeout' in error_type

    def _retry_with_backoff(
        func: Callable[[], T],
        max_retries: int = None,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        operation_name: str = "operation"
    ) -> T:
        """
        带指数退避的重试机制
        
        Args:
            func: 要重试的函数（无参数）
            max_retries: 最大重试次数（None则从环境变量读取，默认3）
            initial_delay: 初始延迟（秒）
            max_delay: 最大延迟（秒）
            backoff_factor: 退避因子
            operation_name: 操作名称（用于日志）
        """
        if max_retries is None:
            max_retries = int(os.getenv('IMAGE_GEN_MAX_RETRIES', '3'))
        
        last_error = None
        delay = initial_delay
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    log_warning(f"{operation_name} failed (attempt {attempt}/{max_retries}), retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)
                
                result = func()
                if attempt > 0:
                    log_success(f"{operation_name} succeeded on retry attempt {attempt + 1}")
                return result
                
            except Exception as e:
                last_error = e
                if not _is_retryable_error(e):
                    log_error(f"{operation_name} failed with non-retryable error: {e}")
                    raise
                
                if attempt >= max_retries:
                    log_error(f"{operation_name} failed after {max_retries + 1} attempts: {e}")
                    raise RuntimeError(f"{operation_name} failed after {max_retries + 1} attempts: {e}") from e
        
        # 理论上不会到达这里
        raise RuntimeError(f"{operation_name} failed: {last_error}") from last_error

    def _encode_file_to_data_url(path: str) -> str:
        mime_type, _ = mimetypes.guess_type(path)
        if not mime_type or not mime_type.startswith("image/"):
            raise ValueError(f"Unsupported image path or mime type: {path}")
        with open(path, "rb") as f:
            b = f.read()
        return f"data:{mime_type};base64,{base64.b64encode(b).decode('utf-8')}"

    def _save_bytes_to_png(image_bytes: bytes, out_dir: str = None):
        """Return a PIL.Image created from raw image bytes. Do not leave temp files behind."""
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
            return img
        except Exception:
            # as a last resort, try to load via PIL's alternative
            from PIL import Image as PILImage
            img = PILImage.open(io.BytesIO(image_bytes)).convert("RGBA")
            return img


    # initial log about invocation
    try:
        log_tool("ImageGen", f"image_generation_tool called with model={model} and {len(image_paths or [])} image(s)")
    except Exception:
        pass

    # normalize image inputs: allow local paths or URLs
    normalized_images = []
    for p in image_paths or []:
        if isinstance(p, str) and (p.startswith("http://") or p.startswith("https://") or p.startswith("data:")):
            normalized_images.append(p)
        else:
            # treat as local file path
            if not os.path.exists(p):
                raise FileNotFoundError(f"Image path not found: {p}")
            normalized_images.append(_encode_file_to_data_url(p))

    # --- APIYI (default) -------------------------------------------------
    # Every model call goes through APIYI's OpenAI-compatible endpoint so that
    # the same code runs locally and on the GPU server, which can reach APIYI
    # but not Google/Aliyun/Volcengine. Set USE_APIYI=0 to fall back to the
    # native SDK paths below (kept intact, but no longer exercised).
    if os.environ.get("USE_APIYI", "1") != "0":
        from src.utils import apiyi as _apiyi
        resolved = _apiyi.resolve_generation_model(model)
        log_debug(f"Selected backend: APIYI (requested={model} -> {resolved})")
        img, usage = _apiyi.generate_image(
            text_prompt,
            image_paths=image_paths or [],
            image_clare_prompt=image_clare_prompt,
            target_ratio=target_ratio,
            model=resolved,
            return_usage=True,
        )
        if usage.get("target_ratio") and abs(usage["returned_ratio"] - usage["target_ratio"]) > 0.12:
            # The OpenAI-compatible schema has no aspect_ratio parameter, so the
            # ratio is only a prompt hint. Warn instead of resizing: a wrong ratio
            # is a real difference from the native-SDK results and should be seen.
            log_warning(f"aspect ratio drift: asked {usage['target_ratio']}, "
                        f"got {usage['returned_ratio']} ({img.size[0]}x{img.size[1]})")
        log_success(f"APIYI generated image {img.size}, tokens={usage.get('total_tokens')}")
        return img

    # --- QWEN via dashscope ---
    if "qwen" in model.lower() or model.lower().startswith("qwen"):
        log_debug(f"Selected backend: QWEN (model={model})")
        try:
            import requests
            import dashscope
            from dashscope import MultiModalConversation
        except Exception as e:
            raise RuntimeError("dashscope package is required for qwen model. Install it and set DASHSCOPE_API_KEY.") from e

        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            log_error("DASHSCOPE_API_KEY environment variable is not set")
            raise RuntimeError("DASHSCOPE_API_KEY environment variable is not set")

        # build messages
        contents = []
        for img, t in zip(normalized_images, image_clare_prompt):
            contents.append({"image": img})
            
        messages = [
            {
                "role": "user",
                "content": [
                    *contents,
                    {"text": text_prompt},
                ],
            }
        ]

        def _call_qwen_api():
            response = MultiModalConversation.call(
                api_key=api_key, model="qwen-image-edit-max", 
                messages=messages, 
                stream=False, 
                watermark=False, 
                negative_prompt=" ")
            if getattr(response, "status_code", None) != 200:
                # try to extract error info if available
                code = getattr(response, "code", None)
                message = getattr(response, "message", None)
                raise RuntimeError(f"QWEN image generation failed: status={response.status_code} code={code} message={message}")
            return response

        response = _retry_with_backoff(
            _call_qwen_api,
            operation_name="QWEN image generation API call"
        )

        try:
            choice = response.output.choices[0].message.content[0]
            img_field = choice.get("image") if isinstance(choice, dict) else None
        except Exception:
            img_field = None

        if not img_field:
            raise RuntimeError("QWEN response did not contain an image field")

        # handle data URL
        if isinstance(img_field, str) and img_field.startswith("data:"):
            head, b64 = img_field.split(",", 1)
            image_bytes = base64.b64decode(b64)
            img = _save_bytes_to_png(image_bytes)
            log_success(f"QWEN image returned as PIL.Image size={getattr(img, 'size', None)}")
            return img

        # otherwise treat as URL
        if isinstance(img_field, str) and img_field.startswith("http"):
            def _download_qwen_image():
                import requests
                r = requests.get(img_field, timeout=30)
                r.raise_for_status()
                return r.content
            
            image_bytes = _retry_with_backoff(
                _download_qwen_image,
                operation_name="QWEN image download"
            )
            img = _save_bytes_to_png(image_bytes)
            log_success(f"QWEN image downloaded and returned as PIL.Image size={getattr(img, 'size', None)}")
            return img

        raise RuntimeError("Unsupported QWEN image field format")

    # --- DOUBAO / Volcengine Ark ---
    if "doubao" in model.lower() or "seedream" in model.lower():
        log_debug(f"Selected backend: Doubao/Volcengine Ark (model={model})")
        try:
            from volcenginesdkarkruntime import Ark
        except Exception as e:
            raise RuntimeError("volcenginesdkarkruntime is required for doubao model. Install it and set ARK_API_KEY.") from e

        ark_api_key = os.environ.get("ARK_IMAGE_API_KEY")
        if not ark_api_key:
            log_error("ARK_API_KEY environment variable is not set")
            raise RuntimeError("ARK_API_KEY environment variable is not set")

        client = Ark(
            base_url="https://ark.cn-beijing.volces.com/api/v3", 
            api_key=ark_api_key
            )
        # client = Ark(
        #     base_url="https://ark.ap-southeast.bytepluses.com/",
        #     api_key=ark_api_key
        # )
    
        doubao_aspect_ratio_to_size = {"1:1": "2048x2048", "4:3": "2304x1728", "3:4": "1728x2304", "16:9": "2560x1440", "9:16": "1440x2560", "3:2": "2496x1664", "2:3": "1664x2496", "21:9": "3024x1296"}
        doubao_aspect_ratio = {"1:1": 1.0, "4:3": 2304/1728, "3:4": 1728/2304, "16:9": 2560/1440, "9:16": 1440/2560, "3:2": 2496/1664, "2:3": 1664/2496, "21:9": 3024/1296}
        closest_ratio_str = min(doubao_aspect_ratio.keys(), key=lambda r: abs(doubao_aspect_ratio[r] - target_ratio))
        closest_size_to_use = doubao_aspect_ratio_to_size[closest_ratio_str]
        # Ark example supports passing URLs; we will pass the normalized_images list (data URLs for locals)
        def _call_doubao_api():
            try:
                resp = client.images.generate(model="doubao-seedream-4-0-250828", prompt=text_prompt, image=normalized_images, size=closest_size_to_use, sequential_image_generation="disabled", response_format="url", watermark=False)
                return resp
            except Exception as e:
                raise RuntimeError(f"Doubao image generation call failed: {e}") from e
        
        resp = _retry_with_backoff(
            _call_doubao_api,
            operation_name="Doubao image generation API call"
        )

        # try to extract URL or base64 from response
        try:
            data0 = resp.data[0]
        except Exception:
            raise RuntimeError("Unexpected response structure from Ark image generation")

        # prefer url
        url = getattr(data0, "url", None)
        b64_field = getattr(data0, "b64", None) or getattr(data0, "base64", None)
        if url:
            def _download_doubao_image():
                import requests
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                return r.content
            
            image_bytes = _retry_with_backoff(
                _download_doubao_image,
                operation_name="Doubao image download"
            )
            img = _save_bytes_to_png(image_bytes)
            log_success(f"Doubao image downloaded and returned as PIL.Image size={getattr(img, 'size', None)}")
            return img
        if b64_field:
            # if b64_field is a str or bytes containing base64
            if isinstance(b64_field, bytes):
                image_bytes = base64.b64decode(b64_field)
            else:
                # sometimes the SDK returns raw base64 string
                image_bytes = base64.b64decode(b64_field)
            img = _save_bytes_to_png(image_bytes)
            log_success(f"Doubao image decoded and returned as PIL.Image size={getattr(img, 'size', None)}")
            return img

        raise RuntimeError("Could not find generated image in Ark response")

    # --- GEMINI / GENAI fallback ---
    if "gemini" in model.lower() or "genai" in model.lower():
        log_debug(f"Selected backend: Gemini/GenAI (model={model})")

        try:
            from google import genai
            from google.genai import types
            from google.genai.types import HttpOptions, Part, FinishReason
        except Exception as e:
            log_error(f"Google GenAI SDK not available: {e}")
            raise RuntimeError("Google GenAI SDK (google-genai) is required for Gemini backend") from e

        try:
            if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "False") == "True":
                # print(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
                os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = "True"
                genai_client = genai.Client(http_options=HttpOptions(api_version="v1"))
            else:
                genai_client = genai.Client()
                
        except Exception as e:
            log_error(f"Failed to initialize GenAI client: {e}")
            raise RuntimeError("Failed to initialize Google GenAI client") from e

        # determine aspect ratio from first image if available
        aspect_ratio_to_use = "1:1"
     
        aspect_ratio_val = target_ratio
        valid_ratios = {
                    "1:1": 1.0, "3:2": 1.5, "2:3": 2/3, "3:4": 0.75, "4:3": 4/3,
                    "4:5": 0.8, "5:4": 1.25, "9:16": 9/16, "16:9": 16/9, "21:9": 21/9
                }
        closest_ratio_str = min(valid_ratios.keys(), key=lambda r: abs(valid_ratios[r] - aspect_ratio_val))
        log_debug(f"Determined aspect ratio ~{aspect_ratio_val:.2f}, using {closest_ratio_str}")
        aspect_ratio_to_use = closest_ratio_str

        # build contents: Parts for images + text prompt
        contents = []
        for p, t in zip(normalized_images, image_clare_prompt):
            if t:
                contents.append(t)
            if isinstance(p, str) and p.startswith("data:"):
                try:
                    header, b64 = p.split(',', 1)
                    img_bytes = base64.b64decode(b64)
                    mime_type = header.split(';', 1)[0].split(':', 1)[1]
                    contents.append(Part.from_bytes(data=img_bytes, mime_type=mime_type))
                except Exception as e:
                    log_warning(f"Failed to decode data URL image: {e}")
            elif isinstance(p, str) and p.startswith("http"):
                try:
                    def _download_image_for_gemini():
                        import requests
                        r = requests.get(p, timeout=30)
                        r.raise_for_status()
                        return r.content, r.headers.get("Content-Type", "image/png")
                    
                    img_bytes, mime_type = _retry_with_backoff(
                        _download_image_for_gemini,
                        operation_name=f"Gemini input image download ({p})",
                        max_retries=2  # 输入图片下载失败重试次数少一些
                    )
                    contents.append(Part.from_bytes(data=img_bytes, mime_type=mime_type))
                except Exception as e:
                    log_warning(f"Could not download image {p}: {e}")
            else:
                # local path (should have been encoded earlier, but handle raw path)
                try:
                    with open(p, 'rb') as f:
                        b = f.read()
                    mime_type = 'image/png' if str(p).lower().endswith('.png') else 'image/jpeg'
                    contents.append(Part.from_bytes(data=b, mime_type=mime_type))
                except Exception as e:
                    log_warning(f"Could not read local image {p}: {e}")

            

        if text_prompt:
            contents.append(text_prompt)

        temperature = float(os.getenv('IMAGE_GEN_TEMPERATURE', 0.7))

        def _call_gemini_api():
            try:
                # Use types.GenerateContentConfig and types.ImageConfig as per official documentation
                response = genai_client.models.generate_content(
                    model="gemini-2.5-flash-image",
                    # model="gemini-3-pro-image-preview",
                    # model="gemini-3.1-flash-image-preview",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        candidate_count=1,
                        temperature=temperature,
                        image_config=types.ImageConfig(
                            aspect_ratio=aspect_ratio_to_use,
                        ),
                    ),
                )
                log_debug(f"Successfully generated image with aspect_ratio={aspect_ratio_to_use}")
                return response
            except Exception as e:
                log_error(f"GenAI generation call failed: {e}")
                raise RuntimeError(f"GenAI generation call failed: {e}") from e

        response = _retry_with_backoff(
            _call_gemini_api,
            operation_name="Gemini image generation API call"
        )

        if not response.candidates or response.candidates[0].finish_reason != FinishReason.STOP:
            reason = response.candidates[0].finish_reason if response.candidates else 'No candidates'
            log_error(f"Image generation failed for Gemini. Reason: {reason}")
            raise RuntimeError(f"Image generation failed for Gemini. Reason: {reason}")

        generated_image_data = None
        for part in response.candidates[0].content.parts:
            if getattr(part, 'inline_data', None):
                generated_image_data = part.inline_data.data
                break

        if generated_image_data:
            img = _save_bytes_to_png(generated_image_data)
            log_success(f"Gemini image returned as PIL.Image size={getattr(img, 'size', None)}")
            return img
        else:
            log_error('No image data found in Gemini response')
            raise RuntimeError('No image data found in Gemini response')

    if "openai" in model.lower() or "gpt" in model.lower() or "gpt-image" in model.lower() or "gptimage" in model.lower():
        log_debug(f"Selected backend: OpenAI/Azure (model={model})")
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("openai is required for openai model. Install it and set AZURE_API_KEY.") from e

        openai_api_key = os.environ.get("AZURE_API_KEY")
        azure_endpoint = os.environ.get("AZURE_ENDPOINT", "https://routinetask.cognitiveservices.azure.com/openai/v1/")
        deployment_name = os.environ.get("AZURE_DEPLOYMENT", "xinzhewei-gptimage1")
        api_version = os.environ.get("AZURE_API_VERSION", "preview")
        
        if not openai_api_key:
            log_error("AZURE_API_KEY environment variable is not set")
            raise RuntimeError("AZURE_API_KEY environment variable is not set")

        # Initialize OpenAI client with Azure endpoint
        client = OpenAI(
            base_url=azure_endpoint,
            api_key=openai_api_key,
            default_headers={"api_version": api_version}
        )

        # Map target_ratio to size options
        # OpenAI/Azure supports: "1024x1024", "1024x1536", "1536x1024"
        size_options = {
            "1024x1024": 1.0,
            "1024x1536": 1024/1536,  # ~0.67 (portrait)
            "1536x1024": 1536/1024,  # 1.5 (landscape)
        }
        closest_size = min(size_options.keys(), key=lambda s: abs(size_options[s] - target_ratio))
        log_debug(f"Determined size ~{target_ratio:.2f}, using {closest_size}")

        # Prepare images parameter if multiple images provided
        # OpenAI images.edit requires file objects with proper MIME types
        # Format: (filename, file_object, content_type) tuple or file object with name attribute
        image_files = []
        opened_files = []  # Track opened files for cleanup
        if normalized_images:
            try:
                # Helper function to determine MIME type from extension
                def _get_mime_type_from_path(path: str) -> str:
                    ext = os.path.splitext(path.lower())[1]
                    mime_map = {
                        '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                        '.png': 'image/png',
                        '.webp': 'image/webp',
                        '.gif': 'image/gif',
                    }
                    return mime_map.get(ext, 'image/png')  # default to png
                
                # Convert normalized_images to file objects with proper MIME types
                # OpenAI SDK needs file objects with 'name' attribute for MIME type detection
                for img_data_url in normalized_images:
                    if isinstance(img_data_url, str) and img_data_url.startswith("data:"):
                        # Extract MIME type from data URL header
                        header, b64 = img_data_url.split(',', 1)
                        mime_type = header.split(';')[0].split(':')[1] if ':' in header else 'image/png'
                        img_bytes = base64.b64decode(b64)
                        # Create BytesIO with name attribute for MIME type detection
                        file_obj = io.BytesIO(img_bytes)
                        # Set name attribute based on MIME type so SDK can detect it
                        ext = mime_type.split('/')[1] if '/' in mime_type else 'png'
                        file_obj.name = f"image.{ext}"
                        image_files.append(file_obj)
                    elif isinstance(img_data_url, str) and img_data_url.startswith("http"):
                        def _download_image_for_openai():
                            import requests
                            r = requests.get(img_data_url, timeout=30)
                            r.raise_for_status()
                            return r.content, r.headers.get("Content-Type", "image/png")
                        
                        try:
                            img_bytes, mime_type = _retry_with_backoff(
                                _download_image_for_openai,
                                operation_name=f"OpenAI input image download ({img_data_url})",
                                max_retries=2  # 输入图片下载失败重试次数少一些
                            )
                            if not mime_type.startswith("image/"):
                                mime_type = "image/png"
                            # Extract filename from URL if possible
                            filename = os.path.basename(img_data_url.split('?')[0]) or "image.png"
                            file_obj = io.BytesIO(img_bytes)
                            file_obj.name = filename  # Set name attribute for MIME type detection
                            image_files.append(file_obj)
                        except Exception as e:
                            log_warning(f"Failed to download input image {img_data_url}: {e}, skipping")
                    else:
                        # Local file path - open() returns file object with name attribute
                        if os.path.exists(img_data_url):
                            f = open(img_data_url, "rb")
                            opened_files.append(f)
                            image_files.append(f)
                
                if image_files:
                    log_debug(f"Prepared {len(image_files)} image(s) with MIME types for OpenAI API")
            except Exception as e:
                log_warning(f"Failed to prepare images for OpenAI API: {e}, proceeding with text-only prompt")
                # Clean up opened files on error
                for f in opened_files:
                    try:
                        f.close()
                    except:
                        pass
                opened_files = []
                image_files = []

        # Call the API
        def _call_openai_api():
            try:
                # Build the generate call
                generate_kwargs = {
                    "model": deployment_name,
                    "prompt": text_prompt + "keep the image to be edited's content/structure/position/oritention.",
                    "n": 1,
                    "size": closest_size,
                    "stream": False,  # Use non-streaming by default for simplicity
                }
                
                # Add image parameter if available
                # OpenAI images.edit accepts file objects with 'name' attribute for MIME type detection
                # For multiple images, pass as list (API may support this for fusion)
                if image_files:
                    # Pass file objects directly - SDK will detect MIME type from file.name attribute
                    if len(image_files) == 1:
                        generate_kwargs["image"] = image_files[0]
                    else:
                        # Multiple images: pass as list
                        generate_kwargs["image"] = image_files
                    log_debug(f"Passing {len(image_files)} reference image(s) to OpenAI API")
                generate_kwargs["input_fidelity"] = 'high'
                if not image_files:
                    response = client.images.generate(**generate_kwargs)
                    return response
                response = client.images.edit(**generate_kwargs)
                return response
            except Exception as e:
                log_error(f"OpenAI/Azure image generation call failed: {e}")
                raise RuntimeError(f"OpenAI/Azure image generation call failed: {e}") from e
        
        try:
            response = _retry_with_backoff(
                _call_openai_api,
                operation_name="OpenAI/Azure image generation API call"
            )
        except Exception as e:
            # Clean up opened files on error
            for f in opened_files:
                try:
                    f.close()
                except:
                    pass
            raise
        finally:
            # Clean up opened files after API call (in finally block to ensure cleanup)
            for f in opened_files:
                try:
                    f.close()
                except:
                    pass

        # Extract image from response
        try:
            # Non-streaming response structure
            if hasattr(response, 'data') and len(response.data) > 0:
                data_item = response.data[0]
                # Try to get b64_json first, then url
                if hasattr(data_item, 'b64_json') and data_item.b64_json:
                    image_bytes = base64.b64decode(data_item.b64_json)
                    img = _save_bytes_to_png(image_bytes)
                    log_success(f"OpenAI/Azure image returned as PIL.Image size={getattr(img, 'size', None)}")
                    return img
                elif hasattr(data_item, 'url') and data_item.url:
                    def _download_openai_image():
                        import requests
                        r = requests.get(data_item.url, timeout=30)
                        r.raise_for_status()
                        return r.content
                    
                    image_bytes = _retry_with_backoff(
                        _download_openai_image,
                        operation_name="OpenAI/Azure image download"
                    )
                    img = _save_bytes_to_png(image_bytes)
                    log_success(f"OpenAI/Azure image downloaded and returned as PIL.Image size={getattr(img, 'size', None)}")
                    return img
                else:
                    raise RuntimeError("OpenAI/Azure response did not contain image data (b64_json or url)")
            else:
                raise RuntimeError("OpenAI/Azure response did not contain data array")
        except Exception as e:
            log_error(f"Failed to extract image from OpenAI/Azure response: {e}")
            raise RuntimeError(f"Failed to extract image from OpenAI/Azure response: {e}") from e
        
    raise ValueError(f"Unsupported model: {model}")