"""
TODO use image generation model to generate image
doubao_image
qwen_image
gemini-2.5-flash-image


# doubao
import os 
# 通过 pip install 'volcengine-python-sdk[ark]' 安装方舟SDK 
from volcenginesdkarkruntime import Ark 
from volcenginesdkarkruntime.types.images.images import SequentialImageGenerationOptions
from dotenv import load_dotenv
load_dotenv()
# 请确保您已将 API Key 存储在环境变量 ARK_API_KEY 中 
# 初始化Ark客户端，从环境变量中读取您的API Key 
client = Ark( 
    # 此为默认路径，您可根据业务所在地域进行配置 
    base_url="https://ark.cn-beijing.volces.com/api/v3", 
    # 从环境变量中获取您的 API Key。此为默认方式，您可根据需要进行修改 
    api_key=os.environ.get("ARK_API_KEY"), 
) 
 
imagesResponse = client.images.generate( 
    model="doubao-seedream-4-0-250828", 
    prompt="将图1的服装换为图2的服装",
    image=["https://ark-project.tos-cn-beijing.volces.com/doc_image/seedream4_imagesToimage_1.png", "https://ark-project.tos-cn-beijing.volces.com/doc_image/seedream4_imagesToimage_2.png"],
    size="2K",
    sequential_image_generation="disabled",
    response_format="url",
    watermark=True
) 
 
print(imagesResponse.data[0].url)


## qwen
import json
import os
from dashscope import MultiModalConversation
import base64
import mimetypes
import dashscope
from dotenv import load_dotenv
load_dotenv()

# 以下为中国（北京）地域url，若使用新加坡地域的模型，需将url替换为：https://dashscope-intl.aliyuncs.com/api/v1
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1/'

# ---用于 Base64 编码 ---
# 格式为 data:{mime_type};base64,{base64_data}
def encode_file(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError("不支持或无法识别的图像格式")

    try:
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(
                image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"
    except IOError as e:
        raise IOError(f"读取文件时出错: {file_path}, 错误: {str(e)}")


# 获取图像的 Base64 编码
# 调用编码函数，请将 "/path/to/your/image.png" 替换为您的本地图片文件路径，否则无法运行
# image = encode_file("/path/to/your/image.png")

style_image = encode_file("data/style/114.jpg")
content_image = encode_file("data/content/23.jpg")

messages = [
    {
        "role": "user",
        "content": [
            {"image": content_image},
            {"image": style_image},
            {"text": "将图2的风格迁移到图1的内容中"}
        ]
    }
]

# 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
# 若没有配置环境变量，请用百炼 API Key 将下行替换为：api_key="sk-xxx"
api_key = os.getenv("DASHSCOPE_API_KEY")


# 模型仅支持单轮对话，复用了多轮对话的接口
response = MultiModalConversation.call(
    api_key=api_key,
    model="qwen-image-edit",
    messages=messages,
    stream=False,
    watermark=False,
    negative_prompt=" "
)

if response.status_code == 200:
    # 如需查看完整响应，请取消下行注释
    # print(json.dumps(response, ensure_ascii=False))
    print("输出图像的URL:", response.output.choices[0].message.content[0]['image'])
else:
    print(f"HTTP返回码：{response.status_code}")
    print(f"错误码：{response.code}")
    print(f"错误信息：{response.message}")
    print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")


## gemini

genai_client = genai.Client(http_options=HttpOptions(api_version="v1"))

async def execute_stage_node(state: State) -> State:
  
    if not genai_client:
        log_warning("GenAI client not initialized. Skipping image generation.")
        return state

    aspect_ratio_to_use = "1:1"
    if state.get('provided_images'):
        try:
            # Use the first provided image to determine the aspect ratio for generated images.
            first_image_path = state['provided_images'][0]
            with Image.open(first_image_path) as img:
                width, height = img.size
            
            aspect_ratio_val = width / height

            valid_ratios = {
                "1:1": 1.0, "3:2": 1.5, "2:3": 2/3, "3:4": 0.75, "4:3": 4/3,
                "4:5": 0.8, "5:4": 1.25, "9:16": 9/16, "16:9": 16/9, "21:9": 21/9
            }

            closest_ratio_str = min(valid_ratios.keys(), key=lambda r: abs(valid_ratios[r] - aspect_ratio_val))
            log_state(f"Original image aspect ratio ~{aspect_ratio_val:.2f}. Using closest valid ratio: {closest_ratio_str}")
            aspect_ratio_to_use = closest_ratio_str
        except Exception as e:
            log_warning(f"Could not determine image aspect ratio: {e}. Defaulting to '1:1'.")
    
    plan = state['style_transfer_plan']
    generated_images_map = state['generated_images_map']
    
    # Ensure the output directory exists
    # output_dir = "out/agent_generated"
    output_dir = state['project_dir']
    os.makedirs(output_dir, exist_ok=True)

    for i, stage in enumerate(plan.stages):
        if stage.generated_image_tag in generated_images_map:
            log_state(f"Skipping Stage {i+1}: {stage.stage_name} (already generated)")
            continue
        log_agent("ExecuteStage", f"Executing Stage {i+1}: {stage.stage_name}")
        
        contents = []
        texts = stage.text_prompt
        image_tags = stage.required_image_tags
        for image_tag in image_tags:
            if image_tag in generated_images_map:
                image_path = generated_images_map[image_tag]
                with open(image_path, "rb") as image_file:
                    image_bytes = image_file.read()
                mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
                contents.append(Part.from_bytes(data=image_bytes, mime_type=mime_type))
        if isinstance(texts, str):
            texts = [texts]
        for text in texts:
            contents.append(text)
      
        # For all stages after the initial sketch, add a strong instruction to prioritize the style image
        # and ensure a significant change from the input image, as suggested by the user.
        if i > 0:
            final_instruction = "\n\n**CRITICAL REMINDER:** Your primary goal is to make the output image look much more like the *style image*. Ensure there is a significant, visible transformation from the input image you received. Do not just make minor tweaks."
            log_state(f"Adding critical reminder to prompt for Stage {i+1}")
            contents.append(final_instruction)

        log_tool("ImageGen", f"Generating image '{stage.generated_image_tag}'...")

        try:
            # Use stage-specific temperature if provided, otherwise default.
            temperature = stage.gen_temperature if stage.gen_temperature is not None else 0.7 # TODO: get from config
            log_debug(f"Using temperature for generation: {temperature}")

            response = genai_client.models.generate_content(
                model=MODEL_ID,
                contents=contents,
                config=GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=ImageConfig(
                        aspect_ratio=aspect_ratio_to_use,
                    ),
                    candidate_count=1,
                    temperature=temperature,
                ),
            )

            if response.candidates[0].finish_reason != FinishReason.STOP:
                reason = response.candidates[0].finish_reason
                log_error(f"Image generation failed for stage '{stage.stage_name}'. Reason: {reason}")
                continue

            generated_image_data = None
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    generated_image_data = part.inline_data.data
                    break
            
            if generated_image_data:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_image_path = os.path.join(output_dir, f"{stage.generated_image_tag}_{timestamp}.png")
                
                with open(new_image_path, "wb") as f:
                    f.write(generated_image_data)
"""
from typing import List
from PIL import Image

def image_generation_tool(text_prompt: str, image_paths: List[str], model: str = "gemini-2.5-flash-image") -> Image.Image:
    """
    TODO
    Generate an image using a selected image-generation backend and return the saved image path.

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
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"image": img} for img in normalized_images],
                    {"text": text_prompt},
                ],
            }
        ]

        response = MultiModalConversation.call(api_key=api_key, model="qwen-image-edit", messages=messages, stream=False, watermark=False, negative_prompt=" ")
        if getattr(response, "status_code", None) != 200:
            # try to extract error info if available
            code = getattr(response, "code", None)
            message = getattr(response, "message", None)
            raise RuntimeError(f"QWEN image generation failed: status={response.status_code} code={code} message={message}")

        # response.output.choices[0].message.content[0]['image'] is expected to be a data url or remote url
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
            import requests
            r = requests.get(img_field, timeout=30)
            r.raise_for_status()
            img = _save_bytes_to_png(r.content)
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

        ark_api_key = os.environ.get("ARK_API_KEY")
        if not ark_api_key:
            log_error("ARK_API_KEY environment variable is not set")
            raise RuntimeError("ARK_API_KEY environment variable is not set")

        client = Ark(
            base_url="https://ark.cn-beijing.volces.com/api/v3", 
            api_key=ark_api_key
            )

        # Ark example supports passing URLs; we will pass the normalized_images list (data URLs for locals)
        try:
            resp = client.images.generate(model="doubao-seedream-4-0-250828", prompt=text_prompt, image=normalized_images, size="2K", sequential_image_generation="disabled", response_format="url", watermark=False)
        except Exception as e:
            raise RuntimeError(f"Doubao image generation call failed: {e}") from e

        # try to extract URL or base64 from response
        try:
            data0 = resp.data[0]
        except Exception:
            raise RuntimeError("Unexpected response structure from Ark image generation")

        # prefer url
        url = getattr(data0, "url", None)
        b64_field = getattr(data0, "b64", None) or getattr(data0, "base64", None)
        if url:
            import requests
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            img = _save_bytes_to_png(r.content)
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
            from google.genai.types import HttpOptions, Part, GenerateContentConfig, ImageConfig, FinishReason
        except Exception as e:
            log_error(f"Google GenAI SDK not available: {e}")
            raise RuntimeError("Google GenAI SDK (google-genai) is required for Gemini backend") from e

        try:
            genai_client = genai.Client(http_options=HttpOptions(api_version="v1"))
        except Exception as e:
            log_error(f"Failed to initialize GenAI client: {e}")
            raise RuntimeError("Failed to initialize Google GenAI client") from e

        # determine aspect ratio from first image if available
        aspect_ratio_to_use = "1:1"
        if normalized_images:
            try:
                from PIL import Image
                # pick the first local image if present, otherwise download the first http
                first = normalized_images[0]
                if first.startswith("data:"):
                    header, b64 = first.split(',', 1)
                    img_bytes = base64.b64decode(b64)
                    img = Image.open(io.BytesIO(img_bytes))
                    width, height = img.size
                elif first.startswith("http"):
                    import requests
                    r = requests.get(first, timeout=30)
                    r.raise_for_status()
                    img = Image.open(io.BytesIO(r.content))
                    width, height = img.size
                else:
                    # data URLs were used for local files, but handle fallback
                    with Image.open(first) as imgf:
                        width, height = imgf.size
                aspect_ratio_val = width / height
                valid_ratios = {
                    "1:1": 1.0, "3:2": 1.5, "2:3": 2/3, "3:4": 0.75, "4:3": 4/3,
                    "4:5": 0.8, "5:4": 1.25, "9:16": 9/16, "16:9": 16/9, "21:9": 21/9
                }
                closest_ratio_str = min(valid_ratios.keys(), key=lambda r: abs(valid_ratios[r] - aspect_ratio_val))
                log_debug(f"Determined aspect ratio ~{aspect_ratio_val:.2f}, using {closest_ratio_str}")
                aspect_ratio_to_use = closest_ratio_str
            except Exception as e:
                log_warning(f"Could not determine image aspect ratio for Gemini: {e}. Defaulting to '1:1'.")

        # build contents: Parts for images + text prompt
        contents = []
        for p in normalized_images:
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
                    import requests
                    r = requests.get(p, timeout=30)
                    r.raise_for_status()
                    mime_type = r.headers.get("Content-Type", "image/png")
                    contents.append(Part.from_bytes(data=r.content, mime_type=mime_type))
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

        try:
            response = genai_client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=contents,
                config=GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=ImageConfig(aspect_ratio=aspect_ratio_to_use),
                    candidate_count=1,
                    temperature=temperature,
                ),
            )
        except Exception as e:
            log_error(f"GenAI generation call failed: {e}")
            raise RuntimeError(f"GenAI generation call failed: {e}") from e

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

    raise ValueError(f"Unsupported model: {model}")