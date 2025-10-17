from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage

from typing import List, Dict

import os
from pathlib import Path
from langchain_core.messages import ToolMessage, BaseMessage, HumanMessage, AIMessage
import base64
from PIL import Image

try:
    import cairosvg
except ImportError:
    cairosvg = None

def show_messages(update: list[BaseMessage]):
    print("\n\n" + "="*100 )
    for m in update:
        if isinstance(m, HumanMessage):
            # print only text
            if 'base64' in m.content:
                continue
            print(f"  [{m.type}] {m.name or ''}: {m.content[:800]}")
            continue
        if isinstance(m, AIMessage):
            print(f"  [{m.type}] {m.name or ''}: {m.content[:800]}")
        if hasattr(m, "tool_calls") and m.tool_calls:
            for tc in m.tool_calls:
                print(f"  [tool-call] {tc['name']}({tc['args']})")
        if isinstance(m, ToolMessage):
            print(f"  [tool-result] {m.content[:200]}")     
        

def svg_to_png(svg_path: str) -> str:
    """
    将 svg 文件转换成同名 png 文件，保存在同一目录下。

    :param svg_path: 输入的 svg 文件路径
    :return: 生成的 png 文件绝对路径；若失败则返回空字符串
    """
    if cairosvg is None:
        print("-> `cairosvg` is not installed. Skipping PNG conversion. "
              "To install, run: pip install cairosvg")
        return ""

    svg_file = Path(svg_path).expanduser().resolve()
    if not svg_file.is_file():
        print(f"-> SVG file not found: {svg_file}")
        return ""

    png_file = svg_file.with_suffix('.png')

    try:
        # 读取 SVG 内容
        svg_content = svg_file.read_text(encoding='utf-8')
        # 生成 PNG
        cairosvg.svg2png(
            bytestring=svg_content.encode('utf-8'),
            write_to=str(png_file),
            background_color="white"
        )
        print(f"-> PNG saved to: {png_file}")
        return str(png_file)
    except Exception as e:
        print(f"-> Failed to convert SVG to PNG: {e}")
        return ""

def convert_svg_to_png_base64(svg_code: str) -> str:
    """svg_code: the svg code of the path you want to draw.
        example: <svg width="100" height="100">
                    <path d="M10,10 L50,50 Q70,30 90,90 Z" fill="red"/>
                </svg>
    return: the base64 str of the png image of the picked paths.
    """
    try:
        # 使用cairosvg或其他SVG渲染库
        import cairosvg
        png_data = cairosvg.svg2png(bytestring=svg_code.encode())
        base64_str = base64.b64encode(png_data).decode('utf-8')
        return f"data:image/png;base64,{base64_str}"
    except ImportError:
        # 备选方案：返回SVG的base64
        return f"data:image/svg+xml;base64,{base64.b64encode(svg_code.encode()).decode()}"

def convert_pil_to_png_base64(pil_image) -> str:
    """pil_image: the PIL image you want to convert to base64.
        example: PIL Image
    return: the base64 str of the png image of the picked paths.
    """
    with open(pil_image, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded_image}"

# ----------------- 使用示例 -----------------

def create_multimodal_message(text: str="", image_data: List[str]=[], mm_hint="Here includes images inputs.."):
    contents = []
    if text:
        contents.append({"type": "text", "text": text})
    if image_data:
        for image in image_data:
            input_image = convert_pil_to_png_base64(image)
            contents.append({"type": "text", "text": f"{mm_hint}"})
            contents.append({"type": "image_url", "image_url": {"url": input_image}})
            
    return HumanMessage(content=contents)

def create_interleaved_multimodal_message(inputs=List[Dict[str, str]]):
    """
    input: List[Dict[str, str]]
    example: [{"type": "text", "text": "Hello"}, {"type": "image", "image": "image.png"}]
    """
    contents = []
    for item in inputs:
        if item["type"] == "text":
            contents.append({"type": "text", "text": item["text"]})
        elif item["type"] == "image":
            input_image = convert_pil_to_png_base64(item["image"])
            contents.append({"type": "image_url", "image_url": {"url": input_image}})
            
    return HumanMessage(content=contents)

if __name__ == "__main__":
    svg_path = input("请输入 SVG 文件路径：").strip()
    png_path = svg_to_png(svg_path)
    if png_path:
        print("转换成功，PNG 路径：", png_path)
