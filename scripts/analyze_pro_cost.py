import tiktoken

def calculate_pro_cost(thinking_text, num_input_images, output_resolution="2K"):
    # 1. 估算文本 Token (使用主流的 cl100k_base 编码器近似计算)
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        text_tokens = len(encoding.encode(thinking_text))
    except ImportError:
        # 如果没有安装 tiktoken，使用粗略估算 (英文字符数 / 4)
        text_tokens = int(len(thinking_text) / 4)
        print("Note: 'tiktoken' not found. Using rough approximation for tokens.")

    # 假设每张输入图像被编码为固定的 Token 数量 (大多数视觉模型一张图约占 258 到 1000 tokens)
    # 这里我们保守假设一张图占用 258 tokens
    tokens_per_image = 258 
    input_image_tokens = num_input_images * tokens_per_image
    
    # 假设用户的原始 prompt 指令约 50 tokens
    input_text_tokens = 50 
    total_input_tokens = input_image_tokens + input_text_tokens

    # 2. 定义定价 ($ / 1M tokens 或 $ / image)
    PRICE_INPUT_1M = 2.00
    PRICE_OUTPUT_1M = 12.00
    PRICE_IMAGE_2K = 0.134
    PRICE_IMAGE_4K = 0.240

    # 3. 计算成本
    input_cost = (total_input_tokens / 1_000_000) * PRICE_INPUT_1M
    output_text_cost = (text_tokens / 1_000_000) * PRICE_OUTPUT_1M
    
    if output_resolution == "4K":
        image_cost = PRICE_IMAGE_4K
    else:
        image_cost = PRICE_IMAGE_2K
        
    total_cost = input_cost + output_text_cost + image_cost

    # 4. 打印结果
    print(f"--- Cost Estimation for Pro Generation ({output_resolution}) ---")
    print(f"Input Tokens: {total_input_tokens} (Cost: ${input_cost:.5f})")
    print(f"Output Text Tokens (Thinking): {text_tokens} (Cost: ${output_text_cost:.5f})")
    print(f"Output Image Cost: ${image_cost:.3f}")
    print(f"Total Cost: ${total_cost:.5f}\n")
    
    return total_cost

# 填入你提取的 thinking 文本
thinking_output = """
Analyzing Visual Concepts
I'm working on combining elements from two sources: a woman from one, and a line art style from another. My current direction involves translating the woman into this new line art style and introducing additional pink abstract shapes and floating leaves against a pale pink backdrop.
Refining Stylistic Integration
I'm now focused on the integration of content and style. The woman's form will be redrawn in the black continuous line art of the reference image. The plan is to shade the face, neck, and hair using the pale pink abstract blocks, with floating pink-outlined leaves on the pink background.
Analyzing Style Transfer Success
I'm assessing the execution of the style transfer, checking how well the content's subject is rendered in the specified artistic style. I'm noting the presence of the watercolor accents and leaf motifs as per the style image. I'm focused on the match between the instructions and the result.
Evaluating Instruction Adherence
I've examined the outcome against the original user input, specifically the continuous line art style and the watercolor accents. I confirm the leaf motifs are present as requested and the result follows the style image exactly. I will proceed accordingly.
"""

# 运行计算 (2张输入图，测试 2K 和 4K 两种输出)
calculate_pro_cost(thinking_output, num_input_images=2, output_resolution="2K")
calculate_pro_cost(thinking_output, num_input_images=2, output_resolution="4K")