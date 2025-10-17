from langgraph.graph import StateGraph
from .schema import State, StyleTransferPlan, ComprehensiveStyleAnalysis, StyleTransferAnalysis, Message, MODALITY, Reflection, Stage
from src.config.manager import config
from langchain.chat_models import init_chat_model
from dataclasses import asdict
from src.utils.multi_modal_utils import create_interleaved_multimodal_message, create_multimodal_message
from src.utils.colored_logger import get_colored_logger, init_default_logger, log_agent, log_state, log_tool, log_success, log_warning, log_error, log_debug
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import END
import base64
import os
from datetime import datetime
from google import genai
from google.genai.types import Part, GenerateContentConfig, ImageConfig, FinishReason
from dotenv import load_dotenv
from google.genai.types import HttpOptions
from PIL import Image
import json
# from src.utils.image_processing import canny_edge_detection

MAX_REFLECTIONS = 3

load_dotenv()

# Initialize colored logger for enhanced visual debugging
init_default_logger(__name__)
# Use the global logger instance for consistency
logger = None

graph = StateGraph(State)

# Load drawing process examples from the text file
try:
    with open("src/agent/drawing_processes.txt", "r") as f:
        drawing_process_examples = f.read()
except FileNotFoundError:
    log_warning("`src/agent/drawing_processes.txt` not found. Using empty examples.")
    drawing_process_examples = ""

try:
    with open("src/agent/stage_examples.txt", "r") as f:
        stage_examples = f.read()
except FileNotFoundError:
    log_warning("`src/agent/stage_examples.txt` not found. Using empty examples.")
    stage_examples = ""

describe_agent_config = config.get_agent_config('describe_agent', 'core')

describe_llm = init_chat_model(**asdict(describe_agent_config.model))
describe_llm_prompt = describe_agent_config.prompt
print(describe_llm_prompt)
describe_sys_message = SystemMessage(content=describe_llm_prompt)

plan_agent_config = config.get_agent_config('plan_agent', 'core')
plan_llm = init_chat_model(**asdict(plan_agent_config.model))
plan_llm_prompt = plan_agent_config.prompt.format(
    # drawing_process_examples=drawing_process_examples
    stage_examples=stage_examples
)
plan_sys_message = SystemMessage(content=plan_llm_prompt)

# execute_agent_config = config.get_agent_config('execute_agent', 'core')
MODEL_ID = "gemini-2.5-flash-image"
PROJECT_ID = os.environ.get("GOOGLE_PROJECT_ID", None)
LOCATION = "us-central1"

genai_client = None
if not PROJECT_ID:
        log_warning("GOOGLE_CLOUD_PROJECT environment variable not set. Skipping GenAI client initialization.")
else:
    try:
        genai_client = genai.Client(http_options=HttpOptions(api_version="v1"))
    except Exception as e:
        log_error(f"Error initializing Google GenAI Client: {e}")
        log_warning("Please ensure you have authenticated with Google Cloud (e.g., `gcloud auth application-default login`).")


async def comprehensive_style_analysis_node(state: State) -> State:
    """
    Analyzes the style image to extract its core characteristics.
    """
    structured_llm = describe_llm.with_structured_output(ComprehensiveStyleAnalysis)
    
    messages = [
        describe_sys_message,
        HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(open(state['style_image_path'], 'rb').read()).decode()}"}},
            {"type": "text", "text": "Please provide a comprehensive, multi-dimensional analysis of the provided style image. Analyze its type, potential author context, creation techniques, and both its general and personal stylistic attributes."},
        ])
    ]

    response = await structured_llm.ainvoke(messages)
    state['comprehensive_style_analysis'] = response
    log_agent("ComprehensiveStyleAnalysis", "Style analysis completed successfully")
    log_debug(f"Analysis response: {response}")
    
    output_dir = state['project_dir']
    os.makedirs(output_dir, exist_ok=True)
    analysis_path = os.path.join(output_dir, "comprehensive_style_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(response.model_dump(), f, indent=4)
    log_success(f"Comprehensive style analysis saved to {analysis_path}")
    return state
    

async def style_transfer_analysis_node(state: State) -> State:
    """
    Analyzes how to transfer the style to the content image.
    """
    # This node will require its own LLM and prompt, which we'll configure in config.yaml
    # For now, let's assume a 'style_transfer_agent' is defined.
    transfer_agent_config = config.get_agent_config('style_transfer_analysis_agent', 'core') # Reusing plan_agent for now
    transfer_llm = init_chat_model(**asdict(transfer_agent_config.model))
    structured_llm = transfer_llm.with_structured_output(StyleTransferAnalysis)

    # We need a dedicated prompt for this agent.
    transfer_prompt = transfer_agent_config.prompt
    transfer_sys_message = SystemMessage(content=transfer_prompt)

    comprehensive_analysis = state['comprehensive_style_analysis']
    
    messages = [
        transfer_sys_message,
        HumanMessage(content=[
            {"type": "text", "text": f"Here is the comprehensive analysis of the style image:\n\n{json.dumps(comprehensive_analysis.model_dump(), indent=2)}"},
            {"type": "text", "text": "Here is the content image:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(open(state['content_image_path'], 'rb').read()).decode()}"}},
            {"type": "text", "text": "Please provide the style transfer analysis for the content image."},
        ])
    ]

    response = await structured_llm.ainvoke(messages)
    state['style_transfer_analysis'] = response
    log_agent("StyleTransferAnalysis", "Style transfer analysis completed successfully")
    log_debug(f"Transfer analysis response: {response}")
    
    output_dir = state['project_dir']
    os.makedirs(output_dir, exist_ok=True)
    analysis_path = os.path.join(output_dir, "style_transfer_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(response.model_dump(), f, indent=4)
    log_success(f"Style transfer analysis saved to {analysis_path}")
    return state


async def plan_stages_node(state: State) -> State:
    """
    Generates a multi-stage style transfer plan based on the image analysis.
    """
    structured_llm = plan_llm.with_structured_output(StyleTransferPlan)
    style_analysis = state['comprehensive_style_analysis']
    transfer_analysis = state['style_transfer_analysis']

    prompt = f"""
    Based on the following analysis, create a multi-stage style transfer plan. 

    **Comprehensive Style Analysis:**
    {json.dumps(style_analysis.model_dump(), indent=2)}

    **Style Transfer Analysis (Instructions for this specific content image):**
    {json.dumps(transfer_analysis.model_dump(), indent=2)}

    The plan should have multiple stages. Each stage must define a `generated_image_tag` and the `context_multimodal_messages` required to generate it.
    The `context_multimodal_messages` is a list of text prompts and image references (using their tags).
    Available initial image tags are: `content_image` and `style_image`.
    Subsequent stages can reference images generated in prior stages by their `generated_image_tag`.
    """

    messages = [plan_sys_message, HumanMessage(content=prompt)]
    response = await structured_llm.ainvoke(messages)
    state['style_transfer_plan'] = response
    log_agent("PlanStages", "Style transfer plan generated successfully")
    # log_debug(f"Plan response: {response}")
    for i, stage in enumerate(response.stages):
        stage_dict = stage.model_dump()
        log_state(f"Stage {i+1}: {stage.stage_name}")
        for key, value in stage_dict.items():
            log_debug(f"\t{key}: {value}")
    output_dir = state['project_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Save the prompt used for planning
    plan_prompt_path = os.path.join(output_dir, "plan_stages_prompt.txt")
    with open(plan_prompt_path, "w") as f:
        f.write("=== SYSTEM MESSAGE ===\n")
        f.write(plan_sys_message.content)
        f.write("\n\n=== USER PROMPT ===\n")
        f.write(prompt)
    log_success(f"Plan stages prompt saved to {plan_prompt_path}")

    plan_path = os.path.join(output_dir, "style_transfer_plan.json")
    with open(plan_path, "w") as f:
        json.dump(response.model_dump(), f, indent=4)
    log_success(f"Style transfer plan saved to {plan_path}")
    return state


async def execute_stage_node(state: State) -> State:
    """
    Executes each stage of the style transfer plan to generate images.
    """
    if not genai_client:
        log_warning("GenAI client not initialized. Skipping image generation.")
        return state

    try:
        with Image.open(state['content_image_path']) as img:
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
        aspect_ratio_to_use = "1:1"
        
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

                log_success(f"Generated image saved to: {new_image_path}")
                generated_images_map[stage.generated_image_tag] = new_image_path
            else:
                log_warning(f"No image data found in response for stage '{stage.stage_name}'.")

        except Exception as e:
            log_error(f"An error occurred during image generation for stage '{stage.stage_name}': {e}")
            continue

    state['generated_images_map'] = generated_images_map

    # Save prompts used for each stage
    for i, stage in enumerate(plan.stages):
        if stage.generated_image_tag in generated_images_map:
            stage_prompt_path = os.path.join(output_dir, f"stage_{i+1}_{stage.generated_image_tag}_prompt.txt")
            with open(stage_prompt_path, "w") as f:
                f.write("=== STAGE PROMPT ===\n")
                f.write(f"Stage Name: {stage.stage_name}\n")
                f.write(f"Generated Image Tag: {stage.generated_image_tag}\n")
                f.write(f"Text Prompt: {stage.text_prompt}\n")
                f.write(f"Required Image Tags: {stage.required_image_tags}\n")
                f.write(f"Generation Temperature: {stage.gen_temperature}\n")
            log_success(f"Stage {i+1} prompt saved to {stage_prompt_path}")

    return state


async def reflect_node(state: State) -> State:
    """
    Reflects on the generated image, critiques it, and plans the next stage if necessary.
    """
    log_agent("Reflection", "Reflecting on the result")
    
    # 1. Configure reflection and planning agents
    reflect_agent_config = config.get_agent_config('reflect_agent', 'core')
    reflect_llm = init_chat_model(**asdict(reflect_agent_config.model))
    structured_reflect_llm = reflect_llm.with_structured_output(Reflection)
    reflect_sys_message = SystemMessage(content=reflect_agent_config.prompt)
    stage_agent_config = config.get_agent_config('stage_agent', 'core')
    stage_llm = init_chat_model(**asdict(stage_agent_config.model))
    structured_stage_llm = stage_llm.with_structured_output(Stage)
    stage_sys_message = SystemMessage(content=stage_agent_config.prompt)

    # 2. Get the last generated image
    last_stage_tag = state['style_transfer_plan'].stages[-1].generated_image_tag
    last_image_path = state['generated_images_map'].get(last_stage_tag)

    if not last_image_path:
        log_warning("Could not find the last generated image to reflect upon. Stopping.")
        # Create a default reflection that stops the process
        stop_reflection = Reflection(critique="Stopping due to missing image.", content_hold_score=0.0, style_transfer_score=0.0, is_satisfied=True)
        state['reflections'].append(stop_reflection)
        return state

    # 3. Perform reflection
    log_state(f"Reflecting on image: {last_image_path}")
    
    # Add the detailed style analysis to the reflection prompt.
    style_analysis_text = json.dumps(state['comprehensive_style_analysis'].model_dump(), indent=2)
    similar_regions_transfer_detail = state['style_transfer_analysis'].similar_regions_transfer_detail


    reflection_prompt_instruction = f"""
Please act as an art critic and critique the `generated_image`.
Your critique should assess how well the style from the `style_image` was transferred to the `content_image`.
You MUST use the following detailed analysis as a reference for your critique.

**Attention on Similar Regions Transfer:**
{similar_regions_transfer_detail}

 Based on your detailed critique, provide a `content_hold_score` and a `style_transfer_score` (both from 0.0 to 10.0), and then decide if you are `is_satisfied` with the result.
"""
    
    reflection_prompt = HumanMessage(content=[
        {"type": "text", "text": reflection_prompt_instruction},
        {"type": "text", "text": "Style Image:"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(open(state['style_image_path'], 'rb').read()).decode()}"}},
        {"type": "text", "text": "Content Image:"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(open(state['content_image_path'], 'rb').read()).decode()}"}},
        {"type": "text", "text": "Generated Image:"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(open(last_image_path, 'rb').read()).decode()}"}},
    ])
    
    # Save reflection prompt before invoking
    output_dir = state['project_dir']
    reflection_prompt_path = os.path.join(output_dir, f"reflection_{state['reflection_count'] + 1}_prompt.txt")
    with open(reflection_prompt_path, "w") as f:
        f.write("=== SYSTEM MESSAGE ===\n")
        f.write(reflect_sys_message.content)
        f.write("\n\n=== REFLECTION INSTRUCTION ===\n")
        f.write(reflection_prompt_instruction)

    reflection: Reflection = await structured_reflect_llm.ainvoke([reflect_sys_message, reflection_prompt])
    state['reflections'].append(reflection)
    state['reflection_count'] += 1

    log_state("Reflection Results:")
    log_debug(f"  Critique: {reflection.critique}")
    log_debug(f"  Content Hold Score: {reflection.content_hold_score}")
    log_debug(f"  Style Transfer Score: {reflection.style_transfer_score}")
    log_debug(f"  Satisfied: {reflection.is_satisfied}")

    # Append reflection result to the same prompt file
    with open(reflection_prompt_path, "a") as f:
        f.write(f"\n\n=== REFLECTION RESULT ===\n")
        f.write(f"Reflection Count: {state['reflection_count']}\n")
        f.write(f"Critique: {reflection.critique}\n")
        f.write(f"Content Hold Score: {reflection.content_hold_score}\n")
        f.write(f"Style Transfer Score: {reflection.style_transfer_score}\n")
        f.write(f"Is Satisfied: {reflection.is_satisfied}\n")
    log_success(f"Reflection saved to {reflection_prompt_path}")

    # 4. If not satisfied, plan the next stage
    if not reflection.is_satisfied and state['reflection_count'] < MAX_REFLECTIONS:
        log_agent("StagePlanner", "Planning next stage based on reflection")
        
        # Prepare context of previous stages for the planner
        previous_stages_summary = "\n".join([f"- Stage '{s.stage_name}' generated '{s.generated_image_tag}'" for s in state['style_transfer_plan'].stages])
        
        stage_append_prompt = HumanMessage(content=f"""
        Based on the critique of the last generated image, create a *single* new stage to improve the result.(e,g,, need_more_colorsim, need_texture_refine, need_abstraction, need_blur, need_lightning, remove_sketch, remove_text, add_xxx, etc.)

        **Critique of previous step:**
        {reflection.critique}

        **Previous Stages:**
        {previous_stages_summary}

        **Available Images for next stage:**
        {list(state['generated_images_map'].keys())}

        Define the next stage to address the critique. Ensure the `generated_image_tag` is unique.
        The `required_image_tags` must be chosen from the list of available images. The most recent image was '{last_stage_tag}'.
        """)

        # Append stage append prompt to the reflection file
        reflection_prompt_path = os.path.join(output_dir, f"reflection_{state['reflection_count']}_prompt.txt")
        with open(reflection_prompt_path, "a") as f:
            f.write(f"\n\n=== STAGE APPEND PROMPT ===\n")
            f.write("=== SYSTEM MESSAGE ===\n")
            f.write(stage_sys_message.content)
            f.write("\n\n=== STAGE APPEND INSTRUCTION ===\n")
            f.write(stage_append_prompt.content)

        new_stage: Stage = await structured_stage_llm.ainvoke([stage_sys_message, stage_append_prompt])
        state['style_transfer_plan'].stages.append(new_stage)

        log_success(f"Appended new stage: '{new_stage.stage_name}'")
        plan_path = os.path.join(state['project_dir'], "style_transfer_plan.json")
        with open(plan_path, "w") as f:
            json.dump(state['style_transfer_plan'].model_dump(), f, indent=4)
        log_success(f"Updated style transfer plan saved to {plan_path}")

    return state


def should_go_direct(state: State) -> str:
    """
    Determines whether to perform direct stylization or the multi-stage process.
    """
    return "direct" if state.get("directly") else "staged"


def should_stop_reflecting(state: State) -> str:
    """
    Determines whether to continue the reflection loop or stop.
    """
    if not state['reflections']:
        return "stop"
        
    last_reflection = state['reflections'][-1]
    if last_reflection.is_satisfied:
        log_success("Reflection satisfied. Stopping.")
        return "stop"

    if state['reflection_count'] >= MAX_REFLECTIONS:
        log_warning("Max reflection steps reached. Stopping.")
        return "stop"
        
    return "continue"


async def direct_stylize_node(state: State) -> State:
    """
    Performs a direct, single-step style transfer using the content and style images.
    """
    log_agent("DirectStyle", "Performing Direct Style Transfer")
    if not genai_client:
        log_warning("GenAI client not initialized. Skipping image generation.")
        return state

    try:
        with Image.open(state['content_image_path']) as img:
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
        aspect_ratio_to_use = "1:1"
        
    generated_images_map = state['generated_images_map']
    output_dir = state['project_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare content for the model
    contents = [
    ]
    direct_prompt_instruction = "Transfer the style from the style image to the content image. Replicate the colors, textures, and overall mood of the style image.",

    
    with open(state['content_image_path'], "rb") as image_file:
        contents.append(Part.from_bytes(data=image_file.read(), mime_type="image/png"))
    
    with open(state['style_image_path'], "rb") as image_file:
        contents.append(Part.from_bytes(data=image_file.read(), mime_type="image/png"))
    
    contents.append(direct_prompt_instruction)

    # Save direct stylize prompt before invoking
    direct_prompt_path = os.path.join(output_dir, "direct_stylize_prompt.txt")
  
    log_success(f"Direct stylize prompt saved to {direct_prompt_path}")

    log_tool("ImageGen", "Generating image 'direct_stylized_image'...")
    try:
        response = genai_client.models.generate_content(
            model=MODEL_ID,
            contents=contents,
            config=GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=ImageConfig(aspect_ratio=aspect_ratio_to_use),
                candidate_count=1,
                temperature=0.7 # Default temperature for direct stylization
            ),
        )

        if response.candidates and response.candidates[0].finish_reason == FinishReason.STOP:
            generated_image_data = response.candidates[0].content.parts[0].inline_data.data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_image_path = os.path.join(output_dir, f"direct_stylized_image_{timestamp}.png")
            
            with open(new_image_path, "wb") as f:
                f.write(generated_image_data)

            log_success(f"Generated image saved to: {new_image_path}")
            generated_images_map["direct_stylized_image"] = new_image_path
        else:
            reason = response.candidates[0].finish_reason if response.candidates else "Unknown"
            log_error(f"Image generation failed. Reason: {reason}")

    except Exception as e:
        log_error(f"An error occurred during image generation: {e}")
    
    state['generated_images_map'] = generated_images_map
    return state


async def aggregate_images_node(state: State) -> State:
    """
    Aggregates the generated images and saves final reflection summary.
    The final image is typically the one from the last stage.
    """
    log_agent("Aggregation", "Aggregating Images")
    log_state("Generated Images Map:")
    for tag, path in state['generated_images_map'].items():
        log_debug(f"  - {tag}: {path}")

    # Potentially select the final image and put its path in a dedicated state field
    if state['style_transfer_plan'] and state['style_transfer_plan'].stages:
        last_stage_tag = state['style_transfer_plan'].stages[-1].generated_image_tag
        if last_stage_tag in state['generated_images_map']:
            log_success(f"Final image is: {state['generated_images_map'][last_stage_tag]}")

    # Save final reflection summary
    output_dir = state['project_dir']
    os.makedirs(output_dir, exist_ok=True)

    if state['reflections']:
        final_reflection_path = os.path.join(output_dir, "final_reflection_summary.txt")
        with open(final_reflection_path, "w") as f:
            f.write("=== FINAL REFLECTION SUMMARY ===\n")
            f.write(f"Total Reflections: {len(state['reflections'])}\n")
            f.write(f"Reflection Count: {state['reflection_count']}\n")
            f.write(f"Final Satisfaction: {state['reflections'][-1].is_satisfied if state['reflections'] else False}\n")
            f.write("\n=== REFLECTIONS ===\n")
            for i, reflection in enumerate(state['reflections']):
                f.write(f"\n--- Reflection {i + 1} ---\n")
                f.write(f"Critique: {reflection.critique}\n")
                f.write(f"Content Hold Score: {reflection.content_hold_score}\n")
                f.write(f"Style Transfer Score: {reflection.style_transfer_score}\n")
                f.write(f"Is Satisfied: {reflection.is_satisfied}\n")
        log_success(f"Final reflection summary saved to {final_reflection_path}")

    return state

async def init_context_node(state: State) -> State:

    style_image_path = state.get("style_image_path")
    content_image_path = state.get("content_image_path")
    generated_images_map = state.get("generated_images_map", {})
    generated_images_map["style_image"] = style_image_path
    generated_images_map["content_image"] = content_image_path
    state["generated_images_map"] = generated_images_map
    state["reflection_count"] = 0
    state["reflections"] = []
 
    return state

graph.add_node("init_context", init_context_node)
graph.add_node("comprehensive_style_analysis", comprehensive_style_analysis_node)
graph.add_node("style_transfer_analysis", style_transfer_analysis_node)
graph.add_node("plan_stages", plan_stages_node)
graph.add_node("execute_stage", execute_stage_node)
graph.add_node("direct_stylize", direct_stylize_node)
graph.add_node("reflect", reflect_node)
graph.add_node("aggregate_images", aggregate_images_node)

graph.set_entry_point("init_context")
graph.add_conditional_edges(
    "init_context",
    should_go_direct,
    {
        "direct": "direct_stylize",
        "staged": "comprehensive_style_analysis",
    },
)
graph.add_edge("comprehensive_style_analysis", "style_transfer_analysis")
graph.add_edge("style_transfer_analysis", "plan_stages")
graph.add_edge("plan_stages", "execute_stage")
graph.add_edge("execute_stage", "reflect")
graph.add_conditional_edges(
    "reflect",
    should_stop_reflecting,
    {
        "continue": "execute_stage",
        "stop": "aggregate_images",
    }
)
graph.add_edge("direct_stylize", "aggregate_images")
graph.add_edge("aggregate_images", END)
