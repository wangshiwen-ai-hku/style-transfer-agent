from langgraph.graph import StateGraph
from .schema import State, StyleTransferPlan, ComprehensiveStyleAnalysis, StyleTransferAnalysis, Message, MODALITY, Reflection, Stage, SystemOrchestration, TaskIdentifier, SkillSelector
from src.config.manager import ConfigManager
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
from pathlib import Path
import asyncio
from collections import deque
from src.utils.colored_logger import log_agent, log_tool, log_error, log_warning, log_success
from src.utils.json_utils import extract_json_from_text
from src.utils.image_processing import canny_edge_detection
from src.utils.image_generation import image_generation_tool

MAX_REFLECTIONS = 5
config_path = Path(__file__).parent / "config.yaml"

config = ConfigManager(config_path)

load_dotenv()

# Initialize colored logger for enhanced visual debugging
init_default_logger(__name__)
# Use the global logger instance for consistency
logger = None

graph = StateGraph(State)


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

# LLM and client initializations
plan_agent_config = config.get_agent_config('plan_agent', 'core')
plan_llm = init_chat_model(**asdict(plan_agent_config.model))

MODEL_ID = "gemini-2.5-flash-image"
PROJECT_ID = os.environ.get("GOOGLE_PROJECT_ID", None)
LOCATION = "us-central1"

genai_client = None
if not PROJECT_ID:
        log_warning("GOOGLE_PROJECT_ID environment variable not set. Skipping GenAI client initialization.")
else:
    try:
        genai_client = genai.Client(http_options=HttpOptions(api_version="v1"))
    except Exception as e:
        log_error(f"Error initializing Google GenAI Client: {e}")
        log_warning("Please ensure you have authenticated with Google Cloud (e.g., `gcloud auth application-default login`).")


async def orchestrator_node(state: State) -> State:
    """
    Acts as a ReAct-style agent to parse user intent, select and read a skill file,
    and then dynamically configure the agent system based on the skill's rules.
    """
    log_agent("Orchestrator", "Initializing ReAct-style orchestration.")
    orchestrator_agent_config = config.get_agent_config('orchestrator_agent')
    orchestrator_llm = init_chat_model(**asdict(orchestrator_agent_config.model))
    
    user_prompt = state['user_prompt']
    
    # --- Action: List available skills (Tool: list_dir) ---
    log_tool("Orchestrator", "Listing available skills from the 'rules' directory.")
    rules_dir = Path(__file__).parent / "rules"
    try:
        skill_files = [f for f in os.listdir(rules_dir) if f.endswith('.md')]
        skill_files_full_path = [os.path.join(rules_dir, f) for f in skill_files]
        log_success(f"Found skills: {skill_files}")
    except Exception as e:
        log_error(f"Could not list skill files in {rules_dir}. Halting. Error: {e}")
        # Early exit if we can't even find the rules.
        return state

    # --- Thought: Decide which skill to use ---
    log_agent("Orchestrator", "Thinking... Which skill is needed for the user's request?")
    # Log ReAct/internal reasoning steps for easier debugging
    log_debug("Orchestrator ReAct: Preparing skill selection prompt with user input and available skills")
    log_debug(f"Orchestrator ReAct: user_prompt={{user_prompt}}")
    skill_selection_prompt = f"""
    You are an AI architect responsible for designing skills and rules in an image processing workflow.
    Based on the user's request, choose the most relevant "skill" to guide your design (Be accurate, If NOT so relevant, you must generate new skill content, i.e. task_name, task_description, task_rules.)

    **User's Request:**
    {user_prompt}

    **Available Skills:**
    {skill_files_full_path}

    Analyze the user's request and determine which single skill file is the most appropriate to read.
    """
    structured_skill_selector_llm = orchestrator_llm.with_structured_output(SkillSelector)
    
    try:
        log_tool("Orchestrator", "Calling skill-selection LLM (tool call)")
        selection_result = await structured_skill_selector_llm.ainvoke(skill_selection_prompt)
        skill_to_read = selection_result.skill_file_to_read
        user_specific_rules = selection_result.user_specific_rules
        log_debug(f"Orchestrator ReAct: user_specific_rules={user_specific_rules}")
        if not skill_to_read:
            log_warning("No suitable skill file found. Generating a new skill file based on the user's request.")
            task_name = selection_result.task_name
            rules_content = selection_result.rules + "\n\n" + user_specific_rules
            generated_skill_path = state.get('project_dir') / "rules" / f"{task_name}.md"
            with open(generated_skill_path, 'w', encoding='utf-8') as gf:
                gf.write(rules_content)
            log_success(f"Auto-generated missing skill file and saved to {generated_skill_path}")
            skill_to_read = generated_skill_path
        log_success(f"Decided to use skill: {skill_to_read}")
        # Detailed ReAct trace
        log_debug("Orchestrator ReAct: Received structured skill selection result")
        try:
            log_debug(f"Orchestrator ReAct: selection_result={selection_result.model_dump()}")
        except Exception:
            log_debug("Orchestrator ReAct: (could not dump structured result)")
    except Exception as e:
        log_error(f"Could not decide which skill to use. Halting. Error: {e}")
        return state

    # --- Action: Read the selected skill file (Tool: read_file) ---
    log_tool("Orchestrator", f"Reading skill file: {skill_to_read}")
    rules_content = "No specific rules loaded."
    try:
        log_tool("Orchestrator", f"Reading skill file: {skill_to_read} (tool call)")
        if Path(skill_to_read).exists():
            with open(skill_to_read, "r", encoding="utf-8") as f:
                # We'll strip the YAML frontmatter for the main prompt, as it's metadata.
                file_content = f.read()
                if file_content.startswith('---'):
                    parts = file_content.split('---', 2)
                    if len(parts) > 2:
                        rules_content = parts[2].strip()
                    else:
                        rules_content = file_content # Fallback if format is weird
                else:
                    rules_content = file_content
            rules_content = rules_content + "\n\n" + user_specific_rules
            log_success(f"Successfully loaded and parsed rules from {skill_to_read}")
            log_debug(f"Orchestrator ReAct: rules_content_length={len(rules_content)}")
        else:
            log_warning(f"Selected skill file '{skill_to_read}' does not exist.")
        
    except Exception as e:
        log_error(f"Error reading or parsing skill file '{skill_to_read}': {e}")


    # --- Thought & Final Answer: Generate the System Orchestration ---
    log_agent("Orchestrator", "Synthesizing skill rules and user request to design the final agent graph.")
    structured_orchestrator_llm = orchestrator_llm.with_structured_output(SystemOrchestration)
    log_debug("Orchestrator ReAct: Prepared structured orchestrator LLM for final orchestration (tool ready)")
    
    prompt_template = orchestrator_agent_config.prompt
    image_tags = list(state['generated_images_map'].keys())
    
    # The `rules_to_be_loaded` placeholder will be filled with the content of the selected skill file
    formatted_prompt = prompt_template.format(user_prompt=user_prompt, image_tags=image_tags, rules_to_be_loaded=rules_content)
    
    # Prepare image messages for multi-modal input
    image_messages = []
    for tag, image_path in state['generated_images_map'].items():
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
        image_messages.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"}
        })
        
    orchestration_message = HumanMessage(content=[
        {"type": "text", "text": formatted_prompt},
        *image_messages
    ])
    log_tool("Orchestrator", "Calling orchestrator LLM to generate SystemOrchestration (tool call)")
    
    system_orchestration: SystemOrchestration = await structured_orchestrator_llm.ainvoke([orchestration_message])
    log_debug("Orchestrator ReAct: Received SystemOrchestration response from LLM")
    try:
        log_debug(f"Orchestrator ReAct: system_orchestration={system_orchestration.model_dump()}")
    except Exception:
        log_debug("Orchestrator ReAct: (could not dump system_orchestration)")
    
    state['system_orchestration'] = system_orchestration
    log_success("Orchestrator configured the system successfully.")
    log_debug(f"Final Task Type from Orchestration: {system_orchestration.task_type}")
    for agent_config in system_orchestration.agent_graph:
        log_debug(f"  - Agent in graph: {agent_config.agent_name}")
    
    # Save the orchestration plan
    output_dir = state['project_dir']
    orchestration_path = os.path.join(output_dir, "system_orchestration.json")
    with open(orchestration_path, "w") as f:
        f.write(system_orchestration.model_dump_json(indent=4))
    log_success(f"System orchestration saved to {orchestration_path}")

    return state


async def execute_graph_node(state: State) -> State:
    """
    Runs the dynamically configured agent graph, which includes both analysis and planning agents.
    It executes agents in topological order and handles the final planning agent specially to
    produce the structured StyleTransferPlan.
    """
    log_agent("GraphExecutor", "Running agent graph.")
    system_orchestration = state['system_orchestration']
    agent_model_config = config.get_agent_config('function_agents', 'core').model
    output_dir = state['project_dir']
    
    analysis_context = state.get('analysis_context', {})
    
    agents = system_orchestration.agent_graph
    agent_map = {agent.agent_name: agent for agent in agents}
    adj = {agent.agent_name: [] for agent in agents}
    in_degree = {agent.agent_name: 0 for agent in agents}
    out_degree = {agent.agent_name: 0 for agent in agents}

    for agent in agents:
        dependencies = agent.dependencies or []
        for dep in dependencies:
            if dep in agent_map:
                adj[dep].append(agent.agent_name)
                in_degree[agent.agent_name] += 1
                out_degree[dep] += 1
            else:
                log_warning(f"Dependency '{dep}' for agent '{agent.agent_name}' not found. It will be ignored.")

    planner_agents = {name for name, degree in out_degree.items() if degree == 0}
    log_state(f"Identified planner agents (terminal nodes): {planner_agents}")

    queue = deque([agent.agent_name for agent in agents if in_degree[agent.agent_name] == 0])
    
    processed_agents_count = 0

    async def run_agent(agent_config):
        is_planner = agent_config.agent_name in planner_agents
        log_tool("AgentRunner", f"Running {'Planner' if is_planner else 'Analysis'} Agent: {agent_config.agent_name}")
        # Detailed logging for tool calls and internal ReAct-style reasoning
        log_debug(f"AgentRunner ReAct: Preparing to run agent '{agent_config.agent_name}'")
        log_debug(f"AgentRunner ReAct: required_image_tags={agent_config.required_image_tags}")

        image_messages = []
        for tag in agent_config.required_image_tags:
            image_path = state['generated_images_map'].get(tag)
            if not image_path:
                log_warning(f"Agent '{agent_config.agent_name}' requires tag '{tag}', but it was not found. Skipping.")
                continue
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
            image_messages.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"}})
            log_tool("AgentRunner", f"Attached image tag '{tag}' for agent '{agent_config.agent_name}' (tool: file read)")

        # Use the existing analysis_context for formatting the prompt
        formatted_prompt = agent_config.prompt
        for dep_name in (agent_config.dependencies or []):
            placeholder = f"{{{dep_name}_output}}"
            if placeholder in formatted_prompt:
                dep_output = analysis_context.get(dep_name, {})
                formatted_prompt = formatted_prompt.replace(placeholder, json.dumps(dep_output, indent=2))

        model_kwargs = asdict(agent_model_config)
        if agent_config.temperature is not None:
            model_kwargs['temperature'] = agent_config.temperature
        agent_llm = init_chat_model(**model_kwargs)

        # suggested_base_image = system_orchestration.suggested_base_image
        combined_prompt = f"Your Task:**\n{formatted_prompt}"
        # suggested_base_image_message = f"{suggested_base_image}"
        
        messages = [
            SystemMessage(content=combined_prompt),
            HumanMessage(content=image_messages if image_messages else "Proceed with your analysis based on text context.")
        ]

        if 'planner' in agent_config.agent_name:
            # if image_messages:
            #     image_messages.append({"type": "text", "text": f"Suggested base image: {suggested_base_image_message}, you had better  make your stages plan from it as an edited base."})
            # messages = [
            #     SystemMessage(content=combined_prompt),
            #     HumanMessage(content=image_messages if image_messages else "Proceed with your analysis based on text context."),
            # ]
            log_tool("AgentRunner", f"Calling planner LLM for agent '{agent_config.agent_name}' (tool call)")
            structured_llm = agent_llm.with_structured_output(StyleTransferPlan)
            plan = await structured_llm.ainvoke(messages)
            log_debug(f"AgentRunner ReAct: planner '{agent_config.agent_name}' returned plan")
            try:
                log_debug(f"AgentRunner ReAct: plan_dump={plan.model_dump()}")
            except Exception:
                log_debug("AgentRunner ReAct: (could not dump plan)")
            return {"type": "plan", "data": plan}
        else:
            log_tool("AgentRunner", f"Calling analysis LLM for agent '{agent_config.agent_name}' (tool call)")
            response = await agent_llm.ainvoke(messages)
            response_content = response.content
            log_debug(f"AgentRunner ReAct: raw response length={len(response_content) if isinstance(response_content, str) else 'unknown'}")
            try:
                parsed_json = extract_json_from_text(response_content)
                log_debug(f"AgentRunner ReAct: parsed JSON for agent '{agent_config.agent_name}'")
                return {"type": "analysis", "agent_name": agent_config.agent_name, "data": parsed_json}
            except Exception as e:
                log_warning(f"Could not parse JSON from agent '{agent_config.agent_name}'. Error: {e}")
                log_debug(f"AgentRunner ReAct: returning raw output for agent '{agent_config.agent_name}'")
                return {"type": "analysis", "agent_name": agent_config.agent_name, "data": {"raw_output": response_content}}

    while queue:
        layer_agents = [agent_map[agent_name] for agent_name in list(queue)]
        queue.clear()
        
        results = await asyncio.gather(*(run_agent(agent_config) for agent_config in layer_agents))
        
        # Process results and update context/state explicitly
        for result in results:
            if result is None: continue
            if result["type"] == "analysis":
                agent_name = result["agent_name"]
                analysis_context[agent_name] = result["data"]
                # Save analysis output for debugging
                output_path = os.path.join(output_dir, f"analysis_{agent_name}.json")
                with open(output_path, "w") as f:
                     json.dump(result["data"], f, indent=4)
                log_success(f"Agent {agent_name} output saved to {output_path}")
            elif result["type"] == "plan":
                state['style_transfer_plan'] = result["data"]
                plan_path = os.path.join(output_dir, "style_transfer_plan.json")
                with open(plan_path, "w") as f:
                    f.write(result["data"].model_dump_json(indent=4))
                log_success(f"Generated Style Transfer Plan saved to {plan_path}")

        processed_agents_count += len(layer_agents)
        for agent_config in layer_agents:
            for neighbor in adj[agent_config.agent_name]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
    
    if processed_agents_count != len(agents):
        log_error("Cycle detected or some agents failed. Halting.")
    
    if 'style_transfer_plan' not in state or not state['style_transfer_plan']:
        log_error("Graph execution completed, but no style transfer plan was generated. Halting.")
        state['style_transfer_plan'] = StyleTransferPlan(stages=[])

    state['analysis_context'] = analysis_context
    log_success("Agent graph execution complete.")
    return state


async def plan_stages_node(state: State) -> State:
    """
    DEPRECATED: This node's logic has been merged into execute_graph_node.
    This function can be removed.
    """
    log_warning("plan_stages_node is deprecated and should be removed from the graph.")
    return state


async def execute_stage_node(state: State) -> State:
    """
    Executes each stage of the style transfer plan to generate images.
    """
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

            # Use the generic image_generation_tool (it will call the appropriate backend)
            model_to_use = state.get('gen_image_model', MODEL_ID) if isinstance(state, dict) else MODEL_ID
            log_debug(f"Using image generation model: {model_to_use}")

            # collect image input paths required by this stage
            image_input_paths = [generated_images_map[tag] for tag in image_tags if tag in generated_images_map]
            # consolidate text prompts
            text_prompt_combined = "\n\n".join(texts) if isinstance(texts, (list, tuple)) else str(texts)

            try:
                new_image_pil = image_generation_tool(text_prompt_combined, image_input_paths, model=model_to_use)
                new_image_path = state['project_dir'] + "/" + stage.generated_image_tag + ".png"
                new_image_pil.save(new_image_path)
                if new_image_path:
                    log_success(f"Generated image saved to: {new_image_path}")
                    generated_images_map[stage.generated_image_tag] = new_image_path
                else:
                    log_warning(f"Image generation tool did not return a path for stage '{stage.stage_name}'.")
            except Exception as e:
                log_error(f"Image generation failed for stage '{stage.stage_name}': {e}")
                continue

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
    orchestration = state['system_orchestration']
    user_prompt = state['user_prompt']
    reflect_agent_config = config.get_agent_config('reflect_agent', 'core')
    
    model_kwargs = asdict(reflect_agent_config.model)
    reflect_llm = init_chat_model(**model_kwargs)
    structured_reflect_llm = reflect_llm.with_structured_output(Reflection)
    
    # Use the dynamically generated prompt, reinforced with the original user request.
    combined_reflect_prompt = f"**Original User Request:**\n{user_prompt}\n\n**Critique Criteria:**\n{orchestration.result_critique_criteria}"
    reflect_sys_message = SystemMessage(content=combined_reflect_prompt)

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
        stop_reflection = Reflection(critique="Stopping due to missing image.", is_satisfied=True)
        state['reflections'].append(stop_reflection)
        return state

    # 3. Perform reflection
    log_state(f"Reflecting on image: {last_image_path}")
    
    reflection_prompt_instruction = f"""
The context is
{json.dumps(state.get("analysis_context"), indent=2)}.

Please act as an art critic and critique the `generated_image`.
"""
    # TODO how to construct the multimodal messages with other tasks such as style integrated.
    image_contents = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(open(image_path, 'rb').read()).decode()}"}} for image_path in state['provided_images']]
    reflection_prompt = HumanMessage(content=[
        {"type": "text", "text": reflection_prompt_instruction},
        *image_contents,
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
    # log_debug(f"  Content Hold Score: {reflection.content_hold_score}")
    # log_debug(f"  Style Transfer Score: {reflection.style_transfer_score}")
    log_debug(f"  Satisfied: {reflection.is_satisfied}")

    # Append reflection result to the same prompt file
    with open(reflection_prompt_path, "a") as f:
        f.write(f"\n\n=== REFLECTION RESULT ===\n")
        f.write(f"Reflection Count: {state['reflection_count']}\n")
        f.write(f"Critique: {reflection.critique}\n")
        # f.write(f"Content Hold Score: {reflection.content_hold_score}\n")
        # f.write(f"Style Transfer Score: {reflection.style_transfer_score}\n")
        f.write(f"Is Satisfied: {reflection.is_satisfied}\n")
    log_success(f"Reflection saved to {reflection_prompt_path}")

    # 4. If not satisfied, plan the next stage
    if not reflection.is_satisfied and state['reflection_count'] < MAX_REFLECTIONS:
        log_agent("StagePlanner", "Planning next stage based on reflection")
        
        # Prepare context of previous stages for the planner
        previous_stages_summary = "\n".join([f"- Stage '{s.stage_name}' generated '{s.generated_image_tag}'" for s in state['style_transfer_plan'].stages])
        
        stage_append_prompt = HumanMessage(content=f"""
        Based on the critique of the last generated image, create a *single* new stage to improve the result.

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
    Performs a direct, single-step image generation using the provided images and prompt.
    """
    log_agent("DirectGenerate", "Performing direct image generation.")
    if not genai_client:
        log_warning("GenAI client not initialized. Skipping image generation.")
        return state

    aspect_ratio_to_use = "1:1"
    if state.get('provided_images'):
        try:
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
        
    generated_images_map = state['generated_images_map']
    output_dir = state['project_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare content for the model
    contents = []
    # Add all provided images to the context
    for image_path in state.get('provided_images', []):
        with open(image_path, "rb") as image_file:
            mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
            contents.append(Part.from_bytes(data=image_file.read(), mime_type=mime_type))

    # Add the user's prompt
    contents.append(state['user_prompt'])

    # Save direct stylize prompt before invoking
    direct_prompt_path = os.path.join(output_dir, "direct_generate_prompt.txt")
    with open(direct_prompt_path, "w") as f:
        f.write(state['user_prompt'])
    log_success(f"Direct generate prompt saved to {direct_prompt_path}")

    log_tool("ImageGen", "Generating image 'direct_generated_image'...")
    model_to_use = state.get('gen_image_model', MODEL_ID) if isinstance(state, dict) else MODEL_ID
    log_debug(f"Using image generation model for direct stylize: {model_to_use}")

    image_input_paths = state.get('provided_images', [])
    try:
        new_image = image_generation_tool(state['user_prompt'], image_input_paths, model=model_to_use)
        new_image_path = os.path.join(output_dir, f"direct_generated_image.png")
        new_image.save(new_image_path)
        if new_image_path:
            log_success(f"Generated image saved to: {new_image_path}")
            generated_images_map["direct_generated_image"] = new_image_path
        else:
            log_error("Image generation tool did not return a path for direct generation")
    except Exception as e:
        log_error(f"An error occurred during direct image generation: {e}")
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
    if state.get('style_transfer_plan', []) and state['style_transfer_plan'].stages:
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
                # f.write(f"Content Hold Score: {reflection.content_hold_score}\n")
                # f.write(f"Style Transfer Score: {reflection.style_transfer_score}\n")
                f.write(f"Is Satisfied: {reflection.is_satisfied}\n")
        log_success(f"Final reflection summary saved to {final_reflection_path}")

    return state

async def create_agents_config_node(state: State) -> State:
    """
    TODO
    1. parse the user instruction
    2. fill the config.yaml, come up with multiagent analysis config
     come up plan/reflect prompt 
    """

async def init_context_node(state: State) -> dict:
    """
    Initializes the state with generic image tags and paths.
    This node returns a dictionary of updates to be merged into the state.
    """
    image_paths = state.get("image_paths", [])
    # Always start with a fresh map for initial tagging.
    generated_images_map = {} 
    
    for i, image_path in enumerate(image_paths):
        tag = f"image_{i+1}"
        generated_images_map[tag] = image_path
        log_state(f"Initial image tagged: '{tag}' -> {image_path}")
    state["generated_images_map"] = generated_images_map
    state["reflection_count"] = 0
    state["reflections"] = []
    state["provided_images"] = image_paths
    return state

graph.add_node("init_context", init_context_node)
graph.add_node("orchestrator", orchestrator_node)
graph.add_node("execute_graph", execute_graph_node)
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
        "staged": "orchestrator",
    },
)
graph.add_edge("orchestrator", "execute_graph")
graph.add_edge("execute_graph", "execute_stage")
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
