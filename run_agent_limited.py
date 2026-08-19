import asyncio

import os
import argparse
import shutil
import yaml
from PIL import Image
from datetime import datetime
from src.utils.colored_logger import init_default_logger
import mimetypes

from dotenv import load_dotenv
load_dotenv()

def set_vertex():
    os.environ["GOOGLE_PROVIDER"] = "google_vertexai"
    if 'google_genai' in os.environ["MODEL_PROVIDER"]:
        os.environ["MODEL_PROVIDER"] = "google_vertexai"
    # os.environ["GOOGLE_API_KEY"] = os.environ["VERTEX_API_KEY"]
    # os.environ["GOOGLE_PROJECT_ID"] = os.environ["VERTEX_PROJECT_ID"]
    # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ["VERTEX_APPLICATION_CREDENTIALS"]
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

def to_jpg(image_path):
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    if args.resize:
        ratio = args.resize / max(width, height)
        new_size = (int(width * ratio), int(height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
       
    image_root = os.path.dirname(image_path) + '_resized'
    os.makedirs(image_root , exist_ok=True)
    image_name = os.path.splitext(os.path.basename(image_path))[0] + ".jpg"
    image_path = os.path.join(image_root, image_name)
    img.save(image_path)
    return image_path

async def main():
    parser = argparse.ArgumentParser(description="Run the style transfer agent.")
    parser.add_argument("--llm-provider","-lm", help="LLM provider identifier (e.g. azure_openai, google_vertexai)", default=None)
    parser.add_argument("--user_vertexai", "-uv", help="Use vertexai for user prompt.", action="store_true", default=None)
    parser.add_argument("--images", "-i", nargs='+', help="Paths to the input images. The first is treated as style, the second as content for 'agent' task.", required=False)
    parser.add_argument("--image_tags", "-it", nargs='+', help="Tags for the input images.", default=None)
    parser.add_argument("--prompt","-p", default=None, help="User prompt for the agent.")

    parser.add_argument("--limited_plan_num", "-lpn", help="Limited plan number.", type=int, default=None)
    parser.add_argument("--limited_response_length", "-lrl", help="Limited response length.", type=int, default=None)
    parser.add_argument("--limited_prompt_length", "-lpl", help="Limited prompt length.", type=int, default=None)
    parser.add_argument("--limited_reflection_count", "-lrc", help="Limited reflection count.", type=int, default=None)
    parser.add_argument("--limited_analysis_agents", "-laa", help="Limited analysis agents.", type=int, default=None)
    
    parser.add_argument("--config", "-c", help="Path to the config file (YAML).", default="config_limited.yaml")

    parser.add_argument("--result_dir", help="Path to the result directory.", default=None)
    parser.add_argument("--task_type", "-t", help="Task type.", default=None)
    parser.add_argument("--gen_image_model", "-g", help="Image generation model.", default=None)
    parser.add_argument("--need_grayscale", "-ng", help="Need grayscale style image.", action="store_true", default=None)
    parser.add_argument("--need_edges", "-ne", help="Need edges style image.", action="store_true", default=None)
    parser.add_argument("--target_image", "-ti", help="Target image for the task.", default=None)
    parser.add_argument("--target_ratio", "-tr", help="Target ratio for the task.", default=None)
    parser.add_argument("--resize", "-rs", help="Resize the content image.", type=int, default=None)
    parser.add_argument(
        "--directly", "-d",
        action="store_true",
        default=None,
        help="Perform direct style transfer, bypassing the multi-stage process."
    )
    parser.add_argument("--content", "-con", help="Content text to inject into the prompt.", default=None)
    global args
    args = parser.parse_args()

    # --- YAML Config Handling ---
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Update args with values from YAML if they are not set via command line
        for key, value in yaml_config.items():
            if hasattr(args, key):
                # If it's a command line argument and it's still at its default (None or False for actions)
                # we use the YAML value.
                current_val = getattr(args, key)
                if current_val is None or current_val is False:
                    setattr(args, key, value)
            else:
                # If it's not a standard argparse argument, we still add it to args
                setattr(args, key, value)

    # Set defaults for those not in YAML or CLI
    defaults = {
        "llm_provider": "gemini",
        "user_vertexai": False,
        "image_tags": ["style_image", "content_image"],
        "prompt": "Transfer the style of the style image to the content image.",
        "limited_plan_num": 2,
        "limited_response_length": 1000,
        "limited_prompt_length": 1000,
        "limited_reflection_count": 1,
        "limited_analysis_agents": 3,
        "result_dir": "result_for_rebuttal",
        "task_type": "general_limited",
        "gen_image_model": "gemini",
        "need_grayscale": False,
        "need_edges": False,
        "resize": 1024,
        "directly": False
    }
    for key, val in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, val)
    
    if args.content and "{content}" in args.prompt:
        args.prompt = args.prompt.format(content=args.content)
    
    if args.images is None:
        raise ValueError("Images must be provided either via command line or config file.")

    # --- Image Handling ---
    image_paths = []
    for path in args.images:
        if mimetypes.guess_type(path)[0] not in ["image/png", "image/jpg", "image/jpeg"]:
            image_paths.append(to_jpg(path))
        else:
            image_paths.append(path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # --- Project Directory Setup ---
    # Use the first image name for the main result directory
    first_image_name = os.path.basename(image_paths[0]).split(".")[0]
    
    content_suffix = f"_{args.content.replace(' ', '_')}" if args.content else ""
    if 'limited' in args.task_type:
        limit_suffix = f"_plan{args.limited_plan_num}_resp{args.limited_response_length}_prmt{args.limited_prompt_length}_ref{args.limited_reflection_count}_agt{args.limited_analysis_agents}"
        project_dir = os.path.join(args.result_dir, first_image_name, args.llm_provider + "_" + args.gen_image_model + limit_suffix + content_suffix + "_" + timestamp)
    else:
        project_dir = os.path.join(args.result_dir, first_image_name, args.llm_provider + "_" + args.gen_image_model + content_suffix + "_" + timestamp)
        
    os.makedirs(project_dir, exist_ok=True)
    
    # Copy YAML config to project directory
    if args.config and os.path.exists(args.config):
        shutil.copy(args.config, os.path.join(project_dir, "config_limited.yaml"))

    
    # Create logs directory and save constraints
    log_dir = os.path.join(project_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    import json
    constraints = {
        "limited_plan_num": args.limited_plan_num,
        "limited_response_length": args.limited_response_length,
        "limited_prompt_length": args.limited_prompt_length,
        "limited_reflection_count": args.limited_reflection_count,
        "limited_analysis_agents": args.limited_analysis_agents,
        "task_type": args.task_type,
        "llm_provider": args.llm_provider,
        "gen_image_model": args.gen_image_model,
        "timestamp": timestamp,
        "prompt": args.prompt,
        "images": args.images
    }
    with open(os.path.join(log_dir, "resource_constraints.json"), "w") as f:
        json.dump(constraints, f, indent=4)
 
    image_paths = [to_jpg(i) for i in image_paths]
    for i, img_path in enumerate(image_paths):
        shutil.copy(img_path, os.path.join(project_dir, f"image_{i+1}{os.path.splitext(img_path)[1]}"))
    
    # --- Model routing: everything goes through APIYI's OpenAI-compatible endpoint.
    # The GPU server can reach APIYI but not Google/Aliyun/Volcengine directly, so a
    # single provider keeps local and server runs on identical model configuration.
    # --llm-provider selects only WHICH understanding model is used.
    from src.utils.apiyi import MODELS as APIYI_MODELS, load_env as _load_apiyi_env
    _load_apiyi_env()
    _UND = {
        'apiyi':      APIYI_MODELS['understanding'],
        'gemini':     APIYI_MODELS['understanding'],
        'gemini-pro': APIYI_MODELS['understanding_pro'],
        'gpt5':       'gpt-5',
        'gpt5mini':   'gpt-5-mini',
        'gpt4o':      'gpt-4o',
        'qwen':       APIYI_MODELS['qwen_understanding'],
        'doubao':     'doubao-seed-1-6-251015',
    }
    if args.llm_provider not in _UND:
        raise ValueError(
            f"Invalid LLM provider: {args.llm_provider}. Choose one of {sorted(_UND)}")
    if not os.environ.get("APIYI_KEY"):
        raise RuntimeError("APIYI_KEY is not set; put APIYI_KEY and BASE_URL in code/.env")
    os.environ["MODEL_PROVIDER"] = "openai"
    os.environ["MODEL"] = _UND[args.llm_provider]
    os.environ["API_KEY"] = os.environ["APIYI_KEY"]
    os.environ.setdefault("BASE_URL", "https://api.apiyi.com/v1")
    # Legacy names still referenced by some config blocks.
    os.environ["GOOGLE_PROVIDER"] = "openai"
    os.environ["GOOGLE_API_KEY"] = os.environ["APIYI_KEY"]

    
    if  args.user_vertexai:
        set_vertex()

    # --- State Initialization ---
    if args.task_type == "agent":
        from src.agent.graph import graph
        from src.agent.schema import State as AgentState
        
        if len(image_paths) < 2:
            raise ValueError("The 'agent' task type requires at least two images (style and content).")

        initial_state = {
            "content_image_path": image_paths[1],
            "style_image_path": image_paths[0],
            "project_dir": project_dir,
            "generated_images_map": {
                "style_image": image_paths[0],
                "content_image": image_paths[1],
            },
            "target_ratio": Image.open(image_paths[1]).size[0] / Image.open(image_paths[1]).size[1],
            "user_prompt": args.prompt,
            "image_analysis": None, 
            "style_transfer_plan": None,
            "directly": args.directly,
            "gen_image_model": args.gen_image_model}

    elif  args.task_type == 'general':
        
        from src.general.graph import graph
        from src.general.schema import State as GeneralState

        if args.target_image:
            target_ratio = Image.open(args.target_image).size[0] / Image.open(args.target_image).size[1]
        else:

            target_image = image_paths[0]
            target_ratio = Image.open(target_image).size[0] / Image.open(target_image).size[1]
        
        initial_state = {
            "image_paths": image_paths,
            "project_dir": project_dir,
            "generated_images_map": dict(zip(args.image_tags, image_paths)),
            "user_prompt": args.prompt,
            "directly": args.directly,
            "target_ratio": target_ratio,
            "gen_image_model": args.gen_image_model,
            "need_grayscale": args.need_grayscale,
            "need_edges": args.need_edges
        }

    elif 'limited' in args.task_type:
        from src.general_limited.graph import graph
        from src.general_limited.schema import State as LimitedState

        if args.target_image:
            target_ratio = Image.open(args.target_image).size[0] / Image.open(args.target_image).size[1]
        else:
            target_image = image_paths[0]
            target_ratio = Image.open(target_image).size[0] / Image.open(target_image).size[1]

        initial_state = {
            "image_paths": image_paths,
            "project_dir": project_dir,
            "generated_images_map": dict(zip(args.image_tags, image_paths)),
            "user_prompt": args.prompt,
            "directly": args.directly,
            "target_ratio": args.target_ratio if args.target_ratio else target_ratio,
            "gen_image_model": args.gen_image_model,
            "need_grayscale": False,
            "need_edges": False,
            "limited_plan_num": args.limited_plan_num,
            "limited_response_length": args.limited_response_length,
            "limited_prompt_length": args.limited_prompt_length,
            "limited_reflection_count": args.limited_reflection_count,
            "limited_analysis_agents": args.limited_analysis_agents,
            "skip_skill": args.skip_skill
        }

    else:
        raise ValueError(f"Invalid task type: {args.task_type}")

    app = graph.compile()
    
    print("--- Starting Image Processing Agent ---")

    # Initialize logging
    init_default_logger(__name__)

    # Run the graph
    final_state = await app.ainvoke(initial_state)

    print("\n--- Image Processing Agent Finished ---")
    print("Final generated images map:")
    for tag, path in final_state['generated_images_map'].items():
        print(f"  - {tag}: {path}")

    # --- Resource Consumption Statistics ---
    from src.utils.llm_helper import calculate_token_consumption
    import csv
    
    log_dir = os.path.join(project_dir, "logs")
    calculate_token_consumption(log_dir)
    
    # Read the summary and append to master stats
    summary_path = os.path.join(log_dir, "llm_token_consumption_summary.csv")
    master_stats_path = "resource_consumption_stats.csv"
    
    if os.path.exists(summary_path):
        stats = {}
        with open(summary_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                stats[row['metric']] = row['value']
        
        # Add experimental parameters
        stats['timestamp'] = timestamp
        stats['llm_provider'] = args.llm_provider
        stats['gen_image_model'] = args.gen_image_model
        stats['task_type'] = args.task_type
        stats['limited_plan_num'] = args.limited_plan_num
        stats['limited_response_length'] = args.limited_response_length
        stats['limited_prompt_length'] = args.limited_prompt_length
        stats['limited_reflection_count'] = args.limited_reflection_count
        stats['limited_analysis_agents'] = args.limited_analysis_agents
        stats['directly'] = args.directly
        
        # Write to master file
        file_exists = os.path.exists(master_stats_path)
        
        fieldnames = [
            'timestamp', 'llm_provider', 'gen_image_model', 'task_type', 
            'limited_plan_num', 'limited_response_length', 'limited_prompt_length', 
            'limited_reflection_count', 'limited_analysis_agents', 'directly',
            'num_llm_calls', 'total_duration', 'total_input_tokens', 'total_output_tokens', 'total_tokens'
        ]
        
        with open(master_stats_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            
            # Filter stats to match fieldnames (ignoring extra details from summary if needed)
            row_to_write = {k: stats.get(k, '') for k in fieldnames}
            writer.writerow(row_to_write)
            print(f"Appended stats to {master_stats_path}")

if __name__ == "__main__":
    # To avoid potential issues with asyncio in different environments,
    # it's good practice to get the running loop or create a new one.
    import glob
    
    
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    
    loop.run_until_complete(main())
    
    
    
