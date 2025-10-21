import asyncio

import os
import argparse
import shutil
from PIL import Image
from datetime import datetime
from src.utils.colored_logger import init_default_logger
import mimetypes

from dotenv import load_dotenv
load_dotenv()
def to_jpg(image_path):
    img = Image.open(image_path)
    image_path = os.path.splitext(image_path)[0] + ".jpg"
    img.save(image_path)
    return image_path

async def main():
    parser = argparse.ArgumentParser(description="Run the style transfer agent.")
    parser.add_argument("--images", "-i", nargs='+', help="Paths to the input images. The first is treated as style, the second as content for 'agent' task.", required=True)
    parser.add_argument("--prompt","-p", default="Transfer the style of the image 2 to image 1.", help="User prompt for the agent.")
    parser.add_argument("--result_dir", help="Path to the result directory.", default="result_exp_general")
    parser.add_argument("--task_type", "-t", help="Task type.", default="general")
    parser.add_argument("--gen_image_model", "-g", help="Image generation model.", default="gemini")
    parser.add_argument(
        "--directly", "-d",
        action="store_true",
        help="Perform direct style transfer, bypassing the multi-stage process."
    )
    args = parser.parse_args()
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
    project_dir = os.path.join(args.result_dir, first_image_name, args.gen_image_model + "_" + timestamp)
    os.makedirs(project_dir, exist_ok=True)
    for i, img_path in enumerate(image_paths):
        shutil.copy(img_path, os.path.join(project_dir, f"image_{i+1}{os.path.splitext(img_path)[1]}"))
        
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
            "generated_images_map": {},
            "user_prompt": args.prompt,
            "image_analysis": None, 
            "style_transfer_plan": None,
            "directly": args.directly,
            "gen_image_model": args.gen_image_model}

    elif args.task_type == "general":
        from src.general.graph import graph
        from src.general.schema import State as GeneralState
        
        initial_state = {
            "image_paths": image_paths,
            "project_dir": project_dir,
            "generated_images_map":{},
            "user_prompt": args.prompt,
            "directly": args.directly,
            "gen_image_model": args.gen_image_model
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
    
    
    
