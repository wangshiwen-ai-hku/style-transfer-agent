import asyncio
from src.agent.graph import graph
from src.agent.schema import State
import os
import argparse
import shutil
from PIL import Image
from datetime import datetime
from src.utils.colored_logger import init_default_logger

def to_jpg(image_path):
    img = Image.open(image_path)
    image_path = os.path.splitext(image_path)[0] + ".jpg"
    img.save(image_path)
    return image_path

async def main():
    parser = argparse.ArgumentParser(description="Run the style transfer agent.")
    parser.add_argument("--style_image_path", help="Path to the style image.", default="styles/style.png")
    parser.add_argument("--content_image_path", help="Path to the content image.", default="contents/content.png")
    parser.add_argument("--prompt", default="Transfer the style of the content image to the style image.", help="User prompt for the agent.")
    parser.add_argument("--result_dir", help="Path to the result directory.", default="result_exp")
    parser.add_argument(
        "--directly",
        action="store_true",
        help="Perform direct style transfer, bypassing the multi-stage process."
    )
    args = parser.parse_args()
    # style_image_paths = glob.glob("styles/*.jpg") + glob.glob("styles/*.png")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")    
    # style_image_path = style_image_paths[0]
    # for style_image_path in style_image_paths:
    if True:
        style_image_path = args.style_image_path

        # if the mimetype of image is not png/jpg
        import mimetypes
        if mimetypes.guess_type(style_image_path)[0] not in ["image/png", "image/jpg"]:
            
            style_image_path = to_jpg(style_image_path)
        
        if mimetypes.guess_type(args.content_image_path)[0] not in ["image/png", "image/jpg"]:
            args.content_image_path = to_jpg(args.content_image_path)
     
        app = graph.compile()
        style_image_name = os.path.basename(style_image_path).split(".")[0]
        content_image_name = os.path.basename(args.content_image_path).split(".")[0]
        project_dir = os.path.join(args.result_dir, style_image_name, timestamp)
        os.makedirs(project_dir, exist_ok=True)
        shutil.copy(style_image_path, os.path.join(project_dir, f"{style_image_name}.png"))
        shutil.copy(args.content_image_path, os.path.join(project_dir, f"{content_image_name}.png"))
        # Define the initial state for the graph
        initial_state: State = {
            "style_image_path": style_image_path,
            "content_image_path": args.content_image_path,
            "project_dir": project_dir,
            "generated_images_map": {},
            "user_prompt": args.prompt,
            # The following will be populated by the graph
            "image_analysis": None, 
            "style_transfer_plan": None,
            "directly": args.directly
        }

        print("--- Starting Style Transfer Agent ---")

        # Initialize logging
        init_default_logger(__name__)

        # Run the graph
        final_state = await app.ainvoke(initial_state)

        print("\n--- Style Transfer Agent Finished ---")
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
    
    
    
