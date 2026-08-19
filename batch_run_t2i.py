import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Batch run text-driven style transfer.")
    parser.add_argument("--style_dir", default="data/t2istyle", help="Directory containing style images.")
    parser.add_argument("--prompt_file", default="data/t2istyle/prompt.txt", help="Path to the prompt (content) file.")
    parser.add_argument("--config", default="config_t2i.yaml", help="Path to the config file.")
    
    args = parser.parse_args()

    if not os.path.exists(args.prompt_file):
        print(f"Error: Prompt file not found at {args.prompt_file}")
        return

    with open(args.prompt_file, "r") as f:
        contents = [line.strip() for line in f if line.strip()]

    style_images = [
        os.path.join(args.style_dir, f) 
        for f in os.listdir(args.style_dir) 
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
    style_images.sort()

    print(f"Found {len(style_images)} style images and {len(contents)} prompts.")

    for style_image in style_images:
        for content in contents:
            print(f"\n>>> Running: Style={os.path.basename(style_image)}, Content='{content}'")
            
            cmd = [
                "python", "run_agent_limited.py",
                "--config", args.config,
                "--images", style_image,
                "--content", content,
                # "--directly", True
            ]
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running command: {e}")
                continue

if __name__ == "__main__":
    main()
