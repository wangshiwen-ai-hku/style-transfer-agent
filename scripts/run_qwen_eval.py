import os
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
import glob

def run_qwen_experiments(dataset_root):
    # Load pipeline
    print("Loading Qwen pipeline...")
    pipeline = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16)
    pipeline.to('cuda')
    pipeline.set_progress_bar_config(disable=None)
    print("Pipeline loaded.")

    # Get all experiment folders
    exp_folders = sorted(glob.glob(os.path.join(dataset_root, "sty_*_cnt_*")))
    
    for exp_dir in exp_folders:
        print(f"\nProcessing experiment: {os.path.basename(exp_dir)}")
        
        # Determine number of stages by looking at stage_N_prompt.txt files
        stage_prompts = glob.glob(os.path.join(exp_dir, "stage_*_prompt.txt"))
        num_stages = len(stage_prompts)
        
        for stage_idx in range(1, num_stages + 1):
            prompt_file = os.path.join(exp_dir, f"stage_{stage_idx}_prompt.txt")
            input_file = os.path.join(exp_dir, f"stage_{stage_idx}_imageinput.txt")
            output_name_file = os.path.join(exp_dir, f"stage_{stage_idx}_outputname.txt")
            
            if not all(os.path.exists(f) for f in [prompt_file, input_file, output_name_file]):
                print(f"Skipping stage {stage_idx} in {exp_dir} due to missing files.")
                continue
            
            # Read stage info
            with open(prompt_file, 'r') as f:
                prompt = f.read().strip()
            
            with open(input_file, 'r') as f:
                input_filenames = f.read().strip().split(',')
            
            with open(output_name_file, 'r') as f:
                output_filename = f.read().strip()
            
            output_path = os.path.join(exp_dir, output_filename)
            
            # Skip if output already exists (optional, but good for resuming)
            if os.path.exists(output_path):
                print(f"Stage {stage_idx} output already exists: {output_filename}. Skipping.")
                continue

            print(f"Running Stage {stage_idx}...")
            print(f"  Inputs: {input_filenames}")
            print(f"  Prompt: {prompt[:50]}...")

            # Load input images
            input_images = []
            for fname in input_filenames:
                img_path = os.path.join(exp_dir, fname)
                if not os.path.exists(img_path):
                    print(f"Error: Input image {img_path} not found!")
                    break
                input_images.append(Image.open(img_path).convert("RGB"))
            
            if len(input_images) != len(input_filenames):
                continue

            # Run pipeline
            inputs = {
                "image": input_images,
                "prompt": prompt,
                "generator": torch.manual_seed(0),
                "true_cfg_scale": 4.0,
                "negative_prompt": " ",
                "num_inference_steps": 40,
                "guidance_scale": 1.0,
                "num_images_per_prompt": 1,
            }
            
            with torch.inference_mode():
                try:
                    output = pipeline(**inputs)
                    output_image = output.images[0]
                    output_image.save(output_path)
                    print(f"  Saved output to {output_path}")
                except Exception as e:
                    print(f"  Error running stage {stage_idx}: {e}")
                    break

if __name__ == "__main__":
    dataset_path = "qwen_eval_dataset"
    if os.path.exists(dataset_path):
        run_qwen_experiments(dataset_path)
    else:
        print(f"Dataset path {dataset_path} not found.")
