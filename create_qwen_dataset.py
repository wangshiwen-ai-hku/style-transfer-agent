import os
import json
import yaml
import shutil
import re

def extract_id(path):
    # Extracts the filename without extension, then tries to find a number
    filename = os.path.basename(path)
    match = re.search(r'(\d+)', filename)
    if match:
        return match.group(1)
    return filename.split('.')[0]

def create_dataset(result_root, output_root):
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    for item in os.listdir(result_root):
        item_path = os.path.join(result_root, item)
        if not os.path.isdir(item_path):
            continue
        
        # item is the ID (e.g., '8', '76')
        for exp_folder in os.listdir(item_path):
            exp_path = os.path.join(item_path, exp_folder)
            if not os.path.isdir(exp_path):
                continue
            
            plan_file = os.path.join(exp_path, 'style_transfer_plan.json')
            config_file = os.path.join(exp_path, 'config_limited.yaml')
            
            if not os.path.exists(plan_file) or not os.path.exists(config_file):
                print(f"Skipping {exp_path} (missing plan or config)")
                continue
            
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            with open(plan_file, 'r') as f:
                plan = json.load(f)
            
            image_paths = config.get('images', [])
            image_tags = config.get('image_tags', [])
            
            if len(image_paths) < 2:
                print(f"Skipping {exp_path} (less than 2 images)")
                continue
                
            style_id = "unknown"
            content_id = "unknown"
            
            tag_to_original_file = {}
            for i, (path, tag) in enumerate(zip(image_paths, image_tags)):
                if tag == 'style_image':
                    style_id = extract_id(path)
                elif tag == 'content_image':
                    content_id = extract_id(path)
                
                # In the results folder, they are named image_1.jpg, image_2.jpg...
                tag_to_original_file[tag] = f"image_{i+1}.jpg"

            dataset_folder = os.path.join(output_root, f"sty_{style_id}_cnt_{content_id}")
            if not os.path.exists(dataset_folder):
                os.makedirs(dataset_folder)
            
            # Copy initial images
            for tag, filename in tag_to_original_file.items():
                src = os.path.join(exp_path, filename)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(dataset_folder, filename))
                else:
                    # Try png if jpg doesn't exist
                    src_png = src.replace('.jpg', '.png')
                    if os.path.exists(src_png):
                        shutil.copy(src_png, os.path.join(dataset_folder, filename.replace('.jpg', '.png')))

            # Process stages
            stages = plan.get('stages', [])
            for i, stage in enumerate(stages):
                stage_num = i + 1
                prompt = stage.get('text_prompt', '')
                output_tag = stage.get('generated_image_tag', f"stage_{stage_num}_output")
                required_tags = stage.get('required_image_tags', [])
                
                # Save prompt
                with open(os.path.join(dataset_folder, f"stage_{stage_num}_prompt.txt"), 'w') as f:
                    f.write(prompt)
                
                # Save output name
                with open(os.path.join(dataset_folder, f"stage_{stage_num}_outputname.txt"), 'w') as f:
                    # We'll use .png as default for generated images
                    f.write(f"{output_tag}.png")
                
                # Save input tags mapping to files
                input_files = []
                for tag in required_tags:
                    if tag in tag_to_original_file:
                        input_files.append(tag_to_original_file[tag])
                    else:
                        # It's a generated tag from a previous stage
                        input_files.append(f"{tag}.png")
                
                with open(os.path.join(dataset_folder, f"stage_{stage_num}_imageinput.txt"), 'w') as f:
                    f.write(",".join(input_files))

    print(f"Dataset created at {output_root}")

if __name__ == "__main__":
    create_dataset("result_for_rebuttal", "qwen_eval_dataset")
