import os
import shutil
from pathlib import Path

def create_eval_set(src_root, dst_root):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    
    style_dir = dst_root / "style"
    content_dir = dst_root / "content"
    result_dir = dst_root / "pro_results"

    for d in [style_dir, content_dir, result_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    sty_idx = 200
    content_copied = False
    
    # Iterate through each ID folder in src_root
    # Filter for directories and sort them numerically if possible
    id_folders = [f for f in src_root.iterdir() if f.is_dir()]
    try:
        id_folders.sort(key=lambda x: int(x.name))
    except ValueError:
        id_folders.sort()
    
    print(f"Found {len(id_folders)} ID folders in {src_root}")
    
    for id_folder in id_folders:
        # Find the experiment folder inside the ID folder
        exp_folders = [f for f in id_folder.iterdir() if f.is_dir() and not f.name.startswith('.')]
        if not exp_folders:
            continue
        print(exp_folders)
        # Take the most recent experiment folder if there are multiple
        exp_folders.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        for exp_folder in exp_folders:
        
            # Look for image_1 and image_2
            image_1 = None
            for ext in ['.jpg', '.png', '.jpeg']:
                if (exp_folder / f"image_1{ext}").exists():
                    image_1 = exp_folder / f"image_1{ext}"
                    break
            
            image_2 = None
            for ext in ['.jpg', '.png', '.jpeg']:
                if (exp_folder / f"image_2{ext}").exists():
                    image_2 = exp_folder / f"image_2{ext}"
                    break
            
            if not image_1:
                print(f"Skipping {id_folder.name}: image_1 not found")
                continue

            # Find all image files
            all_images = []
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                all_images.extend(exp_folder.glob(ext))
            
            # Filter out image_1 and image_2
            result_images = [img for img in all_images if img.name not in ['image_1', 'image_2']]
            
            if not result_images:
                print(f"Skipping {id_folder.name}: no result image found")
                continue
                
            # Sort by modification time to get the "lastly generated image"
            result_images.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            last_image = result_images[0]
            
            # Perform copying
            # 1. Style image: your/folder/style/{i}.jpg
            shutil.copy2(image_2, style_dir / f"{sty_idx}.jpg")
            
            # 2. Content image (only once): your/folder/content/22.jpg
            if not content_copied and image_2:
                shutil.copy(image_1, content_dir / "22.jpg")
                content_copied = True
                print(f"Copied content image from {id_folder.name}")
                
            # 3. Result image: your/folder/result_limited/sty_{sty}_cnt_{22}.jpg
            result_filename = f"sty_{sty_idx}_cnt_22.jpg"
            shutil.copy(last_image, result_dir / result_filename)
            
            print(f"Processed {id_folder.name}: sty{sty_idx} using {last_image.name} save as {result_filename}")
            sty_idx += 1

if __name__ == "__main__":
    # You can change these paths as needed
    workspace_root = "/Users/wangshiwen/Desktop/workspace/style-transfer-agent"
    src = os.path.join(workspace_root, "result_gemini_pro")
    dst = os.path.join(workspace_root, "pro_eval_dataset")
    
    if os.path.exists(src):
        create_eval_set(src, dst)
        print(f"\nFinished! Evaluation dataset created at: {dst}")
    else:
        print(f"Source directory {src} does not exist.")
