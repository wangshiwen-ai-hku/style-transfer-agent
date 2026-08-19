import os
import json
import csv
from pathlib import Path
from collections import defaultdict
from PIL import Image

def load_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        # print(f"Warning: Failed to load {file_path}: {e}")
        return {}

def estimate_image_tokens(image_path):
    """Estimate Gemini image tokens based on size."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            if width <= 512 and height <= 512:
                return 85
            elif width <= 1024 and height <= 1024:
                return 170
            else:
                return 340
    except Exception:
        return 0

def analyze_experiment(exp_dir, base_dir):
    logs_dir = exp_dir / "logs"
    if not logs_dir.exists():
        return None

    stats = {
        "plan": {"duration": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "count": 0},
        "execute": {"duration": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "count": 0},
        "reflection": {"duration": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "count": 0}
    }

    # Find all LLM call report JSONs
    json_files = list(logs_dir.glob("*.json"))
    if not json_files:
        return None

    for json_file in json_files:
        if json_file.name == "resource_constraints.json":
            continue
        
        data = load_json_file(json_file)
        if not data or "duration_seconds" not in data:
            continue

        name = json_file.name.lower()
        if "system_orchestration" in name:
            stage = "plan"
        elif "reflection" in name:
            stage = "reflection"
        else:
            stage = "execute"

        stats[stage]["duration"] += data.get("duration_seconds", 0)
        stats[stage]["input_tokens"] += data.get("input_tokens", {}).get("total", 0)
        stats[stage]["output_tokens"] += data.get("output_tokens", {}).get("total", 0)
        stats[stage]["total_tokens"] += data.get("total_tokens", 0)
        stats[stage]["count"] += 1

    # If no calls were recorded, skip
    if stats["plan"]["count"] + stats["execute"]["count"] + stats["reflection"]["count"] == 0:
        return None

    # Count generated images
    # All images NOT image_1.* or image_2.*
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    generated_images = []
    total_gen_image_tokens = 0
    
    for item in exp_dir.iterdir():
        if item.is_file() and item.suffix.lower() in image_extensions:
            if not (item.stem == "image_1" or item.stem == "image_2"):
                generated_images.append(item.name)
                total_gen_image_tokens += estimate_image_tokens(item)

    return {
        "exp_path": str(exp_dir.relative_to(base_dir)),
        "plan_duration": stats["plan"]["duration"],
        "plan_tokens": stats["plan"]["total_tokens"],
        "plan_calls": stats["plan"]["count"],
        "execute_duration": stats["execute"]["duration"],
        "execute_tokens": stats["execute"]["total_tokens"],
        "execute_calls": stats["execute"]["count"],
        "reflection_duration": stats["reflection"]["duration"],
        "reflection_tokens": stats["reflection"]["total_tokens"],
        "reflection_calls": stats["reflection"]["count"],
        "total_duration": stats["plan"]["duration"] + stats["execute"]["duration"] + stats["reflection"]["duration"],
        "total_tokens": stats["plan"]["total_tokens"] + stats["execute"]["total_tokens"] + stats["reflection"]["total_tokens"],
        "total_calls": stats["plan"]["count"] + stats["execute"]["count"] + stats["reflection"]["count"],
        "gen_images_count": len(generated_images),
        "gen_images": ", ".join(generated_images),
        "gen_image_tokens": total_gen_image_tokens
    }

def run_analysis(base_dir_name):
    base_dir = Path(base_dir_name)
    if not base_dir.exists():
        print(f"Directory {base_dir_name} not found.")
        return

    all_results = []
    print(f"Analyzing {base_dir_name}...")

    # Traverse subdirectories (e.g., result_for_rebuttal/8/ or result_for_stat/1/)
    for first_level_dir in sorted(base_dir.iterdir()):
        if not first_level_dir.is_dir():
            continue
        
        # Check if this directory is itself an experiment or contains experiments
        if (first_level_dir / "logs").exists():
            # It's an experiment directory directly in base_dir
            result = analyze_experiment(first_level_dir, base_dir)
            if result:
                result["style_id"] = "root"
                all_results.append(result)
        else:
            # It's a style_id folder containing experiments
            for exp_dir in sorted(first_level_dir.iterdir()):
                if not exp_dir.is_dir():
                    continue
                if (exp_dir / "logs").exists():
                    result = analyze_experiment(exp_dir, base_dir)
                    if result:
                        result["style_id"] = first_level_dir.name
                        all_results.append(result)

    if not all_results:
        print(f"No valid results found in {base_dir_name}.")
        return

    # Output files
    stats_file = f"stat/stage_resource_consumption_stats_{base_dir_name}.csv"
    summary_file = f"stat/stage_resource_consumption_summary_{base_dir_name}.csv"

    # Save detailed stats
    fieldnames = [
        "style_id", "exp_path", 
        "plan_duration", "plan_tokens", "plan_calls",
        "execute_duration", "execute_tokens", "execute_calls",
        "reflection_duration", "reflection_tokens", "reflection_calls",
        "total_duration", "total_tokens", "total_calls",
        "gen_images_count", "gen_image_tokens", "gen_images"
    ]

    with open(stats_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # Save summary
    num_exps = len(all_results)
    summary_rows = [
        ["Stage", "Average Duration (s)", "Average Tokens", "Average API Calls"],
        ["Plan", 
         f"{sum(r['plan_duration'] for r in all_results) / num_exps:.2f}", 
         f"{sum(r['plan_tokens'] for r in all_results) / num_exps:.2f}",
         f"{sum(r['plan_calls'] for r in all_results) / num_exps:.2f}"],
        ["Execute", 
         f"{sum(r['execute_duration'] for r in all_results) / num_exps:.2f}", 
         f"{sum(r['execute_tokens'] for r in all_results) / num_exps:.2f}",
         f"{sum(r['execute_calls'] for r in all_results) / num_exps:.2f}"],
        ["Reflection", 
         f"{sum(r['reflection_duration'] for r in all_results) / num_exps:.2f}", 
         f"{sum(r['reflection_tokens'] for r in all_results) / num_exps:.2f}",
         f"{sum(r['reflection_calls'] for r in all_results) / num_exps:.2f}"],
        ["Total", 
         f"{sum(r['total_duration'] for r in all_results) / num_exps:.2f}", 
         f"{sum(r['total_tokens'] for r in all_results) / num_exps:.2f}",
         f"{sum(r['total_calls'] for r in all_results) / num_exps:.2f}"]
    ]

    with open(summary_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(summary_rows)

    print(f"Saved stats to {stats_file}")
    print(f"Saved summary to {summary_file}")

def main():
    # Process both directories
    run_analysis("result_for_rebuttal")
    # run_analysis("result_for_rebuttal_qwen")

if __name__ == "__main__":
    main()
