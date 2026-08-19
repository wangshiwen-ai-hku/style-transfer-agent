import os
import json
import csv
from pathlib import Path
from collections import defaultdict

def load_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {}

def analyze_experiment(exp_dir):
    logs_dir = exp_dir / "logs"
    if not logs_dir.exists():
        return None

    stats = {
        "skill_selection": {"duration": 0, "count": 0, "tokens": 0},
        "orchestration": {"duration": 0, "count": 0, "tokens": 0},
        "functional_agents": {"duration": 0, "count": 0, "tokens": 0},
        "reflection": {"duration": 0, "count": 0, "tokens": 0}
    }

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
        if "skill_selection" in name:
            category = "skill_selection"
        elif "system_orchestration" in name:
            category = "orchestration"
        elif "reflection" in name:
            category = "reflection"
        else:
            category = "functional_agents"

        stats[category]["duration"] += data.get("duration_seconds", 0)
        stats[category]["count"] += 1
        stats[category]["tokens"] += data.get("total_tokens", 0)

    return stats

def run_analysis(base_dir_name):
    base_dir = Path(base_dir_name)
    if not base_dir.exists():
        print(f"Directory {base_dir_name} not found.")
        return

    all_stats = []
    print(f"Analyzing {base_dir_name}...")

    # Reuse the traversal logic from analyze_stages_consumption.py
    for first_level_dir in sorted(base_dir.iterdir()):
        if not first_level_dir.is_dir():
            continue
        
        if (first_level_dir / "logs").exists():
            stats = analyze_experiment(first_level_dir)
            if stats:
                all_stats.append(stats)
        else:
            for exp_dir in sorted(first_level_dir.iterdir()):
                if not exp_dir.is_dir():
                    continue
                if (exp_dir / "logs").exists():
                    stats = analyze_experiment(exp_dir)
                    if stats:
                        all_stats.append(stats)

    if not all_stats:
        print(f"No valid results found in {base_dir_name}.")
        return

    num_exps = len(all_stats)
    
    # Calculate averages
    summary_rows = [
        ["Category", "Average Duration (s)", "Average Tokens", "Average Calls"],
        ["Skill Selection", 
         f"{sum(s['skill_selection']['duration'] for s in all_stats) / num_exps:.2f}", 
         f"{sum(s['skill_selection']['tokens'] for s in all_stats) / num_exps:.2f}",
         f"{sum(s['skill_selection']['count'] for s in all_stats) / num_exps:.2f}"],
        ["Orchestration", 
         f"{sum(s['orchestration']['duration'] for s in all_stats) / num_exps:.2f}", 
         f"{sum(s['orchestration']['tokens'] for s in all_stats) / num_exps:.2f}",
         f"{sum(s['orchestration']['count'] for s in all_stats) / num_exps:.2f}"],
        ["Functional Agents (MAS Execution)", 
         f"{sum(s['functional_agents']['duration'] for s in all_stats) / num_exps:.2f}", 
         f"{sum(s['functional_agents']['tokens'] for s in all_stats) / num_exps:.2f}",
         f"{sum(s['functional_agents']['count'] for s in all_stats) / num_exps:.2f}"],
        ["Reflection", 
         f"{sum(s['reflection']['duration'] for s in all_stats) / num_exps:.2f}", 
         f"{sum(s['reflection']['tokens'] for s in all_stats) / num_exps:.2f}",
         f"{sum(s['reflection']['count'] for s in all_stats) / num_exps:.2f}"],
        ["Total", 
         f"{sum(sum(cat['duration'] for cat in s.values()) for s in all_stats) / num_exps:.2f}", 
         f"{sum(sum(cat['tokens'] for cat in s.values()) for s in all_stats) / num_exps:.2f}",
         f"{sum(sum(cat['count'] for cat in s.values()) for s in all_stats) / num_exps:.2f}"]
    ]

    output_file = f"stat/agent_types_consumption_summary_{base_dir_name}.csv"
    os.makedirs("stat", exist_ok=True)
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(summary_rows)

    print(f"Saved summary to {output_file}")

if __name__ == "__main__":
    run_analysis("result_for_stat")
