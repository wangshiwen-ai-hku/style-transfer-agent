import os
import re

def analyze_reflections(root_dir):
    experiment_results = []
    
    # regex to match reflection files
    reflection_file_re = re.compile(r'reflection_(\d+)_prompt\.txt')
    # regex to extract Reflection Count and Is Satisfied
    count_re = re.compile(r'Reflection Count:\s*(\d+)')
    satisfied_re = re.compile(r'Is Satisfied:\s*(True|False)', re.IGNORECASE)

    # Walk through the directory
    for root, dirs, files in os.walk(root_dir):
        reflection_files = []
        for f in files:
            match = reflection_file_re.match(f)
            if match:
                reflection_files.append((int(match.group(1)), f))
        
        if reflection_files:
            # Get the file with the highest index
            max_index, latest_file = max(reflection_files)
            file_path = os.path.join(root, latest_file)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f_in:
                    content = f_in.read()
                    
                    count_match = count_re.search(content)
                    satisfied_match = satisfied_re.search(content)
                    
                    if count_match and satisfied_match:
                        count = int(count_match.group(1))
                        satisfied = satisfied_match.group(1).lower() == 'true'
                        experiment_results.append({
                            'path': root,
                            'count': count,
                            'satisfied': satisfied
                        })
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    if not experiment_results:
        print("No reflection results found.")
        return

    total_count = len(experiment_results)
    sum_reflection = sum(r['count'] for r in experiment_results)
    max_reflection = max(r['count'] for r in experiment_results)
    avg_reflection = sum_reflection / total_count
    
    reached_3_and_failed = [r for r in experiment_results if r['count'] >= 3 and not r['satisfied']]
    fail_ratio = len(reached_3_and_failed) / total_count

    print(f"Total experiments: {total_count}")
    print(f"Average Reflection Count: {avg_reflection:.4f}")
    print(f"Max Reflection Count: {max_reflection}")
    print(f"Experiments reaching 3+ and still False: {len(reached_3_and_failed)}")
    print(f"Ratio (3+ and False): {fail_ratio:.2%}")

if __name__ == "__main__":
    target_dir = "/Users/wangshiwen/Desktop/workspace/style-transfer-agent/result_for_stat"
    print(f"Analyzing directory: {target_dir}")
    analyze_reflections(target_dir)
