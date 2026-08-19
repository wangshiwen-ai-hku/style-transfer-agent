#!/usr/bin/env python3
"""
统计 result_for_stat 文件夹下所有实验的资源消耗
只统计 logs 文件夹下至少有 2 个 JSON 文件的实验
"""

import json
import os
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from PIL import Image


def load_json_file(file_path: Path) -> Dict:
    """加载 JSON 文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {file_path}: {e}")
        return {}


def estimate_text_tokens(text: str) -> int:
    """估算文本 token 数量（Gemini 模型）"""
    if not text:
        return 0
    # Approximate: ~4 characters per token for English text
    return max(1, len(text) // 4)


def estimate_image_tokens(image_path: Path) -> Tuple[int, int, int]:
    """
    根据图像尺寸估算 token 数量（Gemini 模型）
    Returns: (width, height, tokens)
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            
            # Gemini token estimation based on image size
            if width <= 512 and height <= 512:
                tokens = 85
            elif width <= 1024 and height <= 1024:
                tokens = 170
            else:
                tokens = 340
            
            return width, height, tokens
    except Exception as e:
        print(f"Warning: Failed to estimate tokens for {image_path}: {e}")
        return 0, 0, 0


def calculate_direct_call_tokens(exp_dir: Path, output_image_tokens: int) -> Dict:
    """
    计算假想的直接调用 LLM 的 token 消耗
    Prompt: "Transfer the style of the image 2 to the image 1. e.g., color, texture, decorations"
    输入: prompt tokens + image_1 tokens + image_2 tokens (2个输入图像)
    输出: 最终生成图像的 tokens (1个输出图像)
    """
    prompt = "Transfer the style of the image 2 to the image 1. e.g., color, texture, decorations"
    prompt_tokens = estimate_text_tokens(prompt)
    
    # 查找输入图像（2个图像）
    image_1_path = None
    image_2_path = None
    
    for ext in ['.jpg', '.png', '.jpeg']:
        if not image_1_path:
            img1 = exp_dir / f"image_1{ext}"
            if img1.exists():
                image_1_path = img1
        if not image_2_path:
            img2 = exp_dir / f"image_2{ext}"
            if img2.exists():
                image_2_path = img2
    
    # 计算输入图像的 tokens（2个图像）
    image_1_tokens = 0
    image_2_tokens = 0
    
    if image_1_path:
        _, _, image_1_tokens = estimate_image_tokens(image_1_path)
    if image_2_path:
        _, _, image_2_tokens = estimate_image_tokens(image_2_path)
    
    # 输入 tokens: prompt + 2个输入图像
    direct_input_tokens = prompt_tokens + image_1_tokens + image_2_tokens
    
    # 输出 tokens: 1个输出图像
    direct_output_tokens = output_image_tokens
    
    # 总 tokens: 输入 + 输出
    direct_total_tokens = direct_input_tokens + direct_output_tokens
    
    return {
        'direct_call_prompt_tokens': prompt_tokens,
        'direct_call_image_1_tokens': image_1_tokens,
        'direct_call_image_2_tokens': image_2_tokens,
        'direct_call_input_tokens': direct_input_tokens,
        'direct_call_output_tokens': direct_output_tokens,
        'direct_call_total_tokens': direct_total_tokens,
    }


def find_last_generated_image(exp_dir: Path) -> Optional[Tuple[Path, int, int, int]]:
    """
    找到实验目录下最后一个保存的图像文件（排除输入图像）
    Returns: (image_path, width, height, tokens) or None
    """
    # 排除的输入图像文件名模式
    excluded_patterns = ['image_1.jpg', 'image_2.jpg', 'image_1.png', 'image_2.png']
    
    # 查找所有图像文件
    image_extensions = ['.png', '.jpg', '.jpeg']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(exp_dir.glob(f"*{ext}"))
    
    # 过滤掉输入图像和 logs 目录下的文件
    generated_images = [
        img for img in image_files
        if img.name not in excluded_patterns and img.parent == exp_dir
    ]
    
    if not generated_images:
        return None
    
    # 按修改时间排序，获取最后一个
    last_image = max(generated_images, key=lambda p: p.stat().st_mtime)
    
    width, height, tokens = estimate_image_tokens(last_image)
    return (last_image, width, height, tokens)


def analyze_logs_directory(logs_dir: Path) -> Optional[Dict]:
    """分析单个 logs 目录下的所有 JSON 文件"""
    json_files = list(logs_dir.glob("*.json"))
    
    if len(json_files) < 2:
        return None
    
    stats = {
        'call_count': 0,
        'total_duration_seconds': 0.0,
        'total_retry_count': 0,
        'input_tokens_text': 0,
        'input_tokens_images_count': 0,
        'input_tokens_images': 0,
        'input_tokens_total': 0,
        'output_tokens_text': 0,
        'output_tokens_images_count': 0,
        'output_tokens_images': 0,
        'output_tokens_total': 0,
        'total_tokens': 0,
    }
    
    for json_file in json_files:
        data = load_json_file(json_file)
        if not data:
            continue
        
        stats['call_count'] += 1
        
        # 累计时长
        if 'duration_seconds' in data:
            stats['total_duration_seconds'] += float(data['duration_seconds'])
        
        # 累计重试次数
        if 'retry_count' in data:
            stats['total_retry_count'] += int(data['retry_count'])
        
        # 累计输入 tokens
        if 'input_tokens' in data:
            input_tokens = data['input_tokens']
            stats['input_tokens_text'] += int(input_tokens.get('text', 0))
            stats['input_tokens_images_count'] += int(input_tokens.get('images_count', 0))
            stats['input_tokens_images'] += int(input_tokens.get('images', 0))
            stats['input_tokens_total'] += int(input_tokens.get('total', 0))
        
        # 累计输出 tokens
        if 'output_tokens' in data:
            output_tokens = data['output_tokens']
            stats['output_tokens_text'] += int(output_tokens.get('text', 0))
            stats['output_tokens_images_count'] += int(output_tokens.get('images_count', 0))
            stats['output_tokens_images'] += int(output_tokens.get('images', 0))
            stats['output_tokens_total'] += int(output_tokens.get('total', 0))
        
        # 累计总 tokens
        if 'total_tokens' in data:
            stats['total_tokens'] += int(data['total_tokens'])
    
    return stats


def scan_result_for_stat(base_dir: Path) -> List[Dict]:
    """扫描 result_for_stat 目录，收集所有符合条件的实验统计"""
    results = []
    
    # 遍历所有子目录（如 1/, 107/, 11/ 等）
    for style_dir in sorted(base_dir.iterdir()):
        if not style_dir.is_dir():
            continue
        
        # 遍历每个 style 目录下的实验目录（如 gpt5_gpt_20251107_154430/）
        for exp_dir in sorted(style_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            
            logs_dir = exp_dir / "logs"
            if not logs_dir.exists() or not logs_dir.is_dir():
                continue
            
            stats = analyze_logs_directory(logs_dir)
            if stats is None:
                continue
            
            # 查找最后一个生成的图像
            image_info = find_last_generated_image(exp_dir)
            image_path = image_info[0].name if image_info else None
            image_width = image_info[1] if image_info else 0
            image_height = image_info[2] if image_info else 0
            image_tokens = image_info[3] if image_info else 0
            
            # 计算假想的直接调用 token（baseline）
            direct_call_stats = calculate_direct_call_tokens(exp_dir, image_tokens)
            
            # Agent 工作流的统计
            call_count = stats['call_count']
            agent_total_tokens = stats['total_tokens']
            agent_input_tokens = stats['input_tokens_total']
            agent_output_tokens = stats['output_tokens_total']
            agent_duration = stats['total_duration_seconds']
            
            # Baseline（直接调用）的统计
            baseline_total_tokens = direct_call_stats['direct_call_total_tokens']
            baseline_input_tokens = direct_call_stats['direct_call_input_tokens']
            baseline_output_tokens = direct_call_stats['direct_call_output_tokens']
            
            # 计算平均每个图像 token 的资源消耗
            tokens_per_image_token_agent = agent_total_tokens / image_tokens if image_tokens > 0 else 0
            tokens_per_image_token_baseline = baseline_total_tokens / image_tokens if image_tokens > 0 else 0
            
            # 添加实验标识信息和统计
            result = {
                'style_id': style_dir.name,
                'experiment_name': exp_dir.name,
                'experiment_path': f"{style_dir.name}/{exp_dir.name}",
                'last_image': image_path,
                'image_width': image_width,
                'image_height': image_height,
                'image_tokens': image_tokens,
                # Agent 工作流统计
                'agent_call_count': call_count,
                'agent_total_duration_seconds': agent_duration,
                'agent_input_tokens': agent_input_tokens,
                'agent_output_tokens': agent_output_tokens,
                'agent_total_tokens': agent_total_tokens,
                'agent_avg_duration_per_call': agent_duration / call_count if call_count > 0 else 0,
                'agent_avg_input_tokens_per_call': agent_input_tokens / call_count if call_count > 0 else 0,
                'agent_avg_output_tokens_per_call': agent_output_tokens / call_count if call_count > 0 else 0,
                'agent_avg_total_tokens_per_call': agent_total_tokens / call_count if call_count > 0 else 0,
                'agent_tokens_per_image_token': tokens_per_image_token_agent,
                # Baseline（直接调用）统计
                **direct_call_stats,
                'baseline_total_tokens': baseline_total_tokens,
                'baseline_input_tokens': baseline_input_tokens,
                'baseline_output_tokens': baseline_output_tokens,
                'baseline_tokens_per_image_token': tokens_per_image_token_baseline,
                # 对比统计
                'token_ratio_agent_to_baseline': agent_total_tokens / baseline_total_tokens if baseline_total_tokens > 0 else 0,
                'input_token_ratio_agent_to_baseline': agent_input_tokens / baseline_input_tokens if baseline_input_tokens > 0 else 0,
                'output_token_ratio_agent_to_baseline': agent_output_tokens / baseline_output_tokens if baseline_output_tokens > 0 else 0,
            }
            results.append(result)
    
    return results


def write_csv(results: List[Dict], output_file: Path):
    """将结果写入 CSV 文件"""
    if not results:
        print("No results to write.")
        return
    
    fieldnames = [
        'style_id',
        'experiment_name',
        'experiment_path',
        'last_image',
        'image_width',
        'image_height',
        'image_tokens',
        # Agent 工作流详细统计
        'agent_call_count',
        'agent_total_duration_seconds',
        'agent_input_tokens',
        'agent_output_tokens',
        'agent_total_tokens',
        'agent_avg_duration_per_call',
        'agent_avg_input_tokens_per_call',
        'agent_avg_output_tokens_per_call',
        'agent_avg_total_tokens_per_call',
        'agent_tokens_per_image_token',
        # Baseline（直接调用）统计
        'direct_call_prompt_tokens',
        'direct_call_image_1_tokens',
        'direct_call_image_2_tokens',
        'direct_call_input_tokens',
        'direct_call_output_tokens',
        'direct_call_total_tokens',
        'baseline_input_tokens',
        'baseline_output_tokens',
        'baseline_total_tokens',
        'baseline_tokens_per_image_token',
        # 对比统计
        'token_ratio_agent_to_baseline',
        'input_token_ratio_agent_to_baseline',
        'output_token_ratio_agent_to_baseline',
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results written to {output_file}")
    print(f"Total experiments: {len(results)}")


def write_summary_csv(results: List[Dict], output_file: Path):
    """将汇总统计写入 CSV 文件 - 简化的对比表格"""
    if not results:
        print("No results to write summary.")
        return
    
    total_experiments = len(results)
    total_image_tokens = sum(r['image_tokens'] for r in results)
    
    # Agent 工作流统计
    total_agent_duration = sum(r['agent_total_duration_seconds'] for r in results)
    total_agent_input_tokens = sum(r['agent_input_tokens'] for r in results)
    total_agent_output_tokens = sum(r['agent_output_tokens'] for r in results)
    total_agent_total_tokens = sum(r['agent_total_tokens'] for r in results)
    
    # Baseline（直接调用）统计
    total_baseline_input_tokens = sum(r['baseline_input_tokens'] for r in results)
    total_baseline_output_tokens = sum(r['baseline_output_tokens'] for r in results)
    total_baseline_total_tokens = sum(r['baseline_total_tokens'] for r in results)
    
    # Agent 工作流平均值（每个实验）
    avg_agent_duration_per_exp = total_agent_duration / total_experiments if total_experiments > 0 else 0
    avg_agent_input_tokens_per_exp = total_agent_input_tokens / total_experiments if total_experiments > 0 else 0
    avg_agent_output_tokens_per_exp = total_agent_output_tokens / total_experiments if total_experiments > 0 else 0
    avg_agent_total_tokens_per_exp = total_agent_total_tokens / total_experiments if total_experiments > 0 else 0
    
    # Baseline 平均值（每个实验）
    avg_baseline_duration_per_exp = 5.0  # Baseline 单次调用时间设为 5 秒
    avg_baseline_input_tokens_per_exp = total_baseline_input_tokens / total_experiments if total_experiments > 0 else 0
    avg_baseline_output_tokens_per_exp = total_baseline_output_tokens / total_experiments if total_experiments > 0 else 0
    avg_baseline_total_tokens_per_exp = total_baseline_total_tokens / total_experiments if total_experiments > 0 else 0
    
    # 平均每个图像 token 的资源消耗
    avg_agent_tokens_per_image_token = total_agent_total_tokens / total_image_tokens if total_image_tokens > 0 else 0
    avg_baseline_tokens_per_image_token = total_baseline_total_tokens / total_image_tokens if total_image_tokens > 0 else 0
    
    # 计算比率（Agent / Baseline）
    time_ratio = avg_agent_duration_per_exp / avg_baseline_duration_per_exp if avg_baseline_duration_per_exp > 0 else 0
    input_token_ratio = avg_agent_input_tokens_per_exp / avg_baseline_input_tokens_per_exp if avg_baseline_input_tokens_per_exp > 0 else 0
    output_token_ratio = avg_agent_output_tokens_per_exp / avg_baseline_output_tokens_per_exp if avg_baseline_output_tokens_per_exp > 0 else 0
    total_token_ratio = avg_agent_total_tokens_per_exp / avg_baseline_total_tokens_per_exp if avg_baseline_total_tokens_per_exp > 0 else 0
    token_per_image_token_ratio = avg_agent_tokens_per_image_token / avg_baseline_tokens_per_image_token if avg_baseline_tokens_per_image_token > 0 else 0
    
    # 创建对比表格 - 三行数据：Agent, Single LLM Call, Ratio
    summary_rows = [
        {
            'method': 'Agent',
            'time/s': round(avg_agent_duration_per_exp, 2),
            'input tokens': round(avg_agent_input_tokens_per_exp, 2),
            'output tokens': round(avg_agent_output_tokens_per_exp, 2),
            'total tokens': round(avg_agent_total_tokens_per_exp, 2),
            'token per image_token': round(avg_agent_tokens_per_image_token, 2)
        },
        {
            'method': 'Single LLM Call',
            'time/s': round(avg_baseline_duration_per_exp, 2),
            'input tokens': round(avg_baseline_input_tokens_per_exp, 2),
            'output tokens': round(avg_baseline_output_tokens_per_exp, 2),
            'total tokens': round(avg_baseline_total_tokens_per_exp, 2),
            'token per image_token': round(avg_baseline_tokens_per_image_token, 2)
        },
        {
            'method': 'Ratio (Agent/Baseline)',
            'time/s': round(time_ratio, 2),
            'input tokens': round(input_token_ratio, 2),
            'output tokens': round(output_token_ratio, 2),
            'total tokens': round(total_token_ratio, 2),
            'token per image_token': round(token_per_image_token_ratio, 2)
        }
    ]
    
    fieldnames = ['method', 'time/s', 'input tokens', 'output tokens', 'total tokens', 'token per image_token']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    
    print(f"Summary written to {output_file}")


def main():
    base_dir = Path(__file__).parent.parent / "result_for_stat"
    
    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist.")
        return
    
    print(f"Scanning {base_dir}...")
    results = scan_result_for_stat(base_dir)
    
    output_file = Path(__file__).parent.parent / "stat" / "resource_consumption_stats.csv"
    write_csv(results, output_file)
    
    summary_file = Path(__file__).parent.parent /"stat" / "resource_consumption_summary.csv"
    write_summary_csv(results, summary_file)
    
    # 打印汇总统计
    if results:
        total_experiments = len(results)
        total_image_tokens = sum(r['image_tokens'] for r in results)
        
        # Agent 工作流统计
        total_agent_calls = sum(r['agent_call_count'] for r in results)
        total_agent_duration = sum(r['agent_total_duration_seconds'] for r in results)
        total_agent_input_tokens = sum(r['agent_input_tokens'] for r in results)
        total_agent_output_tokens = sum(r['agent_output_tokens'] for r in results)
        total_agent_total_tokens = sum(r['agent_total_tokens'] for r in results)
        
        # Baseline（直接调用）统计
        total_baseline_input_tokens = sum(r['baseline_input_tokens'] for r in results)
        total_baseline_output_tokens = sum(r['baseline_output_tokens'] for r in results)
        total_baseline_total_tokens = sum(r['baseline_total_tokens'] for r in results)
        
        # Agent 工作流平均值（每个实验）
        avg_agent_calls_per_exp = total_agent_calls / total_experiments if total_experiments > 0 else 0
        avg_agent_duration_per_exp = total_agent_duration / total_experiments if total_experiments > 0 else 0
        avg_agent_input_tokens_per_exp = total_agent_input_tokens / total_experiments if total_experiments > 0 else 0
        avg_agent_output_tokens_per_exp = total_agent_output_tokens / total_experiments if total_experiments > 0 else 0
        avg_agent_total_tokens_per_exp = total_agent_total_tokens / total_experiments if total_experiments > 0 else 0
        
        # Agent 工作流平均值（每次调用）
        avg_agent_duration_per_call = total_agent_duration / total_agent_calls if total_agent_calls > 0 else 0
        avg_agent_input_tokens_per_call = total_agent_input_tokens / total_agent_calls if total_agent_calls > 0 else 0
        avg_agent_output_tokens_per_call = total_agent_output_tokens / total_agent_calls if total_agent_calls > 0 else 0
        avg_agent_total_tokens_per_call = total_agent_total_tokens / total_agent_calls if total_agent_calls > 0 else 0
        
        # Baseline 平均值（每个实验）
        avg_baseline_input_tokens_per_exp = total_baseline_input_tokens / total_experiments if total_experiments > 0 else 0
        avg_baseline_output_tokens_per_exp = total_baseline_output_tokens / total_experiments if total_experiments > 0 else 0
        avg_baseline_total_tokens_per_exp = total_baseline_total_tokens / total_experiments if total_experiments > 0 else 0
        
        # 平均每个图像 token 的资源消耗
        avg_agent_tokens_per_image_token = total_agent_total_tokens / total_image_tokens if total_image_tokens > 0 else 0
        avg_baseline_tokens_per_image_token = total_baseline_total_tokens / total_image_tokens if total_image_tokens > 0 else 0
        
        # 对比比率
        token_ratio = total_agent_total_tokens / total_baseline_total_tokens if total_baseline_total_tokens > 0 else 0
        input_token_ratio = total_agent_input_tokens / total_baseline_input_tokens if total_baseline_input_tokens > 0 else 0
        output_token_ratio = total_agent_output_tokens / total_baseline_output_tokens if total_baseline_output_tokens > 0 else 0
        
        print("\n" + "="*70)
        print("RESOURCE CONSUMPTION COMPARISON: Agent Workflow vs Baseline")
        print("="*70)
        
        print(f"\nTotal experiments: {total_experiments}")
        print(f"Total image tokens: {total_image_tokens:,}")
        
        print("\n" + "-"*70)
        print("AGENT WORKFLOW STATISTICS")
        print("-"*70)
        print(f"Total LLM calls: {total_agent_calls:,}")
        print(f"Total duration: {total_agent_duration:.2f} seconds ({total_agent_duration/60:.2f} minutes)")
        print(f"Total input tokens: {total_agent_input_tokens:,}")
        print(f"Total output tokens: {total_agent_output_tokens:,}")
        print(f"Total tokens: {total_agent_total_tokens:,}")
        
        print("\n--- Average per Experiment (Agent Workflow) ---")
        print(f"Average LLM calls per experiment: {avg_agent_calls_per_exp:.2f}")
        print(f"Average duration per experiment: {avg_agent_duration_per_exp:.2f} seconds ({avg_agent_duration_per_exp/60:.2f} minutes)")
        print(f"Average input tokens per experiment: {avg_agent_input_tokens_per_exp:.2f}")
        print(f"Average output tokens per experiment: {avg_agent_output_tokens_per_exp:.2f}")
        print(f"Average total tokens per experiment: {avg_agent_total_tokens_per_exp:.2f}")
        
        print("\n--- Average per Call (Agent Workflow) ---")
        print(f"Average duration per call: {avg_agent_duration_per_call:.2f} seconds")
        print(f"Average input tokens per call: {avg_agent_input_tokens_per_call:.2f}")
        print(f"Average output tokens per call: {avg_agent_output_tokens_per_call:.2f}")
        print(f"Average total tokens per call: {avg_agent_total_tokens_per_call:.2f}")
        
        print("\n" + "-"*70)
        print("BASELINE (Direct LLM Call) STATISTICS")
        print("-"*70)
        print(f"Total input tokens: {total_baseline_input_tokens:,}")
        print(f"Total output tokens: {total_baseline_output_tokens:,}")
        print(f"Total tokens: {total_baseline_total_tokens:,}")
        
        print("\n--- Average per Experiment (Baseline) ---")
        print(f"Average input tokens per experiment: {avg_baseline_input_tokens_per_exp:.2f}")
        print(f"Average output tokens per experiment: {avg_baseline_output_tokens_per_exp:.2f}")
        print(f"Average total tokens per experiment: {avg_baseline_total_tokens_per_exp:.2f}")
        
        print("\n" + "-"*70)
        print("COMPARISON: Agent vs Baseline")
        print("-"*70)
        print(f"Token ratio (Agent/Baseline): {token_ratio:.2f}x")
        print(f"Input token ratio (Agent/Baseline): {input_token_ratio:.2f}x")
        print(f"Output token ratio (Agent/Baseline): {output_token_ratio:.2f}x")
        print(f"\nAgent uses {token_ratio:.2f}x more tokens than baseline")
        print(f"Agent uses {input_token_ratio:.2f}x more input tokens than baseline")
        print(f"Agent uses {output_token_ratio:.2f}x more output tokens than baseline")
        
        print("\n--- Average per Image Token ---")
        print(f"Agent tokens per image token: {avg_agent_tokens_per_image_token:.2f}")
        print(f"Baseline tokens per image token: {avg_baseline_tokens_per_image_token:.2f}")
        print(f"Ratio: {avg_agent_tokens_per_image_token / avg_baseline_tokens_per_image_token:.2f}x" if avg_baseline_tokens_per_image_token > 0 else "Ratio: N/A")


if __name__ == "__main__":
    main()

