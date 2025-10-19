#!/bin/bash

# This script runs the general agent to perform a style fusion test.
content=$1
style=$2
directly=$4
task_type=$3

if [ $directly == "directly" ]; then
  python run_agent.py \
  --task_type $task_type \
  --images "data/content/$content" "data/style/$style" \
  --prompt "第一个图是内容图，第二个图是风格图，把内容图的内容用风格转绘。"\
  --result_dir "result_exp_general" \
  --directly
else
  python run_agent.py \
  --task_type $task_type \
  --images "data/content/$content" "data/style/$style" \
  --prompt "第一个图是内容图，第二个图是风格图，把内容图的内容用风格转绘。"\
  --result_dir "result_exp_general" 
fi
