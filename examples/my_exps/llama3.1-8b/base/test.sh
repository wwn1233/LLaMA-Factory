LLAMA_FACTORY_PATH=/cpfs/user/lengxue/code/multimodel/LLaMA-Factory
ARENA_HARD_PATH=/cpfs/user/lengxue/code/multimodel/Evaluation/arena-hard-auto


########################################### 输入参数 - 自定义设置 ###########################################
dataset_name=arena-hard-v0.1_all
model_name=llama3.1-8b_full_sft_base
deploy_yaml=/cpfs/user/lengxue/code/multimodel/LLaMA-Factory/examples/my_exps/llama3.1-8b/base/deploy/llama3.1_full_sft_ds3_vllm.yaml
model_dir=/cpfs/user/lengxue/code/multimodel/save_models/LF/$model_name
iter="" # 1000  ->  checkpoint-1000


api_yaml=/cpfs/user/lengxue/code/multimodel/LLaMA-Factory/examples/my_exps/llama3.1-8b/base/eval/arena-hard-v0.1_all/api_config.yaml
gen_answer_yaml=/cpfs/user/lengxue/code/multimodel/LLaMA-Factory/examples/my_exps/llama3.1-8b/base/eval/arena-hard-v0.1_all/gen_answer_config.yaml
judge_yaml=/cpfs/user/lengxue/code/multimodel/LLaMA-Factory/examples/my_exps/llama3.1-8b/base/eval/arena-hard-v0.1_all/judge_config.yaml

output_prefix="ablation_para"
gen_parallel=2


########################################### 输入参数 - 自定义设置 ###########################################

#如果iter为空，则遍历 model_dir 下前缀为”checkpoint-“的所有文件夹
# 初始化 checkpoint 列表
checkpoint_list=()

# 获取 checkpoint 列表
if [ -z "$iter" ]; then
  # 遍历所有符合条件的 checkpoints
  for checkpoint in "$model_dir"/checkpoint-*; do
    if [ -d "$checkpoint" ]; then
      checkpoint_list+=($(basename "$checkpoint"))
    fi
  done
  
  # 检查列表是否为空
  if [ ${#checkpoint_list[@]} -eq 0 ]; then
    echo "No checkpoints found in $model_dir."
    exit 1
  fi
else
  checkpoint_name="checkpoint-$iter"
  if [ ! -d "$model_dir/$checkpoint_name" ]; then
    echo "Specified checkpoint $checkpoint_name does not exist in $model_dir."
    exit 1
  fi
  checkpoint_list+=("$checkpoint_name")
fi

# 打印找到的所有 checkpoints
echo "Found the following checkpoints:"
for checkpoint in "${checkpoint_list[@]}"; do
  echo "$checkpoint"
done

# 继续对每个 checkpoint 进行处理
for checkpoint in "${checkpoint_list[@]}"; do
  echo "Processing checkpoint: $checkpoint"

  ## 部署vllm模型
  cd $LLAMA_FACTORY_PATH
  # 使用sed命令替换model_name_or_path的值
  sed -i.bak "s|^\(model_name_or_path:\).*|\1 $model_dir/$checkpoint|" "$deploy_yaml"
#   CUDA_VISIBLE_DEVICES=0,1 API_PORT=8000 llamafactory-cli api $deploy_yaml
  output_log_path=$ARENA_HARD_PATH/data/$dataset_name/$output_prefix/logs
  mkdir -p $output_log_path
  nohup bash -c "CUDA_VISIBLE_DEVICES=0,1 API_PORT=8000 llamafactory-cli api $deploy_yaml" > $output_log_path/$checkpoint"_output.log" 2>&1 &
  
  sleep 120
  max_attempts=10
  attempt=0 
  # 持续检查 llamafactory-cli api 是否在运行，最多尝试 10 次
  while [ "$attempt" -lt "$max_attempts" ]; do
    # 统计 llamafactory-cli api 进程的数量
    process_count=$(ps aux | grep '[l]lamafactory-cli api' | wc -l)

    # 检查是否有进程在运行
    if [ "$process_count" -gt 0 ]; then
      echo "Detected $process_count instance(s) of llamafactory-cli api running."
      break
    else
      echo "Attempt $((attempt + 1)): No llamafactory-cli api processes detected. Checking again in 5 seconds..."
    fi

    # 增加尝试次数计数器
    attempt=$((attempt + 1))

    # 等待 5 秒钟后再次检查
    sleep 5
  done
  # 检查是否是因为达到最大尝试次数而退出循环
  if [ "$attempt" -eq "$max_attempts" ]; then
    echo "Reached maximum attempts without detecting the process."
    exit 1
  fi

  ## arena hard
  cd $ARENA_HARD_PATH
  this_model_name=$model_name"_"$checkpoint
  # Use sed to replace occurrences of TEMPLATE with "sft-1.30"
  sed -i "s/TEMPLATE/${this_model_name}/g" "$api_yaml"
  sed -i "s/TEMPLATE/${this_model_name}/g" "$gen_answer_yaml"

  # 生成答案
  python gen_answer.py --setting-file $gen_answer_yaml   --endpoint-file $api_yaml  --output_prefix $output_prefix

  # set back 
  sed -i "s/${this_model_name}/TEMPLATE/g" "$api_yaml"
  sed -i "s/${this_model_name}/TEMPLATE/g" "$gen_answer_yaml"

  ## 销毁 llamafactory-cli api 进程
  pids=$(ps aux | grep '[l]lamafactory-cli api' | awk '{print $2}')
  # 检查是否找到相关进程
  if [ -z "$pids" ]; then
    echo "No llamafactory-cli api processes found."
  else
    # 结束找到的进程
    echo "Killing llamafactory-cli api processes: $pids"
    kill $pids
    sleep 10
    echo "Processes terminated."
  fi
  fuser -v /dev/nvidia*|awk -F " " '{print $0}' >/tmp/pid.file
  while read pid ; do kill -9 $pid; done </tmp/pid.file
  
  break
done

cd $ARENA_HARD_PATH
## arena hard - judge
python gen_judgment.py --setting_file $judge_yaml   --endpoint_file $api_yaml  --output_prefix $output_prefix


## arena hard - show results
python show_result.py --output_prefix $output_prefix



