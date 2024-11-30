LLAMA_FACTORY_PATH=/cpfs/user/lengxue/code/multimodel/LLaMA-Factory
ARENA_HARD_PATH=/cpfs/user/lengxue/code/multimodel/Evaluation/arena-hard-auto
ALPACA_PATH=/cpfs/user/lengxue/code/multimodel/Evaluation/alpaca_eval

cd $ALPACA_PATH
pip install -e .


########################################### 输入参数 - 自定义设置 ###########################################
dataset_name=alpaca-eval-2
model_name=llama3.1-8b_full_sft_base
deploy_yaml=/cpfs/user/lengxue/code/multimodel/LLaMA-Factory/examples/my_exps/llama3.1-8b/base/deploy/llama3.1_full_sft_ds3_vllm.yaml
model_dir=/cpfs/user/lengxue/code/multimodel/save_models/LF/$model_name
iter="" # 1000  ->  checkpoint-1000


api_yaml=/cpfs/user/lengxue/code/multimodel/LLaMA-Factory/examples/my_exps/llama3.1-8b/base/eval/$dataset_name/api_config.yaml
gen_answer_yaml=/cpfs/user/lengxue/code/multimodel/LLaMA-Factory/examples/my_exps/llama3.1-8b/base/eval/$dataset_name/gen_answer_config.yaml


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

#结果文件
answer_list=()

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

  this_answer_file=$ARENA_HARD_PATH"/data/"$dataset_name"/"$output_prefix"/model_answer/"$this_model_name".jsonl"
  answer_list+=("$this_answer_file")
#   echo $answer_list
  

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
  
done

#处理结果文件，适应alpaca-eval 评估的格式
alpaca_eval_answer_path=$ARENA_HARD_PATH"/data/"$dataset_name"/"$output_prefix"/model_answer_alpaca_eval/"
mkdir -p $alpaca_eval_answer_path

cd $ALPACA_PATH
# pip install -e .

for answer_file in "${answer_list[@]}"; do
  echo "Processing checkpoint: $answer_file"

  file_name=$(basename "$answer_file")
  alpaca_eval_answer_file=$alpaca_eval_answer_path$file_name
  # 使用条件语句检查后缀并重命名文件
if [[ "$alpaca_eval_answer_file" == *.jsonl ]]; then
    # 使用参数扩展去除 .jsonl 后缀
    base_name_this="${alpaca_eval_answer_file%.jsonl}"
    # 构造新的文件名
    alpaca_eval_answer_file="${base_name_this}.json"
    # echo $alpaca_eval_answer_file
fi
#   echo $alpaca_eval_answer_file
  python scripts/convert_to_alpaca_eval.py --raw_file $ARENA_HARD_PATH/data/$dataset_name/question.jsonl  --input_file $answer_file --output_file $alpaca_eval_answer_file
  
  judge_output_path=$ARENA_HARD_PATH"/data/"$dataset_name"/"$output_prefix"/model_judgment"
  log_path=$judge_output_path"/logs"
  mkdir -p $log_path
  python src/alpaca_eval/main.py evaluate \
    --model_outputs $alpaca_eval_answer_file \
    --reference_outputs $ALPACA_PATH/results/gpt4_1106_preview/model_outputs.json \
    --annotators_config $ALPACA_PATH/src/alpaca_eval/evaluators_configs/agi-gpt4o-ptu-lengxue/configs.yaml \
    --output_path $judge_output_path \
    --is_overwrite_leaderboard True \
  > $log_path/${file_name}.log 2>&1

done




