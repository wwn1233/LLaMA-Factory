## 镜像 ali-sg-acr-registry-vpc.ap-southeast-1.cr.aliyuncs.com/xhs-llm/lengxue1:lx-v1
 
 echo -e "111.223.64.33    ci.xiaohongshu.com\n111.223.64.25    ci.xiaohongshu.com" | sudo tee -a /etc/hosts 
 
# ############################# 环境准备 #######################################################################
cd /mnt/workspace/
ln -s /cpfs/user  .
export PATH="/cpfs/user/lengxue/envir/anaconda310/bin:$PATH"
conda init
 
# source ~/.bashrc
# source activate   #/cpfs/user/lengxue/envir/anaconda310
# conda activate #/cpfs/user/lengxue/envir/anaconda310

cd /cpfs/user/lengxue/code/multimodel/LLaMA-Factory

# pip install -e ".[torch,metrics]"
# pip3 install deepspeed

FORCE_TORCHRUN=1 llamafactory-cli train examples/my_exps/llama3.1-8b/base/llama3.1_full_sft_ds3.yaml

