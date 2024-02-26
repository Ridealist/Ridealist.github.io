---
published: true
layout: posts
title: 'Fine-tuning LLMs'
categories: 
  - llm
use_math: true
---

LLM을 파인튜닝하면서 공부한 내용들, 고려해야 할 사항들을 적어본다.



## 1. GPU Cloud 결정

- RunPod
- Vast.ai
	- [관련 링크](https://github.com/TrelisResearch/install-guides/blob/main/llm-notebook-setup.md)



### LLM Notebook Setup

A quick guide for getting set for fine-tuning or inference using a jupyter notebook.

You have a few options for running fine-tuning notebooks:
1. Hosted service (Recommended), e.g. Runpod or Vast.ai:
- Runpod one-click template [here](https://runpod.io/gsc?template=ifyqsvjlzj&ref=jmfkcdio) - easier setup.
    - To support the Trelis Research YouTube channel, you can first sign up for an account with [this link](https://runpod.io?ref=jmfkcdio).
- Vast.ai one-click template [here](https://cloud.vast.ai/?ref_id=98762&creator_id=98762&name=Fine-tuning%20Notebook%20by%20Trelis%20-%20Cuda%2012.1) - offers smaller GPUs (which are cheaper to run).
    - To support the Trelis Research YouTube channel, you can first sign up for an account with [this affiliate link](https://cloud.vast.ai/?ref_id=98762).
2. Google Colab (free and good for 7B models or smaller):
- Upload the .ipynb notebook
- Select a T4 GPU from Runtime -> Change Runtime Type.
- make sure to comment out flash attention when loading the model.
3. Your own computer (assuming you have an AMD or Nvidia GPU) - ADVANCED:
- Set up jupyter lab and a virtual environment using the instructions in the 'jupyter-lab-setup.md' file of this repo. [see here](https://github.com/TrelisResearch/install-guides/blob/main/jupyter-lab-setup.md).



## 2. GPU VRAM / Disk Volume 설정

1. gpu vram은 모델의 크기, batch size에 의해 크게 좌우됨
	- 모델 크기
		- 7b : 모델 크기만 약 14GB
			- [llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b)
		- 13b : 모델 크기만 약 27GB
			- [llava-v1.5-13b](https://huggingface.co/liuhaotian/llava-v1.5-7b)
	- batch size (실제 돌린 GPU 기준)
		- FFT 기준
			- 13b, bs=16 <-> 4* H100은 필요했다 (80 GB VRAM, 125 GB RAM, 12 vCPU)
			- 13b, bs=16 <-> 8* A100 SXM은 아주 원활했다 (80 GB VRAM, 117 GB RAM, 31 vCPU)
			- **<u>13b, bs=16 <-> 2* H100은 CUDA-OOM 에러가 발생했다...</u>**
		- LoRA 기준
			- 7b, bs=16 <-> 8* A5000에서 잘 돌아갔다 (24 GB VRAM, 29 GB RAM,  8 vCPU)
			- 13b, bs=16 <-> 8* A6000에서 잘 돌아갔다 (48 GB VRAM, 50 GB RAM,  8 vCPU)
	- **<u>cuda-oom 에러시 Batch Size를 알맞게 조정하기!!!</u>**

2. disk volume은 넉넉하게 잡아두는 게 좋다
	- 최소 100GB는 확보
	- **<u>학습시킨 모델도 저장할 공간이 필요하므로 생각보다 많이 필요함을 명심하자!</u>**
		- 7b 14GB + (fine-tuned LoRA adapter 7b) 2GB ~= 16GB
		- 7b  14GB + (full fine-tuned 7b) 14GB ~= 28GB



(FFT-학습 스크립트)

```shell
#!/bin/bash

deepspeed /workspace/LLaVA/llava/train/train_mem.py \
    --deepspeed /workspace/LLaVA/scripts/zero3.json \
    --model_name_or_path liuhaotian/llava-v1.5-13b \
    --version v1 \
    --data_path /workspace/LLaVA/dataset/total_line_list_fmt.json \
    --image_folder /workspace/LLaVA/dataset/artwork_images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir  /workspace/LLaVA/checkpoints/llava-v1.5-13b-artwork-tll-fft \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

```



(LoRA 학습 스크립트)

```shell
#!/bin/bash

# # Assign paths to variables
# DEEPSPEED_SCRIPT = "deepspeed llava/train/train_mem.py"
# DEEPSPEED_JSON = "./scripts/zero3.json"
# MODEL_NAME = "liuhaotian/llava-v1.5-7b"
# DATA_PATH = "/workspace/LLaVA/dataset/train/dataset.json"  # Replace with your JSON data path
# IMAGE_FOLDER = "/workspace/LLaVA/dataset/images"  # Replace with your image folder path
# VISION_TOWER = "openai/clip-vit-large-patch14-336"
# OUTPUT_DIR = "/workspace/LLaVA/checkpoints/llava-v1.5-7b-lora"  # Replace with your desired output directory path
# ## ADAPTER_OUTPUT_DIR = "/workspace/LLaVA/checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin"

deepspeed ./llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path liuhaotian/llava-v1.5-13b \
    --version v1 \
    --data_path /workspace/LLaVA/dataset/total_line_list_fmt.json \
    --image_folder /workspace/LLaVA/dataset/artwork_images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /workspace/LLaVA/checkpoints/llava-v1.5-13b-artwork-tll-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

```



## 3. 학습 관련 필요한 HuggingFace 함수들

- 관련 HuggingFace 게시물
	- https://huggingface.co/docs/transformers/v4.15.0/en/model_sharing#use-your-terminal-and-git
- 파일 하나 다운로드
	- [hf_hub_download](https://huggingface.co/docs/huggingface_hub/v0.20.3/en/package_reference/file_download#huggingface_hub.hf_hub_download)

- 전체 Repo 다운로드
	- [snapshot_download](https://huggingface.co/docs/huggingface_hub/v0.20.3/en/package_reference/file_download#huggingface_hub.snapshot_download)

