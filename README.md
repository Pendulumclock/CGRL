# 🚀 UAV Navigation with Curriculum Learning based on VLM Attention

This project explores a curriculum learning strategy for UAV navigation tasks by leveraging the attention scores from the final layer of a Vision-Language Model (VLM) decoder. By analyzing the attention distribution between textual queries and image regions, we estimate the difficulty of each sample based on whether the model is focusing correctly on the target. 

To further enhance training efficiency, we introduce a Gaussian-based difficulty-aware sampling strategy, which prioritizes samples of varying difficulty in a probabilistic manner. This approach significantly improves the performance of the navigation model on challenging scenarios.

## 🔍 Highlights
- 🔎 **Attention-based Difficulty Estimation**: Measures how well VLM attends to the target object.
- 📊 **Gaussian Sampling Strategy**: Selects training samples based on estimated difficulty for curriculum learning.
- 🚁 **Performance Boost**: Demonstrated improved accuracy and robustness in UAV navigation tasks.


## 🛠️ Environment Setup

This project depends on multiple models and tool libraries. It is recommended to use Conda to create an isolated environment.

### Install Conda Environment

```bash
- conda create -n flightgpt python=3.10
- conda activate flightgpt

- pip install -r requirements.txt
```

---


## 🛠️ Model and Data Preparation

* Download model weights to `./model_weight/`  
  Note: Change the value of `max_pixels` in `preprocessor_config.json` to `16032016`.

* Download data to `./data/`

* And for sft, Download the cleaned_final.json to ./LLaMA-Factory/data

### 📦 Project Structure
├── model_weight/ # Directory for model weights (download manually)  
├── experiment/  
├── R1PhotoData/  
├── data/  
    └── citynav/ # Data annotation directory  
    └── rgbd-new/ # Raw image files  
    └── training_data/ # Training data directory  
    └── ...  
├── data_examples/ # Examples of some training data  
├── eval.py # Model inference and evaluation script  
├── open-r1-multimodal/ # GRPO training directory  
├── LLaMA-Factory/ # SFT training directory  
├── requirements.txt # Combined environment dependency file  
├── README.md # This document  
├── ...  

---


## 🚀 Inference

1. Start the vLLM service
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve path/to/your/model \
  --dtype auto \
  --trust-remote-code \
  --served-model-name qwen_2_5_vl_7b \
  --host 0.0.0.0 \
  -tp 4 \
  --uvicorn-log-level debug \
  --port your_port \
  --limit-mm-per-prompt image=2,video=0 \
  --max-model-len=32000
```

2. Start the inference script

```bash
python eval_by_qwen.py
```

3. Result Visualization  
You can use the visualize_prediction function to visualize the predicted target coordinates and the landmark bounding boxes, as well as the actual target coordinates and landmark bounding boxes.

---


## 🚀 Training
```bash
sh ./open-r1-multimodal/run_scripts/run_grpo_rec_lora.sh
```

---

