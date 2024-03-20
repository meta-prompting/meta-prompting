# Meta Prompting for Solving MATH and GSM8K problems

## Setup

We recommend using [Conda](https://docs.conda.io/projects/miniconda) to manage your environment. We use [vLLM](https://github.com/vllm-project/vllm) to accelerate inference. Run the following commands to setup your environment:

```sh
cd src
conda create -n mp python=3.9
conda activate mp
pip3 install torch==2.1.2 torchvision torchaudio
pip install -r requirements.txt
```

### 🪁 Inference

We provide a script for inference, simply config the `MODEL_NAME_OR_PATH` and `DATA` in [./scripts/infer.sh](./scripts/infer.sh) and run the following command:

```sh
bash ./scripts/infer.sh
```
