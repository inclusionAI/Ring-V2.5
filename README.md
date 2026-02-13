# Ring-V2.5
<p align="center"><img src="./figures/ant-bailing.png" width="100"/></p>

<p align="center">🤗 <a href="https://huggingface.co/inclusionAI">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/inclusionAI">ModelScope</a></p>

## Introduction
Introducing Ring-2.5-1T: the world's **first open-source trillion-parameter thinking model based on hybrid linear attention** architecture.

In a major step toward general-purpose AI agents, we're scaling hybrid linear attention across pre-training and RL. Our efficient 1:7 MLA + Lightning Linear Attention boosts reasoning speed and exploration, while expanded RL training enhances deep thinking and long-horizon task execution.

## Performance
Ring-2.5-1T model achieves gold-medal🏅 level performance at both IMO 2025 and CMO 2025. For detailed solutions of our model, please see [examples](https://github.com/inclusionAI/Ring-V2.5/blob/main/examples). 

## Model Downloads

|      **Model**      |    **Context Length**    |                                                                   **Download**                                                                    |
|:-------------------:|:------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------:|
|     Ring-2.5-1T     |    128K -> 256K (YaRN)   | [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ring-2.5-1T) <br>[🤖 ModelScope](https://www.modelscope.cn/models/inclusionAI/Ring-2.5-1T) |

Note: If you are interested in previous version, please visit the past model collections in [Huggingface](https://huggingface.co/inclusionAI) or [ModelScope](https://modelscope.cn/organization/inclusionAI).

## Deployment

### SGLang

#### Environment Preparation

We will later submit our model to SGLang official release, now we can prepare the environment following steps:
```shell
git clone -b ling_2_5 git@github.com:antgroup/sglang.git
cd sglang

# Install the python packages
pip install --upgrade pip
pip install -e "python"
```

#### Run Inference

Both BF16 and FP8 models are supported by SGLang now. It depends on the dtype of the model in ${MODEL_PATH}.
Here is the example to run Ring-1T with multiple GPU nodes, where the master node IP is ${MASTER_IP} and server port is ${PORT}:

- Start server:
```bash
# Node 0:
python -m sglang.launch_server --model-path $MODEL_PATH --tp-size 8 --pp-size 4 --dp-size 1 --trust-remote-code --dist-init-addr $MASTER_IP:2345 --port $PORT --nnodes 4 --node-rank 0 
# Node 1:
python -m sglang.launch_server --model-path $MODEL_PATH --tp-size 8 --pp-size 4 --dp-size 1 --trust-remote-code --dist-init-addr $MASTER_IP:2345 --port $PORT --nnodes 4 --node-rank 1 
# Node 2:
python -m sglang.launch_server --model-path $MODEL_PATH --tp-size 8 --pp-size 4 --dp-size 1 --trust-remote-code --dist-init-addr $MASTER_IP:2345 --port $PORT --nnodes 4 --node-rank 2 
# Node 3:
python -m sglang.launch_server --model-path $MODEL_PATH --tp-size 8 --pp-size 4 --dp-size 1 --trust-remote-code --dist-init-addr $MASTER_IP:2345 --port $PORT --nnodes 4 --node-rank 3

# This is only an example. Please adjust arguments according to your actual environment.
```

- Client:

```shell
curl -s http://${MASTER_IP}:${PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```

More usage can be found [here](https://docs.sglang.ai/basic_usage/send_request.html)

## License

This code repository is licensed under [the MIT License](https://github.com/inclusionAI/Ring-V2.5/blob/main/LICENSE).

## Citation

If you find our work helpful, feel free to give us a cite.

```

```
