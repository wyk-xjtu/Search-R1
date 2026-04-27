## Ascend NPU / CUDA dual backend

This fork can select CUDA, Ascend NPU, or CPU at runtime through `SEARCH_R1_DEVICE`.
CUDA remains the default when CUDA is available. For Ascend, set:

```bash
export SEARCH_R1_DEVICE=npu
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SEARCH_R1_ATTENTION_IMPL=sdpa
export SEARCH_R1_RAY_RESOURCE=NPU
```

Install a PyTorch + torch-npu pair that matches your CANN version before installing this repo. Keep CUDA dependencies such as `flash-attn` and `faiss-gpu` out of the Ascend environment unless your stack explicitly supports them.

### Supported paths

- PPO/GRPO FSDP training path with `actor_rollout_ref.rollout.name=hf`.
- Inference with `infer.py`.
- Dense retriever encoder on NPU.
- FAISS indexing/search on CPU when running on NPU.
- CUDA training and retrieval still work with the original CUDA scripts.

### Current limitations

- The vendored vLLM integration in this repo is CUDA-first. For Ascend, use the provided NPU scripts, which switch rollout to HuggingFace generation. If you want vLLM on Ascend, use a vLLM-Ascend stack and expect additional integration work around the sharding manager.
- `use_remove_padding=True` depends on flash-attn padding utilities, so NPU scripts set it to `False`.
- FAISS GPU is CUDA-only here. For Ascend retrieval, prefer ANN indexes such as `HNSW64` to keep CPU search fast.
- Megatron worker paths still contain CUDA-specific code and are not part of this Ascend pass.

### Retrieval

Build an ANN index with the NPU encoder and CPU FAISS:

```bash
export CORPUS_FILE=/path/to/wiki-18.jsonl
export INDEX_SAVE_DIR=/path/to/index
bash search_r1/search/build_index_npu.sh
```

Launch retrieval:

```bash
export RETRIEVAL_DATA_DIR=/path/to/index_and_corpus
bash retrieval_launch_npu.sh
```

The server still exposes:

```text
http://127.0.0.1:8000/retrieve
```

### Training

Process data as usual:

```bash
python scripts/data_process/nq_search.py --local_dir ./data/nq_search
```

Then run:

```bash
bash train_ppo_npu.sh
```

or:

```bash
bash train_grpo_npu.sh
```

For multi-node Ray, start Ray with the custom NPU resource on every node, for example:

```bash
ray start --head --resources='{"NPU": 8}'
```

Worker nodes should join the same Ray cluster with the same `NPU` resource declaration.

