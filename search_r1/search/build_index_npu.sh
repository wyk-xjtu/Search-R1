export SEARCH_R1_DEVICE=${SEARCH_R1_DEVICE:-npu}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}

corpus_file=${CORPUS_FILE:-/your/corpus/jsonl/file}
save_dir=${INDEX_SAVE_DIR:-/the/path/to/save/index}
retriever_name=${RETRIEVER_NAME:-e5}
retriever_model=${RETRIEVER_MODEL:-intfloat/e5-base-v2}

# FAISS GPU is CUDA-only in this code path; NPU uses the encoder on Ascend and builds FAISS on CPU.
python search_r1/search/index_builder.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 256 \
    --batch_size 512 \
    --pooling_method mean \
    --faiss_type HNSW64 \
    --save_embedding
