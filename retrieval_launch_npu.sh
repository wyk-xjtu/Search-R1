export SEARCH_R1_DEVICE=${SEARCH_R1_DEVICE:-npu}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}

file_path=${RETRIEVAL_DATA_DIR:-/the/path/you/save/corpus}
index_file=$file_path/e5_HNSW64.index
corpus_file=$file_path/wiki-18.jsonl
retriever_name=e5
retriever_path=${RETRIEVER_MODEL:-intfloat/e5-base-v2}

python search_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path
