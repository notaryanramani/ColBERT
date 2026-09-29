import os

# Paths
DATA_DIR   = "data/msmarco"
INDEX_DIR  = "indexes"
CKPT_PATH  = "checkpoints/colbert_msmarco.pt"

# Encoder
BERT_NAME     = "bert-base-uncased"
EMB_DIM       = 128          
DOC_MAXLEN    = 128        # max WordPiece tokens per doc
QUERY_MAXLEN  = 32           
QUERY_TOKEN   = "[Q]"
DOC_TOKEN     = "[D]"

# Indexing throughput 
BUCKET_SIZE     = 100_000
BATCH_SIZE      = 128
NUM_CPU_WORKERS = 2

# Storage
STORE_DTYPE = "float16"      

# FAISS IVFPQ 
FAISS_NLIST  = 2000          
FAISS_M      = 16            
FAISS_NBITS  = 8
FAISS_NPROBE = 10            

PUNCT_TOKENS = [".", ",", "!", "?", ";", ":", "'", '"', "(", ")", "-"]
