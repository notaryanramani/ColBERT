from data import _save_queries, _save_qrels

dataset = "msmarco-passage/train"
out_dir = "/ml_data/aryan/msmarco"

_save_queries(dataset, f"{out_dir}/queries.train.tsv")
_save_qrels(dataset, f"{out_dir}/qrels.train.tsv")
