# Some of the code in this file is adapted from:
#
# pfl-research:
# Copyright 2024, Apple Inc.
# Licensed under the Apache License, Version 2.0 (the "License").

import argparse
import json
import os
import sqlite3
import time
from collections import OrderedDict
from typing import Dict, Iterator, Optional
import torch
import h5py
from contextlib import contextmanager
import fcntl

from sentence_transformers import SentenceTransformer
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
# ---- Low-memory settings to avoid CPU thrash ----
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    torch.set_num_threads(1)
except Exception:
    pass

# ---- Disable TF on GPU to free CUDA memory for PyTorch ----
try:
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

import numpy as np
import torch
from torch.utils.data import DataLoader
from contextlib import nullcontext

CHUNK_SENTENCES = int(os.getenv("CHUNK_SENTENCES", "512"))  # 每次编码的句子数
FLUSH_USERS     = int(os.getenv("FLUSH_USERS", "20"))

@contextmanager
def file_lock(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)   # 阻塞等待独占锁
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def encode_fast(
    model,
    sentences,                 # list[str]
    batch_size=256,
    num_workers=0,             # 低内存机器用 0；富裕可 4/8
    use_fp16=True,
    max_seq_length=128,
    pin_memory=False,          # 低内存建议 False
    prefetch_factor=None       # 仅在 num_workers>0 时生效
):
    """
    Faster/steadier inference for SentenceTransformer-style models.
    - 单进程或少 worker 的 DataLoader，减少内存争抢
    - AMP(fp16) + no per-batch synchronize
    - 在 collate_fn 里做分词，避免先构建大 list 的 tokens
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval().to(device)

    # 缩短序列有助于增大 batch、稳定 GPU
    try:
        if max_seq_length is not None:
            model.max_seq_length = max_seq_length
    except Exception:
        pass

    # 取 tokenizer
    tok = getattr(model, "tokenizer", None)
    if tok is None:
        # SentenceTransformer 通常有 _first_module().tokenizer
        tok = model._first_module().tokenizer

    # 在 DataLoader 的 collate 中做分词（CPU 上执行）
    def collate_fn(batch_sent_list):
        return tok(batch_sent_list, padding=True, truncation=True, return_tensors='pt')

    # 组装 DataLoader（num_workers=0 时不要传 prefetch_factor/persistent_workers）
    dl_kwargs = dict(
        dataset=sentences,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0 and prefetch_factor is not None:
        dl_kwargs["prefetch_factor"] = int(prefetch_factor)

    loader = DataLoader(**dl_kwargs)

    amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if use_fp16 else nullcontext()

    chunks = []
    with torch.inference_mode(), amp_ctx:
        for enc in loader:
            # H2D：只有在 pin_memory=True 时 non_blocking 才有意义
            features = {k: v.to(device, non_blocking=pin_memory) for k, v in enc.items()}

            # SentenceTransformer 可能有 encode_features；没有就 forward
            reps = None
            if hasattr(model, "encode_features"):
                try:
                    reps = model.encode_features(features)
                except Exception:
                    reps = None
            if reps is None:
                reps = model.forward(features)

            # 常见返回：dict 含 'sentence_embedding'；否则可能直接是张量
            if isinstance(reps, dict) and "sentence_embedding" in reps:
                pooled = reps["sentence_embedding"]
            else:
                # 某些实现返回 tuple/list；取第一个
                if isinstance(reps, (list, tuple)):
                    pooled = reps[0]
                else:
                    pooled = reps  # 已是 Tensor

            chunks.append(pooled.detach().cpu())

            # 释放中间引用，避免峰值累积
            del pooled, reps, features, enc

    embeddings = torch.cat(chunks, dim=0).numpy() if len(chunks) else np.empty((0, 0))
    del chunks
    return embeddings

def get_rss_gb():
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])  # resident
        return pages * 4096 / (1024**3)       # 4096 = page size
    except Exception:
        return -1.0

def open_sqlite_conn(db_path: str) -> sqlite3.Connection:
    # 只读方式打开（更安全）；注意 uri=True
    uri = f"file:{os.path.abspath(db_path)}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    cur = conn.cursor()
    # 只读/批量查询提速设置 —— 都是对读安全的
    cur.execute("PRAGMA query_only=ON;")         # 禁止写
    cur.execute("PRAGMA journal_mode=OFF;")      # 只读可关日志
    cur.execute("PRAGMA synchronous=OFF;")       # 只读安全地关闭同步
    cur.execute("PRAGMA temp_store=MEMORY;")     # 临时表进内存
    cur.execute("PRAGMA mmap_size=268435456;")   # 256MB mmap，根据内存可上调
    cur.execute("PRAGMA cache_size=-262144;")    # 256MB page cache（-N，单位KB）
    conn.commit()
    return conn
def add_proto_parsing(ds: tf.data.Dataset) -> tf.data.Dataset:
    """
    输入：每个元素是一个标量 tf.string（TF Example 的二进制）
    输出：每个元素是 (tokens:str, tags:str)
    """
    # 大多数 SO 数据里，tokens/ tags 都是标量字符串
    feature_spec = {
        "tokens": tf.io.FixedLenFeature([], tf.string, default_value=b""),
        "tags":   tf.io.FixedLenFeature([], tf.string, default_value=b""),
    }

    def _parse(serialized_example: tf.Tensor):
        ex = tf.io.parse_single_example(serialized_example, feature_spec)
        # 转为字符串（保持为 tf.string，后续用 .numpy() 或 .decode 再转）
        return ex["tokens"], ex["tags"]

    return ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
# ===== End encode_fast =====

def load_word_counts(cache_dir=None,
                     vocab_size: Optional[int] = None) -> Dict[str, int]:
    """Loads the word counts for the Stack Overflow dataset.

    :param: cache_dir:
        (Optional) directory to cache the downloaded file. If `None`,
        caches in Keras' default cache directory.
    :param: vocab_size:
        (Optional) when specified, only load the first `vocab_size`
        unique words in the vocab file (i.e. the most frequent `vocab_size`
        words).

    :returns:
      A collections.OrderedDict where the keys are string tokens, and the values
      are the counts of unique users who have at least one example in the
      training set containing that token in the body text.
    """
    if vocab_size is not None:
        if not isinstance(vocab_size, int):
            raise TypeError(
                f'vocab_size should be None or int, got {type(vocab_size)}.')
        if vocab_size <= 0:
            raise ValueError(f'vocab_size must be positive, got {vocab_size}.')

    path = tf.keras.utils.get_file(
        'stackoverflow.word_count.tar.bz2',
        origin='https://storage.googleapis.com/tff-datasets-public/'
        'stackoverflow.word_count.tar.bz2',
        file_hash=(
            '1dc00256d6e527c54b9756d968118378ae14e6692c0b3b6cad470cdd3f0c519c'
        ),
        hash_algorithm='sha256',
        extract=True,
        archive_format='tar',
        cache_dir=cache_dir,
    )

    word_counts = OrderedDict()
    dir_path = os.path.dirname(path)
    file_path = os.path.join(dir_path, 'stackoverflow.word_count')
    with open(file_path) as f:
        for line in f:
            word, count = line.split()
            word_counts[word] = int(count)
            if vocab_size is not None and len(word_counts) >= vocab_size:
                break
    return word_counts


# def fetch_client_ids(database_filepath: str,
#                      split_name: Optional[str] = None) -> Iterator[str]:
#     """Fetches the list of client_ids.
#
#     :param database_filepath:
#         A path to a SQL database.
#     :param split_name:
#         An optional split name to filter on. If `None`, all client ids
#          are returned.
#     :returns:
#       An iterator of string client ids.
#     """
#     if split_name == "val":
#         # heldout is used in the raw sqlite database
#         split_name = "heldout"
#     #connection = sqlite3.connect(database_filepath)
#     with open_sqlite_conn(database_filepath) as conn:
#         cur = conn.cursor()
#         cur.execute("SELECT client_id FROM clients WHERE partition=?", (partition,))
#         return [r[0] for r in cur.fetchall()]
#     query = "SELECT DISTINCT client_id FROM client_metadata"
#     if split_name is not None:
#         query += f" WHERE split_name = '{split_name}'"
#     query += ";"
#     result = connection.execute(query)
#     return (x[0] for x in result)

#---------------------------------------------
# def fetch_client_ids(database_filepath: str,
#                      split_name: Optional[str] = None) -> Iterator[str]:
#     """Fetch client ids for a split (train/val/test). If split_name is None, return all."""
#     # sqlite 里 val 叫 heldout
#     if split_name == "val":
#         split_name = "heldout"
#
#     with open_sqlite_conn(database_filepath) as conn:
#         cur = conn.cursor()
#         if split_name is None:
#             cur.execute("SELECT DISTINCT client_id FROM clients;")
#         else:
#             cur.execute("SELECT client_id FROM clients WHERE partition = ?;", (split_name,))
#         rows = cur.fetchall()
#         return [r[0] for r in rows]
#-----------------------------------------------------------------
def fetch_client_ids(database_filepath: str, split_name: Optional[str] = None):
    """
    从 examples 表取 client_id。split_name 可为 train/val/test。
    实际库里分区列叫 split_name，val 可能对应 heldout/validation。
    """
    import os, sqlite3
    uri = f"file:{os.path.abspath(database_filepath)}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    cur = conn.cursor()

    # 先看看库里实际有哪些分区值
    rows = cur.execute("SELECT DISTINCT split_name FROM examples").fetchall()
    splits_present = {r[0] for r in rows}

    def resolve_split(want: Optional[str]) -> Optional[str]:
        if want is None:
            return None
        want = want.lower()
        aliases = {
            "train": ["train"],
            "val":   ["val", "valid", "validation", "heldout"],
            "test":  ["test"],
        }
        for cand in aliases.get(want, [want]):
            if cand in splits_present:
                return cand
        # 回退：用用户传入的原词（即便不在 present，也让 SQL 执行；最多返回空）
        return want

    split_db = resolve_split(split_name)

    if split_db is None:
        q = "SELECT DISTINCT client_id FROM examples"
        params = ()
    else:
        q = "SELECT DISTINCT client_id FROM examples WHERE split_name = ?"
        params = (split_db,)

    ids = [r[0] for r in cur.execute(q, params).fetchall()]
    conn.close()
    return ids


# def query_client_dataset(database_filepath: str,
#                          client_id: str,
#                          split_name: Optional[str] = None) -> tf.data.Dataset:
#
#     def add_proto_parsing(dataset: tf.data.Dataset) -> tf.data.Dataset:
#         """Add parsing of the tf.Example proto to the dataset pipeline."""
#
#         def parse_proto(tensor_proto):
#             parse_spec = OrderedDict(
#                 creation_date=tf.io.FixedLenFeature(dtype=tf.string, shape=()),
#                 score=tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
#                 tags=tf.io.FixedLenFeature(dtype=tf.string, shape=()),
#                 title=tf.io.FixedLenFeature(dtype=tf.string, shape=()),
#                 tokens=tf.io.FixedLenFeature(dtype=tf.string, shape=()),
#                 type=tf.io.FixedLenFeature(dtype=tf.string, shape=()),
#             )
#             parsed_features = tf.io.parse_example(tensor_proto, parse_spec)
#             return OrderedDict(
#                 (key, parsed_features[key]) for key in parse_spec)
#
#         return dataset.map(parse_proto, num_parallel_calls=tf.data.AUTOTUNE)
#
#     query_parts = [
#         "SELECT serialized_example_proto FROM examples WHERE client_id = '",
#         client_id,
#         "'",
#     ]
#     if split_name is not None:
#         if split_name == "val":
#             # heldout is used in the raw sqlite database
#             split_name = "heldout"
#         query_parts.extend([" and split_name ='", split_name, "'"])
#     uri = f"file:{os.path.abspath(database_filepath)}?mode=ro"
#     return add_proto_parsing(
#         tf.data.experimental.SqlDataset(
#             driver_name="sqlite",
#             data_source_name=uri,  # ← 改成只读 URI
#             query=tf.strings.join(query_parts),
#             output_types=(tf.string,),
#         ))
#--------------------------------------------------------------------
# def query_client_dataset(database_filepath, user_id, partition):
#     """
#     返回给定 user_id 在 partition(train/val/test) 下的样本（TF Example 序列化二进制 → 由 add_proto_parsing 解析）
#     """
#     import os, tensorflow as tf
#
#     # 把 'val' 映射到库里的实际值
#     split_map = {
#         "train": "train",
#         "val":   "heldout",     # 如果你跑 inspect 看到是 'validation'，这里改成 'validation'
#         "test":  "test",
#     }
#     split_db = split_map.get(partition, partition)
#
#     uri = f"file:{os.path.abspath(database_filepath)}?mode=ro"
#
#     # 直接取 serialized_example_proto 列，交给 add_proto_parsing 去解码 tokens/tags
#     ds = tf.data.experimental.SqlDataset(
#         driver_name="sqlite",
#         data_source_name=uri,
#         query=tf.strings.join([
#             "SELECT serialized_example_proto FROM examples WHERE client_id = '",
#             tf.constant(user_id),
#             "' AND split_name = '",
#             tf.constant(split_db),
#             "'"
#         ]),
#         output_types=(tf.string,)
#     )
#     return add_proto_parsing(ds)
#-------------------------------------------------------------------------
def query_client_dataset(database_filepath: str, user_id: str, partition: str,
                         max_retries: int = 10, retry_sleep: float = 0.2):
    """
    只读 sqlite3 读取该 user 的 serialized_example_proto，带 busy_timeout + 退避重试。
    使用 immutable=1 避免锁竞争；只设置“只读无副作用”的 PRAGMA。
    """
    import os, sqlite3, time, tensorflow as tf

    split_map = {"train": "train", "val": "heldout", "test": "test"}
    split_db = split_map.get(partition, partition)

    # 关键：mode=ro + immutable=1（数据库不会被其他进程修改时可用）
    uri = f"file:{os.path.abspath(database_filepath)}?mode=ro&immutable=1"

    def _fetch_blobs_once():
        # 只读连接
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False, isolation_level=None)
        cur = conn.cursor()
        try:
            # 只读/无副作用 PRAGMA（OK）：不会写盘
            cur.execute("PRAGMA query_only=ON;")         # 强制只读
            cur.execute("PRAGMA busy_timeout=8000;")     # 8s 等锁
            cur.execute("PRAGMA temp_store=MEMORY;")
            cur.execute("PRAGMA mmap_size=268435456;")   # 256MB
            # 可选：限制 page cache（负值=KB）：这里给 256MB
            cur.execute("PRAGMA cache_size=-262144;")

            cur.execute(
                "SELECT serialized_example_proto FROM examples WHERE client_id=? AND split_name=?",
                (user_id, split_db)
            )
            return [r[0] for r in cur.fetchall()]
        finally:
            conn.close()

    # 带退避重试处理“短暂 busy/locked”
    for t in range(max_retries):
        try:
            blobs = _fetch_blobs_once()
            break
        except sqlite3.OperationalError as e:
            msg = str(e).lower()
            if "locked" in msg or "busy" in msg:
                time.sleep(retry_sleep * (1.5 ** t))
                continue
            raise
    else:
        raise RuntimeError(f"sqlite busy too long for user {user_id} / split {split_db}")

    ds = tf.data.Dataset.from_tensor_slices(blobs)
    return add_proto_parsing(ds)



def process_user(user_id, model, database_filepath, partition, h5_path):
    # Use streaming append to avoid peak RAM
    with h5py.File(h5_path, 'a') as h5:
        return user_id, process_user_stream_and_append(
            user_id, model, database_filepath, partition, h5,
            encode_fn=encode_fast,
            encode_kwargs=dict(batch_size=256, num_workers=0, use_fp16=True, max_seq_length=128, pin_memory=False, prefetch_factor=None),
            chunk_size_sentences=CHUNK_SENTENCES
        )

    # with h5py.File(h5_path, 'a') as h5:
    #     if f'/{partition}/{user_id}' in h5:
    #         return user_id, 0
    # tf Dataset with sentences from user.

    tfdata = query_client_dataset(database_filepath, user_id, partition)

    sentences = []
    tags = []
    for sentence_data in tfdata:
        sentences.append(sentence_data['tokens'].numpy().decode('UTF-8'))
        tags.append(sentence_data['tags'].numpy().decode('UTF-8'))

    #embeddings = model.encode(sentences)
    embeddings = model.encode(sentences,
                              batch_size=64,  # 从 32 → 128
                              convert_to_numpy=True,
                              show_progress_bar=False)
    with h5py.File(h5_path, 'a') as h5:
        # Store encoded inputs.
        h5.create_dataset(f'/{partition}/{user_id}/embeddings', data=embeddings)
        h5.create_dataset(f'/{partition}/{user_id}/tags', data=tags)

    return user_id, len(embeddings)



def process_user_stream_and_append(
    user_id, model, database_filepath, partition, h5,
    encode_fn, encode_kwargs, chunk_size_sentences=CHUNK_SENTENCES
):
    import numpy as np
    grp = h5.require_group(f'/{partition}/{user_id}')
    has_emb = 'embeddings' in grp
    has_tags = 'tags' in grp

    cur_e = grp['embeddings'].shape[0] if has_emb else 0
    cur_t = grp['tags'].shape[0] if has_tags else 0
    already = max(cur_e, cur_t)

    if has_emb and has_tags and cur_e == cur_t and cur_e > 0:
        return cur_e

    # Prepare tags dataset
    if not has_tags:
        dt = h5py.string_dtype(encoding='utf-8')
        dset_tags = grp.create_dataset(
            'tags',
            shape=(0,), maxshape=(None,),
            dtype=dt,
            chunks=True,  # 仍然分块
            compression='lzf'  # ★ 轻量压缩
        )
    else:
        dset_tags = grp['tags']  # 已存在就直接复用（压缩与否保持原样）

    # Prepare embeddings dataset
    dset_emb = grp['embeddings'] if has_emb else None

    tfdata = query_client_dataset(database_filepath, user_id, partition)
    it = iter(tfdata)

    # Skip already processed samples
    skipped = 0
    try:
        while skipped < already:
            next(it)
            skipped += 1
    except StopIteration:
        return max(cur_e, cur_t)

    buf_sent, buf_tags = [], []
    total = already

    def _unpack_tokens_tags(el):
        # 兼容 (tokens, tags) 元组/列表，和 {'tokens':..., 'tags':...} 字典
        if isinstance(el, (tuple, list)) and len(el) == 2:
            tokens_t, tags_t = el
        elif isinstance(el, dict):
            tokens_t, tags_t = el.get('tokens'), el.get('tags')
        else:
            raise TypeError(f"Unsupported element type from dataset: {type(el)}")
        return tokens_t, tags_t

    for el in it:
        tokens_t, tags_t = _unpack_tokens_tags(el)
        # 标量 tf.string → Python str
        tokens = tokens_t.numpy().decode('utf-8', errors='ignore')
        tags = tags_t.numpy().decode('utf-8', errors='ignore')

        buf_sent.append(tokens)
        buf_tags.append(tags)


    # for sample_embedded_data in it:
    #     buf_sent.append(sample_embedded_data['tokens'].numpy().decode('UTF-8'))
    #     buf_tags.append(sample_embedded_data['tags'].numpy().decode('UTF-8'))

        if len(buf_sent) >= chunk_size_sentences:
            emb = encode_fn(model, buf_sent, **encode_kwargs)
            emb = emb.astype(np.float16)  # ★ 统一转 fp16
            if dset_emb is None:
                dim = emb.shape[1]
                dset_emb = grp.create_dataset(
                    'embeddings', shape=(0, dim), maxshape=(None, dim),
                    dtype=np.float16, chunks=True
                )

            old = dset_emb.shape[0]
            dset_emb.resize(old + emb.shape[0], axis=0)
            dset_emb[old: old + emb.shape[0]] = emb

            old_t = dset_tags.shape[0]
            dset_tags.resize(old_t + len(buf_tags), axis=0)
            dt = h5py.string_dtype(encoding='utf-8')
            dset_tags[old_t: old_t + len(buf_tags)] = np.array(buf_tags, dtype=dt)

            total += emb.shape[0]
            del emb
            buf_sent.clear(); buf_tags.clear()

    if buf_sent:
        emb = encode_fn(model, buf_sent, **encode_kwargs)
        emb = emb.astype(np.float16)  # ★ 关键：写入前转 fp16
        if dset_emb is None:
            dim = emb.shape[1]
            dset_emb = grp.create_dataset('embeddings', shape=(0, dim), maxshape=(None, dim), dtype=np.float16, chunks=True)
        old = dset_emb.shape[0]
        dset_emb.resize(old + emb.shape[0], axis=0)
        dset_emb[old: old + emb.shape[0]] = emb

        old_t = dset_tags.shape[0]
        dset_tags.resize(old_t + len(buf_tags), axis=0)
        dt = h5py.string_dtype(encoding='utf-8')
        dset_tags[old_t: old_t + len(buf_tags)] = np.array(buf_tags, dtype=dt)

        total += emb.shape[0]
        del emb
        buf_sent.clear(); buf_tags.clear()

    return total

def dl_preprocess_and_dump_h5(output_dir: str, job_id: int, num_jobs_per_split: int):
    """
    Preprocess StackOverflow dataset.

    :param output_dir:
        Directory for all output files, both raw and processed data.
    :param job_id:
        Which job is being run.
    """
    partition = ['train', 'val', 'test'][job_id // num_jobs_per_split]
    partition_job_id = job_id % num_jobs_per_split

    h5_path = os.path.join(output_dir, f'embedded_stackoverflow_{partition}_{partition_job_id}.hdf5')
    database_filepath = os.path.join(output_dir, "stackoverflow.sqlite")

    # 新增：状态文件相关配置（记录已处理的客户端）
    status_dir = os.path.join(output_dir, 'preprocess_status')  # 状态文件目录
    os.makedirs(status_dir, exist_ok=True)
    status_file = os.path.join(status_dir, f'{partition}_{partition_job_id}.txt')  # 状态文件路径

    print(f'Processing users for partition {partition}')
    client_ids = list(fetch_client_ids(database_filepath, partition))
    block_size = (len(client_ids) // num_jobs_per_split) + 1
    start_idx = partition_job_id * block_size
    end_idx = min(len(client_ids), (partition_job_id + 1) * block_size)
    current_job_clients = client_ids[start_idx: end_idx]  # 必须定义current_job_clients

    # 新增：读取已处理的客户端ID
    processed_clients = set()
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            processed_clients = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(processed_clients)} processed clients from {status_file}")

    # 新增：过滤出未处理的客户端
    remaining_clients = [cid for cid in current_job_clients if cid not in processed_clients]
    print(f"Need to process {len(remaining_clients)} clients (total {len(current_job_clients)} for this job)")

    #model = SentenceTransformer("all-MiniLM-L6-v2")
    model = SentenceTransformer("all-MiniLM-L6-v2",device='cuda',model_kwargs={"dtype": torch.float16})
    print(f"模型运行设备: {model.device}")

    client_ids = client_ids[start_idx: end_idx]
    start = time.time()
    user_num_datapoints = dict()

    # 侧写 TSV 路径
    tsv_path = os.path.join(output_dir, f"user_counts_{partition}_{partition_job_id}.tsv")
    total_remaining = len(remaining_clients)
    start = time.time()

    # ★ 外层只打开一次 HDF5；给它一个较小的 rdcc 缓存，避免常驻内存过大
    lock_path = h5_path + ".lock"
    with file_lock(lock_path):
        with h5py.File(h5_path, 'a',rdcc_nbytes=8 * 1024 ** 2, rdcc_nslots=1_000_003, rdcc_w0=0.25) as h5:
            for i, user_id in enumerate(remaining_clients, 1):
                # 直接把 h5 句柄传进去；函数内部不要再二次 open 文件
                n = process_user_stream_and_append(
                    user_id, model, database_filepath, partition, h5,
                    encode_fn=encode_fast,
                    encode_kwargs=dict(
                        batch_size=192, num_workers=0,
                        use_fp16=True, max_seq_length=96,
                        pin_memory=False, prefetch_factor=None
                    ),
                    chunk_size_sentences=CHUNK_SENTENCES
                )

                # 记录计数（写 TSV，避免把大 dict 留在内存）
                with open(tsv_path, "a") as fout:
                    fout.write(f"{user_id}\t{n}\n")

                # 写状态文件
                with open(status_file, 'a') as f:
                    f.write(f"{user_id}\n")

                # 进度 & 内存
                if i % 100 == 0:
                    print(f'Completed client {i} / {total_remaining} in {time.time() - start:.2f}s')

                # ★ 按批刷新，缓和 HDF5 内部缓存 + 页缓存；顺手回收 Python 临时对象
                if i % FLUSH_USERS == 0:
                    h5.flush()
                    import gc;
                    gc.collect()

                if i % 200 == 0:
                    rss = get_rss_gb()
                    dt = time.time() - start
                    if rss >= 0:
                        print(f"[perf] {i}/{total_remaining} users | {i / dt:.2f} users/s | RSS={rss:.2f} GB")
                    else:
                        print(f"[perf] {i}/{total_remaining} users | {i / dt:.2f} users/s")

            h5.flush()  # 兜底

    #if len(user_num_datapoints):
        # with h5py.File(h5_path, 'a') as h5:
        #     h5[f'/metadata/user_num_datapoints/{partition}'] = json.dumps(
        #         user_num_datapoints)
    tsv = os.path.join(output_dir, f"user_counts_{partition}_{partition_job_id}.tsv")



    # 3) 循环结束后（仅一次）：从 TSV 汇总并写入 HDF5 元数据
    from collections import OrderedDict
    user_num_datapoints = OrderedDict()
    if os.path.exists(tsv):
        with open(tsv, "r") as fin:
            for line in fin:
                cid, num = line.rstrip("\n").split("\t")
                user_num_datapoints[cid] = int(num)

    with h5py.File(h5_path, 'a') as h5:
        meta_grp = h5.require_group('/metadata/user_num_datapoints')
        if partition in meta_grp:
            del meta_grp[partition]
        meta_grp.create_dataset(partition, data=json.dumps(user_num_datapoints))

if __name__ == '__main__':

    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        '--output_dir',
        help=('Output directory for the original sqlite '
              'data and the processed hdf5 files.'),
        default='embedded_data')


    argument_parser.add_argument(
        '--job_id',
        type=int,
        default=0,
        help='Integer in range [0, 3 * num_jobs_per_split], specifiying which portion of the data to process.')

    argument_parser.add_argument(
        '--num_jobs_per_split',
        type=int,
        default=1,
        help='How many jobs per split in [train, test, val]')

    arguments = argument_parser.parse_args()

    dl_preprocess_and_dump_h5(arguments.output_dir,
                              arguments.job_id,
                              arguments.num_jobs_per_split)