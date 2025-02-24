from typing import Optional, Tuple
import torch

import flash_mla_cuda


def get_mla_metadata(
    cache_seqlens: torch.Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    获取MLA（多头注意力）元数据的函数。

    参数:
        cache_seqlens (torch.Tensor): 
            - 形状: (batch_size)
            - 数据类型: torch.int32
            - 说明: 每个样本在缓存中的序列长度。

        num_heads_per_head_k (int): 
            - 说明: 每个K头（Key头）对应的Q头（Query头）数量。
            - 计算方式: seq_len_q * num_heads_q // num_heads_k
            - 其中:
                - seq_len_q: 查询序列的长度
                - num_heads_q: Q头的数量
                - num_heads_k: K头的数量

        num_heads_k (int): 
            - 说明: K头的总数量。

    返回:
        tile_scheduler_metadata (torch.Tensor): 
            - 形状: (num_sm_parts, TileSchedulerMetaDataSize)
            - 数据类型: torch.int32
            - 说明: 用于Tile调度器的元数据，包含了MLA操作所需的调度信息。

        num_splits (torch.Tensor): 
            - 形状: (batch_size + 1)
            - 数据类型: torch.int32
            - 说明: 每个样本的分割数量，用于后续的并行处理。

    调用:
        使用flash_mla_cuda库中的get_mla_metadata函数来计算MLA元数据。
    """
    return flash_mla_cuda.get_mla_metadata(cache_seqlens, num_heads_per_head_k, num_heads_k)


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用键值缓存（KVCache）执行Flash MLA（多头注意力）操作的函数。

    参数:
        q (torch.Tensor): 
            - 形状: (batch_size, seq_len_q, num_heads_q, head_dim)
            - 说明: 查询张量，其中:
                - batch_size: 批次大小
                - seq_len_q: 查询序列的长度
                - num_heads_q: Q头的数量
                - head_dim: 每个头的维度大小

        k_cache (torch.Tensor): 
            - 形状: (num_blocks, page_block_size, num_heads_k, head_dim)
            - 说明: 键缓存张量，其中:
                - num_blocks: 块的数量
                - page_block_size: 每个块的大小
                - num_heads_k: K头的数量
                - head_dim: 每个头的维度大小

        block_table (torch.Tensor): 
            - 形状: (batch_size, max_num_blocks_per_seq)
            - 数据类型: torch.int32
            - 说明: 块表，用于指示每个序列中包含的块。

        cache_seqlens (torch.Tensor): 
            - 形状: (batch_size)
            - 数据类型: torch.int32
            - 说明: 每个样本在缓存中的序列长度。

        head_dim_v (int): 
            - 说明: 值（Value）头的维度大小。

        tile_scheduler_metadata (torch.Tensor): 
            - 形状: (num_sm_parts, TileSchedulerMetaDataSize)
            - 数据类型: torch.int32
            - 说明: 由get_mla_metadata函数返回的Tile调度器元数据。

        num_splits (torch.Tensor): 
            - 形状: (batch_size + 1)
            - 数据类型: torch.int32
            - 说明: 由get_mla_metadata函数返回的分割数量。

        softmax_scale (Optional[float]): 
            - 默认值: None
            - 说明: 在应用softmax之前对QK^T的缩放比例。如果未提供，则默认为1 / sqrt(head_dim)。

        causal (bool): 
            - 默认值: False
            - 说明: 是否应用因果注意力掩码。

    返回:
        out (torch.Tensor): 
            - 形状: (batch_size, seq_len_q, num_heads_q, head_dim_v)
            - 说明: 注意力操作的输出张量。

        softmax_lse (torch.Tensor): 
            - 形状: (batch_size, num_heads_q, seq_len_q)
            - 数据类型: torch.float32
            - 说明: Softmax的对数和，用于后续的梯度计算。

    实现细节:
        - 如果未提供softmax_scale，则根据Q的最后一个维度计算缩放比例。
        - 调用flash_mla_cuda库中的fwd_kvcache_mla函数执行前向MLA操作。
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    out, softmax_lse = flash_mla_cuda.fwd_kvcache_mla(
        q,
        k_cache,
        None,
        head_dim_v,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal,
        tile_scheduler_metadata,
        num_splits,
    )
    return out, softmax_lse
