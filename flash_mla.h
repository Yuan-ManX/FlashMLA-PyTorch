#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @struct Flash_fwd_mla_params
 * @brief 用于多头注意力（Multi-Head Attention）前向传播的参数结构体。
 *
 * 该结构体包含执行多头注意力前向传播所需的所有参数，包括输入和输出张量的指针、步幅信息、缩放因子等。
 */
struct Flash_fwd_mla_params {
    // 定义索引类型为64位整数
    using index_t = int64_t;

    // 以下是多头注意力前向传播所需的参数
    int b;               // 批处理大小（batch size）
    int seqlen_q;        // 查询序列长度（query sequence length）
    int d;               // 特征维度（feature dimension）
    int d_v;             // 值（value）向量的维度（value dimension）
    int h;               // 注意力头的数量（number of attention heads）
    int h_h_k_ratio;     // 注意力头与键（key）向量的维度比率（ratio of heads to key dimension）
    int ngroups;         // 分组数量（number of groups）
    bool is_causal;      // 是否是因果注意力（causal attention），即是否只考虑左侧上下文
    float scale_softmax; // Softmax的缩放因子（scale factor for softmax）
    float scale_softmax_log2; // Softmax缩放因子的对数2（log2 of scale factor for softmax）

    int *__restrict__ cu_seqlens_k;  // CUDA序列长度指针，指向键（key）的序列长度数组

    // 输入和输出张量的指针
    void *__restrict__ q_ptr;    // 查询（query）张量指针
    void *__restrict__ k_ptr;    // 键（key）张量指针
    void *__restrict__ v_ptr;    // 值（value）张量指针
    void *__restrict__ o_ptr;    // 输出（output）张量指针
    void *__restrict__ softmax_lse_ptr; // Softmax的对数求和（log-sum-exp）指针

    // 张量步幅信息
    index_t q_batch_stride;  // 查询张量的批处理步幅（batch stride for query tensor）
    index_t k_batch_stride;  // 键张量的批处理步幅（batch stride for key tensor）
    index_t v_batch_stride;  // 值张量的批处理步幅（batch stride for value tensor）
    index_t o_batch_stride;  // 输出张量的批处理步幅（batch stride for output tensor）
    index_t q_row_stride;    // 查询张量的行步幅（row stride for query tensor）
    index_t k_row_stride;    // 键张量的行步幅（row stride for key tensor）
    index_t v_row_stride;    // 值张量的行步幅（row stride for value tensor）
    index_t o_row_stride;    // 输出张量的行步幅（row stride for output tensor）
    index_t q_head_stride;   // 查询张量的头步幅（head stride for query tensor）
    index_t k_head_stride;   // 键张量的头步幅（head stride for key tensor）
    index_t v_head_stride;   // 值张量的头步幅（head stride for value tensor）
    index_t o_head_stride;   // 输出张量的头步幅（head stride for output tensor）

    // 分块表指针，指向分块信息数组
    int *__restrict__ block_table;
    // 分块表的批处理步幅（batch stride for block table）
    index_t block_table_batch_stride;
    // 分块大小（page block size）
    int page_block_size;

    // Tile调度元数据指针，指向Tile调度元数据数组
    int *__restrict__ tile_scheduler_metadata_ptr;
    // 流多处理器（SM）部分的数量（number of SM parts）
    int num_sm_parts;
    // 分裂数量指针，指向分裂数量数组
    int *__restrict__ num_splits_ptr;

    // Softmax对数求和累加指针
    void *__restrict__ softmax_lseaccum_ptr;
    // 输出累加指针
    void *__restrict__ oaccum_ptr;
};


/**
 * @brief Tile调度元数据的大小常量。
 *
 * 该常量表示Tile调度元数据数组的大小，每个Tile调度元数据包含8个元素：
 * [begin_idx, begin_seqlen, end_idx, end_seqlen, begin_n_split_idx, _, _, _]
 */
static constexpr int TileSchedulerMetaDataSize = 8;
// [begin_idx, begin_seqlen, end_idx, end_seqlen, begin_n_split_idx, _, _, _]


////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief 运行多头注意力前向传播的函数模板。
 *
 * 该函数使用FlashAttention算法对多头注意力进行前向传播，支持键值对（KV）分块。
 *
 * @tparam T 数据类型，如float、half等。
 * @tparam Headdim 注意力头的维度（head dimension）。
 *
 * @param params Flash_fwd_mla_params结构体，包含前向传播所需的参数。
 * @param stream CUDA流，用于异步执行。
 */
template<typename T, int Headdim>
void run_mha_fwd_splitkv_mla(Flash_fwd_mla_params &params, cudaStream_t stream);


/**
 * @struct Mla_metadata_params
 * @brief 用于获取MLA（多头注意力）元数据的参数结构体。
 *
 * 该结构体包含获取MLA元数据所需的所有参数，包括序列长度、Tile调度元数据、分裂数量等。
 */
struct Mla_metadata_params {
    int *__restrict__ seqlens_k_ptr;      // 指向键（key）序列长度数组的指针
    int *__restrict__ tile_scheduler_metadata_ptr; // 指向Tile调度元数据数组的指针
    int *__restrict__ num_splits_ptr;    // 指向分裂数量数组的指针
    int batch_size;                       // 批处理大小（batch size）
    int block_size_n;                     // 块大小（block size）
    int fixed_overhead_num_blocks;        // 固定开销的块数量（number of blocks for fixed overhead）
    int num_sm_parts;                     // 流多处理器（SM）部分的数量（number of SM parts）
};


/**
 * @brief 获取MLA元数据的函数。
 *
 * 该函数计算并填充MLA元数据参数中的各个字段。
 *
 * @param params Mla_metadata_params结构体，包含获取元数据所需的所有参数。
 * @param stream CUDA流，用于异步执行。
 */
void get_mla_metadata_func(Mla_metadata_params &params, cudaStream_t stream);
