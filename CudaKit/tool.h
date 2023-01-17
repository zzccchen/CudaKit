#ifndef TOOL_
#define TOOL_

/**
 * @brief 打印显卡相关信息
 *
 * @see 参考: https://blog.51cto.com/u_15906550/5921417
 */
void C_GetCudaInfo();

/**
 * @brief 内存显存拷贝带宽测试
 *
 * @brief 对比pinned memory页锁定内存和pageable memory可分页内存
 *
 * @see 参考: https://github.com/NVIDIA-developer-blog/code-samples/blob/
 *           master/series/cuda-cpp/optimize-data-transfers/bandwidthtest.cu
 */
void C_ProfileCopies();
#endif
