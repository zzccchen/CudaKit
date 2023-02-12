#include <array>
#include <ctime>
#include <iostream>

#include "cuda_runtime.h"              // define __global__ __device__
#include "device_launch_parameters.h"  // define threadIdx
#include "tool.h"

#define TensorType float
#define LX         1024
#define LY         100
#define LZ         100

#define C_Ptr19D_A std::array<char*, 19>
#define H_Ptr19D_A std::array<TensorType*, 19>
#define D_Ptr19D_A std::array<cudaPitchedPtr, 19>

struct T_19D_S {
  TensorType Q0;
  TensorType Q1;
  TensorType Q2;
  TensorType Q3;
  TensorType Q4;
  TensorType Q5;
  TensorType Q6;
  TensorType Q7;
  TensorType Q8;
  TensorType Q9;
  TensorType Q10;
  TensorType Q11;
  TensorType Q12;
  TensorType Q13;
  TensorType Q14;
  TensorType Q15;
  TensorType Q16;
  TensorType Q17;
  TensorType Q18;
  TensorType _;
};

/**
 * @brief CUDA Grid尺寸计算
 *
 * @param tensor_dim - 三维矩阵长宽高
 * @param block_size - block长宽高
 * @param last_function_name - 父级函数名, 使用 __FUNCTION__ 宏
 * @param last_function_line - 在父级函数中的行号, 使用 __LINE__ 宏
 *
 * @return dim3
 */
dim3 C_GridSizeSet(dim3 tensor_dim,
                   dim3 block_size,
                   const char* last_function_name,
                   const int last_function_line);

/**
 * @brief CUDA异常处理函数
 *
 * @brief 打印异常信息, 不会中断程序运行, CUDA 常见错误类型:
 * @see 参考: https://blog.csdn.net/Bit_Coders/article/details/113181262
 *
 * @param error - CUDA异常
 * @param local_function_name - 当前函数名
 * @param last_function_name - 父级函数名, 使用 __FUNCTION__ 宏
 * @param last_function_line - 在父级函数中的行号, 使用 __LINE__ 宏
 */
void C_CudaErrorHandle(const cudaError_t error,
                       const char* local_function_name,
                       const char* last_function_name,
                       const int last_function_line);

/**
 * @brief make_cudaExtent封装
 *
 * @brief 隐藏make_cudaExtent()细节, 以便统一修改.
 *
 * @param tensor_dim - 三维矩阵长宽高
 *
 * @return cudaExtent
 */
cudaExtent C_MakeCudaExtent(const dim3 tensor_dim);
cudaExtent C_MakeCudaExtent19(const dim3 tensor_dim);

/**
 * @brief make_cudaPitchedPtr封装
 *
 * @brief 隐藏make_cudaPitchedPtr()细节, 以便统一修改.
 *
 * @param host_tensor_ptr - host三维矩阵指针, 使用指针引用
 * @param tensor_dim - 三维矩阵长宽高
 *
 * @return cudaPitchedPtr
 */
cudaPitchedPtr C_MakeCudaPitchedPtr(TensorType*& host_tensor_ptr,
                                    const dim3 tensor_dim);
cudaPitchedPtr C_MakeCudaPitchedPtr19(T_19D_S*& host_tensor_ptr,
                                      const dim3 tensor_dim);

/**
 * @brief CUDA三维矩阵拷贝
 *
 * @brief 支持Host->Deivce和Deivce->Host,
 * 隐藏make_cudaPitchedPtr()和make_cudaExtent()细节.
 * @see 参考: https://www.cnblogs.com/cuancuancuanhao/p/7805892.html
 *
 * @param host_tensor_ptr - host三维矩阵指针, 使用指针引用
 * @param device_tensor_ptr - device三维矩阵指针, 带pitch间隔参数, 使用引用
 * @param cpy_kind - 拷贝类型: cudaMemcpyHostToDevice || cudaMemcpyDeviceToHost
 * @param tensor_dim - 三维矩阵长宽高
 * @param last_function_name - 父级函数名, 使用 __FUNCTION__ 宏
 * @param last_function_line - 在父级函数中的行号, 使用 __LINE__ 宏
 */
void C_CudaMemCpy3D(TensorType*& host_tensor_ptr,
                    cudaPitchedPtr& device_tensor_ptr,
                    const enum cudaMemcpyKind cpy_kind,
                    const dim3 tensor_dim,
                    const char* last_function_name,
                    const int last_function_line);
void C_CudaMemCpy3D19(T_19D_S*& host_tensor_ptr,
                      cudaPitchedPtr& device_tensor_ptr,
                      const enum cudaMemcpyKind cpy_kind,
                      const dim3 tensor_dim,
                      const char* last_function_name,
                      const int last_function_line);

/**
 * @brief Host三维矩阵空间申请
 *
 * @brief 使用cudaMallocHost()申请页锁定内存.
 * @see 参考: https://www.cnblogs.com/1024incn/p/4564726.html
 * @see 参考: https://segmentfault.com/a/1190000007693221
 * @see 参考: https://github.com/NVIDIA-developer-blog/code-samples/blob
 *           /master/series/cuda-cpp/optimize-data-transfers/bandwidthtest.cu
 *
 * @param host_tensor_ptr - host三维矩阵指针, 使用指针引用
 * @param tensor_dim - 三维矩阵长宽高
 * @param last_function_name - 父级函数名, 使用 __FUNCTION__ 宏
 * @param last_function_line - 在父级函数中的行号, 使用 __LINE__ 宏
 */
void C_CudaMalloc3DHost(TensorType*& host_tensor_ptr,
                        const dim3 tensor_dim,
                        const char* last_function_name,
                        const int last_function_line);
void C_CudaMalloc3DHost19(T_19D_S*& host_tensor_ptr,
                          const dim3 tensor_dim,
                          const char* last_function_name,
                          const int last_function_line);

/**
 * @brief Host三维矩阵空间释放
 *
 * @param host_tensor_ptr - Host三维矩阵指针, 使用指针引用
 * @param last_function_name - 父级函数名, 使用 __FUNCTION__ 宏
 * @param last_function_line - 在父级函数中的行号, 使用 __LINE__ 宏
 */
void C_CudaFree3DHost(TensorType*& host_tensor_ptr,
                      const char* last_function_name,
                      const int last_function_line);
void C_CudaFree3DHost19(T_19D_S*& host_tensor_ptr,
                        const char* last_function_name,
                        const int last_function_line);

/**
 * @brief Device三维矩阵空间申请
 *
 * @brief 使用cudaMalloc3D()申请内存对齐显存.
 * @see 参考: https://www.cnblogs.com/cuancuancuanhao/p/7805892.html
 *
 * @param device_tensor_ptr - device三维矩阵指针, 使用引用
 * @param tensor_dim - 三维矩阵长宽高
 * @param last_function_name - 父级函数名, 使用 __FUNCTION__ 宏
 * @param last_function_line - 在父级函数中的行号, 使用 __LINE__ 宏
 */
void C_CudaMalloc3DDevice(cudaPitchedPtr& device_tensor_ptr,
                          const dim3 tensor_dim,
                          const char* last_function_name,
                          const int last_function_line);
void C_CudaMalloc3DDevice19(cudaPitchedPtr& device_tensor_ptr,
                            const dim3 tensor_dim,
                            const char* last_function_name,
                            const int last_function_line);

/**
 * @brief 定制-Device三维矩阵空间释放
 *
 * @param device_tensor_ptr - device三维矩阵指针, 使用引用
 * @param last_function_name - 父级函数名, 使用 __FUNCTION__ 宏
 * @param last_function_line - 在父级函数中的行号, 使用 __LINE__ 宏
 */
void C_CudaFree3DDevice(cudaPitchedPtr& device_tensor_ptr,
                        const char* last_function_name,
                        const int last_function_line);