#include <ctime>
#include <iostream>

#include "cuda_runtime.h"              // define __global__ __device__
#include "device_launch_parameters.h"  // define threadIdx
#include "tool.h"

#define TensorType double
#define LX         1024
#define LY         400
#define LZ         100

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
                   const int last_function_line) {
  dim3 grid_size = {0, 0, 0};
  grid_size.x = ((tensor_dim.x % block_size.x) != 0)
                    ? (tensor_dim.x / block_size.x + 1)
                    : (tensor_dim.x / block_size.x);
  grid_size.y = ((tensor_dim.y % block_size.y) != 0)
                    ? (tensor_dim.y / block_size.y + 1)
                    : (tensor_dim.y / block_size.y);
  grid_size.z = ((tensor_dim.z % block_size.z) != 0)
                    ? (tensor_dim.z / block_size.z + 1)
                    : (tensor_dim.z / block_size.z);
  if (grid_size.x < 0 || grid_size.y < 0 || grid_size.z < 0) {
    fprintf(stderr, "\n====\n%s: %s! (%s:%d)\n====\n", "C_GridSizeSet",
            "grid_size.x/y/z should > 0", last_function_name,
            last_function_line);
  }
  return grid_size;
}

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
                       const int last_function_line) {
  fprintf(stderr, "\n====\n%s: %s: %s! (%s:%d)\n====\n", local_function_name,
          cudaGetErrorName(error), cudaGetErrorString(error),
          last_function_name, last_function_line);
}

/**
 * @brief make_cudaExtent封装
 *
 * @brief 隐藏make_cudaExtent()细节, 以便统一修改.
 *
 * @param tensor_dim - 三维矩阵长宽高
 *
 * @return cudaExtent
 */
cudaExtent C_MakeCudaExtent(const dim3 tensor_dim) {
  return make_cudaExtent(sizeof(TensorType) * tensor_dim.x, tensor_dim.y,
                         tensor_dim.z);
}

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
                                    const dim3 tensor_dim) {
  return make_cudaPitchedPtr((void*)host_tensor_ptr,
                             sizeof(TensorType) * tensor_dim.x, tensor_dim.x,
                             tensor_dim.y);
}

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
                    const int last_function_line) {
  cudaMemcpy3DParms cpy_param = {0};
  cpy_param.extent = C_MakeCudaExtent(tensor_dim);

  switch (cpy_kind) {
    case cudaMemcpyHostToDevice:
      cpy_param.srcPtr = C_MakeCudaPitchedPtr(host_tensor_ptr, tensor_dim);
      cpy_param.dstPtr = device_tensor_ptr;
      cpy_param.kind = cudaMemcpyHostToDevice;
      break;
    case cudaMemcpyDeviceToHost:
      cpy_param.srcPtr = device_tensor_ptr;
      cpy_param.dstPtr = C_MakeCudaPitchedPtr(host_tensor_ptr, tensor_dim);
      cpy_param.kind = cudaMemcpyDeviceToHost;
      break;
    default:
      fprintf(stderr, "\n====\n%s: %s! (%s:%d)\n====\n", "cudaMemcpy3D",
              "C_CudaMemCpy3D() do not support current cpy_kind",
              last_function_name, last_function_line);
      break;
  }

  cudaError_t cuda_status = cudaMemcpy3D(&cpy_param);
  if (cuda_status != cudaSuccess) {
    C_CudaErrorHandle(cuda_status, "cudaMemcpy3D", last_function_name,
                      last_function_line);
  }
}

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
                        const int last_function_line) {
  cudaError_t cuda_status = cudaMallocHost(
      (void**)&host_tensor_ptr,
      sizeof(TensorType) * tensor_dim.x * tensor_dim.y * tensor_dim.z);
  if (cuda_status != cudaSuccess) {
    C_CudaErrorHandle(cuda_status, "cudaMallocHost", last_function_name,
                      last_function_line);
  }
}

/**
 * @brief Host三维矩阵空间释放
 *
 * @param host_tensor_ptr - Host三维矩阵指针, 使用指针引用
 * @param last_function_name - 父级函数名, 使用 __FUNCTION__ 宏
 * @param last_function_line - 在父级函数中的行号, 使用 __LINE__ 宏
 */
void C_CudaFree3DHost(TensorType*& host_tensor_ptr,
                      const char* last_function_name,
                      const int last_function_line) {
  cudaError_t cuda_status = cudaFreeHost(host_tensor_ptr);
  if (cuda_status != cudaSuccess) {
    C_CudaErrorHandle(cuda_status, "cudaFreeHost", last_function_name,
                      last_function_line);
  }
}

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
                          const int last_function_line) {
  cudaError_t cuda_status =
      cudaMalloc3D(&device_tensor_ptr, C_MakeCudaExtent(tensor_dim));
  if (cuda_status != cudaSuccess) {
    C_CudaErrorHandle(cuda_status, "cudaMalloc3D", last_function_name,
                      last_function_line);
  }
}

/**
 * @brief 定制-Device三维矩阵空间释放
 *
 * @param device_tensor_ptr - device三维矩阵指针, 使用引用
 * @param last_function_name - 父级函数名, 使用 __FUNCTION__ 宏
 * @param last_function_line - 在父级函数中的行号, 使用 __LINE__ 宏
 */
void C_CudaFree3DDevice(cudaPitchedPtr& device_tensor_ptr,
                        const char* last_function_name,
                        const int last_function_line) {
  cudaError_t cuda_status = cudaFree(device_tensor_ptr.ptr);
  if (cuda_status != cudaSuccess) {
    C_CudaErrorHandle(cuda_status, "cudaFree", last_function_name,
                      last_function_line);
  }
}

__global__ void MatAdd(cudaPitchedPtr d_tensor_full,
                       cudaPitchedPtr d_tensor_no_full_e,
                       cudaPitchedPtr d_tensor_no_full_w,
                       cudaPitchedPtr d_tensor_no_full_s,
                       cudaPitchedPtr d_tensor_no_full_n,
                       cudaPitchedPtr d_tensor_no_full_t,
                       cudaPitchedPtr d_tensor_no_full_b) {
  unsigned long long int idx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long long int idy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned long long int idz = threadIdx.z + blockIdx.z * blockDim.z;
  char* device_tensor_ptr = (char*)d_tensor_full.ptr;
  char* device_tensor_ptr_e = (char*)d_tensor_no_full_e.ptr;
  char* device_tensor_ptr_w = (char*)d_tensor_no_full_w.ptr;
  char* device_tensor_ptr_s = (char*)d_tensor_no_full_s.ptr;
  char* device_tensor_ptr_n = (char*)d_tensor_no_full_n.ptr;
  char* device_tensor_ptr_t = (char*)d_tensor_no_full_t.ptr;
  char* device_tensor_ptr_b = (char*)d_tensor_no_full_b.ptr;
  TensorType *index, *index_e, *index_w, *index_s, *index_n, *index_t, *index_b;

  unsigned long long int tensor_yz_pitch =
      idz * d_tensor_full.pitch * LY + idy * d_tensor_full.pitch;
  index = (TensorType*)(device_tensor_ptr + tensor_yz_pitch) + idx;
  index_e = (TensorType*)(device_tensor_ptr_e + tensor_yz_pitch) + idx;
  index_w = (TensorType*)(device_tensor_ptr_w + tensor_yz_pitch) + idx;
  index_s = (TensorType*)(device_tensor_ptr_s + tensor_yz_pitch) + idx;
  index_n = (TensorType*)(device_tensor_ptr_n + tensor_yz_pitch) + idx;
  index_t = (TensorType*)(device_tensor_ptr_t + tensor_yz_pitch) + idx;
  index_b = (TensorType*)(device_tensor_ptr_b + tensor_yz_pitch) + idx;

  TensorType mm = *index = *index_e * 0.1 + *index_w * 0.12 + *index_s * 0.23 +
                           *index_n * 0.21 + *index_t * 0.19 + *index_b * 0.15 +
                           1.0;
  *index_e = mm * 0.1 + *index_e * 0.9;
  *index_w = mm * 0.12 + *index_w * 0.88;
  *index_s = mm * 0.23 + *index_s * 0.77;
  *index_n = mm * 0.21 + *index_n * 0.79;
  *index_t = mm * 0.19 + *index_t * 0.81;
  *index_b = mm * 0.15 + *index_b * 0.85;
}

int main() {
  // C_GetCudaInfo();
  // C_ProfileCopies();

  TensorType* h_tensor_full;
  TensorType* h_tensor_no_full_e;
  TensorType* h_tensor_no_full_w;
  TensorType* h_tensor_no_full_s;
  TensorType* h_tensor_no_full_n;
  TensorType* h_tensor_no_full_t;
  TensorType* h_tensor_no_full_b;
  dim3 tensor_dim = {LX, LY, LZ};

  C_CudaMalloc3DHost(h_tensor_full, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DHost(h_tensor_no_full_e, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DHost(h_tensor_no_full_w, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DHost(h_tensor_no_full_s, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DHost(h_tensor_no_full_n, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DHost(h_tensor_no_full_t, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DHost(h_tensor_no_full_b, tensor_dim, __FUNCTION__, __LINE__);

  cudaError_t cudaStatus;

  for (int i = 0; i < LX * LY * LZ; i++) {
    h_tensor_full[i] = 1;
    h_tensor_no_full_e[i] = 2;
    h_tensor_no_full_w[i] = 3;
    h_tensor_no_full_s[i] = 4;
    h_tensor_no_full_n[i] = 5;
    h_tensor_no_full_t[i] = 6;
    h_tensor_no_full_b[i] = 7;
  }

  cudaPitchedPtr d_tensor_full;
  cudaPitchedPtr d_tensor_no_full_e;
  cudaPitchedPtr d_tensor_no_full_w;
  cudaPitchedPtr d_tensor_no_full_s;
  cudaPitchedPtr d_tensor_no_full_n;
  cudaPitchedPtr d_tensor_no_full_t;
  cudaPitchedPtr d_tensor_no_full_b;

  C_CudaMalloc3DDevice(d_tensor_full, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(d_tensor_no_full_e, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(d_tensor_no_full_w, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(d_tensor_no_full_s, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(d_tensor_no_full_n, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(d_tensor_no_full_t, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(d_tensor_no_full_b, tensor_dim, __FUNCTION__, __LINE__);

  int blocksize_x = 32;
  int blocksize_y = 16;
  int blocksize_z = 2;
  dim3 block_size(blocksize_x, blocksize_y, blocksize_z);
  dim3 grid_size =
      C_GridSizeSet(tensor_dim, block_size, __FUNCTION__, __LINE__);

  clock_t start, end;
  start = clock();

  std::cout << h_tensor_no_full_b[(9 * LY + 399) * LX + 999] << "\t";

  C_CudaMemCpy3D(h_tensor_full, d_tensor_full, cudaMemcpyHostToDevice,
                 tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMemCpy3D(h_tensor_no_full_e, d_tensor_no_full_e, cudaMemcpyHostToDevice,
                 tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMemCpy3D(h_tensor_no_full_w, d_tensor_no_full_w, cudaMemcpyHostToDevice,
                 tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMemCpy3D(h_tensor_no_full_s, d_tensor_no_full_s, cudaMemcpyHostToDevice,
                 tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMemCpy3D(h_tensor_no_full_n, d_tensor_no_full_n, cudaMemcpyHostToDevice,
                 tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMemCpy3D(h_tensor_no_full_t, d_tensor_no_full_t, cudaMemcpyHostToDevice,
                 tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMemCpy3D(h_tensor_no_full_b, d_tensor_no_full_b, cudaMemcpyHostToDevice,
                 tensor_dim, __FUNCTION__, __LINE__);

  for (int i = 0; i < 1000; i++) {
    cudaDeviceSynchronize();
    MatAdd<<<grid_size, block_size>>>(d_tensor_full, d_tensor_no_full_e,
                                      d_tensor_no_full_w, d_tensor_no_full_s,
                                      d_tensor_no_full_n, d_tensor_no_full_t,
                                      d_tensor_no_full_b);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "addKernel launch failed: %s\n",
              cudaGetErrorString(cudaStatus));
    }
    cudaDeviceSynchronize();

    // std::cout << i << std::endl;
  }

  C_CudaMemCpy3D(h_tensor_full, d_tensor_full, cudaMemcpyDeviceToHost,
                 tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMemCpy3D(h_tensor_no_full_e, d_tensor_no_full_e, cudaMemcpyDeviceToHost,
                 tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMemCpy3D(h_tensor_no_full_w, d_tensor_no_full_w, cudaMemcpyDeviceToHost,
                 tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMemCpy3D(h_tensor_no_full_s, d_tensor_no_full_s, cudaMemcpyDeviceToHost,
                 tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMemCpy3D(h_tensor_no_full_n, d_tensor_no_full_n, cudaMemcpyDeviceToHost,
                 tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMemCpy3D(h_tensor_no_full_t, d_tensor_no_full_t, cudaMemcpyDeviceToHost,
                 tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMemCpy3D(h_tensor_no_full_b, d_tensor_no_full_b, cudaMemcpyDeviceToHost,
                 tensor_dim, __FUNCTION__, __LINE__);

  cudaDeviceSynchronize();
  std::cout << h_tensor_full[(9 * LY + 399) * LX + 999] << "\t";

  end = clock();  // 结束时间
  std::cout << "耗时 = " << double(end - start) / CLOCKS_PER_SEC << "s"
            << std::endl;  // 输出时间（单位：ｓ）

  C_CudaFree3DDevice(d_tensor_full, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(d_tensor_no_full_e, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(d_tensor_no_full_w, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(d_tensor_no_full_s, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(d_tensor_no_full_n, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(d_tensor_no_full_t, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(d_tensor_no_full_b, __FUNCTION__, __LINE__);

  C_CudaFree3DHost(h_tensor_full, __FUNCTION__, __LINE__);
  C_CudaFree3DHost(h_tensor_no_full_e, __FUNCTION__, __LINE__);
  C_CudaFree3DHost(h_tensor_no_full_w, __FUNCTION__, __LINE__);
  C_CudaFree3DHost(h_tensor_no_full_s, __FUNCTION__, __LINE__);
  C_CudaFree3DHost(h_tensor_no_full_n, __FUNCTION__, __LINE__);
  C_CudaFree3DHost(h_tensor_no_full_t, __FUNCTION__, __LINE__);
  C_CudaFree3DHost(h_tensor_no_full_b, __FUNCTION__, __LINE__);
  return 0;
}