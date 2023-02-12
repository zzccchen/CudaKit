#include "CudaKit.h"

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

void C_CudaErrorHandle(const cudaError_t error,
                       const char* local_function_name,
                       const char* last_function_name,
                       const int last_function_line) {
  fprintf(stderr, "\n====\n%s: %s: %s! (%s:%d)\n====\n", local_function_name,
          cudaGetErrorName(error), cudaGetErrorString(error),
          last_function_name, last_function_line);
}

cudaExtent C_MakeCudaExtent(const dim3 tensor_dim) {
  return make_cudaExtent(sizeof(TensorType) * tensor_dim.x, tensor_dim.y,
                         tensor_dim.z);
}

cudaExtent C_MakeCudaExtent19(const dim3 tensor_dim) {
  return make_cudaExtent(sizeof(T_19D_S) * tensor_dim.x, tensor_dim.y,
                         tensor_dim.z);
}

cudaPitchedPtr C_MakeCudaPitchedPtr(TensorType*& host_tensor_ptr,
                                    const dim3 tensor_dim) {
  return make_cudaPitchedPtr((void*)host_tensor_ptr,
                             sizeof(TensorType) * tensor_dim.x, tensor_dim.x,
                             tensor_dim.y);
}

cudaPitchedPtr C_MakeCudaPitchedPtr19(T_19D_S*& host_tensor_ptr,
                                      const dim3 tensor_dim) {
  return make_cudaPitchedPtr((void*)host_tensor_ptr,
                             sizeof(T_19D_S) * tensor_dim.x, tensor_dim.x,
                             tensor_dim.y);
}

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

void C_CudaMemCpy3D19(T_19D_S*& host_tensor_ptr,
                      cudaPitchedPtr& device_tensor_ptr,
                      const enum cudaMemcpyKind cpy_kind,
                      const dim3 tensor_dim,
                      const char* last_function_name,
                      const int last_function_line) {
  cudaMemcpy3DParms cpy_param = {0};
  cpy_param.extent = C_MakeCudaExtent19(tensor_dim);

  switch (cpy_kind) {
    case cudaMemcpyHostToDevice:
      cpy_param.srcPtr = C_MakeCudaPitchedPtr19(host_tensor_ptr, tensor_dim);
      cpy_param.dstPtr = device_tensor_ptr;
      cpy_param.kind = cudaMemcpyHostToDevice;
      break;
    case cudaMemcpyDeviceToHost:
      cpy_param.srcPtr = device_tensor_ptr;
      cpy_param.dstPtr = C_MakeCudaPitchedPtr19(host_tensor_ptr, tensor_dim);
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

void C_CudaMalloc3DHost19(T_19D_S*& host_tensor_ptr,
                          const dim3 tensor_dim,
                          const char* last_function_name,
                          const int last_function_line) {
  cudaError_t cuda_status =
      cudaMallocHost((void**)&host_tensor_ptr, sizeof(T_19D_S) * tensor_dim.x *
                                                   tensor_dim.y * tensor_dim.z);
  if (cuda_status != cudaSuccess) {
    C_CudaErrorHandle(cuda_status, "cudaMallocHost", last_function_name,
                      last_function_line);
  }
}

void C_CudaFree3DHost(TensorType*& host_tensor_ptr,
                      const char* last_function_name,
                      const int last_function_line) {
  cudaError_t cuda_status = cudaFreeHost(host_tensor_ptr);
  if (cuda_status != cudaSuccess) {
    C_CudaErrorHandle(cuda_status, "cudaFreeHost", last_function_name,
                      last_function_line);
  }
}

void C_CudaFree3DHost19(T_19D_S*& host_tensor_ptr,
                        const char* last_function_name,
                        const int last_function_line) {
  cudaError_t cuda_status = cudaFreeHost(host_tensor_ptr);
  if (cuda_status != cudaSuccess) {
    C_CudaErrorHandle(cuda_status, "cudaFreeHost", last_function_name,
                      last_function_line);
  }
}

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

void C_CudaMalloc3DDevice19(cudaPitchedPtr& device_tensor_ptr,
                            const dim3 tensor_dim,
                            const char* last_function_name,
                            const int last_function_line) {
  cudaError_t cuda_status =
      cudaMalloc3D(&device_tensor_ptr, C_MakeCudaExtent19(tensor_dim));
  if (cuda_status != cudaSuccess) {
    C_CudaErrorHandle(cuda_status, "cudaMalloc3D", last_function_name,
                      last_function_line);
  }
}

void C_CudaFree3DDevice(cudaPitchedPtr& device_tensor_ptr,
                        const char* last_function_name,
                        const int last_function_line) {
  cudaError_t cuda_status = cudaFree(device_tensor_ptr.ptr);
  if (cuda_status != cudaSuccess) {
    C_CudaErrorHandle(cuda_status, "cudaFree", last_function_name,
                      last_function_line);
  }
}