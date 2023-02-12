#include "CudaKit.h"

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

void Test1() {
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
}

int main() {
  // C_GetCudaInfo();
  // C_ProfileCopies();
  Test1();
}