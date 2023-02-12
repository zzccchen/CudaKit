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

__global__ void C_TestSOA(char* ptr_d_0,
                          char* ptr_d_1,
                          char* ptr_d_2,
                          char* ptr_d_3,
                          char* ptr_d_4,
                          char* ptr_d_5,
                          char* ptr_d_6,
                          char* ptr_d_7,
                          char* ptr_d_8,
                          char* ptr_d_9,
                          char* ptr_d_10,
                          char* ptr_d_11,
                          char* ptr_d_12,
                          char* ptr_d_13,
                          char* ptr_d_14,
                          char* ptr_d_15,
                          char* ptr_d_16,
                          char* ptr_d_17,
                          char* ptr_d_18,
                          char* ptr_dd_0,
                          char* ptr_dd_1,
                          char* ptr_dd_2,
                          char* ptr_dd_3,
                          char* ptr_dd_4,
                          char* ptr_dd_5,
                          char* ptr_dd_6,
                          char* ptr_dd_7,
                          char* ptr_dd_8,
                          char* ptr_dd_9,
                          char* ptr_dd_10,
                          char* ptr_dd_11,
                          char* ptr_dd_12,
                          char* ptr_dd_13,
                          char* ptr_dd_14,
                          char* ptr_dd_15,
                          char* ptr_dd_16,
                          char* ptr_dd_17,
                          char* ptr_dd_18,
                          int lx_1,
                          int ly_1,
                          int lz_1,
                          int tensor_pitch_E1,
                          int tensor_pitch_x_dim_y_E1) {
  int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int global_idy = threadIdx.y + blockIdx.y * blockDim.y;
  int global_idz = threadIdx.z + blockIdx.z * blockDim.z;

  if (global_idx > lx_1 || global_idy > ly_1 || global_idz > lz_1) {
    return;
  }

  TensorType f_curr[19];

  const int tensor_yz_pitch_E1 =
      global_idz * tensor_pitch_x_dim_y_E1 + global_idy * tensor_pitch_E1;

  f_curr[0] = *((TensorType*)(ptr_d_0 + tensor_yz_pitch_E1) + global_idx);
  f_curr[1] = *((TensorType*)(ptr_d_1 + tensor_yz_pitch_E1) + global_idx);
  f_curr[2] = *((TensorType*)(ptr_d_2 + tensor_yz_pitch_E1) + global_idx);
  f_curr[3] = *((TensorType*)(ptr_d_3 + tensor_yz_pitch_E1) + global_idx);
  f_curr[4] = *((TensorType*)(ptr_d_4 + tensor_yz_pitch_E1) + global_idx);
  f_curr[5] = *((TensorType*)(ptr_d_5 + tensor_yz_pitch_E1) + global_idx);
  f_curr[6] = *((TensorType*)(ptr_d_6 + tensor_yz_pitch_E1) + global_idx);
  f_curr[7] = *((TensorType*)(ptr_d_7 + tensor_yz_pitch_E1) + global_idx);
  f_curr[8] = *((TensorType*)(ptr_d_8 + tensor_yz_pitch_E1) + global_idx);
  f_curr[9] = *((TensorType*)(ptr_d_9 + tensor_yz_pitch_E1) + global_idx);
  f_curr[10] = *((TensorType*)(ptr_d_10 + tensor_yz_pitch_E1) + global_idx);
  f_curr[11] = *((TensorType*)(ptr_d_11 + tensor_yz_pitch_E1) + global_idx);
  f_curr[12] = *((TensorType*)(ptr_d_12 + tensor_yz_pitch_E1) + global_idx);
  f_curr[13] = *((TensorType*)(ptr_d_13 + tensor_yz_pitch_E1) + global_idx);
  f_curr[14] = *((TensorType*)(ptr_d_14 + tensor_yz_pitch_E1) + global_idx);
  f_curr[15] = *((TensorType*)(ptr_d_15 + tensor_yz_pitch_E1) + global_idx);
  f_curr[16] = *((TensorType*)(ptr_d_16 + tensor_yz_pitch_E1) + global_idx);
  f_curr[17] = *((TensorType*)(ptr_d_17 + tensor_yz_pitch_E1) + global_idx);
  f_curr[18] = *((TensorType*)(ptr_d_18 + tensor_yz_pitch_E1) + global_idx);

  f_curr[0] = f_curr[0] + f_curr[1] + f_curr[2];
  f_curr[3] = f_curr[3] + f_curr[4] + f_curr[5] + f_curr[6];
  f_curr[7] = f_curr[7] + f_curr[8] + f_curr[9] + f_curr[10];
  f_curr[10] = f_curr[10] + f_curr[11] + f_curr[12] + f_curr[13] + f_curr[14];
  f_curr[14] = f_curr[14] + f_curr[15] + f_curr[16] + f_curr[17] + f_curr[18];

  TensorType* f_dd_0 =
      (TensorType*)(ptr_dd_0 + tensor_yz_pitch_E1) + global_idx;
  TensorType* f_dd_1 =
      (TensorType*)(ptr_dd_1 + tensor_yz_pitch_E1) + global_idx;
  TensorType* f_dd_2 =
      (TensorType*)(ptr_dd_2 + tensor_yz_pitch_E1) + global_idx;
  TensorType* f_dd_3 =
      (TensorType*)(ptr_dd_3 + tensor_yz_pitch_E1) + global_idx;
  TensorType* f_dd_4 =
      (TensorType*)(ptr_dd_4 + tensor_yz_pitch_E1) + global_idx;
  TensorType* f_dd_5 =
      (TensorType*)(ptr_dd_5 + tensor_yz_pitch_E1) + global_idx;
  TensorType* f_dd_6 =
      (TensorType*)(ptr_dd_6 + tensor_yz_pitch_E1) + global_idx;
  TensorType* f_dd_7 =
      (TensorType*)(ptr_dd_7 + tensor_yz_pitch_E1) + global_idx;
  TensorType* f_dd_8 =
      (TensorType*)(ptr_dd_8 + tensor_yz_pitch_E1) + global_idx;
  TensorType* f_dd_9 =
      (TensorType*)(ptr_dd_9 + tensor_yz_pitch_E1) + global_idx;
  TensorType* f_dd_10 =
      (TensorType*)(ptr_dd_10 + tensor_yz_pitch_E1) + global_idx;
  TensorType* f_dd_11 =
      (TensorType*)(ptr_dd_11 + tensor_yz_pitch_E1) + global_idx;
  TensorType* f_dd_12 =
      (TensorType*)(ptr_dd_12 + tensor_yz_pitch_E1) + global_idx;
  TensorType* f_dd_13 =
      (TensorType*)(ptr_dd_13 + tensor_yz_pitch_E1) + global_idx;
  TensorType* f_dd_14 =
      (TensorType*)(ptr_dd_14 + tensor_yz_pitch_E1) + global_idx;
  TensorType* f_dd_15 =
      (TensorType*)(ptr_dd_15 + tensor_yz_pitch_E1) + global_idx;
  TensorType* f_dd_16 =
      (TensorType*)(ptr_dd_16 + tensor_yz_pitch_E1) + global_idx;
  TensorType* f_dd_17 =
      (TensorType*)(ptr_dd_17 + tensor_yz_pitch_E1) + global_idx;
  TensorType* f_dd_18 =
      (TensorType*)(ptr_dd_18 + tensor_yz_pitch_E1) + global_idx;

  *f_dd_0 = f_curr[0];
  *f_dd_1 = f_curr[1];
  *f_dd_2 = f_curr[2];
  *f_dd_3 = f_curr[3];
  *f_dd_4 = f_curr[4];
  *f_dd_5 = f_curr[5];
  *f_dd_6 = f_curr[6];
  *f_dd_7 = f_curr[7];
  *f_dd_8 = f_curr[8];
  *f_dd_9 = f_curr[9];
  *f_dd_10 = f_curr[10];
  *f_dd_11 = f_curr[11];
  *f_dd_12 = f_curr[12];
  *f_dd_13 = f_curr[13];
  *f_dd_14 = f_curr[14];
  *f_dd_15 = f_curr[15];
  *f_dd_16 = f_curr[16];
  *f_dd_17 = f_curr[17];
  *f_dd_18 = f_curr[18];
}

void TestSOA() {
  TensorType* h_0;
  TensorType* h_1;
  TensorType* h_2;
  TensorType* h_3;
  TensorType* h_4;
  TensorType* h_5;
  TensorType* h_6;
  TensorType* h_7;
  TensorType* h_8;
  TensorType* h_9;
  TensorType* h_10;
  TensorType* h_11;
  TensorType* h_12;
  TensorType* h_13;
  TensorType* h_14;
  TensorType* h_15;
  TensorType* h_16;
  TensorType* h_17;
  TensorType* h_18;

  cudaPitchedPtr d_0;
  cudaPitchedPtr d_1;
  cudaPitchedPtr d_2;
  cudaPitchedPtr d_3;
  cudaPitchedPtr d_4;
  cudaPitchedPtr d_5;
  cudaPitchedPtr d_6;
  cudaPitchedPtr d_7;
  cudaPitchedPtr d_8;
  cudaPitchedPtr d_9;
  cudaPitchedPtr d_10;
  cudaPitchedPtr d_11;
  cudaPitchedPtr d_12;
  cudaPitchedPtr d_13;
  cudaPitchedPtr d_14;
  cudaPitchedPtr d_15;
  cudaPitchedPtr d_16;
  cudaPitchedPtr d_17;
  cudaPitchedPtr d_18;

  cudaPitchedPtr dd_0;
  cudaPitchedPtr dd_1;
  cudaPitchedPtr dd_2;
  cudaPitchedPtr dd_3;
  cudaPitchedPtr dd_4;
  cudaPitchedPtr dd_5;
  cudaPitchedPtr dd_6;
  cudaPitchedPtr dd_7;
  cudaPitchedPtr dd_8;
  cudaPitchedPtr dd_9;
  cudaPitchedPtr dd_10;
  cudaPitchedPtr dd_11;
  cudaPitchedPtr dd_12;
  cudaPitchedPtr dd_13;
  cudaPitchedPtr dd_14;
  cudaPitchedPtr dd_15;
  cudaPitchedPtr dd_16;
  cudaPitchedPtr dd_17;
  cudaPitchedPtr dd_18;
  dim3 tensor_dim = {LX, LY, LZ};

  C_CudaMalloc3DHost(h_0, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DHost(h_1, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DHost(h_2, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DHost(h_3, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DHost(h_4, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DHost(h_5, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DHost(h_6, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DHost(h_7, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DHost(h_8, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DHost(h_9, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DHost(h_10, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DHost(h_11, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DHost(h_12, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DHost(h_13, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DHost(h_14, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DHost(h_15, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DHost(h_16, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DHost(h_17, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DHost(h_18, tensor_dim, __FUNCTION__, __LINE__);

  cudaError_t cudaStatus;

  for (int i = 0; i < LX * LY * LZ; i++) {
    h_0[i] = 0.01 * i;
    h_1[i] = 0.02 * i;
    h_2[i] = 0.03 * i;
    h_3[i] = 0.04 * i;
    h_4[i] = 0.05 * i;
    h_5[i] = 0.06 * i;
    h_6[i] = 0.07 * i;
    h_7[i] = 0.08 * i;
    h_8[i] = 0.09 * i;
    h_9[i] = 0.10 * i;
    h_10[i] = 0.11 * i;
    h_11[i] = 0.12 * i;
    h_12[i] = 0.13 * i;
    h_13[i] = 0.14 * i;
    h_14[i] = 0.15 * i;
    h_15[i] = 0.16 * i;
    h_16[i] = 0.17 * i;
    h_17[i] = 0.18 * i;
    h_18[i] = 0.19 * i;
  }

  C_CudaMalloc3DDevice(d_0, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(d_1, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(d_2, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(d_3, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(d_4, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(d_5, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(d_6, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(d_7, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(d_8, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(d_9, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(d_10, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(d_11, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(d_12, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(d_13, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(d_14, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(d_15, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(d_16, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(d_17, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(d_18, tensor_dim, __FUNCTION__, __LINE__);

  C_CudaMalloc3DDevice(dd_0, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(dd_1, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(dd_2, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(dd_3, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(dd_4, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(dd_5, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(dd_6, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(dd_7, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(dd_8, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(dd_9, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(dd_10, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(dd_11, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(dd_12, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(dd_13, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(dd_14, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(dd_15, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(dd_16, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(dd_17, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice(dd_18, tensor_dim, __FUNCTION__, __LINE__);

  C_CudaMemCpy3D(h_0, dd_0, cudaMemcpyHostToDevice, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_1, dd_1, cudaMemcpyHostToDevice, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_2, dd_2, cudaMemcpyHostToDevice, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_3, dd_3, cudaMemcpyHostToDevice, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_4, dd_4, cudaMemcpyHostToDevice, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_5, dd_5, cudaMemcpyHostToDevice, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_6, dd_6, cudaMemcpyHostToDevice, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_7, dd_7, cudaMemcpyHostToDevice, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_8, dd_8, cudaMemcpyHostToDevice, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_9, dd_9, cudaMemcpyHostToDevice, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_10, dd_10, cudaMemcpyHostToDevice, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_11, dd_11, cudaMemcpyHostToDevice, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_12, dd_12, cudaMemcpyHostToDevice, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_13, dd_13, cudaMemcpyHostToDevice, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_14, dd_14, cudaMemcpyHostToDevice, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_15, dd_15, cudaMemcpyHostToDevice, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_16, dd_16, cudaMemcpyHostToDevice, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_17, dd_17, cudaMemcpyHostToDevice, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_18, dd_18, cudaMemcpyHostToDevice, tensor_dim, __FUNCTION__,
                 __LINE__);

  char* ptr_d_0 = (char*)d_0.ptr;
  char* ptr_d_1 = (char*)d_1.ptr;
  char* ptr_d_2 = (char*)d_2.ptr;
  char* ptr_d_3 = (char*)d_3.ptr;
  char* ptr_d_4 = (char*)d_4.ptr;
  char* ptr_d_5 = (char*)d_5.ptr;
  char* ptr_d_6 = (char*)d_6.ptr;
  char* ptr_d_7 = (char*)d_7.ptr;
  char* ptr_d_8 = (char*)d_8.ptr;
  char* ptr_d_9 = (char*)d_9.ptr;
  char* ptr_d_10 = (char*)d_10.ptr;
  char* ptr_d_11 = (char*)d_11.ptr;
  char* ptr_d_12 = (char*)d_12.ptr;
  char* ptr_d_13 = (char*)d_13.ptr;
  char* ptr_d_14 = (char*)d_14.ptr;
  char* ptr_d_15 = (char*)d_15.ptr;
  char* ptr_d_16 = (char*)d_16.ptr;
  char* ptr_d_17 = (char*)d_17.ptr;
  char* ptr_d_18 = (char*)d_18.ptr;
  char* ptr_dd_0 = (char*)dd_0.ptr;
  char* ptr_dd_1 = (char*)dd_1.ptr;
  char* ptr_dd_2 = (char*)dd_2.ptr;
  char* ptr_dd_3 = (char*)dd_3.ptr;
  char* ptr_dd_4 = (char*)dd_4.ptr;
  char* ptr_dd_5 = (char*)dd_5.ptr;
  char* ptr_dd_6 = (char*)dd_6.ptr;
  char* ptr_dd_7 = (char*)dd_7.ptr;
  char* ptr_dd_8 = (char*)dd_8.ptr;
  char* ptr_dd_9 = (char*)dd_9.ptr;
  char* ptr_dd_10 = (char*)dd_10.ptr;
  char* ptr_dd_11 = (char*)dd_11.ptr;
  char* ptr_dd_12 = (char*)dd_12.ptr;
  char* ptr_dd_13 = (char*)dd_13.ptr;
  char* ptr_dd_14 = (char*)dd_14.ptr;
  char* ptr_dd_15 = (char*)dd_15.ptr;
  char* ptr_dd_16 = (char*)dd_16.ptr;
  char* ptr_dd_17 = (char*)dd_17.ptr;
  char* ptr_dd_18 = (char*)dd_18.ptr;

  int blocksize_x = 32;
  int blocksize_y = 16;
  int blocksize_z = 2;
  dim3 block_size(blocksize_x, blocksize_y, blocksize_z);
  dim3 grid_size =
      C_GridSizeSet(tensor_dim, block_size, __FUNCTION__, __LINE__);

  int lx_1 = LX - 1;
  int ly_1 = LY - 1;
  int lz_1 = LZ - 1;

  const int tensor_pitch_E1 = int(d_0.pitch);
  const int tensor_pitch_x_dim_y_E1 = tensor_pitch_E1 * tensor_dim.y;

  clock_t start, end;
  start = clock();

  std::cout << h_1[(9 * LY + 399) * LX + 999] << "\t";
  for (int m = 0; m < 1000; m++) {
    cudaDeviceSynchronize();
    C_TestSOA<<<grid_size, block_size>>>(
        ptr_d_0, ptr_d_1, ptr_d_2, ptr_d_3, ptr_d_4, ptr_d_5, ptr_d_6, ptr_d_7,
        ptr_d_8, ptr_d_9, ptr_d_10, ptr_d_11, ptr_d_12, ptr_d_13, ptr_d_14,
        ptr_d_15, ptr_d_16, ptr_d_17, ptr_d_18, ptr_dd_0, ptr_dd_1, ptr_dd_2,
        ptr_dd_3, ptr_dd_4, ptr_dd_5, ptr_dd_6, ptr_dd_7, ptr_dd_8, ptr_dd_9,
        ptr_dd_10, ptr_dd_11, ptr_dd_12, ptr_dd_13, ptr_dd_14, ptr_dd_15,
        ptr_dd_16, ptr_dd_17, ptr_dd_18, lx_1, ly_1, lz_1, tensor_pitch_E1,
        tensor_pitch_x_dim_y_E1);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "TestSOA failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    cudaDeviceSynchronize();
  }

  // std::cout << i << std::endl;
  C_CudaMemCpy3D(h_0, d_0, cudaMemcpyDeviceToHost, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_1, d_1, cudaMemcpyDeviceToHost, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_2, d_2, cudaMemcpyDeviceToHost, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_3, d_3, cudaMemcpyDeviceToHost, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_4, d_4, cudaMemcpyDeviceToHost, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_5, d_5, cudaMemcpyDeviceToHost, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_6, d_6, cudaMemcpyDeviceToHost, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_7, d_7, cudaMemcpyDeviceToHost, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_8, d_8, cudaMemcpyDeviceToHost, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_9, d_9, cudaMemcpyDeviceToHost, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_10, d_10, cudaMemcpyDeviceToHost, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_11, d_11, cudaMemcpyDeviceToHost, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_12, d_12, cudaMemcpyDeviceToHost, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_13, d_13, cudaMemcpyDeviceToHost, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_14, d_14, cudaMemcpyDeviceToHost, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_15, d_15, cudaMemcpyDeviceToHost, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_16, d_16, cudaMemcpyDeviceToHost, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_17, d_17, cudaMemcpyDeviceToHost, tensor_dim, __FUNCTION__,
                 __LINE__);
  C_CudaMemCpy3D(h_18, d_18, cudaMemcpyDeviceToHost, tensor_dim, __FUNCTION__,
                 __LINE__);

  cudaDeviceSynchronize();
  std::cout << h_17[(9 * LY + 399) * LX + 999] << "\t";

  end = clock();  // 结束时间
  std::cout << "耗时 = " << double(end - start) / CLOCKS_PER_SEC << "s"
            << std::endl;  // 输出时间（单位：ｓ）

  C_CudaFree3DDevice(d_0, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(d_1, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(d_2, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(d_3, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(d_4, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(d_5, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(d_6, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(d_7, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(d_8, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(d_9, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(d_10, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(d_11, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(d_12, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(d_13, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(d_14, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(d_15, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(d_16, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(d_17, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(d_18, __FUNCTION__, __LINE__);

  C_CudaFree3DDevice(dd_0, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(dd_1, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(dd_2, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(dd_3, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(dd_4, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(dd_5, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(dd_6, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(dd_7, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(dd_8, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(dd_9, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(dd_10, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(dd_11, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(dd_12, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(dd_13, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(dd_14, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(dd_15, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(dd_16, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(dd_17, __FUNCTION__, __LINE__);
  C_CudaFree3DDevice(dd_18, __FUNCTION__, __LINE__);

  C_CudaFree3DHost(h_0, __FUNCTION__, __LINE__);
  C_CudaFree3DHost(h_1, __FUNCTION__, __LINE__);
  C_CudaFree3DHost(h_2, __FUNCTION__, __LINE__);
  C_CudaFree3DHost(h_3, __FUNCTION__, __LINE__);
  C_CudaFree3DHost(h_4, __FUNCTION__, __LINE__);
  C_CudaFree3DHost(h_5, __FUNCTION__, __LINE__);
  C_CudaFree3DHost(h_6, __FUNCTION__, __LINE__);
  C_CudaFree3DHost(h_7, __FUNCTION__, __LINE__);
  C_CudaFree3DHost(h_8, __FUNCTION__, __LINE__);
  C_CudaFree3DHost(h_9, __FUNCTION__, __LINE__);
  C_CudaFree3DHost(h_10, __FUNCTION__, __LINE__);
  C_CudaFree3DHost(h_11, __FUNCTION__, __LINE__);
  C_CudaFree3DHost(h_12, __FUNCTION__, __LINE__);
  C_CudaFree3DHost(h_13, __FUNCTION__, __LINE__);
  C_CudaFree3DHost(h_14, __FUNCTION__, __LINE__);
  C_CudaFree3DHost(h_15, __FUNCTION__, __LINE__);
  C_CudaFree3DHost(h_16, __FUNCTION__, __LINE__);
  C_CudaFree3DHost(h_17, __FUNCTION__, __LINE__);
  C_CudaFree3DHost(h_18, __FUNCTION__, __LINE__);
}

__global__ void C_TestAOS(char* ptr_d_E19,
                          char* ptr_dd_E19,
                          int lx_1,
                          int ly_1,
                          int lz_1,
                          int tensor_pitch_E19,
                          int tensor_pitch_x_dim_y_E19) {
  int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int global_idy = threadIdx.y + blockIdx.y * blockDim.y;
  int global_idz = threadIdx.z + blockIdx.z * blockDim.z;

  if (global_idx > lx_1 || global_idy > ly_1 || global_idz > lz_1) {
    return;
  }

  T_19D_S f_curr;

  const int tensor_yz_pitch_E19 =
      global_idz * tensor_pitch_x_dim_y_E19 + global_idy * tensor_pitch_E19;

  f_curr = *((T_19D_S*)(ptr_d_E19 + tensor_yz_pitch_E19) + global_idx);

  f_curr.Q0 = f_curr.Q0 + f_curr.Q1 + f_curr.Q2;
  f_curr.Q3 = f_curr.Q3 + f_curr.Q4 + f_curr.Q5 + f_curr.Q6;
  f_curr.Q7 = f_curr.Q7 + f_curr.Q8 + f_curr.Q9 + f_curr.Q10;
  f_curr.Q10 = f_curr.Q10 + f_curr.Q11 + f_curr.Q12 + f_curr.Q13 + f_curr.Q14;
  f_curr.Q14 = f_curr.Q14 + f_curr.Q15 + f_curr.Q16 + f_curr.Q17 + f_curr.Q18;
  f_curr._ = f_curr._ + 1.0;

  T_19D_S* f_dd = (T_19D_S*)(ptr_dd_E19 + tensor_yz_pitch_E19) + global_idx;

  *f_dd = f_curr;
}

void TestAOS() {
  T_19D_S* h;

  cudaPitchedPtr d;
  cudaPitchedPtr dd;
  dim3 tensor_dim = {LX, LY, LZ};

  C_CudaMalloc3DHost19(h, tensor_dim, __FUNCTION__, __LINE__);

  cudaError_t cudaStatus;

  for (int i = 0; i < LX * LY * LZ; i++) {
    h[i].Q0 = 0.01 * i;
    h[i].Q1 = 0.02 * i;
    h[i].Q2 = 0.03 * i;
    h[i].Q3 = 0.04 * i;
    h[i].Q4 = 0.05 * i;
    h[i].Q5 = 0.06 * i;
    h[i].Q6 = 0.07 * i;
    h[i].Q7 = 0.08 * i;
    h[i].Q8 = 0.09 * i;
    h[i].Q9 = 0.10 * i;
    h[i].Q10 = 0.11 * i;
    h[i].Q11 = 0.12 * i;
    h[i].Q12 = 0.13 * i;
    h[i].Q13 = 0.14 * i;
    h[i].Q14 = 0.15 * i;
    h[i].Q15 = 0.16 * i;
    h[i].Q16 = 0.17 * i;
    h[i].Q17 = 0.18 * i;
    h[i].Q18 = 0.19 * i;
  }

  C_CudaMalloc3DDevice19(d, tensor_dim, __FUNCTION__, __LINE__);
  C_CudaMalloc3DDevice19(dd, tensor_dim, __FUNCTION__, __LINE__);

  C_CudaMemCpy3D19(h, d, cudaMemcpyHostToDevice, tensor_dim, __FUNCTION__,
                   __LINE__);

  char* ptr_d = (char*)d.ptr;
  char* ptr_dd = (char*)dd.ptr;

  int blocksize_x = 32;
  int blocksize_y = 16;
  int blocksize_z = 2;
  dim3 block_size(blocksize_x, blocksize_y, blocksize_z);
  dim3 grid_size =
      C_GridSizeSet(tensor_dim, block_size, __FUNCTION__, __LINE__);

  int lx_1 = LX - 1;
  int ly_1 = LY - 1;
  int lz_1 = LZ - 1;

  const int tensor_pitch_E19 = int(d.pitch);
  const int tensor_pitch_x_dim_y_E19 = tensor_pitch_E19 * tensor_dim.y;

  clock_t start, end;
  start = clock();

  std::cout << h[(9 * LY + 399) * LX + 999].Q1 << "\t";
  for (int m = 0; m < 1000; m++) {
    cudaDeviceSynchronize();
    C_TestAOS<<<grid_size, block_size>>>(ptr_d, ptr_dd, lx_1, ly_1, lz_1,
                                         tensor_pitch_E19,
                                         tensor_pitch_x_dim_y_E19);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "TestSOA failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    cudaDeviceSynchronize();
  }

  // std::cout << i << std::endl;
  C_CudaMemCpy3D19(h, dd, cudaMemcpyDeviceToHost, tensor_dim, __FUNCTION__,
                   __LINE__);

  cudaDeviceSynchronize();
  std::cout << h[(9 * LY + 399) * LX + 999].Q17 << "\t";

  end = clock();  // 结束时间
  std::cout << "耗时 = " << double(end - start) / CLOCKS_PER_SEC << "s"
            << std::endl;  // 输出时间（单位：ｓ）

  C_CudaFree3DDevice(d, __FUNCTION__, __LINE__);

  C_CudaFree3DDevice(dd, __FUNCTION__, __LINE__);

  C_CudaFree3DHost19(h, __FUNCTION__, __LINE__);
}

int main() {
  // C_GetCudaInfo();
  // C_ProfileCopies();
  TestSOA();
  // TestAOS();
}