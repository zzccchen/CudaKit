#include <iostream>

#include "cuda_runtime.h"

void C_GetCudaInfo() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  int dev;
  for (dev = 0; dev < deviceCount; dev++) {
    int driver_version(0), runtime_version(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    if (dev == 0)
      if (deviceProp.minor = 9999 && deviceProp.major == 9999)
        std::cout << std::endl;
    std::cout << "使用GPU device " << dev << ": " << deviceProp.name
              << std::endl;
    cudaDriverGetVersion(&driver_version);
    std::cout << "CUDA驱动版本:" << driver_version / 1000 << "."
              << (driver_version % 1000) / 10 << std::endl;
    cudaRuntimeGetVersion(&runtime_version);
    std::cout << "CUDA运行时版本:" << runtime_version / 1000 << "."
              << (runtime_version % 1000) / 10 << std::endl;
    std::cout << "设备计算能力:" << deviceProp.major << "." << deviceProp.minor
              << std::endl;
    std::cout << "显卡时钟频率:" << deviceProp.clockRate * 1e-6f << " GHz"
              << std::endl;
    std::cout << "内存时钟频率:" << deviceProp.memoryClockRate * 1e-3f << " MHz"
              << std::endl;
    std::cout << "内存总线带宽:" << deviceProp.memoryBusWidth << " bit"
              << std::endl;
    std::cout << "总显存大小:" << deviceProp.totalGlobalMem / (1024.0 * 1024.0)
              << " MB" << std::endl;
    std::cout << "总常量内存大小:" << deviceProp.totalConstMem / 1024.0 << " KB"
              << std::endl;
    std::cout << "SM数量:" << deviceProp.multiProcessorCount << std::endl;
    std::cout << "每个SM最大线程数:" << deviceProp.maxThreadsPerMultiProcessor
              << std::endl;
    std::cout << "每个线程块(block)共享内存大小:"
              << deviceProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "每个线程块(block)的最大线程数:"
              << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "每个线程块(block)的最大可用寄存器数:"
              << deviceProp.regsPerBlock << std::endl;
    std::cout << "线程束(wrap)尺寸:" << deviceProp.warpSize << std::endl;
    std::cout << "每个线程块(block)各个维度最大尺寸:"
              << deviceProp.maxThreadsDim[0] << " x "
              << deviceProp.maxThreadsDim[1] << " x "
              << deviceProp.maxThreadsDim[2] << std::endl;
    std::cout << "每个线程格(grid)各个维度最大尺寸:"
              << deviceProp.maxGridSize[0] << " x " << deviceProp.maxGridSize[1]
              << " x " << deviceProp.maxGridSize[2] << std::endl;
    std::cout << "最大存储间距:" << deviceProp.memPitch / (1024.0 * 1024.0)
              << " MB" << std::endl;
  }
}

void _C_ProfileCopies(float* h_a,
                      float* h_b,
                      float* d,
                      unsigned int n,
                      char* desc) {
  printf("\n%s transfers\n", desc);

  unsigned int bytes = n * sizeof(float);

  // events for timing
  cudaEvent_t startEvent, stopEvent;

  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);

  cudaEventRecord(startEvent, 0);
  cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice);
  cudaEventRecord(stopEvent, 0);
  cudaEventSynchronize(stopEvent);

  float time;
  cudaEventElapsedTime(&time, startEvent, stopEvent);
  printf("  Host to Device bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

  cudaEventRecord(startEvent, 0);
  cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost);
  cudaEventRecord(stopEvent, 0);
  cudaEventSynchronize(stopEvent);

  cudaEventElapsedTime(&time, startEvent, stopEvent);
  printf("  Device to Host bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

  for (int i = 0; i < n; ++i) {
    if (h_a[i] != h_b[i]) {
      printf("*** %s transfers failed ***", desc);
      break;
    }
  }

  // clean up events
  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);
}

void C_ProfileCopies() {
  unsigned int nElements = 1000 * 1024 * 1024;
  const unsigned int bytes = nElements * sizeof(float);

  // host arrays
  float *h_aPageable, *h_bPageable;
  float *h_aPinned, *h_bPinned;

  // device array
  float* d_a;

  // allocate and initialize
  h_aPageable = (float*)malloc(bytes);        // host pageable
  h_bPageable = (float*)malloc(bytes);        // host pageable
  cudaMallocHost((void**)&h_aPinned, bytes);  // host pinned
  cudaMallocHost((void**)&h_bPinned, bytes);  // host pinned
  cudaMalloc((void**)&d_a, bytes);            // device

  for (int i = 0; i < nElements; ++i)
    h_aPageable[i] = i;
  memcpy(h_aPinned, h_aPageable, bytes);
  memset(h_bPageable, 0, bytes);
  memset(h_bPinned, 0, bytes);

  // output device info and transfer size
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  printf("\nDevice: %s\n", prop.name);
  printf("Transfer size (MB): %d\n", bytes / (1024 * 1024));

  // perform copies and report bandwidth
  _C_ProfileCopies(h_aPageable, h_bPageable, d_a, nElements, "Pageable");
  _C_ProfileCopies(h_aPinned, h_bPinned, d_a, nElements, "Pinned");

  printf("\n");

  // cleanup
  cudaFree(d_a);
  cudaFreeHost(h_aPinned);
  cudaFreeHost(h_bPinned);
  free(h_aPageable);
  free(h_bPageable);
}