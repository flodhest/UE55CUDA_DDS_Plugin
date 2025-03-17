// Plugins/Lidar/Source/Lidar/Private/CUDAMacros.cpp
#include "CUDAMacros.h"

void CheckCudaErrors(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        UE_LOG(LogTemp, Error, TEXT("CUDA Runtime Error Detected: %s"), *FString(cudaGetErrorString(err)));
        assert(false); // Triggers in debug mode, safe in release
    }
}