// Plugins/CUDAPlugin/Source/GPGPUPlugin/Public/CUDATypes.h
#pragma once

#include "CoreMinimal.h"
#include "cuda.h"           // For CUdeviceptr
#include "cuda_runtime.h"   // For cudaTextureObject_t
#include "CUDATypes.generated.h"

USTRUCT(BlueprintType)
struct FCUDATextureObject
{
    GENERATED_BODY()

    cudaTextureObject_t Value;

    FCUDATextureObject() : Value(0) {}
    FCUDATextureObject(cudaTextureObject_t InValue) : Value(InValue) {}

    operator cudaTextureObject_t() const { return Value; }
    FCUDATextureObject& operator=(cudaTextureObject_t InValue) { Value = InValue; return *this; }
};

USTRUCT(BlueprintType)
struct FCUDADevicePtr
{
    GENERATED_BODY()

    CUdeviceptr Value;

    FCUDADevicePtr() : Value(0) {}
    FCUDADevicePtr(CUdeviceptr InValue) : Value(InValue) {}

    operator CUdeviceptr() const { return Value; }
    FCUDADevicePtr& operator=(CUdeviceptr InValue) { Value = InValue; return *this; }
};