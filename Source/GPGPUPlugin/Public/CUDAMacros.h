#pragma once

#include "CoreMinimal.h"
#include "cuda.h"          // CUDA Driver API
#include "cuda_runtime.h"  // CUDA Runtime API
#include "nvrtc.h"         // NVRTC for kernel compilation
#pragma once

// Cleanup a CUDA module safely
void SafeUnloadModule(CUmodule& Module) {
    if (Module) {
        CUresult err = cuModuleUnload(Module);
        if (err != CUDA_SUCCESS) {
            const char* errStr = nullptr;
            cuGetErrorString(err, &errStr);
            UE_LOG(LogTemp, Error, TEXT("cuModuleUnload failed: %s"),
                errStr ? ANSI_TO_TCHAR(errStr) : TEXT("Unknown error"));
        }
        Module = nullptr;
    }
}


// Macro to check CUDA Driver API calls and log errors
#define CUDA_DRIVER_CHECK(call) \
    do { \
        CUresult result = (call); \
        if (result != CUDA_SUCCESS) { \
            const char* errStr = nullptr; \
            cuGetErrorString(result, &errStr); \
            UE_LOG(LogTemp, Error, TEXT("%s:%d CUDA error in %s: %s (%d)"), \
                TEXT(__FILE__), __LINE__, TEXT(#call), \
                errStr ? ANSI_TO_TCHAR(errStr) : TEXT("Unknown error"), result); \
        } \
    } while (0)

// Macro to check CUDA Runtime API calls (if needed in the future)
#define CUDA_RUNTIME_CHECK(call) \
    do { \
        cudaError_t result = (call); \
        if (result != cudaSuccess) { \
            UE_LOG(LogTemp, Error, TEXT("%s:%d CUDA runtime error in %s: %s (%d)"), \
                TEXT(__FILE__), __LINE__, TEXT(#call), \
                ANSI_TO_TCHAR(cudaGetErrorString(result)), result); \
        } \
    } while (0)

// Preserve CUDA-specific macros that might be defined elsewhere
#define GPU_PUSH_MACROS \
    _Pragma("push_macro(\"__CUDA_ARCH__\")") \
    _Pragma("push_macro(\"__CUDA__\")") \
    _Pragma("push_macro(\"__CUDACC__\")") \
    _Pragma("push_macro(\"CUDA_ARCH\")")

#define GPU_POP_MACROS \
    _Pragma("pop_macro(\"__CUDA_ARCH__\")") \
    _Pragma("pop_macro(\"__CUDA__\")") \
    _Pragma("pop_macro(\"__CUDACC__\")") \
    _Pragma("pop_macro(\"CUDA_ARCH\")")

// CUDA Driver API error checking (void context, no return)
#define CU_CHECK(call) \
    do { \
        CUresult res = (call); \
        if (res != CUDA_SUCCESS) { \
            const char* errStr = nullptr; \
            cuGetErrorString(res, &errStr); \
            UE_LOG(LogTemp, Error, TEXT("CUDA Driver Error [%d]: %s in %s at %s:%d"), \
                   static_cast<int32>(res), UTF8_TO_TCHAR(errStr ? errStr : "Unknown Error"), \
                   TEXT(#call), TEXT(__FILE__), __LINE__); \
        } \
    } while (0)

// CUDA Runtime API error checking (void context, no return)
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            UE_LOG(LogTemp, Error, TEXT("CUDA Runtime Error [%d]: %s in %s at %s:%d"), \
                   static_cast<int32>(err), *FString(cudaGetErrorString(err)), \
                   TEXT(#call), TEXT(__FILE__), __LINE__); \
        } \
    } while (0)

// NVRTC error checking (void context, no return)
#define NVRTC_CHECK(call) \
    do { \
        nvrtcResult res = (call); \
        if (res != NVRTC_SUCCESS) { \
            UE_LOG(LogTemp, Error, TEXT("NVRTC Error [%d]: %s in %s at %s:%d"), \
                   static_cast<int32>(res), *FString(nvrtcGetErrorString(res)), \
                   TEXT(#call), TEXT(__FILE__), __LINE__); \
        } \
    } while (0)

// CUDA Driver API safe call with return value (for bool-returning functions)
#define CU_CHECK_RETURN(call) \
    do { \
        CUresult res = (call); \
        if (res != CUDA_SUCCESS) { \
            const char* errStr = nullptr; \
            cuGetErrorString(res, &errStr); \
            UE_LOG(LogTemp, Error, TEXT("CUDA Driver Error [%d]: %s in %s at %s:%d"), \
                   static_cast<int32>(res), UTF8_TO_TCHAR(errStr ? errStr : "Unknown Error"), \
                   TEXT(#call), TEXT(__FILE__), __LINE__); \
            return false; \
        } \
    } while (0)

// CUDA Runtime API safe call with return value (for bool-returning functions)
#define CUDA_CHECK_RETURN(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            UE_LOG(LogTemp, Error, TEXT("CUDA Runtime Error [%d]: %s in %s at %s:%d"), \
                   static_cast<int32>(err), *FString(cudaGetErrorString(err)), \
                   TEXT(#call), TEXT(__FILE__), __LINE__); \
            return false; \
        } \
    } while (0)

// NVRTC safe call with return value (for bool-returning functions)
#define NVRTC_CHECK_RETURN(call) \
    do { \
        nvrtcResult res = (call); \
        if (res != NVRTC_SUCCESS) { \
            UE_LOG(LogTemp, Error, TEXT("NVRTC Error [%d]: %s in %s at %s:%d"), \
                   static_cast<int32>(res), *FString(nvrtcGetErrorString(res)), \
                   TEXT(#call), TEXT(__FILE__), __LINE__); \
            return false; \
        } \
    } while (0)

// Synchronization check with custom message (void context)
#define CUDA_SYNC_CHECK(msg) \
    do { \
        cudaError_t err = cudaDeviceSynchronize(); \
        if (err != cudaSuccess) { \
            UE_LOG(LogTemp, Warning, TEXT("CUDA Sync Error [%d]: %s at %s:%d - %s"), \
                   static_cast<int32>(err), *FString(cudaGetErrorString(err)), \
                   TEXT(__FILE__), __LINE__, UTF8_TO_TCHAR(msg)); \
        } \
    } while (0)

// Combined macro for NVRTC compilation with logging (returns bool)
#define NVRTC_COMPILE(program, options, numOptions) \
    do { \
        nvrtcResult compileResult = nvrtcCompileProgram(program, numOptions, options); \
        size_t logSize; \
        NVRTC_CHECK_RETURN(nvrtcGetProgramLogSize(program, &logSize)); \
        if (logSize > 1) { \
            TArray<char> log; \
            log.SetNum(logSize); \
            NVRTC_CHECK_RETURN(nvrtcGetProgramLog(program, log.GetData())); \
            UE_LOG(LogTemp, Log, TEXT("NVRTC Compilation Log:\n%s"), ANSI_TO_TCHAR(log.GetData())); \
        } \
        if (compileResult != NVRTC_SUCCESS) { \
            UE_LOG(LogTemp, Error, TEXT("NVRTC Compilation Failed [%d]: %s"), \
                   static_cast<int32>(compileResult), *FString(nvrtcGetErrorString(compileResult))); \
            return false; \
        } \
    } while (0)

// Function declaration for custom error handling (if needed)
void HandleError(const FString& ErrorMessage);