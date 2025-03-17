//#pragma once
//#include "CoreMinimal.h" // For UE_LOG and Unreal types
//#include "cuda_runtime.h"
//#include "nvrtc.h"
//#include <stdexcept> // For std::runtime_error
//
//// Enhanced CUDA error checking with logging and exception throwing
//#define CUDA_CHECK(call) \
//    do { \
//        cudaError_t err = call; \
//        if (err != cudaSuccess) { \
//            FString errorMsg = FString::Printf(TEXT("%s failed: %s (error code %d)"), TEXT(#call), cudaGetErrorString(err), err); \
//            UE_LOG(LogTemp, Error, TEXT("%s"), *errorMsg); \
//            throw std::runtime_error(TCHAR_TO_ANSI(*errorMsg)); \
//        } \
//    } while (0)
//
//// Enhanced NVRTC error checking with logging and exception throwing
//#define NVRTC_CHECK(call) \
//    do { \
//        nvrtcResult res = call; \
//        if (res != NVRTC_SUCCESS) { \
//            FString errorMsg = FString::Printf(TEXT("%s failed: %s (error code %d)"), TEXT(#call), nvrtcGetErrorString(res), res); \
//            UE_LOG(LogTemp, Error, TEXT("%s"), *errorMsg); \
//            throw std::runtime_error(TCHAR_TO_ANSI(*errorMsg)); \
//        } \
//    } while (0)
//
//// Macros to push/pop conflicting definitions between CUDA and Unreal Engine
//#define GPU_PUSH_MACROS \
//    _Pragma("push_macro(\"CONSTEXPR\")") \
//    _Pragma("push_macro(\"dynamic_cast\")") \
//    _Pragma("push_macro(\"check\")") \
//    _Pragma("push_macro(\"PI\")") \
//    _Pragma("push_macro(\"DEPRECATED\")") \
//    _Pragma("push_macro(\"VECTOR_TYPES\")") \
//    _Pragma("undef CONSTEXPR") \
//    _Pragma("undef dynamic_cast") \
//    _Pragma("undef check") \
//    _Pragma("undef PI") \
//    _Pragma("undef DEPRECATED") \
//    _Pragma("undef VECTOR_TYPES")
//
//#define GPU_POP_MACROS \
//    _Pragma("pop_macro(\"VECTOR_TYPES\")") \
//    _Pragma("pop_macro(\"DEPRECATED\")") \
//    _Pragma("pop_macro(\"PI\")") \
//    _Pragma("pop_macro(\"check\")") \
//    _Pragma("pop_macro(\"dynamic_cast\")") \
//    _Pragma("pop_macro(\"CONSTEXPR\")")
//
//// Helper macro usage example (not needed in final code, just for reference)
//// #define PRAGMA_MACRO(action, macro) _Pragma(#action "(" macro ")")