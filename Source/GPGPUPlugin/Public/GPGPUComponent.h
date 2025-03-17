#pragma once
#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "CUDAMacros.h"
#include <cuda.h>           // CUDA Runtime API
#include <cuda_runtime.h>   // Additional runtime functions
#include <nvrtc.h>          // For runtime compilation
#include "KernelContainer.h"
#include "GPGPUComponent.generated.h"

UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class GPGPUPLUGIN_API UGPGPUComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UGPGPUComponent();
    virtual void OnRegister() override;
    virtual void BeginDestroy() override;
    virtual void InitializeComponent() override;
    virtual void UninitializeComponent() override;

    bool IsValid() const;
    void SetKernelContainer(UKernelContainer* InKernelContainer);
    bool CompileAndLoadKernel(const FString& SourceCode, const FString& KernelName);

    bool BeginStreamCapture(CUstream Stream);
    bool EndStreamCapture(CUstream Stream, CUgraph* Graph);
    bool InstantiateGraph(CUgraph Graph, CUgraphExec* GraphExec);
    void LaunchGraph(CUgraphExec GraphExec, CUstream Stream);
    void DestroyGraphExec(CUgraphExec GraphExec);
    void DestroyGraph(CUgraph Graph);

    int CUDADevice = 0;           // Runtime API uses int for device
    cudaStream_t CUDAStream = nullptr; // Note: this is cudaStream_t, not CUstream

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Kernel Functions")
    UKernelContainer* KernelContainer;

    void InitializeCUDA();
    void CleanupCUDA();
    bool LoadPTXModule(const FString& PTXPath, const FString& KernelName);
    void CleanupKernel();

    void UpdateKernel(const FString& PTXPath, const FString& KernelName);

    FCriticalSection CUDAMutex;
    FString CachedKernelName;     // Store the kernel name for launching
    FCriticalSection ExecutionMutex;
    bool bIsValid = false;
    CUfunction CachedKernel = nullptr; // Driver API kernel function handle
    CUmodule CachedModule = nullptr;   // Driver API module handle
    CUcontext CUDAContext = nullptr;   // CUDA context
    FString CachedPTXPath;            // Cached PTX path
    double LastGraphLaunchLogTime = 0.0; // Timing variable
};