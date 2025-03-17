#include "GPGPUComponent.h"
#include "CUDAMacros.h"
#include <cuda.h> // CUDA Driver API
#include <nvrtc.h>

UGPGPUComponent::UGPGPUComponent()
{
    PrimaryComponentTick.bCanEverTick = false; // No ticking required
    CUDAStream = nullptr;
    CachedKernel = nullptr;
    CachedModule = nullptr;
    CUDAContext = nullptr;
    KernelContainer = nullptr;
    bIsValid = false;
    LastGraphLaunchLogTime = 0.0;
}

void UGPGPUComponent::CleanupKernel()
{
    FScopeLock Lock(&CUDAMutex); // Ensure thread safety

    if (CachedKernel)
    {
        CachedKernel = nullptr;
    }

    if (CachedModule)
    {
        cuModuleUnload(CachedModule);
        CachedModule = nullptr;
    }

    CachedPTXPath.Empty();
    CachedKernelName.Empty();
}

void UGPGPUComponent::UpdateKernel(const FString& PTXPath, const FString& KernelName)
{
    if (!IsValid())
    {
        return;
    }

    FScopeLock Lock(&CUDAMutex); // Ensure thread safety

    CleanupKernel();

    if (!PTXPath.IsEmpty() && !KernelName.IsEmpty())
    {
        CUresult err = cuModuleLoad(&CachedModule, TCHAR_TO_ANSI(*PTXPath));
        if (err != CUDA_SUCCESS)
        {
            return;
        }

        err = cuModuleGetFunction(&CachedKernel, CachedModule, TCHAR_TO_ANSI(*KernelName));
        if (err != CUDA_SUCCESS)
        {
            cuModuleUnload(CachedModule);
            CachedModule = nullptr;
            CachedKernel = nullptr;
            return;
        }

        CachedPTXPath = PTXPath;
        CachedKernelName = KernelName;
    }
    else if (!KernelContainer || KernelContainer->KernelCode.IsEmpty() || KernelContainer->KernelEntryPoint.IsEmpty())
    {
        return;
    }
    else
    {
        CompileAndLoadKernel(KernelContainer->KernelCode.ToString(), KernelContainer->KernelEntryPoint);
    }
}

void UGPGPUComponent::OnRegister()
{
    Super::OnRegister();
    InitializeCUDA();
}

void UGPGPUComponent::BeginDestroy()
{
    CleanupCUDA();
    Super::BeginDestroy();
}

void UGPGPUComponent::InitializeComponent()
{
    Super::InitializeComponent();
    if (!bIsValid)
    {
        InitializeCUDA();
    }
}

void UGPGPUComponent::UninitializeComponent()
{
    CleanupCUDA();
    Super::UninitializeComponent();
}

void UGPGPUComponent::InitializeCUDA()
{
    CUresult err;

    err = cuInit(0);
    if (err != CUDA_SUCCESS)
    {
        return;
    }

    CUdevice device;
    err = cuDeviceGet(&device, 0);
    if (err != CUDA_SUCCESS)
    {
        return;
    }

    err = cuCtxCreate(&CUDAContext, 0, device);
    if (err != CUDA_SUCCESS)
    {
        return;
    }

    err = cuStreamCreate(&CUDAStream, CU_STREAM_NON_BLOCKING);
    if (err != CUDA_SUCCESS)
    {
        cuCtxDestroy(CUDAContext);
        CUDAContext = nullptr;
        return;
    }

    bIsValid = true;
}

void UGPGPUComponent::CleanupCUDA()
{
    if (!bIsValid)
    {
        return;
    }

    CUresult err;

    if (CUDAStream)
    {
        err = cuStreamSynchronize(CUDAStream);
        err = cuStreamDestroy(CUDAStream);
        CUDAStream = nullptr;
    }

    if (CachedModule)
    {
        err = cuModuleUnload(CachedModule);
        CachedModule = nullptr;
        CachedKernel = nullptr;
    }

    if (CUDAContext)
    {
        err = cuCtxDestroy(CUDAContext);
        CUDAContext = nullptr;
    }

    bIsValid = false;
}

bool UGPGPUComponent::LoadPTXModule(const FString& PTXPath, const FString& KernelName)
{
    if (!IsValid())
    {
        return false;
    }

    CUresult err = cuModuleLoad(&CachedModule, TCHAR_TO_ANSI(*PTXPath));
    if (err != CUDA_SUCCESS)
    {
        return false;
    }

    err = cuModuleGetFunction(&CachedKernel, CachedModule, TCHAR_TO_ANSI(*KernelName));
    if (err != CUDA_SUCCESS)
    {
        cuModuleUnload(CachedModule);
        CachedModule = nullptr;
        CachedKernel = nullptr;
        return false;
    }

    CachedPTXPath = PTXPath;
    return true;
}

bool UGPGPUComponent::IsValid() const
{
    return bIsValid && CUDAStream != nullptr && CUDAContext != nullptr;
}

void UGPGPUComponent::SetKernelContainer(UKernelContainer* InKernelContainer)
{
    if (!InKernelContainer)
    {
        return;
    }

    KernelContainer = InKernelContainer;
    CompileAndLoadKernel(KernelContainer->KernelCode.ToString(), KernelContainer->KernelEntryPoint);
}

bool UGPGPUComponent::CompileAndLoadKernel(const FString& SourceCode, const FString& KernelName)
{
    if (!IsValid())
    {
        return false;
    }

    nvrtcProgram prog;
    std::string cudaCode = TCHAR_TO_UTF8(*SourceCode);
    nvrtcResult nvrtcErr = nvrtcCreateProgram(&prog, cudaCode.c_str(), "rear_camera_kernel.cu", 0, nullptr, nullptr);
    if (nvrtcErr != NVRTC_SUCCESS)
    {
        nvrtcDestroyProgram(&prog);
        return false;
    }

    CUdevice device;
    cuCtxGetDevice(&device);
    int major, minor;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    FString archOption = FString::Printf(TEXT("-arch=sm_%d%d"), major, minor);

    const char* opts[] = { TCHAR_TO_UTF8(*archOption), "--use_fast_math", "--fmad=true", "--std=c++17" };
    nvrtcErr = nvrtcCompileProgram(prog, UE_ARRAY_COUNT(opts), opts);
    if (nvrtcErr != NVRTC_SUCCESS)
    {
        nvrtcDestroyProgram(&prog);
        return false;
    }

    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    TArray<char> ptx;
    ptx.SetNumUninitialized(ptxSize);
    nvrtcGetPTX(prog, ptx.GetData());

    CUresult err = cuModuleLoadData(&CachedModule, ptx.GetData());
    if (err != CUDA_SUCCESS)
    {
        nvrtcDestroyProgram(&prog);
        return false;
    }

    err = cuModuleGetFunction(&CachedKernel, CachedModule, TCHAR_TO_UTF8(*KernelName));
    if (err != CUDA_SUCCESS)
    {
        cuModuleUnload(CachedModule);
        CachedModule = nullptr;
        CachedKernel = nullptr;
        nvrtcDestroyProgram(&prog);
        return false;
    }

    nvrtcDestroyProgram(&prog);
    CachedPTXPath = TEXT("CompiledInMemory");
    return true;
}

bool UGPGPUComponent::BeginStreamCapture(CUstream Stream)
{
    if (!IsValid() || !Stream)
    {
        return false;
    }

    CUresult err = cuStreamBeginCapture(Stream, CU_STREAM_CAPTURE_MODE_GLOBAL);
    if (err != CUDA_SUCCESS)
    {
        return false;
    }

    return true;
}

bool UGPGPUComponent::EndStreamCapture(CUstream Stream, CUgraph* Graph)
{
    if (!IsValid() || !Stream || !Graph)
    {
        return false;
    }

    CUresult err = cuStreamEndCapture(Stream, Graph);
    if (err != CUDA_SUCCESS)
    {
        return false;
    }

    return true;
}

bool UGPGPUComponent::InstantiateGraph(CUgraph Graph, CUgraphExec* GraphExec)
{
    if (!IsValid() || !Graph || !GraphExec)
    {
        return false;
    }

    CUresult err = cuGraphInstantiate(GraphExec, Graph, 0);
    if (err != CUDA_SUCCESS)
    {
        return false;
    }

    return true;
}

void UGPGPUComponent::LaunchGraph(CUgraphExec GraphExec, CUstream Stream)
{
    if (!IsValid() || !GraphExec || !Stream)
    {
        return;
    }

    CUresult err = cuGraphLaunch(GraphExec, Stream);
    if (err == CUDA_SUCCESS)
    {
        LastGraphLaunchLogTime = FPlatformTime::Seconds();
    }
}

void UGPGPUComponent::DestroyGraphExec(CUgraphExec GraphExec)
{
    if (!GraphExec)
    {
        return;
    }

    cuGraphExecDestroy(GraphExec);
}

void UGPGPUComponent::DestroyGraph(CUgraph Graph)
{
    if (!Graph)
    {
        return;
    }

    cuGraphDestroy(Graph);
}