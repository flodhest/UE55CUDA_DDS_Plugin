#include "Components/dds_front_camera_publisher.h"
#include "CUDAMacros.h"
#include <cuda.h>
bool FFrontBuffer::Init(int32 Size)
{
    BufferA.SetNumUninitialized(Size);
    BufferB.SetNumUninitialized(Size);
    WriteBuffer = BufferA.GetData();
    ReadBuffer = BufferB.GetData();
    bIsBufferAWrite = true;
    return BufferA.Num() == Size && BufferB.Num() == Size;
}

uint8* FFrontBuffer::GetWriteBuffer()
{
    return WriteBuffer.load(std::memory_order_acquire);  
}

const uint8* FFrontBuffer::GetReadBuffer()
{
    return ReadBuffer.load(std::memory_order_acquire);  
}

void FFrontBuffer::SwapBuffers()
{
    bool CurrentIsAWrite = bIsBufferAWrite.load(std::memory_order_acquire);
    bIsBufferAWrite.store(!CurrentIsAWrite, std::memory_order_release);
    WriteBuffer.store(CurrentIsAWrite ? BufferB.GetData() : BufferA.GetData(), std::memory_order_release);
    ReadBuffer.store(CurrentIsAWrite ? BufferA.GetData() : BufferB.GetData(), std::memory_order_release);
}

void FFrontBuffer::Cleanup()
{
    BufferA.Empty();
    BufferB.Empty();
    WriteBuffer.store(nullptr, std::memory_order_release);
    ReadBuffer.store(nullptr, std::memory_order_release);
    bIsBufferAWrite.store(false, std::memory_order_release);
}

AFront_camera_publisher::AFront_camera_publisher()
    : ResolutionX(960), ResolutionY(540), FieldOfView(120.0f), LastPublishTime(FDateTime::UtcNow()),
    d_inputPersistent(0), d_outputPersistent(0), InputSizePersistent(0), OutputSizePersistent(0),
    FrontCameraThread(nullptr), CUDAStreamFront(nullptr), ParticipantFront(DDS_ENTITY_INVALID),
    TopicFrontImage(DDS_ENTITY_INVALID), WriterFrontImage(DDS_ENTITY_INVALID),
    LoanedSample(nullptr)
{
    PrimaryActorTick.bCanEverTick = true;
    GPGPUComponent = CreateDefaultSubobject<UGPGPUComponent>(TEXT("GPGPUComponent"));
    FrontCaptureComponent = CreateDefaultSubobject<USceneCaptureComponent2D>(TEXT("FrontCaptureComponent"));
    RenderTargetA = CreateDefaultSubobject<UTextureRenderTarget2D>(TEXT("RenderTargetA"));
    RenderTargetB = CreateDefaultSubobject<UTextureRenderTarget2D>(TEXT("RenderTargetB"));
    ActiveRenderTarget = RenderTargetA;
    ConfigReader2 = CreateDefaultSubobject<UEsimConfigReaderComponent>(TEXT("ConfigReader2"));
    KernelContainerAsset = NewObject<UKernelContainer>(this, TEXT("DefaultKernelContainer"));
    SurfaceData.SetNumUninitialized(ResolutionX * ResolutionY);
}

void AFront_camera_publisher::BeginPlay()
{
    Super::BeginPlay();

    if (!InitializeDDS("FrontCamera")) { CleanupOnFailure(); return; }

    if (!GPGPUComponent) { CleanupOnFailure(); return; }

    GPGPUComponent->InitializeComponent();
    if (!GPGPUComponent->IsValid()) { CleanupOnFailure(); return; }

    CUDA_DRIVER_CHECK(cuStreamCreate(&CUDAStreamFront, CU_STREAM_NON_BLOCKING));
    CUDA_DRIVER_CHECK(cuEventCreate(&CaptureEvent, CU_EVENT_DEFAULT));
    InputSizePersistent = ResolutionX * ResolutionY * 4;
    OutputSizePersistent = ResolutionX * ResolutionY * 3;
    LastFrameData.SetNumUninitialized(OutputSizePersistent);
    CUDA_DRIVER_CHECK(cuMemAlloc(&d_inputPersistent, InputSizePersistent));
    CUDA_DRIVER_CHECK(cuMemAlloc(&d_outputPersistent, OutputSizePersistent));
    FString PTXPath = FPaths::Combine(FPaths::ProjectPluginsDir(), TEXT("CUDAPlugin/Source/GPGPUPlugin/Public/matSumKernel.ptx"));
    if (FPaths::FileExists(PTXPath)) {
        GPGPUComponent->LoadPTXModule(PTXPath, TEXT("RgbaToRgbKernel"));
    }
    else {
        GPGPUComponent->SetKernelContainer(KernelContainerAsset);
    }

    if (!StartCapturing() || !SetupStreamCapture()) { CleanupOnFailure(); return; }

    FString ThreadName = "FrontCameraThread";
    FrontCameraThread = new CameraThreadProcess(FTimespan::Zero(), *ThreadName, this);
    if (FrontCameraThread && FrontCameraThread->Init()) {
        FrontCameraThread->CameraThreadInit();
    }
    else {
        delete FrontCameraThread;
        FrontCameraThread = nullptr;
        CleanupOnFailure();
    }
}

void AFront_camera_publisher::Tick(float DeltaTime)
{
}

bool AFront_camera_publisher::SetupStreamCapture()
{
    if (!CUDAStreamFront || !GPGPUComponent->BeginStreamCapture(CUDAStreamFront)) { return false; }
    StreamCapture.bIsCaptureActive = true;

    int blockSize = 512;
    int gridSize = (ResolutionX * ResolutionY + blockSize - 1) / blockSize;
    void* kernelArgs[] = { &d_inputPersistent, &d_outputPersistent, &ResolutionX, &ResolutionY };
    CUDA_DRIVER_CHECK(cuLaunchKernel(GPGPUComponent->CachedKernel, gridSize, 1, 1, blockSize, 1, 1, 0, CUDAStreamFront, kernelArgs, nullptr));
    CUDA_DRIVER_CHECK(cuMemcpyDtoHAsync(LastFrameData.GetData(), d_outputPersistent, OutputSizePersistent, CUDAStreamFront));
    CUDA_DRIVER_CHECK(cuEventRecord(CaptureEvent, CUDAStreamFront));  

    if (!GPGPUComponent->EndStreamCapture(CUDAStreamFront, &StreamCapture.Graph) ||
        !GPGPUComponent->InstantiateGraph(StreamCapture.Graph, &StreamCapture.GraphExec)) {
        GPGPUComponent->DestroyGraph(StreamCapture.Graph);
        StreamCapture.Graph = nullptr;
        return false;
    }
    CUresult syncResult = cuEventSynchronize(CaptureEvent);  
    CUDA_DRIVER_CHECK(syncResult);
    StreamCapture.bIsCaptureActive = false;
    return StreamCapture.GraphExec != nullptr;
}
void AFront_camera_publisher::CameraThreadTick()
{
    if (!StreamCapture.GraphExec) return;

    CUDA_DRIVER_CHECK(cuCtxPushCurrent(GPGPUComponent->CUDAContext));

    ReadEntireRenderTargetInternal();
    GPGPUComponent->LaunchGraph(StreamCapture.GraphExec, CUDAStreamFront);
    CUDA_DRIVER_CHECK(cuEventRecord(CaptureEvent, CUDAStreamFront));  
    CUresult syncResult = cuEventSynchronize(CaptureEvent);  
    if (syncResult == CUDA_SUCCESS)  
    {
        uint8* WriteBuffer = FrontBuffer.GetWriteBuffer();
        if (WriteBuffer && LastFrameData.Num() == OutputSizePersistent)
        {
            FMemory::Memcpy(WriteBuffer, LastFrameData.GetData(), LastFrameData.Num());
            FrontBuffer.SwapBuffers();
            const uint8* ReadBuffer = FrontBuffer.GetReadBuffer();
            if (ReadBuffer)
            {
                PublishImageMessage(ReadBuffer, LastFrameData.Num());
            }
        }
    }

    CUDA_DRIVER_CHECK(cuCtxPopCurrent(nullptr));
}
void AFront_camera_publisher::ReadEntireRenderTargetInternal()
{
    UTextureRenderTarget2D* ReadTarget;
    {
        FScopeLock Lock(&RenderTargetMutex);
        ReadTarget = ActiveRenderTarget;
        ActiveRenderTarget = (ActiveRenderTarget == RenderTargetA) ? RenderTargetB : RenderTargetA;
        FrontCaptureComponent->TextureTarget = ActiveRenderTarget;
    }

    AsyncTask(ENamedThreads::GameThread, [this, ReadTarget]()
        {
            if (!ReadTarget) return;

            ENQUEUE_RENDER_COMMAND(ProcessRenderTarget)(
                [this, Resource = ReadTarget->GameThread_GetRenderTargetResource()](FRHICommandListImmediate& RHICmdList)
                {
                    if (!Resource) return;

                    FIntPoint Dimensions = Resource->GetSizeXY();
                    int32 Width = Dimensions.X;
                    int32 Height = Dimensions.Y;

                    SurfaceData.SetNumUninitialized(Width * Height, false);
                    RHICmdList.ReadSurfaceData(
                        Resource->GetTextureRHI(),
                        FIntRect(0, 0, Width, Height),
                        SurfaceData,
                        FReadSurfaceDataFlags()
                    );

                    CUDA_DRIVER_CHECK(cuMemcpyHtoDAsync(d_inputPersistent, SurfaceData.GetData(), InputSizePersistent, CUDAStreamFront));
                });
        });
}

bool AFront_camera_publisher::StartCapturing()
{
    if (!FrontCaptureComponent || !RenderTargetA || !RenderTargetB) return false;

    for (UTextureRenderTarget2D* RT : { RenderTargetA, RenderTargetB })
    {
        RT->InitCustomFormat(ResolutionX, ResolutionY, PF_R8G8B8A8, false);
        RT->bAutoGenerateMips = false;
        RT->ClearColor = FLinearColor::Black;
        RT->bGPUSharedFlag = true;
        RT->UpdateResourceImmediate(true);
    }

    {
        FScopeLock Lock(&RenderTargetMutex);
        ActiveRenderTarget = RenderTargetA;
        FrontCaptureComponent->TextureTarget = ActiveRenderTarget;
    }

    FrontCaptureComponent->bUseRayTracingIfEnabled = false;
    FrontCaptureComponent->ShowFlags.SetPostProcessing(true);
    FrontCaptureComponent->bCaptureEveryFrame = true;
    FrontCaptureComponent->CaptureSource = ESceneCaptureSource::SCS_FinalToneCurveHDR;
    FrontCaptureComponent->bAlwaysPersistRenderingState = true;
    FrontCaptureComponent->FOVAngle = FieldOfView;

    const int32 ColorBufferSize = ResolutionX * ResolutionY * 3;
    bFrontBufferInitialized = FrontBuffer.Init(ColorBufferSize);

    return bFrontBufferInitialized;
}

void AFront_camera_publisher::PublishImageMessage(const uint8* Data, size_t Size)
{
    sensor_msgs_msg_dds__Image_* image_msg = nullptr;
    dds_return_t loanStatus = dds_request_loan(WriterFrontImage, reinterpret_cast<void**>(&image_msg));
    if (loanStatus != DDS_RETCODE_OK || !image_msg) return;
    LoanedSample = image_msg;
    image_msg->header.stamp.sec = 0;
    image_msg->header.stamp.nanosec = 0;
    image_msg->header.frame_id = nullptr;
    image_msg->height = 0;
    image_msg->width = 0;
    image_msg->encoding = nullptr;
    image_msg->is_bigendian = 0;
    image_msg->step = 0;
    image_msg->data._maximum = 0;
    image_msg->data._length = 0;
    image_msg->data._buffer = nullptr;
    image_msg->data._release = false;
    SerializeImageMessage(image_msg, Data, Size);
    dds_return_t writeStatus = dds_write(WriterFrontImage, image_msg);
    if (writeStatus == DDS_RETCODE_OK)
    {
        LoanedSample = nullptr;
    }
    else
    {

        dds_return_loan(WriterFrontImage, reinterpret_cast<void**>(&image_msg), 1);

    }
}


void AFront_camera_publisher::SerializeImageMessage(sensor_msgs_msg_dds__Image_* Msg, const uint8* Data, size_t Size)
{
    double CurrentTimeSeconds = FPlatformTime::Seconds();
    Msg->header.stamp.sec = static_cast<int32_t>(CurrentTimeSeconds);
    Msg->header.stamp.nanosec = static_cast<uint32_t>((CurrentTimeSeconds - Msg->header.stamp.sec) * 1e9);
    Msg->header.frame_id = dds_string_dup("front_camera_frame");
    Msg->height = ResolutionY;
    Msg->width = ResolutionX;
    Msg->encoding = dds_string_dup("bgr8");
    Msg->is_bigendian = 0;
    Msg->step = ResolutionX * 3;

    if (Msg->data._buffer && Msg->data._release)
    {
        dds_free(Msg->data._buffer);
    }
    if (Msg->data._maximum < Size)
    {
        Msg->data._buffer = static_cast<uint8_t*>(dds_alloc(Size));
        Msg->data._maximum = static_cast<uint32_t>(Size);
        Msg->data._release = true;
    }
    Msg->data._length = static_cast<uint32_t>(Size);

    const uint8* src = Data;
    uint8* dst = Msg->data._buffer;
    size_t i = 0;

    for (; i + 31 < Size; i += 32)
    {
        __m256i data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), data);
    }

    for (; i < Size; ++i)
    {
        dst[i] = src[i];
    }
}
bool AFront_camera_publisher::InitializeDDS(const FString& CameraType)
{
    FString AbsolutePath = FPaths::ConvertRelativePathToFull(FPaths::Combine(FPaths::ProjectPluginsDir(), TEXT("CycloneDDS/Source/CycloneDDS/Public/Cyclonedds.xml")));
    FPlatformMisc::SetEnvironmentVar(TEXT("CYCLONEDDS_URI"), *FString::Printf(TEXT("file://%s"), *AbsolutePath));

    ParticipantFront = dds_create_participant(DDS_DOMAIN_DEFAULT, nullptr, nullptr);
    if (ParticipantFront < 0) return false;

    dds_qos_t* qos = dds_create_qos();
    TopicFrontImage = dds_create_topic(ParticipantFront, &sensor_msgs_msg_dds__Image__desc, "rt/FrontCameraImageTopic", qos, nullptr);
    WriterFrontImage = dds_create_writer(ParticipantFront, TopicFrontImage, qos, nullptr);
    dds_delete_qos(qos);
    return WriterFrontImage > 0;
}

void AFront_camera_publisher::CleanupDDS()
{
    if (LoanedSample)
    {
        dds_return_loan(WriterFrontImage, reinterpret_cast<void**>(&LoanedSample), 1);
        LoanedSample = nullptr;
    }

    if (WriterFrontImage != DDS_ENTITY_INVALID)
    {
        dds_delete(WriterFrontImage);
        WriterFrontImage = DDS_ENTITY_INVALID;
    }
    if (TopicFrontImage != DDS_ENTITY_INVALID)
    {
        dds_delete(TopicFrontImage);
        TopicFrontImage = DDS_ENTITY_INVALID;
    }
    if (ParticipantFront != DDS_ENTITY_INVALID)
    {
        dds_delete(ParticipantFront);
        ParticipantFront = DDS_ENTITY_INVALID;
    }
}
void AFront_camera_publisher::CleanupOnFailure()
{
    if (FrontCameraThread)
    {
        FrontCameraThread->Stop();
        while (!FrontCameraThread->ThreadHasStopped()) FPlatformProcess::Sleep(0.01f);
        delete FrontCameraThread;
        FrontCameraThread = nullptr;
    }
    if (StreamCapture.GraphExec) GPGPUComponent->DestroyGraphExec(StreamCapture.GraphExec);
    if (StreamCapture.Graph) GPGPUComponent->DestroyGraph(StreamCapture.Graph);
    if (CaptureEvent) CUDA_DRIVER_CHECK(cuEventDestroy(CaptureEvent));
    if (d_inputPersistent) CUDA_DRIVER_CHECK(cuMemFree(d_inputPersistent));
    if (d_outputPersistent) CUDA_DRIVER_CHECK(cuMemFree(d_outputPersistent));
    if (CUDAStreamFront) CUDA_DRIVER_CHECK(cuStreamDestroy(CUDAStreamFront));
    if (GPGPUComponent) GPGPUComponent->CleanupCUDA();
    FrontBuffer.Cleanup();
    CleanupDDS();
    CleanupRenderTarget();
}

void AFront_camera_publisher::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    Super::EndPlay(EndPlayReason);
    CleanupOnFailure();
}

void AFront_camera_publisher::CleanupRenderTarget()
{
    FScopeLock Lock(&RenderTargetMutex);
    if (RenderTargetA)
    {
        RenderTargetA->ReleaseResource();
        RenderTargetA = nullptr;
    }
    if (RenderTargetB)
    {
        RenderTargetB->ReleaseResource();
        RenderTargetB = nullptr;
    }
    ActiveRenderTarget = nullptr;
}

void AFront_camera_publisher::CleanupCameraThread()
{
    if (FrontCameraThread)
    {
        FrontCameraThread->Stop();
        while (!FrontCameraThread->ThreadHasStopped())
        {
            FPlatformProcess::Sleep(0.01f);
        }
        delete FrontCameraThread;
        FrontCameraThread = nullptr;
    }
}