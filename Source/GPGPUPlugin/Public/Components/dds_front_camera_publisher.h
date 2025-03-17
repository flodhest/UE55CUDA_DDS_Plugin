#pragma once

#include "CoreMinimal.h"
#include "CUDAMacros.h"
#include "dds_macros.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Async/Async.h"
#include "CUDATypes.h"
#include "Async/TaskGraphInterfaces.h"
#include "Templates/SharedPointer.h"
#include "RenderCommandFence.h"
#include "RHICommandList.h"
#include "RHIDefinitions.h"
#include "Misc/Paths.h"
#include "Misc/ScopeLock.h"
#include <RHIGPUReadback.h>
#include "dds/dds.h"
#include <cuda_d3d11_interop.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "CameraBaseActor.h"
#include "EsimConfigReaderComponent.h"
#include "GPGPUComponent.h"
#include "KernelContainer.h"

#include "KernelArguments.h"
#include <atomic>
#include <mutex>
#include <d3d11.h>
#include <condition_variable>
#include "Components/IDL/Image.h"
#include "dds_front_camera_publisher.generated.h"
struct FFrontBuffer {
    TArray<uint8> BufferA;
    TArray<uint8> BufferB;
    std::atomic<uint8*> WriteBuffer;  
    std::atomic<uint8*> ReadBuffer;  
    std::atomic<bool> bIsBufferAWrite;

    bool Init(int32 Size);
    uint8* GetWriteBuffer();          
    const uint8* GetReadBuffer();   
    void SwapBuffers();               
    void Cleanup();

};
UCLASS()
class AFront_camera_publisher : public ACameraBaseActor {
    GENERATED_BODY()

public:
    AFront_camera_publisher();
    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
    virtual void Tick(float DeltaTime) override;
    TArray<FColor> SurfaceData;


    UPROPERTY() USceneCaptureComponent2D* FrontCaptureComponent;
    UPROPERTY() UTextureRenderTarget2D* RenderTargetA;
    UPROPERTY() UTextureRenderTarget2D* RenderTargetB;
    UPROPERTY() UTextureRenderTarget2D* ActiveRenderTarget;
    FCriticalSection RenderTargetMutex;

    UPROPERTY() class UGPGPUComponent* GPGPUComponent;
    UPROPERTY() class UKernelContainer* KernelContainerAsset;
    UPROPERTY() class UEsimConfigReaderComponent* ConfigReader2;

    int32 ResolutionX, ResolutionY;
    float FieldOfView;
    FDateTime LastPublishTime;

    cudaStream_t CUDAStreamFront;  
    CUevent CaptureEvent;
    CUdeviceptr d_inputPersistent, d_outputPersistent;
    size_t InputSizePersistent, OutputSizePersistent;
    TArray<uint8> LastFrameData;
    FFrontBuffer FrontBuffer;
    bool bFrontBufferInitialized;
    sensor_msgs_msg_dds__Image_* LoanedSample;
    dds_entity_t ParticipantFront;
    dds_entity_t TopicFrontImage;
    dds_entity_t WriterFrontImage;

    struct FStreamCapture {
        cudaGraph_t Graph;
        cudaGraphExec_t GraphExec;
        bool bIsCaptureActive;
    } StreamCapture;

    class CameraThreadProcess* FrontCameraThread;

    bool InitializeDDS(const FString& CameraType);
    void CleanupDDS();
    bool StartCapturing();
    bool SetupStreamCapture();
    void ReadEntireRenderTargetInternal();
    void CameraThreadTick();
    void PublishImageMessage(const uint8* Data, size_t Size);
    void SerializeImageMessage(sensor_msgs_msg_dds__Image_* Msg, const uint8* Data, size_t Size);
    void CleanupOnFailure();
    void CleanupRenderTarget();
    void CleanupCameraThread();
};