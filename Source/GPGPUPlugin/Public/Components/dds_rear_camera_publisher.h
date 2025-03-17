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
#include "dds_rear_camera_publisher.generated.h"
struct FRearBuffer {
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
class ARear_camera_publisher : public ACameraBaseActor {
    GENERATED_BODY()

public:
    ARear_camera_publisher();
    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
    virtual void Tick(float DeltaTime) override;
    TArray<FColor> SurfaceData;
    UPROPERTY() USceneCaptureComponent2D* RearCaptureComponent;
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

    cudaStream_t CUDAStreamRear;  
    CUevent CaptureEvent;
    CUdeviceptr d_inputPersistent, d_outputPersistent;
    size_t InputSizePersistent, OutputSizePersistent;
    TArray<uint8> LastFrameData;
    FRearBuffer RearBuffer;
    bool bRearBufferInitialized;
    sensor_msgs_msg_dds__Image_* LoanedSample;
    dds_entity_t ParticipantRear;
    dds_entity_t TopicRearImage;
    dds_entity_t WriterRearImage;

    struct FStreamCapture {
        cudaGraph_t Graph;
        cudaGraphExec_t GraphExec;
        bool bIsCaptureActive;
    } StreamCapture;

    class CameraThreadProcess* RearCameraThread;

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