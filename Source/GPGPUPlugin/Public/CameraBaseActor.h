// Plugins/CUDAPlugin/Source/GPGPUPlugin/Public/CameraBaseActor.h
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "GenericPlatform/GenericPlatformTime.h"
#include "CameraThreadProcess.h"
#include "Camera/CameraComponent.h"
#include "Components/SceneCaptureComponent2D.h"
#include "CUDATypes.h"
#include "cuda.h"
#include "Engine/TextureRenderTarget2D.h"
#include "CameraBaseActor.generated.h"

#define DDS_ENTITY_INVALID 0

USTRUCT()
struct FStreamCaptureData
{
    GENERATED_BODY()

    CUgraph Graph = nullptr;
    CUgraphExec GraphExec = nullptr;
    bool bIsCaptureActive = false;
};



UCLASS()
class GPGPUPLUGIN_API ACameraBaseActor : public AActor
{
    GENERATED_BODY()

public:
    ACameraBaseActor();

    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type Reason) override;

    virtual void CameraThreadTick() {}

    // Thread management
    class CameraThreadProcess* FrontCameraThread = nullptr;
    class CameraThreadProcess* RearCameraThread = nullptr;

protected:
    // If CameraActor exists, declare it here (assumed from the exception)
    // UPROPERTY()
    // AActor* CameraActor = nullptr; // Uncomment and adjust if this is the intended member
};
