#include "CameraBaseActor.h"

ACameraBaseActor::ACameraBaseActor()
{
    PrimaryActorTick.bCanEverTick = true;
}

void ACameraBaseActor::BeginPlay()
{
    Super::BeginPlay();
    // Initialize camera threads
    if (FrontCameraThread)
    {
        FrontCameraThread->CameraThreadInit();
    }
    if (RearCameraThread)
    {
        RearCameraThread->CameraThreadInit();
    }
}

void ACameraBaseActor::EndPlay(EEndPlayReason::Type Reason)
{
    if (FrontCameraThread)
    {
        FrontCameraThread->CameraThreadShutdown();
        delete FrontCameraThread;
        FrontCameraThread = nullptr;
    }
    if (RearCameraThread)
    {
        RearCameraThread->CameraThreadShutdown();
        delete RearCameraThread;
        RearCameraThread = nullptr;
    }
    Super::EndPlay(Reason);
}
