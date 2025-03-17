#pragma once

#include "CoreMinimal.h"
#include "CameraThreadBase.h"

class ACameraBaseActor;  

class GPGPUPLUGIN_API CameraThreadProcess : public CameraThreadBase
{
public:
    typedef CameraThreadBase Super;

    CameraThreadProcess(const FTimespan& InThreadSleepTime, const TCHAR* InThreadName, ACameraBaseActor* InCameraActor)
        : Super(InThreadSleepTime, InThreadName), CameraActor(InCameraActor)
    {
        if (!CameraActor)
        {
            UE_LOG(LogTemp, Error, TEXT("CameraActor pointer is null in CameraThreadProcess constructor."));
        }
    }

    virtual void Process() override;
    bool CameraThreadInit();
    void CameraThreadShutdown();

protected:
    ACameraBaseActor* CameraActor = nullptr;

    // Optional: Use a weak pointer if you want to ensure the actor is valid before use
    // TWeakObjectPtr<ACameraBaseActor> CameraActorWeakPtr;

    // Thread synchronization mechanism (optional, depending on your threading logic)
    // FCriticalSection ThreadCriticalSection;
};
