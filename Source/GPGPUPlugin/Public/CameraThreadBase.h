#pragma once
#include "CoreMinimal.h"
#include "HAL/Runnable.h"
#include "HAL/RunnableThread.h"
#include "HAL/ThreadSafeBool.h"
#include "Misc/SingleThreadRunnable.h"

class GPGPUPLUGIN_API CameraThreadBase : public FRunnable, FSingleThreadRunnable
{
public:
    /**
    * @brief Create the thread by calling this
    * @param InThreadSleepTime The sleep time in main loop of thread.
    * @param InThreadName The thread description (for debugging).
    **/
    CameraThreadBase(const FTimespan& InThreadSleepTime, const TCHAR* InThreadName)
        : bStopping(false), ThreadSleepTime(InThreadSleepTime)
    {
        Paused.AtomicSet(false);
        HasStopped.AtomicSet(true);

        Thread = FRunnableThread::Create(this, InThreadName, 0U, EThreadPriority::TPri_TimeCritical, FPlatformAffinity::GetPoolThreadMask());
        if (Thread == nullptr)
        {
            UE_LOG(LogTemp, Error, TEXT("Thread has not been created in Constructor"));
        }
    }

    virtual ~CameraThreadBase()
    {
        if (Thread != nullptr)
        {
            Stop();  // Set the thread to stop
            WaitForThreadToStop();  // Ensure thread stops before deletion
            delete Thread;
            Thread = nullptr;
            UE_LOG(LogTemp, Warning, TEXT("Thread has been deleted in Destructor"));
        }
    }

public:
    virtual FSingleThreadRunnable* GetSingleThreadInterface() override
    {
        return this;
    }

    virtual void Tick() override
    {
        Process();
    }

    //~~~~~~~~~~~~~~~~~~~~~~~
    //To be Subclassed
    virtual void Process() {}
    //~~~~~~~~~~~~~~~~~~~~~~~

public:
    virtual bool Init() override
    {
        return true;
    }

    virtual uint32 Run() override
    {
        HasStopped.AtomicSet(false);

        while (!bStopping)
        {
            if (Paused)
            {
                if (!IsVerifiedSuspended)
                {
                    IsVerifiedSuspended.AtomicSet(true);
                }

                // Sleep while paused to save CPU cycles
                FPlatformProcess::Sleep(ThreadSleepTime.GetTotalSeconds());
                continue;
            }

            Process();
        }

        HasStopped.AtomicSet(true);
        return 0;
    }

    virtual void Stop() override
    {
        SetPaused(true);
        bStopping = true;
    }

public:
    void SetPaused(bool MakePaused)
    {
        Paused.AtomicSet(MakePaused);
        if (!MakePaused)
        {
            IsVerifiedSuspended.AtomicSet(false);
        }
    }

    bool IsThreadPaused()
    {
        return Paused;
    }

    bool IsThreadVerifiedSuspended()
    {
        return IsVerifiedSuspended;
    }

    bool ThreadHasStopped()
    {
        return HasStopped;
    }

protected:
    void WaitForThreadToStop()
    {
        while (!HasStopped)
        {
            FPlatformProcess::Sleep(0.01f);  // Wait for thread to stop
        }
    }

    FThreadSafeBool Paused;
    FThreadSafeBool IsVerifiedSuspended;
    FThreadSafeBool HasStopped;
    FRunnableThread* Thread = nullptr;
    bool bStopping;

public:
    FTimespan ThreadSleepTime;
};
