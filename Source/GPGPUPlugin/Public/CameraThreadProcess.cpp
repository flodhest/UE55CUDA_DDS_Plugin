#include "CameraThreadProcess.h"
#include "CameraBaseActor.h"

void CameraThreadProcess::Process()
{
    if (!CameraActor)
    {
        return;
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // This is the connection from a wrapper of OS Threading to UE game code.
    // The thread tick should NOT call any UE code that
    //  1. Creates or destroys objects
    //  2. Modifies the game world in any way
    //  3. Tries to debug draw anything
    //  4. Simple raw data calculations are best!
    CameraActor->CameraThreadTick();
}

bool CameraThreadProcess::CameraThreadInit()
{
    return true;}

void CameraThreadProcess::CameraThreadShutdown()
{
    
}