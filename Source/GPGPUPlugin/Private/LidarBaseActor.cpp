#include "LidarBaseActor.h"

ALidarBaseActor::ALidarBaseActor()
{
    PrimaryActorTick.bCanEverTick = true;
}

void ALidarBaseActor::BeginPlay()
{
    Super::BeginPlay();
    // Initialize camera threads

}

void ALidarBaseActor::EndPlay(EEndPlayReason::Type Reason)
{

    Super::EndPlay(Reason);
}
