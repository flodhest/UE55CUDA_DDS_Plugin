// Plugins/CUDAPlugin/Source/GPGPUPlugin/Public/KernelContainer.h
#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "KernelContainer.generated.h"

UENUM()
namespace EKernelContainer {
    enum KernelType {
        kCUDA       UMETA(DisplayName = "CUDA"),
        kOpenCL     UMETA(DisplayName = "Open CL")
    };
}

UCLASS(BlueprintType)
class GPGPUPLUGIN_API UKernelContainer : public UObject
{
    GENERATED_BODY()

public:
    UKernelContainer();

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Kernel Parameters")
    TEnumAsByte<EKernelContainer::KernelType> KernelType;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Kernel Parameters", meta = (MultiLine = "true"))
    FText KernelCode;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Kernel Parameters")
    FString KernelEntryPoint;
};