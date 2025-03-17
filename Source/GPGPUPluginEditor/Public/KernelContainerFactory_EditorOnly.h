// Plugins/CUDAPlugin/Source/GPGPUPluginEditor/Public/KernelContainerFactory_EditorOnly.h
#pragma once

#include "CoreMinimal.h"

#if WITH_EDITOR
#include "Factories/Factory.h"
#include "KernelContainerFactory_EditorOnly.generated.h"

UCLASS()
class GPGPUPLUGINEDITOR_API UKernelContainerFactory : public UFactory
{
    GENERATED_BODY()

public:
    UKernelContainerFactory();
    virtual UObject* FactoryCreateNew(UClass* Class, UObject* InParent, FName Name, EObjectFlags Flags, UObject* Context, FFeedbackContext* Warn) override;
    virtual bool ShouldShowInNewMenu() const override;
};
#endif // WITH_EDITOR