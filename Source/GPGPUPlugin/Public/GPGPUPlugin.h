#pragma once
#include "CoreMinimal.h"
#include "Modules/ModuleInterface.h"

class FGPGPUPluginModule : public IModuleInterface
{
public:
    virtual void StartupModule() override;
    virtual void ShutdownModule() override;
    virtual bool SupportsDynamicReloading() override;
};