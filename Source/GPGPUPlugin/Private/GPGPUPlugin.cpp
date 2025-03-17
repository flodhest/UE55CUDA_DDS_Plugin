#include "GPGPUPlugin.h"

void FGPGPUPluginModule::StartupModule()
{
    // Runtime initialization (e.g., CUDA setup) can go here
    // For now, it's empty as no runtime-specific code was provided
}

void FGPGPUPluginModule::ShutdownModule()
{
    // Runtime cleanup (e.g., CUDA teardown) can go here
}

bool FGPGPUPluginModule::SupportsDynamicReloading()
{
    return true;
}

IMPLEMENT_MODULE(FGPGPUPluginModule, GPGPUPlugin)