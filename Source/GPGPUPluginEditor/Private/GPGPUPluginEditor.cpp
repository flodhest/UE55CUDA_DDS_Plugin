#include "GPGPUPluginEditor.h"
#include "KernelContainerAssetTypeActions.h"
#include "IAssetTools.h"
#include "AssetTypeCategories.h"
#include "IAssetTypeActions.h"
#include "Modules/ModuleManager.h"
#include "Templates/SharedPointer.h"

#define LOCTEXT_NAMESPACE "FGPGPUPluginEditorModule"

void FGPGPUPluginEditorModule::StartupModule()
{
    IAssetTools& AssetTools = FModuleManager::LoadModuleChecked<FAssetToolsModule>("AssetTools").Get();
    GPGPUPluginCategoryBit = AssetTools.RegisterAdvancedAssetCategory(FName(TEXT("GPGPU")), LOCTEXT("GPGPUPluginCategory", "GPGPU"));
    RegisterAssetTypeAction(AssetTools, MakeShareable(new FKernelContainerAssetTypeActions(GPGPUPluginCategoryBit)));
}

void FGPGPUPluginEditorModule::ShutdownModule()
{
    if (FModuleManager::Get().IsModuleLoaded("AssetTools"))
    {
        IAssetTools& AssetTools = FModuleManager::GetModuleChecked<FAssetToolsModule>("AssetTools").Get();
        for (int32 Index = 0; Index < CreatedAssetTypeActions.Num(); ++Index)
        {
            AssetTools.UnregisterAssetTypeActions(CreatedAssetTypeActions[Index].ToSharedRef());
        }
    }
    CreatedAssetTypeActions.Empty();
}

bool FGPGPUPluginEditorModule::SupportsDynamicReloading()
{
    return true;
}

void FGPGPUPluginEditorModule::RegisterAssetTypeAction(IAssetTools& AssetTools, TSharedRef<IAssetTypeActions> Action)
{
    AssetTools.RegisterAssetTypeActions(Action);
    CreatedAssetTypeActions.Add(Action);
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FGPGPUPluginEditorModule, GPGPUPluginEditor)