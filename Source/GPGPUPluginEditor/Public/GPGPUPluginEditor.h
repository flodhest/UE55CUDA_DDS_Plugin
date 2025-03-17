#pragma once
#include "CoreMinimal.h"
#include "Modules/ModuleInterface.h"
#include "AssetTypeCategories.h" // Moved here for EAssetTypeCategories::Type

#if WITH_EDITOR
class IAssetTools;
class IAssetTypeActions;
#endif

class FGPGPUPluginEditorModule : public IModuleInterface
{
public:
    virtual void StartupModule() override;
    virtual void ShutdownModule() override;
    virtual bool SupportsDynamicReloading() override;

#if WITH_EDITOR
private:
    void RegisterAssetTypeAction(class IAssetTools& AssetTools, TSharedRef<class IAssetTypeActions> Action);
    TArray<TSharedPtr<class IAssetTypeActions>> CreatedAssetTypeActions;
    EAssetTypeCategories::Type GPGPUPluginCategoryBit;
#endif
};