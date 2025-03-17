// Plugins/CUDAPlugin/Source/GPGPUPluginEditor/Public/KernelContainerAssetTypeActions.h
#pragma once

#include "CoreMinimal.h"

#if WITH_EDITOR
#include "AssetTypeActions_Base.h"
#include "AssetTypeCategories.h"

class FKernelContainerAssetTypeActions : public FAssetTypeActions_Base
{
public:
    FKernelContainerAssetTypeActions(EAssetTypeCategories::Type InAssetCategory);

    virtual FText GetName() const override;
    virtual FColor GetTypeColor() const override;
    virtual UClass* GetSupportedClass() const override;
    virtual uint32 GetCategories() override;
    virtual void OpenAssetEditor(const TArray<UObject*>& InObjects, TSharedPtr<class IToolkitHost> EditWithinLevelEditor = TSharedPtr<IToolkitHost>()) override;
    virtual void PerformAssetDiff(UObject* OldAsset, UObject* NewAsset, const struct FRevisionInfo& OldRevision, const struct FRevisionInfo& NewRevision) const override;

private:
    EAssetTypeCategories::Type MyAssetCategory;
};
#endif // WITH_EDITOR