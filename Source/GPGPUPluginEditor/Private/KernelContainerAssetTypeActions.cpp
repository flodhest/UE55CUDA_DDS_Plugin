// Plugins/CUDAPlugin/Source/GPGPUPluginEditor/Private/KernelContainerAssetTypeActions.cpp
#include "KernelContainerAssetTypeActions.h"

#if WITH_EDITOR
#include "KernelContainer.h"
#include "Editor/EditorEngine.h"  // For UEditorLoadingSavingSettings
#include "AssetToolsModule.h"     // For IAssetTools
#include "Misc/ScopedSlowTask.h"  // Optional, for progress feedback

#define LOCTEXT_NAMESPACE "AssetTypeActions"

FKernelContainerAssetTypeActions::FKernelContainerAssetTypeActions(EAssetTypeCategories::Type InAssetCategory)
    : MyAssetCategory(InAssetCategory)
{
}

FText FKernelContainerAssetTypeActions::GetName() const
{
    return LOCTEXT("FKernelContainerAssetTypeActionsName", "Kernel Container");
}

FColor FKernelContainerAssetTypeActions::GetTypeColor() const
{
    return FColor::Purple;
}

UClass* FKernelContainerAssetTypeActions::GetSupportedClass() const
{
    return UKernelContainer::StaticClass();
}

uint32 FKernelContainerAssetTypeActions::GetCategories()
{
    return MyAssetCategory;
}

void FKernelContainerAssetTypeActions::OpenAssetEditor(const TArray<UObject*>& InObjects, TSharedPtr<IToolkitHost> EditWithinLevelEditor)
{
    FScopedSlowTask SlowTask(InObjects.Num(), LOCTEXT("OpeningKernelContainerEditor", "Opening Kernel Container Editor..."));
    SlowTask.MakeDialog();

    for (UObject* Obj : InObjects)
    {
        if (UKernelContainer* KernelContainer = Cast<UKernelContainer>(Obj))
        {
            UE_LOG(LogTemp, Log, TEXT("Opening editor for KernelContainer: %s"), *KernelContainer->GetName());
            // Placeholder: Add custom editor logic here if desired
            // FSimpleAssetEditor is not directly available in UE 5.0 without EditorFramework module
            SlowTask.EnterProgressFrame();
        }
    }
}

void FKernelContainerAssetTypeActions::PerformAssetDiff(UObject* OldAsset, UObject* NewAsset, const FRevisionInfo& OldRevision, const FRevisionInfo& NewRevision) const
{
    const UEditorLoadingSavingSettings* Settings = GetDefault<UEditorLoadingSavingSettings>();
    if (Settings && !Settings->TextDiffToolPath.FilePath.IsEmpty())
    {
        FAssetTypeActions_Base::PerformAssetDiff(OldAsset, NewAsset, OldRevision, NewRevision);
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("No text diff tool configured in Editor settings for KernelContainer diff."));
    }
}

#undef LOCTEXT_NAMESPACE
#endif // WITH_EDITOR