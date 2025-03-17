// Plugins/CUDAPlugin/Source/GPGPUPluginEditor/GPGPUPluginEditor.Build.cs
using UnrealBuildTool;
using System.IO;

public class GPGPUPluginEditor : ModuleRules
{
    public GPGPUPluginEditor(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        // Include paths
        PublicIncludePaths.Add(Path.Combine(ModuleDirectory, "Public"));
        PrivateIncludePaths.Add(Path.Combine(ModuleDirectory, "Private"));

        // Dependencies
        PublicDependencyModuleNames.AddRange(new string[]
        {
            "Core",
            "CoreUObject",
            "Engine",
            "Slate",
            "SlateCore",
            "UnrealEd",      // For UFactory and editor tools
            "AssetTools",    // For FAssetTypeActions
            "GPGPUPlugin"    // For UKernelContainer
        });

        // Only build for editor
        if (Target.bBuildEditor)
        {
            PrivateDependencyModuleNames.AddRange(new string[] { "EditorStyle" }); // Optional, for styling
        }
    }
}