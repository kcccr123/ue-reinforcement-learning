using System;
using System.IO;
using UnrealBuildTool;

public class UERLPlugin : ModuleRules
{
    public UERLPlugin(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(
            new string[]
            {
                "Core",
                "CoreUObject",
                "Engine",
                "InputCore",
                "Sockets",
                "Networking",
                "EnhancedInput"
            }
        );

        string OnnxRuntimePath = Path.Combine(ModuleDirectory, "../../OnnxRuntime");

        // Win64
        if (Target.Platform == UnrealTargetPlatform.Win64)
        {
            PublicIncludePaths.Add(Path.Combine(OnnxRuntimePath, "include"));

            string LibPath = Path.Combine(OnnxRuntimePath, "lib");
            PublicAdditionalLibraries.Add(Path.Combine(LibPath, "onnxruntime.lib"));

            string DllPath = Path.Combine(LibPath, "onnxruntime.dll");
            if (File.Exists(DllPath))
            {
                RuntimeDependencies.Add(DllPath);
            }
        }
    }
}
