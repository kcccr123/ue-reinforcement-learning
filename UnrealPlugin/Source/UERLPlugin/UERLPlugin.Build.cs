using System;
using System.IO;
using UnrealBuildTool;

public class UERLPlugin : ModuleRules
{
    public UERLPlugin(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        // msgpack-c throws on malformed input (msgpack::unpack_error, etc.) and
        // ReceiveMessageEnv() wraps unpack in a try/catch. UE disables C++
        // exceptions by default, so enable them for this module.
        bEnableExceptions = true;

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
        string MsgPackPath = Path.Combine(ModuleDirectory, "../../MsgPack");

        // msgpack-c ships its own boost/predef subset; tell it not to look for
        // a system Boost installation.
        PublicDefinitions.Add("MSGPACK_NO_BOOST=1");

        // Win64
        if (Target.Platform == UnrealTargetPlatform.Win64)
        {
            PublicIncludePaths.Add(Path.Combine(OnnxRuntimePath, "include"));
            PublicIncludePaths.Add(Path.Combine(MsgPackPath, "include"));

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
