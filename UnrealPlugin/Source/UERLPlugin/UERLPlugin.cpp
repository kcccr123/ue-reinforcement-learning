#include "UERLPlugin.h"
#include "Modules/ModuleManager.h"

// This class implements the core startup/shutdown for your plugin module.
class FUERLPluginModule : public IModuleInterface
{
public:
    // Called after the module DLL is loaded and object references are set up
    virtual void StartupModule() override
    {
        UE_LOG(LogTemp, Log, TEXT("UERLPlugin: StartupModule() called"));
    }

    // Called before the module is unloaded, right before the DLL is unloaded
    virtual void ShutdownModule() override
    {
        UE_LOG(LogTemp, Log, TEXT("UERLPlugin: ShutdownModule() called"));
    }
};

// The second argument to IMPLEMENT_MODULE must match the 'Name' in .uplugin and .Build.cs
IMPLEMENT_MODULE(FUERLPluginModule, UERLPlugin)
