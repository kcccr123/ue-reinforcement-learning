#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "Tickable.h"
#include "RLBaseBridge.generated.h"

/**
 * Base class for RL communication using TCP.
 * This class provides a framework for setting up your RL environment in Unreal Engine.
 * It includes default implementations for connecting and disconnecting via TCP.
 */
UCLASS(Abstract, Blueprintable)
class UERLPLUGIN_API URLBaseBridge : public UObject, public FTickableGameObject
{
    GENERATED_BODY()

//---------------------------------------------------------
// Functions for general RL communication
//---------------------------------------------------------
public:
    // Initialization function that can be implemented for special behavior in derived classes
    // Called at the start of Connect()
    // Intended to be used for setting up values for TCP handshake
    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "RLBridge")
    void InitializeBridge();
    virtual void InitializeBridge_Implementation();

    // Sends TCP handshake to python environment.
    // Called once connection is made; sends any information the python environment needs before training starts.
    // If overriden, user needs to make sure to send action and observation space size in handshake.
    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "RLBridge")
    void SendHandshake();
    virtual void SendHandshake_Implementation();

    // Connect to TCP socket, passing IP, port, and desired sizes for action and observation spaces.
    UFUNCTION(BlueprintCallable, Category = "RLBridge")
    virtual bool Connect(const FString& IPAddress, int32 port, int32 actionSpaceSize, int32 obsSpaceSize);

    // Disconnect from TCP.
    UFUNCTION(BlueprintCallable, Category = "RLBridge")
    virtual void Disconnect();

    // Starts training; user calls this after environment is initialized.
    UFUNCTION(BlueprintCallable, Category = "RLBridge")
    virtual void StartTraining();

    // Update loop for RL. Implemented by derived classes. Should be called every tick.
    UFUNCTION(BlueprintCallable, Category = "RLBridge")
    virtual void UpdateRL(float DeltaTime);

protected:
    // Pointer to the TCP socket.
    FSocket* ConnectionSocket = nullptr;

    // Socket info (set on connection).
    FString CurrentIP;
    int32 CurrentPort;

    // Flag indicating whether the simulation/training is running.
    bool bIsTraining = false;

    // Flag indicating whether we should send observation state or wait for current action to complete
    bool bIsWaitingForAction = false;

    // Flag indicating if waiting for python response from sending an action
    bool bIsWaitingForPythonResp = false;

    // Action space and observation space size
    int32 ActionSpaceSize = 0;
    int32 ObservationSpaceSize = 0;

    // Default implementations for sending and receiving data over TCP.
    virtual bool SendData(const FString& Data);
    virtual FString ReceiveData();


//---------------------------------------------------------
// Required Environment Overrides
// (Functions the user must implement for custom environment logic)
//---------------------------------------------------------
public:
    // Calculate reward based on the environment state.
    // Must set 'bDone' to true if the episode should end.
    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "RLBridge|Environment")
    float CalculateReward(bool& bDone);
    virtual float CalculateReward_Implementation(bool& bDone);

    // Create a state string that packs observation data to send to the RL model.
    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "RLBridge|Environment")
    FString CreateStateString();
    virtual FString CreateStateString_Implementation();

    // Handle a "RESET" command from the RL model to reset the simulation.
    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "RLBridge|Environment")
    void HandleReset();
    virtual void HandleReset_Implementation();

    // Process received actions from the RL model and apply them to the environment.
    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "RLBridge|Environment")
    void HandleResponseActions(const FString& actions);
    virtual void HandleResponseActions_Implementation(const FString& actions);

    /**
     * Called each frame to check if an in-progress action should block sending new states.
     * If this returns true, UpdateRL will skip its normal logic this tick.
     */
    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "RLBridge|Environment")
    bool IsActionRunning();
    virtual bool IsActionRunning_Implementation();

//---------------------------------------------------------
// FTickableGameObject Interface:
//---------------------------------------------------------
public:
    // Called every frame by the engine.
    virtual void Tick(float DeltaTime) override;

    // Return true so that this object is tickable.
    virtual bool IsTickable() const override;

    // Return false to disable ticking when the game is paused.
    virtual bool IsTickableWhenPaused() const override { return false; }

    // Specifies that this object should always tick.
    virtual ETickableTickType GetTickableTickType() const override { return ETickableTickType::Always; }

    // Returns a stat ID for profiling purposes.
    virtual TStatId GetStatId() const override;
};
