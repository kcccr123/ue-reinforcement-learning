#pragma once

#include "CoreMinimal.h"
#include "TrainingBridges/BaseBridge.h"
#include "SingleEnvBridge.generated.h"

/**
 * A single-environment RL bridge, inherits from UBaseBridge.
 * This class implements a single-env update loop
 *
 */
UCLASS(Blueprintable)
class UERLPLUGIN_API USingleEnvBridge : public UBaseBridge
{
    GENERATED_BODY()

public:
    // -------------------------------------------------------------
    //  RL Loop: UpdateRL Implementation
    // -------------------------------------------------------------
    /**
     * The main RL update function called every frame.
     * Supports training and inference modes.
     * Designed for handling on demand system of OpenAI Gymnasium framework
     */
    virtual void UpdateRL_Implementation(float DeltaTime) override;

protected:

    // Override handshake to send multi enviornment configuration settngs
    FString BuildHandshake_Implementation() override;

    // Implement factory function for creating
    UBaseTcpConnection* CreateTcpConnection_Implementation() override;

    // -------------------------------------------------------------
    //  Environment Callbacks 
    // -------------------------------------------------------------
    /**
     * Computes and returns the reward for the current timestep, setting bIsDone if the episode ends.
     */
    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "SingleEnv|Environment")
    float CalculateReward(bool& bIsDone);
    virtual float CalculateReward_Implementation(bool& bIsDone);

    /**
     * Creates and returns a string representing the current observation state.
     */
    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "SingleEnv|Environment")
    FString CreateStateString();
    virtual FString CreateStateString_Implementation();

    /**
     * Called when a reset is requested (e.g., after the environment is done).
     */
    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "SingleEnv|Environment")
    void HandleReset();
    virtual void HandleReset_Implementation();

    /**
     * Called to parse and apply actions received either from Python or local inference.
     */
    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "SingleEnv|Environment")
    void HandleResponseActions(const FString& Actions);
    virtual void HandleResponseActions_Implementation(const FString& Actions);

    /**
     * Returns true if the environment is currently running an action that has not yet completed.
     * This prevents requesting new actions until the current one finishes.
     */
    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "SingleEnv|Environment")
    bool IsActionRunning();
    virtual bool IsActionRunning_Implementation();

private:

    /** True when we have just applied an action and are waiting for it to finish before requesting another. */
    bool bIsActionRunning = false;

};
