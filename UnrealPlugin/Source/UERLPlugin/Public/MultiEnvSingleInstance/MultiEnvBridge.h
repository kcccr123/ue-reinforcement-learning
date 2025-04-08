#pragma once

#include "CoreMinimal.h"
#include "BaseBridge.h"
#include "MultiEnvBridge.generated.h"

/**
 * UMultiEnvBridge
 *
 * A multi-environment RL bridge, inherits from BaseBridge.
 * This class implements a multi-env update loop and manages per-environment state using internal TArrays.
 */
UCLASS(Blueprintable)
class UERLPLUGIN_API UMultiEnvBridge : public UBaseBridge
{
    GENERATED_BODY()

protected:
    // -------------------------------------------------------------
    //  Multi-Environment Properties
    // -------------------------------------------------------------

    /** Number of sub-environments. Set via InitializeEnvironments(). */
    UPROPERTY(VisibleAnywhere, Category = "MultiEnv|Environment")
    int32 NumEnvironments = 0;

    /** Array of booleans that determines if an enviornment is still running actions */
    UPROPERTY(VisibleAnywhere, Category = "MultiEnv|Environment")
    TArray<bool> bIsActionRunning;



public:
    // -------------------------------------------------------------
    //  Initialization and training loop functions
    // -------------------------------------------------------------
    /** Initialize the number of environments and resize the internal arrays accordingly. */
    UFUNCTION(BlueprintCallable, Category = "MultiEnv")
    void InitializeEnvironments(int32 InNumEnvironments = 1, bool bInInferenceMode = false);

protected:
    // Override handshake to send multi enviornment configuration settngs
    virtual void SendHandshake_Implementation() override;

    // -------------------------------------------------------------
    //  RL Loop: UpdateRL Implementation
    // -------------------------------------------------------------
    virtual void UpdateRL_Implementation(float DeltaTime) override;

    // -------------------------------------------------------------
    //  Environment Callbacks – To be overridden by the user
    // -------------------------------------------------------------
    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "MultiEnv|Environment")
    FString CreateStateStringForEnv(int32 EnvId);
    virtual FString CreateStateStringForEnv_Implementation(int32 EnvId);

    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "MultiEnv|Environment")
    float CalculateRewardForEnv(int32 EnvId, bool& bDone);
    virtual float CalculateRewardForEnv_Implementation(int32 EnvId, bool& bDone);

    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "MultiEnv|Environment")
    void HandleResetForEnv(int32 EnvId);
    virtual void HandleResetForEnv_Implementation(int32 EnvId);

    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "MultiEnv|Environment")
    void HandleResponseActionsForEnv(int32 EnvId, const FString& Actions);
    virtual void HandleResponseActionsForEnv_Implementation(int32 EnvId, const FString& Actions);

    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "MultiEnv|Environment")
    bool IsActionRunningForEnv(int32 EnvId);
    virtual bool IsActionRunningForEnv_Implementation(int32 EnvId);

};
