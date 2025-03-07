#pragma once

#include "CoreMinimal.h"
#include "RLBaseBridge.h"
#include "FighterPawn.h"
#include "RLPawnBridge.generated.h"

/**
 * Implementation example of RLBaseBridge
 * Specialized RL bridge for controlling a single pawn (e.g., a StarFighter).
 */
UCLASS(Blueprintable)
class UERLPLUGIN_API URLPawnBridge : public URLBaseBridge
{
    GENERATED_BODY()

public:
    // Set the pawn that this bridge will control.
    UFUNCTION(BlueprintCallable, Category = "RLBridge")
    virtual void SetControlledPawn(APawn* InPawn);

    // Override UpdateRL to handle state gathering and action application for the pawn.
    virtual void UpdateRL(float DeltaTime) override;


protected:
    // The pawn being controlled.
    UPROPERTY()
    APawn* ControlledPawn;

    int passes = 0;


    virtual float CalculateReward_Implementation(bool& bDone) override;

    virtual void HandleReset_Implementation() override;

    virtual FString CreateStateString_Implementation() override;

    virtual void HandleResponseActions_Implementation(const FString& actions) override;


};
