#pragma once

#include "CoreMinimal.h"
#include "RLBaseBridge.h"
#include "RLPawnBridge.generated.h"

/**
 * Specialized RL bridge for controlling a single pawn (e.g., a StarFighter).
 */
UCLASS(Blueprintable)
class UERLPLUGIN_API URLPawnBridge : public URLBaseBridge
{
    GENERATED_BODY()

public:
    // Set the pawn that this bridge will control.
    UFUNCTION(BlueprintCallable, Category = "RL|PawnBridge")
    virtual void SetControlledPawn(APawn* InPawn);

    // Override UpdateRL to handle state gathering and action application for the pawn.
    virtual void UpdateRL(float DeltaTime) override;

protected:
    // The pawn being controlled.
    UPROPERTY()
    APawn* ControlledPawn;
};
