#include "RLPawnBridge.h"
#include "GameFramework/Pawn.h"
#include "Engine/Engine.h"

void URLPawnBridge::SetControlledPawn(APawn* InPawn)
{
    ControlledPawn = InPawn;
    if (ControlledPawn)
    {
        UE_LOG(LogTemp, Log, TEXT("RLPawnBridge: Controlled pawn set to %s"), *ControlledPawn->GetName());
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("RLPawnBridge: Provided pawn is null."));
    }
}

void URLPawnBridge::UpdateRL(float DeltaTime)
{
    // Call the base implementation (optional)
    Super::UpdateRL(DeltaTime);

    if (!ControlledPawn)
    {
        UE_LOG(LogTemp, Warning, TEXT("RLPawnBridge: No controlled pawn to update."));
        return;
    }

    // ---------------------- MAKE STATE STRING ---------------------------------

    // get pawn state
    FVector PawnLocation = ControlledPawn->GetActorLocation();
    FQuat PawnRotation = ControlledPawn->GetActorQuat(); // Get rotation as a quaternion

    // send position and quat
    FString PosString = FString::Printf(TEXT("%.2f,%.2f,%.2f"), PawnLocation.X, PawnLocation.Y, PawnLocation.Z);
    FString QuatString = FString::Printf(TEXT("%.2f,%.2f,%.2f,%.2f"), PawnRotation.W, PawnRotation.X, PawnRotation.Y, PawnRotation.Z);

    // Combine both into one state string (using a delimiter, e.g., semicolon)
    FString StateString = PosString + TEXT(";") + QuatString;

    // ---------------------- MAKE STATE STRING ---------------------------------

    // send data
    SendData(StateString);

    // recieve response
    FString ActionResponse = ReceiveData();
    if (!ActionResponse.IsEmpty())
    {
        // interpret response and call actions on class here
    }
}
