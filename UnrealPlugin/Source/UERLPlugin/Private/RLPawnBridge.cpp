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
    // should have a default definition of UpdateRL in the base class.

    if (!ControlledPawn)
    {
        UE_LOG(LogTemp, Warning, TEXT("RLPawnBridge: No controlled pawn to update."));
        return;
    }

    if (!ConnectionSocket)
    {
        UE_LOG(LogTemp, Warning, TEXT("RLPawnBridge: Not Connected to a socket."));
        return;
    }

    // ---------------------- MAKE STATE STRING ---------------------------------

    // this entire section should also be converted to a blueprint native function in the future
    // user defined function that outputs a string of data to be sent to external python wrapper.

    // get pawn state
    FVector PawnLocation = ControlledPawn->GetActorLocation();
    FQuat PawnRotation = ControlledPawn->GetActorQuat(); // Get rotation as a quaternion

    bool bDone = false;
    float Reward = CalculateReward(bDone);
    int32 DoneInt = bDone ? 1 : 0;

    // send position and quat
    FString PosString = FString::Printf(TEXT("%.2f,%.2f,%.2f"), PawnLocation.X, PawnLocation.Y, PawnLocation.Z);
    FString QuatString = FString::Printf(TEXT("%.2f,%.2f,%.2f,%.2f"), PawnRotation.W, PawnRotation.X, PawnRotation.Y, PawnRotation.Z);

    // Combine both into one state string (using a delimiter, e.g., semicolon)
    FString DataToSend = FString::Printf(TEXT("%s;%s;%.2f;%d"), *PosString, *QuatString, Reward, DoneInt);


    // ---------------------- MAKE STATE STRING ---------------------------------

    // send data
    SendData(DataToSend);

    // recieve response
    FString ActionResponse = ReceiveData();
    if (!ActionResponse.IsEmpty())
    {
        if (ActionResponse.Equals("RESET"))
        {
            HandleReset();
            return;
        }

        HandleResponseActions(ActionResponse);

        // interpret response and call actions on class here
    }
}


float URLPawnBridge::CalculateReward_Implementation(bool& bDone)
{
    if (!ControlledPawn)
    {
        bDone = false;
        return 0.0f;
    }

    // user can override distance threshold, location, etc. in BP or child class
    float Reward = 0.0f;
    bDone = false;


    // ------- Movement Training -------------
    // Target location
    // This will teach model to only move to 100 100 100 each time. 
    // Need to pass commands through observation state for dynamic movement
    FVector Target(1000.f, 1000.f, 1000.f);
    float Dist = FVector::Dist(ControlledPawn->GetActorLocation(), Target);

    // higher reward as you get closer to location
    float scaleFactor = 1000.f; // adjust based on your environment scale
    Reward += FMath::Exp(-Dist / scaleFactor);
    if (Dist < 100.0f)
    {
        Reward += 10000000000.f; // big reward
        bDone = true;
    }
    // ------- Movement Training -------------


    // reset simulaton if 180 frames have passed
    if (passes >= 1048) {
        bDone = true;
    }

    passes += 1;

    return Reward;
}

void URLPawnBridge::HandleReset_Implementation()
{
    UE_LOG(LogTemp, Log, TEXT("Resetting!"));
    AFighterPawn* FighterPawn = Cast<AFighterPawn>(ControlledPawn);

    // reset pawn to starting point
    FighterPawn->SetActorLocation(FVector(0, 0, 0));
    FighterPawn->SetActorRotation(FRotator(0, 0, 0));
    FighterPawn->InitFighter(1.0, 1.0);
    passes = 0;
}

FString URLPawnBridge::CreateStateString_Implementation()
{
    return FString();
}

void URLPawnBridge::HandleResponseActions_Implementation(const FString& actions)
{
    AFighterPawn* FighterPawn = Cast<AFighterPawn>(ControlledPawn);

    // parse actions
    TArray<FString> ActionStrings;
    actions.ParseIntoArray(ActionStrings, TEXT(","), true);
    TArray<float> ActionValues;
    for (const FString& ActionStr : ActionStrings)
    {
        float Value = FCString::Atof(*ActionStr);
        ActionValues.Add(Value);
    }

    UE_LOG(LogTemp, Log, TEXT("Applying Actions!"));
    // then apply to pawn
    FighterPawn->ForwardBackThrust(ActionValues[0]);
    FighterPawn->LeftRightThrust(ActionValues[1]);
    FighterPawn->UpDownThrust(ActionValues[2]);

    FighterPawn->ApplyTorque(FVector(ActionValues[3], ActionValues[4], ActionValues[5]));


}
