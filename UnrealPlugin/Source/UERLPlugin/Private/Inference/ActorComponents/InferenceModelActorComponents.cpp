#include "Inference/ActorComponents/InferenceModelActorComponents.h"
#include "UERLPlugin/Helpers/BPFL_DataHelpers.h"

// Sets default values for this component's properties
UInferenceModelActorComponents::UInferenceModelActorComponents()
{
    PrimaryComponentTick.bCanEverTick = true;
    SetComponentTickEnabled(false);
}

// Called when the game starts
void UInferenceModelActorComponents::BeginPlay()
{
    Super::BeginPlay();
}

FString UInferenceModelActorComponents::RunLocalModelInference(const FString& Observation)
{
    if (InferenceInterface) {
        return InferenceInterface->RunInference(UBPFL_DataHelpers::ParseStateString(Observation));
    }
    else {
        UE_LOG(LogTemp, Warning, TEXT("InferenceModelActorComponents: Empty InferenceInterface ptr."));
        return "";
    }
}

// Called every frame
void UInferenceModelActorComponents::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

    // If not waiting for the previous action, trigger inference.
    if (!bIsWaitingForAction)
    {
        FString State = CreateStateString();
        FString ActionResponse = RunLocalModelInference(State);

        if (!ActionResponse.IsEmpty())
        {
            HandleResponseActions(ActionResponse);
            bIsWaitingForAction = true;
        }
    }
    else
    {
        // Check if the current action has finished.
        bIsWaitingForAction = IsActionRunning();
    }
}

void UInferenceModelActorComponents::StartInference()
{
    SetComponentTickEnabled(true);
}

void UInferenceModelActorComponents::StopInference()
{
    SetComponentTickEnabled(false);
}

bool UInferenceModelActorComponents::SetInferenceInterface(UInferenceInterface* Interface)
{
    if (Interface) {
        InferenceInterface = Interface;
        return true;
    }
    else {
        UE_LOG(LogTemp, Warning, TEXT("InferenceModelActorComponents: Empty InferenceInterface ptr."));
        return false;
    }
}

FString UInferenceModelActorComponents::CreateStateString_Implementation()
{
    // Default implementation returns an empty string. Users should override this.
    return FString();
}

void UInferenceModelActorComponents::HandleResponseActions_Implementation(const FString& actions)
{
    // Default implementation does nothing. Users should override this.
    UE_LOG(LogTemp, Log, TEXT("HandleResponseActions received: %s"), *actions);
}

bool UInferenceModelActorComponents::IsActionRunning_Implementation()
{
    // Default returns false, meaning new inference can run every tick.
    // Users should override this to provide meaningful action state checking.
    return false;
}
