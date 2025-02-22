#include "RLExampleComponent.h"
#include "GameFramework/Actor.h"
#include "Engine/World.h"

URLExampleComponent::URLExampleComponent()
{
    PrimaryComponentTick.bCanEverTick = true;
}

void URLExampleComponent::BeginPlay()
{
    Super::BeginPlay();
    UE_LOG(LogTemp, Warning, TEXT("RLExampleComponent: BeginPlay() called"));
}

void URLExampleComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction *ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

    AActor *Owner = GetOwner();
    if (Owner)
    {
        CurrentActorLocation = Owner->GetActorLocation();
        // TODO: In a real scenario, you'd store or send this data.
        // E.g., Send this position to a Python RL agent, or log it for debugging.
        UE_LOG(LogTemp, Log, TEXT("Actor Location: %s"), *CurrentActorLocation.ToString());
    }
}
