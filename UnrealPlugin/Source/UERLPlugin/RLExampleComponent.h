#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "RLExampleComponent.generated.h"

UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class UERLPLUGIN_API URLExampleComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    URLExampleComponent();

protected:
    virtual void BeginPlay() override;

public:
    virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

private:
    FVector CurrentActorLocation;
};
