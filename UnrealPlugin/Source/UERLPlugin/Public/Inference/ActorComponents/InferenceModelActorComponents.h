#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Inference/InferenceInterfaces/InferenceInterface.h"
#include "InferenceModelActorComponents.generated.h"

/**
 * Base Actor Component for running inference on Actors.
 * This class provides a framework for loading a trained model and
 * having it control and act as the AI component for a respective actor.
 */
UCLASS(Abstract, ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class UERLPLUGIN_API UInferenceModelActorComponents : public UActorComponent
{
    GENERATED_BODY()

public:
    // Sets default values for this component's properties
    UInferenceModelActorComponents();

protected:
    // Called when the game starts
    virtual void BeginPlay() override;

    // Pointer to model interface
    UInferenceInterface* InferenceInterface = nullptr;

    // Flag to determine if we are waiting for the current action to finish.
    bool bIsWaitingForAction = false;

    // Run inference using loaded model
    FString RunLocalModelInference(const FString& Observation);

public:
    // Called every frame
    virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

    // Starts inference; user calls this after Inference model is initialized.
    UFUNCTION(BlueprintCallable, Category = "RLBridge")
    virtual void StartInference();

    // Stops inference
    UFUNCTION(BlueprintCallable, Category = "RLBridge")
    virtual void StopInference();

    // Called before or during inference mode: load your model inference interface. 
    // Returns true if loaded succesfully
    UFUNCTION(BlueprintCallable, Category = "RLBridge")
    bool SetInferenceInterface(UInferenceInterface* Interface);

    //---------------------------------------------------------
    // Required Model Logic Overrides
    // (Functions the user must implement for custom environment logic)
    //---------------------------------------------------------
    // Create a state string that packs observation data to send to the RL model.
    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "RLBridge")
    FString CreateStateString();
    virtual FString CreateStateString_Implementation();

    // Process received actions from the RL model and apply them to the environment.
    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "RLBridge")
    void HandleResponseActions(const FString& actions);
    virtual void HandleResponseActions_Implementation(const FString& actions);

    /**
     * Called each frame to check if an in-progress action should block sending new states.
     * If this returns true, new inference will not be triggered until the current action finishes.
     */
    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "RLBridge")
    bool IsActionRunning();
    virtual bool IsActionRunning_Implementation();
};
