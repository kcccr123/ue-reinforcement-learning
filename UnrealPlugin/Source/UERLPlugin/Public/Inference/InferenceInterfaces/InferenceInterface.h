#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "InferenceInterface.generated.h"

/**
 * Base class for inference models.
 * Provides Blueprint-callable functions to load a model and run inference.
 * This base implementation does nothing and should be overridden.
 */
UCLASS(Blueprintable)
class UERLPLUGIN_API UInferenceInterface : public UObject
{
	GENERATED_BODY()

public:
	/** Loads the model from a model file.
	 *  @param ModelPath The path to the model file.
	 *  @return True if the model loads successfully.
	 */
	UFUNCTION(BlueprintCallable, Category = "Inference")
	virtual bool LoadModel(const FString& ModelPath);

	/** Runs inference given an array of float observations.
	 *  @param Observation An array of floats representing the input.
	 *  @return A comma-separated string representing the output actions.
	 */
	UFUNCTION(BlueprintCallable, Category = "Inference")
	virtual FString RunInference(const TArray<float>& Observation);
};
