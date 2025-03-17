
#include "Inference/InferenceInterfaces/InferenceInterface.h"

bool UInferenceInterface::LoadModel(const FString& ModelPath)
{
	// Base implementation for blueprint requirements
	return false;
}

FString UInferenceInterface::RunInference(const TArray<float>& Observation)
{
	// Base implementation for blueprint requirements
	return FString();
}
