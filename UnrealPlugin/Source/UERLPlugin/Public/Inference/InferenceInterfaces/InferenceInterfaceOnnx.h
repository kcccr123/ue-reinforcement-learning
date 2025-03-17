#pragma once

#include "CoreMinimal.h"
#include "InferenceInterface.h"
#include "InferenceInterfaceOnnx.generated.h"

#include <onnxruntime_cxx_api.h>
#include <memory>

class Ort::Session;
class Ort::Env;
class Ort::SessionOptions;

/**
 * ONNX-based implementation of the inference interface.
 * Implements model loading and inference using ONNX Runtime.
 */
UCLASS(Blueprintable)
class UERLPLUGIN_API UInferenceInterfaceOnnx : public UInferenceInterface
{
	GENERATED_BODY()

public:
	UInferenceInterfaceOnnx();
	virtual ~UInferenceInterfaceOnnx();

	// Overrides from UInferenceInterface
	virtual bool LoadModel(const FString& ModelPath) override;
	virtual FString RunInference(const TArray<float>& Observation) override;

	/** Returns true if the model is loaded successfully. */
	UFUNCTION(BlueprintCallable, Category = "Inference")
	bool IsModelLoaded() const;

private:
	// Global ONNX environment and session options shared by all instances.
	static std::unique_ptr<Ort::Env> GEnv;
	static std::unique_ptr<Ort::SessionOptions> GSessionOptions;

	// The ONNX Runtime session for this model.
	std::unique_ptr<Ort::Session> SessionPtr;
};
