
#include "Inference/InferenceInterfaces/InferenceInterfaceOnnx.h"
#include "UERLPlugin/Helpers/BPFL_DataHelpers.h" 
#include "Misc/Paths.h"
#include "HAL/PlatformFilemanager.h"

#include <onnxruntime_cxx_api.h>
#include <string>

// Initialize the static globals.
std::unique_ptr<Ort::Env> UInferenceInterfaceOnnx::GEnv = nullptr;
std::unique_ptr<Ort::SessionOptions> UInferenceInterfaceOnnx::GSessionOptions = nullptr;

UInferenceInterfaceOnnx::UInferenceInterfaceOnnx()
{
}

UInferenceInterfaceOnnx::~UInferenceInterfaceOnnx()
{
	// SessionPtr will automatically be cleaned up since is unique ptr
	// add any logic for cleaning up Onnx logic here
}

bool UInferenceInterfaceOnnx::LoadModel(const FString& ModelPath)
{
	if (!GEnv)
	{
		GEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "UnrealONNX");
	}
	if (!GSessionOptions)
	{
		GSessionOptions = std::make_unique<Ort::SessionOptions>();
		GSessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
	}

#if PLATFORM_WINDOWS
	const std::wstring WideModelPath = *ModelPath;
	const ORTCHAR_T* ModelPathCStr = WideModelPath.c_str();
#else
	const std::string NarrowModelPath = TCHAR_TO_UTF8(*ModelPath);
	const ORTCHAR_T* ModelPathCStr = NarrowModelPath.c_str();
#endif

	UE_LOG(LogTemp, Log, TEXT("UInferenceInterfaceOnnx: Loading ONNX model from %s"), *ModelPath);

	try
	{
		SessionPtr = std::make_unique<Ort::Session>(*GEnv, ModelPathCStr, *GSessionOptions);
	}
	catch (const Ort::Exception& e)
	{
		UE_LOG(LogTemp, Error, TEXT("UInferenceInterfaceOnnx: Failed to load ONNX model: %s"), *FString(e.what()));
		SessionPtr.reset();
		return false;
	}

	return true;
}


FString UInferenceInterfaceOnnx::RunInference(const TArray<float>& Observation)
{
	if (!SessionPtr)
	{
		UE_LOG(LogTemp, Warning, TEXT("UInferenceInterfaceOnnx: Model not loaded."));
		return FString();
	}

	// Convert the TArray<float> to a std::vector<float>.
	std::vector<float> InputData(Observation.GetData(), Observation.GetData() + Observation.Num());
	// Assume the model expects an input shape of [1, N].
	std::vector<int64_t> InputShape = { 1, static_cast<int64_t>(Observation.Num()) };

	// NOTE: do we need to pass the observation space size here? or is it already baked into the model?

	// Create an input tensor.
	static Ort::MemoryInfo MemoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value InputTensor = Ort::Value::CreateTensor<float>(
		MemoryInfo,
		InputData.data(),
		InputData.size(),
		InputShape.data(),
		InputShape.size()
	);

	const char* InputNames[] = { "obs" };
	const char* OutputNames[] = { "actions" };

	Ort::RunOptions RunOptions;
	std::vector<Ort::Value> OutputTensors;
	try
	{
		OutputTensors = SessionPtr->Run(RunOptions, InputNames, &InputTensor, 1, OutputNames, 1);
	}
	catch (const Ort::Exception& e)
	{
		UE_LOG(LogTemp, Error, TEXT("UInferenceInterfaceOnnx: RunInference error: %s"), *FString(e.what()));
		return FString();
	}

	if (OutputTensors.empty() || !OutputTensors[0].IsTensor())
	{
		UE_LOG(LogTemp, Error, TEXT("UInferenceInterfaceOnnx: Invalid tensor output."));
		return FString();
	}

	// Extract output data.
	float* OutPtr = OutputTensors[0].GetTensorMutableData<float>();
	auto TensorInfo = OutputTensors[0].GetTensorTypeAndShapeInfo();
	size_t NumElements = TensorInfo.GetElementCount();

	TArray<float> OutputValues;
	OutputValues.SetNum(NumElements);
	for (size_t i = 0; i < NumElements; ++i)
	{
		OutputValues[i] = OutPtr[i];
	}

	// Convert the output array to a comma-separated string.
	FString ActionString = UBPFL_DataHelpers::ArrayToStateString(OutputValues, 2);
	return ActionString;
}

bool UInferenceInterfaceOnnx::IsModelLoaded() const
{
	return SessionPtr != nullptr;
}
