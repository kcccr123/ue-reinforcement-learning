#pragma once

#include "Kismet/BlueprintFunctionLibrary.h"
#include "CoreMinimal.h"
#include "PythonMsgParsingHelpers.generated.h"

/**
 * A Blueprint Function Library providing helper functions for parsing responses
 * from the Python module.
 *
 */
UCLASS()
class UERLPLUGIN_API UPythonMsgParsingHelpers : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()

public:
    /**
     * Parses the environment ID from a message string.
     * Expected formats: "ENV=2;ACT=0.10,-0.20" or "ENV=1;RESET"
     *
     * @param Message The complete message string to parse.
     * @return The extracted environment ID.
     */
    UFUNCTION(BlueprintCallable, Category = "PythonMsgParsingHelpers")
    static int32 ParseEnvId(const FString& Message);

    /**
     * Parses the action substring from a message string.
     * Expected format: "ENV=x;ACT=0.10,-0.20"
     *
     * @param Message The complete message string to parse.
     * @return The extracted action substring.
     */
    UFUNCTION(BlueprintCallable, Category = "PythonMsgParsingHelpers")
    static FString ParseActionString(const FString& Message);

    /**
     * Converts a comma-separated action string into an array of float values.
     *
     * Example:
     *   "0.50,0.25" will be parsed into {0.50, 0.25}.
     *
     * @param ActionString The input action string.
     * @return An array of floats parsed from the action string.
     */
    UFUNCTION(BlueprintCallable, Category = "PythonMsgParsingHelpers")
    static TArray<float> ParseActionFloatArray(const FString& ActionString);
};
