#pragma once

#include "Kismet/BlueprintFunctionLibrary.h"
#include "CoreMinimal.h"
#include "StateStringHelpers.generated.h"

/**
 * A Blueprint Function Library providing helper functions for creating and parsing state strings.
 * In this version, all values are separated by commas, and delimiters are inserted only between values.
 */
UCLASS()
class UERLPLUGIN_API UStateStringHelpers : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()

public:
    /**
     * Parses a comma-separated state string into an array of float values.
     *
     * Example:
     *   "1.0,2.0,3.0,10.0,4.0,5.0" will be parsed into
     *   [1.0, 2.0, 3.0, 10.0, 4.0, 5.0].
     *
     * @param StateString The state string containing comma-separated float values.
     * @return A flat array of parsed float values.
     */
    UFUNCTION(BlueprintCallable, Category = "StateStringHelpers")
    static TArray<float> ParseStateString(const FString& StateString);

    /**
     * Appends a new string value to an existing state string using commas as the delimiter.
     * If BaseState is empty, returns NewValue.
     *
     * @param BaseState The current state string.
     * @param NewValue The new string value to append.
     * @return The concatenated state string.
     */
    UFUNCTION(BlueprintCallable, Category = "StateStringHelpers")
    static FString AppendToStateString(const FString& BaseState, const FString& NewValue);

    /**
     * Appends a float value to an existing state string using commas as the delimiter.
     *
     * @param BaseState The current state string.
     * @param Value The float value to append.
     * @param Precision The number of decimal places to format the float (default is 2).
     * @return The concatenated state string.
     */
    UFUNCTION(BlueprintCallable, Category = "StateStringHelpers")
    static FString AppendToStateString_Float(const FString& BaseState, float Value, int32 Precision = 2);

    /**
     * Appends an integer value to an existing state string using commas as the delimiter.
     *
     * @param BaseState The current state string.
     * @param Value The integer value to append.
     * @return The concatenated state string.
     */
    UFUNCTION(BlueprintCallable, Category = "StateStringHelpers")
    static FString AppendToStateString_Int(const FString& BaseState, int32 Value);

    /**
     * Converts an array of floats into a comma-separated string.
     *
     * @param FloatArray The array of float values.
     * @param Precision The number of decimal places for each float (default is 2).
     * @return A string representation of the float array.
     */
    UFUNCTION(BlueprintCallable, Category = "StateStringHelpers")
    static FString ArrayToStateString(const TArray<float>& FloatArray, int32 Precision = 2);

    /**
     * Appends an array of floats (converted to a comma-separated string) to an existing state string.
     * The array portion is appended using a comma delimiter.
     *
     * @param BaseState The current state string.
     * @param FloatArray The array of float values to append.
     * @param Precision The number of decimal places for each float (default is 2).
     * @return The concatenated state string.
     */
    UFUNCTION(BlueprintCallable, Category = "StateStringHelpers")
    static FString AppendToStateString_Array(const FString& BaseState, const TArray<float>& FloatArray, int32 Precision = 2);

    /**
     * Appends a 3D vector to an existing state string.
     * The vector components (X, Y, Z) are separated by commas.
     *
     * @param BaseState The current state string.
     * @param Vector The 3D vector to append.
     * @param Precision The number of decimal places for each component (default is 2).
     * @return The concatenated state string.
     */
    UFUNCTION(BlueprintCallable, Category = "StateStringHelpers")
    static FString AppendVectorToStateString(const FString& BaseState, const FVector& Vector, int32 Precision = 2);
};
