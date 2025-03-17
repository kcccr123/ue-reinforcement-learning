#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "BPFL_DataHelpers.generated.h"

/**
 * A Blueprint Function Library providing helper functions for handling data and state strings.
 * - When appending to a state string, segments are separated by ";".
 * - When converting an array to a string, elements are separated by ",".
 */
UCLASS()
class UERLPLUGIN_API UBPFL_DataHelpers : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()

public:

    /**
     * Parses a comma-separated action string into an array of float values.
     *
     * @param ActionString The input action string (e.g., "0.50,0.25").
     * @return An array of floats parsed from the string.
     */
    UFUNCTION(BlueprintCallable, Category = "DataHelpers")
    static TArray<float> ParseActionString(const FString& ActionString);

    /**
     * Splits a string by semicolon first, then for each chunk, splits by comma.
     * All parsed numeric values are returned in a single float array.
     *
     * Example:
     *   "1.0,2.0,3.0;10.0;4.0,5.0"
     *   => [1.0, 2.0, 3.0, 10.0, 4.0, 5.0]
     *
     * @param MixedString The string containing multiple segments separated by ";"
     *                    and optionally comma-delimited floats within each segment.
     * @return A flat array of all parsed float values.
     */
    UFUNCTION(BlueprintCallable, Category = "DataHelpers")
    static TArray<float> ParseStateString(const FString& MixedString);

    /**
     * Appends a new string value to an existing state string using ";" as the delimiter.
     * If BaseState is empty, returns NewValue.
     *
     * @param BaseState The current state string.
     * @param NewValue The new string value to append.
     * @return The concatenated state string.
     */
    UFUNCTION(BlueprintCallable, Category = "DataHelpers")
    static FString AppendToStateString(const FString& BaseState, const FString& NewValue);

    /**
     * Appends a float value to an existing state string using ";" as the delimiter.
     *
     * @param BaseState The current state string.
     * @param Value The float value to append.
     * @param Precision The number of decimal places to format the float (default is 2).
     * @return The concatenated state string.
     */
    UFUNCTION(BlueprintCallable, Category = "DataHelpers")
    static FString AppendToStateString_Float(const FString& BaseState, float Value, int32 Precision = 2);

    /**
     * Appends an integer value to an existing state string using ";" as the delimiter.
     *
     * @param BaseState The current state string.
     * @param Value The integer value to append.
     * @return The concatenated state string.
     */
    UFUNCTION(BlueprintCallable, Category = "DataHelpers")
    static FString AppendToStateString_Int(const FString& BaseState, int32 Value);

    /**
     * Converts an array of floats into a string with values separated by commas.
     *
     * @param FloatArray The array of float values.
     * @param Precision The number of decimal places to format each float (default is 2).
     * @return A string representation of the array.
     */
    UFUNCTION(BlueprintCallable, Category = "DataHelpers")
    static FString ArrayToStateString(const TArray<float>& FloatArray, int32 Precision = 2);

    /**
     * Appends an array of floats (converted to a string with comma separation) to an existing state string.
     * The array portion is appended using ";" as the delimiter.
     *
     * @param BaseState The current state string.
     * @param FloatArray The array of float values to append.
     * @param Precision The number of decimal places to format each float (default is 2).
     * @return The concatenated state string.
     */
    UFUNCTION(BlueprintCallable, Category = "DataHelpers")
    static FString AppendToStateString_Array(const FString& BaseState, const TArray<float>& FloatArray, int32 Precision = 2);

    /**
     * Converts a 3D vector into a state string.
     * The vector components (X, Y, Z) are separated by commas and the resulting string is terminated with a semicolon.
     *
     * @param Vector The 3D vector to convert.
     * @param Precision The number of decimal places to format each component (default is 2).
     * @return A string representation of the vector (e.g., "X,Y,Z;").
     */
    UFUNCTION(BlueprintCallable, Category = "DataHelpers")
    static FString AppendVectorToStateString(const FString& BaseState, const FVector& Vector, int32 Precision = 2);
};
