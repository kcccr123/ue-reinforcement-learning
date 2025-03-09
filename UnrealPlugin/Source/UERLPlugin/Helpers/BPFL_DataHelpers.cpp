#include "BPFL_DataHelpers.h"

TArray<float> UBPFL_DataHelpers::ParseActionString(const FString& ActionString)
{
    TArray<float> OutValues;
    TArray<FString> ParsedStrings;
    // Split using comma as delimiter.
    ActionString.ParseIntoArray(ParsedStrings, TEXT(","), true);

    for (const FString& Str : ParsedStrings)
    {
        float Value = FCString::Atof(*Str);
        OutValues.Add(Value);
    }

    return OutValues;
}

FString UBPFL_DataHelpers::AppendToStateString(const FString& BaseState, const FString& NewValue)
{
    return BaseState + NewValue + TEXT(";");
}

FString UBPFL_DataHelpers::AppendToStateString_Float(const FString& BaseState, float Value, int32 Precision)
{
    FString ValueStr = FString::Printf(TEXT("%.*f"), Precision, Value);
    return AppendToStateString(BaseState, ValueStr);
}

FString UBPFL_DataHelpers::AppendToStateString_Int(const FString& BaseState, int32 Value)
{
    FString ValueStr = FString::FromInt(Value);
    return AppendToStateString(BaseState, ValueStr);
}

FString UBPFL_DataHelpers::ArrayToStateString(const TArray<float>& FloatArray, int32 Precision)
{
    TArray<FString> StringArray;

    // Convert each float to a string using comma separation.
    for (float Value : FloatArray)
    {
        StringArray.Add(FString::Printf(TEXT("%.*f"), Precision, Value));
    }
    // Join the array elements with a comma.
    return FString::Join(StringArray, TEXT(","));
}

FString UBPFL_DataHelpers::AppendToStateString_Array(const FString& BaseState, const TArray<float>& FloatArray, int32 Precision)
{
    FString ArrayStr = ArrayToStateString(FloatArray, Precision);
    return AppendToStateString(BaseState, ArrayStr);
}


FString UBPFL_DataHelpers::AppendVectorToStateString(const FString& BaseState, const FVector& Vector, int32 Precision)
{
    // Format each component to the specified precision.
    FString XStr = FString::Printf(TEXT("%.*f"), Precision, Vector.X);
    FString YStr = FString::Printf(TEXT("%.*f"), Precision, Vector.Y);
    FString ZStr = FString::Printf(TEXT("%.*f"), Precision, Vector.Z);

    // Build a comma-separated string and append a semicolon at the end.
    return AppendToStateString(BaseState, XStr + TEXT(",") + YStr + TEXT(",") + ZStr);
}