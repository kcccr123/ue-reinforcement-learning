#include "StateStringHelpers.h"

TArray<float> UStateStringHelpers::ParseStateString(const FString& StateString)
{
    TArray<float> OutValues;
    TArray<FString> ParsedStrings;
    // Split the state string using a comma as the delimiter.
    StateString.ParseIntoArray(ParsedStrings, TEXT(","), true);
    for (const FString& Str : ParsedStrings)
    {
        FString Trimmed = Str;
        Trimmed.TrimStartAndEndInline();
        if (!Trimmed.IsEmpty())
        {
            float Value = FCString::Atof(*Trimmed);
            OutValues.Add(Value);
        }
    }
    return OutValues;
}

FString UStateStringHelpers::AppendToStateString(const FString& BaseState, const FString& NewValue)
{
    // If BaseState is empty, return NewValue; otherwise, insert a comma between.
    return BaseState.IsEmpty() ? NewValue : BaseState + TEXT(",") + NewValue;
}

FString UStateStringHelpers::AppendToStateString_Float(const FString& BaseState, float Value, int32 Precision)
{
    FString ValueStr = FString::Printf(TEXT("%.*f"), Precision, Value);
    return AppendToStateString(BaseState, ValueStr);
}

FString UStateStringHelpers::AppendToStateString_Int(const FString& BaseState, int32 Value)
{
    FString ValueStr = FString::FromInt(Value);
    return AppendToStateString(BaseState, ValueStr);
}

FString UStateStringHelpers::ArrayToStateString(const TArray<float>& FloatArray, int32 Precision)
{
    TArray<FString> StringArray;
    // Convert each float to a string with the specified precision.
    for (float Value : FloatArray)
    {
        StringArray.Add(FString::Printf(TEXT("%.*f"), Precision, Value));
    }
    // Join the array elements with commas.
    return FString::Join(StringArray, TEXT(","));
}

FString UStateStringHelpers::AppendToStateString_Array(const FString& BaseState, const TArray<float>& FloatArray, int32 Precision)
{
    FString ArrayStr = ArrayToStateString(FloatArray, Precision);
    return AppendToStateString(BaseState, ArrayStr);
}

FString UStateStringHelpers::AppendVectorToStateString(const FString& BaseState, const FVector& Vector, int32 Precision)
{
    // Format each vector component to the specified precision.
    FString XStr = FString::Printf(TEXT("%.*f"), Precision, Vector.X);
    FString YStr = FString::Printf(TEXT("%.*f"), Precision, Vector.Y);
    FString ZStr = FString::Printf(TEXT("%.*f"), Precision, Vector.Z);
    // Build a comma-separated string for the vector components.
    FString VectorStr = XStr + TEXT(",") + YStr + TEXT(",") + ZStr;
    return AppendToStateString(BaseState, VectorStr);
}
