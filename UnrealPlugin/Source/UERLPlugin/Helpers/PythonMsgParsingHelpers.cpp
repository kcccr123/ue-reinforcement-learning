#include "PythonMsgParsingHelpers.h"

int32 UPythonMsgParsingHelpers::ParseEnvId(const FString& Message)
{
    TArray<FString> Parts;
    Message.ParseIntoArray(Parts, TEXT(";"), true);

    // Find the token that starts with "ENV="
    for (const FString& Part : Parts)
    {
        if (Part.StartsWith(TEXT("ENV="), ESearchCase::IgnoreCase))
        {
            FString IdStr = Part.Mid(4);
            return FCString::Atoi(*IdStr);
        }
    }

    return -1;
}


FString UPythonMsgParsingHelpers::ParseActionString(const FString& Message)
{
    TArray<FString> Parts;
    Message.ParseIntoArray(Parts, TEXT(";"), true);

    // Find the token that begins with "ACT="
    for (const FString& Part : Parts)
    {
        if (Part.StartsWith(TEXT("ACT="), ESearchCase::IgnoreCase))
        {
            return Part.Mid(4).TrimStartAndEnd();
        }
    }

    return FString();
}


TArray<float> UPythonMsgParsingHelpers::ParseActionFloatArray(const FString& ActionString)
{
    TArray<float> OutValues;
    TArray<FString> ParsedStrings;
    // Split the action string using comma as the delimiter.
    ActionString.ParseIntoArray(ParsedStrings, TEXT(","), true);
    for (const FString& Str : ParsedStrings)
    {
        float Value = FCString::Atof(*Str);
        OutValues.Add(Value);
    }
    return OutValues;
}
