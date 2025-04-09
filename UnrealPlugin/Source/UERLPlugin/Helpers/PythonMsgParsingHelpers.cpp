#include "PythonMsgParsingHelpers.h"

int32 UPythonMsgParsingHelpers::ParseEnvId(const FString& Message)
{
    // Expected format: "ENV=2;ACT=0.10,-0.20" or "ENV=1;RESET"
    int32 EnvId = 0;
    int32 StartIdx = Message.Find(TEXT("ENV="));
    if (StartIdx != INDEX_NONE)
    {
        StartIdx += 4; // Skip "ENV="
        int32 EndIdx;
        if (Message.FindChar(';', EndIdx))
        {
            FString EnvIdStr = Message.Mid(StartIdx, EndIdx - StartIdx);
            EnvId = FCString::Atoi(*EnvIdStr);
        }
    }
    return EnvId;
}

FString UPythonMsgParsingHelpers::ParseActionString(const FString& Message)
{
    // Expected format: "ENV=x;ACT=0.10,-0.20"
    int32 ActIdx = Message.Find(TEXT("ACT="));
    if (ActIdx != INDEX_NONE)
    {
        return Message.Mid(ActIdx + 4).TrimStartAndEnd();
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
