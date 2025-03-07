// Fill out your copyright notice in the Description page of Project Settings.


#include "BPFL_DataHelpers.h"

TArray<float> UBPFL_DataHelpers::ParseActionString(const FString& ActionString)
{
    TArray<float> ActionValues;

    TArray<FString> ParsedStrings;
    ActionString.ParseIntoArray(ParsedStrings, TEXT(","), true);
    for (const FString& Str : ParsedStrings)
    {
        float Value = FCString::Atof(*Str);
        ActionValues.Add(Value);
    }

    return ActionValues;
}
