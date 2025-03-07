// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "BPFL_DataHelpers.generated.h"

/**
 * 
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
    UFUNCTION(BlueprintCallable, Category = "ActionHelpers")
    static TArray<float> ParseActionString(const FString& ActionString);
};
