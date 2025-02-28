#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "RLBaseBridge.generated.h"

/**
 * Base class for RL communication using TCP.
 * This class provides a default implementation for connecting and disconnecting via TCP.
 */
UCLASS(Abstract, Blueprintable)
class UERLPLUGIN_API URLBaseBridge : public UObject
{
    GENERATED_BODY()

public:
    // initalization function that can be implemented for special behaviour in derived classes
    UFUNCTION(BlueprintCallable, Category = "RLBridge")
    virtual void InitializeBridge();

    // connect to tcp socket
    UFUNCTION(BlueprintCallable, Category = "RLBridge")
    virtual bool Connect(const FString& IPAddress, int32 Port);

    // disconnect from TCP
    UFUNCTION(BlueprintCallable, Category = "RLBridge")
    virtual void Disconnect();

    // update loop for RL. Implemented by derived classes. Should be called every tick
    UFUNCTION(BlueprintCallable, Category = "RLBridge")
    virtual void UpdateRL(float DeltaTime);

protected:
    // Default implementation for sending data over TCP.
    virtual bool SendData(const FString& Data);

    // Default implementation for sending data over TCP.
    virtual float CalculateReward();

    // Default implementation for receiving data over TCP.
    virtual FString ReceiveData();

    // ptr to TCP socket
    FSocket* ConnectionSocket = nullptr;

    // socket info (set on connection)
    FString CurrentIP;
    int32 CurrentPort;
};
