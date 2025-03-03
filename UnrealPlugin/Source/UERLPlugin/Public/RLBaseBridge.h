#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "Tickable.h"
#include "RLBaseBridge.generated.h"

/**
 * Base class for RL communication using TCP.
 * This class provides a default implementation for connecting and disconnecting via TCP.
 */
UCLASS(Abstract, Blueprintable)
class UERLPLUGIN_API URLBaseBridge : public UObject, public FTickableGameObject
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

    // abstract implementation for calculating reward (implement in sub classes)
    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "RLBridge")
    float CalculateReward(bool& bDone);

    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "RLBridge")
    FString CreateStateString();

    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "RLBridge")
    void HandleReset();

    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "RLBridge")
    void HandleResponseActions(const FString& actions);


    // Default implementation for receiving data over TCP.
    virtual FString ReceiveData();

    // ptr to TCP socket
    FSocket* ConnectionSocket = nullptr;

    // socket info (set on connection)
    FString CurrentIP;
    int32 CurrentPort;


    //---------------------------------------------------------
    // FTickableGameObject Interface:
    //---------------------------------------------------------
public:
    // Called every frame by the engine.
    virtual void Tick(float DeltaTime) override;

    // Return true so that this object is tickable.
    virtual bool IsTickable() const override { return true; }

    // Return false to disable ticking when the game is paused.
    virtual bool IsTickableWhenPaused() const override { return false; }

    // Specifies that this object should always tick.
    virtual ETickableTickType GetTickableTickType() const override { return ETickableTickType::Always; }

    // Returns a stat ID for profiling purposes.
    virtual TStatId GetStatId() const override;
};
