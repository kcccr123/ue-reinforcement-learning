// BaseBridge.h

#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "Tickable.h" // For FTickableGameObject
#include "Inference/InferenceInterfaces/InferenceInterface.h"
#include "TcpConnection/BaseTcpConnection.h"
#include "BaseBridge.generated.h"

class UBaseTcpConnection;

/**
 * An abstract base class that defines high‑level RL bridging:
 * - TCP connection management (Connect, Disconnect, etc.)
 * - RL modes (StartTraining, StartInference)
 * - Inference interface
 * - An abstract UpdateRL method that subclasses must override.
 *
 * Subclasses override UpdateRL to integrate environment callbacks.
 * Users will implement their own environment callbacks (reward, state, etc.) to
 * fit their simulation requirements.
 */
UCLASS(Abstract, Blueprintable)
class UERLPLUGIN_API UBaseBridge : public UObject, public FTickableGameObject
{
    GENERATED_BODY()

public:
    // -------------------------------------------------------------
    //  Connection & Setup
    // -------------------------------------------------------------

    /**
     * Open a TCP connection on the given IP/Port, then call SendHandshake().
     * Typically also sets ActionSpaceSize/ObservationSpaceSize internally.
     */
    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "Bridge|Connection")
    bool Connect(const FString& IPAddress, int32 Port, int32 InActionSpaceSize, int32 InObservationSpaceSize);
    virtual bool Connect_Implementation(const FString& IPAddress, int32 Port, int32 InActionSpaceSize, int32 InObservationSpaceSize);

    /**
     * Disconnect from the TCP connection (if any).
     */
    UFUNCTION(BlueprintCallable, Category = "Bridge|Connection")
    virtual void Disconnect();


    // -------------------------------------------------------------
    //  RL Modes (Training / Inference)
    // -------------------------------------------------------------

    /** Switch to training mode. Subclasses can define what 'training' means in UpdateRL. */
    UFUNCTION(BlueprintCallable, Category = "Bridge|Mode")
    virtual void StartTraining();

    /** Switch to inference mode. */
    UFUNCTION(BlueprintCallable, Category = "Bridge|Mode")
    virtual void StartInference();


    // -------------------------------------------------------------
    //  Inference 
    // -------------------------------------------------------------

    /**
     * Assign a local inference interface for performing model inference,
     * returns true if successfully set.
     */
    UFUNCTION(BlueprintCallable, Category = "Bridge|Inference")
    virtual bool SetInferenceInterface(UInferenceInterface* Interface);

    /**
     * Run embedded model inference on a given observation string
     */
    UFUNCTION(BlueprintCallable, Category = "Bridge|Inference")
    virtual FString RunLocalModelInference(const FString& Observation);


protected:
    // -------------------------------------------------------------
    //  Protected Members
    // -------------------------------------------------------------

    /** Pointer to our low-level TCP helper for sockets. */
    UPROPERTY()
    UBaseTcpConnection* TcpConnection = nullptr;

    /** Pointer to a local inference interface, if used. */
    UPROPERTY()
    UInferenceInterface* InferenceInterface = nullptr;

    /** True if in training mode. */
    bool bIsTraining = false;

    /** True if in inference mode. */
    bool bIsInference = false;

    /** Action/Observation space sizes, typically assigned on connect. */
    int32 ActionSpaceSize = 0;
    int32 ObservationSpaceSize = 0;


    // -------------------------------------------------------------
    //  RL Loop
    // -------------------------------------------------------------

    /**
     * Subclasses MUST override this. It's the main RL update each frame.
     */
    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "Bridge|Loop")
    void UpdateRL(float DeltaTime);
    virtual void UpdateRL_Implementation(float DeltaTime);


    // -------------------------------------------------------------
    //  Protected Communication
    // -------------------------------------------------------------

    /**
     * Send data across the connection.
     * Subclasses can override if desired.
     */
    UFUNCTION(BlueprintCallable, Category = "Bridge|Communication")
    virtual bool SendData(const FString& Data);

    /**
     * Receive data from the TCP connection, by default reading up to 1024 bytes.
     */
    UFUNCTION(BlueprintCallable, Category = "Bridge|Communication")
    virtual FString ReceiveData();

    /**
     * Sends a handshake message that sets up training on Python Module (subclasses may override).
     */
    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "Bridge|Connection")
    FString BuildHandshake();
    virtual FString BuildHandshake_Implementation();

    /**
     * Factory for the TCP connection object.
     * Subclasses (in C++ or Blueprint) must implement this to return
     * their desired UBaseTcpConnection subclass.
     */
    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "Bridge|Connection")
    UBaseTcpConnection* CreateTcpConnection();
    virtual UBaseTcpConnection* CreateTcpConnection_Implementation();


public:
    // -------------------------------------------------------------
    //  FTickableGameObject Interface
    // -------------------------------------------------------------
    virtual void Tick(float DeltaTime) override;
    virtual bool IsTickable() const override;
    virtual bool IsTickableWhenPaused() const override { return false; }
    virtual ETickableTickType GetTickableTickType() const override { return ETickableTickType::Always; }
    virtual TStatId GetStatId() const override;
};
