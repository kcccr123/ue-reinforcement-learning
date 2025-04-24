// SingleTcpConnection.h
#pragma once

#include "CoreMinimal.h"
#include "BaseTcpConnection.h"
#include "SingleTcpConnection.generated.h"

class FAcceptRunnable;
class FSocket;

/**
 * Basic Single-environment TCP connection:
 * 
 * Uses "\n" as delimiter.
 */
UCLASS()
class UERLPLUGIN_API USingleTcpConnection : public UBaseTcpConnection
{
    GENERATED_BODY()

public:
    virtual ~USingleTcpConnection() override { CloseConnection(); }

    // Listen on IP/Port, spawn acceptance thread
    virtual bool StartListening(const FString& IPAddress, int32 Port) override;

    // Accept environment connection
    virtual bool AcceptEnvConnection(FSocket* InNewSocket) override;

    // Send data to environment. Applies newline char as delimiter.
    virtual bool SendMessageEnv(const FString& Data) override;

    // Receive data from environment. Expects newline char as delimiter.
    virtual FString ReceiveMessageEnv(int32 BufSize = 1024) override;

    // Clean up
    virtual void CloseConnection() override;

    // start thread for accepting incoming connections
    virtual void StartAcceptThread() override;

    // IsConnected returns true only if AdminSocket + EnvSocket are set
    virtual bool IsConnected() const override
    {
        return (AdminSocket != nullptr && EnvSocket != nullptr);
    }

protected:
    // The environment socket
    FSocket* EnvSocket = nullptr;

    // Buffer leftover data until we see a full line (newline-delimited)
    FString PartialData;
};
