#pragma once

#include "CoreMinimal.h"
#include "TcpConnection/BaseTcpConnection.h"
#include "MultiTcpConnection.generated.h"

/**
 * Multi-environment TCP connection:
 *
 * Designed specifically to work with MultiEnvBridge.
 * 
 * An array EnvSockets of size NumEnvironments stores each socket. 
 * Num enviornments must be initalized before connecting in the bridge.
 * 
 * SendMessageEnv() parses "ENV=%d" from the string to find which socket to use.
 * ReceiveMessageEnv() returns a single combined string of new messages from all envs.
 * 
 * Uses "\n" as delimiter.
 */


// % TODO: In the future, why not just use a bunch of the SingleTcpConnection classes inside MultiEnvBridge?
// This class is too coupled with MultiEnvBridge, not very versatile.
// Breaks our abstarct design if we go through with that though.

UCLASS()
class UERLPLUGIN_API UMultiTcpConnection : public UBaseTcpConnection
{
    GENERATED_BODY()

public:
    virtual ~UMultiTcpConnection() override { CloseConnection(); }

    //-------------------------------------------------------------------------
    // UBaseTcpConnection overrides
    //-------------------------------------------------------------------------

    /**
     * Start listening on IP/Port, just like USingleTcpConnection,
     * using FTcpSocketBuilder. We'll allocate EnvSockets array of size NumEnvironments.
     */
    virtual bool StartListening(const FString& IPAddress, int32 Port) override;

    /**
     * Called from AcceptConnection() after Admin is assigned. We place the new socket
     * in the first empty slot in EnvSockets. If full, close the new socket.
     */
    virtual bool AcceptEnvConnection(FSocket* InNewSocket) override;

    /**
     * Instead of needing an EnvId param, we parse "ENV=%d" from the Data string
     * and send the message to that environment index. Applies newline char as delimiter.
     */
    virtual bool SendMessageEnv(const FString& Data) override;

    /**
     * Gather new messages from all environment sockets. 
     * Parses partial messages into buffer.
     * Expects newline char as delimiter.
     */
    virtual FString ReceiveMessageEnv(int32 BufSize = 1024) override;

    /**
     * Close acceptance thread, plus admin and environment sockets.
     */
    virtual void CloseConnection() override;

    /**
     * Spawns a thread that calls AcceptConnection() repeatedly
     * for admin + multiple envs. We stop it once all envs are filled (or at shutdown).
     */
    virtual void StartAcceptThread() override;

    /**
     * Return true if AdminSocket is assigned and all envs are connceted
     */
    virtual bool IsConnected() const override;

    //-------------------------------------------------------------------------
    // Configuration
    //-------------------------------------------------------------------------

    /**
     * Number of environment sockets to accept. We'll store them in EnvSockets[0..NumEnvironments-1].
     */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MultiEnv")
    int32 NumEnvironments = 1;

protected:
    //-------------------------------------------------------------------------
    // Internal data
    //-------------------------------------------------------------------------

    /** Protect EnvSockets and PartialData arrays. */
    FCriticalSection EnvSocketMutex;

    /**
     * Environment sockets array. If EnvSockets[i] is null, that slot is free.
     * We fill them in AcceptEnvConnection until all are assigned.
     */
    TArray<FSocket*> EnvSockets;

    /**
     * Partial data storage for each environment. If we receive a partial message that
     * doesn't end with a newline char, we store the leftover here and combine it on the next read.
     */
    TArray<FString> PartialData;

    //-------------------------------------------------------------------------
    // Helper Methods
    //-------------------------------------------------------------------------

    /**
     * Reads up to BufSize from EnvSocket, merges with leftover partial data,
     * splits by "\n", returns completed messages for that environment.
     */
    FString ReadFromSocket(int32 EnvId, FSocket* EnvSocket, int32 BufSize);


    /**
     * Extract "ENV=%d" from the provided Data string. If not found or invalid, returns -1.
     */
    int32 ExtractEnvIdFromData(const FString& Message) const;

    /**
     * Checks if all EnvSockets[i] are assigned (none are null).
     */
    bool AreAllEnvsAssigned() const;
};
