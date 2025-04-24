#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "Sockets.h"
#include "BaseTcpConnection.generated.h"

class FAcceptRunnable;

/**
 * Abstract base class for framework TCP connection operations.
 */
UCLASS(Abstract)
class UERLPLUGIN_API UBaseTcpConnection : public UObject
{
    GENERATED_BODY()

public:
    virtual ~UBaseTcpConnection() override = default;

    //--------------------------------------------------------------------------
    // Interface methods
    //--------------------------------------------------------------------------

    /**
     * Start listening on the specified IP and port, and typically spawn
     * a dedicated thread to handle acceptance (non-blocking).
     */
    virtual bool StartListening(const FString& IPAddress, int32 Port) PURE_VIRTUAL(UBaseTcpConnection::StartListening, return false;);


    /**
     * Accept an incoming connection from the listening socket.
     *
     * We assume first connection is always meant to be admin socket and fill admin socket first
     * Then, when admin socket is full we fill enviornment sockets
     */
    virtual bool AcceptConnection();

    /**
    * Subclasses must implement environment acceptance (single env, multi env, etc.).
    */
    virtual bool AcceptEnvConnection(FSocket* InNewSocket) PURE_VIRTUAL(UBaseTcpConnection::AcceptEnvConnection, return false;);

    /** Closes sockets, stops threads, and cleans up resources. */
    virtual void CloseConnection() PURE_VIRTUAL(UBaseTcpConnection::CloseConnection, );

    /** Sets handshake message */
    void SetHandshake(const FString& InHandshakeMsg);

    // Get listening socket
    FSocket* GetListeningSocket();

    //--------------------------------------------------------------------------
    // Admin vs. Environment messaging
    //--------------------------------------------------------------------------
    /**
     * Sends a UTF-8 string to the admin socket.
     * Returns false if admin is not connected or if sending fails.
     */
    virtual bool SendMessageAdmin(const FString& Data);

    /**
     * Sends a UTF-8 string to environment socket(s).
     * Subclass must define whether it's single or multiple env sockets.
     */
    virtual bool SendMessageEnv(const FString& Data) PURE_VIRTUAL(UBaseTcpConnection::SendMessageEnv, return false;);

    /**
     * Receives a UTF-8 string from the admin socket, returns empty if none is pending.
     * Applies newline char as delimiter.
     */
    virtual FString ReceiveMessageAdmin(int32 BufSize = 1024);

    /**
     * Receives a UTF-8 string from environment socket(s).
     * Subclass must define logic (single, multi, round-robin, etc.).
     * Expects newline char as delimiter.
     */
    virtual FString ReceiveMessageEnv(int32 BufSize = 1024) PURE_VIRTUAL(UBaseTcpConnection::ReceiveMessageEnv, return TEXT(""););

    /**
     * Checks if admin socket and enviornment sockets are set and ready for training loop logic
     */
    virtual bool IsConnected() const PURE_VIRTUAL(UBaseTcpConnection::IsConnected, return false;);

  
protected:

    // Handshakestring
    FString HandshakeMessage = "";

    // Main listening socket (admin arrives first, then environment).
    FSocket* ListeningSocket = nullptr;
    // The first accepted socket is the admin client.
    FSocket* AdminSocket = nullptr;

    // Accept thread
    FRunnableThread* AcceptThreadRef = nullptr;
    TSharedPtr<FAcceptRunnable> AcceptRunnableRef = nullptr;
    bool bStopAcceptThreadRef = false;

    // Sends handshake message
    void SendHandshake();

    /** Spawn an acceptance thread. */
    virtual void StartAcceptThread() PURE_VIRTUAL(UBaseTcpConnection::StartAcceptThread, );

};
