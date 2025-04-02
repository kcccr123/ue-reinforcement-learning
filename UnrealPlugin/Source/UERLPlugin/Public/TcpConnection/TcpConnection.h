
#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "Sockets.h"
#include "TcpConnection.generated.h"

/**
 * A minimal class for handling raw TCP socket operations:
 * listening, accepting connections, sending data, receiving data.
 */
UCLASS(Blueprintable)
class UERLPLUGIN_API UTcpConnection : public UObject
{
    GENERATED_BODY()

public:
    /** Start listening on IP/port. Returns true if successful. */
    UFUNCTION(BlueprintCallable, Category = "TCP")
    bool StartListening(const FString& IPAddress, int32 Port);

    /** Accept an incoming connection, replacing the listening socket with a client socket. */
    UFUNCTION(BlueprintCallable, Category = "TCP")
    bool AcceptConnection();

    /** Send UTF-8 data over the established client socket. Returns true if successful. */
    UFUNCTION(BlueprintCallable, Category = "TCP")
    bool SendMessage(const FString& Data);

    /** Receive data from the client socket (up to BufSize). Returns the string. Empty if none. */
    UFUNCTION(BlueprintCallable, Category = "TCP")
    FString ReceiveMessage(int32 BufSize = 1024);

    /** Close all sockets. */
    UFUNCTION(BlueprintCallable, Category = "TCP")
    void CloseConnection();

    /** Check if a connection is active. */
    UFUNCTION(BlueprintCallable, Category = "TCP")
    bool IsConnected() const { return ConnectionSocket != nullptr; }

private:
    /** The socket used for listening to incoming connections. */
    FSocket* ListeningSocket = nullptr;

    /** The client socket after accepting a connection. */
    FSocket* ConnectionSocket = nullptr;
};