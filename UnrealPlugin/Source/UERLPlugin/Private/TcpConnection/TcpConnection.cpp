
#include "TcpConnection/TcpConnection.h"
#include "SocketSubsystem.h"
#include "Common/TcpSocketBuilder.h"
#include "HAL/PlatformProcess.h"

bool UTcpConnection::StartListening(const FString& IPAddress, int32 Port)
{
    // TODO MAKE THIS NON BLOCKING IN FUTURE
    // TLDR: SEPERATE THREADING

    // Clean up any previous sockets
    CloseConnection();

    ISocketSubsystem* SocketSubsystem = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
    if (!SocketSubsystem)
    {
        UE_LOG(LogTemp, Error, TEXT("[UTCPConnection] Socket subsystem not found!"));
        return false;
    }

    // Create a listening socket bound to the specified IP/port
    ListeningSocket = FTcpSocketBuilder(TEXT("TCP_Listener"))
        .AsReusable()
        .BoundToAddress(FIPv4Address::Any)
        .BoundToPort(Port)
        .Listening(1);

    if (!ListeningSocket)
    {
        UE_LOG(LogTemp, Error, TEXT("[UTCPConnection] Failed to create listening socket."));
        return false;
    }

    UE_LOG(LogTemp, Log, TEXT("[UTCPConnection] Listening on %s:%d"), *IPAddress, Port);
    return true;
}

bool UTcpConnection::AcceptConnection()
{
    if (!ListeningSocket)
    {
        UE_LOG(LogTemp, Error, TEXT("[UTCPConnection] No valid listening socket to accept from."));
        return false;
    }

    // Wait until there's a pending connection
    bool bHasPendingConnection = false;
    while (!bHasPendingConnection)
    {
        ListeningSocket->HasPendingConnection(bHasPendingConnection);
        FPlatformProcess::Sleep(0.1f);
    }

    // Accept the incoming connection
    ISocketSubsystem* SocketSubsystem = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
    if (!SocketSubsystem)
    {
        UE_LOG(LogTemp, Error, TEXT("[UTCPConnection] Socket subsystem not found!"));
        return false;
    }

    TSharedRef<FInternetAddr> RemoteAddress = SocketSubsystem->CreateInternetAddr();
    FSocket* ClientSocket = ListeningSocket->Accept(*RemoteAddress, TEXT("TCP_ClientSocket"));
    if (!ClientSocket)
    {
        UE_LOG(LogTemp, Error, TEXT("[UTCPConnection] Failed to accept connection."));
        return false;
    }

    // Close the listening socket; we only need the client now
    ListeningSocket->Close();
    SocketSubsystem->DestroySocket(ListeningSocket);
    ListeningSocket = nullptr;

    // Store the connected client socket
    ConnectionSocket = ClientSocket;

    UE_LOG(LogTemp, Log, TEXT("[UTCPConnection] Client connected."));
    return true;
}

bool UTcpConnection::SendMessage(const FString& Data)
{
    if (!ConnectionSocket)
    {
        UE_LOG(LogTemp, Error, TEXT("[UTCPConnection] No connection socket available for sending."));
        return false;
    }

    // Convert FString to UTF-8
    FTCHARToUTF8 Converter(*Data);
    int32 BytesToSend = Converter.Length();
    int32 BytesSent = 0;

    bool bSuccess = ConnectionSocket->Send(reinterpret_cast<const uint8*>(Converter.Get()), BytesToSend, BytesSent);
    if (!bSuccess || BytesSent <= 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("[UTCPConnection] Failed to send data."));
        return false;
    }

    UE_LOG(LogTemp, Log, TEXT("[UTCPConnection] Sent data -> %s"), *Data);
    return true;
}

FString UTcpConnection::ReceiveMessage(int32 BufSize)
{
    if (!ConnectionSocket)
    {
        // Not connected
        return TEXT("");
    }

    TArray<uint8> DataBuffer;
    DataBuffer.SetNumUninitialized(BufSize);

    int32 BytesRead = 0;
    bool bReceived = ConnectionSocket->Recv(DataBuffer.GetData(), BufSize, BytesRead);
    if (!bReceived || BytesRead <= 0)
    {
        // No data available this frame
        return TEXT("");
    }

    // Convert received bytes to FString
    FString ReceivedString = FString(UTF8_TO_TCHAR(reinterpret_cast<const char*>(DataBuffer.GetData())));
    UE_LOG(LogTemp, Log, TEXT("[UTCPConnection] Received data -> %s"), *ReceivedString);

    return ReceivedString;
}

void UTcpConnection::CloseConnection()
{
    if (ListeningSocket)
    {
        ListeningSocket->Close();
        ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(ListeningSocket);
        ListeningSocket = nullptr;
    }

    if (ConnectionSocket)
    {
        ConnectionSocket->Close();
        ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(ConnectionSocket);
        ConnectionSocket = nullptr;
    }
}