#include "RLBaseBridge.h"
#include "Sockets.h"
#include "SocketSubsystem.h"
#include "Interfaces/IPv4/IPv4Address.h"
#include "Common/TcpSocketBuilder.h"
#include "HAL/PlatformProcess.h"


bool URLBaseBridge::Connect(const FString& IPAddress, int32 Port)
{
    CurrentIP = IPAddress;
    CurrentPort = Port;
    UE_LOG(LogTemp, Log, TEXT("RLBaseBridge: Connecting to %s:%d"), *IPAddress, Port);

    // Get the socket subsystem
    ISocketSubsystem* SocketSubsystem = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
    if (!SocketSubsystem)
    {
        UE_LOG(LogTemp, Error, TEXT("RLBaseBridge: Socket subsystem not found!"));
        return false;
    }

    // Create the TCP socket using the TcpSocketBuilder for ease
    ConnectionSocket = FTcpSocketBuilder(TEXT("RL_TcpSocket"))
        .AsReusable()
        .AsBlocking(); // For now, blocking calls are acceptable for our prototype

    if (!ConnectionSocket)
    {
        UE_LOG(LogTemp, Error, TEXT("RLBaseBridge: Failed to create TCP socket."));
        return false;
    }

    // Resolve IP address and set up the remote endpoint
    FIPv4Address IP;
    FIPv4Address::Parse(IPAddress, IP);
    TSharedRef<FInternetAddr> RemoteAddress = SocketSubsystem->CreateInternetAddr();
    RemoteAddress->SetIp(IP.Value);
    RemoteAddress->SetPort(Port);

    // Attempt to connect
    bool bConnected = ConnectionSocket->Connect(*RemoteAddress);
    if (!bConnected)
    {
        UE_LOG(LogTemp, Error, TEXT("RLBaseBridge: Connection failed to %s:%d"), *IPAddress, Port);
        ConnectionSocket->Close();
        ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(ConnectionSocket);
        ConnectionSocket = nullptr;
        return false;
    }

    UE_LOG(LogTemp, Log, TEXT("RLBaseBridge: Successfully connected to %s:%d"), *IPAddress, Port);
    return true;
}

void URLBaseBridge::Disconnect()
{
    if (ConnectionSocket)
    {
        UE_LOG(LogTemp, Log, TEXT("RLBaseBridge: Disconnecting."));
        ConnectionSocket->Close();
        ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(ConnectionSocket);
        ConnectionSocket = nullptr;
    }
}

bool URLBaseBridge::SendData(const FString& Data)
{
    if (!ConnectionSocket)
    {
        UE_LOG(LogTemp, Error, TEXT("RLBaseBridge: No connection socket available for sending."));
        return false;
    }

    // Convert FString to UTF-8
    FTCHARToUTF8 Converter(*Data);
    int32 BytesToSend = Converter.Length();
    int32 BytesSent = 0;

    bool bSuccess = ConnectionSocket->Send(reinterpret_cast<const uint8*>(Converter.Get()), BytesToSend, BytesSent);
    if (!bSuccess || BytesSent <= 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("RLBaseBridge: Failed to send data."));
        return false;
    }

    UE_LOG(LogTemp, Log, TEXT("RLBaseBridge: Sent data -> %s"), *Data);
    return true;
}

FString URLBaseBridge::ReceiveData()
{
    if (!ConnectionSocket)
    {
        UE_LOG(LogTemp, Error, TEXT("RLBaseBridge: No connection socket available for receiving."));
        return TEXT("");
    }

    // Buffer for receiving data
    uint8 DataBuffer[1024];
    FMemory::Memset(DataBuffer, 0, 1024);
    int32 BytesRead = 0;

    bool bReceived = ConnectionSocket->Recv(DataBuffer, 1024, BytesRead);
    if (!bReceived || BytesRead <= 0)
    {
        // If no data is received, return empty string
        return TEXT("");
    }

    // Convert received bytes back to FString (assuming UTF-8)
    FString ReceivedString = FString(UTF8_TO_TCHAR(reinterpret_cast<const char*>(DataBuffer)));
    UE_LOG(LogTemp, Log, TEXT("RLBaseBridge: Received data -> %s"), *ReceivedString);
    return ReceivedString;
}
