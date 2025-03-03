#include "RLBaseBridge.h"
#include "Sockets.h"
#include "SocketSubsystem.h"
#include "Interfaces/IPv4/IPv4Address.h"
#include "Common/TcpSocketBuilder.h"
#include "HAL/PlatformProcess.h"


void URLBaseBridge::InitializeBridge()
{
    UE_LOG(LogTemp, Error, TEXT("Bridge Initialized."));
}

bool URLBaseBridge::Connect(const FString& IPAddress, int32 Port)
{
    CurrentIP = IPAddress;
    CurrentPort = Port;
    UE_LOG(LogTemp, Log, TEXT("RLBaseBridge: Starting TCP server on %s:%d"), *IPAddress, Port);

    // Get the socket subsystem
    ISocketSubsystem* SocketSubsystem = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
    if (!SocketSubsystem)
    {
        UE_LOG(LogTemp, Error, TEXT("RLBaseBridge: Socket subsystem not found!"));
        return false;
    }

    // Create a listening socket that binds to any address on the given port.
    ConnectionSocket = FTcpSocketBuilder(TEXT("RL_TcpServer"))
        .AsReusable()
        .BoundToAddress(FIPv4Address::Any)
        .BoundToPort(Port)
        .Listening(8); // Allow up to 8 pending connections
    if (!ConnectionSocket)
    {
        UE_LOG(LogTemp, Error, TEXT("RLBaseBridge: Failed to create listening socket."));
        return false;
    }

    UE_LOG(LogTemp, Log, TEXT("RLBaseBridge: Listening on port %d"), Port);

    // Wait for an incoming connection by checking the value of bHasPendingConnection.
    bool bHasPendingConnection = false;
    while (!bHasPendingConnection)
    {
        ConnectionSocket->HasPendingConnection(bHasPendingConnection);
        FPlatformProcess::Sleep(0.1f);
    }

    // Accept the incoming connection.
    TSharedRef<FInternetAddr> RemoteAddress = SocketSubsystem->CreateInternetAddr();
    FSocket* ClientSocket = ConnectionSocket->Accept(*RemoteAddress, TEXT("RL_ClientSocket"));
    if (!ClientSocket)
    {
        UE_LOG(LogTemp, Error, TEXT("RLBaseBridge: Failed to accept connection."));
        return false;
    }

    // Destroy the listening socket and use the accepted client socket.
    ConnectionSocket->Close();
    SocketSubsystem->DestroySocket(ConnectionSocket);
    ConnectionSocket = ClientSocket;

    UE_LOG(LogTemp, Log, TEXT("RLBaseBridge: Accepted connection"));
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

void URLBaseBridge::UpdateRL(float DeltaTime)
{
    UE_LOG(LogTemp, Error, TEXT("Updaing"));
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


float URLBaseBridge::CalculateReward_Implementation(bool& bDone)
{
    // Default is zero reward, not done
    bDone = false;
    return 0.0f;
}

FString URLBaseBridge::CreateStateString_Implementation()
{
    return FString();
}

void URLBaseBridge::HandleReset_Implementation()
{
    // Default empty implementation.
}

void URLBaseBridge::HandleResponseActions_Implementation(const FString& actions)
{
    // Default empty implementation.
}


void URLBaseBridge::Tick(float DeltaTime)
{
    // Call UpdateRL each frame
    UE_LOG(LogTemp, Log, TEXT("What the fuck"));
    UpdateRL(DeltaTime);
}

TStatId URLBaseBridge::GetStatId() const
{
    RETURN_QUICK_DECLARE_CYCLE_STAT(URLBaseBridge, STATGROUP_Tickables);
}