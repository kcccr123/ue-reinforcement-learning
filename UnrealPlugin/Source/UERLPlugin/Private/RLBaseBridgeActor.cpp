#include "RLBaseBridgeActor.h"
#include "SocketSubsystem.h"
#include "Interfaces/IPv4/IPv4Address.h"
#include "Common/TcpSocketBuilder.h"
#include "HAL/PlatformProcess.h"

ARLBaseBridgeActor::ARLBaseBridgeActor()
{
    PrimaryActorTick.bCanEverTick = true;
    PrimaryActorTick.TickGroup = TG_PostPhysics;
}

void ARLBaseBridgeActor::BeginPlay()
{
    Super::BeginPlay();
}

void ARLBaseBridgeActor::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    if (bIsTraining)
    {
        UpdateRL(DeltaTime);
    }
}

//---------------------------------------------------------
// Functions for general RL communication
//---------------------------------------------------------
void ARLBaseBridgeActor::InitializeBridge_Implementation()
{
    UE_LOG(LogTemp, Log, TEXT("RLBaseBridgeActor: Initializing Bridge"));
}

bool ARLBaseBridgeActor::Connect(const FString& IPAddress, int32 port, int32 actionSpaceSize, int32 obsSpaceSize)
{
    CurrentIP = IPAddress;
    CurrentPort = port;
    ActionSpaceSize = actionSpaceSize;
    ObservationSpaceSize = obsSpaceSize;

    InitializeBridge();

    UE_LOG(LogTemp, Log, TEXT("RLBaseBridgeActor: Starting TCP server on %s:%d"), *IPAddress, port);

    ISocketSubsystem* SocketSubsystem = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
    if (!SocketSubsystem)
    {
        UE_LOG(LogTemp, Error, TEXT("RLBaseBridgeActor: Socket subsystem not found!"));
        return false;
    }

    // Create a listening socket that binds to any address on the given port.
    ConnectionSocket = FTcpSocketBuilder(TEXT("RL_TcpServer"))
        .AsReusable()
        .BoundToAddress(FIPv4Address::Any)
        .BoundToPort(port)
        .Listening(1);

    if (!ConnectionSocket)
    {
        UE_LOG(LogTemp, Error, TEXT("RLBaseBridgeActor: Failed to create listening socket."));
        return false;
    }

    UE_LOG(LogTemp, Log, TEXT("RLBaseBridgeActor: Listening on port %d"), port);

    // Wait for an incoming connection.
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
        UE_LOG(LogTemp, Error, TEXT("RLBaseBridgeActor: Failed to accept connection."));
        return false;
    }

    // Close the listening socket and replace it with the accepted client socket.
    ConnectionSocket->Close();
    SocketSubsystem->DestroySocket(ConnectionSocket);
    ConnectionSocket = ClientSocket;

    UE_LOG(LogTemp, Log, TEXT("RLBaseBridgeActor: Accepted connection"));

    // send the handshake to the Python environment 
    SendHandshake();

    return true;
}

void ARLBaseBridgeActor::SendHandshake_Implementation()
{
    FString HandshakeMessage = FString::Printf(TEXT("CONFIG:OBS=%d;ACT=%d"), ObservationSpaceSize, ActionSpaceSize);

    bool bSent = SendData(HandshakeMessage);
    if (bSent)
    {
        UE_LOG(LogTemp, Log, TEXT("RLBaseBridgeActor: Handshake sent: %s"), *HandshakeMessage);
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("RLBaseBridgeActor: Failed to send handshake."));
    }
}

void ARLBaseBridgeActor::Disconnect()
{
    if (ConnectionSocket)
    {
        UE_LOG(LogTemp, Log, TEXT("RLBaseBridgeActor: Disconnecting."));
        ConnectionSocket->Close();
        ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(ConnectionSocket);
        ConnectionSocket = nullptr;
    }
}

void ARLBaseBridgeActor::StartTraining()
{
    bIsTraining = true;
}

//---------------------------------------------------------
// Update Loop
//---------------------------------------------------------
void ARLBaseBridgeActor::UpdateRL(float DeltaTime)
{
    UE_LOG(LogTemp, Log, TEXT("Updaing")); // same message as old code

    if (!bIsWaitingForAction)
    {
        // ---------------------- MAKE STATE STRING ---------------------------------
        bool bDone = false;
        float Reward = CalculateReward(bDone);
        int32 DoneInt = bDone ? 1 : 0;

        // Combine both into one state string (using a delimiter, e.g., semicolon)
        FString DataToSend = FString::Printf(TEXT("%s%.2f;%d"), *CreateStateString(), Reward, DoneInt);

        // ---------------------- MAKE STATE STRING ---------------------------------
        // send data
        SendData(DataToSend);

        // receive response
        FString ActionResponse = ReceiveData();
        if (!ActionResponse.IsEmpty())
        {
            if (ActionResponse.Equals("RESET"))
            {
                // reset if simulation is done
                HandleReset();
                return;
            }

            if (ActionResponse.Equals("TRAINING_COMPLETE"))
            {
                // reset if simulation is done
                HandleReset();
                Disconnect();
                return;
            }
            // interpret response and apply given actions
            HandleResponseActions(ActionResponse);
            bIsWaitingForAction = true;
        }
    }
    else
    {
        // Wait if the character is still moving
        bIsWaitingForAction = IsActionRunning();
    }
}

//---------------------------------------------------------
// Default TCP send/recv
//---------------------------------------------------------
bool ARLBaseBridgeActor::SendData(const FString& Data)
{
    if (!ConnectionSocket)
    {
        UE_LOG(LogTemp, Error, TEXT("RLBaseBridgeActor: No connection socket available for sending."));
        return false;
    }

    // Append the "STEP" delimiter to the message.
    FString DataWithDelimiter = Data + TEXT("STEP");

    // Convert FString to UTF-8
    FTCHARToUTF8 Converter(*DataWithDelimiter);
    int32 BytesToSend = Converter.Length();
    int32 BytesSent = 0;

    bool bSuccess = ConnectionSocket->Send(reinterpret_cast<const uint8*>(Converter.Get()), BytesToSend, BytesSent);
    if (!bSuccess || BytesSent <= 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("RLBaseBridgeActor: Failed to send data."));
        return false;
    }

    UE_LOG(LogTemp, Log, TEXT("RLBaseBridgeActor: Sent data -> %s"), *DataWithDelimiter);
    return true;
}

FString ARLBaseBridgeActor::ReceiveData()
{
    if (!ConnectionSocket)
    {
        UE_LOG(LogTemp, Error, TEXT("RLBaseBridgeActor: No connection socket available for receiving."));
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
    UE_LOG(LogTemp, Log, TEXT("RLBaseBridgeActor: Received data -> %s"), *ReceivedString);
    return ReceivedString;
}

//---------------------------------------------------------
// Required Environment Overrides
//---------------------------------------------------------
float ARLBaseBridgeActor::CalculateReward_Implementation(bool& bDone)
{
    // Default is zero reward, not done
    bDone = false;
    return 0.0f;
}

FString ARLBaseBridgeActor::CreateStateString_Implementation()
{
    return FString();
}

void ARLBaseBridgeActor::HandleReset_Implementation()
{
    // Default empty implementation.
}

void ARLBaseBridgeActor::HandleResponseActions_Implementation(const FString& actions)
{
    // Default empty implementation.
}

bool ARLBaseBridgeActor::IsActionRunning_Implementation()
{
    return false;
}
