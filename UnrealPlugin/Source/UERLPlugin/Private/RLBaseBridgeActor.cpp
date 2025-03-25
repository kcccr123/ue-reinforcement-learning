#include "RLBaseBridgeActor.h"
#include "SocketSubsystem.h"
#include "Interfaces/IPv4/IPv4Address.h"
#include "Common/TcpSocketBuilder.h"
#include "HAL/PlatformProcess.h"
#include "UERLPlugin/Helpers/BPFL_DataHelpers.h"

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

    if ((bIsTraining || bIsInference) && ConnectionSocket)
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

    bool bHasPendingConnection = false;
    while (!bHasPendingConnection)
    {
        ConnectionSocket->HasPendingConnection(bHasPendingConnection);
        FPlatformProcess::Sleep(0.1f);
    }

    TSharedRef<FInternetAddr> RemoteAddress = SocketSubsystem->CreateInternetAddr();
    FSocket* ClientSocket = ConnectionSocket->Accept(*RemoteAddress, TEXT("RL_ClientSocket"));
    if (!ClientSocket)
    {
        UE_LOG(LogTemp, Error, TEXT("RLBaseBridgeActor: Failed to accept connection."));
        return false;
    }

    ConnectionSocket->Close();
    SocketSubsystem->DestroySocket(ConnectionSocket);
    ConnectionSocket = ClientSocket;

    UE_LOG(LogTemp, Log, TEXT("RLBaseBridgeActor: Accepted connection"));

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
    bIsInference = false;
}

bool ARLBaseBridgeActor::SetInferenceInterface(UInferenceInterface* Interface)
{
    if (Interface) {
        InferenceInterface = Interface;
        return true;
    }
    else {
        UE_LOG(LogTemp, Warning, TEXT("RLBaseBridgeActor: Empty InferenceInterface ptr."));
        return false;
    }
}

void ARLBaseBridgeActor::StartInference()
{
    bIsTraining = false;
    bIsInference = true;
}

FString ARLBaseBridgeActor::RunLocalModelInference(const FString& Observation)
{
    if (InferenceInterface) {
        return InferenceInterface->RunInference(UBPFL_DataHelpers::ParseStateString(Observation));
    }
    else {
        UE_LOG(LogTemp, Warning, TEXT("RLBaseBridgeActor: Empty InferenceInterface ptr."));
        return "";
    }
}

//---------------------------------------------------------
// Update Loop
//---------------------------------------------------------
void ARLBaseBridgeActor::UpdateRL(float DeltaTime)
{
    UE_LOG(LogTemp, Log, TEXT("Updaing"));

    if (!bIsWaitingForAction)
    {
        if (bIsTraining) {
            if (!bIsWaitingForPythonResp) {
                bool bDone = false;
                float Reward = CalculateReward(bDone);
                int32 DoneInt = bDone ? 1 : 0;
                FString DataToSend = FString::Printf(TEXT("%s%.2f;%d"), *CreateStateString(), Reward, DoneInt);
                SendData(DataToSend);
                bIsWaitingForPythonResp = true;
            }

            FString ActionResponse = ReceiveData();
            if (!ActionResponse.IsEmpty())
            {
                bIsWaitingForPythonResp = false;
                if (ActionResponse.Equals("RESET"))
                {
                    HandleReset();
                    return;
                }

                if (ActionResponse.Equals("TRAINING_COMPLETE"))
                {
                    HandleReset();
                    Disconnect();
                    return;
                }

                HandleResponseActions(ActionResponse);
                bIsWaitingForAction = true;
            }
            else {
                bIsWaitingForPythonResp = true;
            }
        }

        if (bIsInference) {
            FString ActionResponse = RunLocalModelInference(CreateStateString());
            if (!ActionResponse.IsEmpty()) {
                HandleResponseActions(ActionResponse);
                bIsWaitingForAction = true;
            }
        }
    }
    else
    {
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

    FString DataWithDelimiter = Data + TEXT("STEP");

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

    uint8 DataBuffer[1024];
    FMemory::Memset(DataBuffer, 0, 1024);
    int32 BytesRead = 0;

    bool bReceived = ConnectionSocket->Recv(DataBuffer, 1024, BytesRead);
    if (!bReceived || BytesRead <= 0)
    {
        return TEXT("");
    }

    FString ReceivedString = FString(UTF8_TO_TCHAR(reinterpret_cast<const char*>(DataBuffer)));
    UE_LOG(LogTemp, Log, TEXT("RLBaseBridgeActor: Received data -> %s"), *ReceivedString);
    return ReceivedString;
}

//---------------------------------------------------------
// Required Environment Overrides
//---------------------------------------------------------
float ARLBaseBridgeActor::CalculateReward_Implementation(bool& bDone)
{
    bDone = false;
    return 0.0f;
}

FString ARLBaseBridgeActor::CreateStateString_Implementation()
{
    return FString();
}

void ARLBaseBridgeActor::HandleReset_Implementation()
{
}

void ARLBaseBridgeActor::HandleResponseActions_Implementation(const FString& actions)
{
}

bool ARLBaseBridgeActor::IsActionRunning_Implementation()
{
    return false;
}
