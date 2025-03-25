#include "RLBaseBridge.h"
#include "Sockets.h"
#include "SocketSubsystem.h"
#include "Interfaces/IPv4/IPv4Address.h"
#include "Common/TcpSocketBuilder.h"
#include "UERLPlugin/Helpers/BPFL_DataHelpers.h" 
#include "HAL/PlatformProcess.h"

void URLBaseBridge::InitializeBridge_Implementation()
{
    UE_LOG(LogTemp, Log, TEXT("RLBaseBridge: Initializing Bridge"));
}

bool URLBaseBridge::Connect(const FString& IPAddress, int32 port, int32 actionSpaceSize, int32 obsSpaceSize)
{
    // Optionally, store the sizes in member variables if needed for handshake.
    // For this example, we'll pass them directly in the handshake.
    CurrentIP = IPAddress;
    CurrentPort = port;
    ActionSpaceSize = actionSpaceSize;
    ObservationSpaceSize = obsSpaceSize;

    InitializeBridge();
        
    UE_LOG(LogTemp, Log, TEXT("RLBaseBridge: Starting TCP server on %s:%d"), *IPAddress, port);

    // Get the socket subsystem.
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
        .BoundToPort(port)
        .Listening(1);
    if (!ConnectionSocket)
    {
        UE_LOG(LogTemp, Error, TEXT("RLBaseBridge: Failed to create listening socket."));
        return false;
    }

    UE_LOG(LogTemp, Log, TEXT("RLBaseBridge: Listening on port %d"), port);

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
        UE_LOG(LogTemp, Error, TEXT("RLBaseBridge: Failed to accept connection."));
        return false;
    }

    // Close the listening socket and replace it with the accepted client socket.
    ConnectionSocket->Close();
    SocketSubsystem->DestroySocket(ConnectionSocket);
    ConnectionSocket = ClientSocket;

    UE_LOG(LogTemp, Log, TEXT("RLBaseBridge: Accepted connection"));



    // send the handshake to the Python environment 
    SendHandshake();

    return true;
}


void URLBaseBridge::SendHandshake_Implementation()
{
    FString HandshakeMessage = FString::Printf(TEXT("CONFIG:OBS=%d;ACT=%d"), ObservationSpaceSize, ActionSpaceSize);

    bool bSent = SendData(HandshakeMessage);
    if (bSent)
    {
        UE_LOG(LogTemp, Log, TEXT("RLBaseBridge: Handshake sent: %s"), *HandshakeMessage);
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("RLBaseBridge: Failed to send handshake."));
    }
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

void URLBaseBridge::StartTraining()
{
    bIsTraining = true;
    bIsInference = false;
}

bool URLBaseBridge::SetInferenceInterface(UInferenceInterface* Interface)
{
    if (Interface) {
        InferenceInterface = Interface;
        return true;
    }
    else {
        UE_LOG(LogTemp, Warning, TEXT("RLBaseBridge: Empty InferenceInterface ptr."));
        return false;
    }
    
}


void URLBaseBridge::StartInference()
{
    bIsTraining = false;
    bIsInference = true;
}


FString URLBaseBridge::RunLocalModelInference(const FString& Observation)
{
    if (InferenceInterface) {
        return InferenceInterface->RunInference(UBPFL_DataHelpers::ParseStateString(Observation));
    }
    else {
        UE_LOG(LogTemp, Warning, TEXT("RLBaseBridge: Empty InferenceInterface ptr."));
        return "";
    }
    
}

void URLBaseBridge::UpdateRL(float DeltaTime)
{
    UE_LOG(LogTemp, Log, TEXT("Updaing"));

    if (!bIsWaitingForAction)
    {
        if (bIsTraining) {
            if (!bIsWaitingForPythonResp) {
                // ---------------------- MAKE STATE STRING ---------------------------------
                bool bDone = false;
                float Reward = CalculateReward(bDone);
                int32 DoneInt = bDone ? 1 : 0;

                // Combine both into one state string (using a delimiter, e.g., semicolon)
                FString DataToSend = FString::Printf(TEXT("%s%.2f;%d"), *CreateStateString(), Reward, DoneInt);

                // ---------------------- MAKE STATE STRING ---------------------------------
                // send data
                SendData(DataToSend);
                bIsWaitingForPythonResp = true;
            }


            // receive response
            FString ActionResponse = ReceiveData();
            if (!ActionResponse.IsEmpty())
            {
                bIsWaitingForPythonResp = false;
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
            else {
                bIsWaitingForPythonResp = true;
            }
        }
        if (bIsInference) {
            // if inference mode, run inference through loaded model instead
            FString ActionResponse = RunLocalModelInference(CreateStateString());
            if (!ActionResponse.IsEmpty()) {
                HandleResponseActions(ActionResponse);
                bIsWaitingForAction = true;
            }
            

        }
       
    }
    else
    {
        // Wait if the character is still moving
        bIsWaitingForAction = IsActionRunning();
    }
}

bool URLBaseBridge::SendData(const FString& Data)
{
    if (!ConnectionSocket)
    {
        UE_LOG(LogTemp, Error, TEXT("RLBaseBridge: No connection socket available for sending."));
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
        UE_LOG(LogTemp, Warning, TEXT("RLBaseBridge: Failed to send data."));
        return false;
    }

    UE_LOG(LogTemp, Log, TEXT("RLBaseBridge: Sent data -> %s"), *DataWithDelimiter);
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


bool URLBaseBridge::IsActionRunning_Implementation()
{
    return false;
}

void URLBaseBridge::Tick(float DeltaTime)
{ 
    
    UpdateRL(DeltaTime);

}

bool URLBaseBridge::IsTickable() const
{

    return (bIsTraining && ConnectionSocket) || bIsInference;
}



TStatId URLBaseBridge::GetStatId() const
{
    RETURN_QUICK_DECLARE_CYCLE_STAT(URLBaseBridge, STATGROUP_Tickables);
}