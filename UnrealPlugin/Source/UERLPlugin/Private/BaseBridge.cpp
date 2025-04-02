#include "BaseBridge.h"
#include "TcpConnection/TcpConnection.h"
#include "Inference/InferenceInterfaces/InferenceInterface.h"
#include "UERLPlugin/Helpers/BPFL_DataHelpers.h"

// -------------------------------------------------------------
//  Connection & Setup
// -------------------------------------------------------------
bool UBaseBridge::Connect_Implementation(const FString& IPAddress, int32 Port, int32 InActionSpaceSize, int32 InObservationSpaceSize)
{
    // Store user-provided details
    ActionSpaceSize = InActionSpaceSize;
    ObservationSpaceSize = InObservationSpaceSize;

    // Initialize the TCP connection if needed
    if (!TcpConnection)
    {
        TcpConnection = NewObject<UTcpConnection>(this);
    }

    // Start listening
    if (!TcpConnection->StartListening(IPAddress, Port))
    {
        UE_LOG(LogTemp, Error, TEXT("[UBaseBridge] Failed to start listening on %s:%d"), *IPAddress, Port);
        return false;
    }

    // Accept the incoming connection
    if (!TcpConnection->AcceptConnection())
    {
        UE_LOG(LogTemp, Error, TEXT("[UBaseBridge] Failed to accept connection."));
        return false;
    }

    SendHandshake();

    return true;
}

void UBaseBridge::Disconnect()
{
    if (TcpConnection)
    {
        TcpConnection->CloseConnection();
        TcpConnection = nullptr;
    }
}

void UBaseBridge::SendHandshake_Implementation()
{
    FString HandshakeMessage = FString::Printf(TEXT("CONFIG:OBS=%d;ACT=%d"), ObservationSpaceSize, ActionSpaceSize);
    SendData(HandshakeMessage);
    UE_LOG(LogTemp, Log, TEXT("[UBaseBridge] Default handshake sent: %s"), *HandshakeMessage);
}

// -------------------------------------------------------------
//  RL Modes
// -------------------------------------------------------------
void UBaseBridge::StartTraining()
{
    bIsTraining = true;
    bIsInference = false;
}

void UBaseBridge::StartInference()
{
    bIsTraining = false;
    bIsInference = true;
}

// -------------------------------------------------------------
//  Inference
// -------------------------------------------------------------
bool UBaseBridge::SetInferenceInterface(UInferenceInterface* Interface)
{
    if (Interface)
    {
        InferenceInterface = Interface;
        return true;
    }
    UE_LOG(LogTemp, Warning, TEXT("[UBaseBridge] Empty InferenceInterface ptr."));
    return false;
}

FString UBaseBridge::RunLocalModelInference(const FString& Observation)
{
    if (InferenceInterface)
    {
        // Example: parse CSV float data
        TArray<float> ParsedObs = UBPFL_DataHelpers::ParseStateString(Observation);
        return InferenceInterface->RunInference(ParsedObs);
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("[UBaseBridge] No InferenceInterface set."));
        return TEXT("");
    }
}

// -------------------------------------------------------------
//  RL Loop
// -------------------------------------------------------------
void UBaseBridge::UpdateRL_Implementation(float DeltaTime)
{
    // No default environment logic here.
    // Child classes (SingleEnvBridge, MultiEnvBridge, etc.)
    // override this to implement their step logic.
}

// -------------------------------------------------------------
//  Communication
// -------------------------------------------------------------
bool UBaseBridge::SendData(const FString& Data)
{
    if (!TcpConnection || !TcpConnection->IsConnected())
    {
        UE_LOG(LogTemp, Error, TEXT("[UBaseBridge] SendData: No valid TCP connection."));
        return false;
    }

    FString DataWithDelimiter = Data + TEXT("STEP");
    bool bResult = TcpConnection->SendMessage(DataWithDelimiter);
    if (!bResult)
    {
        UE_LOG(LogTemp, Warning, TEXT("[UBaseBridge] Failed to send data: %s"), *Data);
    }
    return bResult;
}

FString UBaseBridge::ReceiveData()
{
    if (!TcpConnection || !TcpConnection->IsConnected())
    {
        UE_LOG(LogTemp, Error, TEXT("[UBaseBridge] ReceiveData: No valid TCP connection."));
        return TEXT("");
    }
    return TcpConnection->ReceiveMessage(1024);
}

// -------------------------------------------------------------
//  FTickableGameObject
// -------------------------------------------------------------
void UBaseBridge::Tick(float DeltaTime)
{
    // Just call UpdateRL. Subclasses override UpdateRL_Implementation with their logic.
    UpdateRL(DeltaTime);
}

bool UBaseBridge::IsTickable() const
{
    return (bIsTraining && TcpConnection && TcpConnection->IsConnected()) || bIsInference;
}

TStatId UBaseBridge::GetStatId() const
{
    RETURN_QUICK_DECLARE_CYCLE_STAT(UBaseBridge, STATGROUP_Tickables);
}
