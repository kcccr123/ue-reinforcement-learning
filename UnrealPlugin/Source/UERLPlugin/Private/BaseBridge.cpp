// BaseBridge.cpp

#include "BaseBridge.h"
#include "UERLPlugin/Helpers/BPFL_DataHelpers.h"

bool UBaseBridge::Connect_Implementation(const FString& IPAddress, int32 Port, int32 InActionSpaceSize, int32 InObservationSpaceSize)
{
    ActionSpaceSize = InActionSpaceSize;
    ObservationSpaceSize = InObservationSpaceSize;

    if (!TcpConnection)
    {
        TcpConnection = CreateTcpConnection();
        if (!TcpConnection)
        {
            UE_LOG(LogTemp, Error, TEXT("[UBaseBridge] CreateTcpConnection returned null. Please override CreateTcpConnection in C++ or Blueprint."));
            return false;
        }
    }

    if (!TcpConnection->StartListening(IPAddress, Port))
    {
        UE_LOG(LogTemp, Error, TEXT("[UBaseBridge] Failed to start listening on %s:%d"), *IPAddress, Port);
        return false;
    }

    if (!TcpConnection->IsConnected())
    {
        UE_LOG(LogTemp, Error, TEXT("[UBaseBridge] No client connected after StartListening."));
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
    const FString Msg = FString::Printf(TEXT("CONFIG:OBS=%d;ACT=%d"), ObservationSpaceSize, ActionSpaceSize);
    SendData(Msg);
    UE_LOG(LogTemp, Log, TEXT("[UBaseBridge] Handshake sent: %s"), *Msg);
}

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
    if (!InferenceInterface)
    {
        UE_LOG(LogTemp, Warning, TEXT("[UBaseBridge] No InferenceInterface set."));
        return TEXT("");
    }
    TArray<float> Parsed = UBPFL_DataHelpers::ParseStateString(Observation);
    return InferenceInterface->RunInference(Parsed);
}

void UBaseBridge::UpdateRL_Implementation(float)
{
    // Must be overridden by subclass.
}

bool UBaseBridge::SendData(const FString& Data)
{
    if (!TcpConnection || !TcpConnection->IsConnected())
    {
        UE_LOG(LogTemp, Error, TEXT("[UBaseBridge] SendData: No valid TCP connection."));
        return false;
    }
    return TcpConnection->SendMessageEnv(Data + TEXT("STEP"));
}

FString UBaseBridge::ReceiveData()
{
    if (!TcpConnection || !TcpConnection->IsConnected())
    {
        UE_LOG(LogTemp, Error, TEXT("[UBaseBridge] ReceiveData: No valid TCP connection."));
        return TEXT("");
    }
    return TcpConnection->ReceiveMessageEnv(1024);
}

UBaseTcpConnection* UBaseBridge::CreateTcpConnection_Implementation()
{
    // No default implementation; must be provided by subclass.
    return nullptr;
}

void UBaseBridge::Tick(float DeltaTime)
{
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
