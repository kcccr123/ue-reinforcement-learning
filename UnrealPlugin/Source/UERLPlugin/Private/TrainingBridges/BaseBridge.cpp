
#include "TrainingBridges/BaseBridge.h"
#include "UERLPlugin/Helpers/BPFL_DataHelpers.h"
#include "TcpConnection/SingleTcpConnection.h"

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

    // For single-socket connections, pre-load the handshake so AcceptEnvConnection
    // can fire it immediately when Python connects (before any tick runs).
    if (USingleTcpConnection* Single = Cast<USingleTcpConnection>(TcpConnection))
    {
        Single->SetHandshakeMsg(BuildHandshake());
    }

    if (!TcpConnection->StartListening(IPAddress, Port))
    {
        UE_LOG(LogTemp, Error, TEXT("[UBaseBridge] Failed to start listening on %s:%d"), *IPAddress, Port);
        return false;
    }

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

FHandshakeMessage UBaseBridge::BuildHandshake()
{
    FHandshakeMessage Msg;
    Msg.env_id = "base_env";

    FAgentInfo Agent;
    Agent.id          = "agent_1";
    Agent.obs_shape   = { ObservationSpaceSize };
    Agent.act_shape   = { ActionSpaceSize };
    Agent.act_low     = -1.0f;
    Agent.act_high    =  1.0f;
    Agent.is_scripted = false;
    Agent.team        = "t_1";

    Msg.agents.push_back(Agent);
    return Msg;
}

bool UBaseBridge::SendHandshake()
{
    if (!TcpConnection || !TcpConnection->IsConnected())
    {
        UE_LOG(LogTemp, Warning, TEXT("[UBaseBridge] SendHandshake: no connected TCP connection."));
        return false;
    }

    USingleTcpConnection* Single = Cast<USingleTcpConnection>(TcpConnection);
    if (!Single)
    {
        UE_LOG(LogTemp, Warning, TEXT("[UBaseBridge] SendHandshake: connection is not a USingleTcpConnection."));
        return false;
    }

    const FHandshakeMessage Msg = BuildHandshake();
    return Single->SendMessageEnv(Msg);
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
    return TcpConnection->SendMessageEnv(Data);
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
    // Flush a queued handshake on the game thread (set by the accept thread when
    // the env socket connects). Keeps all socket I/O on the game thread.
    if (USingleTcpConnection* Single = Cast<USingleTcpConnection>(TcpConnection))
    {
        Single->FlushPendingHandshake();
    }

    UpdateRL(DeltaTime);
}

bool UBaseBridge::IsTickable() const
{
    // Tick once a connection is established (not just while training) so the
    // handshake can be flushed to a probing client before StartTraining() is
    // called. UpdateRL_Implementation early-returns when not in training mode.
    return (TcpConnection && TcpConnection->IsConnected()) || bIsInference;
}

TStatId UBaseBridge::GetStatId() const
{
    RETURN_QUICK_DECLARE_CYCLE_STAT(UBaseBridge, STATGROUP_Tickables);
}

