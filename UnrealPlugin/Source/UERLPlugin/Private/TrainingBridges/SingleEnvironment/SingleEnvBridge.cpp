#include "TrainingBridges/SingleEnvironment/SingleEnvBridge.h"
#include "TcpConnection/SingleTcpConnection.h"
#include "TcpConnection/EnvMessages.h"
#include "UERLPlugin/Helpers/BPFL_DataHelpers.h"


FHandshakeMessage USingleEnvBridge::BuildHandshake()
{
    FHandshakeMessage Msg;
    Msg.env_id = "ue5_single_env";

    FAgentInfo Agent;
    Agent.id          = TCHAR_TO_UTF8(*AgentId);
    Agent.obs_shape   = { ObservationSpaceSize };
    Agent.act_shape   = { ActionSpaceSize };
    Agent.act_low     = -1.0f;
    Agent.act_high    =  1.0f;
    Agent.is_scripted = false;
    Agent.team        = "t_1";

    Msg.agents.push_back(Agent);
    return Msg;
}

UBaseTcpConnection* USingleEnvBridge::CreateTcpConnection_Implementation()
{
    return NewObject<USingleTcpConnection>(this, USingleTcpConnection::StaticClass());
}

// -------------------------------------------------------------------------
// RL Loop: UpdateRL Implementation
// -------------------------------------------------------------------------
void USingleEnvBridge::UpdateRL_Implementation(float DeltaTime)
{
    if (!bIsTraining) return;

    USingleTcpConnection* Conn = Cast<USingleTcpConnection>(TcpConnection);
    if (!Conn || !Conn->IsConnected()) return;

    const std::string AgentIdStd = TCHAR_TO_UTF8(*AgentId);

    // ---- If an action is still running from a previous tick, check completion ----
    if (bIsActionRunning)
    {
        bIsActionRunning = IsActionRunning();
        if (!bIsActionRunning)
        {
            // Action finished — collect obs/reward/done and send step_result.
            bool bDone = false;
            const float Reward = CalculateReward(bDone);

            FString ObsStr = CreateStateString();
            TArray<float> ObsArray = UBPFL_DataHelpers::ParseStateString(ObsStr);

            FAgentStepResult AgentResult;
            AgentResult.obs.assign(ObsArray.GetData(), ObsArray.GetData() + ObsArray.Num());
            AgentResult.reward = Reward;
            AgentResult.done   = bDone;

            FStepResultMessage Result;
            Result.agents[AgentIdStd] = AgentResult;
            Result.global.done        = bDone;

            Conn->SendMessageEnv(Result);
        }
        // Don't try to read new commands while an action is mid-flight.
        return;
    }

    // ---- Poll for the next command from Python (non-blocking) ----
    FClientMessage Msg;
    if (!Conn->ReceiveMessageEnv(Msg))
    {
        return; // No complete frame yet — come back next tick.
    }

    if (Msg.type == "reset")
    {
        HandleReset();

        FString ObsStr = CreateStateString();
        TArray<float> ObsArray = UBPFL_DataHelpers::ParseStateString(ObsStr);

        FAgentResetResult AgentResult;
        AgentResult.obs.assign(ObsArray.GetData(), ObsArray.GetData() + ObsArray.Num());

        FResetResultMessage Result;
        Result.agents[AgentIdStd] = AgentResult;

        Conn->SendMessageEnv(Result);
    }
    else if (Msg.type == "step")
    {
        if (Msg.actions.empty())
        {
            UE_LOG(LogTemp, Warning, TEXT("[USingleEnvBridge] Received step with no actions."));
            return;
        }

        // Look up this agent's actions; fall back to first entry if ID not matched.
        const std::vector<float>* ActionVec = nullptr;
        auto It = Msg.actions.find(AgentIdStd);
        if (It != Msg.actions.end())
        {
            ActionVec = &It->second;
        }
        else
        {
            ActionVec = &Msg.actions.begin()->second;
            UE_LOG(LogTemp, Warning, TEXT("[USingleEnvBridge] Agent ID '%s' not in step actions; using first entry."), *AgentId);
        }

        // Convert float vector to comma-separated FString for the Blueprint callback.
        FString ActionStr;
        for (int32 i = 0; i < static_cast<int32>(ActionVec->size()); ++i)
        {
            if (i > 0) ActionStr += TEXT(",");
            ActionStr += FString::SanitizeFloat((*ActionVec)[i]);
        }

        HandleResponseActions(ActionStr);
        bIsActionRunning = true;
    }
    else if (Msg.type == "close")
    {
        // Client is done with this episode/run (e.g. Coordinator probe closing,
        // or training/eval finishing). Tear down just the env socket and keep
        // listening so the next client (training after probe, or a later run)
        // can connect. Reset action state so the next client starts clean.
        UE_LOG(LogTemp, Log, TEXT("[USingleEnvBridge] Received close. Re-arming for next client."));
        bIsActionRunning = false;
        Conn->ResetEnvConnection();
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("[USingleEnvBridge] Unknown message type: %s"),
            UTF8_TO_TCHAR(Msg.type.c_str()));
    }
}

// -------------------------------------------------------------------------
// Environment Callbacks (default no-op implementations; override in Blueprint)
// -------------------------------------------------------------------------
float USingleEnvBridge::CalculateReward_Implementation(bool& bIsDone)
{
    bIsDone = false;
    return 0.f;
}

FString USingleEnvBridge::CreateStateString_Implementation()
{
    return FString();
}

void USingleEnvBridge::HandleReset_Implementation()
{
}

void USingleEnvBridge::HandleResponseActions_Implementation(const FString& Actions)
{
}

bool USingleEnvBridge::IsActionRunning_Implementation()
{
    return false;
}
