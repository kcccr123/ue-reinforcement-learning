#include "MultiEnvSingleInstance/MultiEnvBridge.h"
#include "Misc/Parse.h"
#include "HAL/PlatformProcess.h"

void UMultiEnvBridge::InitializeEnvironments(int32 InNumEnvironments, bool bInInferenceMode)
{

    // If we're in inference mode, force only one environment.
    if (bInInferenceMode) {
        NumEnvironments = 1;
    }
    else {
        NumEnvironments = FMath::Max(InNumEnvironments, 1);
    }

    // Resize the arrays to match the number of environments.
    bIsActionRunning.SetNum(NumEnvironments);

    // Initialize each environment's state.
    for (int32 i = 0; i < NumEnvironments; i++)
    {
        bIsActionRunning[i] = false;
    }
}

// -------------------------------------------------------------------------
// Overridden Handshake: Inform Python this is a multi-env configuration
// -------------------------------------------------------------------------
void UMultiEnvBridge::SendHandshake_Implementation()
{
    // Here we override the base handshake to indicate multi-env specifics.
    FString HandshakeMessage = FString::Printf(TEXT("CONFIG:ENV_TYPE=MULTI;ENV_COUNT=%d;OBS=%d;ACT=%d"),
        NumEnvironments, ObservationSpaceSize, ActionSpaceSize);
    SendData(HandshakeMessage);
    UE_LOG(LogTemp, Log, TEXT("[UMultiEnvBridge] Multi-Env handshake sent: %s"), *HandshakeMessage);
}

// -------------------------------------------------------------------------
// Environment Callbacks – Default empty implementations (to be overridden by the user)
// -------------------------------------------------------------------------
FString UMultiEnvBridge::CreateStateStringForEnv_Implementation(int32 EnvId)
{
    return FString();
}

float UMultiEnvBridge::CalculateRewardForEnv_Implementation(int32 EnvId, bool& bDone)
{
    bDone = false;
    return 0.0f;
}

void UMultiEnvBridge::HandleResetForEnv_Implementation(int32 EnvId)
{
    // Empty by default.
}

void UMultiEnvBridge::HandleResponseActionsForEnv_Implementation(int32 EnvId, const FString& Actions)
{
    // Empty by default.
}

bool UMultiEnvBridge::IsActionRunningForEnv_Implementation(int32 EnvId)
{
    return false;
}

// -------------------------------------------------------------------------
// Training Loop
// -------------------------------------------------------------------------
void UMultiEnvBridge::UpdateRL_Implementation(float DeltaTime)
{

    UE_LOG(LogTemp, Log, TEXT("Updating"));

    if (bIsTraining) {
        // receive response
        FString PythonMessage = ReceiveData();

        FString StateString;

        if (!PythonMessage.IsEmpty())
        {
            
            TArray<FString> actionMsgArray;
            PythonMessage.ParseIntoArray(actionMsgArray, TEXT("||"), true);

            for (int i = 0; i < actionMsgArray.Num(); i++) {
                FString ActionString = ParseActionString(actionMsgArray[i]);
                int32 EnvId = ParseEnvId(ActionString);
                if (ActionString.Contains("RESET"))
                {
                    // reset if simulation is done
                    HandleResetForEnv(EnvId);
                }
                else {
                    // interpret response and apply given actions
                    HandleResponseActionsForEnv(EnvId, ActionString);
                    bIsActionRunning[EnvId] = true;
                }
         
            }
          
        }
        for (int i = 0; i < bIsActionRunning.Num(); i++) {
            if (bIsActionRunning[i] == true) {
                bIsActionRunning[i] = IsActionRunningForEnv(i);

                if (bIsActionRunning[i] == false) {
                    // if isActionRunning returns false, action has completed send new obs state
                    // ---------------------- MAKE STATE STRING ---------------------------------
                    bool bDone = false;
                    float Reward = CalculateRewardForEnv(i, bDone);
                    int32 DoneInt = bDone ? 1 : 0;

                    FString Response = FString::Printf(TEXT("OBS=%s;REW=%.2f;DONE=%d;ENV=%d||"),
                        *CreateStateStringForEnv(i), Reward, DoneInt, i);
                    StateString += Response;
                    // ---------------------- MAKE STATE STRING ---------------------------------
                }

            }
        }
        if (!StateString.IsEmpty()) {
            SendData(StateString);
        }
    }
    else if (bIsInference) {
        // if inference mode, run inference through loaded model instead
        // inference is tick driven rather then on demand

        if (bIsActionRunning[0] == true) {
            bIsActionRunning[0] = IsActionRunningForEnv(0);

        }
        else {
            FString ActionResponse = RunLocalModelInference(CreateStateStringForEnv(0));
            if (!ActionResponse.IsEmpty()) {
                HandleResponseActionsForEnv(0, ActionResponse);
                bIsActionRunning[0] = true;
            }
        }


    }

}

// helper functions for message parsing
int32 UMultiEnvBridge::ParseEnvId(const FString& Message) const
{
    // Expected format: "ENV=2;ACT=0.10,-0.20" or "ENV=1;RESET"
    int32 EnvId = 0;
    int32 StartIdx = Message.Find(TEXT("ENV="));
    if (StartIdx != INDEX_NONE)
    {
        StartIdx += 4; // Skip "ENV="
        int32 EndIdx;
        if (Message.FindChar(';', EndIdx))
        {
            FString EnvIdStr = Message.Mid(StartIdx, EndIdx - StartIdx);
            EnvId = FCString::Atoi(*EnvIdStr);
        }
    }
    return EnvId;
}

FString UMultiEnvBridge::ParseActionString(const FString& Message) const
{
    // Expected format: "ENV=x;ACT=0.10,-0.20"
    int32 ActIdx = Message.Find(TEXT("ACT="));
    if (ActIdx != INDEX_NONE)
    {
        return Message.Mid(ActIdx + 4).TrimStartAndEnd();
    }
    return FString();
}
