#include "TrainingBridges/MultiEnvironment/MultiEnvBridge.h"
#include "Misc/Parse.h"
#include "TcpConnection/MultiTcpConnection.h"    
#include "UERLPlugin/Helpers/PythonMsgParsingHelpers.h"
#include "HAL/PlatformProcess.h"

UBaseTcpConnection* UMultiEnvBridge::CreateTcpConnection_Implementation()
{
    // set num of environments after creation
    UMultiTcpConnection* newBridge = NewObject<UMultiTcpConnection>(this, UMultiTcpConnection::StaticClass());
    newBridge->NumEnvironments = NumEnvironments;
    return newBridge;
}

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

FString UMultiEnvBridge::BuildHandshake_Implementation()
{
    return FString::Printf(TEXT("CONFIG:OBS=%d;ACT=%d;ENV_TYPE=MULTI;ENV_COUNT=%d"),
        ObservationSpaceSize, ActionSpaceSize, NumEnvironments);
}

// -------------------------------------------------------------------------
// Training Loop
// -------------------------------------------------------------------------
void UMultiEnvBridge::UpdateRL_Implementation(float DeltaTime)
{

    if (bIsTraining) {
        // receive response
        FString PythonMessage = ReceiveData();

        if (!PythonMessage.IsEmpty())
        {

            TArray<FString> actionMsgArray;
            PythonMessage.ParseIntoArray(actionMsgArray, TEXT("||"), true);

            for (int i = 0; i < actionMsgArray.Num(); i++) {
                FString ActionString = UPythonMsgParsingHelpers::ParseActionString(actionMsgArray[i]);
                int32 EnvId = UPythonMsgParsingHelpers::ParseEnvId(actionMsgArray[i]);
                if (ActionString.Contains("RESET"))
                {
                    // reset if simulation is done
                    HandleResetForEnv(EnvId);
                    bIsActionRunning[EnvId] = false;
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

                    FString Response = FString::Printf(TEXT("OBS=%s;REW=%.2f;DONE=%d;ENV=%d"),
                        *CreateStateStringForEnv(i), Reward, DoneInt, i);
                    SendData(Response);
                    // ---------------------- MAKE STATE STRING ---------------------------------
                }

            }
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


// -------------------------------------------------------------------------
// Environment Callbacks
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
}

void UMultiEnvBridge::HandleResponseActionsForEnv_Implementation(int32 EnvId, const FString& Actions)
{
}

bool UMultiEnvBridge::IsActionRunningForEnv_Implementation(int32 EnvId)
{
    return false;
}