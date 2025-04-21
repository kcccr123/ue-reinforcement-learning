#include "TrainingBridges/SingleEnvironment/SingleEnvBridge.h"
#include "TcpConnection/SingleTcpConnection.h"     
#include "UERLPlugin/Helpers/PythonMsgParsingHelpers.h"
#include "UERLPlugin/Helpers/BPFL_DataHelpers.h"         


FString USingleEnvBridge::BuildHandshake_Implementation()
{
    return FString::Printf(TEXT("CONFIG:OBS=%d;ACT=%d;ENV_TYPE=SINGLE"),
        ObservationSpaceSize, ActionSpaceSize);
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

    if (bIsTraining) {
        // receive response
        FString PythonMessage = ReceiveData();

        // if command recieved
        if (!PythonMessage.IsEmpty())
        {
            FString ActionString = UPythonMsgParsingHelpers::ParseActionString(PythonMessage);
            if (ActionString.Contains("RESET"))
            {
                // reset if simulation is done
                HandleReset();
                bIsActionRunning = false;
            }
            else {
                // interpret response and apply given actions
                HandleResponseActions(ActionString);

                // Set action running to true
                bIsActionRunning = true;
            }
        }

        // check if an action is running
        if (bIsActionRunning == true) {
            bIsActionRunning = IsActionRunning();
            // if action has concluded send state data as a result of the action
            if (bIsActionRunning == false) {
                bool bDone = false;
                float Reward = CalculateReward(bDone);
                int32 DoneInt = bDone ? 1 : 0;

                FString ObsStr = CreateStateString();
                FString DataToSend = FString::Printf(TEXT("OBS=%s;REW=%.2f;DONE=%d"), *ObsStr, Reward, DoneInt);

                // Send environment observation, reward, done to Python
                SendData(DataToSend);
            }
        }

    }
    else if (bIsInference) {
        // if inference mode, run inference through loaded model instead
        // inference is tick driven rather then on demand

        if (bIsActionRunning == true) {
            bIsActionRunning = IsActionRunning();

        }
        else {
            FString ActionResponse = RunLocalModelInference(CreateStateString());
            if (!ActionResponse.IsEmpty()) {
                HandleResponseActions(ActionResponse);
                bIsActionRunning = true;
            }
        }


    }

}

// -------------------------------------------------------------------------
// Environment Callbacks 
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
