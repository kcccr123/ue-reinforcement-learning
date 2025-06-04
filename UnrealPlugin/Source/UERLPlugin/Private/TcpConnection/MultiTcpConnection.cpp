#include "TcpConnection/MultiTcpConnection.h"
#include "Common/TcpSocketBuilder.h"
#include "SocketSubsystem.h"
#include "TcpConnection/Threads/AcceptRunnable.h"
#include "Misc/ScopeLock.h"

bool UMultiTcpConnection::StartListening(const FString& IPAddress, int32 Port)
{
    CloseConnection(); // ensure we start clean

    ISocketSubsystem* SocketSubsystem = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
    if (!SocketSubsystem)
    {
        UE_LOG(LogTemp, Error, TEXT("[UMultiTcpConnection] Socket subsystem not found!"));
        return false;
    }

    {
        // Allocate arrays for env sockets + partial data
        FScopeLock Lock(&EnvSocketMutex);
        EnvSockets.SetNum(NumEnvironments);
        PartialData.SetNum(NumEnvironments);
        for (int32 i = 0; i < NumEnvironments; i++)
        {
            EnvSockets[i] = nullptr;
            PartialData[i] = TEXT("");
        }
    }

    // Build a listening socket just like USingleTcpConnection
    ListeningSocket = FTcpSocketBuilder(TEXT("MultiEnvListener"))
        .AsReusable()
        .BoundToAddress(FIPv4Address::Any) // ignoring IPAddress param, same as Single
        .BoundToPort(Port)
        .Listening(NumEnvironments + 1); // backlog for admin + multiple envs

    if (!ListeningSocket)
    {
        UE_LOG(LogTemp, Error, TEXT("[UMultiTcpConnection] Failed to create listening socket on %s:%d"), *IPAddress, Port);
        return false;
    }

    UE_LOG(LogTemp, Log, TEXT("[UMultiTcpConnection] Listening on %s:%d for up to %d environment(s)."), *IPAddress, Port, NumEnvironments);

    StartAcceptThread();
    return true;
}

bool UMultiTcpConnection::AcceptEnvConnection(FSocket* InNewSocket)
{
    if (!InNewSocket)
    {
        UE_LOG(LogTemp, Warning, TEXT("[UMultiTcpConnection] AcceptEnvConnection called with a null socket."));
        return false;
    }

    FScopeLock Lock(&EnvSocketMutex);

    // Find the first free slot in EnvSockets
    int32 FreeIndex = INDEX_NONE;
    for (int32 i = 0; i < EnvSockets.Num(); i++)
    {
        if (EnvSockets[i] == nullptr)
        {
            FreeIndex = i;
            break;
        }
    }

    if (FreeIndex == INDEX_NONE)
    {
        // Array is full
        UE_LOG(LogTemp, Warning, TEXT("[UMultiTcpConnection] EnvSockets array is full. Rejecting new connection."));
        InNewSocket->Close();
        ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(InNewSocket);
        return false;
    }

    // Assign new socket to this free slot
    EnvSockets[FreeIndex] = InNewSocket;
    UE_LOG(LogTemp, Log, TEXT("[UMultiTcpConnection] Accepted environment socket => EnvId=%d. (Array slot %d/%d filled)"),
        FreeIndex, FreeIndex + 1, NumEnvironments);

    // If that was the last free slot, we can stop accepting more
    if (AreAllEnvsAssigned() && AcceptRunnableRef.IsValid())
    {
        UE_LOG(LogTemp, Log, TEXT("[UMultiTcpConnection] All env sockets assigned. Stopping accept thread."));
        AcceptRunnableRef->Stop(); // This means no further AcceptConnection calls
    }

    return true;
}

bool UMultiTcpConnection::SendMessageEnv(const FString& Data)
{
   
    // % TODO: MIGHT WANT TO CONSIDER MAKING A NEW MESSAGE FUNCTION FOR MULTITCPCONNECTIONS THAT TAKES
    // ENV ID'S INSTEAD OF THIS GARBAGE THAT EXTRACTS THE ENVID FROM THE STRING

    // Parse "ENV=%d" from Data to figure out which environment socket to target
    int32 EnvId = ExtractEnvIdFromData(Data);
    if (EnvId < 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("[UMultiTcpConnection] SendMessageEnv: Could not parse ENV=%%d in => %s"), *Data);
        return false;
    }

    FScopeLock Lock(&EnvSocketMutex);

    if (EnvId >= EnvSockets.Num() || !EnvSockets[EnvId])
    {
        UE_LOG(LogTemp, Warning, TEXT("[UMultiTcpConnection] SendMessageEnv: EnvId=%d is out of range or not connected."), EnvId);
        return false;
    }

    FSocket* EnvSock = EnvSockets[EnvId];

    // Append "\n" delimiter
    FString DataWithStep = Data + TEXT("\n");

    FTCHARToUTF8 Converter(*DataWithStep);
    int32 BytesSent = 0;
    bool bSuccess = EnvSock->Send(
        reinterpret_cast<const uint8*>(Converter.Get()),
        Converter.Length(),
        BytesSent
    );

    if (!bSuccess || BytesSent <= 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("[UMultiTcpConnection] Failed to send data to EnvId=%d => %s"), EnvId, *DataWithStep);
        return false;
    }

    UE_LOG(LogTemp, Log, TEXT("[UMultiTcpConnection] Sent to EnvId=%d => %s"), EnvId, *DataWithStep);
    return true;
}


FString UMultiTcpConnection::ReceiveMessageEnv(int32 BufSize)
{
    // Gather new messages from all environment sockets
    TArray<FString> AllMessages;

    {
        FScopeLock Lock(&EnvSocketMutex);

        for (int32 i = 0; i < EnvSockets.Num(); i++)
        {
            FSocket* EnvSock = EnvSockets[i];
            if (!EnvSock) // not connected yet
            {
                continue;
            }
            FString line = ReadFromSocket(i, EnvSock, BufSize);
            if (!line.IsEmpty()) {
                // Env no longer added on Python side
                // EnvID now based on index of socket inside socket array
                line += FString::Printf(TEXT(";ENV=%d"), i);
                AllMessages.Add(line);
            }
      
        }
    }

    if (AllMessages.Num() == 0)
    {
        return TEXT("");
    }

    // Join them with "||" so bridging code can parse them easily.
    FString Combined = FString::Join(AllMessages, TEXT("||"));

    UE_LOG(LogTemp, Log, TEXT("[UMultiTcpConnection] Received from env(s) => %s"), *Combined);
    return Combined;
}

void UMultiTcpConnection::CloseConnection()
{
    bStopAcceptThreadRef = true;

    if (AcceptRunnableRef.IsValid())
    {
        AcceptRunnableRef->Stop();
    }
    if (AcceptThreadRef)
    {
        AcceptThreadRef->Kill(true);
        delete AcceptThreadRef;
        AcceptThreadRef = nullptr;
    }
    AcceptRunnableRef = nullptr;

    ISocketSubsystem* SocketSubsystem = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);

    // listening
    if (ListeningSocket)
    {
        ListeningSocket->Close();
        SocketSubsystem->DestroySocket(ListeningSocket);
        ListeningSocket = nullptr;
    }

    // admin
    if (AdminSocket)
    {
        AdminSocket->Close();
        SocketSubsystem->DestroySocket(AdminSocket);
        AdminSocket = nullptr;
    }

    // env
    {
        FScopeLock Lock(&EnvSocketMutex);
        for (int32 i = 0; i < EnvSockets.Num(); i++)
        {
            if (EnvSockets[i])
            {
                EnvSockets[i]->Close();
                SocketSubsystem->DestroySocket(EnvSockets[i]);
                EnvSockets[i] = nullptr;
            }
        }
        EnvSockets.Empty();
        PartialData.Empty();
    }

    UE_LOG(LogTemp, Log, TEXT("[UMultiTcpConnection] Closed sockets (admin + multi-env)."));
}

void UMultiTcpConnection::StartAcceptThread()
{
    bStopAcceptThreadRef = false;

    AcceptRunnableRef = MakeShareable(new FAcceptRunnable(this));
    AcceptThreadRef = FRunnableThread::Create(
        AcceptRunnableRef.Get(),
        TEXT("MultiEnvAcceptThread"),
        0,
        TPri_Normal
    );

    if (!AcceptThreadRef)
    {
        UE_LOG(LogTemp, Error, TEXT("[UMultiTcpConnection] Failed to start accept thread."));
    }
    else
    {
        UE_LOG(LogTemp, Log, TEXT("[UMultiTcpConnection] Accept thread started."));
    }
}

bool UMultiTcpConnection::IsConnected() const
{
    // We consider ourselves connected if admin is assigned and at all environments are connected.
    return AdminSocket && AreAllEnvsAssigned();
}

FString UMultiTcpConnection::ReadFromSocket(int32 EnvId, FSocket* EnvSocket, int32 BufSize)
{

    if (!EnvSocket)
    {
        UE_LOG(LogTemp, Error, TEXT("Bridge: No connection socket available for receiving."));
        return TEXT("");
    }

    // Read any new bytes into PartialData
    uint32 Pending = 0;
if (EnvSocket->HasPendingData(Pending) && Pending > 0)
{
    int32 ToRead = FMath::Min((int32)Pending, BufSize);

    TArray<uint8> Buffer;
    Buffer.SetNumUninitialized(ToRead + 1);          // +1 for the null byte

    int32 Read = 0;
    if (EnvSocket->Recv(Buffer.GetData(), ToRead, Read) && Read > 0)
    {
        Buffer[Read] = 0;                            // ensure null-termination

        // length-aware conversion: copies exactly “Read” bytes
        FUTF8ToTCHAR Converter(reinterpret_cast<const ANSICHAR*>(Buffer.GetData()), Read);
        PartialData[EnvId].AppendChars(Converter.Get(), Converter.Length());
    }
}

    // If we have a full line ending in '\n', extract it
    int32 NewlineIdx;
    if (PartialData[EnvId].FindChar('\n', NewlineIdx))
    {
        FString Line = PartialData[EnvId].Left(NewlineIdx);
        PartialData[EnvId] = PartialData[EnvId].Mid(NewlineIdx + 1);
        UE_LOG(LogTemp, Log, TEXT("[UMultiTcpConnection] Env %d Received: %s"), EnvId, *Line);
        return Line;
    }

    return TEXT("");
}




int32 UMultiTcpConnection::ExtractEnvIdFromData(const FString& Message) const 
{

    //  Find the "ENV=" tag
    int32 StartIdx = Message.Find(TEXT("ENV="), ESearchCase::IgnoreCase, ESearchDir::FromStart);
    if (StartIdx == INDEX_NONE) return -1;
    StartIdx += 4;

    // Look for the next ';', or if none, use the end of the string:
    int32 EndIdx = Message.Find(TEXT(";"), ESearchCase::IgnoreCase, ESearchDir::FromStart, StartIdx);
    if (EndIdx == INDEX_NONE) {
        EndIdx = Message.Len();
    }

    FString EnvIdStr = Message.Mid(StartIdx, EndIdx - StartIdx).TrimStartAndEnd();
    return FCString::Atoi(*EnvIdStr);
}




// Helper: check if all env slots are assigned
bool UMultiTcpConnection::AreAllEnvsAssigned() const
{
    // might wanna remove this later
    // this means were looping through the entire array every single tick since
    // probably really slows down performance (YIKES)
    // maybe replace in future with a counter that checks how many envs are connnected
    // or perodic heartbeats that check if envs are connected. (when recieving data)
    for (FSocket* Sock : EnvSockets)
    {
        if (!Sock)
        {
            return false;
        }
    }
    return true;
}
