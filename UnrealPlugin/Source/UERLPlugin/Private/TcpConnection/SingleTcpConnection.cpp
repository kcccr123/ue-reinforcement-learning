#include "TcpConnection/SingleTcpConnection.h"
#include "SocketSubsystem.h"
#include "Common/TcpSocketBuilder.h"
#include "HAL/PlatformProcess.h"
#include "TcpConnection/Threads/AcceptRunnable.h"

bool USingleTcpConnection::StartListening(const FString& IPAddress, int32 Port)
{
    CloseConnection();

    ISocketSubsystem* SocketSubsystem = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
    if (!SocketSubsystem)
    {
        UE_LOG(LogTemp, Error, TEXT("[USingleTcpConnection] Socket subsystem not found!"));
        return false;
    }

    // 2 connections: admin then single env
    ListeningSocket = FTcpSocketBuilder(TEXT("SingleEnvListener"))
        .AsReusable()
        .BoundToAddress(FIPv4Address::Any)
        .BoundToPort(Port)
        .Listening(2);

    if (!ListeningSocket)
    {
        UE_LOG(LogTemp, Error, TEXT("[USingleTcpConnection] Failed to create listening socket."));
        return false;
    }

    UE_LOG(LogTemp, Log, TEXT("[USingleTcpConnection] Listening on %s:%d"), *IPAddress, Port);
    StartAcceptThread();
    return true;
}

void USingleTcpConnection::StartAcceptThread()
{
    bStopAcceptThreadRef = false;
    AcceptRunnableRef = MakeShareable(new FAcceptRunnable(this));
    AcceptThreadRef = FRunnableThread::Create(
        AcceptRunnableRef.Get(),
        TEXT("SingleEnvAcceptThread"),
        0,
        TPri_Normal
    );

    if (!AcceptThreadRef)
    {
        UE_LOG(LogTemp, Error, TEXT("[USingleTcpConnection] Failed to start accept thread."));
    }
    else
    {
        UE_LOG(LogTemp, Log, TEXT("[USingleTcpConnection] Accept thread started."));
    }
}

bool USingleTcpConnection::AcceptEnvConnection(FSocket* InNewSocket)
{
    if (EnvSocket)
    {
        UE_LOG(LogTemp, Warning, TEXT("[USingleTcpConnection] Already have env socket. Rejecting new."));
        InNewSocket->Close();
        ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(InNewSocket);
        return false;
    }

    EnvSocket = InNewSocket;
    UE_LOG(LogTemp, Log, TEXT("[USingleTcpConnection] Environment socket connected. Single env is ready."));

    if (AcceptRunnableRef.IsValid())
    {
        AcceptRunnableRef->Stop();
    }
    return true;
}

bool USingleTcpConnection::SendMessageEnv(const FString& Data)
{
    if (!EnvSocket)
    {
        UE_LOG(LogTemp, Warning, TEXT("[USingleTcpConnection] No env socket to send data."));
        return false;
    }

    // Append newline delimiter
    FString Payload = Data + TEXT("\n");

    FTCHARToUTF8 Converter(*Payload);
    int32 BytesSent = 0;
    bool bSuccess = EnvSocket->Send(
        reinterpret_cast<const uint8*>(Converter.Get()),
        Converter.Length(),
        BytesSent
    );

    if (!bSuccess || BytesSent <= 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("[USingleTcpConnection] Failed to send env data."));
        return false;
    }

    UE_LOG(LogTemp, Log, TEXT("[USingleTcpConnection] Sent to env => %s"), *Payload);
    return true;
}

FString USingleTcpConnection::ReceiveMessageEnv(int32 BufSize)
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
        TArray<uint8> Buffer;
        Buffer.SetNumUninitialized(FMath::Min((int32)Pending, BufSize));
        int32 Read = 0;
        if (EnvSocket->Recv(Buffer.GetData(), Buffer.Num(), Read) && Read > 0)
        {
            PartialData += FString(UTF8_TO_TCHAR(reinterpret_cast<const char*>(Buffer.GetData())));
        }
    }

    // If we have a full line ending in '\n', extract it
    int32 NewlineIdx;
    if (PartialData.FindChar('\n', NewlineIdx))
    {
        FString Line = PartialData.Left(NewlineIdx);
        PartialData = PartialData.Mid(NewlineIdx + 1);
        UE_LOG(LogTemp, Log, TEXT("[USingleTcpConnection] Received: %s"), *Line);
        return Line;
    }

    return TEXT("");
}

void USingleTcpConnection::CloseConnection()
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
    if (EnvSocket)
    {
        EnvSocket->Close();
        ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(EnvSocket);
        EnvSocket = nullptr;
    }

    UE_LOG(LogTemp, Log, TEXT("[USingleTcpConnection] Closed sockets (admin + env)."));
}
