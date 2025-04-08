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

void USingleTcpConnection::SendHandshake()
{
    SendMessageAdmin(HandshakeMessage);
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

    // We can kill the accept thread if we don't expect more env connections
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

    FTCHARToUTF8 Converter(*Data);
    int32 BytesSent = 0;
    bool bSuccess = EnvSocket->Send(reinterpret_cast<const uint8*>(Converter.Get()), Converter.Length(), BytesSent);

    if (!bSuccess || BytesSent <= 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("[USingleTcpConnection] Failed to send env data."));
        return false;
    }

    UE_LOG(LogTemp, Log, TEXT("[USingleTcpConnection] Sent to env => %s"), *Data);
    return true;
}

FString USingleTcpConnection::ReceiveMessageEnv(int32 BufSize)
{
    if (!EnvSocket)
    {
        return TEXT("");
    }

    uint32 PendingSize = 0;
    if (!EnvSocket->HasPendingData(PendingSize) || PendingSize == 0)
    {
        return TEXT("");
    }

    TArray<uint8> DataBuffer;
    DataBuffer.SetNumUninitialized(FMath::Min((int32)PendingSize, BufSize));

    int32 BytesRead = 0;
    bool bOk = EnvSocket->Recv(DataBuffer.GetData(), DataBuffer.Num(), BytesRead);
    if (!bOk || BytesRead <= 0)
    {
        return TEXT("");
    }

    FString Received = FString(UTF8_TO_TCHAR(reinterpret_cast<const char*>(DataBuffer.GetData())));
    UE_LOG(LogTemp, Log, TEXT("[USingleTcpConnection] Received from env => %s"), *Received);
    return Received;
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
        SocketSubsystem->DestroySocket(EnvSocket);
        EnvSocket = nullptr;
    }

    UE_LOG(LogTemp, Log, TEXT("[USingleTcpConnection] Closed sockets (admin + env)."));
}

