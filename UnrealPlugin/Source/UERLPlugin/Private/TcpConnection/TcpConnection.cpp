#include "TcpConnection/TcpConnection.h"
#include "SocketSubsystem.h"
#include "Common/TcpSocketBuilder.h"
#include "HAL/PlatformProcess.h"
#include "TcpConnection/Threads/AcceptRunnable.h"

FSocket* UTcpConnection::GetListeningSocket()
{
    return ListeningSocket;
}

bool UTcpConnection::StartListening(const FString& IPAddress, int32 Port)
{
    CloseConnection();

    ISocketSubsystem* SocketSubsystem = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
    if (!SocketSubsystem)
    {
        UE_LOG(LogTemp, Error, TEXT("[UTcpConnection] Socket subsystem not found!"));
        return false;
    }

    // Only one env socket is used at a time, but a small backlog lets the next
    // client queue while the previous one is being torn down.
    ListeningSocket = FTcpSocketBuilder(TEXT("SingleEnvListener"))
        .AsReusable()
        .BoundToAddress(FIPv4Address::Any)
        .BoundToPort(Port)
        .Listening(2);

    if (!ListeningSocket)
    {
        UE_LOG(LogTemp, Error, TEXT("[UTcpConnection] Failed to create listening socket."));
        return false;
    }

    UE_LOG(LogTemp, Log, TEXT("[UTcpConnection] Listening on %s:%d"), *IPAddress, Port);
    StartAcceptThread();
    return true;
}

void UTcpConnection::StartAcceptThread()
{
    // Ensure any previous accept thread is fully torn down before starting a new
    // one (ResetEnvConnection re-arms accept across client reconnects).
    TeardownAcceptThread();

    bStopAcceptThreadRef = false;
    AcceptRunnableRef = MakeShared<FAcceptRunnable>(this);
    AcceptThreadRef = FRunnableThread::Create(
        AcceptRunnableRef.Get(),
        TEXT("SingleEnvAcceptThread"),
        0,
        TPri_Normal
    );

    if (!AcceptThreadRef)
    {
        UE_LOG(LogTemp, Error, TEXT("[UTcpConnection] Failed to start accept thread."));
    }
    else
    {
        UE_LOG(LogTemp, Log, TEXT("[UTcpConnection] Accept thread started."));
    }
}

void UTcpConnection::TeardownAcceptThread()
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
    AcceptRunnableRef.Reset();
}

bool UTcpConnection::AcceptConnection()
{
    // Single-socket protocol: treat the very first connection as the env socket.
    ISocketSubsystem* SocketSubsystem = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
    if (!SocketSubsystem || !ListeningSocket)
    {
        UE_LOG(LogTemp, Error, TEXT("[UTcpConnection] AcceptConnection: no listening socket."));
        return false;
    }

    TSharedRef<FInternetAddr> RemoteAddr = SocketSubsystem->CreateInternetAddr();
    FSocket* NewSock = ListeningSocket->Accept(*RemoteAddr, TEXT("EnvSocket"));
    if (!NewSock)
    {
        return false;
    }

    if (EnvSocket)
    {
        UE_LOG(LogTemp, Warning, TEXT("[UTcpConnection] Already have env socket. Rejecting new."));
        NewSock->Close();
        ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(NewSock);
        return false;
    }

    EnvSocket = NewSock;

    // Stop accepting more connections — we only need one.
    if (AcceptRunnableRef.IsValid())
    {
        AcceptRunnableRef->Stop();
    }

    // Defer the handshake send to the game thread (FlushPendingHandshake) so all
    // socket Send/Recv stays on one thread. Setting this atomic last also
    // publishes the EnvSocket write to the game thread.
    bHandshakePending = true;
    UE_LOG(LogTemp, Log, TEXT("[UTcpConnection] Env socket connected. Handshake queued for game thread."));

    return true;
}

void UTcpConnection::ResetEnvConnection()
{
    // Tear down only the env socket; keep the listening socket open so the next
    // client can connect. Called on the game thread after a "close" / disconnect.
    if (EnvSocket)
    {
        EnvSocket->Close();
        ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(EnvSocket);
        EnvSocket = nullptr;
    }

    // Drop any buffered partial frame and a not-yet-sent handshake so the next
    // client starts from a clean state.
    PartialDataBytes.Reset();
    bHandshakePending = false;

    if (!ListeningSocket)
    {
        UE_LOG(LogTemp, Warning, TEXT("[UTcpConnection] ResetEnvConnection: no listening socket; cannot re-arm accept."));
        return;
    }

    // Re-arm acceptance for the next client.
    StartAcceptThread();
    UE_LOG(LogTemp, Log, TEXT("[UTcpConnection] Env connection reset; accepting new connections."));
}

void UTcpConnection::CloseConnection()
{
    TeardownAcceptThread();

    ISocketSubsystem* SocketSubsystem = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);

    // listening
    if (ListeningSocket)
    {
        ListeningSocket->Close();
        SocketSubsystem->DestroySocket(ListeningSocket);
        ListeningSocket = nullptr;
    }

    // env
    if (EnvSocket)
    {
        EnvSocket->Close();
        SocketSubsystem->DestroySocket(EnvSocket);
        EnvSocket = nullptr;
    }

    PartialDataBytes.Reset();
    bHandshakePending = false;

    UE_LOG(LogTemp, Log, TEXT("[UTcpConnection] Closed listening + env sockets."));
}
