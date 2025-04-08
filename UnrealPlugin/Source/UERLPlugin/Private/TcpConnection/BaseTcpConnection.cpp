#include "TcpConnection/BaseTcpConnection.h"
#include "SocketSubsystem.h"

bool UBaseTcpConnection::AcceptConnection()
{
    if (!ListeningSocket)
    {
        UE_LOG(LogTemp, Error, TEXT("[UBaseTcpConnection] No valid listening socket."));
        return false;
    }

    ISocketSubsystem* SocketSubsystem = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
    if (!SocketSubsystem)
    {
        UE_LOG(LogTemp, Error, TEXT("[UBaseTcpConnection] Socket subsystem not found!"));
        return false;
    }

    // Attempt to accept a pending connection.
    TSharedRef<FInternetAddr> RemoteAddr = SocketSubsystem->CreateInternetAddr();
    FSocket* NewSock = ListeningSocket->Accept(*RemoteAddr, TEXT("AdminOrEnvSocket"));
    if (!NewSock)
    {
        UE_LOG(LogTemp, Warning, TEXT("[UBaseTcpConnection] Failed to accept a connection."));
        return false;
    }

    // If we have no admin yet, this new socket becomes the admin client.
    if (!AdminSocket)
    {
        AdminSocket = NewSock;
        UE_LOG(LogTemp, Log, TEXT("[UBaseTcpConnection] Admin socket connected."));
        SendHandshake();
        return true;
    }
    else
    {
        // We already have an admin => pass to environment acceptance
        return AcceptEnvConnection(NewSock);
    }
}

bool UBaseTcpConnection::SendMessageAdmin(const FString& Data)
{
    if (!AdminSocket)
    {
        UE_LOG(LogTemp, Warning, TEXT("[UBaseTcpConnection] No admin socket to send to."));
        return false;
    }

    FTCHARToUTF8 Converter(*Data);
    const int32 BytesToSend = Converter.Length();
    int32 BytesSent = 0;

    bool bSuccess = AdminSocket->Send(reinterpret_cast<const uint8*>(Converter.Get()), BytesToSend, BytesSent);
    if (!bSuccess || BytesSent <= 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("[UBaseTcpConnection] Failed to send data to admin."));
        return false;
    }

    UE_LOG(LogTemp, Log, TEXT("[UBaseTcpConnection] Sent to admin => %s"), *Data);
    return true;
}

FString UBaseTcpConnection::ReceiveMessageAdmin(int32 BufSize)
{
    if (!AdminSocket)
    {
        UE_LOG(LogTemp, Warning, TEXT("[UBaseTcpConnection] No admin socket to receive from."));
        return TEXT("");
    }

    uint32 PendingSize = 0;
    if (!AdminSocket->HasPendingData(PendingSize) || PendingSize == 0)
    {
        return TEXT("");
    }

    TArray<uint8> DataBuffer;
    DataBuffer.SetNumUninitialized(FMath::Min(static_cast<int32>(PendingSize), BufSize));

    int32 BytesRead = 0;
    bool bOk = AdminSocket->Recv(DataBuffer.GetData(), DataBuffer.Num(), BytesRead);
    if (!bOk || BytesRead <= 0)
    {
        return TEXT("");
    }

    FString ReceivedString = FString(UTF8_TO_TCHAR(reinterpret_cast<const char*>(DataBuffer.GetData())));
    UE_LOG(LogTemp, Log, TEXT("[UBaseTcpConnection] Received from admin => %s"), *ReceivedString);
    return ReceivedString;
}

FSocket* UBaseTcpConnection::GetListeningSocket()
{
    return ListeningSocket;
}


void UBaseTcpConnection::SetHandshake(const FString& InHandshakeMsg)
{
    HandshakeMessage = InHandshakeMsg;
}

