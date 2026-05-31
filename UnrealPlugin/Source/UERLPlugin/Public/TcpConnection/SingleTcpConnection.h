// SingleTcpConnection.h
#pragma once

#include "CoreMinimal.h"
#include "BaseTcpConnection.h"
#include "TcpConnection/EnvMessages.h"
#include "HAL/ThreadSafeBool.h"
#include "SingleTcpConnection.generated.h"

class FAcceptRunnable;
class FSocket;

/**
 * Basic Single-environment TCP connection.
 *
 * Wire format: 4-byte big-endian length prefix + MessagePack map body.
 * All message schemas are defined in EnvMessages.h.
 */
UCLASS()
class UERLPLUGIN_API USingleTcpConnection : public UBaseTcpConnection
{
    GENERATED_BODY()

public:
    virtual ~USingleTcpConnection() override { CloseConnection(); }

    // Listen on IP/Port, spawn acceptance thread
    virtual bool StartListening(const FString& IPAddress, int32 Port) override;

    // Store the handshake to be auto-sent when the first connection arrives.
    // Call this before StartListening so it is ready when Python connects.
    void SetHandshakeMsg(const FHandshakeMessage& Msg) { StoredHandshake = Msg; }

    // Game-thread-only: if a connection was just accepted on the accept thread,
    // send the stored handshake from the game thread. This keeps all socket
    // Send/Recv on the game thread, avoiding concurrent access with the accept
    // thread. Safe to call every tick; it no-ops unless a send is pending.
    void FlushPendingHandshake()
    {
        if (bHandshakePending && EnvSocket)
        {
            bHandshakePending = false;
            SendMessageEnv(StoredHandshake);
        }
    }

    // Single-socket protocol: the first (and only) incoming connection is the env
    // socket. Overrides the base two-socket logic so we never use AdminSocket.
    virtual bool AcceptConnection() override;

    // Accept environment connection
    virtual bool AcceptEnvConnection(FSocket* InNewSocket) override;

    // Game-thread-only: tear down just the current env socket (e.g. after the
    // client sends "close" or disconnects) and re-arm the accept thread so the
    // next client can connect. The listening socket stays open. This lets one
    // worker serve the Coordinator's probe connection, then the training
    // connection, then later eval / re-probe runs — without relaunching.
    void ResetEnvConnection();

    // Accepts any MSGPACK_DEFINE_MAP struct (FHandshakeMessage, FStepResultMessage, FResetResultMessage, ...)
    template<typename T>
    bool SendMessageEnv(const T& Msg)
    {
        if (!EnvSocket)
        {
            UE_LOG(LogTemp, Warning, TEXT("[USingleTcpConnection] No socket to send on."));
            return false;
        }

        msgpack::sbuffer MsgBuf;
        msgpack::pack(MsgBuf, Msg);

        const uint32 PayloadLen = static_cast<uint32>(MsgBuf.size());
        const uint8 LenBytes[4] = {
            static_cast<uint8>((PayloadLen >> 24) & 0xFF),
            static_cast<uint8>((PayloadLen >> 16) & 0xFF),
            static_cast<uint8>((PayloadLen >>  8) & 0xFF),
            static_cast<uint8>( PayloadLen        & 0xFF),
        };

        TArray<uint8> Packet;
        Packet.Reserve(4 + static_cast<int32>(MsgBuf.size()));
        Packet.Append(LenBytes, 4);
        Packet.Append(reinterpret_cast<const uint8*>(MsgBuf.data()), static_cast<int32>(MsgBuf.size()));

        int32 BytesSent = 0;
        if (!EnvSocket->Send(Packet.GetData(), Packet.Num(), BytesSent) || BytesSent != Packet.Num())
        {
            UE_LOG(LogTemp, Warning, TEXT("[USingleTcpConnection] Send failed (%d/%d bytes)."), BytesSent, Packet.Num());
            return false;
        }

        UE_LOG(LogTemp, Log, TEXT("[USingleTcpConnection] Sent %d bytes (%d payload)."), Packet.Num(), PayloadLen);
        return true;
    }

    // Receive one full length-prefixed msgpack frame and unpack into OutMsg.
    // Non-blocking: returns true only when a complete frame is available; otherwise
    // buffers what's there and returns false. Caller polls each tick.
    // T must be a MSGPACK_DEFINE_MAP struct (FStepActionMessage, FResetMessage, ...).
    template<typename T>
    bool ReceiveMessageEnv(T& OutMsg)
    {
        if (!EnvSocket)
        {
            UE_LOG(LogTemp, Error, TEXT("[USingleTcpConnection] No socket to receive on."));
            return false;
        }

        // Drain any pending bytes from the socket into PartialDataBytes.
        uint32 Pending = 0;
        if (EnvSocket->HasPendingData(Pending) && Pending > 0)
        {
            const int32 OldNum = PartialDataBytes.Num();
            PartialDataBytes.SetNumUninitialized(OldNum + static_cast<int32>(Pending));
            int32 BytesRead = 0;
            if (!EnvSocket->Recv(PartialDataBytes.GetData() + OldNum, static_cast<int32>(Pending), BytesRead) || BytesRead <= 0)
            {
                PartialDataBytes.SetNum(OldNum);
                return false;
            }
            PartialDataBytes.SetNum(OldNum + BytesRead);
        }

        // Need the 4-byte big-endian length prefix.
        if (PartialDataBytes.Num() < 4)
        {
            return false;
        }

        const uint32 PayloadLen =
            (static_cast<uint32>(PartialDataBytes[0]) << 24) |
            (static_cast<uint32>(PartialDataBytes[1]) << 16) |
            (static_cast<uint32>(PartialDataBytes[2]) <<  8) |
            (static_cast<uint32>(PartialDataBytes[3])      );

        // Need the full body too.
        if (static_cast<uint32>(PartialDataBytes.Num()) < 4 + PayloadLen)
        {
            return false;
        }

        bool bOk = false;
        try
        {
            msgpack::object_handle Handle = msgpack::unpack(
                reinterpret_cast<const char*>(PartialDataBytes.GetData() + 4),
                PayloadLen);
            Handle.get().convert(OutMsg);
            bOk = true;
        }
        catch (const std::exception& e)
        {
            UE_LOG(LogTemp, Warning, TEXT("[USingleTcpConnection] msgpack unpack failed: %s"), UTF8_TO_TCHAR(e.what()));
        }

        // Always consume the frame so a bad message doesn't wedge the buffer.
        PartialDataBytes.RemoveAt(0, static_cast<int32>(4 + PayloadLen));

        if (bOk)
        {
            UE_LOG(LogTemp, Log, TEXT("[USingleTcpConnection] Received %u byte msgpack message."), PayloadLen);
        }
        return bOk;
    }

    // Clean up
    virtual void CloseConnection() override;

    // start thread for accepting incoming connections
    virtual void StartAcceptThread() override;

protected:
    // Stop and destroy the accept thread (if any). Safe to call when none runs.
    void TeardownAcceptThread();

public:

    // IsConnected returns true once the single env socket is established.
    // AdminSocket is unused in the single-socket protocol.
    virtual bool IsConnected() const override
    {
        return EnvSocket != nullptr;
    }

protected:
    // The environment socket (the one and only socket for the new protocol).
    // Written once on the accept thread, then only touched on the game thread.
    FSocket* EnvSocket = nullptr;

    // Handshake to send once Python connects.
    FHandshakeMessage StoredHandshake;

    // Set on the accept thread when EnvSocket is published; consumed on the
    // game thread by FlushPendingHandshake(). Atomic so the EnvSocket write is
    // safely visible to the game thread before it sends.
    FThreadSafeBool bHandshakePending = false;

    // Buffer leftover bytes until we have a full length-prefixed msgpack frame
    TArray<uint8> PartialDataBytes;
};