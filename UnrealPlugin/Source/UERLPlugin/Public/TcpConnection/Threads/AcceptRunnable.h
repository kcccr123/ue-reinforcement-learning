#pragma once

#include "HAL/Runnable.h"
#include "HAL/ThreadSafeBool.h"

class UTcpConnection;

/**
 * FRunnable that polls a listening socket for new connections
 * and calls AcceptConnection() on its owner when one arrives.
 */
class FAcceptRunnable : public FRunnable
{
public:
    explicit FAcceptRunnable(UTcpConnection* InOwner);
    virtual ~FAcceptRunnable() override;

    // FRunnable interface
    virtual uint32 Run() override;
    virtual void Stop() override;

private:
    UTcpConnection* Owner;
    FThreadSafeBool bStop;
};
