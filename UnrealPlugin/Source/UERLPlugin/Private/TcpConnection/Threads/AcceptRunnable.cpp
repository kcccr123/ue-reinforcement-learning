// Fill out your copyright notice in the Description page of Project Settings.


#include "TcpConnection/Threads/AcceptRunnable.h"
#include "TcpConnection/SingleTcpConnection.h"
#include "HAL/PlatformProcess.h"

FAcceptRunnable::FAcceptRunnable(USingleTcpConnection* InOwner)
    : Owner(InOwner)
    , bStop(false)
{
}

FAcceptRunnable::~FAcceptRunnable() = default;

uint32 FAcceptRunnable::Run()
{
    while (!bStop && Owner && Owner->GetListeningSocket())
    {
        bool bHasPending = false;
        Owner->GetListeningSocket()->HasPendingConnection(bHasPending);
        if (bHasPending)
        {
            Owner->AcceptConnection();
        }
        FPlatformProcess::Sleep(0.1f);
    }
    return 0;
}

void FAcceptRunnable::Stop()
{
    bStop = true;
}
