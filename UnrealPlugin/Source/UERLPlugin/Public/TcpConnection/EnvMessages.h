#pragma once

#include "CoreMinimal.h"

// msgpack-c is a third-party header-only library that does not satisfy UE's
// strict warning-as-error settings. Wrap it so its warnings don't fail the build.
THIRD_PARTY_INCLUDES_START
#include "msgpack.hpp"
THIRD_PARTY_INCLUDES_END

#include <map>
#include <string>
#include <vector>

//------------------------------------------------------------------------------
// UE → Python  (send)
//------------------------------------------------------------------------------

/** One agent's entry in the handshake agent list. */
struct FAgentInfo
{
    std::string           id;
    std::vector<int32_t>  obs_shape;
    std::vector<int32_t>  act_shape;
    float                 act_low      = -1.0f;
    float                 act_high     =  1.0f;
    bool                  is_scripted  = false;
    std::string           team;

    MSGPACK_DEFINE_MAP(id, obs_shape, act_shape, act_low, act_high, is_scripted, team);
};

/** Sent once on connect: describes the environment and its agents. */
struct FHandshakeMessage
{
    std::string              type    = "handshake";
    std::string              env_id;
    std::vector<FAgentInfo>  agents;

    MSGPACK_DEFINE_MAP(type, env_id, agents);
};

/** Per-agent payload inside a step_result. */
struct FAgentStepResult
{
    std::vector<float>  obs;
    float               reward  = 0.0f;
    bool                done    = false;

    MSGPACK_DEFINE_MAP(obs, reward, done);
};

/** Episode-level flags in a step_result. */
struct FGlobalStepResult
{
    bool done = false;

    MSGPACK_DEFINE_MAP(done);
};

/** Sent after every simulation step. */
struct FStepResultMessage
{
    std::string                            type    = "step_result";
    std::map<std::string, FAgentStepResult> agents;
    FGlobalStepResult                      global;

    MSGPACK_DEFINE_MAP(type, agents, global);
};

/** Per-agent payload inside a reset_result (observations only). */
struct FAgentResetResult
{
    std::vector<float> obs;

    MSGPACK_DEFINE_MAP(obs);
};

/** Sent after the environment resets. */
struct FResetResultMessage
{
    std::string                              type    = "reset_result";
    std::map<std::string, FAgentResetResult> agents;

    MSGPACK_DEFINE_MAP(type, agents);
};

//------------------------------------------------------------------------------
// Python → UE  (receive)
//------------------------------------------------------------------------------

/** Received from Python each training step. */
struct FStepActionMessage
{
    std::string                                  type;    // "step"
    std::map<std::string, std::vector<float>>    actions;

    MSGPACK_DEFINE_MAP(type, actions);
};

/** Received from Python to trigger a reset. */
struct FResetMessage
{
    std::string type;    // "reset"

    MSGPACK_DEFINE_MAP(type);
};

/** Generic inbound message that covers "step", "reset", and "close".
 *
 *  With MSGPACK_DEFINE_MAP, keys missing in the wire payload simply leave the
 *  corresponding member at its default value, so a bare {"type":"reset"} message
 *  will deserialize correctly with an empty actions map. */
struct FClientMessage
{
    std::string                                 type;    // "step" | "reset" | "close"
    std::map<std::string, std::vector<float>>   actions; // present only for "step"

    MSGPACK_DEFINE_MAP(type, actions);
};
