from rl_platform.plugins.sb3 import SB3Plugin

LEARNER_REGISTRY: dict[str, type] = {
    "sb3": SB3Plugin,
}
