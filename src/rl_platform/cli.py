import argparse
import sys
from pathlib import Path


def _cmd_train(args: argparse.Namespace) -> None:
    from rl_platform.core.coordinator import Coordinator

    coordinator = Coordinator.from_yaml(args.config)
    result = coordinator.run()
    print(f"\nTraining complete — run_id: {result['run_id']}")
    if "results" in result:
        for k, v in result["results"].items():
            print(f"  {k}: {v}")


def _cmd_eval(args: argparse.Namespace) -> None:
    import gymnasium as gym
    import numpy as np

    from rl_platform.artifacts.runtime import ConfigLoader
    from rl_platform.artifacts.database import Database
    from rl_platform.core.eval import EvalDriver, EvalScenario
    from rl_platform.plugins import LEARNER_REGISTRY
    from rl_platform.core.specifications import EnvSpec
    from rl_platform.utils.logging import setup_logging

    setup_logging()

    cfg = ConfigLoader.load(args.config)
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    learner_cls = LEARNER_REGISTRY.get(cfg.learner.type)
    if learner_cls is None:
        print(f"Unknown learner type: {cfg.learner.type!r}", file=sys.stderr)
        sys.exit(1)

    if cfg.env.env_id is None:
        print("rlp eval requires env.env_id in config (gym env only for M0)", file=sys.stderr)
        sys.exit(1)

    env = gym.make(cfg.env.env_id)
    agent_id = "agent_0"
    env_spec = EnvSpec(
        env_id=cfg.env.env_id,
        agent_ids=[agent_id],
        observation_spaces={agent_id: env.observation_space},
        action_spaces={agent_id: env.action_space},
        is_multi_agent=False,
    )
    env.close()

    learner = learner_cls()
    learner.configure(env_spec, cfg.learner.params)
    learner.load(checkpoint_path)
    policy = learner.get_policy()

    db = Database(cfg.data_dir / "registry.db")
    eval_driver = EvalDriver(database=db)

    num_episodes = args.num_episodes or cfg.eval.num_episodes
    scenario = EvalScenario(name="cli_eval", num_episodes=num_episodes)

    result = eval_driver.evaluate(
        policy=policy,
        env_fn=lambda: gym.make(cfg.env.env_id),
        scenario=scenario,
    )

    db.close()

    print(f"\nEvaluation complete — {result.num_episodes} episodes")
    print(f"  mean_reward:  {result.mean_reward:.2f}")
    print(f"  mean_length:  {result.mean_length:.1f}")
    print(f"  min_reward:   {min(result.episode_rewards):.2f}")
    print(f"  max_reward:   {max(result.episode_rewards):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rlp",
        description="RL Platform CLI",
    )
    subparsers = parser.add_subparsers(dest="command")

    # rlp train
    train_parser = subparsers.add_parser("train", help="Run a training session")
    train_parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to YAML config file",
    )

    # rlp eval
    eval_parser = subparsers.add_parser("eval", help="Evaluate a checkpoint")
    eval_parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to YAML config file",
    )
    eval_parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint file",
    )
    eval_parser.add_argument(
        "--num-episodes", type=int, default=None,
        help="Number of eval episodes (overrides config)",
    )

    args = parser.parse_args()

    if args.command == "train":
        _cmd_train(args)
    elif args.command == "eval":
        _cmd_eval(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
