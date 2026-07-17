"""Reward v1 package: the original hybrid verification reward."""

from __future__ import annotations

from typing import Optional

from reward.v1.reward import CONFIG_PATH, HybridRewardV1, RewardV1Config, create_reward

__all__ = ["CONFIG_PATH", "HybridRewardV1", "RewardV1Config", "create_reward", "get_reward"]

_instance: Optional[HybridRewardV1] = None


def get_reward() -> HybridRewardV1:
    global _instance
    if _instance is None:
        _instance = create_reward()
    return _instance
