"""Reward v2 package: the upgraded hybrid reward (per-field judges + rules + gates)."""

from __future__ import annotations

from typing import Optional

from reward.v2.reward import CONFIG_PATH, HybridRewardV2, RewardV2Config, create_reward

__all__ = ["CONFIG_PATH", "HybridRewardV2", "RewardV2Config", "create_reward", "get_reward"]

_instance: Optional[HybridRewardV2] = None


def get_reward() -> HybridRewardV2:
    global _instance
    if _instance is None:
        _instance = create_reward()
    return _instance
