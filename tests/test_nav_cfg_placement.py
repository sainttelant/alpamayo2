# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Device placement regressions without loading the release weights."""

import importlib.util
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers.cache_utils import DynamicCache


DEMO_PATH = Path(__file__).resolve().parents[1] / "examples/two_gpu_nav_cfg_demo.py"


@pytest.fixture
def demo():
    spec = importlib.util.spec_from_file_location("nav_cfg_demo", DEMO_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_three_gpu_map_keeps_blocks_whole_and_reserves_expert_space(demo):
    device_map = demo.build_three_gpu_device_map(64)
    prefix = "vlm.model.language_model"
    layers = [device_map[f"{prefix}.layers.{index}"] for index in range(64)]
    assert layers == ["cuda:0"] * 24 + ["cuda:1"] * 26 + ["cuda:2"] * 14
    assert Counter(layers) == {"cuda:0": 24, "cuda:1": 26, "cuda:2": 14}
    assert device_map["expert"] == "cuda:2"
    for name in (
        "vlm.model.visual",
        f"{prefix}.embed_tokens",
        f"{prefix}.rotary_emb",
        f"{prefix}.norm",
        "vlm.lm_head",
    ):
        assert device_map[name] == "cuda:0"
    # Parent placement would move an entire sharded subtree during dispatch.
    assert not any(
        other.startswith(name + ".")
        for name in device_map
        for other in device_map
    )
    assert len(device_map) == 70


@pytest.mark.parametrize("count", [0, 32, 65])
def test_three_gpu_map_rejects_other_architectures(demo, count):
    with pytest.raises(ValueError, match="64 VLM layers"):
        demo.build_three_gpu_device_map(count)


def test_cache_transfer_updates_tensors_and_device_metadata(demo):
    cache = DynamicCache()
    for index in range(3):
        keys = torch.randn(1, 2, 4, 8)
        cache.update(keys, keys.clone(), index)
        cache.layers[index].device = torch.device(f"cuda:{index}")
        cache.layers[index].nested = {"tensor": [keys.clone()]}
    result = demo.move_cache_to_device(cache, "cpu")
    assert result is cache
    assert cache.get_seq_length() == 4
    for layer in cache.layers:
        assert layer.device == torch.device("cpu")
        assert layer.keys.device == layer.values.device == torch.device("cpu")
        assert layer.nested["tensor"][0].device == torch.device("cpu")
    cache.update(torch.ones(1, 2, 1, 8), torch.ones(1, 2, 1, 8), 0)
    cache.crop(4)
    assert cache.get_seq_length() == 4


def test_three_gpu_loader_dispatches_without_moving_whole_vlm(demo, monkeypatch):
    from alpamayo2_super.config import Alpamayo2SuperConfig
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

    config = SimpleNamespace(
        vlm_class="Qwen3VLForConditionalGeneration",
        enable_expert=True,
        vlm_config=SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=64)),
    )
    model = SimpleNamespace(expert=torch.nn.Linear(1, 1), vlm=Mock(), eval=Mock())
    load = Mock(return_value=model)
    monkeypatch.setattr(Alpamayo2SuperConfig, "from_pretrained", Mock(return_value=config))
    monkeypatch.setattr(Alpamayo2Super, "from_pretrained", load)
    move_plain = Mock()
    monkeypatch.setattr(demo, "move_plain_tensor_attrs_to_device", move_plain)
    assert demo.load_model_on_three_gpus("nvidia/Alpamayo2-Super") is model
    load.assert_called_once_with(
        "nvidia/Alpamayo2-Super",
        config=config,
        dtype=torch.bfloat16,
        device_map=demo.build_three_gpu_device_map(64),
    )
    model.vlm.to.assert_not_called()
    move_plain.assert_called_once_with(model.expert, "cuda:2")
    model.eval.assert_called_once()


def test_two_gpu_loader_preserves_whole_module_placement(demo, monkeypatch):
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

    model = Mock()
    load = Mock(return_value=model)
    monkeypatch.setattr(Alpamayo2Super, "from_pretrained", load)
    monkeypatch.setattr(demo, "move_plain_tensor_attrs_to_device", Mock())
    assert demo.load_model_on_two_gpus(
        "nvidia/Alpamayo2-Super", "cuda:0", "cuda:1"
    ) is model
    load.assert_called_once_with("nvidia/Alpamayo2-Super", dtype=torch.bfloat16)
    model.vlm.to.assert_called_once_with("cuda:0")
    model.expert.to.assert_called_once_with("cuda:1")


def test_three_gpu_requires_three_visible_devices_before_loading_data(demo, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="only 2 CUDA devices"):
        demo.run_two_gpu_nav_cfg("nvidia/Alpamayo2-Super", "clip", 0, three_gpu=True)


def test_three_gpu_rejects_conflicting_device_overrides(demo, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    with pytest.raises(ValueError, match="requires --vlm-device cuda:0"):
        demo.run_two_gpu_nav_cfg(
            "nvidia/Alpamayo2-Super", "clip", 0, expert_device="cuda:1", three_gpu=True
        )
