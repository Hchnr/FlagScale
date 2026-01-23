"""
Unit tests for train_robobrain_x0.py dataloader consistency.

This test verifies that lerobot and energon dataloaders produce consistent data
when initialized with equivalent mock datasets.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

_mock_modules = [
    # megatron modules
    "megatron",
    "megatron.core",
    "megatron.core.config",
    "megatron.core.parallel_state",
    "megatron.core.datasets",
    "megatron.core.datasets.blended_megatron_dataset_builder",
    "megatron.core.datasets.gpt_dataset",
    "megatron.core.enums",
    "megatron.core.energy_monitor",
    "megatron.core.models",
    "megatron.core.models.gpt",
    "megatron.core.models.gpt.gpt_layer_specs",
    "megatron.core.models.gpt.heterogeneous",
    "megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs",
    "megatron.core.rerun_state_machine",
    "megatron.core.transformer",
    "megatron.core.transformer.spec_utils",
    "megatron.core.utils",
    "megatron.core.num_microbatches_calculator",
    "megatron.core.mpu",
    "megatron.energon",
    "megatron.legacy",
    "megatron.legacy.model",
    "megatron.plugin",
    "megatron.plugin.utils",
    "megatron.training",
    "megatron.training.arguments",
    "megatron.training.checkpointing",
    "megatron.training.dist_signal_handler",
    "megatron.training.global_vars",
    "megatron.training.spiky_loss",
    "megatron.training.tokenizer",
    "megatron.training.tokenizer.tokenizer",
    "megatron.training.training",
    "megatron.training.utils",
    "megatron.training.yaml_arguments",
    # external dependencies
    "webdataset",
    "webdataset.autodecode",
    # tools modules
    "tools",
    "tools.datasets",
    "tools.datasets.vla",
    "tools.datasets.vla.data",
    "tools.datasets.vla.data.dataset_helpers_vlm",
    "tools.datasets.vla.data.energon",
    "tools.datasets.vla.data.energon.chatml",
    # flagscale internal modules
    "flagscale.models.megatron.qwen2_5_vl",
    "flagscale.models.megatron.qwen2_5_vl.layer_specs",
    "flagscale.models.megatron.qwen2_5_vl.qwen2_5_vl_model",
    "flagscale.models.megatron.qwen2_5_vl.tensor_parallel",
    "flagscale.models.megatron.qwen2_5_vl.transformer_config",
    "flagscale.train.datasets.lerobot_dataset",
]

# Store original modules for cleanup
_original_modules = {}

for mod in _mock_modules:
    _original_modules[mod] = sys.modules.get(mod)
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

from flagscale.train.megatron.train_robobrain_x0 import (
    EnergonDataloader,
    LeRobotDataloader,
    lerobot_collate_fn,
)


@pytest.fixture(scope="module", autouse=True)
def cleanup_mocked_modules():
    """Cleanup mocked modules after all tests in this module complete."""
    yield
    # Restore original modules or remove mocks
    for mod in _mock_modules:
        if _original_modules[mod] is None:
            sys.modules.pop(mod, None)
        else:
            sys.modules[mod] = _original_modules[mod]


class MockArgs:
    max_padding_length = 128
    temporal_patch_size = 2
    spatial_merge_size = 2
    patch_size = 14
    tensor_model_parallel_size = 1
    context_parallel_size = 1
    sequence_parallel = False
    micro_batch_size = 2
    num_workers = 0
    enable_variable_seq_lengths = False
    transformer_pipeline_model_parallel_size = 0
    video_backend = "pyav"


class MockTokenizer:
    pad_token_id = 0
    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652
    vocab = {"<|im_start|>": 151644, "<|im_end|>": 151645}
    processor = MagicMock()

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text[:10]]


mock_args = MockArgs()
mock_tokenizer = MockTokenizer()


def create_mock_batch_data(batch_size, seq_len, num_images=1, seed=42):
    """Create mock batch data for both lerobot and energon formats."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    img_dim = 3 * 2 * 14 * 14
    num_patches = num_images * 4

    return {
        "text": np.random.randint(0, 1000, (batch_size, seq_len), dtype=np.int64),
        "target": np.random.randint(0, 1000, (batch_size, seq_len), dtype=np.int64),
        "imgs": np.random.randn(num_patches, img_dim).astype(np.float32),
        "videos": np.empty([0, img_dim], dtype=np.float32),
        "image_thw_grids": np.array([[1, 2, 2]] * num_images, dtype=np.int64),
        "video_thw_grids": np.empty([0, 3], dtype=np.int64),
        "image_input_mask": np.zeros((batch_size, seq_len), dtype=bool),
        "video_input_mask": np.zeros((batch_size, seq_len), dtype=bool),
        "second_per_grid_ts": np.empty([0], dtype=np.float32),
    }


def to_lerobot_samples(data, batch_size):
    """Convert batch data to individual lerobot-style samples."""
    return [
        {
            "text": data["text"][i],
            "target": data["target"][i],
            "imgs": data["imgs"]
            if i == 0
            else np.empty([0, data["imgs"].shape[1]], dtype=np.float32),
            "videos": data["videos"],
            "image_thw_grids": data["image_thw_grids"]
            if i == 0
            else np.empty([0, 3], dtype=np.int64),
            "video_thw_grids": data["video_thw_grids"],
            "image_input_mask": data["image_input_mask"][i],
            "video_input_mask": data["video_input_mask"][i],
            "second_per_grid_ts": data["second_per_grid_ts"],
        }
        for i in range(batch_size)
    ]


def to_energon_batch(data):
    """Convert data to energon batch format (torch tensors)."""
    return {k: torch.from_numpy(v) for k, v in data.items()}


class TestDataloaderConsistency:
    """Test data consistency between lerobot and energon dataloaders."""

    @pytest.fixture
    def batch_data(self):
        return create_mock_batch_data(
            mock_args.micro_batch_size, mock_args.max_padding_length, num_images=2
        )

    def _get_lerobot_batch(self, data):
        samples = to_lerobot_samples(data, mock_args.micro_batch_size)
        with (
            patch("flagscale.train.megatron.train_robobrain_x0.get_args", return_value=mock_args),
            patch(
                "flagscale.train.megatron.train_robobrain_x0.get_tokenizer",
                return_value=mock_tokenizer,
            ),
        ):
            return lerobot_collate_fn(samples)

    def test_output_keys_match(self, batch_data):
        """Verify both formats output the same keys."""
        lerobot_batch = self._get_lerobot_batch(batch_data)
        energon_batch = to_energon_batch(batch_data)
        assert set(lerobot_batch.keys()) == set(energon_batch.keys())

    def test_text_consistency(self, batch_data):
        """Verify text tokens match."""
        lerobot_batch = self._get_lerobot_batch(batch_data)
        energon_batch = to_energon_batch(batch_data)
        assert torch.allclose(lerobot_batch["text"].float(), energon_batch["text"].float())

    def test_target_consistency(self, batch_data):
        """Verify target tokens match."""
        lerobot_batch = self._get_lerobot_batch(batch_data)
        energon_batch = to_energon_batch(batch_data)
        assert torch.allclose(lerobot_batch["target"].float(), energon_batch["target"].float())

    def test_image_mask_consistency(self, batch_data):
        """Verify image masks match."""
        lerobot_batch = self._get_lerobot_batch(batch_data)
        energon_batch = to_energon_batch(batch_data)
        assert torch.equal(lerobot_batch["image_input_mask"], energon_batch["image_input_mask"])

    def test_grid_consistency(self, batch_data):
        """Verify grid info matches."""
        lerobot_batch = self._get_lerobot_batch(batch_data)
        energon_batch = to_energon_batch(batch_data)
        assert torch.equal(lerobot_batch["image_thw_grids"], energon_batch["image_thw_grids"])
        assert torch.equal(lerobot_batch["video_thw_grids"], energon_batch["video_thw_grids"])

    def test_multiple_seeds(self):
        """Verify consistency across multiple random seeds."""
        for seed in [42, 123, 456]:
            data = create_mock_batch_data(
                mock_args.micro_batch_size, mock_args.max_padding_length, seed=seed
            )
            lerobot_batch = self._get_lerobot_batch(data)
            energon_batch = to_energon_batch(data)
            assert torch.allclose(lerobot_batch["text"].float(), energon_batch["text"].float())

    def test_dataloader_iteration_consistency(self, batch_data):
        """Verify data consistency when iterating through both dataloaders."""
        samples = to_lerobot_samples(batch_data, mock_args.micro_batch_size)

        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, d):
                self.d = d

            def __len__(self):
                return len(self.d)

            def __getitem__(self, i):
                return self.d[i]

        def collate_fn(x):
            with (
                patch(
                    "flagscale.train.megatron.train_robobrain_x0.get_args", return_value=mock_args
                ),
                patch(
                    "flagscale.train.megatron.train_robobrain_x0.get_tokenizer",
                    return_value=mock_tokenizer,
                ),
            ):
                return lerobot_collate_fn(x)

        lerobot_loader = LeRobotDataloader(
            torch.utils.data.DataLoader(
                SimpleDataset(samples), batch_size=len(samples), collate_fn=collate_fn
            )
        )

        class MockEnergonIter:
            def __init__(self, b):
                self.b = b

            def __iter__(self):
                return self

            def __next__(self):
                return self.b

        energon_loader = EnergonDataloader(MockEnergonIter(to_energon_batch(batch_data)))

        lerobot_batch = next(lerobot_loader)
        energon_batch = next(energon_loader)

        assert torch.allclose(lerobot_batch["text"].float(), energon_batch["text"].float())
        assert torch.equal(lerobot_batch["image_input_mask"], energon_batch["image_input_mask"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
