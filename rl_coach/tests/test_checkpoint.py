import os
import pytest
import tempfile

from rl_coach import checkpoint


@pytest.mark.unit_test
def test_get_checkpoint_state():
    files = ['4.test.ckpt.ext', '2.test.ckpt.ext', '3.test.ckpt.ext', '1.test.ckpt.ext', 'prefix.10.test.ckpt.ext']
    with tempfile.TemporaryDirectory() as temp_dir:
        [open(os.path.join(temp_dir, fn), 'a').close() for fn in files]
        checkpoint_state = checkpoint.get_checkpoint_state(temp_dir, all_checkpoints=True)
        assert checkpoint_state.model_checkpoint_path == os.path.join(temp_dir, '4.test.ckpt')
        assert checkpoint_state.all_model_checkpoint_paths == \
               [os.path.join(temp_dir, f[:-4]) for f in sorted(files[:-1])]

        reader = checkpoint.CheckpointStateReader(temp_dir, checkpoint_state_optional=False)
        assert reader.get_latest() is None
        assert len(reader.get_all()) == 0

        reader = checkpoint.CheckpointStateReader(temp_dir)
        assert reader.get_latest().num == 4
        assert [ckp.num for ckp in reader.get_all()] == [1, 2, 3, 4]
