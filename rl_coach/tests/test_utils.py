import pytest

from rl_coach import utils


@pytest.mark.unit_test
def test_get_checkpoint_state_default():
    files = ['4.test.ckpt.ext', '2.test.ckpt.ext', '3.test.ckpt.ext', '1.test.ckpt.ext']
    checkpoint_state = utils.get_checkpoint_state(files)
    assert checkpoint_state.model_checkpoint_path == '4.test.ckpt'
    assert checkpoint_state.all_model_checkpoint_paths == [f[:-4] for f in sorted(files)]


@pytest.mark.unit_test
def test_get_checkpoint_state_custom():
    files = ['prefix.4.test.ckpt.ext', 'prefix.2.test.ckpt.ext', 'prefix.3.test.ckpt.ext', 'prefix.1.test.ckpt.ext']
    assert len(utils.get_checkpoint_state(files).all_model_checkpoint_paths) == 0  # doesn't match the default pattern
    checkpoint_state = utils.get_checkpoint_state(files, filename_pattern=r'([0-9]+)[^0-9].*?\.ckpt')
    assert checkpoint_state.model_checkpoint_path == '4.test.ckpt'
    assert checkpoint_state.all_model_checkpoint_paths == [f[7:-4] for f in sorted(files)]

