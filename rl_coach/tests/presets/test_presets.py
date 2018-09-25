# nasty hack to deal with issue #46
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
import os
import time
import shutil
from subprocess import Popen, DEVNULL
from rl_coach.logger import screen


def all_presets():
    result = []
    for f in sorted(os.listdir('rl_coach/presets')):
        if f.endswith('.py') and f != '__init__.py':
            result.append(f.split('.')[0])
    return result


@pytest.fixture(params=all_presets())
def preset(request):
    return request.param


@pytest.mark.integration_test
def test_preset_runs(preset):
    test_failed = False

    print("Testing preset {}".format(preset))

    # TODO: this is a temporary workaround for presets which define more than a single available level.
    # we should probably do this in a more robust way
    level = ""
    if "Atari" in preset:
        level = "breakout"
    elif "Mujoco" in preset:
        level = "inverted_pendulum"
    elif "ControlSuite" in preset:
        level = "pendulum:swingup"

    experiment_name = ".test-" + preset

    params = ["python3", "rl_coach/coach.py", "-p", preset, "-ns", "-e", experiment_name]
    if level != "":
        params += ["-lvl", level]

    p = Popen(params, stdout=DEVNULL)

    # wait 10 seconds overhead of initialization etc.
    time.sleep(10)
    return_value = p.poll()

    if return_value is None:
        screen.success("{} passed successfully".format(preset))
    else:
        test_failed = True
        screen.error("{} failed".format(preset), crash=False)

    p.kill()
    if os.path.exists("experiments/" + experiment_name):
        shutil.rmtree("experiments/" + experiment_name)

    assert not test_failed
