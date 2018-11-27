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

FAILING_PRESETS = [
    'Fetch_DDPG_HER_baselines',
    'MontezumaRevenge_BC',
    'ControlSuite_DDPG',
    'Doom_Basic_BC',
    'CARLA_CIL',
    'CARLA_DDPG',
    'CARLA_Dueling_DDQN',
    'CARLA_3_Cameras_DDPG',
    'Starcraft_CollectMinerals_A3C',
    'Starcraft_CollectMinerals_Dueling_DDQN',
]

def all_presets():
    result = []
    for f in sorted(os.listdir('rl_coach/presets')):
        if f.endswith('.py') and f != '__init__.py':
            preset = f.split('.')[0]
            if preset not in FAILING_PRESETS:
                result.append(preset)
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

    p = Popen(params)

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
