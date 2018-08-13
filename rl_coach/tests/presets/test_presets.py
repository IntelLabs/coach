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


@pytest.mark.integration_test
def test_all_presets_are_running():
    # os.chdir("../../")
    test_failed = False
    all_presets = sorted([f.split('.')[0] for f in os.listdir('rl_coach/presets') if f.endswith('.py') and f != '__init__.py'])
    for preset in all_presets:
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
        params = ["python3", "rl_coach/coach.py", "-p", preset, "-ns", "-e", ".test"]
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
        if os.path.exists("experiments/.test"):
            shutil.rmtree("experiments/.test")

    assert not test_failed


if __name__ == "__main__":
    test_all_presets_are_running()
