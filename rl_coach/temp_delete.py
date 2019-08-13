# Adding module path to sys path if not there, so rl_coach submodules can be imported
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
resources_path = os.path.abspath(os.path.join('Resources'))
if module_path not in sys.path:
    sys.path.append(module_path)
if resources_path not in sys.path:
    sys.path.append(resources_path)

from rl_coach.coach import CoachInterface

coach = CoachInterface(preset='CartPole_DQN', f='mxnet')
coach.run()

a = 1

#coach = CoachInterface({'preset': 'CartPole_DQN'} )

