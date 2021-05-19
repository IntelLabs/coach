#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np

from robosuite.utils.mjcf_utils import CustomMaterial

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.environments.manipulation.lift import Lift

from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler

TAMAR_LAB_TABLE_TOP_SIZE = (0.84, 1.25, 0.05)
TAMAR_LAB_TABLE_OFFSET = (0, 0, 0.82)


class CubeExp(Lift):
    """
    This class corresponds to multi-colored cube exploration for a single robot arm.
    """

    def __init__(
        self,
        robots,
        table_full_size=TAMAR_LAB_TABLE_TOP_SIZE,
        table_offset=TAMAR_LAB_TABLE_OFFSET,
        placement_initializer=None,
        penalize_reward_on_collision=False,
        end_episode_on_collision=False,
        **kwargs
    ):
        """
        Args:
            robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
                (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
                Note: Must be a single single-arm robot!
            table_full_size (3-tuple): x, y, and z dimensions of the table.
            placement_initializer (ObjectPositionSampler instance): if provided, will
                be used to place objects on every reset, else a UniformRandomSampler
                is used by default.
            Rest of kwargs follow Lift class arguments
        """
        if placement_initializer is None:
            placement_initializer = UniformRandomSampler(
                # Placement range for Tamar Lab setup
                name="ObjectSampler",
                x_range=[0.0, 0.0],
                y_range=[0.0, 0.0],
                rotation=(0.0, 0.0),
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=table_offset,
                z_offset=0.9,
            )

        super().__init__(
            robots=robots,
            table_full_size=table_full_size,
            placement_initializer=placement_initializer,
            initialization_noise=None,
            **kwargs
        )

        self._max_episode_steps = self.horizon

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        SingleArmEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        cube_material = self._get_cube_material()
        self.cube = BoxObject(
            name="cube",
            size_min=(0.025, 0.025, 0.025),
            size_max=(0.025, 0.025, 0.025),
            rgba=[1, 0, 0, 1],
            material=cube_material,
        )

        self.placement_initializer.reset()
        self.placement_initializer.add_objects(self.cube)

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
        )

    @property
    def action_spec(self):
        """
        Action space (low, high) for this environment
        """
        low, high = super().action_spec
        return low[:3], high[:3]

    def _get_cube_material(self):
        from robosuite.utils.mjcf_utils import array_to_string
        rgba = (1, 0, 0, 1)
        cube_material = CustomMaterial(
            texture=rgba,
            tex_name="solid",
            mat_name="solid_mat",
        )
        cube_material.tex_attrib.pop('file')
        cube_material.tex_attrib["type"] = "cube"
        cube_material.tex_attrib["builtin"] = "flat"
        cube_material.tex_attrib["rgb1"] = array_to_string(rgba[:3])
        cube_material.tex_attrib["rgb2"] = array_to_string(rgba[:3])
        cube_material.tex_attrib["width"] = "100"
        cube_material.tex_attrib["height"] = "100"

        return cube_material

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        from robosuite.utils.mjmod import Texture

        super()._reset_internal()

        self._action_dim = 3

        geom_id = self.sim.model.geom_name2id('cube_g0_vis')
        mat_id = self.sim.model.geom_matid[geom_id]
        tex_id = self.sim.model.mat_texid[mat_id]
        texture = Texture(self.sim.model, tex_id)
        bitmap_to_set = texture.bitmap
        bitmap = np.zeros_like(bitmap_to_set)
        bitmap[:100, :, :] = 255
        bitmap[100:200, :, 0] = 255
        bitmap[200:300, :, 1] = 255
        bitmap[300:400, :, 2] = 255
        bitmap[400:500, :, :2] = 255
        bitmap[500:, :, 1:] = 255
        bitmap_to_set[:] = bitmap
        for render_context in self.sim.render_contexts:
            render_context.upload_texture(texture.id)

    def _pre_action(self, action, policy_step=False):
        """ explicitly shut the gripper """
        joined_action = np.append(action, [0., 0., 0., 1.])
        self._action_dim = 7
        super()._pre_action(joined_action, policy_step)

    def _post_action(self, action):
        ret = super()._post_action(action)
        self._action_dim = 3
        return ret

    def reward(self, action=None):
        return 0

    def _check_success(self):
        return False
