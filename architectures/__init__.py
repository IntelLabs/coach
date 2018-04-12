#
# Copyright (c) 2017 Intel Corporation
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
import logger

try:
    from architectures.tensorflow_components import general_network as ts_gn
    from architectures.tensorflow_components import architecture as ts_arch
except ImportError:
    logger.failed_imports.append("TensorFlow")

try:
    from architectures.neon_components import general_network as neon_gn
    from architectures.neon_components import architecture as neon_arch
except ImportError:
    logger.failed_imports.append("Neon")
