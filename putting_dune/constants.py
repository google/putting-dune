# Copyright 2022 The Putting Dune Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyformat: mode=pyink
"""A collection of useful constants."""

import numpy as np

# Atomic numbers.
CARBON = 6
SILICON = 14

CARBON_BOND_DISTANCE_ANGSTROMS = 1.42

# Silicon-doped Graphene (SiGr) prior rates.
SIGR_PRIOR_RATE_MEAN = np.array((0.85, 0))
SIGR_PRIOR_RATE_COV = np.array(((0.1, 0), (0, 0.1)))
SIGR_PRIOR_MAX_RATE = np.log(2) / 3
