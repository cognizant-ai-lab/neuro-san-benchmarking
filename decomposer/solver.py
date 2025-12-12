# Copyright © 2025 Cognizant Technology Solutions Corp, www.cognizant.com.
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
#
# END COPYRIGHT

from typing import Any

import os


class Solver:
    """
    Interface for a generic solver.
    """
    # agents end their final answer on the last line after this token
    FINAL_TOKEN: str = os.getenv("FINAL_TOKEN", "vote:")

    def solve(self, problem: str, depth: int, max_depth: int, path: str) -> dict[str, Any]:
        """
        Internal recursive solver that returns (response, trace_node).
        Builds a complete trace tree of the decomposition process.

        :return: The root trace node of the decomposition process
        """
        raise NotImplementedError
