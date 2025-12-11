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

import os


class Solver:
    """
    Interface for a generic solver.
    """
    # agents end their final answer on the last line after this token
    FINAL_TOKEN: str = os.getenv("FINAL_TOKEN", "vote:")

    def extract_final(self, text: str, token: str = FINAL_TOKEN) -> str:
        """
        Return the text after the last occurrence of token (case-insensitive),
        or the last non-empty line if not found. Preserves original casing.
        """
        raise NotImplementedError

    def solve_trace(self, problem: str, depth: int, max_depth: int, path: str) -> tuple[str, dict]:
        """
        Internal recursive solver that returns (response, trace_node).
        Builds a complete trace tree of the decomposition process.
        """
        raise NotImplementedError
