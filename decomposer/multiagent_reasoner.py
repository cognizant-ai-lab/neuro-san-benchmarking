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

import asyncio
import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path

from decomposer.neuro_san_solver import NeuroSanSolver
from decomposer.trace_data import TraceData


class MultiAgentReasoner:
    """
    This guy is the main entry point for the multi-agent reasoner.
    """

    # Tuning knobs with environment variable overrides
    MAX_DEPTH: int = int(os.getenv("MAX_DEPTH", "5"))
    WINNING_VOTE_COUNT: int = int(os.getenv("WINNING_VOTE_COUNT", "2"))
    CANDIDATE_COUNT: int = (2 * WINNING_VOTE_COUNT) - 1
    NUMBER_OF_VOTES: int = (2 * WINNING_VOTE_COUNT) - 1
    SOLUTION_CANDIDATE_COUNT: int = (2 * WINNING_VOTE_COUNT) - 1

    LOG_FAILURES_JSONL: str = os.getenv("LOG_FAILURES_JSONL")

    def __init__(self):

        self.trace_data = TraceData()
        self.thread_id: int = threading.get_ident()

    def setup(self) -> str:
        log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

        logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s", stream=sys.stderr)
        logger = logging.getLogger()

        log_file: str = None
        log_dir: str = os.getenv("LOG_DIR")
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            log_file: str = Path(log_dir) / f"mr_{os.getpid()}_{self.thread_id}_{int(time.time())}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            logger.addHandler(file_handler)
            logging.info(f"Logging to {log_file}")

        os.environ["AGENT_MANIFEST_FILE"] = "./registries/manifest.hocon"
        os.environ["AGENT_TOOL_PATH"] = "coded_tools"

        return log_file

    def _parse_number(self, text: str) -> int | None:
        """Extract and parse a number from text, stripping commas/spaces/underscores."""
        if not text:
            return None
        cleaned = text.strip().replace(",", "").replace("_", "").replace(" ", "")
        try:
            return int(cleaned)
        except ValueError:
            numbers: list[str] = re.findall(r"\d+", cleaned)
            if numbers:
                try:
                    longest: str = max(numbers, key=len)
                    return int(longest)
                except ValueError:
                    pass
        return None

    def _extract_multiplication_problem(self, problem: str) -> tuple[int | None, int | None]:
        """Extract A and B from 'What is A × B?' or similar formats."""
        patterns = [
            r"What is (\d+)\s*[×x*]\s*(\d+)",
            r"(\d+)\s*[×x*]\s*(\d+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, problem, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1)), int(match.group(2))
                except ValueError:
                    pass
        return None, None

    def _classify_failure(self, trace: dict, _expected: int, actual: int | None) -> list[str]:
        """Classify failure patterns based on trace data."""
        patterns = []

        if actual is None:
            patterns.append("malformed_final")
            return patterns

        decomp_info = trace.get("decomposition")
        if decomp_info:
            p2_text = decomp_info.get("p2", "")
            c_text = decomp_info.get("c", "")

            if p2_text and (
                "result of P1" in p2_text
                or "result of p1" in p2_text.lower()
                or "use P1" in p2_text
                or "use p1" in p2_text.lower()
            ):
                patterns.append("non_independent_subproblems")

            if c_text:
                has_add = any(word in c_text.lower() for word in ["add", "sum", "plus"])
                has_subtract = any(word in c_text.lower() for word in ["subtract", "minus", "difference"])
                if has_add and has_subtract:
                    patterns.append("ambiguous_composition_op")

        solve_info = trace.get("solve")
        if solve_info:
            s1_final = solve_info.get("s1_final")
            s2_final = solve_info.get("s2_final")

            if s1_final is not None or s2_final is not None:
                patterns.append("composed_miscalc")
            else:
                patterns.append("atomic_miscalc")
        else:
            patterns.append("atomic_miscalc")

        if not patterns:
            patterns.append("unknown_failure")

        return patterns

    def _flatten_tree(self, node: dict, result: list[dict] | None = None) -> list[dict]:
        """
        Flatten a trace tree into a list of nodes for easy jq querying.
        Each node gets a flat representation with key fields.
        """
        if result is None:
            result = []

        decomp_info = node.get("decomposition")
        if decomp_info is None:
            node_type = "atomic"
        elif decomp_info.get("decision") == "no_decomposition":
            node_type = "no_decomposition"
        else:
            node_type = "decomposed"

        flat_node = {
            "path": node.get("path", ""),
            "depth": node.get("depth", 0),
            "type": node_type,
            "problem": node.get("problem", "")[:200],
            "final": node.get("final", ""),
            "final_num": node.get("final_num"),
            "error": node.get("error"),
        }

        if decomp_info:
            flat_node["decomposition_winner"] = decomp_info.get("chosen", "")[:100]
            if decomp_info.get("decision"):
                flat_node["decomposition_decision"] = decomp_info["decision"]

        result.append(flat_node)

        # Recurse on children
        for child in node.get("children", []):
            self._flatten_tree(child, result)

        return result

    def _annotate_failure(self, node: dict, expected: int | None = None) -> None:
        """
        Annotate each node in the tree with error information.
        Recursively processes children first (post-order).
        """
        for child in node.get("children", []):
            self._annotate_failure(child, expected)

        final_text = node.get("final", "")
        node["final_num"] = self._parse_number(final_text)

        decomp_info = node.get("decomposition")

        if decomp_info is None:
            problem = node.get("problem", "")
            a, b = self._extract_multiplication_problem(problem)
            if a is not None and b is not None:
                expected_val = a * b
                if node["final_num"] != expected_val:
                    node["error"] = {
                        "code": "atomic_miscalc",
                        "details": f"Expected {expected_val}, got {node['final_num']}",
                        "expected": expected_val,
                        "actual": node["final_num"],
                    }
            elif node["final_num"] is None:
                node["error"] = {
                    "code": "malformed_final",
                    "details": "Could not parse final answer",
                }
        else:
            p2_text = decomp_info.get("p2", "")
            c_text = decomp_info.get("c", "")

            if p2_text and (
                "result of P1" in p2_text
                or "result of p1" in p2_text.lower()
                or "use P1" in p2_text
                or "use p1" in p2_text.lower()
            ):
                node["error"] = {
                    "code": "non_independent_subproblems",
                    "details": "P2 depends on P1 result",
                    "p2": p2_text[:100],
                }

            if c_text:
                has_add = any(word in c_text.lower() for word in ["add", "sum", "plus"])
                has_subtract = any(word in c_text.lower() for word in ["subtract", "minus", "difference"])
                if has_add and has_subtract:
                    if node.get("error") is None:
                        node["error"] = {
                            "code": "ambiguous_composition_op",
                            "details": "Composition operator is ambiguous",
                            "c": c_text[:100],
                        }

            children = node.get("children", [])
            if len(children) == 2 and node["final_num"] is not None:
                s1_num = children[0].get("final_num")
                s2_num = children[1].get("final_num")

                if s1_num is not None and s2_num is not None and c_text:
                    c_lower = c_text.lower()
                    expected_comp = None
                    op_name = None

                    if any(word in c_lower for word in ["add", "sum", "plus", "+"]):
                        expected_comp = s1_num + s2_num
                        op_name = "addition"
                    elif any(word in c_lower for word in ["subtract", "minus", "difference", "-"]):
                        expected_comp = s1_num - s2_num
                        op_name = "subtraction"
                    elif any(word in c_lower for word in ["multiply", "product", "times", "*", "×"]):
                        expected_comp = s1_num * s2_num
                        op_name = "multiplication"
                    elif any(word in c_lower for word in ["divide", "quotient", "/"]):
                        if s2_num != 0:
                            expected_comp = s1_num // s2_num  # Integer division
                            op_name = "division"

                    if expected_comp is not None and node["final_num"] != expected_comp:
                        if node.get("error") is None:
                            details = (
                                f"Composition {op_name} error: {s1_num} op {s2_num} = "
                                f"{expected_comp}, got {node['final_num']}"
                            )
                            node["error"] = {
                                "code": "composed_miscalc",
                                "details": details,
                                "expected": expected_comp,
                                "actual": node["final_num"],
                                "operation": op_name,
                            }

            if node["final_num"] is None and node.get("error") is None:
                node["error"] = {
                    "code": "malformed_final",
                    "details": "Could not parse final answer at decomposed node",
                }

    def _find_failure_node(self, node: dict) -> tuple[str, dict] | None:
        """
        Find the deepest node with an error in the tree (post-order traversal).
        Returns (path, error_dict) or None if no errors found.
        """
        for child in node.get("children", []):
            failure = self._find_failure_node(child)
            if failure:
                return failure

        if node.get("error"):
            return node.get("path", "unknown"), node["error"]

        return None

    def solve(self, problem: str, depth: int = 0, max_depth: int = MAX_DEPTH) -> tuple[str, str]:
        """
        Recursive solver with tree tracing.
        :returns: A tuple of the final agent response (which includes the {FINAL_TOKEN} line).
                  and the extracted final answer from that line
        """
        solver = NeuroSanSolver(winning_vote_count=self.WINNING_VOTE_COUNT)

        node: dict[str, Any] = asyncio.run(solver.solve(problem, depth, max_depth, "0"))

        self.trace_data.tree = node

        if depth == 0:
            if node.get("decomposition"):
                self.trace_data.decomposition = {
                    "candidates": node["decomposition"].get("candidates", []),
                    "winner_idx": node["decomposition"].get("winner_idx", 0),
                    "votes": node["decomposition"].get("votes", []),
                    "p1": node["decomposition"].get("p1"),
                    "p2": node["decomposition"].get("p2"),
                    "c": node["decomposition"].get("c"),
                }

            composition = node.get("composition")
            if composition:
                sub_finals = node.get("sub_finals")
                self.trace_data.solve = {
                    "s1_final": sub_finals["s1_final"] if sub_finals else None,
                    "s2_final": sub_finals["s2_final"] if sub_finals else None,
                    "c": composition["c_text"],
                    "composed_candidates": composition["composed_candidates"],
                    "composition_votes": composition["composition_votes"],
                    "composition_winner_idx": composition["composition_winner_idx"],
                }

        resp: str = node["response"]
        extracted_final: str = node["extracted_final"]
        return resp, extracted_final

    def reason(self, problem: str) -> str:
        log_file: str = self.setup()
        if self.LOG_FAILURES_JSONL:
            log_dir = Path(self.LOG_FAILURES_JSONL).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"[main] Failure logging enabled: {self.LOG_FAILURES_JSONL}")

        self.trace_data.tree = None
        self.trace_data.decomposition = None
        self.trace_data.solve = None

        final_resp, extracted_final = self.solve(problem, depth=0, max_depth=self.MAX_DEPTH)

        logging.info(f"[main] final answer: {extracted_final!r}")

        a, b = self._extract_multiplication_problem(problem)
        if a is None or b is None:
            logging.info(f"[main] Could not extract multiplication problem from: {problem[:100]}")
        elif not self.LOG_FAILURES_JSONL:
            logging.info("[main] LOG_FAILURES_JSONL not set; skipping failure logging")
        else:
            expected = a * b
            actual = self._parse_number(extracted_final)
            logging.info(f"[main] Checking: expected={expected}, actual={actual}")

            if actual != expected:
                self.report(problem, final_resp, extracted_final, expected, actual, log_file)
            else:
                logging.info(f"[main] Correct answer; not logging to {self.LOG_FAILURES_JSONL}")

        return final_resp

    def report(self, problem: str, final_resp: str, extracted_final: str, expected: int, actual: int, log_file: str):

        trace_tree = getattr(self.trace_data, "tree", None)
        if trace_tree:
            self._annotate_failure(trace_tree, expected)
            trace_flat = self._flatten_tree(trace_tree)
            failure_node_info = self._find_failure_node(trace_tree)

            if failure_node_info:
                failure_node_path, failure_node_error = failure_node_info
                logging.info(f"[main] Failure identified at path={failure_node_path}: {failure_node_error}")
            else:
                failure_node_path = None
                failure_node_error = None
        else:
            trace_flat = []
            failure_node_path = None
            failure_node_error = None

        trace = {
            "decomposition": getattr(self.trace_data, "decomposition", None),
            "solve": getattr(self.trace_data, "solve", None),
        }

        failure_patterns = self._classify_failure(trace, expected, actual)

        diff = None if actual is None else actual - expected
        abs_diff = None if actual is None else abs(actual - expected)
        rel_error = None if actual is None or expected == 0 else abs_diff / abs(expected)

        failure_record = {
            "problem": problem,
            "expected": expected,
            "actual": actual,
            "extracted_final": extracted_final,
            "final_resp": final_resp,
            "failure_patterns": failure_patterns,
            "error": {
                "diff": diff,
                "abs_diff": abs_diff,
                "relative_error": rel_error,
            },
            "trace": trace,
            "trace_tree": trace_tree,
            "trace_flat": trace_flat,
            "failure_node_path": failure_node_path,
            "failure_node_error": failure_node_error,
            "config": {
                "WINNING_VOTE_COUNT": self.WINNING_VOTE_COUNT,
                "MAX_DEPTH": self.MAX_DEPTH,
                "CANDIDATE_COUNT": self.CANDIDATE_COUNT,
                "NUMBER_OF_VOTES": self.NUMBER_OF_VOTES,
                "SOLUTION_CANDIDATE_COUNT": self.SOLUTION_CANDIDATE_COUNT,
            },
        }

        log_dir: str = os.getenv("LOG_DIR")
        if log_dir:
            failure_record["log_file"] = str(log_file) if "log_file" in locals() else None

        try:
            with open(self.LOG_FAILURES_JSONL, "a", encoding="utf-8") as f:
                f.write(json.dumps(failure_record) + "\n")
            logging.info(f"[main] Failure logged to {self.LOG_FAILURES_JSONL}")
        except IOError as e:
            logging.error(f"[main] Failed to write failure log: {e}")

    def main(self):
        problem = sys.stdin.read().strip()
        if not problem:
            print("[ERROR] No input provided.", file=sys.stderr)
            sys.exit(1)

        final_resp: str = self.reason(problem)
        print(final_resp)


if __name__ == "__main__":
    MultiAgentReasoner().main()
