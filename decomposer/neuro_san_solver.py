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

import logging
import os
import re
import sys
import threading

from neuro_san.client.agent_session_factory import AgentSession
from neuro_san.client.streaming_input_processor import StreamingInputProcessor

from decomposer.session_manager import SessionManager

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="[%(levelname)s] %(message)s", stream=sys.stderr)

_DECOMP_FIELD_RE = re.compile(r"(P1|P2|C)\s*=\s*\[(.*?)]", re.DOTALL)

os.environ["AGENT_MANIFEST_FILE"] = "./registries/manifest.hocon"
os.environ["AGENT_TOOL_PATH"] = "coded_tools"

FINAL_TOKEN = os.getenv("FINAL_TOKEN", "vote:")  # agents end their final answer on the last line after this token

# Tuning knobs with environment variable overrides
WINNING_VOTE_COUNT = int(os.getenv("WINNING_VOTE_COUNT", "2"))
CANDIDATE_COUNT = (2 * WINNING_VOTE_COUNT) - 1
NUMBER_OF_VOTES = (2 * WINNING_VOTE_COUNT) - 1
SOLUTION_CANDIDATE_COUNT = (2 * WINNING_VOTE_COUNT) - 1


class NeuroSanSolver:
    """
    Generic solver implementation that uses Neuro SAN.
    """

    def decomposer_session(self) -> AgentSession:
        return SessionManager.get_session("decomposer")

    def solution_discriminator_session(self) -> AgentSession:
        return SessionManager.get_session("solution_discriminator")

    def composition_discriminator_session(self) -> AgentSession:
        return SessionManager.get_session("composition_discriminator")

    def problem_solver_session(self) -> AgentSession:
        return SessionManager.get_session("problem_solver")

    # Unique temp file per *call*
    def _tmpfile(self, stem: str) -> str:
        return f"/tmp/{stem}_{os.getpid()}_{threading.get_ident()}.txt"

    def call_agent(self, agent_session: AgentSession, text: str, timeout_ms: float = 100000.0) -> str:
        """
        Call a single agent with given text, return its response.
        """
        thread = {
            "last_chat_response": None,
            "prompt": "",
            "timeout": timeout_ms,
            "num_input": 0,
            "user_input": text,
            "sly_data": None,
            "chat_filter": {"chat_filter_type": "MAXIMAL"},
        }
        inp = StreamingInputProcessor("DEFAULT", self._tmpfile("program_mode_thinking"), agent_session, None)
        thread = inp.process_once(thread)
        logging.debug(f"call_agent({agent_session}): sending {len(text)} chars")
        resp = thread.get("last_chat_response") or ""
        logging.debug(f"call_agent({agent_session}): received {len(resp)} chars")
        return resp

    def extract_final(self, text: str, token: str = FINAL_TOKEN) -> str:
        """
        Return the text after the last occurrence of token (case-insensitive),
        or the last non-empty line if not found. Preserves original casing.
        """
        if not text:
            return ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return ""
        tkn = (token or "").strip()
        if not tkn:
            return lines[-1]

        tkn_lower = tkn.lower()
        for ln in reversed(lines):
            # Find LAST occurrence of token in this line (case-insensitive)
            idx = ln.lower().rfind(tkn_lower)
            if idx != -1:
                return ln[idx + len(tkn):].strip()
        return lines[-1]

    def _extract_decomposition_text(self, resp: str) -> str | None:
        """
        Scan the FULL agent response (multi-line) for P1=[...], P2=[...], C=[...].
        Returns a canonical single-line 'P1=[...], P2=[...], C=[...]' or None.
        """
        fields = {}
        for label, val in _DECOMP_FIELD_RE.findall(resp or ""):
            fields[label] = val.strip()

        if fields:
            p1 = fields.get("P1", "None")
            p2 = fields.get("P2", "None")
            c = fields.get("C", "None")
            return f"P1=[{p1}], P2=[{p2}], C=[{c}]"

        # Fallback: if the last line already contains the canonical string
        tail = self.extract_final(resp)
        if "P1=" in tail and "C=" in tail:
            return tail
        return None

    def _parse_decomposition(self, decomp_line: str) -> tuple[str | None, str | None, str | None]:
        """
        Parses: P1=[p1], P2=[p2], C=[c]
        Returns (p1, p2, c) with 'None' coerced to None.
        """
        parts = {
            seg.split("=", 1)[0].strip(): seg.split("=", 1)[1].strip() for seg in decomp_line.split(",") if "=" in seg
        }

        p1 = self.unbracket(parts.get("P1"))
        p2 = self.unbracket(parts.get("P2"))
        c = self.unbracket(parts.get("C"))
        return p1, p2, c

    def unbracket(self, s: str | None) -> str | None:
        if not s:
            return None
        s = s.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip()
        return None if s == "None" else s

    def _compose_prompt(self, c: str, s1: str, s2: str) -> str:
        """
        Build a prompt for the final composition solve: C(s1, s2).
        We pass the original problem, the composition description, and the sub-solutions.
        """
        return f"Solve C(P1, P2) such that C={c}, P1={s1}, P2={s2}"

    def _solve_atomic(self, problem: str) -> str:
        """
        Single call to problem_solver; returns the full agent response.
        """
        return self.call_agent(self.problem_solver_session(), problem)

    def _solve_atomic_with_voting(self, problem: str) -> tuple[str, list[str], list[int], int, list[str]]:
        """
        Generate multiple atomic solutions and vote on them.
        Returns (chosen_response, finals, votes, winner_idx, solutions).
        """
        solutions: list[str] = []
        finals: list[str] = []
        for k in range(SOLUTION_CANDIDATE_COUNT):
            r = self.call_agent(self.problem_solver_session(), problem)
            solutions.append(r)
            finals.append(self.extract_final(r))
            logging.info(f"[atomic] candidate {k + 1}: {finals[-1]}")

        numbered = "\n".join(f"{i + 1}. {ans}" for i, ans in enumerate(finals))
        numbered = f"problem: {problem}, {numbered}"
        logging.info(f"[atomic] composition_discriminator query: {numbered}")
        votes = [0] * len(finals)
        winner_idx = None
        for _ in range(NUMBER_OF_VOTES):
            vresp = self.call_agent(self.composition_discriminator_session(), f"{numbered}\n\n")
            vote_txt = self.extract_final(vresp)
            logging.info(f"[atomic] solution vote: {vote_txt}")
            try:
                idx = int(vote_txt) - 1
                if idx >= len(finals):
                    logging.error(f"Invalid solution index: {idx}")
                if 0 <= idx < len(finals):
                    votes[idx] += 1
                    logging.info(f"[atomic] tally: {votes}")
                    if votes[idx] >= WINNING_VOTE_COUNT:
                        winner_idx = idx
                        logging.info(f"[atomic] early solution winner: {winner_idx + 1}")
                        break
            except ValueError:
                logging.warning(f"[atomic] malformed vote ignored: {vote_txt!r}")

        if winner_idx is None:
            winner_idx = max(range(len(votes)), key=lambda i: votes[i])

        logging.info(f"[atomic] final (chosen): {finals[winner_idx]!r}")

        return solutions[winner_idx], finals, votes, winner_idx, solutions

    def solve_trace(self, problem: str, depth: int, max_depth: int, path: str) -> tuple[str, dict]:
        """
        Internal recursive solver that returns (response, trace_node).
        Builds a complete trace tree of the decomposition process.
        """
        logging.info(f"[solve] depth={depth} path={path} problem: {problem[:120]}{'...' if len(problem) > 120 else ''}")

        node = {
            "depth": depth,
            "path": path,
            "problem": problem,
            "decomposition": None,
            "children": [],
            "sub_finals": None,
            "composition": None,
            "response": None,
            "final": None,
            "final_num": None,
            "error": None,
        }

        if depth >= max_depth:
            logging.info(f"[solve] depth={depth} -> atomic (max depth)")
            resp, finals, votes, winner_idx, solutions = self._solve_atomic_with_voting(problem)
            node["response"] = resp
            node["final"] = finals[winner_idx]
            node["atomic"] = {
                "atomic_candidates": finals,
                "atomic_votes": votes,
                "atomic_winner_idx": winner_idx,
                "final_choice": finals[winner_idx],
            }
            return resp, node

        p1, p2, c, decomp_meta = self.decompose(problem)

        if not p1 or not p2 or not c:
            logging.info(f"[solve] depth={depth} -> atomic (no decomp)")
            if decomp_meta:
                node["decomposition"] = {**decomp_meta, "decision": "no_decomposition"}
            resp, finals, votes, winner_idx, solutions = self._solve_atomic_with_voting(problem)
            node["response"] = resp
            node["final"] = finals[winner_idx]
            node["atomic"] = {
                "atomic_candidates": finals,
                "atomic_votes": votes,
                "atomic_winner_idx": winner_idx,
                "final_choice": finals[winner_idx],
            }
            return resp, node

        logging.info(f"[solve] depth={depth} using decomposition")
        node["decomposition"] = decomp_meta

        s1_resp, s1_node = self.solve_trace(p1, depth + 1, max_depth, f"{path}.0")
        s2_resp, s2_node = self.solve_trace(p2, depth + 1, max_depth, f"{path}.1")
        node["children"] = [s1_node, s2_node]

        s1 = self.extract_final(s1_resp)
        s2 = self.extract_final(s2_resp)
        node["sub_finals"] = {"s1_final": s1, "s2_final": s2}

        logging.info(f"[solve] depth={depth} sub-answers -> s1_final={s1!r}, s2_final={s2!r}")

        comp_prompt = self._compose_prompt(c, s1, s2)
        logging.info(f"[solve] depth={depth} composing with C={c!r}")

        solutions: list[str] = []
        finals: list[str] = []
        for k in range(SOLUTION_CANDIDATE_COUNT):
            r = self.call_agent(self.problem_solver_session(), comp_prompt)
            solutions.append(r)
            finals.append(self.extract_final(r))
            logging.info(f"[solve] depth={depth} composed candidate {k + 1}: {finals[-1]}")

        numbered = "\n".join(f"{i + 1}. {ans}" for i, ans in enumerate(finals))
        numbered = f"problem: {comp_prompt}, {numbered}"
        logging.info(f"[solve] depth={depth} composition_discriminator query: {numbered}")
        votes = [0] * len(finals)
        winner_idx = None
        for _ in range(NUMBER_OF_VOTES):
            vresp = self.call_agent(self.composition_discriminator_session(), f"{numbered}\n\n")
            vote_txt = self.extract_final(vresp)
            logging.info(f"[solve] depth={depth} solution vote: {vote_txt}")
            try:
                idx = int(vote_txt) - 1
                if idx >= len(finals):
                    logging.error(f"Invalid solution index: {idx}")
                if 0 <= idx < len(finals):
                    votes[idx] += 1
                    logging.info(f"[solve] depth={depth} tally: {votes}")
                    if votes[idx] >= WINNING_VOTE_COUNT:
                        winner_idx = idx
                        logging.info(f"[solve] depth={depth} early solution winner: {winner_idx + 1}")
                        break
            except ValueError:
                logging.warning(f"[solve] depth={depth} malformed vote ignored: {vote_txt!r}")

        if winner_idx is None:
            winner_idx = max(range(len(votes)), key=lambda i: votes[i])

        resp = solutions[winner_idx]
        node["response"] = resp
        node["final"] = finals[winner_idx]
        node["composition"] = {
            "c_text": c,
            "composed_candidates": finals,
            "composition_votes": votes,
            "composition_winner_idx": winner_idx,
            "final_choice": finals[winner_idx],
        }

        logging.info(f"[solve] depth={depth} composed final (chosen): {finals[winner_idx]!r}")

        return resp, node

    def decompose(self, problem: str) -> tuple[str | None, str | None, str | None, dict]:
        """
        Collect CANDIDATE_COUNT decompositions from the 'decomposer' agent,
        then run a voting round via 'solution_discriminator'.
        Returns (p1, p2, c, metadata_dict).
        """
        candidates: list[str] = []
        for _ in range(CANDIDATE_COUNT):
            resp = self.call_agent(self.decomposer_session(), problem)
            cand = self._extract_decomposition_text(resp)
            if cand:
                candidates.append(cand)

        for i, c in enumerate(candidates, 1):
            logging.info(f"[decompose] candidate {i}: {c}")

        if not candidates:
            return None, None, None, {}

        numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))
        numbered = f"problem: {problem}, {numbered}"
        logging.info(f"[decompose] solution_discriminator query: {numbered}")

        votes = [0] * len(candidates)
        winner_idx = None
        for _ in range(NUMBER_OF_VOTES):
            disc_prompt = f"{numbered}\n\n"
            vresp = self.call_agent(self.solution_discriminator_session(), disc_prompt)
            vote_txt = self.extract_final(vresp)
            logging.info(f"[decompose] discriminator raw vote: {vote_txt}")
            try:
                idx = int(vote_txt) - 1
                if idx >= len(candidates):
                    logging.error(f"Invalid vote index: {idx}")
                if 0 <= idx < len(candidates):
                    votes[idx] += 1
                    logging.info(f"[decompose] tally: {votes}")
                    if votes[idx] >= WINNING_VOTE_COUNT:
                        winner_idx = idx
                        logging.info(f"[decompose] early winner: {winner_idx + 1}")
                        break
            except ValueError:
                logging.warning(f"[decompose] malformed vote ignored: {vote_txt!r}")

        if winner_idx is None:
            winner_idx = max(range(len(votes)), key=lambda v: votes[v])

        logging.info(f"[decompose] final winner: {winner_idx + 1} -> {candidates[winner_idx]}")

        p1, p2, c = self._parse_decomposition(candidates[winner_idx])

        metadata = {
            "candidates": candidates,
            "winner_idx": winner_idx,
            "votes": votes,
            "chosen": candidates[winner_idx],
            "p1": p1,
            "p2": p2,
            "c": c,
        }

        return p1, p2, c, metadata
