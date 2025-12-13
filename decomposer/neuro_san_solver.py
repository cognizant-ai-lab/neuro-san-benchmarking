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

import logging
import os
import threading

from neuro_san.client.agent_session_factory import AgentSession
from neuro_san.client.streaming_input_processor import StreamingInputProcessor

from decomposer.session_manager import SessionManager
from decomposer.solver import Solver
from decomposer.solver_parsing import SolverParsing


class NeuroSanSolver(Solver):
    """
    Generic solver implementation that uses Neuro SAN.
    """

    def __init__(self, winning_vote_count: int = 2,
                 candidate_count: int = None,
                 number_of_votes: int = None,
                 solution_candidate_count: int = None):
        """
        Constructor.
        """

        self.winning_vote_count: int = winning_vote_count
        default_count: int = (2 * winning_vote_count) - 1

        self.candidate_count: int = candidate_count
        if self.candidate_count is None:
            self.candidate_count = default_count

        self.number_of_votes: int = number_of_votes
        if self.number_of_votes is None:
            self.number_of_votes = default_count

        self.solution_candidate_count: int = solution_candidate_count
        if self.solution_candidate_count is None:
            self.solution_candidate_count = default_count

        # Initialize the Neuro SAN agent sessions
        self.composition_discriminator_session: AgentSession = SessionManager.get_session("composition_discriminator")
        self.decomposer_session: AgentSession = SessionManager.get_session("decomposer")
        self.problem_solver_session: AgentSession = SessionManager.get_session("problem_solver")
        self.solution_discriminator_session: AgentSession = SessionManager.get_session("solution_discriminator")
        self.parsing = SolverParsing()

    def solve(self, problem: str, depth: int, max_depth: int, path: str = "0") -> dict[str, Any]:
        """
        Internal recursive solver that returns (response, trace_node).
        Builds a complete trace tree of the decomposition process.

        :return: The root trace node of the decomposition process
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
            "extracted_final": None,
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
            node["extracted_final"] = self.parsing.extract_final(resp)
            return node

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
            node["extracted_final"] = self.parsing.extract_final(resp)
            return node

        logging.info(f"[solve] depth={depth} using decomposition")
        node["decomposition"] = decomp_meta

        s1_node = self.solve(p1, depth + 1, max_depth, f"{path}.0")
        s2_node = self.solve(p2, depth + 1, max_depth, f"{path}.1")
        node["children"] = [s1_node, s2_node]

        s1: str = s1_node.get("extracted_final")
        s2: str = s2_node.get("extracted_final")
        node["sub_finals"] = {"s1_final": s1, "s2_final": s2}

        logging.info(f"[solve] depth={depth} sub-answers -> s1_final={s1!r}, s2_final={s2!r}")

        comp_prompt = self._compose_prompt(c, s1, s2)
        logging.info(f"[solve] depth={depth} composing with C={c!r}")

        solutions: list[str] = []
        finals: list[str] = []
        for k in range(self.solution_candidate_count):
            r = self.call_agent(self.problem_solver_session, comp_prompt)
            solutions.append(r)
            finals.append(self.parsing.extract_final(r))
            logging.info(f"[solve] depth={depth} composed candidate {k + 1}: {finals[-1]}")

        numbered = "\n".join(f"{i + 1}. {ans}" for i, ans in enumerate(finals))
        numbered = f"problem: {comp_prompt}, {numbered}"
        logging.info(f"[solve] depth={depth} composition_discriminator query: {numbered}")
        votes = [0] * len(finals)
        winner_idx = None
        for _ in range(self.number_of_votes):
            vresp = self.call_agent(self.composition_discriminator_session, f"{numbered}\n\n")
            vote_txt = self.parsing.extract_final(vresp)
            logging.info(f"[solve] depth={depth} solution vote: {vote_txt}")
            try:
                idx = int(vote_txt) - 1
                if idx >= len(finals):
                    logging.error(f"Invalid solution index: {idx}")
                if 0 <= idx < len(finals):
                    votes[idx] += 1
                    logging.info(f"[solve] depth={depth} tally: {votes}")
                    if votes[idx] >= self.winning_vote_count:
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
        node["extracted_final"] = self.parsing.extract_final(resp)

        logging.info(f"[solve] depth={depth} composed final (chosen): {finals[winner_idx]!r}")

        return node

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
        return self.call_agent(self.problem_solver_session, problem)

    def _solve_atomic_with_voting(self, problem: str) -> tuple[str, list[str], list[int], int, list[str]]:
        """
        Generate multiple atomic solutions and vote on them.
        Returns (chosen_response, finals, votes, winner_idx, solutions).
        """
        solutions: list[str] = []
        finals: list[str] = []
        for k in range(self.solution_candidate_count):
            r = self.call_agent(self.problem_solver_session, problem)
            solutions.append(r)
            finals.append(self.parsing.extract_final(r))
            logging.info(f"[atomic] candidate {k + 1}: {finals[-1]}")

        numbered = "\n".join(f"{i + 1}. {ans}" for i, ans in enumerate(finals))
        numbered = f"problem: {problem}, {numbered}"
        logging.info(f"[atomic] composition_discriminator query: {numbered}")
        votes = [0] * len(finals)
        winner_idx = None
        for _ in range(self.number_of_votes):
            vresp = self.call_agent(self.composition_discriminator_session, f"{numbered}\n\n")
            vote_txt = self.parsing.extract_final(vresp)
            logging.info(f"[atomic] solution vote: {vote_txt}")
            try:
                idx = int(vote_txt) - 1
                if idx >= len(finals):
                    logging.error(f"Invalid solution index: {idx}")
                if 0 <= idx < len(finals):
                    votes[idx] += 1
                    logging.info(f"[atomic] tally: {votes}")
                    if votes[idx] >= self.winning_vote_count:
                        winner_idx = idx
                        logging.info(f"[atomic] early solution winner: {winner_idx + 1}")
                        break
            except ValueError:
                logging.warning(f"[atomic] malformed vote ignored: {vote_txt!r}")

        if winner_idx is None:
            winner_idx = max(range(len(votes)), key=lambda i: votes[i])

        logging.info(f"[atomic] final (chosen): {finals[winner_idx]!r}")

        return solutions[winner_idx], finals, votes, winner_idx, solutions

    def decompose(self, problem: str) -> tuple[str | None, str | None, str | None, dict]:
        """
        Collect CANDIDATE_COUNT decompositions from the 'decomposer' agent,
        then run a voting round via 'solution_discriminator'.
        Returns (p1, p2, c, metadata_dict).
        """
        candidates: list[str] = []
        for _ in range(self.candidate_count):
            resp = self.call_agent(self.decomposer_session, problem)
            cand = self.parsing.extract_decomposition_text(resp)
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
        for _ in range(self.number_of_votes):
            disc_prompt = f"{numbered}\n\n"
            vresp = self.call_agent(self.solution_discriminator_session, disc_prompt)
            vote_txt = self.parsing.extract_final(vresp)
            logging.info(f"[decompose] discriminator raw vote: {vote_txt}")
            try:
                idx = int(vote_txt) - 1
                if idx >= len(candidates):
                    logging.error(f"Invalid vote index: {idx}")
                if 0 <= idx < len(candidates):
                    votes[idx] += 1
                    logging.info(f"[decompose] tally: {votes}")
                    if votes[idx] >= self.winning_vote_count:
                        winner_idx = idx
                        logging.info(f"[decompose] early winner: {winner_idx + 1}")
                        break
            except ValueError:
                logging.warning(f"[decompose] malformed vote ignored: {vote_txt!r}")

        if winner_idx is None:
            winner_idx = max(range(len(votes)), key=lambda v: votes[v])

        logging.info(f"[decompose] final winner: {winner_idx + 1} -> {candidates[winner_idx]}")

        p1, p2, c = self.parsing.parse_decomposition(candidates[winner_idx])

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
