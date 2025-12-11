from __future__ import annotations

from typing import Dict, List, Tuple
from copy import deepcopy
import sys

Move = List[int]          # [disk_id, from_peg, to_peg]
State = Dict[str, List[int]]   # three pegs, each a stack (bottom->top left->right)
Plan = List[Move]         # list of moves


class TowerOfHanoi:
    """Tower of Hanoi puzzle simulator

    This module implements a deterministic simulator for the Tower of Hanoi puzzle as
    described in *The Illusion of Thinking* (Shojaee et al.,2025), Appendix A.1.1.

    It is designed so other programs (e.g. LLM evaluation harnesses) can verify a
    sequence of moves produced by a model without requiring any custom logic.  The
    simulator focuses **only** on state-tracking and rule-enforcement — it does *not*
    include any search or solving heuristics.

    It respects the OpenAI Gym interface, i.e. it has a `reset` method to
    initialize the puzzle, an `act` method to execute a single move, and an
    `apply_moves` method to execute a sequence of moves atomically.

    Example usage:
        from tower_of_hanoi_simulator import TowerOfHanoiSimulator

        sim = TowerOfHanoiSimulator(num_disks=3)
        moves = [[1, 0, 2], [2, 0, 1], [1, 2, 1],
                [3, 0, 2], [1, 1, 0], [2, 1, 2], [1, 0, 2]]

        ok, msg, final_state = sim.apply_moves(moves)
        assert ok and sim.is_solved()

    Internally each peg is represented as a *stack* (Python list) whose **last**
    element is the *top* disk.  Disk identifiers are integers ``1…N`` where
    ``1`` is the *smallest* disk and ``N`` the largest.  The initial state is::

        [[N, N-1, …, 2, 1], [], []]

    i.e. peg 0 contains the full tower, peg 1 and peg 2 are empty.
    """
    def __init__(self, num_disks: int):
        if num_disks < 1:
            raise ValueError("num_disks must be >= 1")
        self._num_disks = num_disks
        self.reset()

    def reset(self) -> Tuple[State, int]:
        """Reset the puzzle to its initial configuration."""
        self._state: State = {'0': list(range(self._num_disks, 0, -1)), '1': [], '2': []}
        self._move_count: int = 0
        return self.get_state(), 0

    def copy_state(self) -> State:
        """Return a *deep copy* of the current state (safe for mutation)."""
        return deepcopy(self._state)

    def get_state(self) -> State:
        """Return the current state of the puzzle.
        """
        return self._state

    def is_solved(self) -> bool:
        """Return ``True`` iff all disks are stacked (largest→smallest) on peg 2."""
        return (
            len(self._state['2']) == self._num_disks
            and self._state['2'] == list(range(self._num_disks, 0, -1))
        )

    def act(self, action: Move) -> Tuple[State, int, bool, dict]:
        """Attempt to execute *one* move.

        Parameters
        ----------
        move
            An iterable ``[disk_id, from_peg, to_peg]``.

        Returns
        -------
        (state, ok, done, message)
        state
            The current state of the puzzle after the move.
        ok
            A boolean indicating whether the move was valid.
        done
            A boolean indicating whether the simulation has stopped.
        message
            A dictionary with a human-readable error message if the move was invalid.
        """
        try:
            disk_id, from_peg, to_peg = map(int, action)
        except (TypeError, ValueError):
            return self.get_state(), 0, True, {"reason": "move must be a sequence of three ints [disk, from, to]"}

        # Basic input validation
        if disk_id < 1 or disk_id > self._num_disks:
            return self.get_state(), 0, True, {"reason": f"disk_id {disk_id} out of range"}
        if from_peg not in (0, 1, 2) or to_peg not in (0, 1, 2):
            return self.get_state(), 0, True, {"reason": "peg indices must be 0, 1, or 2"}
        if from_peg == to_peg:
            return self.get_state(), 0, True, {"reason": "from_peg and to_peg must differ"}

        src_stack = self._state[str(from_peg)]
        dst_stack = self._state[str(to_peg)]

        # Rule 1: source peg must not be empty
        if not src_stack:
            return self.get_state(), 0, True, {"reason": f"peg {from_peg} is empty"}

        # Rule 2: the specified disk must be the *top* disk on the source peg
        top_disk = src_stack[-1]
        if disk_id != top_disk:
            return self.get_state(), 0, True, {
                "reason": f"disk {disk_id} is not on top of peg {from_peg} (top is {top_disk})"
            }

        # Rule 3: cannot place larger disk atop a smaller one
        if dst_stack and dst_stack[-1] < disk_id:
            return self.get_state(), 0, True, {
                "reason": f"illegal move: cannot place larger disk {disk_id} on smaller disk {dst_stack[-1]}"
            }

        # Perform the move
        src_stack.pop()          # remove from source
        dst_stack.append(disk_id)  # push onto destination
        self._move_count += 1
        return self.get_state(), 1, False, {"reason": "ok"}

    # pylint: disable=redefined-outer-name
    def apply_moves(self, moves: Plan) -> Tuple[State, int, bool, dict]:
        """Execute a sequence of moves atomically.

        If any move is invalid, the simulator stops *immediately* and returns the
        error along with the move index (0‑based).  The internal state *does* keep
        partial progress up to the failure point – call :py:meth:`reset` to start
        over.
        """
        done = False
        moves_executed = 0
        for idx, mv in enumerate(moves):
            state, ok, done, msg = self.act(mv)
            moves_executed += ok
            if not ok:
                return state, moves_executed, done, {"reason": f"move {idx}: {msg['reason']}"}
        return self.get_state(), len(moves), done, {"reason": "all moves valid"}

    def minimal_solution_length(self) -> int:
        """Return the theoretical minimum number of moves ``2ⁿ - 1``."""
        return (1 << self._num_disks) - 1


if __name__ == "__main__":

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    sim = TowerOfHanoi(n)
    print("Initial state:", sim.get_state())
    print("Minimal solution length:", sim.minimal_solution_length())

    moves = [
        [1, 0, 2], [2, 0, 1], [1, 2, 1],
        [3, 0, 2], [1, 1, 0], [2, 1, 2], [1, 0, 2]
    ]

    moves = [
        [3, 1, 2],
    ]

    state, ok, done, msg = sim.apply_moves(moves)
    print("Result:", msg)
    print("Moves executed:", ok)
    print("Final state:", state)
    print("Solved?", sim.is_solved())
