# Prompt encoder for Towers of Hanoi domain
#
# These were the prompts used in the paper: https://arxiv.org/abs/2511.09030 (See appendix)
#

SYSTEM_PROMPT = """
You are a helpful assistant. Solve this puzzle for me.

There are three pegs and n disks of different sizes stacked on the first peg. The disks are
numbered from 1 (smallest) to n (largest). Disk moves in this puzzle should follow:
1. Only one disk can be moved at a time.
2. Each move consists of taking the upper disk from one stack and placing it on top of
another stack.
3. A larger disk may not be placed on top of a smaller disk.
The goal is to move the entire stack to the third peg.

Example: With 3 disks numbered 1 (smallest), 2, and 3 (largest), the initial state is [[3, 2, 1], [], []], and a solution might be:
moves = [[1, 0, 2], [2, 0, 1], [1, 2, 1], [3, 0, 2], [1, 1, 0], [2, 1, 2], [1, 0, 2]]
This means: Move disk 1 from peg 0 to peg 2, then move disk 2 from peg 0 to peg 1, and so on.

Requirements:
- The positions are 0-indexed (the leftmost peg is 0).
- Ensure your answer includes a single next move in this EXACT FORMAT:
```move = [disk id, from peg, to peg]```
- Ensure your answer includes the next state resulting from applying the move to the current state in this EXACT FORMAT:
```next_state = [[...], [...], [...]]```
"""

USER_TEMPLATE = """
Rules:
- Only one disk can be moved at a time.
- Only the top disk from any stack can be moved.
- A larger disk may not be placed on top of a smaller disk.

For all moves, follow the standard Tower of Hanoi procedure:
If the previous move did not move disk 1, move disk 1 clockwise one peg (0 -> 1 -> 2 -> 0).
If the previous move did move disk 1, make the only legal move that does not involve moving disk1.
Use these clear steps to find the next move given the previous move and current state.

Previous move: {previous_move}

Current State: {current_state}

Based on the previous move and current state, find the single next move that follows the procedure and the resulting next state.
"""

def create_prompts(previous_move, current_state):

    return SYSTEM_PROMPT, USER_TEMPLATE.format(previous_move=previous_move, current_state=current_state)
