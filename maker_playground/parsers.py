#
# Parsers for Tower of Hanoi domain
#
# These were the parsers used in the paper: https://arxiv.org/abs/2511.09030 (See Appendix)
#

import ast
import re


# Repair parser

def extract_balanced_brackets(text, start_idx):
    """Extract a substring with balanced brackets [[...]] starting at start_idx"""
    bracket_stack = []
    i = start_idx
    while i < len(text):
        if text[i] == '[':
            bracket_stack.append('[')
        elif text[i] == ']':
            if not bracket_stack:
                break
            bracket_stack.pop()
            if not bracket_stack:
                return text[start_idx:i + 1]
        i += 1
    return text[start_idx:i] + ']'


def parse_move_state_repair(response_text):
    try:
        move_matches = re.findall(r"(?i)\bmove\b\s*=\s*(\[[^\[\]]*\])", response_text)
        if not move_matches:
            raise ValueError("No 'move' found in response.")
        move = ast.literal_eval(move_matches[-1].strip())
    except Exception as e:
        raise ValueError("Could not parse 'move' from response.") from e

    try:
        # Match last occurrence of 'next_state = [ [' with any whitespace
        pattern = re.compile(r"(?i)\bnext_state\b\s*=\s*(\[\s*\[)", re.DOTALL)
        matches = list(pattern.finditer(response_text))
        if not matches:
            raise ValueError("No 'next_state' found in response.")
        start_idx = matches[-1].start(1)  # last match
        next_state_str = extract_balanced_brackets(response_text, start_idx).strip()
        next_state = ast.literal_eval(next_state_str)
    except Exception as e:
        raise ValueError("Could not parse 'next_state' from response.") from e

    return tuple(move), tuple(tuple(peg) for peg in next_state)


# Flag parser

def _validate_move(move):
    if not isinstance(move, list) or len(move) != 3 or not all(isinstance(x, int) for x in move):
        raise ValueError("'move' must be a list of exactly 3 integers.")
    return tuple(move)


def _validate_state(state, num_disks=20):
    if not (isinstance(state, list) and len(state) == 3 and all(isinstance(t, list) for t in state)):
        raise ValueError("'next_state' must be a list of three lists.")
    flat = [x for t in state for x in t]
    if not all(isinstance(x, int) for x in flat):
        raise ValueError("All entries in 'next_state' must be integers.")
    if len(flat) != num_disks or set(flat) != set(range(1, num_disks+1)):
        missing = sorted(set(range(1, 21)) - set(flat))
        extra = sorted(set(flat) - set(range(1, 21)))
        raise ValueError(f"State must contain 1..{num_disks} exactly once. "
                         f"Missing: {missing or '[]'}, Extras: {extra or '[]'}")
    return tuple(tuple(peg) for peg in state)


# pylint: disable=invalid-name
def parse_move_state_flag(response_text: str, num_disks=20):
    # Match square brackets
    move_pat = re.compile(r"(?is)\bmove\b\s*=\s*(\[[^\[\]]*\])")
    state_pat = re.compile(
        r"(?is)\bnext_state\b\s*=\s*(\[\s*\[[^\[\]]*\]\s*,\s*\[[^\[\]]*\]\s*,\s*\[[^\[\]]*\]\s*\])"
    )

    move_matches = list(move_pat.finditer(response_text))
    if not move_matches:
        raise ValueError("No 'move = [...]' found.")
    move_str = move_matches[-1].group(1)  # last 'move'

    state_matches = list(state_pat.finditer(response_text))
    if not state_matches:
        raise ValueError("No 'next_state = [[...],[...],[...]]' found.")
    state_str = state_matches[-1].group(1)  # last 'next_state'

    try:
        move = ast.literal_eval(move_str)
    except Exception as e:
        raise ValueError("Could not parse 'move' as a Python list.") from e
    try:
        next_state = ast.literal_eval(state_str)
    except Exception as e:
        raise ValueError("Could not parse 'next_state' as Python lists.") from e

    return _validate_move(move), _validate_state(next_state, num_disks)
