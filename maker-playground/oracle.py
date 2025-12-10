#
# Oracle program for generating ground truth for Towers of Hanoi with an even number of disks.
#

from tqdm import tqdm
from toh_simulator import TowerOfHanoi

def oracle_step(previous_move, current_state):

    pegs = [list(peg) for peg in current_state]

    move = None

    if previous_move is None:
        move = [1, 0, 1] # First move

    elif previous_move[0] == 1: # If moved disk 1 last move, make only other legal move

        for from_peg in range(3):
            if len(pegs[from_peg]) > 0:
                top_disk = pegs[from_peg][-1]
                if top_disk > 1:
                    for to_peg in range(3):
                        if len(pegs[to_peg]) == 0 or pegs[to_peg][-1] > top_disk:
                            move = [top_disk, from_peg, to_peg]

    else: # Otherwise, move disk 1 clockwise

        for from_peg in range(3):
            if len(pegs[from_peg]) > 0:
                if pegs[from_peg][-1] == 1:
                    to_peg = (from_peg + 1) % 3
                    move = [1, from_peg, to_peg]

    disk = move[0]
    from_peg = move[1]
    to_peg = move[2]

    pegs[from_peg].pop()
    pegs[to_peg].append(disk)

    next_state = [list(pegs[0]), list(pegs[1]), list(pegs[2])]

    return move, next_state


def generate_oracle_data(n_disks):

    N = n_disks
    max_moves = 2**N
    toh = TowerOfHanoi(N)
    current_state = [[i+1 for i in range(N)][::-1], [], []]
    previous_move = "null"
    oracle_states = [current_state]
    oracle_moves = [previous_move]
    for move_idx in tqdm(range(max_moves)):
        previous_move, current_state = call_oracle(previous_move,
                                                   current_state)

        sim_state, move_valid, done, sim_message = toh.act(previous_move)

        oracle_moves.append(previous_move)
        oracle_states.append(current_state)

        if not move_valid:
            print("INVALID MOVE!")
            break

        if toh.is_solved():
            print("Disks:", N, "; Move:", move_idx, "; Done.")
            break

    return oracle_moves, oracle_states
