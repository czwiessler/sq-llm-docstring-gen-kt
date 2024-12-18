import chess
import time
from evaluation.antichess_eval import *
from evaluation.threecheck_eval import threecheck_eval
import random
import chess.variant

inf = float('inf')
poscount = 0

DEPTH = 3

def search(node, color, variant, depth):
	moves = list(node.legal_moves)

	if not moves:
		print('Game over.')
		return

	move = negamax(node, -inf, inf, color, variant, depth)[1]

	if not move:
		return random.choice(moves)
	else:
		return move

def negamax(node, a, b, color, variant, depth=DEPTH):
	global poscount

	if (depth == 0):
		return (threecheck_eval(node, color, variant) * color, None)

	moves = list(node.legal_moves)

	best_move = None
	best_value = -inf

	for move in moves:
		poscount+=1

		node.push(move)
		result = negamax(node, -b, -a, -color, variant, depth-1)
		value = -result[0]
		node.pop()
		if value > best_value:
			best_value = value
			best_move = move

		a = max(a, value)

		if a >= b:
			break

	return (best_value, best_move)


if __name__ == "__main__":
	board = chess.variant.ThreeCheckBoard()
	moves = []

	c = 0

	while not board.is_variant_end():
		if c%2==0:
			move = input("move: \n\n")
			move = chess.Move.from_uci(move)
			if not move in board.legal_moves:
				continue
		else:
			start_time = time.time()

			#book_move = book.get_moves(moves)
			book_move = None
			if book_move:
				move = random.choice(book_move)
				move = chess.Move.from_uci(move)
			else:
				move = search(board, color=-1, variant="3check", depth=DEPTH)
			elapsed = time.time() - start_time
			print("--- %s moves ---" % (len(list(board.legal_moves))))
			print("--- number of nodes: %s --" % poscount)
			print("--- %s seconds ---" % (elapsed))
			print("--- nodes per second: %s ---" % str(poscount / elapsed))

		print(move)
		moves.append(str(move))
		board.push(move)
		c+=1