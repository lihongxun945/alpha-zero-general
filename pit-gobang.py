import Arena
from MCTS import MCTS
from gobang.GobangGame import GobangGame
from gobang.GobangPlayers import *
from gobang.keras.NNet import NNetWrapper as NNet


import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

board_size = 10 # Play in 6x6 instead of the normal 8x8.
row_count = 5
human_vs_cpu = True

g = GobangGame(board_size, row_count)

# all players
rp = RandomPlayer(g).play
gp = GreedyGobangPlayer(g).play
hp = HumanGobangPlayer(g).play



# nnet players
n1 = NNet(g)
n1.load_checkpoint('./checkpoint/gobang{}_{}'.format(board_size, row_count),'best.h5')
args1 = dotdict({
    'numMCTSSims': 200,
    'cpuct':1.0,
    'showMCTSInfo': True,
    'board_size': board_size,
})
mcts1 = MCTS(g, n1, args1)
def n1p(x):
  actions = mcts1.getActionProb(x, temp=0)
  print('actions', actions)
  return np.argmax(actions)

if human_vs_cpu:
    player2 = hp
# else:
#     n2 = NNet(g)
#     n2.load_checkpoint('./pretrained_models/othello/pytorch/', '8x8_100checkpoints_best.pth.tar')
#     args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
#     mcts2 = MCTS(g, n2, args2)
#     n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

    # player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

arena = Arena.Arena(n1p, player2, g, display=GobangGame.display)

print(arena.playGames(2, verbose=True))
