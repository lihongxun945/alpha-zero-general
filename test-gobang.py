import unittest

import Arena
import datetime
from MCTS import MCTS

from othello.OthelloPlayers import RandomPlayer
from gobang.GobangGame import GobangGame
from gobang.keras.NNet import NNetWrapper as GobangKerasNNet

import numpy as np
from utils import *

class TestAllGames(unittest.TestCase):

    @staticmethod
    def execute_game_test(game, neural_net):
        # 0 0 0 0 0 0
        # 0 0 0 0 0 0
        # 0 0 0 0 0 0
        # 0 0 1 1 0 0
        # 0 0 -1 0 0 0
        # 0 0 0 0 0 0
        # 0 0 0 0 0 0
        args = dotdict({'numMCTSSims': 25, 'cpuct': 1.0})
        board = game.getInitBoard()
        nnet = neural_net(game)
        start_time = datetime.datetime.now()
        predict = nnet.predict(game.getCanonicalForm(board, -1))
        end_time = datetime.datetime.now()
        delta_time = end_time - start_time
        print('predict ', "程序执行时间为：", delta_time.total_seconds()*1000, "毫秒", predict)
        nnet.load_checkpoint('./checkpoint/gobang6_4/', 'best.h5')
        mcts = MCTS(game, nnet, args)
        currentPlayer = 1
        board, currentPlayer = game.getNextState(board, currentPlayer, 20)
        board, currentPlayer = game.getNextState(board, currentPlayer, 26)
        board, currentPlayer = game.getNextState(board, currentPlayer, 21)
        game.display(board)
        actions = mcts.getActionProb(game.getCanonicalForm(board, -1), temp=0)
        print('actions:', actions, len(actions))
        move = np.argmax(actions)
        print('move:', int(move/6), move%6)
        p, v = nnet.predict(board=game.getCanonicalForm(board, -1))
        print('p:', p)
        print('v:', v)
        print(np.array(p).reshape(6, 6), v)

    def test_gobang_keras(self):
        self.execute_game_test(GobangGame(6, 4), GobangKerasNNet)


if __name__ == '__main__':
    unittest.main()
