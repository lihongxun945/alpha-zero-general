import logging

import coloredlogs

from Coach import Coach
from gobang.GobangGame import GobangGame as Game
from gobang.keras.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

board_size = 11
row_count = 5

args = dotdict({
    'numIters': 100,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 10,        # 蒙特卡洛搜索温度阈值，每一次self play的前tempThreshold 步，会采用更随机的策略，否则采用更优策略
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    # 'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'numMCTSSims': 400,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 20,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './checkpoint/gobang{}_{}'.format(board_size, row_count),
    'load_model': True, # 是否加载之前训练的模型
    'load_folder_file': ('./checkpoint/gobang{}_{}'.format(board_size, row_count), 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 40, # 神经网络要拟合的训练集
    'load_examples': True, # 是否加载之前的数据集

    'reduce': 0.95, # 衰减因子
    'board_size': board_size,
    'showMCTSInfo': False,

})

def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(board_size, row_count)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_examples:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
