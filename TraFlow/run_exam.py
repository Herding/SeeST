import argparse

from examples import agcrn_example
from examples import dcrnn_example
from examples import graphwave_example
from examples import mtgnn_example
from examples import stgcn_examples


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run an example')
    parser.add_argument('--model',  help='choose a model to train')
    args = parser.parse_args()

    if args.model == 'AGCRN':
        agcrn_example.run()
    elif args.model == 'DCRNN':
        dcrnn_example.run()
    elif args.model == 'GraphWave':
        graphwave_example.run()
    elif args.model == 'MTGNN':
        mtgnn_example.run()
    elif args.model == 'STGCN':
        stgcn_examples.run()
    else:
        print('Model is not defined!')
