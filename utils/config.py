from bunch import Bunch
import logging
from collections import OrderedDict
import argparse
import wandb
import datetime
import os

logger = logging.getLogger("logger")


def GetArgs():
    """" collects command line arguments """

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--config', default=None, type=str, help='configuration file')
    argparser.add_argument('--wandb_project_name', default=None, type=str, help='wandb project name')
    argparser.add_argument('--run', default='debug', type=str, help='Run mode')
    argparser.add_argument('--experiment', default='synthetic_mot', type=str, help='Experiment type')
    argparser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    argparser.add_argument('--epochs', default=55, type=int, help='Number of epochs')
    argparser.add_argument('--lr', default=5e-4, type=float, help='Learning rate')
    argparser.add_argument('--n', default=5000, type=int, help='Sample size')
    argparser.add_argument('--k', default=3, type=int, help='Number of iterations')
    argparser.add_argument('--eps', default=0.5, type=float, help='Epsilon value for regularisation')
    argparser.add_argument('--cost', default='quad', type=str, choices=['quad', 'quad_gw', 'ip_gw'], help='Cost function')
    argparser.add_argument('--alg', default='ne_mot', type=str, choices=['ne_mot', 'sinkhorn_mot', 'ne_gw', 'sinkhorn_gw'],
                        help='Algorithm')
    argparser.add_argument('--hidden_dim', default=32, type=int, help='Dimension of hidden layers')
    argparser.add_argument('--mod', default='mot', type=str, choices=['mot', 'mgw'], help='Model type')
    argparser.add_argument('--seed', default=42, type=int, help='Random seed')
    argparser.add_argument('--data_dist', default='uniform', type=str, help='Data distribution type')
    argparser.add_argument('--dims', default=[13, 13, 13, 13, 13], nargs='+', type=int, help='Dimensions of the data')
    argparser.add_argument('--dim', default=13, nargs='+', type=int, help='Dimensions of the data')
    argparser.add_argument('--device', default='gpu', type=str, help='Device to use')
    argparser.add_argument('--cuda_visible', default=3, type=int, help='CUDA visible device')
    argparser.add_argument('--using_wandb', default=0, type=int, help='Use Weights & Biases logging')
    argparser.add_argument('--cost_graph', default='full', type=str, choices=['full', 'circle', 'tree'],
                        help='Graphical structure of the cost function')


    argparser.set_defaults(quiet=False)

    args = argparser.parse_args()
    return args


def PreprocessMeta():
    """
    steps:
    0. get config
    1. parse args
    2. initiate wandb
    """
    args = GetArgs()
    config = GetConfig(args)

    # add wandb:
    if config.using_wandb:
        wandb_proj = "mot" if not hasattr(config, 'wandb_project_name') else config.wandb_project_name
        wandb.init(project=wandb_proj,
                   entity=config.wandb_entity,
                   config=config)
    return config


def GetConfig(args):
    config = {
        'run': 'debug',
        'experiment': 'synthetic_mot',
        'batch_size': 64,
        'epochs': 55,
        'lr': 5e-4,
        'n': 5000,
        'k': 3,
        'eps': 0.5,
        'cost': 'quad',  # options - quad, quad_gw, ip_gw
        'alg': 'ne_mot',  # options - ne_mot, sinkhorn_mot,ne_gw, sinkhorn_gw
        'hidden_dim': 32,
        'mod': 'mot',  # options - mot, mgw
        'seed': 42,
        'data_dist': 'uniform',
        # 'dims': [1,1,1,1,1,1,1,1],
        # 'dims': [100,100,100,100,100,100,100,100],
        'dims': [13,13,13,13,13],
        'device': 'gpu',
        'cuda_visible': 3,
        'using_wandb': 0,
        'cost_graph': 'full'  # The cost function graphical structure for decomposition. Options - full, circle, tree
    }
    # TD: ADJUST DIMS TO K
    config['batch_size'] = min(config['batch_size'], config['n'])

    # Turn into Bunch object
    config = Config(config)

    # Add args values to the config attributes
    for key in sorted(vars(args)):
        val = getattr(args, key)
        if val is not None:
            setattr(config, key, val)


    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    config.figDir = f"results/{config.run}/{config.alg}/k_{config.k}_stamp_{now_str}"
    os.makedirs(config.figDir, exist_ok=True)

    config.print()
    return config


class Config(Bunch):
    """ class for handling dictionary as class attributes """
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
    def print(self):
        line_width = 132
        line = "-" * line_width
        logger.info(line + "\n" +
              "| {:^35s} | {:^90} |\n".format('Feature', 'Value') +
              "=" * line_width)
        for key, val in sorted(self.items(), key= lambda x: x[0]):
            if isinstance(val, OrderedDict):
                raise NotImplementedError("Nested configs are not implemented")
            else:
                logger.info("| {:35s} | {:90} |\n".format(key, str(val)) + line)
        logger.info("\n")