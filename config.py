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
        'batch_size': 128,
        'epochs': 2000,
        'lr': 1e-4,
        'n': 50000,
        'k': 2,
        'eps': 0.5,
        'cost': 'quad',  # options - quad, quad_gw, ip_gw
        'alg': 'ne_mot',  # options - ne_mot, sinkhorn_mot,ne_gw, sinkhorn_gw
        'hidden_dim': 32,
        'mod': 'mot',  # options - mot, mgw
        'seed': 42,
        'data_dist': 'uniform',
        # 'dims': [1,1,1,1,1,1,1,1],
        # 'dims': [100,100,100,100,100,100,100,100],
        'dims': [5, 5, 5, 5, 5, 5],
        'device': 'gpu',
        'cuda_visible': 0,
        'using_wandb': 0
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
    config.figDir = f"results/{config.run}/{config.experiment}/k_{config.k}_stamp_{now_str}"
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