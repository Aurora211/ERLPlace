import argparse
parser = argparse.ArgumentParser("EfficientPlace ERL Edition")
parser.add_argument("-s", "--seed", type=int, default=42)
parser.add_argument("-g", "--gpu", type=str, default="0")
parser.add_argument("-f", "--folder", type=str, default="VSCode_Debug")
parser.add_argument("-b", "--benchmark", type=str, default="adaptec1")
parser.add_argument("-l", "--log", type=str, default="info")
args = parser.parse_args()
# ===================================================================================================================================================
import os
if os.path.exists(f'./results/{args.folder}') == False:
    os.makedirs(f'./results/{args.folder}', exist_ok=True)
if os.path.exists(f'./tensorboard/{args.folder}') == False:
    os.makedirs(f'./tensorboard/{args.folder}', exist_ok=True)
# ===================================================================================================================================================
import logging
logging_terminal_hander = logging.StreamHandler()
logging_file_handler = logging.FileHandler(f'./results/{args.folder}/log.log', mode='w', encoding='utf-8')
loglevel = str(args.log).upper()
if loglevel == "DEBUG":
    logging_terminal_hander.setLevel(logging.DEBUG)
    logging_file_handler.setLevel(logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', handlers=[logging_terminal_hander, logging_file_handler])
elif loglevel == "INFO":
    logging_terminal_hander.setLevel(logging.INFO)
    logging_file_handler.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', handlers=[logging_terminal_hander, logging_file_handler])
elif loglevel == "WARNING":
    logging_terminal_hander.setLevel(logging.WARNING)
    logging_file_handler.setLevel(logging.WARNING)
    logging.basicConfig(level=logging.WARNING, format='[%(asctime)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', handlers=[logging_terminal_hander, logging_file_handler])
elif loglevel == "ERROR":
    logging_terminal_hander.setLevel(logging.ERROR)
    logging_file_handler.setLevel(logging.ERROR)
    logging.basicConfig(level=logging.ERROR, format='[%(asctime)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', handlers=[logging_terminal_hander, logging_file_handler])
elif loglevel == "CRITICAL":
    logging_terminal_hander.setLevel(logging.CRITICAL)
    logging_file_handler.setLevel(logging.CRITICAL)
    logging.basicConfig(level=logging.CRITICAL, format='[%(asctime)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', handlers=[logging_terminal_hander, logging_file_handler])
else:
    logging_terminal_hander.setLevel(logging.INFO)
    logging_file_handler.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', handlers=[logging_terminal_hander, logging_file_handler])
logger = logging.getLogger(__name__)
logger.info(f"args: {args}")
# ===================================================================================================================================================
import yaml
from typing import Dict
if os.path.exists(f"./config.yaml"):
    try:
        with open(f"./config.yaml", "r") as f:
            global_config: Dict = yaml.load(f, Loader=yaml.FullLoader)
    except Exception as e:
        logger.critical("Global config file is not a valid yaml file.")
        logger.critical(e)
        exit()
else:
    logger.critical("Global config file not found.")
    exit()
if os.path.exists(f"./configs/{args.benchmark}.yaml"):
    try:
        with open(f"./configs/{args.benchmark}.yaml", "r") as f:
            benchmark_config: Dict = yaml.load(f, Loader=yaml.FullLoader)
    except Exception as e:
        logger.critical("Benchmark config file is not a valid yaml file.")
        logger.critical(e)
        exit()
else:
    logger.critical("Benchmark config file not found.")
    exit()
# ===================================================================================================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:{str(global_config.get("max_split_size_mb", 512))}'
os.environ['OMP_NUM_THREADS'] = str(global_config.get("number_of_cpus", 8))
os.environ['OPENBLAS_NUM_THREADS'] = str(global_config.get("number_of_cpus", 8))
os.environ['MKL_NUM_THREADS'] = str(global_config.get("number_of_cpus", 8))
os.environ['VECLIB_MAXIMUM_THREADS'] = str(global_config.get("number_of_cpus", 8))
os.environ['NUMEXPR_NUM_THREADS'] = str(global_config.get("number_of_cpus", 8))
# ===================================================================================================================================================
import random
import numpy as np
import torch
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.set_num_threads(int(global_config.get("number_of_cpus", 8)))
# ===================================================================================================================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info("Using GPU: " + str(torch.cuda.get_device_name()))
else:
    device = torch.device("cpu")
    logger.info("Using CPU")
# ===================================================================================================================================================
from tensorboardX import SummaryWriter
writer = SummaryWriter(f'./tensorboard/{args.folder}')
# ===================================================================================================================================================
from typing import List, Tuple
# ===================================================================================================================================================
from models import Agent
from envs import Environment_ISPD2005 as Environment
from trainer import Trainer
# ===================================================================================================================================================
logger.info("Initialization started.")
# ===================================================================================================================================================
population: List[Tuple[int, Agent, Dict]] = []
for i in range(benchmark_config.get("population_size", 10)):
    agent = Agent(benchmark_config, i)
    statistics = {
        'hpwl': [],
        'last': 1e9,
        'best': 1e9,
        'reinforce': False,
        'crossover': False,
        'rollback': False,
        'mutation': False,
        'elite': False,
        'from': 'ea',
    }
    agent.set_up(device, writer)
    population.append((i, agent, statistics))
# ===================================================================================================================================================
environment: Environment = Environment(
    args.benchmark,
    f"./benchmarks/{args.benchmark}",
    benchmark_config,
)
# ===================================================================================================================================================
trainer: Trainer = Trainer(
    benchmark_config,
    f'./results/{args.folder}'
)
trainer.setup_tb_writer(writer)
# ===================================================================================================================================================
logger.info("Training started.")
# ===================================================================================================================================================
trainer.train(
    population,
    environment,
    device,
)
# ===================================================================================================================================================
logger.info("Training finished.")


