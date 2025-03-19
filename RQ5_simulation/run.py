import json
import time
import torch
import random
import logging
import argparse

from utils import *
from simulation import simulation
from Invoker import load_invoker
from Locators import load_locator
from Generators import load_generator
from record import statistics_analyzer
from CoEdPilot_estimator import load_model_estimator
from CoEdPilot_dependency_analyzer import load_dep_model
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%Y/%m/%d %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def main(args):
    # Add logging of arguments
    logger.info("Running with arguments:")
    max_arg_length = max(len(arg) for arg in vars(args).keys())
    for arg, value in vars(args).items():
        logger.info(f"\t{arg:<{max_arg_length}}\t{value}")
    
    with open(args.testset_path, "r") as f:
        """
        testset must follow the format: 
        {
            "lang(py/go/java/js/ts)": [
                {
                    "commit_url": "https://github.com/user/repo/commit/commit_hash",
                    "edit_order_seq": list[list[int] | None], list[int] is the edit hunk ids, if sorted, should equal to range(edit_hunk_num)
                    "edit_order_graph": networkx.DiGraph() | None
                }
            ]
        }
        """
        testset = json.load(f)
    
    # Load models
    generator, generator_tokenizer = load_generator(args, logger)
    models = {
        "generator": generator,
        "generator_tokenizer": generator_tokenizer
    }
    if args.system == "TRACE":
        invoker, invoker_tokenizer = load_invoker(args, logger)
        models["invoker"] = invoker
        models["invoker_tokenizer"] = invoker_tokenizer
    if args.system != "CodeCloneDetector":
        locator, locator_tokenizer = load_locator(args, logger)
        models["locator"] = locator
        models["locator_tokenizer"] = locator_tokenizer
    if args.system == "CoEdPilot":
        # load dependency classifier
        dependency_analyzer, dependency_tokenizer = load_dep_model(args, logger)
        # load prior edit estimator
        estimator, estimator_tokenizer = load_model_estimator(dependency_analyzer, args, logger)
        models["estimator"] = estimator
        models["estimator_tokenizer"] = estimator_tokenizer
        models["dependency_tokenizer"] = dependency_tokenizer
    
    simulation_statistics = []
    for lang, test_samples in testset.items():
        if lang != args.lang:
            continue
        for sample_idx, test_sample in enumerate(test_samples):
            if sample_idx >= args.idx:
                continue
            logger.info(f"==> Start simulation of {test_sample['commit_url']}")
            simulation_staticstic = simulation(test_sample, args, lang, logger, models)
            simulation_staticstic.check()
            simulation_statistics.append(simulation_staticstic)
            logger.info(f"==> Simulation progress({lang}): {sample_idx+1}/{len(test_samples)}\n\n")
    
    # Use StatisticsAnalyzer to process the statistics
    statistics_analyzer(simulation_statistics, args, logger)
    
if __name__ == "__main__":
    random.seed(42)
    parser = argparse.ArgumentParser()
    if True:
        parser.add_argument("--system", 
                            type=str, 
                            choices= [
                                "TRACE", 
                                "TRACE-wo-Invoker", 
                                "PlainSemantics",
                                "EnrichedSemantics",
                                "CoEdPilot",
                                "CodeCloneDetector"
                            ],
                            required=True)
        parser.add_argument("--testset_path", type=str, required=True)
        parser.add_argument("--locator_model_path", type=str)
        parser.add_argument("--invoker_model_path", type=str)
        parser.add_argument("--generator_model_path", type=str)
        parser.add_argument("--dependency_model_path", type=str)
        parser.add_argument("--estimator_model_path", type=str)
        parser.add_argument("--output_dir", type=str, required=True)
        parser.add_argument("--locator_batch_size", type=int, default=1)
        parser.add_argument("--random_order", action="store_true", help="Simulate the edits in a random order. If False, simulate the edits in the given order from sequences or edit order graph")
        parser.add_argument("--label_correction", action="store_true", help="If True, user will correct the input label for generator, otherwise, the input label is the predicted label")
        parser.add_argument("--lsp_log", action="store_true", help="Log the LSP communication")
        parser.add_argument("--init_diagnose_msg", type=list, default=[])
        parser.add_argument("--debug", action="store_true", help="Debug mode")
        parser.add_argument("--lang", type=str)
        parser.add_argument("--idx", type=int)
        
    args = parser.parse_args()
    
    args.simulation_start_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.system == "TRACE":
        args.use_lsp = True
        args.use_invoker = True
        args.label_num = 6
    elif args.system == "TRACE-wo-Invoker":
        args.use_lsp = True
        args.use_invoker = False
        args.label_num = 6
    elif args.system == "EnrichedSemantics":
        args.use_lsp = False
        args.use_invoker = False
        args.label_num = 6
    elif args.system in ["PlainSemantics", "CoEdPilot", "CodeCloneDetector"]:
        args.use_lsp = False
        args.use_invoker = False
        args.label_num = 3
    
       
    main(args)