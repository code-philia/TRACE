import os
import time

from Invoker import ask_invoker
from logic_gate import logic_gate
from utils import merge_predictions
def TRACE(commit, models, args, record, logger, LSP):
    """
    Use LSP & Invoker to predict the next edit location
    """
    if args.system == "TRACE":
        service, service_info = ask_invoker(commit, models, args, logger)
    elif args.system == "TRACE-wo-Invoker":
        # Invoker will not tell what service it is, service == "all"
        service, service_info = ask_invoker(commit, models, args, logger)
    
    predictions = None 
    # STEP 1: logic based code navigation
    # STEP 1.1: rename edit composition
    if service == "rename" or service == "all":
        start = time.time()
        predictions = LSP.process_rename(commit, service_info)
        end = time.time()
        record.locator_runtime[-1] += end - start
        if predictions is None or predictions == {}:
            logger.info(f"LSP rename returned empty")
        else:
            return predictions
        
    # STEP 1.2: def&ref edit composition
    if service == "def&ref" or service == "all":
        start = time.time()
        predictions = LSP.process_def_ref(service_info, commit, models, args)
        end = time.time()
        record.locator_runtime[-1] += end - start
        if predictions is None:
            logger.info(f"LSP def&ref returned empty")
        # we don't return prediction directly, as we may receive diagnostics from LSP
        
    # STEP 1.3: code clone edit composition
    if service == "clone" or service == "all":
        start = time.time()
        predictions = LSP.process_code_clone(service_info, commit, models, args)
        end = time.time()
        record.locator_runtime[-1] += end - start
        if predictions is None:
            logger.info(f"LSP clone returned empty")

    # STEP 2: error based code navigation
    try:
        start = time.time()
        diagnose_predictions = LSP.process_diagnose(commit, models, args)
        end = time.time()
        if diagnose_predictions is not None:
            record.locator_runtime[-1] += end - start
            if predictions is not None:
                # merge the predictions from logic and error
                predictions = merge_predictions(predictions, diagnose_predictions)
            else:
                predictions = diagnose_predictions
    except:
        return predictions
    
    return predictions

