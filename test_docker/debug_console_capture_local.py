#!/usr/bin/env python3
import time
import sys
import logging

from certain_library.tracking.tracker import tracker

# quick smoke test for console capture: create experiment and run, then print many lines
experiment_name = "local_debug_capture"
tracker.set_experiment(experiment_name=experiment_name)

with tracker.start_run(run_name="debug_capture_test") as run:
    print(f"Run started: {run.info.run_id}")
    sys.stdout.flush()

    # emit many print lines and logging messages
    logger = logging.getLogger("debug_logger")
    for i in range(1000):
        print(f"PRINT_LINE {i}")
        logger.info(f"LOG_LINE {i}")
        if i % 100 == 0:
            sys.stdout.flush()
        # tiny sleep to allow threads to process
        time.sleep(0.001)

    print("Run complete")
    sys.stdout.flush()

print("Script finished")
