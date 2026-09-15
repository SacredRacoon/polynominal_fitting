import json
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import ExperimentPipeline

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    config_path = 'config.json'
    if not os.path.exists(config_path):
        logger.error(f"Configuration file {config_path} not found!")
        sys.exit(1)
        
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        pipeline = ExperimentPipeline(config)
        pipeline.run()
        
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
    except Exception as e:
        logger.error(f"Critical error {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()