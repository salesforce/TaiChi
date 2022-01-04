import os
import time
import shutil
import unittest
import sys
sys.path.insert(0, '../')
    
import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

try:
    logger.info('importing dnnc from taichi...')
    from taichi import dnnc
    logger.info('ImportSuccess!')
except ImportError:
    logger.info('ImportError!')

    

# add all paths and hyperparameters in test config file    
config = "test_dnnc_config.json"


class ExperimentRunTest(unittest.TestCase):

    def test_run(self):
        d = dnnc.DNNC(config)
        logger.info('initialize the DNNC model with training data and model parameters...')
        d.init()
        logger.info('initialization completed!')
        logger.info('begin the model training...')
        d.train()
        logger.info('model trained')
        logger.info('checking if trained model has been saved...')
        self.assertTrue(os.path.exists(d.config.checkpoint_dir), 'saved model folder was not created!') # change path to checkpoint_dir in test_uslp_config
        files = [f for f in os.walk(d.config.checkpoint_dir)]
        logger.info(files[0])
        self.assertEqual(len(files[0][-1]), 2, 'model checkpoint file is missing!') # count number of files in checkpoint_dir
        logger.info('model was saved successfully!')
        logger.info('begin the model evaluation...')
        d.eval()
        logger.info('checking if results have been saved...')
        self.assertTrue(os.path.exists(d.config.error_analysis_dir), 'error analysis folder was not created!') # change path to save_results_fp in test_uslp_config
        files = [f for f in os.walk(d.config.error_analysis_dir)]
        self.assertEqual(len(files[0][-1]), 5, 'error analysis files are missing!') # count files in error_analysis_dir
        self.assertTrue(os.path.exists(d.config.save_result_fp), 'results file missing') # check if result file created 
        shutil.rmtree(d.config.checkpoint_dir)
        shutil.rmtree(d.config.error_analysis_dir)
        os.remove(d.config.save_result_fp)
        logger.info('testing completed!')

if __name__ == '__main__':
    unittest.main()
