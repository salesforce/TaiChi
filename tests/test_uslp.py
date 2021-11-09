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
    logger.info('importing uslp from taichi...')
    from taichi import uslp
    logger.info('ImportSuccess!')
except ImportError:
    logger.info('ImportError!')


# add all paths and hyperparameters in test config file    
config = "tests/test_uslp_config.json"


class ExperimentRunTest(unittest.TestCase):

    def test_run(self):
        u = uslp.USLP()
        logger.info('initialize the USLP model with training data and model parameters...')
        u.init()
        logger.info('initialization completed!')
        logger.info('begin the model training...')
        u.train()
        logger.info('model trained')
        logger.info('checking if trained model has been saved...')
        self.assertTrue(os.path.exists('./tmp'), 'tmp folder was not created!') # change path to saved_model_dir in test_uslp_config
        files = [f for f in os.walk('./tmp')]
        self.assertEqual(len(files[0]), 3, 'model or results.txt is missing!') # count number of files in saved_model_dir
        logger.info('model was saved successfully!')
        logger.info('begin the model evaluation...')
        u.eval()
        logger.info('checking if results have been saved...')
        self.assertTrue(os.path.exists('./tmp'), 'tmp folder was not created!') # change path to save_results_fp in test_uslp_config
        files = [f for f in os.walk('./tmp')]
        self.assertEqual(len(files[0]), 3, 'model or results.txt is missing!') # count number of files in saved_model_dir        
        shutil.rmtree('./tmp')
        logger.info('testing completed!')

if __name__ == '__main__':
    unittest.main()
