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
config = "test_uslp_config.json"


class ExperimentRunTest(unittest.TestCase):

    def test_run(self):
        u = uslp.USLP(config)
        logger.info('initialize the USLP model with training data and model parameters...')
        u.init()
        logger.info('initialization completed!')
        logger.info('begin the model training...')
        u.train()
        logger.info('model trained')
        logger.info('checking if trained model has been saved...')
        self.assertTrue(os.path.exists(u.config.checkpoint_dir), 'model checkpoint directory was not created!')
        files = [f for f in os.walk(u.config.checkpoint_dir)]
        self.assertEqual(len(files[0][-1]), 2, 'model checkpoint files are missing!') # count files in checkpoint_dir
        logger.info('model was saved successfully!')
        logger.info('begin the model evaluation...')
        u.eval()
        logger.info('checking if results have been saved...')

        # check if result file created
        self.assertTrue(os.path.exists(u.config.save_result_fp), 'results file missing') 
        shutil.rmtree(u.config.checkpoint_dir)
        os.remove(u.config.save_result_fp)
        logger.info('testing completed!')

if __name__ == '__main__':
    unittest.main()
