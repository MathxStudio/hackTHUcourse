import os

CONSTFILEPATH = os.path.abspath(__file__)
PROJECT_PATH = os.path.dirname(os.path.dirname(CONSTFILEPATH))
DATAPATH = os.path.join(PROJECT_PATH, 'dataset')

SEGMENTED_presampling = os.path.join(DATAPATH, r'segmented')
SEGMENTED_aftersampling = os.path.join(DATAPATH, r'segmented_aftersampling')

SEGMENTED_afterpre = os.path.join(DATAPATH, r'segmented_afterpre')

TRAIN_SAMPLES_PATH = os.path.join(SEGMENTED_afterpre, r'train')
TEST_SAMPLES_PATH = os.path.join(SEGMENTED_afterpre, r'test')

BATCH_SIZE = 64

CLASSES = ['#', '+', '2', '3', '4', '6', '7', '8', '9', 'B', 'C', 'E', 'F',
           'G', 'H', 'J', 'K', 'M', 'P', 'Q', 'R', 'T', 'V', 'W', 'X', 'Y']
CLASSES_TO_ID = {k: i for i, k in enumerate(CLASSES)}
num_classes = len(CLASSES)

