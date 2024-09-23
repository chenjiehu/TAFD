BATCH_SIZE = 4
EPOCH_NUMNBER = 100

DATASET = ['CamVid', 6]

crop_size = (256,256)
crop_size_label = (256, 256)

class_dict_path = './Datasets/' + DATASET[0] + '/class_dict.csv'
TRAIN_ROOT = './Datasets/'+ DATASET[0] + '/train'
TRAIN_LABEL = './Datasets/' + DATASET[0] + '/train_labels'
VAL_ROOT = './Datasets/' + DATASET[0] + '/val'
VAL_LABEL = './Datasets/' + DATASET[0] + '/val_labels'
TEST_ROOT = './Datasets/'+ DATASET[0] + '/test'
TEST_LABEL = './Datasets/'+DATASET[0] +'/test_labels'