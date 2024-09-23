"""Training Script"""
#20230506 modified A TPN
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
import torch.optim
import argparse
from model_distangle.disentangle_module import Disentangle_new1
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose

from TADF_model import Feature_extractor, MGCN_TPN_Classifier_union_gai
from util.utils import set_seed, CLASS_LABELS
from dataset_GID5 import DataSet_GID5_te, get_shot_data
from dataset_WHDLD import DataSet_WHDLD_tr
from dataset_Seg import images_seg
import numpy as np
from utils import Averager, Averager_vector
from evalution_segmentaion import eval_semantic_segmentation_enhance, calc_semantic_segmentation_confusion_enhance, eval_semantic_segmentation_enhance_3
from Dice_loss import SoftDiceLoss, DiceLoss_gai
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 10 #12
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY_S = './dataset/WHDLD/'
DATA_DIRECTORY_T = './dataset/GID5_new/'
DATA_LIST_PATH = './dataset/GID5_new/train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '256,256'

# DATA_DIRECTORY_TARGET = './dataset/Target/'
# DATA_LIST_PATH_TARGET = './dataset/Target_list/train.txt'
INPUT_SIZE_TARGET = '256,256'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 5
GAN = 'Vanilla'
NUM_STEPS = 6
NUM_STEPS_STOP = 20000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
# RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
RESTORE_FROM = './dataset/DeepLab_resnet_pretrained_init-f81d91e8.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 3

SNAPSHOT_DIR_PRETRAIN = './snapshots/5shot/WHDLD_GID5_Pretrain'
WEIGHT_DECAY = 0.0005

TARGET = 'cityscapes'
SET = 'train'
LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001
SHOT_NUMBER = 3
INIT_PATH = './pretrained_model/resnet50-19c8e357.pth'

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : True")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir-s", type=str, default=DATA_DIRECTORY_S,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-dir-t", type=str, default=DATA_DIRECTORY_T,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir-pretrain", type=str, default=SNAPSHOT_DIR_PRETRAIN,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    parser.add_argument("--tensorboard", action='store_true', help="choose whether to use tensorboard.")
    parser.add_argument("--init_path", type=str, default=INIT_PATH,
                        help="Path to the directory of log.")
    parser.add_argument("--shot", type=int, default=5,
                        help="Number of shots")
    parser.add_argument("--way", type=int, default=6,
                        help="Number of ways")
    parser.add_argument("--rn", type=int, default=300,
                        help="")
    parser.add_argument("--k", type=int, default=3,
                        help="")
    parser.add_argument("--feature_dim", type=int, default=32,
                        help="")
    parser.add_argument('--alpha', type=float, default=0.99, metavar='ALPHA',
                        help="Initial alpha in label propagation")
    return parser.parse_args()

def compute_miou(query_pred, query_labels, nclass=5):

    pre_label_query = query_pred.permute(0, 3, 1, 2).max(dim=1)[1].data.cpu().numpy()
    pre_label = [i for i in pre_label_query]
    true_label = query_labels.data.cpu().numpy()
    true_label = [i for i in true_label]

    eval_matrics = eval_semantic_segmentation_enhance(pre_label, true_label, ignore_label=255, nclass=nclass)
    eval_acc = eval_matrics['mean_class_accuracy']
    eval_imou = eval_matrics['miou']
    eval_iou = eval_matrics['iou']
    acc_class = eval_matrics['class_accuracy']

    return eval_acc, eval_imou, eval_iou, acc_class


def compute_miou_3(confusion_matrix):

    eval_matrics = eval_semantic_segmentation_enhance_3(confusion_matrix, ignore_label=255)
    eval_acc = eval_matrics['mean_class_accuracy']
    eval_imou = eval_matrics['miou']
    eval_iou = eval_matrics['iou']
    acc_class = eval_matrics['class_accuracy']
    F1_score = eval_matrics['F1_score']
    m_F1_score = eval_matrics['m_F1_score']
    OA = eval_matrics['OA']
    confusion_matrix = eval_matrics['confusion_matrix']

    return eval_acc, eval_imou, eval_iou, acc_class, F1_score, m_F1_score, OA, confusion_matrix

def compute_confusion(pre_label_query, query_labels, nclass):

    pre_label = [i for i in pre_label_query]
    true_label = query_labels.data.cpu().numpy()
    true_label = [i for i in true_label]

    eval_matrics = calc_semantic_segmentation_confusion_enhance(pre_label, true_label, ignore_label=255, n_class=nclass)
    return eval_matrics
def main():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    args = get_arguments()
    device = torch.device("cuda" if not args.cpu else "cpu")
    set_seed(3)
    cudnn.enabled = True
    cudnn.benchmark = True
    print('SLIC_100_TPN')

    if not os.path.exists(args.snapshot_dir_pretrain):
        print('save path does not exist')

    print('###### Load data ######')
    #
    trainset = DataSet_WHDLD_tr(root=args.data_dir_s, list_path=os.path.join(args.data_dir_s, 'image_list.txt'))
    train_loader = DataLoader(
        trainset,
        batch_size=15,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )

    for shot_num in range(0,SHOT_NUMBER):

        print('#-----------------------shot number:',shot_num, '-----------------#')

        model_extractor = Feature_extractor(pretrained_path=args.init_path, cfg=args.model, args=args).cuda()
        model_disentangle = Disentangle_new1().cuda()
        model_classifier = MGCN_TPN_Classifier_union_gai(cfg=args.model, args=args).cuda()

        model_extractor.train()
        model_disentangle.train()
        model_classifier.train()

        optimizer_extractor = torch.optim.SGD(model_extractor.parameters(), lr=1e-4, momentum=0.9, weight_decay=0.0005)
        #scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma=0.1)
        scheduler_extractor = MultiStepLR(optimizer_extractor, milestones=[10000, 20000, 30000], gamma=0.1)
        optimizer_disentangle = torch.optim.SGD(model_disentangle.parameters(), lr=1e-4, momentum=0.9, weight_decay=0.0005)
        scheduler_disentangle = MultiStepLR(optimizer_disentangle, milestones=[10000, 20000, 30000], gamma=0.1)
        optimizer_classifier = torch.optim.SGD(model_classifier.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0005)
        #scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma=0.1)
        scheduler_classifier = MultiStepLR(optimizer_classifier, milestones=[10000, 20000, 30000], gamma=0.1)
        criterion = nn.CrossEntropyLoss(ignore_index=255)

        dice_loss = DiceLoss_gai(ignore_index=255)

        print('###### Training ######')
        miou_best = 0
        for epoch in range(args.num_steps):

            query_loss_ave = Averager()
            align_loss_ave = Averager()
            for i_iter, batch in enumerate(train_loader):
                # Prepare input
                # if i_iter > 2:
                #     break
                images, labels, _, _, = batch
                images = images.to(device)
                labels = labels.long().to(device)
                _, slic_npy_refine = images_seg(images,labels,seg_number=60)
                # slic_npy = slic_npy.to(device)
                shot_images, query_images = images[:args.shot], images[args.shot:]
                shot_labels, query_labels = labels[:args.shot], labels[args.shot:]
                # shot_label_mask, query_label_mask = label_mask[:args.shot], label_mask[args.shot:]
                # shot_slic, query_slic = slic_npy[:args.shot], slic_npy[args.shot:]

                temp_unique = torch.unique(shot_labels)

                if len(torch.unique(shot_labels)) < 6:
                    continue

                optimizer_extractor.zero_grad()
                optimizer_disentangle.zero_grad()
                optimizer_classifier.zero_grad()
                superpixel_fts, label_superpixel_shot, label_superpixel_query, fts_map  = model_extractor(shot_images, shot_labels, query_images, query_labels, slic_npy_refine)
                superpixel_fts_irrevelant, superpixel_fts_specific = model_disentangle(superpixel_fts)
                query_pred, superpixel_pred = model_classifier(superpixel_fts_irrevelant, label_superpixel_shot, label_superpixel_query, slic_npy_refine, class_num=6)

                query_superpixel_loss = criterion(superpixel_pred, label_superpixel_query)
                query_loss = criterion(query_pred.permute(0, 3, 1, 2), query_labels)
                # output = torch.argmax(query_pred.permute(0, 3, 1, 2), axis=1)
                query_dice_loss = dice_loss(query_pred.permute(0, 3, 1, 2), query_labels)
                #
                loss = query_loss + query_dice_loss

                loss.backward()
                optimizer_extractor.step()
                scheduler_extractor.step()
                optimizer_disentangle.step()
                scheduler_disentangle.step()
                optimizer_classifier.step()
                scheduler_classifier.step()

                # Log loss
                query_loss = query_loss.detach().data.cpu().numpy()
                # align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0

                query_loss_ave.add(query_loss)
                # align_loss_ave.add(align_loss)
                if i_iter % 100 == 0:
                    print('epoch:', epoch)
                    print('query_loss', query_loss_ave.item())
                    # print('align_loss', align_loss_ave.item())

            # test===========================================================
            with torch.no_grad():
                model_extractor.eval()
                model_disentangle.eval()
                model_classifier.eval()

                list_shot_path = os.path.join(args.data_dir_t,'shot_GID5', '5_shot' + str(shot_num) + '.csv')
                shot_images, shot_labels, _, shot_name_all = get_shot_data(root=args.data_dir_t,
                                                                           shot_list_path=list_shot_path)
                shot_images_t = shot_images.to(device)
                shot_labels_t = shot_labels.long().to(device)
                _, shot_slic_npy_tgt = images_seg(shot_images, shot_labels, seg_number=60)
                shot_slic_npy_tgt = torch.from_numpy(shot_slic_npy_tgt)

                print('---------------------val----------------------')
                confusion_matrix_all_tgt = np.zeros((5, 5))
                list_query_path = os.path.join(args.data_dir_t,'query_GID5', '5_query' + str(shot_num) + '.csv')
                target_set = DataSet_GID5_te(root=args.data_dir_t, list_path=list_query_path)
                target_loader = DataLoader(
                    target_set,
                    batch_size=10,
                    shuffle=True,
                    num_workers=8,
                    pin_memory=True,
                    drop_last=True
                )

                for j_iter, (images_tgt, labels_tgt, _, _) in enumerate(target_loader):
                    # Prepare input
                    if j_iter > 50:
                        break
                    images_tgt = images_tgt.to(device)
                    labels_tgt = labels_tgt.long().to(device)
                    slic_npy_tgt, _ = images_seg(images_tgt, labels_tgt, seg_number=60)
                    slic_npy_tgt = torch.from_numpy(slic_npy_tgt)
                    # slic_npy = slic_npy.to(device)

                    shot_images_tgt, query_images_tgt = shot_images_t, images_tgt
                    shot_labels_tgt, query_labels_tgt = shot_labels_t, labels_tgt
                    if len(torch.unique(shot_labels_tgt)) < 5:
                        continue

                    slic_npy_tgt_union = torch.cat([shot_slic_npy_tgt, slic_npy_tgt], dim=0)


                    # pred source and target data

                    # slic_npy_tgt_union = torch.cat([shot_slic_npy, slic_npy_tgt], dim=0)
                    superpixel_fts_tgt, superpixel_label_shot_tgt, superpixel_label_query_tgt, fts_map_tgt = model_extractor(
                        shot_images_tgt, shot_labels_tgt, query_images_tgt, query_labels_tgt, slic_npy_tgt_union)
                    superpixel_fts_tgt_irrevelant, superpixel_fts_tgt_specific = model_disentangle(superpixel_fts_tgt)
                    query_pred_tgt, superpixel_pred_tgt = model_classifier(superpixel_fts_tgt_irrevelant,
                                                                           superpixel_label_shot_tgt,
                                                                           superpixel_label_query_tgt, slic_npy_tgt_union, class_num=5)



                    pre_label_query = query_pred_tgt.permute(0, 3, 1, 2).max(dim=1)[1].data.cpu().numpy()
                    confusion_matrix_ = compute_confusion(pre_label_query, query_labels_tgt, nclass = 5)
                    confusion_matrix_all_tgt = confusion_matrix_all_tgt + confusion_matrix_

                acc_ave_t, miou_t, iou_t, acc_t, F1_score_t, m_F_score_t, OA_t, confusion_matrix_t = compute_miou_3(confusion_matrix_all_tgt)

                print('target: ========================================================================')
                print('iou_t:', iou_t)
                print('miou_ave_t:', miou_t)
                print('F1_t:', F1_score_t)
                print('mF1_ave_t:', m_F_score_t)
                print('OA_t:', OA_t)
                print('epoch:', epoch)
                print('target: ========================================================================')

            if miou_best < miou_t:
                print('-------------save model (shot_num)',shot_num)
                miou_best = miou_t
                iou_everyclass = iou_t
                print('best miou:', miou_best)
                print('iou:',iou_everyclass)
                torch.save(model_extractor.state_dict(), os.path.join(args.snapshot_dir_pretrain,'TAFD_extractor_best_' + str(shot_num) +'.pth'))
                torch.save(model_disentangle.state_dict(), os.path.join(args.snapshot_dir_pretrain, 'TAFD_disentangle_best_'+ str(shot_num) +'.pth'))
                torch.save(model_classifier.state_dict(), os.path.join(args.snapshot_dir_pretrain, 'TAFD_classifier_best_' + str(shot_num) +'.pth'))

            if epoch % args.save_pred_every == 0 and epoch != 0:
                print('best miou:', miou_best)
                print('iou:', iou_t)
                print('taking snapshot ...')
                torch.save(model_extractor.state_dict(),os.path.join(args.snapshot_dir_pretrain, 'TAFD_extractor_' + str(epoch) + '_' + str(shot_num) +'.pth'))
                torch.save(model_disentangle.state_dict(),os.path.join(args.snapshot_dir_pretrain, 'TAFD_disentangle_' + str(epoch) + '_' +  str(shot_num) +'.pth'))
                torch.save(model_classifier.state_dict(),os.path.join(args.snapshot_dir_pretrain, 'TAFD_classifier_' + str(epoch) +'_' +  str(shot_num) +'.pth'))

if __name__=='__main__':
    main()
