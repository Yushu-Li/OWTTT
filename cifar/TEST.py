import argparse
import torch
import torch.optim as optim
import torch.utils.data as data

from utils.misc import *
from utils.test_helpers import *
from utils.prepare_dataset import *

# ----------------------------------
import copy
import random
import numpy as np
from utils.contrastive import *
from utils.offline import *
from torch import nn
import torch.nn.functional as F
# ----------------------------------


def compute_os_variance(os, th):
    """
    Calculate the area of a rectangle.

    Parameters:
        os : OOD score queue.
        th : Given threshold to separate weak and strong OOD samples.

    Returns:
        float: Weighted variance at the given threshold th.
    """
    
    thresholded_os = np.zeros(os.shape)
    thresholded_os[os >= th] = 1

    # compute weights
    nb_pixels = os.size
    nb_pixels1 = np.count_nonzero(thresholded_os)
    weight1 = nb_pixels1 / nb_pixels
    weight0 = 1 - weight1

    # if one the classes is empty, eg all pixels are below or above the threshold, that threshold will not be considered
    # in the search for the best threshold
    if weight1 == 0 or weight0 == 0:
        return np.inf

    # find all pixels belonging to each class
    val_pixels1 = os[thresholded_os == 1]
    val_pixels0 = os[thresholded_os == 0]

    # compute variance of these classes
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0

    return weight0 * var0 + weight1 * var1


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10OOD')
parser.add_argument('--strong_OOD', default='noise')
parser.add_argument('--strong_ratio', default=1, type=float)
parser.add_argument('--dataroot', default="./data", help='path to dataset')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--outf', help='folder to output log')
parser.add_argument('--level', default=5, type=int)
parser.add_argument('--N_m', default=512, type=int, help='queue length')
parser.add_argument('--corruption', default='snow')
parser.add_argument('--resume', default='/cluster/personal/code/TTT/TTAC-master/cifar/results/cifar10_joint_resnet50', help='directory of pretrained model')
parser.add_argument('--model', default='resnet50', help='resnet50')
parser.add_argument('--seed', default=0, type=int)


# ----------- Args and Dataloader ------------
args = parser.parse_args()

print(args)
print('\n')




class_num = 10 if args.dataset == 'cifar10OOD' else 100

net, ext, head, ssh, classifier = build_resnet50(args)

teset, _ = prepare_test_data(args)
teloader = data.DataLoader(teset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, worker_init_fn=seed_worker, pin_memory=True, drop_last=False)

# -------------------------------
print('Resuming from %s...' %(args.resume))

load_resnet50(net, head, ssh, classifier, args)

# ----------- Offline Feature Summarization ------------
args_align = copy.deepcopy(args)

_, offlineloader = prepare_train_data(args_align)
ext_src_mu, ext_src_cov, ssh_src_mu, ssh_src_cov, mu_src_ext, cov_src_ext, mu_src_ssh, cov_src_ssh = offline(args,offlineloader, ext, classifier, head, class_num)

ext_src_mu = torch.stack(ext_src_mu)
weak_prototype = F.normalize(ext_src_mu.clone()).cuda()

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# ----------- Open-World Test-time Training ------------

correct = []
unseen_correct= []
all_correct=[]
cumulative_error = []
num_open = 0
predicted_list=[]
label_list=[]

os_inference_queue = []
queue_length = args.N_m

ema_total_n = 0.

print('\n-----Test-Time Training with TEST-----')
for te_idx, (te_inputs, te_labels) in enumerate(teloader):    

        
    ####-------------------------- Test ----------------------------####

    with torch.no_grad():
        if isinstance(te_inputs,list):
            inputs = te_inputs[0].cuda()
        else:
            inputs = te_inputs.cuda()
        net.eval()
        feat_ext = ext(inputs) #b,2048
        logit = torch.mm(F.normalize(feat_ext), weak_prototype.t())
        update = 1
        softmax_logit = logit.softmax(dim=-1)
        pro, predicted = softmax_logit.max(dim=-1)
        
        ood_score, max_index = logit.max(1)
        ood_score = 1-ood_score
        os_inference_queue.extend(ood_score.detach().cpu().tolist())
        os_inference_queue = os_inference_queue[-queue_length:]

        threshold_range = np.arange(0,1,0.01)
        criterias = [compute_os_variance(np.array(os_inference_queue), th) for th in threshold_range]
        best_threshold = threshold_range[np.argmin(criterias)]
        unseen_mask = (ood_score > best_threshold)
        args.ts = best_threshold
        predicted[unseen_mask] = class_num

        one = torch.ones_like(te_labels)*class_num
        false = torch.ones_like(te_labels)*-1
        predicted = torch.where(predicted>class_num-1, one.cuda(), predicted)
        all_labels = torch.where(te_labels>class_num-1, one, te_labels)
        seen_labels = torch.where(te_labels>class_num-1, false, te_labels)
        unseen_labels = torch.where(te_labels>class_num-1, one, false)
        correct.append(predicted.cpu().eq(seen_labels))
        unseen_correct.append(predicted.cpu().eq(unseen_labels))
        all_correct.append(predicted.cpu().eq(all_labels))
        num_open += torch.gt(te_labels, 99).sum()

        predicted_list.append(predicted.long().cpu())
        label_list.append(all_labels.long().cpu())


    seen_acc = round(torch.cat(correct).numpy().sum() / (len(torch.cat(correct).numpy())-num_open.numpy()),4)
    unseen_acc = round(torch.cat(unseen_correct).numpy().sum() / num_open.numpy(),4)
    h_score = round((2*seen_acc*unseen_acc) /  (seen_acc + unseen_acc),4)
    print('Batch:(', te_idx,'/',len(teloader),\
        '\t Cumulative Results: ACC_S:', seen_acc,\
        '\tACC_N:', unseen_acc,\
        '\tACC_H:',h_score\
        )


print('\nTest time training result:',' ACC_S:', seen_acc,\
        '\tACC_N:', unseen_acc,\
        '\tACC_H:',h_score,'\n\n\n\n'\
        )


if args.outf != None:
    my_makedir(args.outf)
    with open (args.outf+'/results.txt','a') as f:
        f.write(str(args)+'\n')
        f.write(
        'ACC_S:'+ str(seen_acc)+\
            '\tACC_N:'+ str(unseen_acc)+\
            '\tACC_H:'+str(h_score)+'\n\n\n\n'\
        )