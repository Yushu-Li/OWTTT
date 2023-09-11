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




class Prototype_Pool(nn.Module):
    
    """
    Prototype pool containing strong OOD prototypes.

    Methods:
        __init__: Constructor method to initialize the prototype pool, storing the values of delta, the number of weak OOD categories, and the maximum count of strong OOD prototypes.
        forward: Method to farward pass, return the cosine similarity with strong OOD prototypes.
        update_pool: Method to append and delete strong OOD prototypes.
    """
    
    
    def __init__(self, delta=0.1, class_num=10, max=100):
        super(Prototype_Pool, self).__init__()
        
        self.class_num=class_num
        self.max_length = max
        self.flag = 0
        self.delta = delta


    def forward(self, x, all=False):
        
        # if the flag is 0, the prototype pool is empty, return None.
        if not self.flag:
            return None
        
        # compute the cosine similarity between the features and the strong OOD prototypes.
        out = torch.mm(x, self.memory.t())
        
        if all==True:
            # if all is True, return the cosine similarity with all the strong OOD prototypes.
            return out
        else:
            # if all is False, return the cosine similarity with the nearest strong OOD prototype. 
            return torch.max(out/(self.delta),dim=1)[0].unsqueeze(1)


    def update_pool(self, feature):

        if not self.flag:
            # if the flag is 0, the prototype pool is empty, use the feature to init the prototype pool.
            self.register_buffer('memory', feature.detach())
            self.flag = 1
        else:
            if self.memory.shape[0] < self.max_length:
                # if the number of strong OOD prototypes is less than the maximum count of strong OOD prototypes, append the feature to the prototype pool.
                self.memory = torch.cat([self.memory, feature.detach()],dim=0)
            else:
                # else then delete the earlest appended strong OOD prototype and append the feature to the prototype pool.
                self.memory = torch.cat([self.memory[1:], feature.detach()],dim=0)
        self.memory = F.normalize(self.memory)


def append_prototypes(pool, feat_ext, logit, ts, ts_pro):
    """
    Append strong OOD prototypes to the prototype pool.

    Parameters:
        pool : Prototype pool.
        feat_ext : Normalized features of the input images.
        logit : Cosine similarity between the features and the weak OOD prototypes.
        ts : Threshold to separate weak and strong OOD samples.
        ts_pro : Threshold to append strong OOD prototypes.

    """
    added_list=[]
    update = 1

    while update:
        feat_mat = pool(F.normalize(feat_ext),all=True)
        if  not feat_mat==None:
            new_logit = torch.cat([logit, feat_mat], 1)
        else:
            new_logit = logit

        r_i_pro, _ = new_logit.max(dim=-1)

        r_i, _ = logit.max(dim=-1)

        if added_list!=[]:
            for add in added_list:
                # if added_list is not empty, set the cosine similarity between the added features and the strong OOD prototypes to 1, to avoid the added features to be appended to the prototype pool again.
                r_i[add]=1
        min_logit , min_index = r_i.min(dim=0)


        if (1-min_logit) > ts :
            # if the cosine similarity between the feature and the weak OOD prototypes is less than the threshold ts, the feature is a strong OOD sample.
            added_list.append(min_index)
            if (1-r_i_pro[min_index]) > ts_pro:
                # if this strong OOD sample is far away from all the strong OOD prototypes, append it to the prototype pool.
                pool.update_pool(F.normalize(feat_ext[min_index].unsqueeze(0)))
        else:
            # all the features are weak OOD samples, stop the loop.
            update=0


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10OOD')
parser.add_argument('--strong_OOD', default='noise')
parser.add_argument('--strong_ratio', default=1, type=float)
parser.add_argument('--dataroot', default="./data", help='path to dataset')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--delta', default=0.1, type=float)
parser.add_argument('--ce_scale', default=0, type=float, help='cross entropy loss scale')
parser.add_argument('--outf', help='folder to output log')
parser.add_argument('--level', default=5, type=int)
parser.add_argument('--N_m', default=512, type=int, help='queue length')
parser.add_argument('--corruption', default='snow')
parser.add_argument('--resume', default='/cluster/personal/code/TTT/TTAC-master/cifar/results/cifar10_joint_resnet50', help='directory of pretrained model')
parser.add_argument('--da_scale', default=1, type=float, help='distribution alignment loss scale')
parser.add_argument('--model', default='resnet50', help='resnet50')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--max_prototypes', default=100, type=int)
parser.add_argument('--save', action='store_true', default=False, help='save the model final checkpoint')


# ----------- Args and Dataloader ------------
args = parser.parse_args()

print(args)
print('\n')




class_num = 10 if args.dataset == 'cifar10OOD' else 100

net, ext, head, ssh, classifier = build_resnet50(args)

teset, _ = prepare_test_data(args)
teloader = data.DataLoader(teset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, worker_init_fn=seed_worker, pin_memory=True, drop_last=False)

pool = Prototype_Pool(args.delta,class_num=class_num,max = args.max_prototypes).cuda()

# -------------------------------
print('Resuming from %s...' %(args.resume))

load_resnet50(net, head, ssh, classifier, args)

optimizer = optim.SGD(ext.parameters(), lr=args.lr, momentum=0.9)

# ----------- Offline Feature Summarization ------------
args_align = copy.deepcopy(args)

_, offlineloader = prepare_train_data(args_align)
ext_src_mu, ext_src_cov, ssh_src_mu, ssh_src_cov, mu_src_ext, cov_src_ext, mu_src_ssh, cov_src_ssh = offline(args,offlineloader, ext, classifier, head, class_num)

ext_src_mu = torch.stack(ext_src_mu)
ext_src_cov = torch.stack(ext_src_cov)

ema_ext_mu = ext_src_mu.clone()
ema_ext_cov = ext_src_cov.clone()
ema_ext_total_mu = torch.zeros(2048).float()
ema_ext_total_cov = torch.zeros(2048, 2048).float()

if class_num == 10:
    loss_scale = 0.05
    ema_length = 128
else:
    loss_scale = 0.05
    ema_length = 64

ema_n = torch.zeros(class_num).cuda()
ema_total_n = 0.
weak_prototype = F.normalize(ext_src_mu.clone()).cuda()
args.ts_pro = 0.0
bias = cov_src_ext.max().item() / 30.
template_ext_cov = torch.eye(2048).cuda() * bias

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

os_training_queue = []
os_inference_queue = []
queue_length = args.N_m
ce_scale = args.ce_scale

ema_total_n = 0.

print('\n-----Test-Time Training with OURS-----')
for te_idx, (te_inputs, te_labels) in enumerate(teloader):
    classifier.eval()
    ext.eval()
    
    optimizer.zero_grad()
    loss = torch.tensor(0.).cuda()
    
    if isinstance(te_inputs,list):
        inputs = te_inputs[0].cuda()
    else:
        inputs = te_inputs.cuda()

    # features extracted by backbone
    feat_ext = ext(inputs)
    
    # logits of the input images, used to compute the cosine similarity between the features and the weak OOD prototypes.
    logit = torch.mm(F.normalize(feat_ext), weak_prototype.t()) / args.delta

    
    # compute the cosine similarity between the features and the strong OOD prototypes.
    feat_mat = pool(F.normalize(feat_ext))
    if  not feat_mat==None:
        new_logit = torch.cat([logit, feat_mat], 1)
    else:
        new_logit = logit

    pro, predicted = new_logit[:,:class_num].max(dim=-1)

    # compute the ood score of the input images.
    ood_score = 1-pro*args.delta
    os_training_queue.extend(ood_score.detach().cpu().tolist())
    os_training_queue = os_training_queue[-queue_length:]


    threshold_range = np.arange(0,1,0.01)
    criterias = [compute_os_variance(np.array(os_training_queue), th) for th in threshold_range]

    # best threshold is the one minimizing the variance of the two classes
    best_threshold = threshold_range[np.argmin(criterias)]
    args.ts = best_threshold
    seen_mask = (ood_score < args.ts)
    unseen_mask = (ood_score >= args.ts)
    r_i, pseudo_labels = new_logit.max(dim=-1)

    if unseen_mask.sum().item()!=0:
        #compute ts_pro to append new strong OOD prototypes to the prototype pool.

        min_logit , min_index = r_i.min(dim=0)
        
        in_score = 1-r_i*args.delta
        threshold_range = np.arange(0,1,0.01)
        criterias = [compute_os_variance(in_score[unseen_mask].detach().cpu().numpy(), th) for th in threshold_range]

        best_threshold = threshold_range[np.argmin(criterias)]
        args.ts_pro = best_threshold
        
        # append new strong OOD prototypes to the prototype pool.
        append_prototypes(pool, feat_ext, logit.detach()*args.delta, args.ts, args.ts_pro)

    len_memory = len(new_logit[0])

    
    if len_memory!=class_num:
        
        if seen_mask.sum().item()!=0:
            pseudo_labels[seen_mask] = new_logit[seen_mask,:class_num].softmax(dim=-1).max(dim=-1)[1]
        if unseen_mask.sum().item()!=0:
            pseudo_labels[unseen_mask] = class_num
    else:
        pseudo_labels = new_logit[seen_mask,:class_num].softmax(dim=-1).max(dim=-1)[1]


    # ------distribuution alignment------
    if seen_mask.sum().item()!=0:
        ext.train()
        feat_global = ext(inputs[seen_mask])
        # Global Gaussian
        b = feat_global.shape[0]
        ema_total_n += b
        alpha = 1. / 1280 if ema_total_n > 1280 else 1. / ema_total_n
        delta_pre = (feat_global - ema_ext_total_mu.cuda())        
        delta = alpha * delta_pre.sum(dim=0)                    
        tmp_mu = ema_ext_total_mu.cuda() + delta                
        tmp_cov = ema_ext_total_cov.cuda() + alpha * (delta_pre.t() @ delta_pre - b * ema_ext_total_cov.cuda()) - delta[:, None] @ delta[None, :]
        with torch.no_grad():
            ema_ext_total_mu = tmp_mu.detach().cpu()
            ema_ext_total_cov = tmp_cov.detach().cpu()

        source_domain = torch.distributions.MultivariateNormal(mu_src_ext, cov_src_ext + template_ext_cov)
        target_domain = torch.distributions.MultivariateNormal(tmp_mu, tmp_cov + template_ext_cov)
        loss += args.da_scale*(torch.distributions.kl_divergence(source_domain, target_domain) + torch.distributions.kl_divergence(target_domain, source_domain)) * loss_scale
    
    
    # we only use 50% of samples with ood score far from τ∗ to perform prototype clustering for each batch
    if len_memory!=class_num and seen_mask.sum().item()!=0 and unseen_mask.sum().item()!=0:
        a, idx1 = torch.sort((ood_score[seen_mask]), descending=True)
        filter_down = a[-int(seen_mask.sum().item()*(1/2))]
        a, idx1 = torch.sort((ood_score[unseen_mask]), descending=True)
        filter_up= a[int(unseen_mask.sum().item()*(1/2))]
        for j in range(len(pseudo_labels)):
            
            if ood_score[j] >=filter_down and seen_mask[j]:
                seen_mask[j]=False
            if ood_score[j] <=filter_up and unseen_mask[j]:
                unseen_mask[j]=False


    if len_memory!=class_num:
        entropy_seen = nn.CrossEntropyLoss()(new_logit[seen_mask,:class_num],pseudo_labels[seen_mask])
        entropy_unseen= nn.CrossEntropyLoss()(new_logit[unseen_mask],pseudo_labels[unseen_mask])
        loss += ce_scale*(entropy_seen+ entropy_unseen)/2

    try:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    except:
        print('can not backward')
    torch.cuda.empty_cache()

    

    ####-------------------------- Test ----------------------------####
    
    with torch.no_grad():

        net.eval()
        feat_ext = ext(inputs) #b,2048
        logit = torch.mm(F.normalize(feat_ext), weak_prototype.t())/args.delta
        update = 1


        softmax_logit = logit.softmax(dim=-1)
        # _, recall_predicted = softmax_logit.max(1)
        pro, predicted = softmax_logit.max(dim=-1)
        
        ood_score, max_index = logit.max(1)
        ood_score = 1-ood_score*args.delta
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
    print('Batch:(', te_idx,'/',len(teloader), ')\tloss:',"%.2f" % loss.item(),\
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
    if args.save:
        torch.save(net.state_dict(), os.path.join(args.outf, 'final.pth'))