import torch
import statistics
import os

def covariance(features):
    assert len(features.size()) == 2, "TODO: multi-dimensional feature map covariance"
    n = features.shape[0]
    tmp = torch.ones((1, n), device=features.device) @ features
    cov = (features.t() @ features - (tmp.t() @ tmp) / n) / n
    return cov

def coral(cs, ct):
    d = cs.shape[0]
    loss = (cs - ct).pow(2).sum() / (4. * d ** 2)
    return loss


def linear_mmd(ms, mt):
    loss = (ms - mt).pow(2).mean()
    return loss

def offline(args,trloader, ext, classifier, head, class_num=10):
    if class_num == 10:
        if os.path.exists(args.resume+'/offline_cifar10.pth'):
            data = torch.load(args.resume+'/offline_cifar10.pth')
            return data
    elif class_num == 100:
        if os.path.exists(args.resume+'/offline_cifar100.pth'):
            data = torch.load(args.resume+'/offline_cifar100.pth')
            return data
    else:
        raise Exception("This function only handles CIFAR10 and CIFAR100 datasets.")
    ext.eval()
    
    feat_stack = [[] for i in range(class_num)]
    ssh_feat_stack = [[] for i in range(class_num)]

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(trloader):

            feat = ext(inputs.cuda())
            predict_logit = classifier(feat)
            ssh_feat = predict_logit
            
            pseudo_label = predict_logit.max(dim=1)[1]

            for label in pseudo_label.unique():
                label_mask = pseudo_label == label
                feat_stack[label].extend(feat[label_mask, :])
                ssh_feat_stack[label].extend(ssh_feat[label_mask, :])
    ext_mu = []
    ext_cov = []
    ext_all = []

    ssh_mu = []
    ssh_cov = []
    ssh_all = []
    for feat in feat_stack:
        ext_mu.append(torch.stack(feat).mean(dim=0))
        ext_cov.append(covariance(torch.stack(feat)))
        ext_all.extend(feat)
    
    for feat in ssh_feat_stack:
        ssh_mu.append(torch.stack(feat).mean(dim=0))
        ssh_cov.append(covariance(torch.stack(feat)))
        ssh_all.extend(feat)
    
    ext_all = torch.stack(ext_all)
    ext_all_mu = ext_all.mean(dim=0)
    ext_all_cov = covariance(ext_all)

    ssh_all = torch.stack(ssh_all)
    ssh_all_mu = ssh_all.mean(dim=0)
    ssh_all_cov = covariance(ssh_all)
    if class_num == 10:
        torch.save((ext_mu, ext_cov, ssh_mu, ssh_cov, ext_all_mu, ext_all_cov, ssh_all_mu, ssh_all_cov), args.resume+'/offline_cifar10.pth')
    if class_num == 100:
        torch.save((ext_mu, ext_cov, ssh_mu, ssh_cov, ext_all_mu, ext_all_cov, ssh_all_mu, ssh_all_cov), args.resume+'/offline_cifar100.pth')
    return ext_mu, ext_cov, ssh_mu, ssh_cov, ext_all_mu, ext_all_cov, ssh_all_mu, ssh_all_cov


