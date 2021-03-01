import time

import numpy as np
import os
import sys
import argparse
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torch

from custom_modelCompose import ComposedModel
import scipy.io as sio

from art.utils import to_categorical
from art.estimators.classification import BlackBoxClassifier, BlackBoxClassifierNeuralNetwork
from art.attacks.evasion import HopSkipJump, SquareAttack, BoundaryAttack


print(f" GPU: {torch.cuda.current_device()}")

model = "resnet50"
imagesPath = "/thecube/students/lagnolucci/val_by_class/"
K = 50
qf = 40
defence = "gan"
jpeg_pass = 1
delta_qf = 20
model_iterations = 3
multi_gan = True
batch_size = 1


def imshow(img, title):
    npimg = img.numpy()
    fig = plt.figure(figsize=(5, 15))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()


def imsave(img, title):
    npimg = img.numpy()
    fig = plt.figure(figsize=(5, 15))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(title)


class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def get_logits(self, data):
        return self.model(data)

    def cuda(self):
        self.model = self.model.cuda()
        return


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


count_queries = 0
def black_box_decision_attack(data):
    global count_queries
    count_queries += 1
    output = modelwrap(torch.from_numpy(data).cuda())
    prediction = output.argmax(dim=1, keepdim=True).detach().cpu().numpy()
    return to_categorical(prediction, nb_classes=1000)


def black_box_score_attack(data):
    output = modelwrap(torch.from_numpy(data).cuda()).detach().cpu().numpy()
    return output


print(' Model - {}\n Defence - {}\n Num images - {}\n QF - {}\n Delta QF - {}\n Model iterations {}\n'.format(model, defence, K, qf, delta_qf, model_iterations))

# Instantiate composed model
modelwrap = ComposedModel(arch=model, qf=qf, defence=defence, jpeg_pass=jpeg_pass,
                          delta_qf=delta_qf, model_iterations=model_iterations, multi_gan=multi_gan)

dataset = datasets.ImageFolder(imagesPath, transforms.Compose([transforms.Resize(256),
                                                              transforms.CenterCrop(224),
                                                              transforms.ToTensor()]))
imagenet_res = np.load('imagenet_preds_' + model +'.npz')
sorted_idx = np.argsort(imagenet_res['all_probs'])[::-1]#[-K:]
all_paths = [f'{imagesPath}{x}' for x in imagenet_res['all_filenames'][sorted_idx]]
sorted_scores = imagenet_res['all_probs'][sorted_idx]

# Take the best image for each class
all_classes = [x.split('/')[-2] for x in all_paths]
cls_ids = np.unique(all_classes)
top_k_paths = []
top_k_scores = []
for c in cls_ids:
    cur_cls_idx = np.where(np.array(all_classes) == c)[0]
    top_k_paths.append(all_paths[cur_cls_idx[0]])
    top_k_scores.append(sorted_scores[cur_cls_idx[0]])

dataset.samples = list(filter(lambda x: x[0] in top_k_paths, dataset.samples))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

modelwrap.eval()
modelwrap.cuda()

start_time = time.time()

results = []

gan_pass = 1
jpg_pass = jpeg_pass
if defence == 'jpeg':
    gan_pass = 0
elif defence is None:
    gan_pass = 0
    jpg_pass = 0


# classifier = BlackBoxClassifier(black_box_decision_attack, dataset[0][0].shape, nb_classes=1000)
# attack = HopSkipJump(classifier=classifier, norm=2, targeted=False, max_iter=0, max_eval=10000, init_eval=100)
estimator = BlackBoxClassifierNeuralNetwork(black_box_score_attack, dataset[0][0].shape, nb_classes=1000, clip_values=(0, 1))
attack = SquareAttack(estimator=estimator, norm=2, eps=0.01, max_iter=10, nb_restarts=1, p_init=0.1, batch_size=batch_size) # p_init: 0.1 for L2 and 0.05 for L_inf


for count, (data, target) in enumerate(dataloader):
    # if count < 6:
    #     continue
    print(f'{count} results_qf{qf}_jpg{jpg_pass}_gan{gan_pass}_K{K}_deltaQF{delta_qf}_modelIterations{model_iterations}')
    if count >= K:
        # Exit after K batches
        break

    data_cuda = data.cuda()
    data_cpu = data.detach().cpu().clone().numpy()
    target = target.cuda()

    # Get original classification
    output_orig = modelwrap(data_cuda)
    loss_orig = F.nll_loss(output_orig, target, reduction='sum').item()
    pred_orig = output_orig.argmax(dim=1, keepdim=True).detach().cpu().numpy()
    correct_orig = (pred_orig.T == target.cpu().numpy()).sum()

    # Attack
    # adv_img, iter_l2, iter_cls = BPDA_attack_cuda(data, args.target_label, modelwrap, step_size=0.001, iterations=200)
    # output_adv = modelwrap(adv_img)
    # loss_adv = F.nll_loss(output_adv, target, reduction='sum').item()
    # pred_adv = output_adv.argmax(dim=1, keepdim=True).detach().cpu().numpy()
    # correct_adv = (pred_adv.T == target.cpu().numpy()).sum()

    '''
    # Hop skip jump
    # plt.imshow(np.transpose(data_cpu.squeeze(), (1, 2, 0)))
    # plt.title(f"Original {count}")
    # plt.show(block=False)
    
    iter_step = 5
    adv_img = None
    for i in range(10):
        adv_img = attack.generate(data_cpu, x_adv_init=adv_img, resume=True)
        l2 = np.linalg.norm(np.reshape(adv_img * 255 - data_cpu * 255, -1) / np.linalg.norm(data_cpu * 255))
        print("L2 error: ", l2)
        output_adv = modelwrap(torch.from_numpy(adv_img).cuda())
        pred_adv = output_adv.argmax(dim=1, keepdim=True).detach().cpu().numpy()
        print(f"Total queries: {count_queries}")
        print(f"Predicted label: {pred_adv.squeeze()}\n")
        attack.max_iter = iter_step
        # plt.imshow(np.transpose(adv_img.squeeze(), (1, 2, 0)))
        # plt.title(f"Adversarial image {i} iteration, l2: {l2}")
        # plt.show(block=False)
    '''
    
    # Square attack
    adv_img = attack.generate(data_cpu)
    l2 = np.linalg.norm(np.reshape(data_cpu - adv_img, -1)) / np.linalg.norm(np.reshape(data_cpu, -1))
    print("L2 error: ", l2)
    a = np.sqrt(np.sum(np.square(data_cpu - adv_img))) / np.sqrt(np.sum(np.square(data_cpu)))
    print("a: ", a)
    l_inf = np.max(np.abs(data_cpu - adv_img)) / np.max(np.abs(data_cpu))
    print("L_inf error: ", l_inf)
    plt.imshow(np.transpose(data_cpu.squeeze(), (1, 2, 0)))
    plt.title(f"Original {count}")
    plt.show(block=False)
    plt.imshow(np.transpose(adv_img.squeeze(), (1, 2, 0)))
    plt.title(f"Adversarial image l2: {l2}")
    plt.show(block=False)


    output_adv = modelwrap(torch.from_numpy(adv_img).cuda())
    loss_adv = F.nll_loss(output_adv, target, reduction='sum').item()
    pred_adv = output_adv.argmax(dim=1, keepdim=True).detach().cpu().numpy()
    correct_adv = (pred_adv.T == target.cpu().numpy()).sum()

    # plt.imshow(np.reshape(adv_img[0].astype(np.float32), (224, 224, 3)))
    # plt.show(block=False)


    for t, po, pa in zip(target, pred_orig, pred_adv):
        print(f'=== Batch {count} - target {t.item()} - pred orig {po.item()} - pred adv {pa.item()} ===')
    print(f"Correct original: {correct_orig}")

    # TODO: Understand how to constraint a maximum l2 distance
    # TODO: Try to increase batch size
    # TODO: Create Github repo
    # TODO: Choose the correct parameters and count the queries

    print(f'Took {time.time() - start_time}s')



'''
    assert args.batch_size == 1
    for bi, _ in enumerate(range(args.batch_size)):
        result_dict = {'idx': count * args.batch_size + bi,
                       'original_pred': pred_orig[bi],
                       'img_cls': target[bi].cpu().numpy(),
                       'final_pred': pred_adv[bi],
                       'iter_l2': iter_l2,  # Da fixare per batch size > 1
                       'iter_cls': iter_cls,  # Da fixare per batch size > 1
                       'attack_target': args.target_label}
        results.append(result_dict)


mat_name = f'{args.model}_results_BPDA_EOT_qf{args.qf}_jpg{jpg_pass}_gan{gan_pass}_K{args.K}_deltaQF{args.delta_qf}_modelIterations{args.model_iterations}_multiGAN{str(args.multi_gan)}.mat'
print(mat_name)
sio.savemat(mat_name, {'results': results})
'''