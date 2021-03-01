import time
import os
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torch

from custom_modelCompose import ComposedModel

from art.estimators.classification import BlackBoxClassifierNeuralNetwork
from art.attacks.evasion import SquareAttack
import math
import pandas as pd


def black_box_score_attack(data):
    output = model(torch.from_numpy(data).cuda()).detach().cpu().numpy()
    return output


if __name__ == '__main__':
    BASE_MODEL = "resnet50"
    IMAGES_PATH = "/thecube/students/lagnolucci/val_by_class/"
    K = 100
    QF = 40
    DEFENCE = "gan"
    JPEG_PASS = 1
    DELTA_QF = 20
    MODEL_ITERATIONS = 3
    MULTI_GAN = True
    BATCH_SIZE = 1
    NORM = 2

    print(f" GPU: {torch.cuda.current_device()}")
    print(' Model - {}\n Defence - {}\n Num images - {}\n QF - {}\n Delta QF - {}\n Model iterations {}\n'.format(BASE_MODEL, DEFENCE, K, QF, DELTA_QF, MODEL_ITERATIONS))

    # Instantiate composed model
    model = ComposedModel(arch=BASE_MODEL, qf=QF, defence=DEFENCE, jpeg_pass=JPEG_PASS,
                          delta_qf=DELTA_QF, model_iterations=MODEL_ITERATIONS, multi_gan=MULTI_GAN)

    dataset = datasets.ImageFolder(IMAGES_PATH, transforms.Compose([transforms.Resize(256),
                                                                  transforms.CenterCrop(224),
                                                                  transforms.ToTensor()]))
    imagenet_res = np.load('imagenet_preds_' + BASE_MODEL + '.npz')
    sorted_idx = np.argsort(imagenet_res['all_probs'])[::-1]#[-K:]
    all_paths = [f'{IMAGES_PATH}{x}' for x in imagenet_res['all_filenames'][sorted_idx]]
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
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model.eval()
    model.cuda()

    estimator = BlackBoxClassifierNeuralNetwork(black_box_score_attack, dataset[0][0].shape, nb_classes=1000, clip_values=(0, 1))
    attack = SquareAttack(estimator=estimator, norm=NORM, eps=0.08, max_iter=5000, nb_restarts=1, p_init=0.1, batch_size=BATCH_SIZE, max_queries=5000) # p_init: 0.1 for L2 and 0.05 for L_inf

    for count, (data, target) in enumerate(dataloader):
        start_time = time.time()

        if count >= K:
            # Exit after K batches
            break

        data_cuda = data.cuda()
        data_cpu = data.detach().cpu().clone().numpy()
        target = target.cuda()

        # Get original classification
        output_orig = model(data_cuda)
        loss_orig = F.nll_loss(output_orig, target, reduction='sum').item()
        pred_orig = output_orig.argmax(dim=1, keepdim=True).detach().cpu().numpy()
        correct_orig = (pred_orig.T == target.cpu().numpy()).sum()

        # Square attack
        adv_img = attack.generate(data_cpu)
        output_adv = model(torch.from_numpy(adv_img).cuda())
        loss_adv = F.nll_loss(output_adv, target, reduction='sum').item()
        pred_adv = output_adv.argmax(dim=1, keepdim=True).detach().cpu().numpy()
        correct_adv = (pred_adv.T == target.cpu().numpy()).sum()

        # plt.imshow(np.transpose(data_cpu.squeeze(), (1, 2, 0)))
        # plt.title(f"Original {count}")
        # plt.show(block=False)
        # plt.imshow(np.transpose(adv_img.squeeze(), (1, 2, 0)))
        # plt.title(f"Adversarial image metric L{norm}: {l2}")
        # plt.show(block=False)


        if NORM == 2:
            perturbation = np.linalg.norm(np.reshape(data_cpu - adv_img, -1)) / np.linalg.norm(np.reshape(data_cpu, -1))
            print("L2 error: ", perturbation)
            if math.isnan(perturbation):
                print("L2 is nan")
            perturbation_metric = "L2 metric"

        elif NORM == np.inf:
            perturbation = np.max(np.abs(data_cpu - adv_img)) / np.max(np.abs(data_cpu))
            print("L_inf error: ", perturbation)
            perturbation_metric = "L_inf metric"


        num_queries = attack.num_queries

        column_names = ["Correct label", "Predicted label", "Adversarial label", perturbation_metric, "Num queries", "Time (s)"]
        for t, po, pa in zip(target, pred_orig, pred_adv):
            print(f'=== Batch {count} - target {t.item()} - pred orig {po.item()} - pred adv {pa.item()} ===')
        print(f"Correct original: {correct_orig}")
        elapsed_time = time.time() - start_time
        print(f'Took {elapsed_time}s')


        result = [[target.detach().cpu().numpy().squeeze(), pred_orig.squeeze(), pred_adv.squeeze(), perturbation, num_queries, elapsed_time]]
        df = pd.DataFrame(result)

        if not os.path.isfile(f"results_square_{DEFENCE}_defence.csv"):
            df.columns = column_names
            df.to_csv(f"results_square_{DEFENCE}_defence.csv", index=False)
        else:
            df.to_csv(f"results_square_{DEFENCE}_defence.csv", header=False, mode="a", index=False)

        '''
        assert BATCH_SIZE == 1
        for bi, _ in enumerate(range(BATCH_SIZE)):
            result_dict = {'idx': count * BATCH_SIZE + bi,
                           'original_pred': pred_orig[bi],
                           'img_cls': target[bi].cpu().numpy(),
                           'final_pred': pred_adv[bi],
                           "L2": l2}
            results.append(result_dict)
    
        '''
