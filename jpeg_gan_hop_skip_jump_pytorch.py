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
import math
import pandas as pd

from art.utils import to_categorical
from art.estimators.classification import BlackBoxClassifier
from art.attacks.evasion import HopSkipJump

from modelCompose import ComposedModel


def black_box_decision_attack(data):
    output = model(torch.from_numpy(data).cuda())
    prediction = output.argmax(dim=1, keepdim=True).detach().cpu().numpy()
    return to_categorical(prediction, nb_classes=1000)


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

    parser = argparse.ArgumentParser(description="Run square attack.", usage='Use -h for more information.')
    parser.add_argument("--start", type=int, help="Image to start with")
    args = parser.parse_args()
    start = args.start

    print(f" GPU: {torch.cuda.current_device()}")
    print(' Model - {}\n Defence - {}\n Num images - {}\n QF - {}\n Delta QF - {}\n Model iterations {}\n'
          .format(BASE_MODEL, DEFENCE, K, QF, DELTA_QF, MODEL_ITERATIONS))

    # Instantiate composed model
    model = ComposedModel(arch=BASE_MODEL, qf=QF, defence=DEFENCE, jpeg_pass=JPEG_PASS,
                          delta_qf=DELTA_QF, model_iterations=MODEL_ITERATIONS, multi_gan=MULTI_GAN)

    dataset = datasets.ImageFolder(IMAGES_PATH, transforms.Compose([transforms.Resize(256),
                                                                    transforms.CenterCrop(224),
                                                                    transforms.ToTensor()]))
    imagenet_res = np.load('imagenet_preds_' + BASE_MODEL + '.npz')
    sorted_idx = np.argsort(imagenet_res['all_probs'])[::-1]  # [-K:]
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

    results = []

    classifier = BlackBoxClassifier(black_box_decision_attack, dataset[0][0].shape, nb_classes=1000)
    attack = HopSkipJump(classifier=classifier, norm=NORM, targeted=False, max_iter=150, max_eval=10000, init_eval=100, max_queries=5000, budget=0.01, verbose=False)

    for count, (data, target) in enumerate(dataloader):
        if count < start:
            continue

        start_time = time.time()

        if count >= K + start:
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

        # Attack
        adv_img = attack.generate(data_cpu, x_adv_init=None)
        pred_adv = attack.last_adv_pred


        if NORM == 2:
            perturbation = attack.best_loss
            print("L2 error: ", perturbation)
            if math.isnan(perturbation):
                print("L2 is nan")
            perturbation_metric = "L2 metric"


        # plt.imshow(np.transpose(data_cpu.squeeze(), (1, 2, 0)))
        # plt.title(f"Original {count}")
        # plt.show(block=False)
        # plt.imshow(np.transpose(adv_img.squeeze(), (1, 2, 0)))
        # plt.title(f"Adversarial image {perturbation_metric}: {perturbation}")
        # plt.show(block=False)


        num_queries = attack.num_queries

        column_names = ["Correct label", "Predicted label", "Adversarial label", perturbation_metric, "Num queries", "Time (s)"]
        for t, po, pa in zip(target, pred_orig, pred_adv):
            print(f'=== Batch {count} - target {t.item()} - pred orig {po.item()} - pred adv {pa.item()} ===')
        elapsed_time = time.time() - start_time
        print(f'Took {elapsed_time}s')


        result = [[target.detach().cpu().numpy().squeeze(), pred_orig.squeeze(), pred_adv.squeeze(), perturbation, num_queries, elapsed_time]]
        df = pd.DataFrame(result)

        if not os.path.isfile(f"results_HSJ_{DEFENCE}_defence_start_{start}_K_{K}.csv"):
            df.columns = column_names
            df.to_csv(f"results_HSJ_{DEFENCE}_defence_start_{start}_K_{K}.csv", index=False)
        else:
            df.to_csv(f"results_HSJ_{DEFENCE}_defence_start_{start}_K_{K}.csv", header=False, mode="a", index=False)
