import os
import statistics
import sys
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
import argparse

from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader
from models.encoder_and_projection import Encoder_and_projection
from models.classifier import MLPClassifier
from AutomaticWeightedLoss import AutomaticWeightedLoss
from get_dataset import finetune_dataset_add_feature
from sklearn.model_selection import train_test_split
import torch.fft
from models.encoder_and_projection import AttentionFusionModule


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # All GPU
    os.environ["PYTHONHASHSEED"] = str(seed)  # 禁止hash随机化
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn.functional as F
import random


def augment_signal(x, sr=5000000, shift_range=0.05, mask_range=0.05, p_shift_freq=0.4, p_mask=0.4, p_noise=0.4,
                   p_crop=0.4, p_shift_time=0.4, noise_level=0.02, crop_ratio=0.1, time_shift_ratio=0.2):
    """
    对信号进行频率域和时域数据增强。
    """
    batch_size, signal_length = x.shape[0], x.shape[-1]
    x = x.to(x.device)

    # 1. 时域增强 (高斯噪声)
    if random.random() < p_noise:
        noise = torch.randn_like(x) * noise_level
        x = x + noise

    # 2. 时域增强 (信号裁剪)
    if random.random() < p_crop:
        crop_len = int(signal_length * crop_ratio)
        if crop_len > 0:
            start_idx = random.randint(0, crop_len)
            end_idx = signal_length - random.randint(0, crop_len)
            x_cropped = x[..., start_idx:end_idx]
            x = F.pad(x_cropped, (0, signal_length - x_cropped.shape[-1]), 'constant', 0)

    # 3. 时域增强 (循环移位)
    if random.random() < p_shift_time:
        max_shift = int(signal_length * time_shift_ratio)
        if max_shift > 0:
            shift_amount = random.randint(-max_shift, max_shift)
            x = torch.roll(x, shifts=shift_amount, dims=-1)

    # 4. 频率域增强
    if not torch.is_complex(x):
        x_complex = torch.view_as_complex(torch.stack([x.float(), torch.zeros_like(x).float()], dim=-1))
    else:
        x_complex = x.float()

    spectrum = torch.fft.fft(x_complex)

    # 5. 频率平移 (Frequency Shift)
    if random.random() < p_shift_freq:
        max_shift_bins = int(signal_length * shift_range)
        if max_shift_bins > 0:
            shift_bins = random.randint(-max_shift_bins, max_shift_bins)
        else:
            shift_bins = 0
        if shift_bins != 0:
            shifted_spectrum = torch.roll(spectrum, shifts=shift_bins, dims=-1)
            spectrum = shifted_spectrum

    # 6. 频率掩蔽 (Frequency Masking)
    if random.random() < p_mask:
        mask_width_bins = int(signal_length * mask_range)
        if mask_width_bins > 0 and signal_length - mask_width_bins > 0:
            start_bin = random.randint(0, signal_length - mask_width_bins)
            end_bin = start_bin + mask_width_bins
            mask = torch.ones_like(spectrum, dtype=torch.bool)
            mask[..., start_bin:end_bin] = False
            spectrum = spectrum * mask.to(spectrum.dtype)

    # 7. IFFT 逆变换回时域
    augmented_x_complex = torch.fft.ifft(spectrum)
    augmented_x = augmented_x_complex.real

    return augmented_x


# --- train 函数 ---
def train(
        online_network,
        classifier,
        classifier_fused,
        classifier2,
        criterion,  # <--- 修改：从 loss_nll 改为 criterion
        train_dataloader,
        optimizer,
        scheduler,
        epoch,
        device,
        writer,
        awl1,
):
    online_network.train()
    classifier.train()
    classifier_fused.train()
    classifier2.train()
    correct = 0
    nll_loss_sum = 0
    nll_loss_feature_sum = 0
    nll_loss_fused_sum = 0
    for data, target in train_dataloader:
        target = target.long()
        if torch.cuda.is_available():
            data = data.to(device)
            target = target.to(device)

        data = augment_signal(data)

        optimizer.zero_grad()

        fused_embedding, _, embedding, embedding_new = online_network(data)

        # === 核心修改：移除 F.log_softmax，直接输出 logits ===
        output = classifier(embedding)
        nll_loss_batch = criterion(output, target)

        outpute_feature = classifier2(embedding_new)
        nll_loss_batch_feature = criterion(outpute_feature, target)

        output_fused = classifier_fused(fused_embedding)
        nll_loss_batch_fused = criterion(output_fused, target)
        # ====================================================

        result_loss_batch, weight_pa = awl1(
            nll_loss_batch,
            nll_loss_batch_feature,
        )

        total_loss = result_loss_batch + nll_loss_batch_fused

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(online_network.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(classifier_fused.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(classifier2.parameters(), 1.0)

        optimizer.step()

        nll_loss_sum += nll_loss_batch.item()
        nll_loss_feature_sum += nll_loss_batch_feature.item()
        nll_loss_fused_sum += nll_loss_batch_fused.item()

        # 在计算准确率时，需要对logits应用softmax
        pred = F.softmax(output_fused, dim=1).argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    scheduler.step()

    nll_loss_avg = nll_loss_sum / len(train_dataloader)
    nll_loss_feature_avg = nll_loss_feature_sum / len(train_dataloader)
    nll_loss_fused_avg = nll_loss_fused_sum / len(train_dataloader)

    # 精简输出
    print(
        "Train Epoch: {} \tClass_Loss: {:.6f}, feature_Loss: {:.6f}, Fused_Loss: {:.6f}, Accuracy: {}/{} ({:0f}%)".format(
            epoch,
            nll_loss_avg,
            nll_loss_feature_avg,
            nll_loss_fused_avg,
            correct,
            len(train_dataloader.dataset),
            100.0 * correct / len(train_dataloader.dataset),
        )
    )
    return nll_loss_avg, nll_loss_feature_avg, nll_loss_fused_avg


def evaluate(
        online_network,
        classifier,
        classifier_fused,
        classifier2,
        criterion,  # <--- 修改：从 loss_nll 改为 criterion
        val_dataloader,
        epoch,
        device,
        writer,
        awl1,
):
    online_network.eval()
    classifier.eval()
    classifier_fused.eval()
    classifier2.eval()
    awl1.eval()
    test_loss = 0
    feature_loss = 0
    fused_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)

            fused_embedding, _, embedding, embedding_new = online_network(data)

            # === 核心修改：移除 F.log_softmax，直接输出 logits ===
            output = classifier(embedding)
            outpute_feature = classifier2(embedding_new)
            output_fused = classifier_fused(fused_embedding)

            test_loss += criterion(output, target).item()
            feature_loss += criterion(outpute_feature, target).item()
            fused_loss += criterion(output_fused, target).item()
            # ====================================================

            pred = F.softmax(output_fused, dim=1).argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_dataloader)
    feature_loss /= len(val_dataloader)
    fused_loss /= len(val_dataloader)
    fmt = "\nValidation set: IQ_loss: {:.4f}, AF_loss: {:.4f}, Fused_loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n"
    print(
        fmt.format(
            test_loss,
            feature_loss,
            fused_loss,
            correct,
            len(val_dataloader.dataset),
            100.0 * correct / len(val_dataloader.dataset),
        )
    )
    return test_loss, feature_loss, fused_loss, 100.0 * correct / len(val_dataloader.dataset)


def test(online_network, classifier, classifier_fused, classifier2, test_dataloader, device, awl1):
    online_network.eval()
    classifier.eval()
    classifier_fused.eval()
    classifier2.eval()
    test_loss = 0
    feature_loss = 0
    fused_loss = 0
    correct = 0
    # === 核心修改：使用带标签平滑的损失函数 ===
    loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    # =========================================
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
                loss = loss.to(device)

            fused_embedding, _, embedding, embedding_new = online_network(data)

            # === 核心修改：移除 F.log_softmax，直接输出 logits ===
            output = classifier(embedding)
            outpute_feature = classifier2(embedding_new)
            output_fused = classifier_fused(fused_embedding)

            test_loss += loss(output, target).item()
            feature_loss += loss(outpute_feature, target).item()
            fused_loss += loss(output_fused, target).item()
            # ====================================================

            pred = F.softmax(output_fused, dim=1).argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader)
    feature_loss /= len(test_dataloader)
    fused_loss /= len(test_dataloader)
    fmt = "\nTest set: IQ_loss: {:.4f}, AF_loss: {:.4f}, Fused_loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n"
    print(
        fmt.format(
            test_loss,
            feature_loss,
            fused_loss,
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )
    return 100.0 * correct / len(test_dataloader.dataset), test_loss, feature_loss, fused_loss


def train_and_test(
        online_network,
        classifier,
        classifier_fused,
        classifier2,
        criterion,  # <--- 修改：从 loss_nll 改为 criterion
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
        epochs,
        save_path_online_network,
        save_path_classifier,
        save_path_classifier_fused,
        save_path_classifier2,
        device,
        writer,
        awl1,
):
    best_val_accuracy = 0.0
    patience_counter = 0
    patience = 70

    for epoch in range(1, epochs + 1):
        train_loss, train_feature_loss, train_fused_loss = train(
            online_network,
            classifier,
            classifier_fused,
            classifier2,
            criterion,  # <--- 修改
            train_dataloader,
            optimizer,
            scheduler,
            epoch,
            device,
            writer,
            awl1=awl1,
        )

        test_loss, feature_loss, fused_loss, current_val_accuracy = evaluate(
            online_network,
            classifier,
            classifier_fused,
            classifier2,
            criterion,  # <--- 修改
            val_dataloader,
            epoch,
            device,
            writer,
            awl1=awl1,
        )

        if current_val_accuracy > best_val_accuracy:
            print(
                "Validation accuracy improved from {:.4f}% to {:.4f}%, saving new model.".format(
                    best_val_accuracy, current_val_accuracy
                )
            )
            best_val_accuracy = current_val_accuracy
            patience_counter = 0

            torch.save(online_network, save_path_online_network)
            torch.save(classifier, save_path_classifier)
            torch.save(classifier_fused, save_path_classifier_fused)
            torch.save(classifier2, save_path_classifier2)
        else:
            print("Validation accuracy not improved. Current best: {:.4f}%".format(best_val_accuracy))
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

        print("------------------------------------------------")


def run(
        checkpoints_folder,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        epochs,
        save_path_online_network,
        save_path_classifier,
        save_path_classifier_fused,
        save_path_classifier2,
        device,
        writer,
        config,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training with: {device}")

    online_network = torch.load(os.path.join(checkpoints_folder, "model_val_best.pth"))

    online_network.fusion_module = AttentionFusionModule(
        feature_dim_iq=1024, feature_dim_af=20
    ).to(device)

    online_network.train()

    lr = config["finetune"]["lr"]
    dropout_rate = config["finetune"]["dropout_rate"]
    mid_dim_new_hidden = config["network"]["projection_head"]["mlp_hidden_size"]

    hidden_dim = 512
    num_classes = 13

    classifier = MLPClassifier(
        input_dim=1024,
        hidden_dim=hidden_dim,
        output_dim=num_classes,
        dropout_rate=dropout_rate,
    ).to(device)

    classifier_fused = MLPClassifier(
        input_dim=1024 + 20,
        hidden_dim=hidden_dim,
        output_dim=num_classes,
        dropout_rate=dropout_rate,
    ).to(device)

    classifier2 = MLPClassifier(
        input_dim=20,
        hidden_dim=256,
        output_dim=num_classes,
        dropout_rate=dropout_rate,
    ).to(device)

    awl1 = AutomaticWeightedLoss(2)

    # === 核心修改：定义带标签平滑的损失函数 ===
    # smoothing_value 可以作为超参数进行调整，0.1 是一个常用的默认值
    smoothing_value = 0.1
    criterion = nn.CrossEntropyLoss(label_smoothing=smoothing_value)
    # =========================================

    if torch.cuda.is_available():
        online_network = online_network.to(device)
        classifier = classifier.to(device)
        classifier_fused = classifier_fused.to(device)
        criterion = criterion.to(device)  # <--- 修改
        classifier2 = classifier2.to(device)
        awl1 = awl1.to(device)

    trainable_params = list(online_network.parameters()) + \
                       list(awl1.parameters()) + \
                       list(classifier2.parameters()) + \
                       list(classifier.parameters()) + \
                       list(classifier_fused.parameters()) + \
                       list(online_network.fusion_module.parameters())

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config["finetune"]["lr"],
        # momentum=0.9,
        weight_decay=0.01
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    train_and_test(
        online_network,
        classifier,
        classifier_fused,
        classifier2,
        criterion,  # <--- 修改
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        save_path_online_network=save_path_online_network,
        save_path_classifier=save_path_classifier,
        save_path_classifier_fused=save_path_classifier_fused,
        save_path_classifier2=save_path_classifier2,
        device=device,
        writer=writer,
        awl1=awl1,
    )

    print("Test_result:")
    online_network = torch.load(save_path_online_network)
    classifier = torch.load(save_path_classifier)
    classifier_fused = torch.load(save_path_classifier_fused)
    classifier2 = torch.load(save_path_classifier2)

    test_acc, test_loss, feature_loss, fused_loss = test(
        online_network, classifier, classifier_fused, classifier2, test_dataloader, device, awl1=awl1
    )

    return test_acc, test_loss, feature_loss, fused_loss


def main():
    config = yaml.load(open("config/config.yaml", "r", encoding='utf-8'), Loader=yaml.FullLoader)
    config_ft = config["finetune"]

    device = torch.device("cuda:0")
    test_acc_all = []
    test_loss_all = []
    feature_loss_all = []
    fused_loss_all = []
    start = time.time()

    for i in range(config["iteration"]):
        print(f"iteration: {i}--------------------------------------------------------")
        set_seed(i)
        writer = SummaryWriter(
            f"./log_finetune/PT_{config['trainer']['class_start']}-{config['trainer']['class_end']}_FT_{config_ft['class_start']}-{config_ft['class_end']}_{config_ft['k_shot']}shot"
        )
        save_path_classifier = f"./model_weight/classifier_PT_{config['trainer']['class_start']}-{config['trainer']['class_end']}_FT_{config_ft['class_start']}-{config_ft['class_end']}_{config_ft['k_shot']}shot_{i}.pth"
        save_path_classifier2 = f"./model_weight/classifier2_PT_{config['trainer']['class_start']}-{config['trainer']['class_end']}_FT_{config_ft['class_start']}-{config_ft['class_end']}_{config_ft['k_shot']}shot_{i}.pth"
        save_path_classifier_fused = f"./model_weight/classifier_fused_PT_{config['trainer']['class_start']}-{config['trainer']['class_end']}_FT_{config_ft['class_start']}-{config_ft['class_end']}_{config_ft['k_shot']}shot_{i}.pth"
        save_path_online_network = f"./model_weight/online_network_PT_{config['trainer']['class_start']}-{config['trainer']['class_end']}_FT_{config_ft['class_start']}-{config_ft['class_end']}_{config_ft['k_shot']}shot_{i}.pth"

        X_train, X_test, X_val, Y_train, Y_test, Y_val = finetune_dataset_add_feature()

        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
        train_dataloader = DataLoader(
            train_dataset, batch_size=config_ft["batch_size"], shuffle=True
        )

        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val))
        val_dataloader = DataLoader(
            val_dataset, batch_size=config_ft["batch_size"], shuffle=True
        )

        test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
        test_dataloader = DataLoader(
            test_dataset, batch_size=config_ft["test_batch_size"], shuffle=True
        )

        checkpoints_folder = os.path.join(
            "runs",
            f"byol_s{config['trainer']['class_start']}-e{config['trainer']['class_end']}",
            "checkpoints",
        )

        test_acc, test_loss, feature_loss, fused_loss = run(
            checkpoints_folder,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            epochs=config_ft["epochs"],
            save_path_online_network=save_path_online_network,
            save_path_classifier=save_path_classifier,
            save_path_classifier_fused=save_path_classifier_fused,
            save_path_classifier2=save_path_classifier2,
            device=device,
            writer=writer,
            config=config,
        )
        test_acc_all.append(test_acc)
        test_loss_all.append(test_loss)
        feature_loss_all.append(feature_loss)
        fused_loss_all.append(fused_loss)
        writer.close()

    end = time.time()
    average_test_acc = statistics.mean(test_acc_all)
    average_test_loss = statistics.mean(test_loss_all)
    average_feature_loss = statistics.mean(feature_loss_all)
    average_fused_loss = statistics.mean(fused_loss_all)

    print("average grade:", average_test_acc)
    print("average test loss:", average_test_loss)
    print("average feature loss:", average_feature_loss)
    print("average fused loss:", average_fused_loss)
    print("eppch grade:", test_acc_all)
    print("eppch test loss:", test_loss_all)
    print("eppch feature loss:", feature_loss_all)
    print("eppch fused loss:", fused_loss_all)
    print("all time:", end - start)

    df = pd.DataFrame(test_acc_all)
    df.to_excel(
        f"test_result/PT_{config['trainer']['class_start']}-{config['trainer']['class_end']}_FT_{config_ft['class_start']}-{config_ft['class_end']}_{config_ft['k_shot']}shot.xlsx"
    )


if __name__ == "__main__":
    main()
