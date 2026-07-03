import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.manifold import TSNE
from torch.utils.data import TensorDataset, DataLoader

# =========================================================
# 导入模型
# =========================================================

sys.path.insert(0, 'models')

from models.encoder_and_projection import Encoder_and_projection

# =========================================================
# 路径配置
# =========================================================

CONFIG_PATH = "config/config.yaml"

DATA_DIR = r"F:\KDM_ADS\ADSB"

# =========================================================
# 加载配置文件
# =========================================================

def load_config(config_path):

    with open(config_path, "r", encoding="utf-8") as f:

        config = yaml.load(
            f,
            Loader=yaml.FullLoader
        )

    return config

# =========================================================
# 加载测试集
# =========================================================

def get_test_loader(config):

    X_path = os.path.join(
        DATA_DIR,
        "X_test_10Class.npy"
    )

    Y_path = os.path.join(
        DATA_DIR,
        "Y_test_10Class.npy"
    )

    X_test = np.load(X_path)
    Y_test = np.load(Y_path)

    # =====================================================
    # 数据归一化
    # =====================================================

    processed = []

    for i in range(len(X_test)):

        x = X_test[i, :4800]

        x_max = x.max()
        x_min = x.min()

        if (x_max - x_min) == 0:

            x = x - x_min

        else:

            x = (x - x_min) / (x_max - x_min)

        processed.append(x)

    X_test = np.array(processed)

    X_test = X_test.transpose(0, 2, 1)

    dataset = TensorDataset(

        torch.Tensor(X_test),
        torch.Tensor(Y_test)

    )

    loader = DataLoader(

        dataset,

        batch_size=config["finetune"]["test_batch_size"],

        shuffle=False

    )

    return loader

# =========================================================
# 加载模型
# =========================================================

def load_model(config):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Encoder_and_projection(
        **config["network"]
    ).to(device)

    # =====================================================
    # 权重路径
    # =====================================================

    config_ft = config["finetune"]

    epoch_id = 23

    weights_path = (
        f"./model_weight/"
        f"online_network_PT_"
        f"{config['trainer']['class_start']}-"
        f"{config['trainer']['class_end']}_"
        f"FT_"
        f"{config_ft['class_start']}-"
        f"{config_ft['class_end']}_"
        f"{config_ft['k_shot']}shot_"
        f"{epoch_id}.pth"
    )

    print("\nLoading weight:")
    print(weights_path)

    model = torch.load(
        weights_path,
        map_location=device
    )

    model.eval()

    return model, device

# =========================================================
# 提取特征
# =========================================================

def extract_features(model, loader, device):

    all_features = []
    all_labels = []

    with torch.no_grad():

        for data, labels in tqdm(loader):

            data = data.to(device)

            _, _, features, _ = model(data)

            all_features.append(
                features.cpu().numpy()
            )

            all_labels.append(
                labels.cpu().numpy()
            )

    features = np.concatenate(
        all_features,
        axis=0
    )

    labels = np.concatenate(
        all_labels,
        axis=0
    )

    return features, labels

# =========================================================
# SCI级 t-SNE 绘图
# =========================================================

def draw_tsne(features,
              labels,
              shot):

    print("\nRunning t-SNE ...")

    tsne = TSNE(

        n_components=2,

        perplexity=30,

        n_iter=1000,

        random_state=42

    )

    tsne_result = tsne.fit_transform(
        features
    )

    print("t-SNE finished.")

    # =====================================================
    # SCI风格
    # =====================================================

    plt.rcParams['font.family'] = 'Times New Roman'

    plt.rcParams['font.size'] = 14

    fig, ax = plt.subplots(
        figsize=(6.5, 5.5)
    )

    # =====================================================
    # 类别颜色
    # =====================================================

    num_classes = len(
        np.unique(labels)
    )

    cmap = plt.cm.get_cmap(
        'tab10',
        num_classes
    )

    # =====================================================
    # 绘制散点
    # =====================================================

    for c in range(num_classes):

        idx = labels == c

        ax.scatter(

            tsne_result[idx, 0],
            tsne_result[idx, 1],

            s=10,

            alpha=0.85,

            color=cmap(c),

            label=str(c)

        )

    # =====================================================
    # 标题
    # =====================================================

    ax.set_title(

        f'{shot}-shot t-SNE Visualization',

        fontsize=16

    )

    # =====================================================
    # 去坐标轴
    # =====================================================

    ax.set_xticks([])
    ax.set_yticks([])

    # =====================================================
    # 边框
    # =====================================================

    for spine in ax.spines.values():

        spine.set_linewidth(1.0)

    # =====================================================
    # 图例
    # =====================================================

    ax.legend(

        fontsize=9,

        frameon=False,

        loc='best',

        ncol=2

    )

    plt.tight_layout()

    # =====================================================
    # 保存高分辨率
    # =====================================================

    save_name = f"TSNE_{shot}shot"

    plt.savefig(

        save_name + ".pdf",

        dpi=600,

        bbox_inches='tight'

    )

    plt.savefig(

        save_name + ".png",

        dpi=600,

        bbox_inches='tight'

    )

    plt.savefig(

        save_name + ".tiff",

        dpi=600,

        bbox_inches='tight'

    )

    print("\nSaved:")
    print(save_name)

    plt.show()

# =========================================================
# 主函数
# =========================================================

if __name__ == "__main__":

    # =====================================================
    # 加载yaml
    # =====================================================

    config = load_config(CONFIG_PATH)

    current_shot = config["finetune"]["k_shot"]

    print(f"\nCurrent shot: {current_shot}")

    # =====================================================
    # 数据
    # =====================================================

    loader = get_test_loader(config)

    # =====================================================
    # 模型
    # =====================================================

    model, device = load_model(config)

    # =====================================================
    # 特征提取
    # =====================================================

    features, labels = extract_features(

        model,
        loader,
        device

    )

    # =====================================================
    # 绘制 t-SNE
    # =====================================================

    draw_tsne(

        features,
        labels,
        current_shot

    )