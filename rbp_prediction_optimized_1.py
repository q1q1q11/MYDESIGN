# -*- coding: utf-8 -*-
# 本科毕设：基于深度学习的 RBP 结合位点预测
# **完整流程**：数据预处理 → 特征提取 (One-hot) → 多尺度CNN-BiLSTM-Attention 模型训练 → 多指标评估 (AUC, AUPRC, F1) → Attention 可视化 → Motif Logo 提取


# 导入核心库 & 参数全局配置
import pandas as pd
import numpy as np
import random
from pyfaidx import Fasta
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import logomaker
from Bio.Seq import Seq
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# 随机种子 (保证实验可复现)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)          # 如果使用多卡
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 固定 DataLoader 的随机性（需要在创建 DataLoader 时应用）
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)

# 超参数配置
SEQ_LEN = 101           # 序列截取长度 (为奇数是为了保证有中心点)
BATCH_SIZE = 64         # 批次大小 (加快训练)
TRAIN_RATIO = 0.7       # 训练集比例
VAL_RATIO = 0.15        # 验证集比例
TEST_RATIO = 0.15       # 测试集比例
LR = 0.001              # 初始学习率
EPOCHS = 20             # 训练轮数
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用计算设备: {DEVICE}")


# 挂载 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 文件路径
bed_path = "/content/drive/MyDrive/RBP_Project/ENCFF752JNY.bed"
fasta_path = "/content/drive/MyDrive/RBP_Project/hg38.fa"

test1_bed_path = "/content/drive/MyDrive/RBP_Project/ENCFF445ENC.bed"

# 统一读取 BED 文件的函数
def load_bed(file_path):
    df = pd.read_csv(
        file_path,
        sep='\t',
        header=None,
        usecols=[0,1,2,3,4,5],
        names=['chr','start','end','name','score','strand']
    )
    df['chr'] = df['chr'].astype(str)
    df = df[df['chr'].isin([f"chr{i}" for i in range(1,23)] + ["chrX","chrY"])]
    df['center'] = ((df['start'] + df['end']) // 2).astype(int)
    print(f"{file_path} 加载完成，Peak数量: {len(df)}")
    return df

# 读取原始数据集
train_peaks = load_bed(bed_path)

# 读取独立测试集
test1_peaks = load_bed(test1_bed_path)

# 读取基因组
genome = Fasta(fasta_path)

print(f"成功加载基因组，原始数据集有 {len(train_peaks)} 个Peak，测试集有 {len(test1_peaks)} 个Peak")


# 核心函数定义

def get_seq(chrom, center, strand, length=SEQ_LEN):
    """从基因组中截取中心定长的序列，处理边界越界情况"""
    half = length // 2
    start = center - half
    end = center + half + 1
    # 边界保护
    if chrom not in genome or start < 0 or end > len(genome[chrom]):
        return None
    seq = genome[chrom][start:end].seq.upper()
    if len(seq) != length or 'N' in seq:
        return None
    if strand == '-':
        seq = str(Seq(seq).reverse_complement())
    return seq

def generate_negative_samples(num_samples, peaks, genome, length=SEQ_LEN, max_tries=100000):
    """
    从非peak区域随机采样负样本
    """
    neg_seqs = []
    peak_regions = {}

    #构建 peak 区间字典（加速判断）
    for _, row in peaks.iterrows():
        chrom = row['chr']
        start = row['start']
        end = row['end']
        if chrom not in peak_regions:
            peak_regions[chrom] = []
        peak_regions[chrom].append((start, end))

    #随机采样
    chrom_list = list(genome.keys())
    tries = 0

    while len(neg_seqs) < num_samples and tries < max_tries:
        tries += 1

        chrom = random.choice(chrom_list)
        chrom_len = len(genome[chrom])

        # 随机选中心点
        center = random.randint(length//2, chrom_len - length//2 - 1)

        # 检查是否落在 peak 区域（±50bp buffer 更安全）
        overlap = False
        if chrom in peak_regions:
            for start, end in peak_regions[chrom]:
                if (center >= start - 50) and (center <= end + 50):
                    overlap = True
                    break

        if overlap:
            continue

        # 随机链方向
        strand = random.choice(['+', '-'])

        seq = get_seq(chrom, center, strand, length)

        if seq is not None:
            neg_seqs.append(seq)

    print(f"负样本采样完成: {len(neg_seqs)} 条（尝试 {tries} 次）")

    return neg_seqs
# def dinucleotide_shuffle(seq):
#     """二核苷酸打乱，保留二核苷酸频率"""
#     if len(seq) < 2:
#         return seq
#     di = [seq[i:i+2] for i in range(len(seq)-1)]
#     random.shuffle(di)
#     shuffled = di[0][0] + ''.join([d[1] for d in di])
#     return shuffled[:len(seq)]

def one_hot_encode(seq):
    """将 RNA 序列转换为 4xL 的独热编码矩阵"""
    mapping = {'A':0, 'C':1, 'G':2, 'U':3, 'T':3}  # T视作U
    L = len(seq)
    encoding = np.zeros((4, L), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in mapping:
            encoding[mapping[base], i] = 1.0
    return encoding



# 提取 BED 文件对应的序列
def extract_sequences(peaks_df, genome, seq_len=SEQ_LEN):
    seqs_raw = peaks_df.apply(lambda row: get_seq(row['chr'], row['center'], row['strand'], seq_len), axis=1)
    seqs = [seq for seq in seqs_raw if seq is not None]
    return seqs


# 原始数据集构建
print("正在提取原始数据集序列...")
pos_seqs = extract_sequences(train_peaks, genome)
X_pos = np.array([one_hot_encode(seq.replace('T','U')) for seq in pos_seqs], dtype=np.float32)
y_pos = np.ones((len(X_pos),1), dtype=np.float32)

neg_seqs = generate_negative_samples(num_samples=len(pos_seqs), peaks=train_peaks, genome=genome, length=SEQ_LEN)
X_neg = np.array([one_hot_encode(seq.replace('T','U')) for seq in neg_seqs], dtype=np.float32)
y_neg = np.zeros((len(X_neg),1), dtype=np.float32)

X = np.concatenate([X_pos, X_neg], axis=0)
y = np.concatenate([y_pos, y_neg], axis=0)
seqs = np.array(pos_seqs + neg_seqs)

# 内部分割 Train/Val/Test
X_train, X_temp, y_train, y_temp, seq_train, seq_temp = train_test_split(
    X, y, seqs, test_size=0.3, random_state=SEED, shuffle=True
)
X_val, X_test, y_val, y_test, seq_val, seq_test = train_test_split(
    X_temp, y_temp, seq_temp, test_size=0.5, random_state=SEED, shuffle=True
)

# 独立测试集构建（含负样本）
print("正在提取独立测试集序列...")

def build_test_dataset(peaks_df, genome, seq_len=SEQ_LEN):
    # 正样本
    pos_seqs = extract_sequences(peaks_df, genome, seq_len)
    X_pos = np.array([one_hot_encode(seq.replace('T','U')) for seq in pos_seqs], dtype=np.float32)
    y_pos = np.ones((len(X_pos),1), dtype=np.float32)

    # 负样本
    neg_seqs = generate_negative_samples(num_samples=len(pos_seqs), peaks=peaks_df, genome=genome, length=seq_len)
    X_neg = np.array([one_hot_encode(seq.replace('T','U')) for seq in neg_seqs], dtype=np.float32)
    y_neg = np.zeros((len(X_neg),1), dtype=np.float32)

    # 合并
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([y_pos, y_neg], axis=0)
    seqs = np.array(pos_seqs + neg_seqs)

    return X, y, seqs

# 构建独立测试集1
X_test1, y_test1, seq_test1 = build_test_dataset(test1_peaks, genome)


# 构建 PyTorch DataLoader
train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                  torch.tensor(y_train, dtype=torch.float32)),
    batch_size=BATCH_SIZE, shuffle=True,
    worker_init_fn=seed_worker,
    generator=g
)

val_loader = DataLoader(
    TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                  torch.tensor(y_val, dtype=torch.float32)),
    batch_size=BATCH_SIZE, shuffle=False,
    worker_init_fn=seed_worker,
    generator=g
)

test_loader = DataLoader(
    TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                  torch.tensor(y_test, dtype=torch.float32)),
    batch_size=BATCH_SIZE, shuffle=False,
    worker_init_fn=seed_worker,
    generator=g
)

# 独立测试集 DataLoader
test1_loader = DataLoader(
    TensorDataset(torch.tensor(X_test1, dtype=torch.float32),
                  torch.tensor(y_test1, dtype=torch.float32)),
    batch_size=BATCH_SIZE,
    shuffle=False,
    worker_init_fn=seed_worker,
    generator=g
)

print("\n========== 数据统计汇总 ==========")

# ===== 主数据集（整体）=====
print("\n--- 主数据集（整体）---")
print(f"正样本: {len(X_pos)}")
print(f"负样本: {len(X_neg)}")

# ===== 训练集 =====
train_pos = int(np.sum(y_train))
train_neg = len(y_train) - train_pos
print("\n--- 训练集 ---")
print(f"总样本: {len(X_train)}")
print(f"正样本: {train_pos}")
print(f"负样本: {train_neg}")

# ===== 验证集 =====
val_pos = int(np.sum(y_val))
val_neg = len(y_val) - val_pos
print("\n--- 验证集 ---")
print(f"总样本: {len(X_val)}")
print(f"正样本: {val_pos}")
print(f"负样本: {val_neg}")

# ===== 内部测试集 =====
test_pos = int(np.sum(y_test))
test_neg = len(y_test) - test_pos
print("\n--- 内部测试集 ---")
print(f"总样本: {len(X_test)}")
print(f"正样本: {test_pos}")
print(f"负样本: {test_neg}")

# ===== 独立测试集1 =====
test1_pos = int(np.sum(y_test1))
test1_neg = len(y_test1) - test1_pos
print("\n--- 独立测试集1 ---")
print(f"总样本: {len(X_test1)}")
print(f"正样本: {test1_pos}")
print(f"负样本: {test1_neg}")

print("\n===================================")



class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Attention 权重计算层
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        # lstm_output shape: [batch_size, seq_len, hidden_dim]
        attn_weights = self.attn(lstm_output)
        attn_weights = F.softmax(attn_weights, dim=1)
        # 加权求和得到上下文向量
        context = torch.sum(attn_weights * lstm_output, dim=1)
        attn_weights = attn_weights.squeeze(-1)
        return context, attn_weights

class RNARBPModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 多尺度 CNN: padding='same' 确保输出序列长度一致，避免硬截断导致信息丢失
        self.conv7  = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=7, padding='same')
        self.conv11  = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=11, padding='same')
        self.conv13 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=13, padding='same')

        self.bn = nn.BatchNorm1d(192) # 64 * 3 = 192 通道
        self.pool = nn.MaxPool1d(kernel_size=2)

        # BiLSTM
        # self.lstm = nn.LSTM(
        #   input_size=192,
        #   hidden_size=64,
        #   batch_first=True,
        #   bidirectional=True,
        #   dropout=0.2
        # )
        self.lstm = nn.LSTM(
          input_size=192,
          hidden_size=64,
          num_layers=2,
          batch_first=True,
          bidirectional=True,
          dropout=0.2
        )

        # Attention (双向LSTM，维度翻倍为 128)
        self.attention = Attention(128)

        # 全连接分类层
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x, return_attn=False):
        # x shape: [batch, 4, seq_len]
        x7  = F.relu(self.conv7(x))
        x11 = F.relu(self.conv11(x))
        x13 = F.relu(self.conv13(x))

        # 特征拼接 [batch, 192, seq_len]
        x = torch.cat((x7, x11, x13), dim=1)
        x = self.bn(x)
        x = self.pool(x) # shape: [batch, 192, seq_len/2]

        # 维度转换适应 LSTM: [batch, seq_len/2, 192]
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)

        # 提取 Attention
        context, attn_weights = self.attention(lstm_out)
        out = self.fc(context)

        if return_attn:
            return out, attn_weights
        return out
# class RNARBPModel_Ablation_Conv13(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 消融实验：仅保留 kernel_size=13 的单尺度卷积
#         self.conv13 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=13, padding='same')

#         self.bn = nn.BatchNorm1d(64)          # 输入通道改为 64
#         self.pool = nn.MaxPool1d(kernel_size=2)

#         # BiLSTM：input_size 改为 64
#         self.lstm = nn.LSTM(
#             input_size=64,
#             hidden_size=64,
#             num_layers=2,
#             batch_first=True,
#             bidirectional=True,
#             dropout=0.2
#         )

#         # Attention（双向 LSTM，hidden_dim 为 128）
#         self.attention = Attention(128)

#         # 全连接分类层（保持不变）
#         self.fc = nn.Sequential(
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(64, 1)
#         )

#     def forward(self, x, return_attn=False):
#         # 仅使用 conv13 提取特征
#         x = F.relu(self.conv13(x))          # shape: [batch, 64, seq_len]

#         x = self.bn(x)
#         x = self.pool(x)                     # shape: [batch, 64, seq_len/2]

#         # 维度转换适应 LSTM: [batch, seq_len/2, 64]
#         x = x.permute(0, 2, 1)
#         lstm_out, _ = self.lstm(x)

#         # Attention
#         context, attn_weights = self.attention(lstm_out)
#         out = self.fc(context)

#         if return_attn:
#             return out, attn_weights
#         return out
# class RNARBPModel_NoAttention(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 1. 多尺度 CNN（与原模型完全一致）
#         self.conv7  = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=7, padding='same')
#         self.conv11 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=11, padding='same')
#         self.conv13 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=13, padding='same')

#         self.bn = nn.BatchNorm1d(192)   # 64*3 = 192
#         self.pool = nn.MaxPool1d(kernel_size=2)

#         # 2. BiLSTM（与原模型完全一致）
#         self.lstm = nn.LSTM(
#             input_size=192,
#             hidden_size=64,
#             num_layers=2,
#             batch_first=True,
#             bidirectional=True,
#             dropout=0.2
#         )

#         # 3. 移除 Attention，改用平均池化
#         # 因此全连接层输入维度仍为 128（双向 LSTM 输出维度）

#         # 4. 全连接分类层（与原模型完全一致）
#         self.fc = nn.Sequential(
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(64, 1)
#         )

#     def forward(self, x, return_attn=False):
#         # CNN 特征提取
#         x7  = F.relu(self.conv7(x))
#         x11 = F.relu(self.conv11(x))
#         x13 = F.relu(self.conv13(x))
#         x = torch.cat((x7, x11, x13), dim=1)   # [batch, 192, seq_len]
#         x = self.bn(x)
#         x = self.pool(x)                        # [batch, 192, seq_len/2]

#         # LSTM
#         x = x.permute(0, 2, 1)                  # [batch, seq_len/2, 192]
#         lstm_out, _ = self.lstm(x)              # [batch, seq_len/2, 128]

#         # 平均池化代替 Attention
#         pooled = torch.mean(lstm_out, dim=1)    # [batch, 128]

#         # 全连接分类
#         out = self.fc(pooled)

#         # 保持接口兼容：如果外部调用时要求返回 attention 权重，则返回 None
#         if return_attn:
#             return out, None
#         return out


import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# 训练准备 

model = RNARBPModel().to(DEVICE)
#model = RNARBPModel_Ablation_Conv13().to(DEVICE)
#model = RNARBPModel_NoAttention().to(DEVICE)
optimizer = optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=1e-5
)

# 使用更稳定的 BCEWithLogitsLoss
criterion = nn.BCEWithLogitsLoss()

# 学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=3
)

# Early stopping 参数
best_auc = 0
patience = 10
counter = 0

# 用于记录训练曲线
train_losses = []
val_aucs = []

print("开始模型训练...")

for epoch in range(EPOCHS):

    # 训练阶段
    model.train()
    total_loss = 0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)

    avg_train_loss = total_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    # 验证阶段
    model.eval()
    val_preds, val_labels = [], []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(DEVICE)
            logits = model(batch_x)
            preds = torch.sigmoid(logits).cpu().numpy()
            val_preds.extend(preds.flatten())
            val_labels.extend(batch_y.numpy().flatten())

    val_auc = roc_auc_score(val_labels, val_preds)
    val_aucs.append(val_auc)

    # 学习率调度
    scheduler.step(val_auc)
    current_lr = optimizer.param_groups[0]['lr']

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"| LR: {current_lr:.6f} "
        f"| Train Loss: {avg_train_loss:.4f} "
        f"| Val AUC: {val_auc:.4f}"
    )

    # Early Stopping
    if val_auc > best_auc:
        best_auc = val_auc
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")
        print("Best model saved")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered")
            break

# 绘制训练曲线
plt.figure(figsize=(10,4))

# 绘制训练 loss 曲线
plt.figure(figsize=(6,4))
plt.plot(range(1,len(train_losses)+1), train_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('Train Loss Curve')
plt.grid(True)

plt.savefig("TrainingLossConvergenceCurve.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()


# 绘制验证 AUC 曲线
plt.figure(figsize=(6,4))
plt.plot(range(1,len(val_aucs)+1), val_aucs, marker='o', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Validation AUC')
plt.title('Validation AUC Curve')
plt.grid(True)

plt.savefig("ValidationAUCCurve.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()


import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, data_loader, name="test"):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(DEVICE)
            batch_preds = torch.sigmoid(model(batch_x)).cpu().numpy()
            preds.extend(batch_preds.flatten())
            labels.extend(batch_y.numpy().flatten())

    preds = np.array(preds)
    labels = np.array(labels)
    preds_binary = (preds > 0.5).astype(int)

    auc   = roc_auc_score(labels, preds)
    auprc = average_precision_score(labels, preds)
    acc   = accuracy_score(labels, preds_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds_binary, average='binary'
    )

    cm = confusion_matrix(labels, preds_binary)

    print(f"\n========== {name} 评估结果 ==========")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"AUC-ROC   : {auc:.4f}")
    print(f"AUPRC     : {auprc:.4f}")
    print("Confusion Matrix:\n", cm)
    print("====================================\n")

    # 画混淆矩阵热力图
    plt.figure(figsize=(5,4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"]
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    filename = f"confusion_matrix_{name}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return preds, preds_binary, labels, cm

test_preds, test_preds_binary, y_test_arr, cm_test = evaluate_model(
    model, test_loader, name="internal_test"
)

test1_preds, test1_preds_binary, y_test1_arr, cm_test1 = evaluate_model(
    model, test1_loader, name="external_test1"
)


pos_correct = np.where((y_test_arr.flatten() == 1) & (test_preds_binary == 1))[0]

if len(pos_correct) == 0:
    print("没有找到预测正确的正样本")
else:
    # 随机取一条
    idx_single = np.random.choice(pos_correct, 1)[0]
    sample_seq = seq_test[idx_single]
    sample_tensor = torch.tensor(X_test[idx_single:idx_single+1]).to(DEVICE)

    # 提取模型注意力
    model.eval()
    with torch.no_grad():
        logits, attn = model(sample_tensor, return_attn=True)
        pred_prob = torch.sigmoid(logits).item()

    # 拉伸 attention 对齐原始序列
    attn_weights = attn.squeeze().cpu().numpy()
    attn_weights_stretched = np.repeat(attn_weights, 2)
    if len(attn_weights_stretched) < len(sample_seq):
        attn_weights_stretched = np.append(attn_weights_stretched, 0)
    elif len(attn_weights_stretched) > len(sample_seq):
        attn_weights_stretched = attn_weights_stretched[:len(sample_seq)]

    # 绘制单条样本 attention heatmap
    plt.figure(figsize=(14,3))
    positions = np.arange(len(sample_seq))
    plt.bar(positions, attn_weights_stretched)
    plt.xticks(positions, list(sample_seq), fontsize=8)
    plt.title(f"Attention Heatmap (Predicted Prob: {pred_prob:.4f})")
    plt.xlabel("RNA Sequence Position")
    plt.ylabel("Attention Weight")
    plt.tight_layout()

    #保存热力图（高分辨率 PNG）
    plt.savefig("attention_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()   # 释放内存

# 多条预测正确正样本，用于 motif logo
num_samples_to_use = min(len(pos_correct), 200)
if num_samples_to_use < 50:
    print(f"只有 {num_samples_to_use} 条预测正确正样本，可能不足50条")
selected_idx = np.random.choice(pos_correct, size=num_samples_to_use, replace=False)

attn_matrix = []
seq_onehot_matrix = []

model.eval()
with torch.no_grad():
    for idx in selected_idx:
        # 提取 attention
        sample_tensor = torch.tensor(X_test[idx:idx+1]).to(DEVICE)
        _, attn = model(sample_tensor, return_attn=True)
        attn_weights = attn.squeeze().cpu().numpy()
        # 拉伸
        seq_len = len(seq_test[idx])
        attn_weights_stretched = np.repeat(attn_weights, 2)
        if len(attn_weights_stretched) < seq_len:
            attn_weights_stretched = np.append(attn_weights_stretched, 0)
        elif len(attn_weights_stretched) > seq_len:
            attn_weights_stretched = attn_weights_stretched[:seq_len]
        attn_matrix.append(attn_weights_stretched)

        # 构建 one-hot 序列矩阵
        seq = seq_test[idx].replace('T','U')
        onehot = np.zeros((len(seq), 4), dtype=float)
        mapping = {'A':0,'C':1,'G':2,'U':3}
        for i, b in enumerate(seq):
            if b in mapping:
                onehot[i, mapping[b]] = 1.0
        seq_onehot_matrix.append(onehot)

attn_matrix = np.array(attn_matrix)            # [num_samples, seq_len]
seq_onehot_matrix = np.array(seq_onehot_matrix) # [num_samples, seq_len, 4]

# 注意力加权
weighted_matrix = np.mean(seq_onehot_matrix * attn_matrix[:,:,np.newaxis], axis=0)

# 转 DataFrame 给 logomaker
df_logo = pd.DataFrame(weighted_matrix, columns=['A','C','G','U'])

# 绘制 motif logo
plt.figure(figsize=(12,4))
logo = logomaker.Logo(df_logo)
logo.style_spines(visible=False)
logo.style_spines(spines=['left','bottom'], visible=True)
logo.style_xticks(rotation=90, fmt='%d', anchor=0)
plt.title(f"RBP Motif Logo (n={num_samples_to_use} positive samples)")


#保存 Motif Logo（高分辨率 PNG）
plt.savefig("motif_logo.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()
