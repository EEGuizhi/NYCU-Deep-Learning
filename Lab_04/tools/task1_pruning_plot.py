import pandas as pd
import matplotlib.pyplot as plt

PATH = "/home/bschen/deep_learning/Lab_04/task1_log.csv"

# # === 1️⃣ 讀取 CSV ===
# # 假設你的檔名為 "loss_log.csv"
# df = pd.read_csv(PATH, sep=r'\s+', engine='python')  # 自動處理 tab 或空白分隔
# df.columns = [c.strip() for c in df.columns]  # 清理欄位名稱（避免有多餘空格）

# # === 2️⃣ 建立一個統一的 x 軸（Phase+Epoch累計） ===
# # 讓不同 Phase 接續在一起畫
# phase_offsets = {}
# offset = 0
# x_values = []
# for phase in df['Phase'].unique():
#     phase_len = len(df[df['Phase'] == phase])
#     phase_offsets[phase] = offset
#     offset += phase_len
# for i, row in df.iterrows():
#     x_values.append(phase_offsets[row['Phase']] + row['Epoch'])
# df['x'] = x_values

# # === 3️⃣ 畫圖 ===
# plt.figure(figsize=(12, 6))

# plt.plot(df['x'], df['Train'], label='Train Loss', color='steelblue', linewidth=2)
# plt.plot(df['x'], df['Test'], label='Test Loss', color='darkorange', linewidth=2)

# # === 4️⃣ 用垂直虛線分隔每個 Phase ===
# for phase in df['Phase'].unique():
#     phase_data = df[df['Phase'] == phase]
#     start_x = phase_data['x'].min()
#     end_x = phase_data['x'].max()
#     plt.axvline(x=start_x, color='gray', linestyle='--', alpha=0.5)
#     plt.text((start_x + end_x)/2, max(df['Train'].max(), df['Test'].max()) * 1.02,
#              f'Phase {phase}', ha='center', va='bottom', fontsize=10, color='black')

# plt.xlabel('Training Progress (Phases separated by dashed lines)')
# plt.ylabel('Loss')
# plt.title('Train/Test Loss Convergence by Phase')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.3)
# plt.tight_layout()
# plt.savefig("/home/bschen/deep_learning/Lab_04/task1_loss_plot.png")

import pandas as pd
import matplotlib.pyplot as plt
import re

# === Step 1: 讀取檔案，試著自動清理 ===
file_path = PATH

# 偵測分隔符號
with open(file_path, "r") as f:
    first_line = f.readline()
if "," in first_line:
    sep = ","
elif "\t" in first_line:
    sep = "\t"
else:
    sep = r"\s+"

df = pd.read_csv(file_path, sep=sep, engine="python")
df.columns = [c.strip() for c in df.columns]

# 清理欄位名稱
def clean_col(col):
    col = col.lower()
    col = re.sub(r'[^a-z0-9]+', '_', col)
    col = re.sub(r'_+', '_', col)
    return col.strip('_')

df.columns = [clean_col(c) for c in df.columns]

# 自動欄位對應
phase_col = [c for c in df.columns if 'phase' in c][0]
epoch_col = [c for c in df.columns if 'epoch' in c][0]
train_col = [c for c in df.columns if 'train' in c and 'loss' in c][0]
test_col = [c for c in df.columns if ('test' in c or 'val' in c) and 'loss' in c][0]

# 型別轉換
df[phase_col] = df[phase_col].astype(int)
df[epoch_col] = df[epoch_col].astype(int)
df[train_col] = df[train_col].astype(float)
df[test_col] = df[test_col].astype(float)

# === Step 2: 生成連續的 global epoch 編號 ===
global_epochs = []
offset = 0
for phase in sorted(df[phase_col].unique()):
    sub_df = df[df[phase_col] == phase]
    global_epochs.extend(list(range(offset + 1, offset + len(sub_df) + 1)))
    offset += len(sub_df)
df["global_epoch"] = global_epochs

# === Step 3: 畫圖 ===
plt.figure(figsize=(12, 6))

# 統一顏色與標籤
plt.plot(df["global_epoch"], df[train_col], label="Train Loss", color="tab:blue")
plt.plot(df["global_epoch"], df[test_col], label="Test Loss", color="tab:orange")

# 加上垂直線分隔每個 phase
phase_boundaries = []
offset = 0
for phase in sorted(df[phase_col].unique()):
    phase_len = len(df[df[phase_col] == phase])
    offset += phase_len
    phase_boundaries.append(offset)

for boundary in phase_boundaries[:-1]:  # 最後一個不用畫
    plt.axvline(boundary, color="gray", linestyle="--", alpha=0.6)

# === Step 4: 標籤與排版 ===
plt.xlabel("Global Epoch")
plt.ylabel("Loss")
plt.title("Training and Testing Loss across Phases")
plt.legend(loc="best")
plt.grid(True, linestyle="--", alpha=0.4)

# 在上方標示 Phase 編號
midpoints = []
offset = 0
for phase in sorted(df[phase_col].unique()):
    phase_len = len(df[df[phase_col] == phase])
    mid = offset + phase_len / 2
    text = f"Phase {phase}" if phase < 2 or phase > 10 else ""
    plt.text(mid, plt.ylim()[1] * 0.98, text, ha="center", va="top", fontsize=9, color="black")
    offset += phase_len

plt.tight_layout()
plt.savefig("/home/bschen/deep_learning/Lab_04/task1_loss_plot.png")
