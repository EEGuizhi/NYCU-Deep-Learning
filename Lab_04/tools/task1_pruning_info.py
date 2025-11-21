# BSChen
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from resnet20 import CifarResNet
@torch.no_grad()
def profile_model_per_layer(
    model: nn.Module,
    input_size: Tuple[int, ...],   # e.g. (1,3,32,32)  (包含 batch 維度)
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """
    跑一次前向，量每個模組輸出形狀，並對 Conv/Linear 計算：
      - dense_params, dense_MACs
      - nonzero_params（非零權重數）
      - zero_ratio(%)
      - est_sparse_MACs（用通道級活躍度縮放）
    其他層會列出名稱但參數/MACs 為 0（不影響總和）
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    was_training = model.training
    model.eval().to(device)

    # 掛 hooks 來收集輸出空間大小
    outputs: Dict[str, Tuple[int,int,int,int]] = {}
    handles = []

    def _make_hook(name):
        def _hook(_m, _in, out):
            # 統一轉成 Tensor 後取 shape；支援 tuple/list 只記第一個
            if isinstance(out, (list, tuple)):
                out = out[0]
            if isinstance(out, torch.Tensor):
                outputs[name] = tuple(out.shape)  # e.g. (N, C, H, W) or (N, F)
        return _hook

    # 為了有穩定名稱序，走訪 named_modules
    names = []
    for name, m in model.named_modules():
        names.append((name, m))
        # 只對會輸出的模組掛 hook；避免 nn.Sequential 再掛一次
        if len(list(m.children())) == 0:
            handles.append(m.register_forward_hook(_make_hook(name)))

    # 做一次 dummy forward，量輸出
    dummy = torch.randn(*input_size, device=device)
    _ = model(dummy)

    for h in handles:
        h.remove()

    # 開始統計
    rows = []
    idx = 0
    for name, m in names:
        class_name = m.__class__.__name__
        match_key = f"{class_name}:{name}"
        dense_params = nonzero_params = 0
        dense_macs = est_sparse_macs = 0
        zero_ratio = np.nan

        if isinstance(m, (nn.Conv2d, nn.Linear)):
            W = m.weight.detach()
            dense_params = int(W.numel())
            nonzero_params = int((W != 0).sum().item())
            zero_ratio = 100.0 * (1 - nonzero_params / max(1, dense_params))

            if isinstance(m, nn.Conv2d):
                # 取得輸出空間大小
                out_shape = outputs.get(name, None)
                if out_shape is not None and len(out_shape) >= 4:
                    # (N, C_out, H_out, W_out)
                    _, Cout, Hout, Wout = out_shape
                else:
                    # 若沒捕到，就用 0，避免炸掉
                    Cout, Hout, Wout = m.out_channels, 0, 0

                Cin = m.in_channels
                Kh, Kw = m.kernel_size
                G = m.groups if hasattr(m, "groups") else 1

                # Dense MACs
                dense_macs = int(Cout * Hout * Wout * (Cin // G) * Kh * Kw)

                # 通道級活躍度（輸出/輸入）
                # 活躍輸出通道：該 filter 是否全 0
                if W.dim() == 4:
                    active_out = int((W.abs().sum(dim=(1,2,3)) > 0).sum().item())
                    # 活躍輸入通道：對所有輸出與 kernel 求和，沿 dim=(0,2,3)
                    active_in  = int((W.abs().sum(dim=(0,2,3)) > 0).sum().item())
                else:
                    # 理論上不會進來
                    active_out = Cout
                    active_in  = Cin

                out_scale = active_out / max(1, Cout)
                in_scale  = active_in  / max(1, Cin)
                # groups 的情況：等效輸入通道是 Cin/G
                # 但我們的 scale 是在 Cin/Cout 上的比例，對 dense_macs 乘以 scale 即可
                est_sparse_macs = int(dense_macs * out_scale * in_scale)

            elif isinstance(m, nn.Linear):
                out_f, in_f = m.weight.shape
                # Dense MACs（忽略 batch）
                dense_macs = int(in_f * out_f)

                # 活躍輸出/輸入
                active_out = int((W.abs().sum(dim=1) > 0).sum().item())
                active_in  = int((W.abs().sum(dim=0) > 0).sum().item())
                out_scale = active_out / max(1, out_f)
                in_scale  = active_in  / max(1, in_f)
                est_sparse_macs = int(dense_macs * out_scale * in_scale)

        rows.append({
            "idx": idx,
            "name": name,
            "type": class_name,
            "match_key": match_key,
            "dense_params": dense_params,
            "nonzero_params": nonzero_params,
            "zero_ratio(%)": zero_ratio,
            "dense_MACs": dense_macs,
            "est_sparse_MACs": est_sparse_macs,
        })
        idx += 1

    df = pd.DataFrame(rows)
    # 附上一些單位友善列
    for col in ("dense_params", "nonzero_params"):
        df[col + "_M"] = df[col] / 1e6
    for col in ("dense_MACs", "est_sparse_MACs"):
        df[col + "_M"] = df[col] / 1e6

    model.train(was_training)
    return df

def compare_sparse_aware(df_before: pd.DataFrame, df_after: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    用 match_key 對齊（如失敗會自動用 idx 對齊），輸出差異表與總結。
    """
    # 先試 match_key
    merged = pd.merge(
        df_before, df_after, on="match_key", how="outer", suffixes=("_before", "_after"), validate="one_to_one"
    )

    # 若發現對齊很差（大量 NaN），退回 idx 對齊
    bad_align = merged["dense_params_before"].isna().sum() + merged["dense_params_after"].isna().sum()
    if bad_align > max(5, 0.2 * len(merged)):
        df_b = df_before.copy(); df_b["idx_key"] = df_b["idx"]
        df_a = df_after.copy();  df_a["idx_key"] = df_a["idx"]
        merged = pd.merge(df_b, df_a, on="idx_key", how="outer", suffixes=("_before", "_after"))
        merged["match_key"] = merged.get("match_key_before", merged.get("match_key_after"))

    # 填 0
    for col in ("dense_params_before","dense_params_after","nonzero_params_before","nonzero_params_after",
                "dense_MACs_before","dense_MACs_after","est_sparse_MACs_before","est_sparse_MACs_after"):
        merged[col] = merged[col].fillna(0).astype(int)

    # 顯示欄
    merged["type_show"] = merged["type_before"].fillna(merged["type_after"])
    merged["name_show"] = merged["name_before"].fillna(merged["name_after"])
    merged["idx_show"]  = merged["idx_before"].fillna(merged["idx_after"]).astype("Int64")

    merged["nonzero_params_diff"] = merged["nonzero_params_after"] - merged["nonzero_params_before"]
    merged["dense_params_diff"]   = merged["dense_params_after"]   - merged["dense_params_before"]
    merged["dense_MACs_diff"]     = merged["dense_MACs_after"]     - merged["dense_MACs_before"]
    merged["sparse_MACs_diff"]    = merged["est_sparse_MACs_after"]- merged["est_sparse_MACs_before"]

    out = merged[[
        "idx_show","type_show","name_show",
        "dense_params_before","dense_params_after","dense_params_diff",
        "nonzero_params_before","nonzero_params_after","nonzero_params_diff",
        "dense_MACs_before","dense_MACs_after","dense_MACs_diff",
        "est_sparse_MACs_before","est_sparse_MACs_after","sparse_MACs_diff",
    ]].sort_values(by=["idx_show"], kind="mergesort").reset_index(drop=True)

    # 總結
    def _sum(df, col): return int(df[col].sum())
    summary = {
        "dense_params_before": _sum(df_before,"dense_params"),
        "dense_params_after":  _sum(df_after,"dense_params"),
        "nonzero_params_before": _sum(df_before,"nonzero_params"),
        "nonzero_params_after":  _sum(df_after,"nonzero_params"),
        "dense_MACs_before": _sum(df_before,"dense_MACs"),
        "dense_MACs_after":  _sum(df_after,"dense_MACs"),
        "est_sparse_MACs_before": _sum(df_before,"est_sparse_MACs"),
        "est_sparse_MACs_after":  _sum(df_after,"est_sparse_MACs"),
    }
    summary.update({
        "param_zero_ratio_before(%)": 100.0 * (1 - summary["nonzero_params_before"] / max(1, summary["dense_params_before"])),
        "param_zero_ratio_after(%)":  100.0 * (1 - summary["nonzero_params_after"]  / max(1, summary["dense_params_after"])),
        "est_MACs_reduction_from_dense_before(%)": 100.0 * (1 - summary["est_sparse_MACs_before"] / max(1, summary["dense_MACs_before"])),
        "est_MACs_reduction_from_dense_after(%)":  100.0 * (1 - summary["est_sparse_MACs_after"]  / max(1, summary["dense_MACs_after"])),
        "delta_nonzero_params(absolute)": summary["nonzero_params_after"] - summary["nonzero_params_before"],
        "delta_est_sparse_MACs(absolute)": summary["est_sparse_MACs_after"] - summary["est_sparse_MACs_before"],
    })
    return out, summary


def compare_model_summaries(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame
    ) -> Tuple[pd.DataFrame, dict]:
    """Compare two model summaries (before and after pruning)."""
    # 以 match_key 左右連接；若你的模型在剪枝後層序完全改名，也能回退到 'idx' 對齊
    merged = pd.merge(
        df_before,
        df_after,
        on="match_key",
        how="outer",
        suffixes=("_before", "_after"),
        validate="one_to_one",
    )

    # 填空值（代表新增或消失的層）
    for col in ("num_params_before", "num_params_after", "mult_adds_before", "mult_adds_after"):
        merged[col] = merged[col].fillna(0).astype(int)

    merged["params_diff"] = merged["num_params_after"] - merged["num_params_before"]
    merged["MACs_diff"] = merged["mult_adds_after"] - merged["mult_adds_before"]

    # 便於閱讀的列
    show_cols = [
        "idx_before", "type_before", "name_before",
        "num_params_before", "num_params_after", "params_diff",
        "mult_adds_before", "mult_adds_after", "MACs_diff",
    ]
    # 有些層只出現在 after，補齊類型/名稱
    merged["type_before"] = merged["type_before"].fillna(merged["type_after"])
    merged["name_before"] = merged["name_before"].fillna(merged["name_after"])
    merged["idx_before"] = merged["idx_before"].fillna(merged["idx_after"])

    out = merged[show_cols].sort_values(by=["idx_before"], kind="mergesort").reset_index(drop=True)

    # 總結
    total_before_params = int(df_before["num_params"].sum())
    total_after_params  = int(df_after["num_params"].sum())
    total_before_macs   = int(df_before["mult_adds"].sum())
    total_after_macs    = int(df_after["mult_adds"].sum())

    summary = {
        "total_params_before": total_before_params,
        "total_params_after": total_after_params,
        "total_params_diff": total_after_params - total_before_params,
        "total_MACs_before": total_before_macs,
        "total_MACs_after": total_after_macs,
        "total_MACs_diff": total_after_macs - total_before_macs,
        "pruned_params_ratio(%)": 100.0 * (1 - (total_after_params / max(1, total_before_params))),
        "pruned_MACs_ratio(%)": 100.0 * (1 - (total_after_macs / max(1, total_before_macs))),
    }
    return out, summary



if __name__ == "__main__":
    # Settings
    device = "cpu"
    path = "313510156_pruning.pt"
    log_file = "pruning_info.txt"

    # Initialize log
    with open(log_file, "w") as f:
        f.write("Pruning Comparison Report\n\n")

    # Original pre-trained model
    model = torch.hub.load(
        'chenyaofo/pytorch-cifar-models', 'cifar10_resnet20', pretrained=True
    ).to(device)
    model.eval()

    # Pruned model from file
    state_dict = torch.load(path, weights_only=True)
    if not state_dict:
        raise RuntimeError("Failed to load the pruned model state_dict.")
    model_pruned = CifarResNet()
    model_pruned.load_state_dict(state_dict)
    model_pruned.to(device)
    model_pruned.eval()

    # Input size for CIFAR-10
    input_size = (1, 3, 32, 32)  # (batch, C, H, W)
    df_before = profile_model_per_layer(model, input_size=input_size, device=torch.device(device))
    df_after  = profile_model_per_layer(model_pruned, input_size=input_size, device=torch.device(device))
    # df_before = torchinfo_perlayer_df(model, input_size=input_size, batch_dim_included=True)
    # df_after = torchinfo_perlayer_df(model_pruned, input_size=input_size, batch_dim_included=True)

    # Compare
    per_layer_diff, overall = compare_sparse_aware(df_before, df_after)
    # per_layer_diff, overall = compare_model_summaries(df_before, df_after)

    # Print results
    print("\n[Per-layer params/MACs diff]", file=open(log_file, "a"))
    print(tabulate(per_layer_diff, headers="keys", tablefmt="github", floatfmt=".3f"), file=open(log_file, "a"))

    print("\n[Overall change]", file=open(log_file, "a"))
    for k, v in overall.items():
        if "ratio" in k:
            print(f"{k}: {v:.2f}%", file=open(log_file, "a"))
        else:
            print(f"{k}: {v:,}", file=open(log_file, "a"))

    # # Optional: Save to CSV
    # per_layer_diff.to_csv("pruning_compare_per_layer.csv", index=False)
    # df_before.to_csv("model_before_perlayer.csv", index=False)
    # df_after.to_csv("model_after_perlayer.csv", index=False)
    # print("\n已輸出: pruning_compare_per_layer.csv / model_before_perlayer.csv / model_after_perlayer.csv")
