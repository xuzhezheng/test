import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error


def evaluate_prediction(T_true, T_hat, save_path=None):
    """
    评估预测结果，计算 R² 和 RMSE，并绘制预测 vs 真实曲线图。

    参数:
        T_true: np.ndarray, shape (n_samples, d) 或 (n_samples,)
        T_hat: np.ndarray, 同上，预测值
        save_path: str, 可选，若指定则保存图像到文件路径

    返回:
        r2: float, 决定系数
        rmse: float, 均方根误差
    """
    T_true = np.array(T_true)
    T_hat = np.array(T_hat)

    # 若是多维 T，每列分别评估
    if T_true.ndim == 2 and T_true.shape[1] > 1:
        for i in range(T_true.shape[1]):
            _plot_and_score(T_true[:, i], T_hat[:, i], i, save_path)
    else:
        r2 = r2_score(T_true, T_hat)
        rmse = mean_squared_error(T_true, T_hat, squared=True)
        rmae = mean_absolute_error(T_true, T_hat)

        plt.figure(figsize=(8, 6))
        plt.plot(T_true, label="True T", linewidth=2)
        plt.plot(T_hat, label="Predicted T", linestyle='--', linewidth=2)
        plt.title(f"Prediction vs True\n$R^2$ = {r2:.3f}, RMSE = {rmse:.3f}")
        plt.xlabel("Sample index")
        plt.ylabel("T value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.show()

        return r2, rmse,rmae


def _plot_and_score(true_col, pred_col, index, save_path=None):
    r2 = r2_score(true_col, pred_col)
    rmse = mean_squared_error(true_col, pred_col, squared=False)

    plt.figure(figsize=(8, 6))
    plt.plot(true_col, label=f"True T[:,{index}]", linewidth=2)
    plt.plot(pred_col, label=f"Predicted T[:,{index}]", linestyle='--', linewidth=2)
    plt.title(f"T dimension {index}\n$R^2$ = {r2:.3f}, RMSE = {rmse:.3f}")
    plt.xlabel("Sample index")
    plt.ylabel("T value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}_dim{index}.png")
    plt.show()
