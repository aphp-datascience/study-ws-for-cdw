import pandas as pd
import matplotlib.pyplot as plt


def plot_loss(loss_tracking, task_name, window_rolling_mean=10):
    # Sample time series data
    data = {
        # 'date': pd.date_range(start='2023-01-01', periods=30),
        "value": loss_tracking[task_name]  # just some pattern data
    }
    df = pd.DataFrame(data)
    # Calculate rolling mean with a window of 10

    df["rolling_mean"] = df["value"].rolling(window=window_rolling_mean).mean()

    # Plot original data and rolling mean
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["value"], label="Original")
    plt.plot(
        df.index,
        df["rolling_mean"],
        label=f"Rolling Mean ({window_rolling_mean})",
        color="red",
    )
    plt.title(f"Loss for {task_name}")
    plt.xlabel("step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_metric(
    metrics_tracking,
    task_name,
    metric="fscore",
    pos_label="1",
):
    values = [i[pos_label][metric] for i in metrics_tracking[task_name]]
    steps = [i["step"] for i in metrics_tracking[task_name]]
    data = {"value": values, "step": steps}
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 5))
    plt.plot(df["step"], df["value"], label="Original")

    plt.title(f"{metric} for {task_name}")
    plt.xlabel("step")
    plt.ylabel(f"{metric}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
