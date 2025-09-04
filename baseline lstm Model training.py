import tensorflow as tf
import numpy as np
import os
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------- 自定义 R² Callback（适用于单输出模型） ---------------------- #
class R2Callback(tf.keras.callbacks.Callback):
    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.train_r2 = []
        self.val_r2 = []

    def on_epoch_end(self, epoch, logs=None):
        # 对于 LSTM 模型，predict 返回单个输出
        y_train_pred = self.model.predict(self.X_train, verbose=0)
        y_val_pred = self.model.predict(self.X_val, verbose=0)
        y_train_true = self.y_train.flatten()
        y_val_true = self.y_val.flatten()
        y_train_pred = y_train_pred.flatten()
        y_val_pred = y_val_pred.flatten()
        ss_res_train = np.sum((y_train_true - y_train_pred) ** 2)
        ss_tot_train = np.sum((y_train_true - np.mean(y_train_true)) ** 2)
        r2_train = 1 - ss_res_train / ss_tot_train if ss_tot_train != 0 else 0
        ss_res_val = np.sum((y_val_true - y_val_pred) ** 2)
        ss_tot_val = np.sum((y_val_true - np.mean(y_val_true)) ** 2)
        r2_val = 1 - ss_res_val / ss_tot_val if ss_tot_val != 0 else 0
        self.train_r2.append(r2_train)
        self.val_r2.append(r2_val)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train R²: {r2_train:.4f}, Val R²: {r2_val:.4f}")

# ---------------------- 数据加载与增强 ---------------------- #
def load_data(path, noise_level=0.02):
    if path.endswith('.zip'):
        with zipfile.ZipFile(path, 'r') as z:
            csv_files = [f for f in z.namelist() if f.lower().endswith('.csv')]
            if not csv_files:
                raise ValueError("ZIP文件中未找到CSV文件")
            extract_path = os.path.join(os.path.dirname(path), "temp_extract")
            os.makedirs(extract_path, exist_ok=True)
            z.extract(csv_files, path=extract_path)
            path = os.path.join(extract_path, csv_files[0])

    encodings = ['utf-8', 'gbk', 'utf-16', 'latin1']
    for encoding in encodings:
        try:
            with open(path, 'r', encoding=encoding) as f:
                first_line = f.readline()
                sep = '\t' if '\t' in first_line else ','
            print(f"检测到编码: {encoding}，分隔符: {repr(sep)}")
            break
        except (UnicodeDecodeError, IsADirectoryError):
            continue
    else:
        raise ValueError("无法自动识别文件编码")

    df = pd.read_csv(path, sep=sep, engine='python', encoding=encoding, on_bad_lines='warn')
    required_columns = ['Nm', 'Pw', 'mw', 'ma', 'Cs', 'q', 'H', 'D', 'u', 'actual_U']
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"缺失关键列: {missing_cols}")

    df = df.dropna().reset_index(drop=True)
    df = df[(df['q'] > 0) & (df['H'] > 0) & (df['D'] > 0)]
    features = ['Nm', 'Pw', 'mw', 'ma', 'q', 'H', 'D']
    target = 'actual_U'

    scaler = MinMaxScaler(feature_range=(1e-5, 1))
    X_scaled = scaler.fit_transform(df[features])
    # 如有需要可添加高斯噪声：X_scaled += np.random.normal(0, noise_level, X_scaled.shape)

    return X_scaled, df[target].values.reshape(-1, 1), scaler, features

# ---------------------- 定义两层 LSTM 模型 ---------------------- #
def build_lstm_model(num_features):
    model = tf.keras.Sequential([
        # 此处输入shape为 (timesteps, num_features)；这里我们设置 timesteps=1
        tf.keras.layers.Input(shape=(1, num_features)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(1)
    ])
    return model

# ---------------------- 训练与评估过程 ---------------------- #
def main():
    DATA_PATH = r"E:\dataset\water.csv"  # 请修改为你的数据路径
    FIG_PATH = r"E:\PINN explore\pinn+lstm"  # 请修改为你保存结果的文件夹
    os.makedirs(FIG_PATH, exist_ok=True)

    try:
        X, y, scaler, features = load_data(DATA_PATH)
        print(f"数据加载成功，样本数: {len(X)}")
        # 将输入调整为三维数据，每个样本为 (1, num_features)
        X_data = X.reshape(-1, 1, len(features))
        print("输入数据形状:", X_data.shape)
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        return

    X_train, X_val, y_train, y_val = train_test_split(X_data, y, test_size=0.2, random_state=42)

    # 构建两层 LSTM 模型
    model = build_lstm_model(num_features=len(features))
    # 设置学习率和优化器
    model.compile(optimizer=tf.keras.optimizers.Adam(0.005),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['mae'])

    # 创建 R² Callback
    r2_callback = R2Callback(X_train, y_train, X_val, y_val)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=3000,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=800, restore_best_weights=True, monitor='val_loss'),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(FIG_PATH, 'best_model.ckpt.lstm'),
                save_best_only=True,
                monitor='val_loss',
                save_weights_only=True
            ),
            r2_callback
        ],
        verbose=2
    )

    # ------------------- 使用 Plotly 绘制训练过程图 ------------------- #
    fig = make_subplots(rows=3, cols=1, subplot_titles=(
        'Training and Validation Loss', 'Training and Validation MAE', 'Training and Validation R²'))
    fig.add_trace(
        go.Scatter(x=np.arange(len(history.history['loss'])), y=history.history['loss'],
                   mode='lines', name='Training Loss'),
        row=1, col=1)
    fig.add_trace(
        go.Scatter(x=np.arange(len(history.history['val_loss'])), y=history.history['val_loss'],
                   mode='lines', name='Validation Loss'),
        row=1, col=1)
    fig.add_trace(
        go.Scatter(x=np.arange(len(history.history['mae'])), y=history.history['mae'],
                   mode='lines', name='Training MAE'),
        row=2, col=1)
    fig.add_trace(
        go.Scatter(x=np.arange(len(history.history['val_mae'])), y=history.history['val_mae'],
                   mode='lines', name='Validation MAE'),
        row=2, col=1)
    fig.add_trace(
        go.Scatter(x=np.arange(len(r2_callback.train_r2)), y=r2_callback.train_r2,
                   mode='lines', name='Training R²'),
        row=3, col=1)
    fig.add_trace(
        go.Scatter(x=np.arange(len(r2_callback.val_r2)), y=r2_callback.val_r2,
                   mode='lines', name='Validation R²'),
        row=3, col=1)
    fig.update_layout(height=900, width=1600, title_text="Model Training Process (Interactive Chart)", showlegend=True)
    fig.update_xaxes(title_text="Epochs", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1, tickformat='.2f')
    fig.update_xaxes(title_text="Epochs", row=2, col=1)
    fig.update_yaxes(title_text="MAE", row=2, col=1, tickformat='.2f')
    fig.update_xaxes(title_text="Epochs", row=3, col=1)
    fig.update_yaxes(title_text="R²", row=3, col=1, tickformat='.2f')
    fig.write_html(os.path.join(FIG_PATH, 'training_process_interactive_chart.html'))

    # ------------------- 使用 Plotly 绘制实际值与预测值对比图 ------------------- #
    y_pred = model.predict(X_val)
    num_samples = min(len(y_val), 20)
    selected_indices = np.sort(np.random.choice(len(y_val), num_samples, replace=False))
    y_val_selected = y_val.flatten()[selected_indices]
    y_pred_selected = y_pred.flatten()[selected_indices]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=selected_indices,
        y=y_val_selected,
        mode='markers+lines',
        name='Actual Values'
    ))
    fig2.add_trace(go.Scatter(
        x=selected_indices,
        y=y_pred_selected,
        mode='markers+lines',
        name='Predicted Values'
    ))
    fig2.update_layout(
        title="Comparison of Actual and Predicted Values (20 Samples)",
        xaxis_title="Sample Index",
        yaxis_title="Value",
        showlegend=True
    )
    fig2.write_html(os.path.join(FIG_PATH, 'actual_vs_predicted_interactive_chart.html'))

    # ------------------- 计算评估指标 ------------------- #
    y_val_flat = y_val.flatten()
    y_pred_flat = y_pred.flatten()
    ss_res = np.sum((y_val_flat - y_pred_flat) ** 2)
    ss_tot = np.sum((y_val_flat - np.mean(y_val_flat)) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"R²: {r2:.4f}")

    mape = np.mean(np.abs((y_val_flat - y_pred_flat) / (y_val_flat + 1e-8))) * 100.0
    print(f"LSTM预测 MAPE: {mape:.2f}%")

    # ------------------- 使用 Matplotlib 绘制结果图 ------------------- #
    # 1. Loss 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Loss Decline During Training", fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_PATH, 'loss_plot.png'), dpi=300)
    plt.close()

    # 2. MAE 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['mae'], label='Training MAE', linewidth=2)
    plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("MAE", fontsize=14)
    plt.title("MAE Changes During Training", fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_PATH, 'mae_plot.png'), dpi=300)
    plt.close()

    # 3. R² 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(r2_callback.train_r2, label='Training R²', linewidth=2)
    plt.plot(r2_callback.val_r2, label='Validation R²', linewidth=2)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("R²", fontsize=14)
    plt.title("R² Changes During Training", fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_PATH, 'r2_plot.png'), dpi=300)
    plt.close()

    # 4. 散点图：实际值 vs. LSTM 预测
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(y_val_flat)), y_val_flat, c='black', s=30, label='Actual Values', marker='o')
    plt.scatter(range(len(y_pred_flat)), y_pred_flat, c='red', s=30, label='Predicted Values', marker='^')
    plt.xlabel("Sample Index", fontsize=14)
    plt.ylabel("U Value", fontsize=14)
    plt.title("Actual vs. LSTM Predicted Values", fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_PATH, 'scatter_plot.png'), dpi=300)
    plt.close()

    print(f"训练完成！结果已保存至：{FIG_PATH}")

if __name__ == "__main__":
    main()
