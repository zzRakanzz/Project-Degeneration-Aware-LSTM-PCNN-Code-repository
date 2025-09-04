import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

tfd = tfp.distributions
tfk = tf.keras


# 自定义 Softplus 层（带完整序列化支持）
class SoftplusLayer(tfk.layers.Layer):
    def call(self, inputs):
        return 1e-3 + tf.nn.softplus(inputs)

    def get_config(self):
        return super().get_config()


# 数据加载函数（与原代码保持一致）
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
    return X_scaled, df[target].values.reshape(-1, 1), scaler, features


# 计算 R² 与 MAPE 的函数
def compute_r2(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - ss_res / ss_total


def compute_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100


# PINN+LSTM 模型
class PINNLSTMModel(tfk.Model):
    def __init__(self, num_features, scaler_min, scaler_scale, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        # 保存缩放参数（可序列化）
        self.scaler_min = scaler_min.tolist() if isinstance(scaler_min, np.ndarray) else scaler_min
        self.scaler_scale = scaler_scale.tolist() if isinstance(scaler_scale, np.ndarray) else scaler_scale
        self.scaler_min_tensor = tf.constant(self.scaler_min, dtype=tf.float32)
        self.scaler_scale_tensor = tf.constant(self.scaler_scale, dtype=tf.float32)

        # 网络结构
        self.lstm_layer = tfk.layers.LSTM(16, return_sequences=False,
                                          kernel_regularizer=tfk.regularizers.l2(0.01))
        self.dropout = tfk.layers.Dropout(0.01)
        self.loc_layer = tfk.layers.Dense(1, kernel_regularizer=tfk.regularizers.l2(0.001))
        self.scale_layer = tfk.layers.Dense(1, kernel_regularizer=tfk.regularizers.l2(0.001))
        self.softplus_layer = SoftplusLayer()
        # physics_corrector 无偏置，确保梯度传播
        self.physics_corrector = tfk.Sequential([
            tfk.layers.Dense(8, activation='relu', input_shape=(8,), kernel_regularizer=tfk.regularizers.l2(0.001)),
            tfk.layers.Dense(1, use_bias=False)
        ])

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_features': self.num_features,
            'scaler_min': self.scaler_min,
            'scaler_scale': self.scaler_scale
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['scaler_min'] = np.array(config['scaler_min'], dtype=np.float32)
        config['scaler_scale'] = np.array(config['scaler_scale'], dtype=np.float32)
        return cls(**config)

    def call(self, inputs, training=False):
        # LSTM部分
        x = self.lstm_layer(inputs)
        x = self.dropout(x, training=training)
        loc = self.loc_layer(x)
        scale = self.softplus_layer(self.scale_layer(x))
        temporal_dist = tfd.Normal(loc=loc, scale=scale)
        temporal_mean = temporal_dist.mean()
        # 物理修正部分
        inputs_flat = tf.squeeze(inputs, axis=1)
        correction_input = tf.concat([inputs_flat, temporal_mean], axis=-1)
        correction = self.physics_corrector(correction_input)
        # 物理公式计算
        physics_output = self.physics_model(inputs, training=training)

        corrected_output = physics_output * tf.exp(0.05 * correction)
        # 返回元组，便于损失函数同时获取修正预测、物理预测及不确定性参数
        return corrected_output, physics_output, loc, scale

    @tf.function
    def physics_model(self, inputs, training=False):
        inputs_flat = tf.cast(tf.squeeze(inputs, axis=1), tf.float32)
        inputs_raw = (inputs_flat - self.scaler_min_tensor) / self.scaler_scale_tensor
        Nm, Pw, mw, ma, q, H, D = tf.unstack(inputs_raw, axis=1)
        numerator = Nm * (tf.maximum(Pw, 1e-5) ** 1.25) * (tf.maximum(mw, 1e-5) ** 0.687) * (
                    tf.maximum(ma, 1e-5) ** 0.343)
        denominator = 9040 * tf.maximum(q, 1e-5) * tf.maximum(H, 1e-5) * (tf.maximum(D, 1e-5) ** 0.618) + 1e-8
        return tf.expand_dims((numerator / denominator) ** 1.15, axis=-1)


# 自定义损失函数，传入真实值、预测值（元组中的修正预测与物理预测）以及权重参数
def compute_loss(y_true, corrected_output, physics_output, loc, scale, model,
                 weight_data=0.6, weight_physics=0.1, weight_reg=0.3):
    # 确保 y_true 为 float32 类型
    y_true = tf.cast(y_true, tf.float32)
    # 数据损失：修正预测与真实值之间的均方误差
    data_loss = tf.reduce_mean(tf.square(y_true - corrected_output))
    # 物理损失：物理公式预测与修正预测之间的均方误差
    physics_mae = tf.reduce_mean(tf.abs(corrected_output - physics_output))
    adaptive_weight = tf.minimum(0.5, 1.0 / (physics_mae + 1e-8))
    physics_loss = adaptive_weight * tf.reduce_mean(tf.square((corrected_output - physics_output)))
    # KL散度作为正则化损失：这里利用不确定性分布估计
    temporal_dist = tfd.Normal(loc=loc, scale=scale)
    log_prob = temporal_dist.log_prob(corrected_output)
    kl_loss = -tf.reduce_mean(log_prob)
    total_loss = weight_data * data_loss + weight_physics * physics_loss + weight_reg * kl_loss
    return total_loss, data_loss, physics_loss, kl_loss


# 自定义训练循环（参考代码B架构，同时保存每个epoch的日志信息）
def train_model(model, X_train, y_train, X_val, y_val,
                weight_data=0.6, weight_physics=0.1, weight_reg=0.3,
                epochs=2300, batch_size=32, learning_rate=0.0005, patience=400):
    optimizer = tf.keras.optimizers.Adam(learning_rate, clipvalue=0.5)
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)

    # 历史记录
    history_epoch = []
    history_total_loss = []
    history_val_total_loss = []   # 验证集损失记录
    history_data_loss = []
    history_physics_loss = []
    history_reg_loss = []
    history_train_r2 = []
    history_val_r2 = []
    history_train_mape = []
    history_val_mape = []

    # 早停机制变量
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_data_loss = tf.keras.metrics.Mean()
        epoch_physics_loss = tf.keras.metrics.Mean()
        epoch_reg_loss = tf.keras.metrics.Mean()

        # 遍历每个batch进行训练
        for x_batch, y_batch in dataset:
            with tf.GradientTape() as tape:
                corrected_output, physics_output, loc, scale = model(x_batch, training=True)
                loss, data_loss, physics_loss, kl_loss = compute_loss(
                    y_batch, corrected_output, physics_output, loc, scale, model,
                    weight_data=weight_data,
                    weight_physics=weight_physics,
                    weight_reg=weight_reg
                )
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            epoch_loss_avg.update_state(loss)
            epoch_data_loss.update_state(data_loss)
            epoch_physics_loss.update_state(physics_loss)
            epoch_reg_loss.update_state(kl_loss)

        # 训练集评估
        corrected_train, _, _, _ = model(X_train, training=False)
        train_r2 = compute_r2(y_train.flatten(), corrected_train.numpy().flatten())
        train_mape = compute_mape(y_train.flatten(), corrected_train.numpy().flatten())
        # 验证集评估
        corrected_val, physics_val, loc_val, scale_val = model(X_val, training=False)
        val_r2 = compute_r2(y_val.flatten(), corrected_val.numpy().flatten())
        val_mape = compute_mape(y_val.flatten(), corrected_val.numpy().flatten())
        val_loss, _, _, _ = compute_loss(y_val, corrected_val, physics_val, loc_val, scale_val, model,
                                         weight_data=weight_data,
                                         weight_physics=weight_physics,
                                         weight_reg=weight_reg)

        # 保存历史记录
        history_epoch.append(epoch)
        history_total_loss.append(epoch_loss_avg.result().numpy())
        history_val_total_loss.append(val_loss.numpy())
        history_data_loss.append(epoch_data_loss.result().numpy())
        history_physics_loss.append(epoch_physics_loss.result().numpy())
        history_reg_loss.append(epoch_reg_loss.result().numpy())
        history_train_r2.append(train_r2)
        history_val_r2.append(val_r2)
        history_train_mape.append(train_mape)
        history_val_mape.append(val_mape)

        # 输出当前epoch的日志信息
        print(f"Epoch {epoch}: Train Loss = {epoch_loss_avg.result():.6f}, Val Loss = {val_loss:.6f}, "
              f"Data Loss = {epoch_data_loss.result():.6f}, Physics Loss = {epoch_physics_loss.result():.6f}, "
              f"KL Loss = {epoch_reg_loss.result():.6f}, Train R² = {train_r2:.5f}, Val R² = {val_r2:.5f}, "
              f"Train MAPE = {train_mape:.2f}%, Val MAPE = {val_mape:.2f}%")

        # 判断早停条件：如果验证损失有改善则重置计数器，否则累加
        if val_loss.numpy() < best_val_loss:
            best_val_loss = val_loss.numpy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"验证损失在连续 {patience} 个epoch内没有改善，提前停止训练。")
            break

    print(f"Final Validation Loss: {val_loss:.6f}")

    history = {
        "epoch": history_epoch,
        "total_loss": history_total_loss,
        "val_total_loss": history_val_total_loss,
        "data_loss": history_data_loss,
        "physics_loss": history_physics_loss,
        "reg_loss": history_reg_loss,
        "train_r2": history_train_r2,
        "val_r2": history_val_r2,
        "train_mape": history_train_mape,
        "val_mape": history_val_mape
    }
    return model, history


def plot_history(history, fig_path):
    epochs = history["epoch"]
    # 使用 Plotly 绘制交互式图表
    fig = make_subplots(rows=3, cols=1, subplot_titles=(
        'Training and Validation Loss',
        'Training and Validation MAPE',
        'Training and Validation R²'
    ))
    # 损失图：添加训练损失和验证损失曲线
    fig.add_trace(go.Scatter(x=epochs, y=history["total_loss"], mode='lines', name='Train Loss'), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history["val_total_loss"], mode='lines', name='Validation Loss'), row=1, col=1)
    # MAPE图
    fig.add_trace(go.Scatter(x=epochs, y=history["train_mape"], mode='lines', name='Train MAPE'), row=2, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history["val_mape"], mode='lines', name='Val MAPE'), row=2, col=1)
    # R²图
    fig.add_trace(go.Scatter(x=epochs, y=history["train_r2"], mode='lines', name='Train R²'), row=3, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history["val_r2"], mode='lines', name='Val R²'), row=3, col=1)

    fig.update_layout(height=900, width=1600, title_text="Training Process (Interactive Chart)", showlegend=True)
    fig.write_html(os.path.join(fig_path, 'training_process_interactive_chart.html'))

    # 用 matplotlib 绘制静态图：损失图中同时绘制训练和验证损失曲线
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["total_loss"], label='Train Loss', linewidth=2)
    plt.plot(epochs, history["val_total_loss"], label='Validation Loss', linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'loss_plot.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_mape"], label='Train MAPE', linewidth=2)
    plt.plot(epochs, history["val_mape"], label='Val MAPE', linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("MAPE (%)")
    plt.title("MAPE Changes During Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'mape_plot.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_r2"], label='Train R²', linewidth=2)
    plt.plot(epochs, history["val_r2"], label='Val R²', linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("R²")
    plt.title("R² Changes During Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'r2_plot.png'), dpi=300)
    plt.close()


def plot_predictions(y_true, y_pred_corrected, y_pred_physics, fig_path):
    num_samples = min(len(y_true), 20)
    selected_indices = np.sort(np.random.choice(len(y_true), num_samples, replace=False))
    y_true_selected = y_true.flatten()[selected_indices]
    y_pred_selected = y_pred_corrected.flatten()[selected_indices]

    # 使用 Plotly 绘制实际值与预测值对比的交互图
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=selected_indices, y=y_true_selected,
                             mode='markers+lines', name='Actual Values'))
    fig.add_trace(go.Scatter(x=selected_indices, y=y_pred_selected,
                             mode='markers+lines', name='Corrected Prediction'))
    fig.write_html(os.path.join(fig_path, 'actual_vs_predicted_interactive_chart.html'))

    # 使用 matplotlib 绘制散点图：实际值、物理预测、修正预测对比
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(y_true)), y_true, c='black', s=30, label='Actual Values', marker='o')
    plt.scatter(range(len(y_pred_physics)), y_pred_physics, c='blue', s=30, label='Physics Prediction', marker='d')
    plt.scatter(range(len(y_pred_corrected)), y_pred_corrected, c='red', s=30, label='Corrected Prediction', marker='^')
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.title("Comparison: Actual vs. Physics vs. Corrected")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'scatter_plot.png'), dpi=300)
    plt.close()


def main():
    DATA_PATH = r"E:\dataset\water.csv"
    FIG_PATH = r"E:\PINN explore\picture"
    os.makedirs(FIG_PATH, exist_ok=True)

    try:
        X, y, scaler, features = load_data(DATA_PATH)
        print(f"数据加载成功，样本数: {len(X)}")
        X_seq = X.reshape(-1, 1, len(features))
        print("输入数据形状:", X_seq.shape)
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        return

    X_train, X_val, y_train, y_val = train_test_split(X_seq, y, test_size=0.2, random_state=42)

    # 实例化模型
    model = PINNLSTMModel(
        num_features=len(features),
        scaler_min=scaler.min_,
        scaler_scale=scaler.scale_
    )

    # 使用自定义训练循环训练模型，并获取历史记录
    trained_model, history = train_model(model, X_train, y_train, X_val, y_val,
                                         weight_data=0.6, weight_physics=0.1, weight_reg=0.3,
                                         epochs=2000, batch_size=32, learning_rate=0.0005,
                                         patience=400)

    # 模型预测（取修正预测值）
    corrected_output, physics_output, loc, scale = trained_model(X_seq, training=False)
    y_pred_corrected = corrected_output.numpy().flatten()
    y_pred_physics = physics_output.numpy().flatten()
    y_true_full = y.flatten()

    print("y_pred_corrected:", y_pred_corrected)
    print("y_pred_physics:", y_pred_physics)
    mape_nn = compute_mape(y_true_full, y_pred_corrected)
    mape_physics = compute_mape(y_true_full, y_pred_physics)
    print(f"神经网络修正预测与真实值之间的 MAPE: {mape_nn:.2f}%")
    print(f"物理公式预测与真实值之间的 MAPE: {mape_physics:.2f}%")

    # 计算R²
    r2_train = compute_r2(y_train.flatten(), trained_model(X_train, training=False)[0].numpy().flatten())
    r2_val = compute_r2(y_val.flatten(), trained_model(X_val, training=False)[0].numpy().flatten())
    print(f"训练集 R²: {r2_train:.5f}, 验证集 R²: {r2_val:.5f}")

    # 绘制训练过程图表
    plot_history(history, FIG_PATH)
    # 绘制预测对比图
    plot_predictions(y_true_full, y_pred_corrected, y_pred_physics, FIG_PATH)

    # 保存完整模型
    full_model_path = os.path.join(FIG_PATH, 'full_model')
    tfk.models.save_model(trained_model, full_model_path, overwrite=True,
                          include_optimizer=True, save_format='tf')
    print(f"完整模型已保存至: {full_model_path}")


if __name__ == "__main__":
    main()
