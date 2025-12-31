#!/usr/bin/env python3
"""
验证弯曲时间假说：引力波次峰频移(Δf)与黑洞软毛熵(S_soft)的相关性分析
依赖：GWpy, bilby, numpy, matplotlib, scipy
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from gwpy import timeseries, astro
from bilby import run_sampler, result
from bilby.gw import models, detectors
from scipy.stats import pearsonr

# =============================================================
#                1. 数据获取与预处理
# =============================================================

def fetch_gw_data(event: str, out_dir: str = "data") -> timeseries.TimeSeries:
    """
    从GWOSC获取引力波事件数据
    Args:
        event: 事件ID (e.g., "GW170817")
        out_dir: 数据保存目录
    Returns:
        处理后的时序数据
    """
    ensure_dir(out_dir)
    try:
        # 获取事件元数据（质量、自旋等）
        event_info = astro.Event(event).query()
        # 下载数据（LIGO Hanford (H1)和Livingston (L1)探测器）
        data = timeseries.TimeSeries.fetch(
            "LIGO", event, 
            start=event_info["gps_time"] - 32,  # 32秒数据窗
            end=event_info["gps_time"] + 32,
            outdir=out_dir
        )
        print(f"下载事件 {event} 数据完成")
        return data
    except Exception as e:
        print(f"下载失败: {str(e)}")
        exit(1)


def preprocess_data(data: timeseries.TimeSeries) -> np.ndarray:
    """
    数据预处理：带通滤波+去除噪声
    Args:
        data: 原始时序数据
    Returns:
        预处理后的频域数据
    """
    # 1. 带通滤波（20Hz-2000Hz，LIGO敏感频段）
    filtered = data.bandpass(20, 2000)
    # 2. 去除瞬态噪声（如非引力波事件）
    cleaned = filtered.remove_outliers(3)  # 3σ阈值
    # 3. 计算时频谱（Q-transform）
    qtransform = cleaned.qtransform(
        logf=True,  # 对数频率轴
        norm=True,  # 归一化功率
        logfsteps=100  # 频率分辨率
    )
    return qtransform


# =============================================================
#                2. 次峰特征提取（Δf）
# =============================================================

def find_secondary_peak(qtransform: np.ndarray, f_range: Tuple[float, float] = (50, 500)) -> float:
    """
    在时频谱中识别次峰并计算Δf
    Args:
        qtransform: 时频谱数据 (shape: [时间点, 频率点])
        f_range: 搜索频率范围 (Hz)
    Returns:
        次峰频率偏移量Δf (Hz)
    """
    # 1. 获取频率轴和时间轴
    freqs = qtransform.frequencies.value
    times = qtransform.times.value
    
    # 2. 限制搜索范围
    f_mask = (freqs >= f_range[0]) & (freqs <= f_range[1])
    qspec = qtransform.value[:, f_mask]
    freqs_lim = freqs[f_mask]
    
    # 3. 找到主峰（最大功率时间点）
    peak_time_idx = np.argmax(np.sum(qspec, axis=1))
    peak_time = times[peak_time_idx]
    
    # 4. 在主峰后30ms内搜索次峰（典型黑洞合并次峰位置）
    search_window = (times >= peak_time) & (times <= peak_time + 0.03)
    qspec_window = qspec[search_window]
    
    # 5. 找到次峰频率（次大功率频率）
    peak_freq_idx = np.argmax(np.sum(qspec_window, axis=0))
    f_secondary = freqs_lim[peak_freq_idx]
    
    # 6. 主峰频率（典型黑洞合并主峰~100-300Hz）
    f_primary = 150  # 假设主峰150Hz（需根据实际事件调整）
    delta_f = abs(f_secondary - f_primary)
    print(f"次峰频率: {f_secondary:.1f} Hz，主峰频率: {f_primary} Hz，Δf: {delta_f:.1f} Hz")
    return delta_f


# =============================================================
#                3. 黑洞软毛熵计算（S_soft）
# =============================================================

def calculate_soft_hair_entropy(event_info: Dict) -> float:
    """
    根据黑洞参数计算软毛熵（理论模型假设）
    Args:
        event_info: 事件元数据（质量、自旋等）
    Returns:
        软毛熵 S_soft（自然单位）
    """
    # 假设软毛熵与黑洞质量和自旋相关（简化模型）
    # S_soft = k * M^2 * (1 + a^2)，k为常数
    # 实际需根据具体理论调整（如 <searchIndex index="4" ></searchIndex> ）
    M = event_info["mass1"] + event_info["mass2"]  # 总质量 (太阳质量)
    a1 = event_info["spin1"]  # 黑洞1自旋参数（0-1）
    a2 = event_info["spin2"]  # 黑洞2自旋参数
    
    # 假设k=0.1（需标定）
    k = 0.1
    S_soft = k * (M**2) * (1 + 0.5*(a1**2 + a2**2))  # 简化自旋平均
    print(f"总质量: {M:.1f} M☉，平均自旋: {0.5*(a1+a2):.2f}，S_soft: {S_soft:.2f}")
    return S_soft


# =============================================================
#                4. 多事件数据收集与相关性分析
# =============================================================

def collect_events(event_list: List[str], data_dir: str = "data") -> Tuple[List[float], List[float]]:
    """
    收集多个事件的Δf和S_soft数据
    Args:
        event_list: 事件ID列表
        data_dir: 数据保存目录
    Returns:
        (delta_f_list, S_soft_list)
    """
    delta_f_list = []
    S_soft_list = []
    
    for event in event_list:
        print(f"\n处理事件: {event}")
        # 1. 获取数据
        data = fetch_gw_data(event, os.path.join(data_dir, event))
        # 2. 预处理并提取Δf
        qtransform = preprocess_data(data)
        delta_f = find_secondary_peak(qtransform)
        # 3. 获取事件参数并计算S_soft
        event_info = astro.Event(event).query()
        S_soft = calculate_soft_hair_entropy(event_info)
        # 4. 存储结果
        delta_f_list.append(delta_f)
        S_soft_list.append(S_soft)
    
    return delta_f_list, S_soft_list


def analyze_correlation(delta_f: List[float], S_soft: List[float]) -> Tuple[float, float]:
    """
    计算Δf与S_soft的皮尔逊相关系数
    Args:
        delta_f: 次峰频移列表
        S_soft: 软毛熵列表
    Returns:
        (相关系数r, p值)
    """
    r, p = pearsonr(delta_f, S_soft)
    print(f"\nΔf与S_soft相关系数: r = {r:.3f}, p = {p:.4f}")
    if p < 0.05:
        print("结果显著（p<0.05），支持弯曲时间假说")
    else:
        print("结果不显著，需更多数据验证")
    return r, p


# =============================================================
#                       主函数
# =============================================================

def main():
    # 1. 定义待分析事件（示例：O3和O4事件）
    event_list = [
        "GW170817",  # 双中子星并合
        "GW190521",  # 大质量黑洞合并
        "GW200105",  # 中等质量黑洞合并
        "GW231123"   # 2023年最新事件
    ]
    
    # 2. 收集数据（首次运行会下载数据）
    delta_f, S_soft = collect_events(event_list)
    
    # 3. 分析相关性
    analyze_correlation(delta_f, S_soft)
    
    # 4. 可视化结果
    plt.figure(figsize=(8, 5))
    plt.scatter(delta_f, S_soft, alpha=0.7)
    plt.xlabel("Δf (Hz)")
    plt.ylabel("S_soft (自然单位)")
    plt.title("引力波次峰频移与黑洞软毛熵相关性")
    plt.grid(True)
    plt.savefig("correlation_plot.png", dpi=200)
    plt.close()
    print("\n结果已保存到 correlation_plot.png")


if __name__ == "__main__":
    main()