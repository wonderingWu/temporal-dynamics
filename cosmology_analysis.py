# =============================================================
#           Cosmology Module: CMB Data Interface
# =============================================================
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# 尝试导入astropy，如果缺失提供友好错误
try:
    from astropy.io import fits
    ASTROPY_AVAILABLE = True
except ImportError:
    print("警告: astropy模块未安装，将使用模拟数据")
    ASTROPY_AVAILABLE = False


def ensure_dir(path):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

class CMBDataLoader:
    """CMB实验数据加载器（支持Planck/LiteBIRD/CMB-S4模拟）"""
    PLANCK_URL = "https://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE_R3.01.txt"
    SIM_DATA_URLS = {
        "LiteBIRD": "https://cmb-s4.org/files/simulations/LiteBIRD_ns_sim_v1.fits",
        "CMB-S4": "https://cmb-s4.org/files/simulations/CMBS4_ns_sim_v2.fits"
    }
    
    def __init__(self, experiment="Planck"):
        self.experiment = experiment
        self.data = None
        
    def load(self):
        """加载观测/模拟数据"""
        if self.experiment == "Planck":
            self._load_planck()
        else:
            self._load_simulated()
        return self.data
    
    def _load_planck(self):
        """加载Planck真实观测数据"""
        try:
            response = requests.get(self.PLANCK_URL, timeout=10)
            raw_data = response.text.split('\n')
            # 解析Planck数据格式
            data_lines = [line for line in raw_data if not line.startswith('#') and len(line) > 10]
            ell = []
            dl_tt = []
            for line in data_lines:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        ell_val = int(parts[0])
                        dl_val = float(parts[1])
                        ell.append(ell_val)
                        dl_tt.append(dl_val)
                    except ValueError:
                        # 跳过无法转换为数字的行
                        continue
            if not ell or not dl_tt:
                raise ValueError("未找到有效的Planck数据")
            self.data = pd.DataFrame({"ell": ell, "Dl_tt": dl_tt})
            # 添加err列，假设相对误差为4.2%
            self.data["err"] = self.data["Dl_tt"] * 0.042
            
        except Exception as e:
            print(f"Planck数据加载失败: {e}")
            # 生成模拟数据作为后备
            self._generate_sim_data(0.966, 0.0042)
    
    def _load_simulated(self):
        """加载未来实验模拟数据"""
        try:
            url = self.SIM_DATA_URLS.get(self.experiment)
            if not url:
                raise ValueError("未知实验名称")
                
            if ASTROPY_AVAILABLE:
                with fits.open(url) as hdul:
                    data = hdul[1].data
                    self.data = pd.DataFrame({
                        "ell": data["ell"],
                        "Dl_tt": data["Dl_tt"],
                        "err": data["err"]
                    })
            else:
                # 当astropy不可用时，生成模拟数据
                print(f"astropy不可用，为{self.experiment}生成模拟数据")
                self._generate_sim_data(0.965, 0.0018 if self.experiment=="LiteBIRD" else 0.0020)
                
        except Exception as e:
            print(f"{self.experiment}模拟数据加载失败: {e}")
            # 生成高精度模拟数据
            self._generate_sim_data(0.965, 0.0018 if self.experiment=="LiteBIRD" else 0.0020)
    
    def _generate_sim_data(self, ns, err_ns):
        """生成CMB功率谱模拟数据"""
        ell = np.arange(2, 2501, dtype=float)  # 转换为浮点数避免整数负幂问题
        dl_tt = 1e6 * 2 * np.pi * (ell*(ell+1))**-1 * (ell/60)**(ns-1)
        noise = np.random.normal(0, err_ns * dl_tt.mean(), len(ell))
        self.data = pd.DataFrame({
            "ell": ell,
            "Dl_tt": dl_tt + noise,
            "err": np.full(len(ell), err_ns * dl_tt.mean())
        })

class InflationModel:
    """对数时间暴胀模型计算器"""
    def __init__(self, ns_base=0.965, delta_ns=0.003):
        """
        ns_base: 标准暴胀模型谱指数
        delta_ns: 对数时间模型预测的偏移量
        """
        self.ns_base = ns_base
        self.delta_ns = delta_ns
        
    def compute_power_spectrum(self, ell):
        """计算功率谱"""
        # 标准标量功率谱
        ps_base = (ell/60)**(self.ns_base - 1)
        
        # 对数时间修正项 (τ = ln(t/t0))
        # 慢滚参数修正: ε -> ε + δ(ln H)/δτ
        correction = 1 - 0.1 * self.delta_ns * np.log(ell/60)
        
        return ps_base * correction
    
    def compute_ns(self, data, ell_range=(50, 500)):
        """计算有效谱指数
        data: 包含'ell'和'Dl_tt'的数据框
        """
        if data is None or data.empty:
            raise ValueError("没有数据可供计算谱指数")
            
        ell_min, ell_max = ell_range
        mask = (ell_min <= data["ell"]) & (data["ell"] <= ell_max)
        ell_subset = data["ell"][mask]
        ps_subset = data["Dl_tt"][mask]
        
        if len(ell_subset) < 2:
            raise ValueError(f"在ell范围[{ell_min}, {ell_max}]内没有足够的数据点")
        
        # 线性回归拟合 n_s-1 = dlnP/dlnk
        log_ell = np.log(ell_subset)
        log_ps = np.log(ps_subset)
        slope = np.polyfit(log_ell, log_ps, 1)[0]
        
        return slope + 1

# =============================================================
#                   CLI 接口与可视化
# =============================================================
def run_cosmology_analysis():
    parser = argparse.ArgumentParser(description='CMB功率谱分析')
    parser.add_argument('--experiment', choices=['Planck', 'LiteBIRD', 'CMB-S4'], 
                        default='Planck', help='CMB实验数据源')
    parser.add_argument('--delta-ns', type=float, default=0.003,
                        help='对数时间模型预测的ns偏移量')
    parser.add_argument('--out', default='./out/cosmology', help='输出目录')
    args = parser.parse_args()
    
    ensure_dir(args.out)
    
    # 加载数据
    loader = CMBDataLoader(args.experiment)
    cmb_data = loader.load()
    
    # 初始化模型
    model = InflationModel(delta_ns=args.delta_ns)
    
    # 计算模型预测
    cmb_data["model_Dl"] = model.compute_power_spectrum(cmb_data["ell"])
    
    # 计算谱指数
    ns_obs = model.compute_ns(cmb_data)  # 使用观测数据
    ns_model = model.compute_ns(cmb_data[['ell', 'model_Dl']].rename(columns={'model_Dl': 'Dl_tt'}))  # 使用模型预测数据
    
    # 保存结果
    cmb_data.to_csv(os.path.join(args.out, f"cmb_{args.experiment}.csv"), index=False)
    with open(os.path.join(args.out, "ns_results.txt"), "w", encoding="utf-8") as f:
        f.write(f"观测谱指数: {ns_obs:.4f}\n")
        f.write(f"模型预测谱指数: {ns_model:.4f}\n")
        f.write(f"Δn_s: {abs(ns_model - ns_obs):.4f}\n")
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.errorbar(cmb_data["ell"], cmb_data["Dl_tt"], yerr=cmb_data["err"], 
                fmt='o', alpha=0.5, label=f"{args.experiment}观测数据")
    plt.plot(cmb_data["ell"], cmb_data["model_Dl"], 'r-', lw=2, 
             label=f"对数时间模型(Δn_s={args.delta_ns})")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'多极矩$\ell$')
    plt.ylabel(r'$D_\ell^{TT}$ [$\mu K^2$]')
    # 设置中文字体支持
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "sans-serif"]
    plt.title(f'CMB温度功率谱: {args.experiment} vs 对数时间模型')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(args.out, f"cmb_{args.experiment}_comparison.png"), dpi=150)
    plt.close()

if __name__ == "__main__":
    run_cosmology_analysis()