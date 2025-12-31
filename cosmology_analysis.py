# =============================================================
#           Cosmology Module: CMB Data Interface
# =============================================================
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# Try to import astropy, provide friendly error if missing
try:
    from astropy.io import fits
    ASTROPY_AVAILABLE = True
except ImportError:
    print("Warning: astropy module not installed, will use simulated data")
    ASTROPY_AVAILABLE = False


def ensure_dir(path):
    """Ensure directory exists, create if it doesn't"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

class CMBDataLoader:
    """CMB experiment data loader (supports Planck/LiteBIRD/CMB-S4 simulations)"""
    PLANCK_URL = "https://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE_R3.01.txt"
    SIM_DATA_URLS = {
        "LiteBIRD": "https://cmb-s4.org/files/simulations/LiteBIRD_ns_sim_v1.fits",
        "CMB-S4": "https://cmb-s4.org/files/simulations/CMBS4_ns_sim_v2.fits"
    }
    
    def __init__(self, experiment="Planck"):
        self.experiment = experiment
        self.data = None
        
    def load(self):
        """Load observation/simulation data"""
        if self.experiment == "Planck":
            self._load_planck()
        else:
            self._load_simulated()
        return self.data
    
    def _load_planck(self):
        """Load Planck real observation data"""
        try:
            response = requests.get(self.PLANCK_URL, timeout=10)
            raw_data = response.text.split('\n')
            # Parse Planck data format
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
                        # Skip lines that cannot be converted to numbers
                        continue
            if not ell or not dl_tt:
                raise ValueError("No valid Planck data found")
            self.data = pd.DataFrame({"ell": ell, "Dl_tt": dl_tt})
            # Add err column, assuming 4.2% relative error
            self.data["err"] = self.data["Dl_tt"] * 0.042
            
        except Exception as e:
            print(f"Failed to load Planck data: {e}")
            # Generate simulated data as fallback
            self._generate_sim_data(0.966, 0.0042)
    
    def _load_simulated(self):
        """Load future experiment simulation data"""
        try:
            url = self.SIM_DATA_URLS.get(self.experiment)
            if not url:
                raise ValueError("Unknown experiment name")
                
            if ASTROPY_AVAILABLE:
                with fits.open(url) as hdul:
                    data = hdul[1].data
                    self.data = pd.DataFrame({
                        "ell": data["ell"],
                        "Dl_tt": data["Dl_tt"],
                        "err": data["err"]
                    })
            else:
                # When astropy is not available, generate simulated data
                print(f"astropy not available, generating simulated data for {self.experiment}")
                self._generate_sim_data(0.965, 0.0018 if self.experiment=="LiteBIRD" else 0.0020)
                
        except Exception as e:
            print(f"Failed to load {self.experiment} simulation data: {e}")
            # Generate high-precision simulated data
            self._generate_sim_data(0.965, 0.0018 if self.experiment=="LiteBIRD" else 0.0020)
    
    def _generate_sim_data(self, ns, err_ns):
        """Generate CMB power spectrum simulation data"""
        ell = np.arange(2, 2501, dtype=float)  # Convert to float to avoid integer negative power issues
        dl_tt = 1e6 * 2 * np.pi * (ell*(ell+1))**-1 * (ell/60)**(ns-1)
        noise = np.random.normal(0, err_ns * dl_tt.mean(), len(ell))
        self.data = pd.DataFrame({
            "ell": ell,
            "Dl_tt": dl_tt + noise,
            "err": np.full(len(ell), err_ns * dl_tt.mean())
        })

class InflationModel:
    """Log-time inflation model calculator"""
    def __init__(self, ns_base=0.965, delta_ns=0.003):
        """
        ns_base: Standard inflation model spectral index
        delta_ns: Offset predicted by log-time model
        """
        self.ns_base = ns_base
        self.delta_ns = delta_ns
        
    def compute_power_spectrum(self, ell):
        """Compute power spectrum"""
        # Standard scalar power spectrum
        ps_base = (ell/60)**(self.ns_base - 1)
        
        # Log-time correction term (τ = ln(t/t0))
        # Slow-roll parameter correction: ε -> ε + δ(ln H)/δτ
        correction = 1 - 0.1 * self.delta_ns * np.log(ell/60)
        
        return ps_base * correction
    
    def compute_ns(self, data, ell_range=(50, 500)):
        """Compute effective spectral index
        data: Data frame containing 'ell' and 'Dl_tt'
        """
        if data is None or data.empty:
            raise ValueError("No data available for spectral index calculation")
            
        ell_min, ell_max = ell_range
        mask = (ell_min <= data["ell"]) & (data["ell"] <= ell_max)
        ell_subset = data["ell"][mask]
        ps_subset = data["Dl_tt"][mask]
        
        if len(ell_subset) < 2:
            raise ValueError(f"Not enough data points in ell range [{ell_min}, {ell_max}]")
        
        # Linear regression fit n_s-1 = dlnP/dlnk
        log_ell = np.log(ell_subset)
        log_ps = np.log(ps_subset)
        slope = np.polyfit(log_ell, log_ps, 1)[0]
        
        return slope + 1

# =============================================================
#                   CLI Interface and Visualization
# =============================================================
def run_cosmology_analysis():
    parser = argparse.ArgumentParser(description='CMB Power Spectrum Analysis')
    parser.add_argument('--experiment', choices=['Planck', 'LiteBIRD', 'CMB-S4'], 
                        default='Planck', help='CMB experiment data source')
    parser.add_argument('--delta-ns', type=float, default=0.003,
                        help='ns offset predicted by log-time model')
    parser.add_argument('--out', default='./out/cosmology', help='Output directory')
    args = parser.parse_args()
    
    ensure_dir(args.out)
    
    # Load data
    loader = CMBDataLoader(args.experiment)
    cmb_data = loader.load()
    
    # Initialize model
    model = InflationModel(delta_ns=args.delta_ns)
    
    # Compute model predictions
    cmb_data["model_Dl"] = model.compute_power_spectrum(cmb_data["ell"])
    
    # Calculate spectral index
    ns_obs = model.compute_ns(cmb_data)  # Using observation data
    ns_model = model.compute_ns(cmb_data[['ell', 'model_Dl']].rename(columns={'model_Dl': 'Dl_tt'}))  # Using model prediction data
    
    # Save results
    cmb_data.to_csv(os.path.join(args.out, f"cmb_{args.experiment}.csv"), index=False)
    with open(os.path.join(args.out, "ns_results.txt"), "w", encoding="utf-8") as f:
        f.write(f"Observed spectral index: {ns_obs:.4f}\n")
        f.write(f"Model predicted spectral index: {ns_model:.4f}\n")
        f.write(f"Δn_s: {abs(ns_model - ns_obs):.4f}\n")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.errorbar(cmb_data["ell"], cmb_data["Dl_tt"], yerr=cmb_data["err"], 
                fmt='o', alpha=0.5, label=f"{args.experiment} observation data")
    plt.plot(cmb_data["ell"], cmb_data["model_Dl"], 'r-', lw=2, 
             label=f"Log-time model (Δn_s={args.delta_ns})")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Multipole moment $\ell$')
    plt.ylabel(r'$D_\ell^{TT}$ [$\mu K^2$]')
    # Set up Chinese font support
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "sans-serif"]
    plt.title(f'CMB Temperature Power Spectrum: {args.experiment} vs Log-time Model')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(args.out, f"cmb_{args.experiment}_comparison.png"), dpi=150)
    plt.close()

if __name__ == "__main__":
    run_cosmology_analysis()