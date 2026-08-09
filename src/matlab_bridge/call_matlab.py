# IMPORTANT REMINDER: make sure that matlab.engine is the correct version matching your MATLAB release

import matlab.engine
from pathlib import Path
import numpy as np
import matlab.engine

eng = matlab.engine.start_matlab()
ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA = ROOT / "data" / "processed"
MATLAB_SCRIPTS_DIR = Path(__file__).resolve().parent / "matlab_scripts"


class MatlabBridge:
    """Wraps a single MATLAB engine process. Start once for the duration
    of the entire session. REMINDER: It may take several seconds for the engine
    to fully load. Use as a context manager so the engine starts cleanly,
    minimizing errors."""

    def __init__(self):
        self.eng = None

    def start(self):
        if self.eng is None:
            print("STARTING MATLAB ENGINE (may take several seconds)")
            self.eng = matlab.engine.start_matlab()
            self.eng.addpath(str(MATLAB_SCRIPTS_DIR), nargout=0)

    def stop(self):
        if self.eng is not None:
            self.eng.quit()
            self.eng = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def bandpower_live(self, signal: np.ndarray, sfreq: float, band=(4, 8)) -> float:
        """Live equivalent of the bandpower() cross-check in compute_stats.m"""
        self.start()
        matlab_signal = matlab.double(signal.tolist())
        matlab_band = matlab.double(list(band))
        result = self.eng.bandpower(matlab_signal, float(sfreq), matlab_band)
        return float(result)

    def anova_live(self, values, groups) -> float:
        """Live equivalent of the anova1() call function in compute_stats.m

        values: 1D array
        groups: 1D array, same length as values, workload level per epoch (1, 2, or 3)

        Returns the p-value"""
        self.start()
        matlab_values = matlab.double(list(values))
        matlab_groups = matlab.double(list(groups))
        # making nargout=3 because anova1 returns 3 items only
        p = self.eng.anova1(matlab_values, matlab_groups, "off", nargout=1)
        return float(p)


if __name__ == "__main__":
    # runs an example
    from scipy.io import loadmat

    crosscheck = loadmat(PROCESSED_DATA / "matlab_check.mat")
    signal = crosscheck["example_signal"].ravel()
    sfreq = float(crosscheck["sfreq"].item())
    python_theta_power = float(crosscheck["python_theta_power"].item())

    with MatlabBridge() as bridge:
        live_power = bridge.bandpower_live(signal, sfreq, band=(4, 8))
        print(f"Python (welch): {python_theta_power:.4e}")
        print(f"MATLAB (live): {live_power:.4e}")

        import pandas as pd

        features = pd.read_csv(PROCESSED_DATA / "theta_power_features.csv")
        p_value = bridge.anova_live(features["theta_power"], features["test"])
        print(f"ANOVA p-value is: {p_value:.4f}")
