import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

# ========== Wavelet Denoising Function (from your code) ==========
def wavelet_denoise_csi(raw_mat: np.ndarray, level: int = 4, threshold_mode: str = 'soft'):
    T, F = raw_mat.shape
    denoised = np.zeros_like(raw_mat)
    for f in range(F):
        signal = raw_mat[:, f]
        max_level = int(np.log2(len(signal))) - 1
        safe_level = max(1, min(level, max_level))
        try:
            coeffs = haar_wavelet_decompose(signal, safe_level)
        except Exception:
            denoised[:, f] = signal
            continue
        if coeffs is None or len(coeffs) < 2:
            denoised[:, f] = signal
            continue
        cA, *cD_list = coeffs
        if len(cD_list) == 0:
            denoised[:, f] = signal
            continue
        all_cd = np.concatenate(cD_list)
        if all_cd.size == 0:
            denoised[:, f] = signal
            continue
        sigma = np.median(np.abs(all_cd)) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(all_cd)))
        denoised_cD = []
        for cd in cD_list:
            if threshold_mode == 'soft':
                cd_denoised = np.sign(cd) * np.maximum(np.abs(cd) - threshold, 0)
            else:
                cd_denoised = np.where(np.abs(cd) < threshold, 0, cd)
            denoised_cD.append(cd_denoised)
        try:
            recon = haar_wavelet_reconstruct([cA] + denoised_cD)
            denoised[:, f] = recon[:T]
        except Exception:
            denoised[:, f] = signal
    return denoised

# ========== STFT Function (from your code) ==========
def apply_stft(csi: np.ndarray, nperseg=2, noverlap=1, reduce=True):
    T, F = csi.shape
    stft_list = []
    for f in range(F):
        _, _, Zxx = stft(csi[:, f], nperseg=nperseg, noverlap=noverlap)
        Zxx = np.abs(Zxx)
        if reduce:
            Zxx = Zxx[:Zxx.shape[0]//2, :]
        stft_list.append(Zxx)
    stft_stack = np.stack(stft_list, axis=-1)
    stft_stack = stft_stack.transpose(1, 0, 2)
    out = stft_stack.reshape(stft_stack.shape[0], -1)
    return out.astype(np.float32)

# ========== Wavelet Utils ==========
def haar_wavelet_decompose(signal, level):
    import pywt
    return pywt.wavedec(signal, 'haar', level=level)

def haar_wavelet_reconstruct(coeffs):
    import pywt
    return pywt.waverec(coeffs, 'haar')

# ==============================
# Main Plotting
# ==============================
if __name__ == '__main__':

    # Load CSI data
    file_path = "/home/chm/CSI-LSTM-Transformer/data_raw_1/walk/user_1_sample_2_walk_A.csv"
    csi_raw = np.loadtxt(file_path, delimiter=",", dtype=np.float32)

    # Normalization
    csi_raw = (csi_raw - csi_raw.mean()) / (csi_raw.std() + 1e-8)

    # Wavelet denoising
    csi_denoised = wavelet_denoise_csi(csi_raw, level=4, threshold_mode='soft')

    # STFT
    csi_stft = apply_stft(csi_denoised, nperseg=2, noverlap=1)

    # Plot figure
    plt.figure(figsize=(16, 5))

    # 1. Original waveform
    plt.subplot(1, 3, 1)
    plt.plot(csi_raw[:, 0], color='#1f77b4')
    plt.title("Original CSI Waveform", fontsize=12)
    plt.xlabel("Sample Points")
    plt.grid(alpha=0.3)

    # 2. Wavelet denoised waveform
    plt.subplot(1, 3, 2)
    plt.plot(csi_denoised[:, 0], color='#ff6b6b')
    plt.title("Wavelet Denoised Waveform", fontsize=12)
    plt.xlabel("Sample Points")
    plt.grid(alpha=0.3)

    # 3. STFT time-frequency map
    plt.subplot(1, 3, 3)
    plt.imshow(csi_stft.T, aspect='auto', cmap='jet', origin='lower')
    plt.title("STFT Time-Frequency Map", fontsize=12)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("action_preprocessing_comparison.png", dpi=300)
    plt.show()