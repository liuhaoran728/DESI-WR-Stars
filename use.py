import os
import pickle
import numpy as np
import torch
import csv
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from baseline import MODEL_REGISTRY

def _validate_spectrum(wavelength, flux, start, end):
    return not (np.max(wavelength) < start or np.min(wavelength) > end or
                np.any(np.isnan(wavelength)) or np.any(np.isnan(flux)) or
                np.any(np.isinf(wavelength)) or np.any(np.isinf(flux)))


def _crop_spectrum(wavelength, flux, start, end):
    mask = (wavelength >= start) & (wavelength <= end)
    return wavelength[mask], flux[mask]


def _resample_spectrum(wavelength, flux, start, end):
    target_wl = np.arange(start, end + 1)
    if len(wavelength) < 2:
        return None, None
    try:
        func = interp1d(wavelength, flux, kind='linear', fill_value='extrapolate')
        return target_wl, func(target_wl)
    except Exception:
        return None, None


def _denoise_spectrum(flux):
    return median_filter(flux, size=9, mode='reflect')


def _normalize_spectrum(flux):
    mean_val = np.mean(flux)
    std_val = np.std(flux)
    if std_val < 1e-8:
        return np.zeros_like(flux)
    return (flux - mean_val) / std_val


def preprocess_spectrum(wavelength, flux, start_wave=4050, end_wave=6800):
    if not _validate_spectrum(wavelength, flux, start_wave, end_wave):
        raise ValueError

    wl_c, fl_c = _crop_spectrum(wavelength, flux, start_wave, end_wave)
    if len(wl_c) == 0:
        raise ValueError

    _, fl_r = _resample_spectrum(wl_c, fl_c, start_wave, end_wave)
    if fl_r is None:
        raise ValueError

    fl_d = _denoise_spectrum(fl_r)
    fl_n = _normalize_spectrum(fl_d)
    return fl_n

def extract_full_sample_data(raw_data, sample_index):
    try:
        item = raw_data[sample_index]
        ra = item[0] if len(item) > 0 else None
        dec = item[1] if len(item) > 1 else None
        path = item[2] if len(item) > 2 else None
        fiber_index = item[3] if len(item) > 3 else None
        wavelength = np.asarray(item[-2])
        flux = np.asarray(item[-1])
        return [ra, dec, path, fiber_index, wavelength, flux]
    except Exception:
        return None


def load_pkl_for_preprocessing(file_path):
    print(f"load {os.path.basename(file_path)}")
    try:
        with open(file_path, 'rb') as f:
            raw_data = pickle.load(f)

        wls, fls = [], []
        for item in raw_data:
            if len(item) >= 2:
                wls.append(item[-2])
                fls.append(item[-1])
        print(f"samples {len(wls)}")
        return wls, fls, raw_data
    except Exception as e:
        print(f"stop{e}")
        return None, None, None


def load_model_for_inference(model_path, device, model_name_override=None):
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"stop{e}")

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            num_classes = int(checkpoint.get('num_classes', 2))
            spectrum_length = int(checkpoint.get('spectrum_length', 2751))
            saved_model_name = checkpoint.get('model_name', 'unknown').lower()
            print(f"metadata: "
                  f"num_classes={num_classes}, length={spectrum_length}, model='{saved_model_name}'")
        else:
            state_dict = checkpoint
            num_classes = 2
            spectrum_length = 2751
            saved_model_name = "unknown"
    else:
        raise ValueError

    if 'conv1.weight' in state_dict:
        in_channel = state_dict['conv1.weight'].shape[1]
    else:
        in_channel = 1
        print("in_channel=1")

    target_model_name = (model_name_override or saved_model_name).lower().strip()

    try:
        from baseline import get_model_class
        ModelClass = get_model_class(target_model_name)
    except Exception as e:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError

    try:
        model = ModelClass(
            in_channels=in_channel,
            num_classes=num_classes,
            spec_length=spectrum_length,
            output_prob=False
        )
    except TypeError:
        try:
            model = ModelClass(
                in_channel=in_channel,
                out_channel=num_classes,
                spectrum_size=spectrum_length,
                output_prob=False
            )
        except Exception as e:
            raise RuntimeError from e

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"model {target_model_name} | device={device}")
    return model


def process_single_file(file_path, model, start_wave=4050, end_wave=6800,
                        class_threshold=0.9, device='cpu', chunk_size=2000):
    print(f"\n: {os.path.basename(file_path)}")
    wls, fls, raw_data = load_pkl_for_preprocessing(file_path)
    if wls is None or len(wls) == 0:
        return []

    total_samples = len(wls)
    print(f"{total_samples}, {chunk_size}")

    high_confidence_results = []
    spectra_processed = 0

    with torch.no_grad():
        for i in range(0, total_samples, chunk_size):
            chunk_wls = wls[i:i + chunk_size]
            chunk_fls = fls[i:i + chunk_size]
            processed_fluxes = []
            valid_indices = []

            for j, (wl, fl) in enumerate(zip(chunk_wls, chunk_fls)):
                try:
                    proc_flux = preprocess_spectrum(wl, fl, start_wave, end_wave)
                    processed_fluxes.append(proc_flux)
                    valid_indices.append(j)
                except Exception:
                    pass
                finally:
                    spectra_processed += 1

            if not processed_fluxes:
                continue

            data_array = np.array(processed_fluxes)
            data_tensor = torch.tensor(data_array[:, np.newaxis, :], dtype=torch.float32).to(device)

            all_probs = []
            batch_size_gpu = 128
            for b in range(0, len(data_tensor), batch_size_gpu):
                batch = data_tensor[b:b + batch_size_gpu]
                logits = model(batch)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu())
            all_probs = torch.cat(all_probs, dim=0).numpy()
            class_1_probs = all_probs[:, 1]

            high_conf_mask = class_1_probs >= class_threshold
            high_conf_local_indices = np.where(high_conf_mask)[0]

            for local_idx in high_conf_local_indices:
                global_idx = i + valid_indices[local_idx]
                sample_data = extract_full_sample_data(raw_data, global_idx)
                if sample_data is not None:
                    prob = class_1_probs[local_idx]
                    current_num = len(high_confidence_results) + 1
                    print(f"[{current_num}] prob={prob:.6f} ({spectra_processed}/{total_samples})")
                    high_confidence_results.append(sample_data)

    print(f"total{total_samples}")
    print(f"{len(high_confidence_results)} (≥{class_threshold})")
    return high_confidence_results

def save_results_append(samples, output_path):
    fieldnames = ["ra", "dec", "path", "fiber_index", "wavelength", "flux"]
    with open(output_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for s in samples:
            ra, dec, path, fiber_idx, wl, fl = s
            writer.writerow({
                "ra": ra or "",
                "dec": dec or "",
                "path": path or "",
                "fiber_index": fiber_idx or "",
                "wavelength": str(wl.tolist()),
                "flux": str(fl.tolist())
            })

def main():
    # 参数配置
    PKL_FOLDER = r'H:\DESI Dataset'
    MODEL_PATH = r"E:\WR\codes\trained_models\model.pth"
    OUTPUT_CSV = r"E:\WR\WR_results.csv"
    START_WAVE = 4050
    END_WAVE = 6800
    CLASS_THRESHOLD = 0.99
    CHUNK_SIZE = 6000

    MODEL_NAME = "LPA_MLK_CNN"

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{device}")

    # 加载模型
    model = load_model_for_inference(
        model_path=MODEL_PATH,
        device=device,
        model_name_override=MODEL_NAME
    )

    pkl_files = [
        os.path.join(PKL_FOLDER, f) for f in os.listdir(PKL_FOLDER)
        if f.lower().endswith('.pkl') and os.path.isfile(os.path.join(PKL_FOLDER, f))
    ]


    print(f"{len(pkl_files)}")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    fieldnames = ["ra", "dec", "path", "fiber_index", "wavelength", "flux"]
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    print(f"{OUTPUT_CSV}")

    total_high_confidence = 0
    for i, file in enumerate(pkl_files, 1):
        print(f"\n{i}/{len(pkl_files)}")
        results = process_single_file(
            file_path=file,
            model=model,
            start_wave=START_WAVE,
            end_wave=END_WAVE,
            class_threshold=CLASS_THRESHOLD,
            device=device,
            chunk_size=CHUNK_SIZE
        )

        if results:
            current_count = len(results)
            total_high_confidence += current_count
            save_results_append(results, OUTPUT_CSV)
        else:
            print(f"no results for {file}")



if __name__ == "__main__":
    main()