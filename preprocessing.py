import os
import shutil
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter


def _read_txt(file_path):
    wavelength = []
    flux = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    wavelength.append(float(parts[0]))
                    flux.append(float(parts[1]))
                except ValueError:
                    continue
    return np.array(wavelength), np.array(flux)


def _validate_spectrum(wavelength, flux, start, end):
    if len(wavelength) == 0 or len(flux) == 0:
        return False
    if np.any(np.isnan(wavelength)) or np.any(np.isnan(flux)):
        return False
    if np.any(np.isinf(wavelength)) or np.any(np.isinf(flux)):
        return False
    if np.max(wavelength) < start or np.min(wavelength) > end:
        return False
    return True


def _preprocess_single(wavelength, flux, start, end):

    mask = (wavelength >= start) & (wavelength <= end)
    wave_cropped = wavelength[mask]
    flux_cropped = flux[mask]

    sampling_list = np.arange(start, end + 1)

    interp_func = interp1d(wave_cropped, flux_cropped, kind='linear', fill_value='extrapolate')
    flux_resampled = interp_func(sampling_list)

    flux_denoised = median_filter(flux_resampled, size=9, mode='reflect')

    mean_flux = np.mean(flux_denoised)
    std_flux = np.std(flux_denoised)

    if std_flux != 0:
        flux_normalized = (flux_denoised - mean_flux) / std_flux
    else:
        flux_normalized = np.zeros_like(flux_denoised)

    return sampling_list, flux_normalized


def process_spectrum_with_labels(input_spectrum_folder, input_label_folder,
                                 output_spectrum_folder, output_label_folder,
                                 label_extension=".txt", spectrum_extension=".txt",
                                 start_wave=4050, end_wave=6800):

    os.makedirs(output_spectrum_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)

    # 遍历所有光谱文件
    for spectrum_filename in os.listdir(input_spectrum_folder):
        if not spectrum_filename.endswith(spectrum_extension):
            continue

        spectrum_path = os.path.join(input_spectrum_folder, spectrum_filename)
        print(f"{spectrum_filename}")

        try:
            base_name = os.path.splitext(spectrum_filename)[0]

            wavelength, flux = _read_txt(spectrum_path)
            if not _validate_spectrum(wavelength, flux, start_wave, end_wave):
                print(f"continue")
                continue

            wave_proc, flux_proc = _preprocess_single(wavelength, flux, start_wave, end_wave)

            output_spectrum_file = os.path.join(output_spectrum_folder, f"{base_name}{spectrum_extension}")
            with open(output_spectrum_file, 'w') as f:
                for w, f_val in zip(wave_proc, flux_proc):
                    f.write(f"{w:.2f} {f_val:.6f}\n")

            label_filename = f"{base_name}{label_extension}"
            input_label_path = os.path.join(input_label_folder, label_filename)

            if os.path.exists(input_label_path):
                output_label_path = os.path.join(output_label_folder, label_filename)
                shutil.copy2(input_label_path, output_label_path)  # 保留文件元数据
                print(f"{output_spectrum_file}")
                print(f"{output_label_path}")
            else:
                print(f"{output_spectrum_file}")
                print(f"{label_filename}")

        except Exception as e:
            print(f"{str(e)}")

    print("\n")


if __name__ == "__main__":
    INPUT_SPECTRUM_FOLDER = "E:\WR\datasets_middle/val\spectra"
    INPUT_LABEL_FOLDER = "E:\WR\datasets_middle/val\labels"
    OUTPUT_SPECTRUM_FOLDER = "E:\WR\datasets_processed/val\spectra"
    OUTPUT_LABEL_FOLDER = "E:\WR\datasets_processed/val\labels"

    process_spectrum_with_labels(
        input_spectrum_folder=INPUT_SPECTRUM_FOLDER,
        input_label_folder=INPUT_LABEL_FOLDER,
        output_spectrum_folder=OUTPUT_SPECTRUM_FOLDER,
        output_label_folder=OUTPUT_LABEL_FOLDER,
        label_extension=".txt",
        spectrum_extension=".txt"
    )
