from pathlib import Path

import cv2
from joblib import Parallel, delayed
from tqdm import tqdm

from trajectory_prediction.camera import IM_H, IM_W, D, K, distort_image


def split_list(input_list, sublist_length):
    return [input_list[i : i + sublist_length] for i in range(0, len(input_list), sublist_length)]


if __name__ == "__main__":
    INPUT_DIR = Path("../../trajectory_optimization/data/fixed_speed_50k/images")
    OUTPUT_DIR = Path("../../trajectory_optimization/data/fixed_speed_50k/processed_images")
    EXT = ".jpg"
    THREADS = 8

    OUTPUT_DIR.mkdir(exist_ok=True)

    # end crop: top left coords, wh
    x, y, w, h = 250, 169, 375, 250

    scaled_K = K / 4  # 3264x2464 -> 816x616
    scaled_K[2][2] = 1

    def process_paths(paths, job_index):
        for i, pth in enumerate(paths):
            # TODO not the best to report progress, can we do smth better?
            if i % 500 == 0:
                print(f"job {job_index}: {i}/{len(paths)}")
            if pth.suffix == EXT:
                out_pth = OUTPUT_DIR / pth.name

                img = cv2.imread(str(pth))
                dist = distort_image(img, scaled_K, D, crop_output=False)
                crop_img = dist[y : y + h, x : x + w]
                cv2.imwrite(str(out_pth), crop_img)

    # split into multiple jobs
    all_paths = [pth for pth in INPUT_DIR.iterdir()]
    job_paths = split_list(all_paths, sublist_length=(len(all_paths) // THREADS))

    Parallel(n_jobs=THREADS)(
        delayed(process_paths)(job_paths[i], i) for i in range(len(job_paths))
    )
