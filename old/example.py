import os
import glob
import cv2
import numpy as np

def main():
    # ————————————————
    # CONFIGURATION
    # ————————————————
    base_dir      = 'dataset3'
    rgb_pattern   = os.path.join(base_dir, 'rgb',   'r-*-*.png')
    depth_pattern = os.path.join(base_dir, 'depth', 'd-*-*.png')

    # Optical‐flow parameters
    fb_params = dict(
        pyr_scale = 0.5,
        levels    = 3,
        winsize   = 15,
        iterations= 3,
        poly_n    = 5,
        poly_sigma= 1.2,
        flags     = 0
    )

    # Depth‐edge thresholds
    depth_canny_thresh1 = 25
    depth_canny_thresh2 = 150

    # Morphology kernel for mask cleanup
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

    # ————————————————
    # PAIR UP RGB & DEPTH FRAMES BY INDEX
    # ————————————————
    def build_dict(pattern):
        d = {}
        for path in glob.glob(pattern):
            name = os.path.basename(path)
            # filename format: prefix-<timestamp>-<index>.png
            idx = int(name.rstrip('.png').split('-')[-1])
            d[idx] = path
        return d

    rgb_dict   = build_dict(rgb_pattern)
    depth_dict = build_dict(depth_pattern)
    common_ids = sorted(set(rgb_dict) & set(depth_dict))
    if not common_ids:
        print("No matching RGB/depth frame pairs found.")
        return

    # ————————————————
    # READ FIRST FRAME
    # ————————————————
    first_id  = common_ids[0]
    prev_rgb  = cv2.imread(rgb_dict[first_id])
    prev_gray = cv2.cvtColor(prev_rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # ————————————————
    # PROCESS EACH FRAME PAIR
    # ————————————————
    for idx in common_ids[1:]:
        rgb = cv2.imread(rgb_dict[idx])
        dep = cv2.imread(depth_dict[idx])

        # convert to grayscale
        gray  = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
        dep_g = cv2.cvtColor(dep, cv2.COLOR_BGR2GRAY)

        # 1) Compute Farneback flow & residual
        flow   = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **fb_params)
        fx, fy = flow[...,0], flow[...,1]
        Ix     = cv2.Sobel(prev_gray, cv2.CV_32F, 1, 0, ksize=3)
        Iy     = cv2.Sobel(prev_gray, cv2.CV_32F, 0, 1, ksize=3)
        It     = gray - prev_gray
        residual = np.abs(Ix*fx + Iy*fy + It)

        # normalize & threshold residual → motion mask
        res_norm, motion_mask = None, None
        res_norm = cv2.normalize(residual, None, 0,255, cv2.NORM_MINMAX).astype(np.uint8)
        _, motion_mask = cv2.threshold(res_norm, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 2) Depth‐edge mask
        depth_mask = cv2.Canny(dep_g, depth_canny_thresh1, depth_canny_thresh2)

        # 3) Fuse masks and clean up
        fused = cv2.bitwise_or(motion_mask, depth_mask)
        fused = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, morph_kernel)
        fused = cv2.morphologyEx(fused, cv2.MORPH_OPEN,  morph_kernel)

        # 4) Save the per‐pixel change mask
        mask_filename = f"mask_{idx:04d}.png"
        cv2.imwrite(mask_filename, fused)
        print(f"Saved mask for frame {idx} → {mask_filename}")

        # 5) (Optional) Overlay fused mask in red on the RGB frame
        overlay = rgb.copy()
        overlay[fused > 0] = (0, 0, 255)
        vis = cv2.addWeighted(overlay, 0.5, rgb, 0.5, 0)

        # Display results
        cv2.imshow('Fused Change Mask', fused)
        cv2.imshow('Overlay on RGB',    vis)
        if cv2.waitKey(1) & 0xFF == 27:  # press ESC to exit early
            break

        # prepare for next iteration
        prev_gray = gray.copy()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
