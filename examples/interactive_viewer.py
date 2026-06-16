"""Interactive single-camera viewer.

WASD/Space/Shift to translate, Q/E to roll, arrow keys to pitch/yaw, ESC to quit.

Run with the package installed (``pip install -e .[examples]``):
    python examples/interactive_viewer.py
"""
import cv2
import torch

from gaterenderer import GateRenderer, matrix_from_quat, quat_from_euler_xyz
from gaterenderer.sample_scene import CALIB_RES, SAMPLE_D, SAMPLE_K, SAMPLE_GATE_CONFIG, scale_intrinsics


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    render_res = (CALIB_RES[0]*4, CALIB_RES[1]*4)  # Render at 4x calibration resolution

    renderer = GateRenderer(
        gate_config=SAMPLE_GATE_CONFIG,
        n_cams=1,
        K=scale_intrinsics(SAMPLE_K, CALIB_RES, render_res),
        D=SAMPLE_D,
        resolution=render_res,
        device=device,
        verbose=True,
    )

    cam_pos = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float32, device=device)
    view = torch.eye(4, dtype=torch.float32, device=device)
    roll = torch.tensor(0.0, device=device)
    pitch = torch.tensor(0.0, device=device)
    yaw = torch.tensor(0.0, device=device)

    print("#"*80)
    print("Press ESC to exit. Use WASD to move, Q/E to roll, Arrow keys to pitch/yaw.")
    print("#"*80)

    while True:
        key = cv2.waitKey(0)
        norm = torch.norm(view[:2, 0]) + 1e-6
        if   key == ord('w'): cam_pos[:2] += view[:2, 0] * 0.1 / norm  # Forward
        elif key == ord('s'): cam_pos[:2] -= view[:2, 0] * 0.1 / norm  # Backward
        elif key == ord('a'): cam_pos[:2] += view[:2, 1] * 0.1 / norm  # Left
        elif key == ord('d'): cam_pos[:2] -= view[:2, 1] * 0.1 / norm  # Right
        elif key == ord(' '): cam_pos[2] += 0.1  # Up
        elif key == 225:      cam_pos[2] -= 0.1  # Down
        elif key == ord('q'): roll -= 5.0   # Roll left
        elif key == ord('e'): roll += 5.0   # Roll right
        elif key == 81:       yaw += 5.0    # Left arrow
        elif key == 83:       yaw -= 5.0    # Right arrow
        elif key == 82:       pitch += 5.0  # Up arrow
        elif key == 84:       pitch -= 5.0  # Down arrow
        elif key == 27:       break         # ESC

        rot = matrix_from_quat(quat_from_euler_xyz(roll * torch.pi / 180, pitch * torch.pi / 180, yaw * torch.pi / 180))
        view[:3, :3] = rot
        view[:3, 3] = cam_pos

        segment_map = renderer.render(view.unsqueeze(0))
        segment_map_np = segment_map[0].cpu().numpy() * 255
        segment_map_np = cv2.resize(segment_map_np, (CALIB_RES[1], CALIB_RES[0]), interpolation=cv2.INTER_AREA).astype('uint8')

        cv2.imshow("Segment Map", segment_map_np)
        cv2.waitKey(1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
