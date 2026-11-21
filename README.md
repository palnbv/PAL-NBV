# PALLET NBV

Run single-robot pallet pose estimation in an Isaac Sim warehouse. An RGB-D camera and an ONNX keypoint model locate the pallet, an octree accumulates target and obstacle observations, and the geometric next-best-view planner uses OMPL to navigate to a clearer view.

This directory runs independently of the original `nbv_pose_estimation` task. It uses the same warehouse scenarios and core geometric algorithms, with one controller and no RL agents, timing reports, plots, or experiment scripts.

## Install on Ubuntu 24.04 with Docker

**Reproduction target:** [Isaac Lab v2.3.0](https://github.com/isaac-sim/IsaacLab/releases/tag/v2.3.0). Its [official Docker configuration](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/docker/.env.base) pins **Isaac Sim 5.1.0**. The project was smoke-tested with Isaac Lab 2.3.0 and a 5.1.0 release-candidate build of Isaac Sim. [Isaac Lab v3.0.0](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0) was the latest upstream release as of November 21, 2026, but is not tested here. Use the pinned tag below, not the default branch.

### 1. Set up the host

Use Ubuntu 24.04 on x86-64 with an RTX-capable NVIDIA GPU. Install a driver that meets the [Isaac Sim 5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html). On a fresh host, start with the Ubuntu-recommended driver, then reboot and check `nvidia-smi`. Upgrade the driver if it does not meet those requirements. The host does not need the CUDA Toolkit or a separate Python environment.

```bash
sudo apt update
sudo apt install -y ca-certificates curl git gnupg ubuntu-drivers-common
sudo ubuntu-drivers install
sudo reboot
```

After reboot, install [Docker Engine and Compose](https://docs.docker.com/engine/install/ubuntu/) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Follow Docker's [post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/) so the Isaac Lab launcher can run Docker without `sudo`.


### 2. Start the Isaac Lab container with your files

Keep a checkout or copy of this project on the host. Obtain the external inputs separately; this repository does not distribute them:

- **Isaac Sim 4.5 USD assets:** Mount the `Isaac` directory containing `Environments/Simple_Warehouse` and `Props/KLT_Bin`. Keep the asset directory tree intact so USD references resolve. The **asset version** is separate from the **simulator version**.
- **Pallet ONNX weights:** Mount the folder containing the compatible ONNX export and any companion weight files. The export takes RGB NCHW at 640×640 and returns detection rows with a pixel-space bounding box, score, and eight ordered keypoints with confidences. A generic YOLO pose model is not a substitute.

The upstream container sets `ACCEPT_EULA=Y`. Review the [NVIDIA Isaac Sim license](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/common/NVIDIA_Omniverse_License_Agreement.html) before building or running the image.

```bash
git clone --depth 1 --branch v2.3.0 https://github.com/isaac-sim/IsaacLab.git "$HOME/IsaacLab"
cd "$HOME/IsaacLab"
```

In the Isaac Lab checkout, save this as `docker/pallet-nbv.yaml`. Replace the paths **before the colons** with your absolute host paths. The model and assets are mounted read-only; the project is writable so results appear on the host.

```yaml
services:
  isaac-lab-base:
    volumes:
      - "/home/you/PALLET_NBV:/pallet-nbv"
      - "/home/you/isaac-assets/Assets/Isaac/4.5/Isaac:/assets/Isaac:ro"
      - "/home/you/pallet-model:/weights:ro"
```

From the Isaac Lab checkout on the host, start and enter the container. You can decline X11 forwarding for a headless run.

```bash
./docker/container.py start --files pallet-nbv.yaml
./docker/container.py enter
```

### 3. Run inside the container

```bash
cd /pallet-nbv
export PALLET_NBV_ASSET_ROOT=/assets/Isaac
export PALLET_NBV_MODEL=/weights/best.onnx  # Change if your export has a different name.
/isaac-sim/python.sh -m pip install -r requirements.txt
/isaac-sim/python.sh run.py --headless --limit 1 --max-steps 18
```

The smoke run writes one record to the project's `outputs/` directory on the host. `step_limit` is normal for an 18-step smoke test. Remove both `--limit` and `--max-steps` for the full batch, or use `--scenario KEY` (repeatable) to choose specific cases. Exit the container shell with `exit`, then run `./docker/container.py stop --files pallet-nbv.yaml` from the Isaac Lab checkout to stop the container.

**Already have Isaac Lab?** From the project directory, use its Python interpreter to install the requirements and run `python run.py --headless --limit 1 --asset-root /path/to/Isaac --model /path/to/best.onnx`. Inference uses the CPU provider; Isaac Sim rendering still uses the GPU.

## Results

The runner writes one JSON object per completed episode, in selected scenario order, to a new JSON Lines file under `outputs/` by default. It flushes each record immediately. Each record contains exactly four fields:

| Field | Meaning |
| --- | --- |
| `initial_pose_estimation` | Estimate from the first settled RGB frame. |
| `final_pose_estimation` | Fresh estimate from the actual terminal frame on every end path, including failure. |
| `steps` | Number of actions applied with `env.step()`, excluding setup and settling. |
| `termination_reason` | `goal_reached`, `collision`, `step_limit`, or `stalled:<reason>`. |

Pose snapshots contain success, bounding-box confidence, the 4×4 `camera_from_pallet` transform in camera optical coordinates, and translation and symmetry-aware rotation errors against simulation ground truth. `pnp_reprojection_error_px` describes the PnP fit before its rotation is constrained to the floor. A failed pose has `success: false` and null transform/error fields; the final snapshot never substitutes an earlier, better estimate. `goal_reached` means the robot reached the viewing pose, not that PnP necessarily succeeded.

## Optional LLM selection

The default `--nbv-mode default` never imports OpenAI or sends network requests to it. `--nbv-mode hybrid` requests an LLM candidate if geometric navigation stalls; `--nbv-mode chatgpt` requests a candidate instead of using geometric viewpoint selection. Both require `requirements-llm.txt` and an `OPENAI_API_KEY` environment variable set outside the project. These modes send the current RGB-D image and map to the configured OpenAI model. The runner uses only a locally generated candidate, not coordinates from the response. LLM prompting was not exercised in the first version.

In `chatgpt` mode, the console prints the feedback cycle duration in seconds, including input preparation, the API call (if sent), and response handling. It excludes route planning and is not written to the episode results.

## Development

```sh
python -m unittest discover -s tests -v
```

The lifecycle tests do not launch Isaac Sim. Main modules: `simulation` owns the scene and sensors, `controller` connects perception and navigation, `navigation` owns path state, `episode` defines the result, and `llm` is opt-in. Supporting modules preserve the geometry, occupancy mapping, and PnP algorithms from the original task.

Derived from the original Isaac Lab pallet NBV task. See the included BSD-3-Clause license and retained source notices.
