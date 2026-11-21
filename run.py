#!/usr/bin/env python3
# Copyright (c) 2025, Benedikt Hein
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Run a batch of pallet next-best-view episodes in Isaac Sim."""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path

import yaml

from pallet_nbv.assets import validate_assets
from pallet_nbv.episode import run_episode

HERE = Path(__file__).resolve().parent
LOCAL_DATA = Path("/home/assets_and_data")


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def select_scenarios(path, keys, limit):
    with path.open(encoding="utf-8") as stream:
        scenarios = yaml.safe_load(stream)["test_scenarios"]
    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError("test_scenarios must be a nonempty mapping")
    if keys:
        unknown = [key for key in keys if key not in scenarios]
        if unknown:
            raise ValueError(f"Unknown scenario(s): {', '.join(unknown)}")
    selected = [(key, scenarios[key]) for key in (keys or scenarios)][:limit]
    for key, scenario in selected:
        for field in ("robot_position", "robot_yaw", "object_position", "object_yaw"):
            if field not in scenario:
                raise ValueError(f"Scenario {key!r} is missing {field!r}")
        positive_int(scenario.get("max_steps_per_episode", 1800))
    return selected


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=os.getenv(
        "PALLET_NBV_ASSET_ROOT", str(LOCAL_DATA / "isaac_sim_assets/1/Assets/Isaac/4.5/Isaac")))
    parser.add_argument("--model", type=Path, default=os.getenv(
        "PALLET_NBV_MODEL", str(LOCAL_DATA / "yolov11_december/runs/pose/pallet6/weights/best.onnx")))
    parser.add_argument("--scenarios", type=Path, default=HERE / "scenarios.yaml")
    parser.add_argument("--scenario", action="append", help="Run this scenario key; repeat to select multiple.")
    parser.add_argument("--limit", type=positive_int, help="Run only the first N selected scenarios.")
    parser.add_argument("--max-steps", type=positive_int, help="Override the step limit for each selected scenario.")
    parser.add_argument("--output", type=Path, help="New JSON Lines results path (will not overwrite).")
    parser.add_argument("--bbox-confidence", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nbv-mode", choices=("default", "hybrid", "chatgpt"), default="default")
    parser.add_argument("--llm-model", default="gpt-5.1")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True

    try:
        validate_assets(args.asset_root.expanduser().resolve(), args.model.expanduser().resolve())
        scenarios = select_scenarios(args.scenarios, args.scenario, args.limit)
        if not 0.0 <= args.bbox_confidence <= 1.0:
            raise ValueError("--bbox-confidence must be between 0 and 1")
        if args.nbv_mode != "default" and not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required for hybrid and chatgpt modes")
        output = args.output or HERE / "outputs" / datetime.now().strftime("episodes-%Y%m%d-%H%M%S-%f.jsonl")
        if output.exists():
            raise ValueError(f"Output already exists: {output}")
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError, argparse.ArgumentTypeError) as error:
        parser.error(str(error))

    app = AppLauncher(args).app
    env = None
    try:
        from pallet_nbv.controller import Controller
        from pallet_nbv.simulation import PalletEnv, PalletEnvCfg

        selector = None
        if args.nbv_mode != "default":
            from pallet_nbv.llm import LLMSelector

            selector = LLMSelector(args.llm_model)
        cfg = PalletEnvCfg(asset_root=str(args.asset_root.expanduser().resolve()), seed=args.seed)
        cfg.sim.device = args.device
        env = PalletEnv(cfg)
        controller = Controller(env, args.model.expanduser().resolve(), args.bbox_confidence, args.nbv_mode, selector)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            for key, scenario in scenarios:
                max_steps = args.max_steps or int(scenario.get("max_steps_per_episode", 1800))
                result = run_episode(controller, scenario, max_steps)
                print(json.dumps(asdict(result), allow_nan=False), file=stream, flush=True)
                print(f"{key}: {result.steps} steps ({result.termination_reason})", flush=True)
        print(f"Episode results: {output.resolve()}", flush=True)
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            app.close()


if __name__ == "__main__":
    main()