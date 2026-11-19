import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from results import (
    plot_load, plot_spec, read_diagnostics, read_json, read_measurement,
    recover_measurement, write_csv, write_json,
)


PROJECT = Path(__file__).resolve().parent
DATASET = PROJECT / "data/requests.jsonl"
BASELINE = "baseline"
TUNE_RPM = 120
SPEC_RPM = [60, 120, 240, 480, 960]
SERVER_TIMEOUT = 1800

CONFIGS = {
    BASELINE: [],
    "seq64": ["--max-num-seqs", "64"],
    "seq512": ["--max-num-seqs", "512"],
    "seq1024": ["--max-num-seqs", "1024"],
    "seq2048": ["--max-num-seqs", "2048"],
    "prefill256": ["--max-num-batched-tokens", "256"],
    "prefill512": ["--max-num-batched-tokens", "512"],
    "prefill1024": ["--max-num-batched-tokens", "1024"],
    "prefill4096": ["--max-num-batched-tokens", "4096"],
    "flashinfer": ["--attention-backend", "FLASHINFER"],
    "sglang": ["--engine", "sglang", "--dtype", "bfloat16", "--enable-metrics"],
}
NGRAM_CONFIGS = ("ngram2", "ngram4")
for tokens, name in zip((2, 4), NGRAM_CONFIGS):
    CONFIGS[name] = [
        "--speculative-config",
        json.dumps({
            "method": "ngram",
            "num_speculative_tokens": tokens,
            "prompt_lookup_min": 2,
            "prompt_lookup_max": 5,
        }),
    ]

DRAFT_TOKENS = 4
DRAFT_MODELS = {
    "draft_qwen3_06b": "Qwen/Qwen3-0.6B",
    "draft_qwen3_17b": "Qwen/Qwen3-1.7B",
    "draft_qwen35_08b": "Qwen/Qwen3.5-0.8B",
}
for name, model in DRAFT_MODELS.items():
    CONFIGS[name] = [
        "--speculative-config",
        json.dumps({
            "method": "draft_model",
            "model": model,
            "num_speculative_tokens": DRAFT_TOKENS,
        }),
    ]

INCOMPATIBLE = {
    "draft_qwen35_08b": "vLLM 0.22.0 detects the draft as MTP and rejects the Qwen3 target (Invalid model_type qwen3).",
}
EAGLE_MODEL = "RedHatAI/Qwen3-8B-speculator.eagle3"
CONFIGS["eagle3"] = [
    "--speculative-config",
    json.dumps({"method": "eagle3", "model": EAGLE_MODEL, "num_speculative_tokens": 3}),
]
CONFIGS["suffix"] = [
    "--speculative-config",
    json.dumps({
        "method": "suffix",
        "num_speculative_tokens": 16,
        "suffix_decoding_max_cached_requests": 0,
    }),
]
SPEC_MODELS = {**DRAFT_MODELS, "eagle3": EAGLE_MODEL, "suffix": None}
EXTRAS = Path("/opt/hw-inference-extras")
OPTIONAL_DEPENDENCIES = {
    "sglang": EXTRAS / "sglang/bin/sglang",
    "suffix": EXTRAS / "arctic/arctic_inference/suffix_decoding/__init__.py",
}


def missing_dependency(config):
    dependency = OPTIONAL_DEPENDENCIES.get(config)
    if dependency and not dependency.is_file():
        return f"{config} requires {dependency}. See the optional setup in README.md."
    return None


def timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def cached(row, stage):
    if row is None:
        return False
    if row["status"] in {"ok", "pass", "fail"}:
        return True
    # Retain completed overload runs for the spec sweep. Their successful-only
    # latency must stay off the comparison curve.
    return (
        stage == "spec"
        and row["status"] == "inconclusive"
        and not row.get("detail")
        and row.get("rate_ok")
        and bool(row.get("requests_failed"))
    )


class Experiment:
    def __init__(self, directory, port):
        self.directory = directory
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        dataset = DATASET.read_bytes()
        self.dataset_count = len(dataset.splitlines())
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
        ).splitlines()
        environment = {
            "dataset_sha256": hashlib.sha256(dataset).hexdigest(),
            "dataset_count": self.dataset_count,
            "gpus": gpu,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "configs": CONFIGS,
        }
        directory.mkdir(parents=True, exist_ok=True)
        info = directory / "experiment.json"
        if info.exists():
            previous = read_json(info)
            previous_configs = previous.pop("configs")
            same_environment = all(previous.get(key) == value for key, value in environment.items()
                                   if key != "configs")
            same_existing_configs = all(previous_configs.get(name, args) == args for name, args in CONFIGS.items())
            if not same_environment or not same_existing_configs:
                raise ValueError(f"Dataset, GPU or configuration changed. Use a new results directory: {directory}")
            environment["configs"] = {**previous_configs, **CONFIGS}
        write_json(info, environment)

    def latest(self, stage, config):
        rows = {}
        attempts = self.directory / stage / "runs" / config
        for result in sorted(attempts.glob("*/*/result.json")):
            row = read_json(result)
            recover_measurement(result, row, self.dataset_count, load=stage == "load")
            rpm = row["rpm"]
            if cached(row, stage) or not cached(rows.get(rpm), stage):
                if rows.get(rpm, {}).get("status") == "ok" and row["status"] != "ok":
                    continue
                rows[rpm] = row
        if stage == "spec":
            # Count-mode AIPerf runs use the same protocol in tune and spec.
            tune_row = self.latest("tune", config).get(TUNE_RPM)
            if not cached(rows.get(TUNE_RPM), stage) and cached(tune_row, "tune"):
                rows[TUNE_RPM] = tune_row
        if stage == "tune":
            for row in rows.values():
                profile = self.directory / row["run"] / "aiperf" / "profile"
                row.update(read_diagnostics(profile))
        return rows

    @contextmanager
    def server(self, stage, config):
        if detail := missing_dependency(config):
            raise RuntimeError(detail)
        logs = self.directory / stage / "servers" / config / timestamp()
        logs.mkdir(parents=True)
        log_path = logs / "server.log"
        command = ["bash", "serve.sh", *CONFIGS[config], "--port", str(self.port)]
        (logs / "command.txt").write_text(shlex.join(command) + "\n")
        with log_path.open("w") as log:
            probe = subprocess.run(
                ["curl", "--silent", "--max-time", "1", "--output", "/dev/null", self.url + "/health"],
                check=False,
            )
            if probe.returncode == 0:
                raise RuntimeError(f"Port {self.port} already has an HTTP server. Choose another --port.")

            print(f"Starting {config}. Log {log_path}", flush=True)
            process = subprocess.Popen(
                command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            try:
                deadline = time.monotonic() + SERVER_TIMEOUT
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(f"{config} server exited with code {process.returncode}. See {log_path}")
                    try:
                        with urlopen(self.url + "/health", timeout=2) as response:
                            if response.status == 200:
                                break
                    except (URLError, TimeoutError):
                        pass
                    time.sleep(2)
                else:
                    raise RuntimeError(f"{config} server was not ready after {SERVER_TIMEOUT} s. See {log_path}")
                yield log_path
            finally:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()

    def measure(self, stage, config, rpm, server_log):
        run = self.directory / stage / "runs" / config / f"{rpm}rpm" / timestamp()
        run.mkdir(parents=True)
        output = run / "aiperf"
        load = stage == "load"
        command = ["bash", "load_test.sh", str(rpm), str(output)] if load else [
            "bash", "benchmark.sh", str(output), str(rpm),
        ]
        row = {
            "config": config,
            "rpm": rpm,
            "run": str(run.relative_to(self.directory)),
            "server_log": str(server_log.relative_to(self.directory)),
            "status": "interrupted",
        }
        write_json(run / "result.json", row)
        print(f"Running {stage}: {config} at {rpm} RPM. Output {run}", flush=True)
        reset_path = "/flush_cache" if config == "sglang" else "/reset_prefix_cache"
        result = subprocess.run(
            command, cwd=PROJECT, env={**os.environ, "URL": self.url, "RESET_CACHE_PATH": reset_path},
            check=False,
        )
        try:
            row.update(read_measurement(output / "profile", rpm, self.dataset_count, load))
            row["status"] = row["sla_status"] if load else ("ok" if row["usable"] else "inconclusive")
            if result.returncode:
                row.update(status="inconclusive", detail=f"AIPerf exit code {result.returncode}")
        except (OSError, KeyError, TypeError, ValueError) as error:
            row.update(status="inconclusive", detail=f"AIPerf exit code {result.returncode}: {error}")
        write_json(run / "result.json", row)
        print(f"{config}, {rpm} RPM: {row['status']}, P90 {row.get('p90_s')} s", flush=True)
        return row

    def measure_rates(self, stage, config, rates):
        history = self.latest(stage, config)
        pending = []
        for rpm in sorted(set(rates)):
            row = history.get(rpm)
            if cached(row, stage):
                print(f"Reusing {stage}: {config} at {rpm} RPM ({row['run']})", flush=True)
            else:
                pending.append(rpm)
        if pending:
            with self.server(stage, config) as log:
                for rpm in pending:
                    self.measure(stage, config, rpm, log)


def winner(rows, configs):
    candidates = [row for row in rows if row["config"] in configs and row["status"] == "ok"]
    if not candidates:
        return None
    return min(candidates, key=lambda row: (row["mean_s"], row["p90_s"]))["config"]


def tune(experiment):
    directory = experiment.directory / "tune"
    failed_servers = {}
    for config in CONFIGS:
        if config in INCOMPATIBLE:
            continue
        if detail := missing_dependency(config):
            if not cached(experiment.latest("tune", config).get(TUNE_RPM), "tune"):
                failed_servers[config] = {"status": "unavailable", "detail": detail}
            continue
        try:
            experiment.measure_rates("tune", config, [TUNE_RPM])
        except RuntimeError as error:
            failed_servers[config] = {"status": "server_error", "detail": str(error)}
            print(error, file=sys.stderr)
            if config == BASELINE:
                break

    rows = []
    for config in CONFIGS:
        row = experiment.latest("tune", config).get(TUNE_RPM, {
            "config": config, "rpm": TUNE_RPM, "status": "pending",
        })
        row.update(failed_servers.get(config, {}))
        if config in INCOMPATIBLE:
            row.update(status="incompatible", detail=INCOMPATIBLE[config])
        row["draft_model"] = SPEC_MODELS.get(config)
        rows.append(row)
    baseline_mean = rows[0].get("mean_s") if rows[0]["status"] == "ok" else None
    for row in rows:
        row["speedup"] = baseline_mean / row["mean_s"] if baseline_mean and row["status"] == "ok" else None
    write_csv(directory / "comparison.csv", rows)
    if baseline_mean is None:
        raise RuntimeError(f"No valid baseline at {TUNE_RPM} RPM. See {directory / 'comparison.csv'}")
    selection = {
        "configs": list(CONFIGS),
        "best": winner(rows, CONFIGS),
        "spec": winner(rows, NGRAM_CONFIGS),
        "drafts": [row["config"] for row in rows if row["config"] in DRAFT_MODELS and row["status"] == "ok"],
        "spec_configs": [row["config"] for row in rows if row["config"] in SPEC_MODELS and row["status"] == "ok"],
    }
    write_json(directory / "winners.json", selection)
    print(f"Best at {TUNE_RPM} RPM: {selection['best']}. N-gram: {selection['spec']}. "
          f"Valid drafts: {', '.join(selection['drafts'])}", flush=True)


def tune_selection(experiment):
    path = experiment.directory / "tune" / "winners.json"
    if path.is_file():
        selection = read_json(path)
        if selection.get("configs") == list(CONFIGS):
            return selection
    raise RuntimeError("Run the tune stage with the current configuration list first.")


def spec(experiment, rates):
    selection = tune_selection(experiment)
    ngram = selection["spec"]
    configs = ([ngram] if ngram else []) + selection["spec_configs"]
    if not configs:
        raise RuntimeError("No valid speculative decoding result at 120 RPM. See tune/comparison.csv.")
    directory = experiment.directory / "spec"
    existing = [experiment.latest("spec", config) for config in [BASELINE, *configs]]
    rates = sorted(set(rates).union(*existing))
    experiment.measure_rates("spec", BASELINE, rates)
    failed_servers = {}
    for config in configs:
        try:
            experiment.measure_rates("spec", config, rates)
        except RuntimeError as error:
            failed_servers[config] = str(error)
            print(error, file=sys.stderr)
    histories = {config: experiment.latest("spec", config) for config in [BASELINE, *configs]}

    pairs = []
    crossovers = {}
    for config in configs:
        config_pairs = []
        for rpm in rates:
            baseline = histories[BASELINE].get(rpm, {})
            heuristic = histories.get(ngram, {}).get(rpm, {})
            speculative = histories[config].get(rpm, {})
            before = baseline.get("p90_s") if baseline.get("status") == "ok" else None
            after = speculative.get("p90_s") if speculative.get("status") == "ok" else None
            ngram_p90 = heuristic.get("p90_s") if heuristic.get("status") == "ok" else None
            status = speculative.get("status")
            if status == "inconclusive" and cached(speculative, "spec"):
                status = "request_errors"
            elif status is None and config in failed_servers:
                status = "server_error"
            row = {
                "rpm": rpm,
                "spec_config": config,
                "draft_model": SPEC_MODELS.get(config),
                "baseline_p90_s": before,
                "spec_p90_s": after,
                "ngram_p90_s": ngram_p90,
                "spec_minus_baseline_s": after - before if before is not None and after is not None else None,
                "spec_minus_ngram_s": after - ngram_p90 if after is not None and ngram_p90 is not None else None,
                "baseline_status": baseline.get("status"),
                "spec_status": status,
                "spec_requests_failed": speculative.get("requests_failed"),
                "baseline_run": baseline.get("run"),
                "spec_run": speculative.get("run"),
            }
            config_pairs.append(row)
        pairs.extend(config_pairs)
        slower = next((row["rpm"] for row in config_pairs if (row["spec_minus_baseline_s"] or 0) > 0), None)
        faster_below = [
            row["rpm"] for row in config_pairs if slower is not None and row["rpm"] < slower
            and (row["spec_minus_baseline_s"] or 0) < 0
        ]
        crossovers[config] = {
            "first_slower_rpm": slower,
            "last_faster_before_rpm": max(faster_below, default=None),
        }
        print(f"{config}: first measured RPM slower than baseline: {slower}", flush=True)

    write_csv(directory / "comparison.csv", pairs)
    write_json(directory / "crossover.json", {
        "ngram_config": ngram,
        "configs": crossovers,
        "server_errors": failed_servers,
    })
    plot_spec(directory, pairs, crossovers)


def search_capacity(at, start, max_rpm, step, history):
    low = max((rpm for rpm, row in history.items() if rpm <= max_rpm and row["status"] == "pass"), default=None)
    high = min(
        (rpm for rpm, row in history.items() if (low or 0) < rpm <= max_rpm and row["status"] == "fail"),
        default=None,
    )
    if low is None and high is None:
        if at(start) == "pass":
            low = start
        else:
            high = start

    if high is None:
        while low < max_rpm:
            high = min(low * 2, max_rpm)
            if at(high) == "fail":
                break
            low = high
        else:
            return
    elif low is None:
        while high > 1:
            low = max(1, high // 2)
            if at(low) == "pass":
                break
            high = low
        else:
            return
    while high - low > step:
        middle = (low + high) // 2
        if at(middle) == "pass":
            low = middle
        else:
            high = middle


def load(experiment, args):
    config = tune_selection(experiment)["best"]
    directory = experiment.directory / "load"
    history = experiment.latest("load", config)
    try:
        if args.rpm:
            experiment.measure_rates("load", config, args.rpm)
        else:
            with ExitStack() as stack:
                log = None

                def at(rpm):
                    nonlocal log
                    if not cached(history.get(rpm), "load"):
                        if log is None:
                            log = stack.enter_context(experiment.server("load", config))
                        history[rpm] = experiment.measure("load", config, rpm, log)
                    status = history[rpm]["status"]
                    if status not in {"pass", "fail"}:
                        raise RuntimeError(f"Inconclusive run at {rpm} RPM. See {history[rpm]['run']}")
                    return status

                search_capacity(at, args.start, args.max_rpm, args.step, history)
    finally:
        rows = sorted(experiment.latest("load", config).values(), key=lambda row: row["rpm"])
        passing = [row["rpm"] for row in rows if row["status"] == "pass"]
        failing = [row["rpm"] for row in rows if row["status"] == "fail"]
        best = max(passing, default=None)
        failed = min((rpm for rpm in failing if best is None or rpm > best), default=None)
        if rows:
            write_csv(directory / "measurements.csv", rows)
            write_json(directory / "boundary.json", {
                "config": config,
                "highest_passing_rpm": best,
                "next_failing_rpm": failed,
                "bracket_width_rpm": failed - best if best is not None and failed is not None else None,
                "fail_below_highest_pass": best is not None and any(rpm < best for rpm in failing),
            })
            plot_load(directory, rows, config, best, failed)
            print(f"Highest measured passing RPM: {best}. Next measured failing RPM: {failed}", flush=True)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=PROJECT / "results" / "hw")
    parser.add_argument("--port", type=positive_int, default=8000)
    stages = parser.add_subparsers(dest="stage", required=True)
    stages.add_parser("tune")
    spec_parser = stages.add_parser("spec")
    spec_parser.add_argument("--rpm", type=positive_int, nargs="+", default=SPEC_RPM)
    load_parser = stages.add_parser("load")
    load_parser.add_argument("--rpm", type=positive_int, nargs="+")
    load_parser.add_argument("--start", type=positive_int, default=TUNE_RPM)
    load_parser.add_argument("--max-rpm", type=positive_int, default=3840)
    load_parser.add_argument("--step", type=positive_int, default=5)
    args = parser.parse_args()
    if args.port > 65535:
        parser.error("--port must be at most 65535")
    if args.stage == "load" and not args.rpm and args.start > args.max_rpm:
        parser.error("--start must be at most --max-rpm")

    try:
        experiment = Experiment(args.out.resolve(), args.port)
        if args.stage == "tune":
            tune(experiment)
        elif args.stage == "spec":
            spec(experiment, args.rpm)
        else:
            load(experiment, args)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted. Finished measurements are saved.\n")


if __name__ == "__main__":
    main()