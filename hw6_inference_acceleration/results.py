import csv
import json
import math
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SLA_SECONDS = 15
LOAD_SECONDS = 300


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    if rows:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)


def seconds(data, metric, statistic):
    stats = data.get(metric, {})
    value = stats.get(statistic)
    if value is None or not math.isfinite(float(value)):
        return None
    unit = {"s": 1, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "ns": 1e-9}
    return float(value) * unit[stats["unit"]]


def read_diagnostics(profile):
    diagnostics = {}
    try:
        summary = read_json(profile / "profile_export_aiperf.json")
        diagnostics.update(
            ttft_mean_s=seconds(summary, "time_to_first_token", "avg"),
            decode_mean_s=seconds(summary, "decode_duration", "avg"),
            itl_mean_s=seconds(summary, "inter_token_latency", "avg"),
            completion_tokens_avg=summary.get("usage_completion_tokens", {}).get("avg"),
        )
    except (OSError, KeyError, TypeError, ValueError):
        pass

    try:
        endpoints = read_json(profile / "phases/profiling/server_metrics.json")["data"]["endpoint_summaries"]
        totals = {}
        for endpoint in endpoints.values():
            for metric in ("accepted_tokens", "draft_tokens", "drafts"):
                entry = endpoint["metrics"].get(f"vllm:spec_decode_num_{metric}", {})
                totals[metric] = totals.get(metric, 0) + sum(
                    series["stats"]["total"] for series in entry.get("series", ())
                )
        if totals["draft_tokens"] and totals["drafts"]:
            diagnostics.update(
                draft_acceptance_rate=totals["accepted_tokens"] / totals["draft_tokens"],
                mean_acceptance_length=1 + totals["accepted_tokens"] / totals["drafts"],
            )
    except (OSError, KeyError, TypeError, ValueError):
        pass
    return diagnostics


def read_measurement(profile, rpm, dataset_count, load):
    summary = read_json(profile / "profile_export_aiperf.json")
    if summary.get("aiperf_version") != "0.13.0" or summary.get("was_cancelled") is not False:
        raise ValueError("Expected a completed AIPerf 0.13.0 run")

    successes = int(summary.get("request_count", {}).get("avg", 0))
    errors = int(summary.get("error_request_count", {}).get("avg", 0))
    count = successes + errors
    # AIPerf's per-request export includes warmup. Its summary covers profiling only.
    starts = []
    with (profile / "profile_export.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            metadata = json.loads(line)["metadata"]
            if metadata["benchmark_phase"] == "profiling":
                starts.append(metadata["request_start_ns"])
    if count < 2 or len(starts) != count or not all(starts):
        raise ValueError("Request counts or timestamps are incomplete")

    span = (max(starts) - min(starts)) / 1e9
    interval = 60 / rpm
    expected_span = LOAD_SECONDS - interval if load else (dataset_count - 1) * interval
    expected_count = LOAD_SECONDS * rpm / 60 if load else dataset_count
    if load:
        count_ok = abs(count - expected_count) <= max(2, math.ceil(expected_count * 0.01))
    else:
        count_ok = count == dataset_count
    rate_ok = count_ok and abs(span - expected_span) <= max(3, 2 * interval)

    mean = seconds(summary, "request_latency", "avg")
    p90 = seconds(summary, "request_latency", "p90")
    latency_count = summary.get("request_latency", {}).get("count", 0)
    complete_latency = latency_count == successes

    row = {
        "requests_ok": successes,
        "requests_failed": errors,
        "send_span_s": span,
        "observed_rpm": (count - 1) * 60 / span if span > 0 else None,
        "rate_ok": rate_ok,
        "mean_s": mean,
        "p90_s": p90,
        "request_throughput_rps": summary.get("request_throughput", {}).get("avg"),
        "output_tokens_avg": summary.get("output_sequence_length", {}).get("avg"),
        "usable": rate_ok and complete_latency and errors == 0 and bool(mean) and p90 is not None,
    }
    if load:
        # AIPerf adds an infinite latency for each failed request and uses
        # the nearest rank in that distribution for error-adjusted percentiles.
        infinite = bool(errors) and round(0.9 * (count - 1)) >= successes
        sla_p90 = seconds(summary, "adj_request_latency", "p90") if errors else p90
        status = "inconclusive"
        if rate_ok and complete_latency:
            if infinite:
                status = "fail"
            elif sla_p90 is not None:
                status = "pass" if sla_p90 < SLA_SECONDS else "fail"
        row.update(sla_p90_s=sla_p90, sla_p90_infinite=infinite, sla_status=status)
    return row


def recover_measurement(path, row, dataset_count, load):
    status = row["status"]
    parse_failed = status == "inconclusive" and row.get("detail", "").startswith("AIPerf exit code 0:")
    if status != "interrupted" and not parse_failed:
        return

    try:
        metrics = read_measurement(path.parent / "aiperf" / "profile", row["rpm"], dataset_count, load)
    except (OSError, KeyError, TypeError, ValueError):
        return

    new_status = metrics["sla_status"] if load else ("ok" if metrics["usable"] else "inconclusive")
    if new_status == "inconclusive":
        return
    row.update(metrics, status=new_status, recovered_from_status=status)
    row.pop("detail", None)
    write_json(path, row)


def plot_spec(directory, pairs, crossovers):
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    baseline = [row for row in pairs if row["spec_config"] == pairs[0]["spec_config"]]
    ax.plot([row["rpm"] for row in baseline], [row["baseline_p90_s"] for row in baseline],
            "o-", color="0.25", label="Baseline")
    for config, crossover in crossovers.items():
        series = [row for row in pairs if row["spec_config"] == config]
        if not any(row["spec_p90_s"] is not None for row in series):
            continue
        label = series[0]["draft_model"] or config
        slower = crossover["first_slower_rpm"]
        if slower is not None:
            label += f" (slower at {slower} RPM)"
        line, = ax.plot([row["rpm"] for row in series], [row["spec_p90_s"] for row in series],
                        "s-", label=label)
        if slower is not None:
            point = next(row for row in series if row["rpm"] == slower)
            ax.scatter(slower, point["spec_p90_s"], color=line.get_color(), marker="x", s=90, zorder=4)
    ax.set(xlabel="Offered load (RPM)", ylabel="P90 E2E latency (s)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(directory / "latency_vs_rpm.png", dpi=160)
    plt.close(fig)


def plot_load(directory, rows, config, best, failed):
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.axhline(SLA_SECONDS, linestyle="--", color="0.4", label=f"SLA {SLA_SECONDS} s")
    ax.plot([row["rpm"] for row in rows], [row.get("p90_s") for row in rows],
            "o-", color="tab:blue", label="P90 of successful requests")
    for status, color, marker in [("pass", "tab:green", "s"), ("fail", "tab:red", "X")]:
        points = [row for row in rows if row["status"] == status and row.get("sla_p90_s") is not None]
        if points:
            ax.scatter([row["rpm"] for row in points], [row["sla_p90_s"] for row in points],
                       color=color, marker=marker, zorder=3, label=f"SLA P90, {status}")
    for rpm, label in [(best, "Highest passing"), (failed, "Next failing")]:
        if rpm is not None:
            ax.axvline(rpm, linestyle=":", color="0.45", label=f"{label}: {rpm} RPM")
    infinite = [row["rpm"] for row in rows if row["status"] == "fail" and row.get("sla_p90_infinite")]
    if infinite:
        ax.scatter(infinite, [0.96] * len(infinite), transform=ax.get_xaxis_transform(),
                   marker="^", color="tab:red", label="SLA P90 unbounded (errors)")
    ax.set(title=config, xlabel="Offered load (RPM)", ylabel="P90 E2E latency (s)", ylim=(0, None))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(directory / "latency_vs_rpm.png", dpi=160)
    plt.close(fig)