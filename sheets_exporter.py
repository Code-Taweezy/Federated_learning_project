"""Google Sheets exporter for FL experiment results.

Setup (one-time):
  1. Go to https://console.cloud.google.com and create a project.
  2. Enable the Google Sheets API and Google Drive API.
  3. Create a Service Account, grant it "Editor" role, and download the JSON key.
  4. Copy the JSON key to this project folder (any name, e.g. service_account.json).
  5. Open your Google Spreadsheet, click Share, and add the service account email
     (found in the JSON key as "client_email") with Editor access.
  6. Copy sheets_config.template.json -> sheets_config.json and fill in:
       - "credentials_file": path to your downloaded JSON key
       - "spreadsheet_id":   the long ID from your sheet's URL
         e.g. https://docs.google.com/spreadsheets/d/<THIS_PART>/edit
  7. pip install gspread google-auth

Once configured, results are exported automatically after every experiment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

# Column header row written once per tab
HEADER = [
    "Timestamp", "Run ID", "Aggregator", "Dataset", "Topology",
    "Num Nodes", "Total Rounds", "Attack Ratio", "Attack Type",
    "Round", "Avg Acc", "Std Acc", "Avg Loss", "Honest Acc", "Compromised Acc",
    "Final Acc", "Final Loss",
]

# Extended header for advanced metrics tab
METRICS_HEADER = [
    "Timestamp", "Run ID", "Aggregator", "Dataset", "Topology",
    "Num Nodes", "Attack Ratio", "Attack Type",
    "Round", "Drift Mean", "Drift Std",
    "Peer Dev Mean", "Consensus Score",
    "Regression Slope", "R²",
    "Detection Flags",
    "TP (cumul)", "FP (cumul)", "TN (cumul)", "FN (cumul)",
    "T_detect", "Time w/o Detection (s)", "Time w/ Detection (s)",
]

# Maps internal dataset names -> display names matching your spreadsheet tabs
_DATASET_LABEL: dict[str, str] = {
    "femnist":     "Femnist",
    "shakespeare": "Shakespeare",
}

# Maps internal attack type names -> display names matching your spreadsheet tabs
_ATTACK_LABEL: dict[str, str] = {
    "directed": "Direct",
    "gaussian": "Gaussian",
    "none":     "None",
}


def tab_name_for(dataset: str, attack_type: str, attack_ratio: float) -> str:
    """Return the spreadsheet tab name for a given dataset / attack combination.

    Tab names follow the pattern "<Dataset> - <AttackType>", e.g.:
      "Femnist - Direct", "Shakespeare - Gaussian", etc.
    """
    ds  = _DATASET_LABEL.get(dataset.lower(), dataset.title())
    atk = _ATTACK_LABEL.get(attack_type.lower(), attack_type.title())
    return f"{ds} - {atk}"


def _load_config(config_path: str = "sheets_config.json") -> Optional[dict]:
    p = Path(config_path)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def export_results(results_path: str, config_path: str = "sheets_config.json") -> bool:
    """Read a JSON results file and append per-round rows to the matching Sheets tab.

    Returns True on success.
    Returns False (silently or with a message) when Sheets export is not configured
    or when a recoverable error occurs — the simulator continues either way.
    """
    cfg = _load_config(config_path)
    if cfg is None:
        return False                        # sheets_config.json not present -> skip

    credentials_file = cfg.get("credentials_file", "service_account.json")
    spreadsheet_id   = cfg.get("spreadsheet_id", "")

    if not spreadsheet_id or spreadsheet_id.startswith("YOUR_"):
        print("[Sheets] sheets_config.json not filled in — skipping export.")
        return False

    if not Path(credentials_file).exists():
        print(f"[Sheets] Credentials file '{credentials_file}' not found — skipping export.")
        return False

    # -- dependencies ---------------------------------------------------------
    try:
        import gspread                                          # type: ignore
        from google.oauth2.service_account import Credentials  # type: ignore
    except ImportError:
        print("[Sheets] Required packages missing. Run:  pip install gspread google-auth")
        return False

    # -- load JSON results ----------------------------------------------------
    try:
        with open(results_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[Sheets] Could not read results file: {exc}")
        return False

    # Extract fields (support both flat and nested config sections)
    c             = data.get("config", {})
    dataset       = c.get("dataset",      data.get("dataset",      "unknown"))
    aggregation   = c.get("aggregation",  data.get("aggregation",  "unknown"))
    topology      = c.get("topology",     data.get("topology",     "ring"))
    num_nodes     = c.get("num_nodes",    data.get("num_nodes",    0))
    total_rounds  = c.get("num_rounds",   data.get("num_rounds",   0))
    attack_ratio  = c.get("attack_ratio", data.get("attack_ratio", 0.0))
    attack_type   = c.get("attack_type",  data.get("attack_type",  "directed"))
    timestamp     = data.get("run_timestamp", "")
    run_id        = data.get("run_id", Path(results_path).stem)
    round_results = data.get("round_results", [])
    summary       = data.get("summary", {})
    final_acc     = summary.get("final_accuracy", "")
    # final_loss lives in the last round's avg_loss, not in summary
    final_loss_v  = round_results[-1].get("avg_loss", "") if round_results else ""

    if not round_results:
        print("[Sheets] No round_results found in JSON — nothing to export.")
        return False

    # -- authenticate ---------------------------------------------------------
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds  = Credentials.from_service_account_file(credentials_file, scopes=scopes)
        client = gspread.authorize(creds)
        spread = client.open_by_key(spreadsheet_id)
    except Exception as exc:
        print(f"[Sheets] Authentication / open failed: {exc}")
        return False

    # -- find or create the right worksheet -----------------------------------
    tab = tab_name_for(dataset, attack_type, attack_ratio)
    try:
        all_sheets = spread.worksheets()
    except Exception as exc:
        print(f"[Sheets] Could not list worksheets: {exc}")
        return False

    # Case-insensitive match so manually-named tabs like "FEMNIST - None" still hit
    ws = None
    for sheet in all_sheets:
        if sheet.title == tab:          # exact match first
            ws = sheet
            break
    if ws is None:
        for sheet in all_sheets:
            if sheet.title.lower() == tab.lower():
                ws = sheet
                break
    if ws is None:
        ws = spread.add_worksheet(title=tab, rows=2000, cols=len(HEADER))
        print(f"[Sheets] Created new tab '{tab}'")

    # Convenience: last column letter for range strings (HEADER has 17 cols → 'Q')
    last_col = chr(ord('A') + len(HEADER) - 1)

    def _fmt_header_row():
        """Apply bold + dark-blue background to the header row (row 1)."""
        try:
            ws.format(f'A1:{last_col}1', {
                'backgroundColor': {'red': 0.145, 'green': 0.165, 'blue': 0.278},
                'textFormat': {
                    'bold': True,
                    'foregroundColor': {'red': 0.878, 'green': 0.894, 'blue': 0.949},
                },
                'horizontalAlignment': 'CENTER',
            })
        except Exception:
            pass  # formatting is best-effort

    # Write header if the first row is missing or doesn't look like our header
    existing = ws.get_all_values()
    header_is_new = False
    if not existing:
        ws.append_row(HEADER, value_input_option="USER_ENTERED")
        existing = [HEADER]
        header_is_new = True
    elif existing[0] != HEADER:
        ws.insert_row(HEADER, index=1, value_input_option="USER_ENTERED")
        existing = [HEADER] + existing
        header_is_new = True
        print(f"[Sheets] Inserted header row into existing tab '{tab}'")

    if header_is_new:
        _fmt_header_row()

    # -- build rows (one per round) -------------------------------------------
    last_round = total_rounds or (round_results[-1].get("round", 0) if round_results else 0)
    rows = []
    for rr in round_results:
        rnd = rr.get("round", "")
        is_last = (rnd == last_round)
        rows.append([
            timestamp,
            run_id,
            aggregation,
            dataset,
            topology,
            num_nodes,
            total_rounds,
            attack_ratio,
            attack_type,
            rnd,
            rr.get("avg_accuracy",        ""),
            rr.get("std_accuracy",         ""),
            rr.get("avg_loss",             ""),
            rr.get("honest_accuracy",      rr.get("avg_accuracy", "")),
            rr.get("compromised_accuracy", ""),
            final_acc    if is_last else "",
            final_loss_v if is_last else "",
        ])

    # -- write in one batch request -------------------------------------------
    try:
        ws.append_rows(rows, value_input_option="USER_ENTERED")
        print(f"[Sheets] Exported {len(rows)} rows -> tab '{tab}'")
    except Exception as exc:
        print(f"[Sheets] Write failed: {exc}")
        return False

    # -- export advanced metrics to a separate tab ----------------------------
    try:
        _export_metrics_tab(
            spread, data, timestamp, run_id,
            aggregation, dataset, topology, num_nodes, attack_ratio, attack_type,
        )
    except Exception as exc:
        print(f"[Sheets] Metrics tab export failed (non-fatal): {exc}")

    return True


def _export_metrics_tab(spread, data: dict,
                        timestamp, run_id, aggregation, dataset,
                        topology, num_nodes, attack_ratio, attack_type):
    """Write per-round advanced metrics to a dedicated '<Dataset> - Metrics' tab."""
    round_results = data.get("round_results", [])
    summary = data.get("summary", {})
    detection = summary.get("detection", {}) or {}
    t_detect = detection.get("detection_time", "")

    metrics_rows = []
    for rr in round_results:
        rnd = rr.get("round", "")
        flags_list = rr.get("detection_flags", [])
        n_flagged = sum(1 for f in flags_list if f.get("flagged")) if isinstance(flags_list, list) else ""
        metrics_rows.append([
            timestamp,
            run_id,
            aggregation,
            dataset,
            topology,
            num_nodes,
            attack_ratio,
            attack_type,
            rnd,
            rr.get("drift_mean", ""),
            rr.get("drift_std", ""),
            rr.get("peer_deviation_mean", ""),
            rr.get("consensus_score", ""),
            rr.get("regression_slope", ""),
            rr.get("regression_r_squared", ""),
            n_flagged,
            detection.get("true_positives", ""),
            detection.get("false_positives", ""),
            detection.get("true_negatives", ""),
            detection.get("false_negatives", ""),
            t_detect,
            rr.get("time_without_detection", data.get("summary", {}).get("overhead_avg", {}).get("without_detection", "")),
            rr.get("time_with_detection", data.get("summary", {}).get("overhead_avg", {}).get("with_detection", "")),
        ])

    if not metrics_rows:
        return

    ds_label = _DATASET_LABEL.get(dataset.lower(), dataset.title())
    tab_name = f"{ds_label} - Metrics"
    last_col = chr(ord('A') + len(METRICS_HEADER) - 1)

    # Find or create the metrics tab
    all_sheets = spread.worksheets()
    ws = None
    for sheet in all_sheets:
        if sheet.title == tab_name:
            ws = sheet
            break
    if ws is None:
        for sheet in all_sheets:
            if sheet.title.lower() == tab_name.lower():
                ws = sheet
                break
    if ws is None:
        ws = spread.add_worksheet(title=tab_name, rows=2000, cols=len(METRICS_HEADER))
        print(f"[Sheets] Created new metrics tab '{tab_name}'")

    # Header
    existing = ws.get_all_values()
    header_is_new = False
    if not existing:
        ws.append_row(METRICS_HEADER, value_input_option="USER_ENTERED")
        header_is_new = True
    elif existing[0] != METRICS_HEADER:
        ws.insert_row(METRICS_HEADER, index=1, value_input_option="USER_ENTERED")
        header_is_new = True

    if header_is_new:
        try:
            ws.format(f'A1:{last_col}1', {
                'backgroundColor': {'red': 0.145, 'green': 0.165, 'blue': 0.278},
                'textFormat': {
                    'bold': True,
                    'foregroundColor': {'red': 0.878, 'green': 0.894, 'blue': 0.949},
                },
                'horizontalAlignment': 'CENTER',
            })
        except Exception:
            pass

    ws.append_rows(metrics_rows, value_input_option="USER_ENTERED")
    print(f"[Sheets] Exported {len(metrics_rows)} metrics rows -> tab '{tab_name}'")
