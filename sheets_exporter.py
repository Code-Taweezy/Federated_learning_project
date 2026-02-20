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

# Maps internal dataset names -> display names matching your spreadsheet tabs
_DATASET_LABEL: dict[str, str] = {
    "femnist":     "Femnist",
    "celeba":      "Celeba",
    "shakespeare": "Shakespear",   # matches your existing tab name
    "reddit":      "Reddit",
}

# Maps internal attack type names -> display names matching your spreadsheet tabs
_ATTACK_LABEL: dict[str, str] = {
    "directed": "Direct",
    "gaussian": "Gaussian",
    "none":     "None",
}


def tab_name_for(dataset: str, attack_type: str, attack_ratio: float) -> str:
    """Return the spreadsheet tab name for a given dataset / attack combination."""
    ds  = _DATASET_LABEL.get(dataset.lower(), dataset.title())
    if attack_ratio == 0.0:
        return f"{ds} - None"
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
    final_acc     = summary.get("final_accuracy",  "")
    final_loss_v  = summary.get("final_loss",       "")

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
        ws = spread.worksheet(tab)
    except gspread.WorksheetNotFound:
        ws = spread.add_worksheet(title=tab, rows=2000, cols=len(HEADER))
        print(f"[Sheets] Created new tab '{tab}'")

    # Write header if the sheet is completely empty
    existing = ws.get_all_values()
    if not existing:
        ws.append_row(HEADER, value_input_option="USER_ENTERED")

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
        return True
    except Exception as exc:
        print(f"[Sheets] Write failed: {exc}")
        return False
