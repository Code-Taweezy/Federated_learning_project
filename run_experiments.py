"""
Automated Experiment Runner for FL Simulations
Runs multiple experiments and generates comparison reports
"""

import re
import subprocess
import json
import os
import time
import sys
from datetime import datetime
from typing import List, Dict

# Matches a round-metrics table row from decentralised_fl_sim.py output:
#   <round> | <avg_acc> | <std_acc> | <avg_loss> [ | <honest> | <compromised> ]
ROUND_RE = re.compile(
    r'^\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)'
    r'(?:\s*\|\s*([\d.]+)\s*\|\s*([\d.]+))?'
)

# Matches a METRICS|... structured line emitted after each round
METRICS_RE = re.compile(
    r'^METRICS\|(\d+)\|'                          # round
    r'([\d.eE+-]+)\|([\d.eE+-]+)\|'               # drift_mean | drift_std
    r'([\d.eE+-]+)\|([\d.eE+-]+)\|'               # peer_dev_mean | consensus
    r'([\d.eE+-]+)\|([\d.eE+-]+)\|'               # slope | r_squared
    r'(\d+)\|(\d+)\|(\d+)\|(\d+)\|(\d+)\|'        # n_flagged | tp | fp | tn | fn
    r'([\d.eE+-]+)\|([\d.eE+-]+)'                 # time_without | time_with
)


# ── Experiment Configurations ──────────────────────────────────────


class Experiment:
    """Simple experiment configuration."""
    def __init__(self, name, **kwargs):
        self.name = name
        # Defaults
        self.dataset = kwargs.get('dataset', 'femnist')
        self.num_nodes = kwargs.get('num_nodes', 32)
        self.rounds = kwargs.get('rounds', 50)
        self.topology = kwargs.get('topology', 'ring')
        self.aggregation = kwargs.get('aggregation', 'fedavg')
        self.attack_ratio = kwargs.get('attack_ratio', 0.0)
        self.attack_type = kwargs.get('attack_type', 'directed')
        self.seed = kwargs.get('seed', 42)
        self.k = kwargs.get('k', 4)
    
    def to_command(self, output_file):
        """Generate command line arguments."""
        cmd = [
            sys.executable, 'decentralised_fl_sim.py',
            '--dataset', self.dataset,
            '--num-nodes', str(self.num_nodes),
            '--rounds', str(self.rounds),
            '--topology', self.topology,
            '--aggregation', self.aggregation,
            '--attack-ratio', str(self.attack_ratio),
            '--attack-type', self.attack_type,
            '--seed', str(self.seed),
            '--k', str(self.k),
            '--output', output_file
        ]
        return cmd


# ── Experiment Suites ──────────────────────────────────────────────


# Suite 1: Baseline (Quick Test)
BASELINE_SUITE = [
    Experiment('baseline_no_attack', 
               aggregation='fedavg', 
               attack_ratio=0.0, 
               rounds=50),
    
    Experiment('baseline_with_attack', 
               aggregation='fedavg', 
               attack_ratio=0.25, 
               rounds=50),
]

# Suite 2: Algorithm Comparison (Main Test)
ALGORITHM_SUITE = [
    Experiment('fedavg', 
               aggregation='fedavg', 
               attack_ratio=0.25,
               rounds=50),
    
    Experiment('balance', 
               aggregation='balance', 
               attack_ratio=0.25,
               rounds=50),
    
    Experiment('ubar', 
               aggregation='ubar', 
               attack_ratio=0.25,
               rounds=50),
]

# Suite 3: Topology Comparison (30-45 min)
TOPOLOGY_SUITE = [
    Experiment('ring', 
               topology='ring', 
               aggregation='balance', 
               attack_ratio=0.25),
    
    Experiment('k_regular', 
               topology='k-regular', 
               aggregation='balance', 
               attack_ratio=0.25,
               k=4),
    
    Experiment('fully', 
               topology='fully', 
               aggregation='balance', 
               attack_ratio=0.25,
               num_nodes=6),
]

# Suite 4: Quick Debug (2-3 min)
DEBUG_SUITE = [
    Experiment('quick_test', 
               num_nodes=4, 
               rounds=3, 
               aggregation='fedavg'),
]

# Suite 5: Shakespeare Dataset - Algorithm Comparison
SHAKESPEARE_SUITE = [
    Experiment('shakespeare_fedavg',
               dataset='shakespeare', aggregation='fedavg',
               attack_ratio=0.25, rounds=50),
    Experiment('shakespeare_balance',
               dataset='shakespeare', aggregation='balance',
               attack_ratio=0.25, rounds=50),
    Experiment('shakespeare_ubar',
               dataset='shakespeare', aggregation='ubar',
               attack_ratio=0.25, rounds=50),
]

# Suite 6: Cross-Dataset Comparison  – built dynamically (see _build_cross_dataset_suite)
def _build_cross_dataset_suite(attack_ratio: float = 0.25,
                               attack_type: str = 'directed',
                               topology: str = 'ring') -> List[Experiment]:
    """FedAvg, BALANCE and UBAR across femnist and shakespeare (6 experiments)."""
    datasets     = ['femnist', 'shakespeare']
    aggregators  = ['fedavg', 'balance', 'ubar']
    experiments  = []
    for ds in datasets:
        for agg in aggregators:
            experiments.append(Experiment(
                f'{ds}_{agg}',
                dataset=ds,
                aggregation=agg,
                attack_ratio=attack_ratio,
                attack_type=attack_type,
                topology=topology,
                rounds=50,
            ))
    return experiments


# ── Experiment Runner ──────────────────────────────────────────────


class ExperimentRunner:
    """Manages experiment execution and result collection."""
    
    def __init__(self, suite_name: str, experiments: List[Experiment]):
        self.suite_name = suite_name
        self.experiments = experiments
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = f'results/{suite_name}_{self.timestamp}'
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f'{self.output_dir}/individual', exist_ok=True)
        
        print(f"\nExperiment Suite: {suite_name}")
        print(f" Output directory: {self.output_dir}")
        print(f"Number of experiments: {len(experiments)}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def run_all(self, dashboard=None):
        """Run all experiments in the suite."""
        results_files = {}
        
        for idx, exp in enumerate(self.experiments, 1):
            print(f"\n{'─'*80}")
            print(f"Experiment {idx}/{len(self.experiments)}: {exp.name}")
            print(f"{'─'*80}")
            
            output_file = f'{self.output_dir}/individual/{exp.name}.json'

            if dashboard:
                dashboard.notify_start(exp.name, idx, len(self.experiments))

            # Run experiment
            success = self._run_experiment(exp, output_file, dashboard=dashboard)
            
            if success:
                results_files[exp.name] = output_file
                print(f"Completed: {exp.name}")
            else:
                print(f" Failed: {exp.name}")
        
        if dashboard:
            dashboard.notify_suite_done()

        # Generate comparison visualizations
        if len(results_files) > 1:
            print("\nGenerating comparison analysis\n")
            self._generate_comparison(results_files)
        
        # Print summary
        self._print_summary(results_files)
        
        return results_files
    
    def _run_experiment(self, exp: Experiment, output_file: str, dashboard=None) -> bool:
        """Run a single experiment, streaming output line-by-line."""
        try:
            start_time = time.time()

            cmd = exp.to_command(output_file)

            print(f"Configuration:")
            print(f"   Dataset:     {exp.dataset}")
            print(f"   Nodes:       {exp.num_nodes}, Rounds: {exp.rounds}")
            print(f"   Topology:    {exp.topology}")
            print(f"   Aggregation: {exp.aggregation}")
            print(f"   Attack:      {exp.attack_ratio:.0%} ({exp.attack_type})")
            print(f"\nRunning...")

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Buffer for pending metrics line (arrives right after the round row)
            _pending_metrics = {}

            for line in proc.stdout:
                line_stripped = line.rstrip()
                print(line_stripped)
                if dashboard:
                    # Try METRICS line first (emitted right after the round row)
                    mm = METRICS_RE.match(line_stripped)
                    if mm:
                        rnd = int(mm.group(1))
                        _pending_metrics[rnd] = {
                            "drift_mean":        float(mm.group(2)),
                            "drift_std":         float(mm.group(3)),
                            "peer_dev_mean":     float(mm.group(4)),
                            "consensus":         float(mm.group(5)),
                            "slope":             float(mm.group(6)),
                            "r_squared":         float(mm.group(7)),
                            "n_flagged":         int(mm.group(8)),
                            "tp":                int(mm.group(9)),
                            "fp":                int(mm.group(10)),
                            "tn":                int(mm.group(11)),
                            "fn":                int(mm.group(12)),
                            "time_without_det":  float(mm.group(13)),
                            "time_with_det":     float(mm.group(14)),
                        }
                        # If the round row was already sent, send a metrics-only update
                        dashboard.notify_metrics(exp.name, rnd, _pending_metrics[rnd])
                        continue

                    m = ROUND_RE.match(line_stripped)
                    if m:
                        rnd = int(m.group(1))
                        row = {
                            "round":                rnd,
                            "avg_accuracy":         float(m.group(2)),
                            "std_accuracy":         float(m.group(3)),
                            "avg_loss":             float(m.group(4)),
                            "honest_accuracy":      float(m.group(5)) if m.group(5) else None,
                            "compromised_accuracy": float(m.group(6)) if m.group(6) else None,
                        }
                        # Merge any pre-received metrics (shouldn't happen, but be safe)
                        if rnd in _pending_metrics:
                            row.update(_pending_metrics.pop(rnd))
                        dashboard.notify_round(exp.name, row)

            proc.wait()
            elapsed = time.time() - start_time

            if proc.returncode == 0:
                print(f"Success in {elapsed:.1f}s")
                self._analyze_individual(output_file, exp.name)

                if dashboard:
                    try:
                        with open(output_file, encoding='utf-8') as fh:
                            saved = json.load(fh)
                        result = dict(saved.get("summary", {}))
                        result['success']  = True
                        result['duration'] = f"{elapsed:.1f}s"
                        dashboard.notify_done(exp.name, result)
                    except Exception:
                        dashboard.notify_done(exp.name, {'success': True,
                                                          'duration': f"{elapsed:.1f}s"})
                return True
            else:
                print(f"Failed after {elapsed:.1f}s")
                if dashboard:
                    dashboard.notify_done(exp.name, {'error': True,
                                                      'duration': f"{elapsed:.1f}s"})
                return False

        except Exception as e:
            print(f"Exception: {e}")
            if dashboard:
                dashboard.notify_done(exp.name, {'error': str(e)})
            return False
    
    def _analyze_individual(self, results_file: str, exp_name: str):
        """Generate analysis for individual experiment."""
        try:
            from visualisations import analyze_experiment
            
            analysis_dir = f'{self.output_dir}/individual/{exp_name}_analysis'
            analyze_experiment(results_file, analysis_dir)
            
        except Exception as e:
            print(f"Could not generate individual analysis: {e}")
    
    def _generate_comparison(self, results_files: Dict[str, str]):
        """Generate comparison visualizations."""
        try:
            from visualisations import compare_experiments
            
            comparison_dir = f'{self.output_dir}/comparison'
            compare_experiments(results_files, comparison_dir)
            
            print(f" Comparison saved to: {comparison_dir}")
            
        except Exception as e:
            print(f" Could not generate comparison: {e}")
    
    def _print_summary(self, results_files: Dict[str, str]):
        """Print final summary."""
        print("\nExperiment Suite Summary")
        print(f"Suite: {self.suite_name}")
        print(f"Completed: {len(results_files)}/{len(self.experiments)}")
        print(f"Output: {self.output_dir}")
        print()
        
        if results_files:
            print(" Results:")
            for name, filepath in results_files.items():
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    final_acc = data['results']['accuracies'][-1]
                    avg_acc = sum(final_acc) / len(final_acc)
                    
                    if data['compromised_nodes']:
                        honest_acc = data['results']['honest_accuracies'][-1]
                        print(f"  {name:20s}: Honest={honest_acc:.4f}, Overall={avg_acc:.4f}")
                    else:
                        print(f"  {name:20s}: Accuracy={avg_acc:.4f}")
                
                except:
                    print(f"  {name:20s}: (error reading results)")
        
        print()



# MAIN MENU

def print_menu():
    """Print experiment suite menu."""
    print("\nFEDERATED LEARNING EXPERIMENT RUNNER")
    print("\nAvailable Experiment Suites:\n")
    print("  1. BASELINE SUITE")
    print("     - 2 experiments (32 nodes, 50 rounds)")
    print("     - FedAvg no-attack vs directed-attack baseline\n")
    print("  2. ALGORITHM COMPARISON")
    print("     - 3 experiments (32 nodes, 50 rounds)")
    print("     - FedAvg vs BALANCE vs UBAR  (femnist)\n")
    print("  3. TOPOLOGY COMPARISON")
    print("     - 3 experiments (32 nodes, 50 rounds)")
    print("     - Ring vs k-regular vs fully-connected\n")
    print("  4. SHAKESPEARE DATASET")
    print("     - 4 experiments (32 nodes, 50 rounds)")
    print("     - Algorithm comparison on Shakespeare (character-level LSTM)\n")
    print("  5. CROSS-DATASET COMPARISON")
    print("     - 6 experiments (32 nodes, 50 rounds)")
    print("     - FedAvg / BALANCE / UBAR across femnist and shakespeare")
    print("     - Configurable attack ratio + attack type (dialog shown before launch)\n")
    print("  6. CUSTOM (define your own)\n")
    print("  0. Exit\n")


def run_custom_suite():
    """Run custom experiments."""
    print("\n Custom Experiment Configuration")
    
    experiments = []
    
    while True:
        print(f"\nCurrent experiments: {len(experiments)}")
        
        name = input("Experiment name (or 'done' to finish): ").strip()
        if name.lower() == 'done':
            break
        
        print("\nAggregation: 1=FedAvg, 2=BALANCE, 3=UBAR")
        agg_choice = input("Choose (1-3): ").strip()
        agg_map = {'1': 'fedavg', '2': 'balance', '3': 'ubar'}
        aggregation = agg_map.get(agg_choice, 'fedavg')
        
        print("\nTopology: 1=Ring, 2=k-Regular, 3=Fully")
        topo_choice = input("Choose (1-3) [default 1]: ").strip()
        topo_map = {'1': 'ring', '2': 'k-regular', '3': 'fully'}
        topology = topo_map.get(topo_choice, 'ring')

        attack_ratio = float(input("Attack ratio (0.0-0.5): ").strip() or "0.25")
        rounds = int(input("Number of rounds (default 50): ").strip() or "50")

        exp = Experiment(name,
                        aggregation=aggregation,
                        topology=topology,
                        attack_ratio=attack_ratio,
                        rounds=rounds)
        experiments.append(exp)
        print(f" Added: {name}")
    
    return experiments


def _configure_cross_dataset() -> dict:
    """
    Show a styled tkinter config dialog for the cross-dataset suite.
    Returns {'attack_ratio': float, 'attack_type': str} or None if cancelled.
    """
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        # Fall back to terminal prompts
        try:
            ratio = float(input("Attack ratio (0.0-0.5) [default 0.25]: ").strip() or "0.25")
            atype = input("Attack type – directed or gaussian [default directed]: ").strip() or "directed"
            topo  = input("Topology – ring / k-regular / fully [default ring]: ").strip() or "ring"
            return {'attack_ratio': max(0.0, min(0.5, ratio)),
                    'attack_type': atype if atype in ('directed', 'gaussian') else 'directed',
                    'topology':    topo  if topo  in ('ring', 'k-regular', 'fully') else 'ring'}
        except Exception:
            return None

    # Pull design tokens from the dashboard so both UIs stay in sync
    try:
        from results_dashboard import (
            BG, SURFACE, CARD, BORDER, HDR_BG, HDR_FG,
            ACCENT, GREEN, AMBER, DIM, FG, MUTED,
            FONT_UI, FONT_MONO,
        )
    except ImportError:
        BG = '#0d1021'; SURFACE = '#141729'; CARD = '#1a1e35'
        BORDER = '#252a47'; HDR_BG = '#10142a'; HDR_FG = '#e8ecf8'
        ACCENT = '#6b8ef5'; GREEN = '#4ec98b'; AMBER = '#f0a832'
        DIM = '#8b96b8'; FG = '#e0e4f2'; MUTED = '#434c6e'
        FONT_UI = 'Segoe UI'; FONT_MONO = 'Consolas'

    result = {}

    # Enable DPI awareness for sharp text on high-DPI displays
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    root = tk.Tk()
    root.title("Configure Cross-Dataset Suite")
    root.geometry("500x520")
    root.resizable(False, False)
    root.configure(bg=BG)

    # ── header bar (matches dashboard header) ─────────────────────────────────
    hdr = tk.Frame(root, bg=HDR_BG, height=62)
    hdr.pack(fill='x'); hdr.pack_propagate(False)
    tk.Label(hdr, text='Federated Learning Dashboard',
             font=(FONT_UI, 13, 'bold'), bg=HDR_BG, fg=HDR_FG
             ).pack(side='left', padx=20, pady=(12, 0), anchor='sw')
    tk.Label(hdr, text='Cross-Dataset Configuration',
             font=(FONT_UI, 9), bg=HDR_BG, fg=DIM
             ).pack(side='left', padx=(6, 0), pady=(28, 8), anchor='sw')
    # thin accent line
    tk.Frame(root, bg=ACCENT, height=2).pack(fill='x')

    body = tk.Frame(root, bg=BG)
    body.pack(fill='both', expand=True, padx=24, pady=18)

    def _section_label(text):
        """Styled section heading matching the dashboard's MUTED caps labels."""
        tk.Label(body, text=text.upper(),
                 font=(FONT_UI, 9, 'bold'), bg=BG, fg=MUTED
                 ).pack(anchor='w', pady=(14, 4))
        tk.Frame(body, bg=BORDER, height=1).pack(fill='x', pady=(0, 10))

    # ── attack ratio slider ───────────────────────────────────────────────────
    _section_label('Attack Ratio')

    slider_row = tk.Frame(body, bg=BG)
    slider_row.pack(fill='x')

    ratio_var = tk.DoubleVar(value=0.25)
    ratio_lbl = tk.Label(slider_row, text='0.25', width=5,
                         font=(FONT_MONO, 12, 'bold'), bg=BG, fg=AMBER)
    ratio_lbl.pack(side='right')

    def _on_slider(val):
        v = round(float(val), 2)
        ratio_lbl.config(text=f'{v:.2f}')

    slider = tk.Scale(slider_row, from_=0.0, to=0.5, resolution=0.05,
                      orient='horizontal', variable=ratio_var,
                      command=_on_slider,
                      bg=BG, fg=FG, troughcolor=BORDER,
                      highlightthickness=0, activebackground=ACCENT,
                      sliderrelief='flat', length=300, showvalue=False)
    slider.pack(side='left', fill='x', expand=True)

    tk.Label(body, text='0.0 = no attack    0.5 = 50% compromised',
             font=(FONT_UI, 8), bg=BG, fg=MUTED
             ).pack(anchor='w', pady=(0, 2))

    # ── attack type toggle ────────────────────────────────────────────────────
    _section_label('Attack Type')

    type_var = tk.StringVar(value='directed')
    toggle_row = tk.Frame(body, bg=BG)
    toggle_row.pack(anchor='w')

    for label, value, colour in [
        ('Directed',  'directed', ACCENT),
        ('Gaussian',  'gaussian', GREEN),
    ]:
        tk.Radiobutton(
            toggle_row, text=label, variable=type_var, value=value,
            font=(FONT_UI, 11), bg=BG, fg=colour,
            selectcolor=CARD, activebackground=BG,
            activeforeground=colour, indicatoron=True,
        ).pack(side='left', padx=(0, 24))

    tk.Label(body,
             text='Directed: push models toward a target class\n'
                  'Gaussian: add noise to gradients',
             font=(FONT_UI, 9), bg=BG, fg=MUTED, justify='left'
             ).pack(anchor='w', pady=(8, 2))

    # ── topology selector ─────────────────────────────────────────────────────
    _section_label('Topology')

    topo_var = tk.StringVar(value='ring')
    topo_row = tk.Frame(body, bg=BG)
    topo_row.pack(anchor='w')

    for label, value, colour in [
        ('Ring',      'ring',      ACCENT),
        ('k-Regular', 'k-regular', GREEN),
        ('Fully',     'fully',     AMBER),
    ]:
        tk.Radiobutton(
            topo_row, text=label, variable=topo_var, value=value,
            font=(FONT_UI, 11), bg=BG, fg=colour,
            selectcolor=CARD, activebackground=BG,
            activeforeground=colour, indicatoron=True,
        ).pack(side='left', padx=(0, 20))

    tk.Label(body,
             text='Ring: chain  |  k-Regular: k nearest  |  Fully: all connected',
             font=(FONT_UI, 9), bg=BG, fg=MUTED
             ).pack(anchor='w', pady=(8, 0))

    # ── buttons (matches dashboard footer style) ───────────────────────────────
    tk.Frame(root, bg=BORDER, height=1).pack(fill='x')
    btn_row = tk.Frame(root, bg=SURFACE, height=54)
    btn_row.pack(fill='x', side='bottom'); btn_row.pack_propagate(False)

    def _launch():
        result['attack_ratio'] = round(ratio_var.get(), 2)
        result['attack_type']  = type_var.get()
        result['topology']     = topo_var.get()
        root.destroy()

    def _cancel():
        root.destroy()

    tk.Button(btn_row, text='Cancel', command=_cancel,
              font=(FONT_UI, 10), bg=CARD, fg=DIM,
              relief='flat', padx=18, pady=6, cursor='hand2',
              activebackground=BORDER, activeforeground=FG,
              ).pack(side='right', padx=(0, 10), pady=10)
    tk.Button(btn_row, text='Launch Suite  ›', command=_launch,
              font=(FONT_UI, 10, 'bold'), bg=ACCENT, fg=BG,
              relief='flat', padx=18, pady=6, cursor='hand2',
              activebackground='#8fa8ff', activeforeground=BG,
              ).pack(side='right', padx=(0, 4), pady=10)

    root.mainloop()
    return result if result else None


def _run_suite(suite_name: str, experiments: List[Experiment]):
    """Launch a suite with the live results dashboard."""
    runner = ExperimentRunner(suite_name, experiments)

    try:
        from results_dashboard import ResultsDashboard
    except ImportError as e:
        print(f"results_dashboard not found ({e}), running without dashboard.")
        runner.run_all()
        return

    import threading

    dashboard = ResultsDashboard(suite_name, [e.name for e in experiments])

    def _worker():
        try:
            runner.run_all(dashboard=dashboard)
        except Exception as e:
            import traceback
            print(f"Worker error: {e}")
            traceback.print_exc()
            dashboard.notify_suite_done()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    dashboard.run()   # blocks in tkinter mainloop until window closed
    thread.join(timeout=60)


def main():
    """Main execution."""
    while True:
        print_menu()
        choice = input("Select suite (0-6): ").strip()

        if choice == '0':
            print("\nGoodbye!\n")
            break
        elif choice == '1':
            _run_suite('baseline', BASELINE_SUITE)
        elif choice == '2':
            _run_suite('algorithm_comparison', ALGORITHM_SUITE)
        elif choice == '3':
            _run_suite('topology_comparison', TOPOLOGY_SUITE)
        elif choice == '4':
            _run_suite('shakespeare_comparison', SHAKESPEARE_SUITE)
        elif choice == '5':
            cfg = _configure_cross_dataset()
            if cfg:
                exps = _build_cross_dataset_suite(
                    attack_ratio=cfg['attack_ratio'],
                    attack_type=cfg['attack_type'],
                    topology=cfg.get('topology', 'ring'),
                )
                _run_suite('cross_dataset', exps)
        elif choice == '6':
            custom_exps = run_custom_suite()
            if custom_exps:
                _run_suite('custom', custom_exps)
        else:
            print("Invalid choice.")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
