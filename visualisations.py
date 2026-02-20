"""Visualizations and Analysis for Federated Learning. 
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict, Optional
import json
import os
from scipy import stats

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


""" Visualisation Functions"""


def plot_accuracy_evolution(results_file: str, output_path: str):
    """Plot accuracy evolution over rounds."""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    accuracies = np.array(data['results']['accuracies'])
    rounds = len(accuracies)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot individual nodes
    for node_idx in range(accuracies.shape[1]):
        is_compromised = node_idx in data['compromised_nodes']
        color = 'red' if is_compromised else 'lightblue'
        alpha = 0.3
        ax.plot(range(rounds), accuracies[:, node_idx], 
               color=color, alpha=alpha, linewidth=1)
    
    # Plot averages
    if data['compromised_nodes']:
        honest_accs = data['results']['honest_accuracies']
        comp_accs = data['results']['compromised_accuracies']
        
        ax.plot(range(rounds), honest_accs, 'g-o', 
               label='Honest Nodes (Avg)', linewidth=3, markersize=6)
        ax.plot(range(rounds), comp_accs, 'r-s', 
               label='Compromised Nodes (Avg)', linewidth=3, markersize=6)
    else:
        avg_accs = accuracies.mean(axis=1)
        ax.plot(range(rounds), avg_accs, 'b-o', 
               label='All Nodes (Avg)', linewidth=3, markersize=6)
    
    ax.set_xlabel('Round', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test Accuracy', fontsize=14, fontweight='bold')
    ax.set_title(f"Accuracy Evolution - {data['config']['aggregation'].upper()}", 
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f" Saved: {output_path}")


def plot_network_topology(results_file: str, output_path: str):
    """Visualize network topology with compromised nodes."""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    num_nodes = data['config']['num_nodes']
    topology = data['config']['topology']
    compromised = set(data['compromised_nodes'])
    
    # Recreate graph
    from decentralised_fl_sim import NetworkGraph
    graph = NetworkGraph(num_nodes, topology, k=4)
    
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(graph.get_edge_list())
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Layout
    if topology == "ring":
        pos = nx.circular_layout(G)
    elif topology == "fully":
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    else:
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
    
    # Node colors and sizes
    node_colors = ['#e74c3c' if i in compromised else '#3498db' 
                   for i in range(num_nodes)]
    node_sizes = [1200 if i in compromised else 800 
                  for i in range(num_nodes)]
    
    # Draw
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=node_sizes, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.4, width=2, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, 
                           font_weight='bold', font_color='white', ax=ax)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label=f'Honest Nodes ({num_nodes - len(compromised)})'),
        Patch(facecolor='#e74c3c', label=f'Compromised Nodes ({len(compromised)})')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=14)
    
    ax.set_title(f"Network Topology: {topology.capitalize()} ({num_nodes} nodes)", 
                fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_convergence_comparison(results_files: Dict[str, str], output_path: str):
    """Compare convergence of different algorithms."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    
    for idx, (label, filepath) in enumerate(results_files.items()):
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if 'honest_accuracies' in data['results'] and data['results']['honest_accuracies']:
            accs = data['results']['honest_accuracies']
        else:
            accs = np.array(data['results']['accuracies']).mean(axis=1)
        
        ax.plot(range(len(accs)), accs, '-o', 
               label=label, color=colors[idx % len(colors)], 
               linewidth=2.5, markersize=6)
    
    ax.set_xlabel('Round', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Algorithm Convergence Comparison', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


""" Analysis  Functions """


def analyze_robustness(results_file: str) -> Dict:
    """Analyze robustness metrics."""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    if not data['compromised_nodes']:
        return {"error": "No attack present"}
    
    honest_accs = np.array(data['results']['honest_accuracies'])
    comp_accs = np.array(data['results']['compromised_accuracies'])
    
    # Metrics
    final_gap = honest_accs[-1] - comp_accs[-1]
    avg_gap = np.mean(honest_accs - comp_accs)
    
    # Convergence rate (last 5 rounds improvement)
    if len(honest_accs) >= 5:
        recent_improvement = honest_accs[-1] - honest_accs[-5]
    else:
        recent_improvement = honest_accs[-1] - honest_accs[0]
    
    # Attack success rate (lower is better for defense)
    attack_success = 1.0 - (honest_accs[-1] - comp_accs[-1])
    
    analysis = {
        "final_honest_accuracy": float(honest_accs[-1]),
        "final_compromised_accuracy": float(comp_accs[-1]),
        "final_accuracy_gap": float(final_gap),
        "average_accuracy_gap": float(avg_gap),
        "convergence_rate": float(recent_improvement),
        "attack_success_rate": float(max(0, attack_success)),
        "robustness_score": float(final_gap / max(0.01, honest_accs[-1]))  # Normalized
    }
    
    return analysis


def analyze_statistical_significance(results_files: Dict[str, str]) -> pd.DataFrame:
    """Perform statistical tests between algorithms."""
    final_accs = {}
    
    for label, filepath in results_files.items():
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if 'honest_accuracies' in data['results'] and data['results']['honest_accuracies']:
            final_accs[label] = data['results']['honest_accuracies'][-1]
        else:
            final_accs[label] = np.mean(data['results']['accuracies'][-1])
    
    # Create comparison matrix
    algorithms = list(final_accs.keys())
    comparison_df = pd.DataFrame(index=algorithms, columns=algorithms)
    
    for alg1 in algorithms:
        for alg2 in algorithms:
            if alg1 == alg2:
                comparison_df.loc[alg1, alg2] = "-"
            else:
                diff = final_accs[alg1] - final_accs[alg2]
                comparison_df.loc[alg1, alg2] = f"{diff:+.4f}"
    
    return comparison_df


def generate_summary_report(results_files: Dict[str, str], output_path: str):
    #Generate comprehensive summary report.
    report = []
    report.append("FEDERATED LEARNING EXPERIMENT SUMMARY REPORT")
    report.append("")
    
    for label, filepath in results_files.items():
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        report.append("\n----------------------------------------")
        report.append(f"EXPERIMENT: {label}")
        report.append("----------------------------------------")
        
        # Configuration
        config = data['config']
        report.append(f"\nConfiguration:")
        report.append(f"  Dataset: {config['dataset']}")
        report.append(f"  Nodes: {config['num_nodes']}")
        report.append(f"  Rounds: {config['num_rounds']}")
        report.append(f"  Topology: {config['topology']}")
        report.append(f"  Aggregation: {config['aggregation']}")
        report.append(f"  Attack Ratio: {config['attack_ratio']:.1%}")
        report.append(f"  Attack Type: {config['attack_type']}")
        
        # Results
        report.append(f"\nResults:")
        final_accs = data['results']['accuracies'][-1]
        report.append(f"  Final Accuracy: {np.mean(final_accs):.4f} ± {np.std(final_accs):.4f}")
        
        if data['compromised_nodes']:
            honest_acc = data['results']['honest_accuracies'][-1]
            comp_acc = data['results']['compromised_accuracies'][-1]
            report.append(f"  Honest Nodes: {honest_acc:.4f}")
            report.append(f"  Compromised Nodes: {comp_acc:.4f}")
            report.append(f"  Protection Gap: {honest_acc - comp_acc:.4f}")
            
            # Robustness analysis
            analysis = analyze_robustness(filepath)
            report.append(f"\nRobustness Analysis:")
            report.append(f"  Robustness Score: {analysis['robustness_score']:.4f}")
            report.append(f"  Attack Success Rate: {analysis['attack_success_rate']:.4f}")
            report.append(f"  Convergence Rate: {analysis['convergence_rate']:.4f}")
    
    # Save report
    report_text = "\n".join(report)
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    print(f" Saved report: {output_path}")
    print("\n" + report_text)


def plot_robustness_comparison(results_files: Dict[str, str], output_path: str):
    """Compare robustness metrics across algorithms."""
    metrics = {}
    
    for label, filepath in results_files.items():
        analysis = analyze_robustness(filepath)
        if "error" not in analysis:
            metrics[label] = analysis
    
    if not metrics:
        print("No attack data available for robustness comparison")
        return
    
    # Create comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Robustness Metrics Comparison', fontsize=16, fontweight='bold')
    
    algorithms = list(metrics.keys())
    
    # Metric 1: Final Accuracy Gap
    ax = axes[0, 0]
    gaps = [metrics[alg]['final_accuracy_gap'] for alg in algorithms]
    bars = ax.bar(algorithms, gaps, color='#2ecc71', alpha=0.8)
    ax.set_ylabel('Accuracy Gap', fontweight='bold')
    ax.set_title('Final Accuracy Gap (Higher = Better)')
    ax.tick_params(axis='x', rotation=15)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Metric 2: Robustness Score
    ax = axes[0, 1]
    scores = [metrics[alg]['robustness_score'] for alg in algorithms]
    bars = ax.bar(algorithms, scores, color='#3498db', alpha=0.8)
    ax.set_ylabel('Robustness Score', fontweight='bold')
    ax.set_title('Robustness Score (Higher = Better)')
    ax.tick_params(axis='x', rotation=15)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Metric 3: Attack Success Rate
    ax = axes[1, 0]
    attack_success = [metrics[alg]['attack_success_rate'] for alg in algorithms]
    bars = ax.bar(algorithms, attack_success, color='#e74c3c', alpha=0.8)
    ax.set_ylabel('Attack Success Rate', fontweight='bold')
    ax.set_title('Attack Success Rate (Lower = Better)')
    ax.tick_params(axis='x', rotation=15)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Metric 4: Final Honest Accuracy
    ax = axes[1, 1]
    honest_accs = [metrics[alg]['final_honest_accuracy'] for alg in algorithms]
    bars = ax.bar(algorithms, honest_accs, color='#f39c12', alpha=0.8)
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Final Honest Accuracy (Higher = Better)')
    ax.tick_params(axis='x', rotation=15)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


"""Complete Analysis and Comparison Functions"""


def analyze_experiment(results_file: str, output_dir: str):
    """Run complete analysis on a single experiment."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n Analyzing: {results_file}")
    
    # Generate visualizations
    plot_accuracy_evolution(results_file, f"{output_dir}/accuracy.png")
    plot_network_topology(results_file, f"{output_dir}/topology.png")
    
    # Generate analysis
    analysis = analyze_robustness(results_file)
    if "error" not in analysis:
        with open(f"{output_dir}/analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f" Saved analysis: {output_dir}/analysis.json")


def compare_experiments(results_files: Dict[str, str], output_dir: str):
    """Compare multiple experiments."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n Comparing {len(results_files)} experiments")
    
    # Generate comparison plots
    plot_convergence_comparison(results_files, f"{output_dir}/convergence_comparison.png")
    plot_robustness_comparison(results_files, f"{output_dir}/robustness_comparison.png")
    
    # Generate summary report
    generate_summary_report(results_files, f"{output_dir}/summary_report.txt")
    
    # Statistical comparison
    comparison_df = analyze_statistical_significance(results_files)
    comparison_df.to_csv(f"{output_dir}/statistical_comparison.csv")
    print(f" Saved statistical comparison: {output_dir}/statistical_comparison.csv")



"""Main Execution"""


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single experiment: python visualizations.py results/experiment.json")
        print("  Comparison: python visualizations.py results/*.json")
        sys.exit(1)
    
    results_files = sys.argv[1:]
    
    if len(results_files) == 1:
        # Single experiment analysis
        analyze_experiment(results_files[0], "results/analysis")
    else:
        # Multiple experiment comparison
        labeled_files = {f"Exp{i+1}": f for i, f in enumerate(results_files)}
        compare_experiments(labeled_files, "results/comparison")