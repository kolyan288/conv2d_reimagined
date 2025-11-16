from torch import nn
from experiments.common.conv2d_img2col import Conv2dImg2Col
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional


def custom_conv_model(k = 1):
    return nn.Sequential(
            Conv2dImg2Col(3, 16, kernel_size=k, stride=1, padding=k//2, bias=True),
            nn.ReLU(),
            Conv2dImg2Col(16, 32, kernel_size=k, stride=1, padding=k//2, bias=True),
            nn.ReLU(),
            Conv2dImg2Col(32, 64, kernel_size=k, stride=1, padding=k//2, bias=True),
        ) 


def basicconv_model(k = 1):
    return nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=k, stride=1, padding=k//2, bias=True),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=k, stride=1, padding=k//2, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=k, stride=1, padding=k//2, bias=True),
        )

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast


def compare_models(csv_file: str, key = 'latency_gpu_mean_ms'):
    """Compare models grouped by type, batch size, and kernel size"""
    df = pd.read_csv(csv_file)
    label_dev = 'GPU' if 'gpu' in key else 'CPU'

    # Extract batch size from input_shape
    df["batch_size_from_shape"] = df["input_shape"].apply(
        lambda x: int(x.split("torch.Size([")[1].split(",")[0])
    )

    # Extract model type and kernel size from model_name
    df["model_type"] = df["model_name"].apply(
        lambda x: "custom" if "custom" in x else "basic"
    )
    df["kernel_size"] = df["model_name"].apply(
        lambda x: int(x.split("k")[1].split("_")[0])
    )

    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: GPU Latency by Model Type and Kernel Size (for batch_size=1)
    batch_1_data = df[df["batch_size_from_shape"] == 1]

    kernel_sizes = sorted(df["kernel_size"].unique())
    model_types = ["basic", "custom"]

    width = 0.35
    x = np.arange(len(kernel_sizes))

    for i, model_type in enumerate(model_types):
        latencies = []
        for ks in kernel_sizes:
            data = batch_1_data[
                (batch_1_data["model_type"] == model_type)
                & (batch_1_data["kernel_size"] == ks)
            ]
            latencies.append(data[key].mean() if len(data) > 0 else 0)

        ax1.bar(x + i * width, latencies, width, label=model_type, alpha=0.7)

    ax1.set_xlabel("Kernel Size")
    ax1.set_ylabel(f"{label_dev} Latency (ms)")
    ax1.set_title(f"{label_dev} Latency by Model Type and Kernel Size (Batch Size=1)")
    ax1.set_xticks(x + width / 2)
    ax1.set_xticklabels([f"K{ks}" for ks in kernel_sizes])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Batch Size Scaling for Different Model Types
    batch_sizes = sorted(df["batch_size_from_shape"].unique())

    for model_type in model_types:
        for ks in kernel_sizes:
            latencies = []
            for bs in batch_sizes:
                data = df[
                    (df["model_type"] == model_type)
                    & (df["kernel_size"] == ks)
                    & (df["batch_size_from_shape"] == bs)
                ]
                if len(data) > 0:
                    latencies.append(data[key].mean())
                else:
                    latencies.append(np.nan)

            label = f"{model_type} K{ks}"
            ax2.plot(batch_sizes, latencies, "o-", label=label, markersize=5)

    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel(f"{label_dev} Latency (ms)")
    ax2.set_title("Batch Size Scaling by Model Type and Kernel Size")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Custom vs Basic Speedup Ratio
    speedup_data = []

    for bs in batch_sizes:
        for ks in kernel_sizes:
            custom_data = df[
                (df["model_type"] == "custom")
                & (df["kernel_size"] == ks)
                & (df["batch_size_from_shape"] == bs)
            ]
            basic_data = df[
                (df["model_type"] == "basic")
                & (df["kernel_size"] == ks)
                & (df["batch_size_from_shape"] == bs)
            ]

            if len(custom_data) > 0 and len(basic_data) > 0:
                custom_lat = custom_data[key].iloc[0]
                basic_lat = basic_data[key].iloc[0]
                speedup_ratio = custom_lat / basic_lat
                speedup_data.append(
                    {"batch_size": bs, "kernel_size": ks, "speedup": speedup_ratio}
                )

    speedup_df = pd.DataFrame(speedup_data)

    for ks in kernel_sizes:
        ks_data = speedup_df[speedup_df["kernel_size"] == ks]
        if len(ks_data) > 0:
            ax3.plot(
                ks_data["batch_size"],
                ks_data["speedup"],
                "s-",
                label=f"K{ks}",
                markersize=6,
                linewidth=2,
            )

    ax3.axhline(y=1, color="red", linestyle="--", alpha=0.5, label="Equal Performance")
    ax3.set_xlabel("Batch Size")
    ax3.set_ylabel("Speedup Ratio (Custom / Basic)")
    ax3.set_title(
        "Custom vs Basic Performance Ratio\n>1 = Custom slower, <1 = Custom faster"
    )
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Performance Heatmap
    pivot_data = df.pivot_table(
        values=key,
        index="kernel_size",
        columns=["model_type", "batch_size_from_shape"],
        aggfunc="mean",
    )

    im = ax4.imshow(pivot_data.values, cmap="YlOrRd", aspect="auto")

    # Set labels
    ax4.set_xticks(range(len(pivot_data.columns)))
    ax4.set_yticks(range(len(pivot_data.index)))

    # Create column labels
    col_labels = []
    for col in pivot_data.columns:
        model_type, bs = col
        col_labels.append(f"{model_type}\nbs{bs}")

    ax4.set_xticklabels(col_labels, rotation=45, ha="right")
    ax4.set_yticklabels([f"K{ks}" for ks in pivot_data.index])
    ax4.set_title(f"{label_dev} Latency Heatmap (ms)")
    ax4.set_xlabel("Model Type × Batch Size")
    ax4.set_ylabel("Kernel Size")

    # Add colorbar
    plt.colorbar(im, ax=ax4, label="Latency (ms)")

    # Add text annotations
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            text = ax4.text(
                j,
                i,
                f"{pivot_data.iloc[i, j]:.1f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("=== Performance Summary ===")
    for model_type in model_types:
        model_data = df[df["model_type"] == model_type]
        avg_gpu = model_data["latency_gpu_mean_ms"].mean()
        avg_cpu = model_data["latency_cpu_ms"].mean()
        print(f"{model_type.upper()} Models: GPU={avg_gpu:.2f}ms, CPU={avg_cpu:.2f}ms")

    print(f"\n=== Kernel Size Impact ===")
    for ks in kernel_sizes:
        ks_data = df[df["kernel_size"] == ks]
        avg_latency = ks_data[key].mean()
        print(f"Kernel Size {ks}: {avg_latency:.2f}ms average {label_dev} latency")

    print(f"\n=== Batch Size Impact ===")
    for bs in batch_sizes:
        bs_data = df[df["batch_size_from_shape"] == bs]
        avg_latency = bs_data[key].mean()
        print(f"Batch Size {bs}: {avg_latency:.2f}ms average {label_dev} latency")



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast

def setup_plotting():
    """Initialize plotting style"""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

def extract_model_type(model_name):
    """Extract whether model has ReplacedConv and precision type from model name"""
    has_replaced_conv = 'ReplacedConv' in model_name
    is_half_model = model_name.endswith('_half')
    base_name = model_name.replace('ReplacedConv_', '').replace('_half', '')
    return has_replaced_conv, is_half_model, base_name

def plot_latency_vs_batch_size(df):
    """Plot latency vs batch size for different precisions and model types"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # GPU Latency
    for (precision, replaced_conv), group in df.groupby(['precision', 'has_replaced_conv']):
        label = f"{precision} {'ReplacedConv' if replaced_conv else 'Standard'}"
        ax1.plot(group['batch_size'], group['latency_gpu_mean_ms'], 
                marker='o', label=label, linewidth=2)
    
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('GPU Latency (ms)')
    ax1.set_title('GPU Latency vs Batch Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # CPU Latency (if available)
    cpu_data = df.dropna(subset=['latency_cpu_ms'])
    if not cpu_data.empty:
        for (precision, replaced_conv), group in cpu_data.groupby(['precision', 'has_replaced_conv']):
            label = f"{precision} {'ReplacedConv' if replaced_conv else 'Standard'}"
            ax2.plot(group['batch_size'], group['latency_cpu_ms'], 
                    marker='s', label=label, linewidth=2)
        
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('CPU Latency (ms)')
        ax2.set_title('CPU Latency vs Batch Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_accuracy_vs_latency(df):
    """Plot accuracy vs latency trade-off"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(df['precision'].unique())))
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, (precision, precision_group) in enumerate(df.groupby('precision')):
        marker = markers[i % len(markers)]
        
        for j, (replaced_conv, model_group) in enumerate(precision_group.groupby('has_replaced_conv')):
            color = colors[i]
            label = f"{precision} {'ReplacedConv' if replaced_conv else 'Standard'}"
            
            ax1.scatter(model_group['latency_gpu_mean_ms'], model_group['val_iou'],
                       c=[color], marker=marker, s=100, label=label, alpha=0.7)
            ax2.scatter(model_group['latency_gpu_mean_ms'], model_group['test_iou'],
                       c=[color], marker=marker, s=100, label=label, alpha=0.7)
    
    ax1.set_xlabel('GPU Latency (ms)')
    ax1.set_ylabel('Validation IoU')
    ax1.set_title('Validation Accuracy vs Latency')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('GPU Latency (ms)')
    ax2.set_ylabel('Test IoU')
    ax2.set_title('Test Accuracy vs Latency')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_accuracy_vs_latency_cpu(df):
    """Plot accuracy vs latency trade-off"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(df['precision'].unique())))
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, (precision, precision_group) in enumerate(df.groupby('precision')):
        marker = markers[i % len(markers)]
        
        for j, (replaced_conv, model_group) in enumerate(precision_group.groupby('has_replaced_conv')):
            color = colors[i]
            label = f"{precision} {'ReplacedConv' if replaced_conv else 'Standard'}"
            
            ax1.scatter(model_group['latency_cpu_ms'], model_group['val_iou'],
                       c=[color], marker=marker, s=100, label=label, alpha=0.7)
            ax2.scatter(model_group['latency_cpu_ms'], model_group['test_iou'],
                       c=[color], marker=marker, s=100, label=label, alpha=0.7)
    
    ax1.set_xlabel('CPU Latency (ms)')
    ax1.set_ylabel('Validation IoU')
    ax1.set_title('Validation Accuracy vs Latency')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('CPU Latency (ms)')
    ax2.set_ylabel('Test IoU')
    ax2.set_title('Test Accuracy vs Latency')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_precision_comparison(df, metric='latency_gpu_mean_ms'):
    """Compare different precisions for the same model type"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    comparisons = [
        ('Standard Conv', False),
        ('ReplacedConv', True)
    ]
    
    for idx, (title, replaced_conv) in enumerate(comparisons):
        if idx >= len(axes):
            break
            
        model_data = df[df['has_replaced_conv'] == replaced_conv]
        
        # Latency comparison
        sns.barplot(data=model_data, x='precision', y='latency_gpu_mean_ms', 
                   hue='batch_size', ax=axes[idx])
        axes[idx].set_title(f'{title} - GPU Latency by Precision')
        axes[idx].set_ylabel('Latency (ms)')
        axes[idx].tick_params(axis='x', rotation=45)
        
        # Accuracy comparison
        if idx + 2 < len(axes):
            sns.barplot(data=model_data, x='precision', y='test_iou',
                       hue='batch_size', ax=axes[idx + 2])
            axes[idx + 2].set_title(f'{title} - Test IoU by Precision')
            axes[idx + 2].set_ylabel('Test IoU')
            axes[idx + 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def plot_latency_std_deviation(df):
    """Plot latency standard deviation across different configurations"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create combined labels for x-axis
    df['config'] = df['precision'] + '_' + df['batch_size'].astype(str) + '_' + df['has_replaced_conv'].astype(str)
    
    # Plot mean latency with error bars
    x_pos = range(len(df))
    ax.errorbar(x_pos, df['latency_gpu_mean_ms'], 
                yerr=df['latency_gpu_std_ms'], 
                fmt='o', capsize=5, capthick=2,
                label='Mean ± Std Dev')
    
    ax.set_xlabel('Configuration (Precision_BatchSize_ReplacedConv)')
    ax.set_ylabel('GPU Latency (ms)')
    ax.set_title('GPU Latency with Standard Deviation')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df['config'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig

def create_summary_table(df):
    """Create a summary table of the benchmark results"""
    summary = df.groupby(['has_replaced_conv', 'precision', 'batch_size']).agg({
        'latency_gpu_mean_ms': 'mean',
        'latency_gpu_std_ms': 'mean',
        'val_iou': 'mean',
        'test_iou': 'mean'
    }).round(4)
    
    return summary

# Usage example:
def analyze_benchmark_data(csv_file_path):
    """Main function to load and analyze benchmark data"""
    # Load data
    df = pd.read_csv(csv_file_path)
    
    # Data preprocessing
    df['has_replaced_conv'] = df['model_name'].apply(lambda x: 'ReplacedConv' in x)
    df['is_half_model'] = df['model_name'].apply(lambda x: x.endswith('_half'))
    
    # Fix potential typo
    df['precision'] = df['precision'].replace('fp16-halp', 'fp16-half')
    
    # Set up plotting
    setup_plotting()
    
    # Create visualizations
    plots = {}
    
    plots['latency_vs_batch'] = plot_latency_vs_batch_size(df)
    plots['accuracy_vs_latency'] = plot_accuracy_vs_latency(df)

    plots['accuracy_vs_latency_cpu'] = plot_accuracy_vs_latency_cpu(df)
    plots['precision_comparison'] = plot_precision_comparison(df)
    plots['latency_std'] = plot_latency_std_deviation(df)
    
    # Create summary table
    summary = create_summary_table(df)
    
    return plots, summary, df

# Example usage:
# plots, summary, processed_df = analyze_benchmark_data('your_benchmark_data.csv')