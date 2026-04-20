import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def plot_average_metrics(df_combined, output_dir):
    """Plot the average MAE and pixel accuracy thresholds comparison across models."""
    # Filter rows containing 'Average' for each model
    df_avg = df_combined[df_combined['Object'] == 'Average'].copy()
    
    # 1. Bar chart for Average MAE comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_avg, x='Model', y='Mean_Angular_Error', hue='Model', palette='viridis', legend=False)
    plt.title('Average Mean Angular Error (Lower is Better)', fontsize=14, fontweight='bold')
    plt.ylabel('MAE (Degrees)', fontsize=12)
    plt.xlabel('Models', fontsize=12)
    
    for index, value in enumerate(df_avg['Mean_Angular_Error']):
        plt.text(index, value + 0.1, f'{value:.2f}', ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_mae_comparison.png'), dpi=300)
    plt.close()

    # 2. Line chart for percentage of accurate pixels comparison
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['Pixels_under_10', 'Pixels_under_20', 'Pixels_under_30']
    
    for model in df_avg['Model'].unique():
        model_data = df_avg[df_avg['Model'] == model][metrics_to_plot].values.flatten()
        plt.plot(['< 10°', '< 20°', '< 30°'], model_data, marker='o', linewidth=2, label=model)

    plt.title('Percentage of Accurate Pixels (Higher is Better)', fontsize=14, fontweight='bold')
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.xlabel('Error Threshold', fontsize=12)
    plt.legend(title='Models')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pixel_accuracy_comparison.png'), dpi=300)
    plt.close()

def plot_per_object_mae(df_combined, output_dir):
    """Plot a grouped bar chart comparing MAE per object."""
    # Exclude 'Average' rows to plot detailed per-object metrics
    df_objects = df_combined[df_combined['Object'] != 'Average']

    plt.figure(figsize=(14, 7))
    sns.barplot(data=df_objects, x='Object', y='Mean_Angular_Error', hue='Model', palette='Set2')
    
    plt.title('Per-Object Mean Angular Error Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('MAE (Degrees)', fontsize=12)
    plt.xlabel('Test Objects', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Models')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_object_mae.png'), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate statistics and plots from metric CSVs.")
    parser.add_argument("--csv_files", nargs='+', required=True, help="List of CSV files to process")
    parser.add_argument("--model_names", nargs='+', default=None, help="Custom names for models (must match number of csv files)")
    parser.add_argument("--out_dir", type=str, default="./output_stats", help="Directory to save plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model_names = args.model_names if args.model_names else [os.path.basename(f).replace('.csv', '') for f in args.csv_files]
    
    if len(args.csv_files) != len(model_names):
        print("ERROR: Number of --csv_files must match number of --model_names")
        return

    # Read and merge all CSV files into a single DataFrame
    dfs = []
    for file, name in zip(args.csv_files, model_names):
        print(f"Reading: {file}")
        df = pd.read_csv(file)
        df['Model'] = name  # Add a column for model identification
        dfs.append(df)
    
    df_combined = pd.concat(dfs, ignore_index=True)

    print("\nGenerating plots...")
    plot_average_metrics(df_combined, args.out_dir)
    plot_per_object_mae(df_combined, args.out_dir)

    print(f"Plots have been saved to: {args.out_dir}/")

if __name__ == "__main__":
    main()