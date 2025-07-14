#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def visualize_benchmark_results(csv_file='out.csv'):
    """
    Visualize benchmark results from CSV file
    """
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        print("Make sure to run the benchmark script first and enable CSV output.")
        return
    
    try:
        # Load data from CSV file
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} rows from {csv_file}")
        print("Columns available:", df.columns.tolist())
        print("\nFirst few rows:")
        print(df.head())
        
        # Identify key columns for plotting
        # Adjust these column names based on your actual CSV structure
        possible_x_cols = ['init_table_size', 'table_size', 'operations_num', 'thread_num']
        possible_y_cols = ['throughput', 'latency', 'memory_usage']
        possible_group_cols = ['index_type', 'index']
        
        # Find actual column names
        x_col = None
        y_col = None
        group_col = None
        
        for col in possible_x_cols:
            if col in df.columns:
                x_col = col
                break
        
        for col in possible_y_cols:
            if col in df.columns:
                y_col = col
                break
                
        for col in possible_group_cols:
            if col in df.columns:
                group_col = col
                break
        
        print(f"\nUsing: X-axis={x_col}, Y-axis={y_col}, Grouping={group_col}")
        
        if not all([x_col, y_col]):
            print("Error: Could not identify X and Y columns for plotting")
            print("Available columns:", df.columns.tolist())
            return
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Benchmark Results Analysis ({len(df)} data points)', fontsize=16)
        
        # Plot 1: Main performance metric vs parameter
        ax1 = axes[0, 0]
        if group_col and len(df[group_col].unique()) > 1:
            # Multiple index types - line plot
            for idx_type in df[group_col].unique():
                subset = df[df[group_col] == idx_type]
                if len(subset) > 0:
                    ax1.plot(subset[x_col], subset[y_col], marker='o', label=str(idx_type))
            ax1.legend(title=group_col.replace('_', ' ').title())
        else:
            # Single index type - scatter plot
            ax1.scatter(df[x_col], df[y_col], alpha=0.7)
        
        ax1.set_xlabel(x_col.replace('_', ' ').title())
        ax1.set_ylabel(y_col.replace('_', ' ').title())
        ax1.set_title(f'{y_col.title()} vs {x_col.title()}')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Bar chart if we have categorical data
        ax2 = axes[0, 1]
        if group_col:
            group_stats = df.groupby(group_col)[y_col].mean()
            bars = ax2.bar(range(len(group_stats)), group_stats.values)
            ax2.set_xticks(range(len(group_stats)))
            ax2.set_xticklabels(group_stats.index, rotation=45)
            ax2.set_ylabel(f'Average {y_col.replace("_", " ").title()}')
            ax2.set_title(f'Average {y_col.title()} by Index Type')
            
            # Add value labels on bars
            for bar, value in zip(bars, group_stats.values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.1f}', ha='center', va='bottom')
        else:
            ax2.text(0.5, 0.5, 'No grouping column found', ha='center', va='center', transform=ax2.transAxes)
        
        # Plot 3: Distribution histogram
        ax3 = axes[1, 0]
        ax3.hist(df[y_col], bins=20, alpha=0.7, edgecolor='black')
        ax3.set_xlabel(y_col.replace('_', ' ').title())
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'Distribution of {y_col.title()}')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary statistics
        summary_stats = df[y_col].describe()
        stats_text = f"""
Summary Statistics for {y_col.title()}:

Count: {summary_stats['count']:.0f}
Mean: {summary_stats['mean']:.2f}
Std: {summary_stats['std']:.2f}
Min: {summary_stats['min']:.2f}
25%: {summary_stats['25%']:.2f}
50%: {summary_stats['50%']:.2f}
75%: {summary_stats['75%']:.2f}
Max: {summary_stats['max']:.2f}
        """
        
        if group_col:
            stats_text += f"\n\nBy {group_col.title()}:\n"
            for idx_type in df[group_col].unique():
                subset_mean = df[df[group_col] == idx_type][y_col].mean()
                stats_text += f"{idx_type}: {subset_mean:.2f}\n"
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save the plot
        output_file = csv_file.replace('.csv', '_visualization.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved as: {output_file}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        print("Make sure the CSV file has proper headers and data.")

def main():
    csv_file = 'out.csv'
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    print(f"Visualizing results from: {csv_file}")
    visualize_benchmark_results(csv_file)

if __name__ == "__main__":
    main()
