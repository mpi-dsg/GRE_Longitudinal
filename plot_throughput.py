#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_throughput_comparison(csv_file='out.csv'):
    """
    Create a throughput comparison plot similar to your example
    """
    try:
        # Load data from CSV file
        df = pd.read_csv(csv_file)
        print(f"Loaded data from {csv_file}")
        print("Columns:", df.columns.tolist())
        print("\nData preview:")
        print(df.head())
        
        # Determine the appropriate columns for plotting
        # Adjust these based on your actual CSV structure
        x_column = None
        y_column = 'throughput'  # Default to throughput
        group_column = None
        
        # Try to find the right columns
        possible_x = ['init_table_size', 'table_size', 'operations_num', 'thread_num', 'error_bound']
        possible_group = ['index_type', 'index']
        
        for col in possible_x:
            if col in df.columns and len(df[col].unique()) > 1:
                x_column = col
                break
        
        for col in possible_group:
            if col in df.columns:
                group_column = col
                break
        
        # If throughput doesn't exist, try other performance metrics
        if 'throughput' not in df.columns:
            possible_y = ['latency', 'ops_per_sec', 'performance', 'time']
            for col in possible_y:
                if col in df.columns:
                    y_column = col
                    break
        
        print(f"\nUsing columns: X={x_column}, Y={y_column}, Group={group_column}")
        
        if not x_column or y_column not in df.columns:
            print("Could not determine appropriate columns for plotting!")
            return
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        if group_column and len(df[group_column].unique()) > 1:
            # Multiple index types - create pivot and plot lines
            try:
                pivot_df = df.pivot(index=x_column, columns=group_column, values=y_column)
                
                for index_type in pivot_df.columns:
                    if not pivot_df[index_type].isna().all():
                        plt.plot(pivot_df.index, pivot_df[index_type], 
                                marker='o', linewidth=2, markersize=8, label=index_type)
                
                plt.legend(title='Index Type', loc='best')
            except:
                # If pivot fails, plot individual lines
                for index_type in df[group_column].unique():
                    subset = df[df[group_column] == index_type]
                    if len(subset) > 0:
                        plt.plot(subset[x_column], subset[y_column], 
                                marker='o', linewidth=2, markersize=8, label=str(index_type))
                plt.legend(title='Index Type', loc='best')
        else:
            # Single series - scatter plot with line
            plt.plot(df[x_column], df[y_column], 
                    marker='o', linewidth=2, markersize=8, color='blue')
        
        # Customize the plot
        plt.xlabel(x_column.replace('_', ' ').title())
        plt.ylabel(y_column.replace('_', ' ').title())
        
        # Create a descriptive title
        title = f'{y_column.replace("_", " ").title()} vs {x_column.replace("_", " ").title()}'
        if 'thread_num' in df.columns:
            threads = df['thread_num'].iloc[0]
            title += f' ({threads} threads)'
        if 'operations_num' in df.columns:
            ops = df['operations_num'].iloc[0]
            title += f' - {ops:,} operations'
        
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        output_file = csv_file.replace('.csv', '_throughput_plot.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as: {output_file}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the CSV file exists and has the right format.")

if __name__ == "__main__":
    csv_file = 'out.csv'
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    plot_throughput_comparison(csv_file)
