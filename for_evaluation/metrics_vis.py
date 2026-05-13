import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from pathlib import Path

# ==============================================================================
# 🎨 Global Configuration Area (Control what to plot here)
# ==============================================================================

PLOT_CONFIG = {
    # 1. Color Palette (20 colors cycling)
    'COLOR_PALETTE': [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173', 
        '#5254a3', '#8c1b1b', '#e7969c', '#6b6ecf', '#ce6dbd'
    ],

    # 2. Specify metrics to plot (whitelist mode)
    'METRICS_TO_PLOT': {
        'CLIENT': [
            # --- Core Metrics ---
            'completion_rate',              # Completion Rate (%)
            'average_turns_to_target',      # Average Turns
            'humanness_mean',               # Humanness (mean)
            'trust_rationality_mean',       # Trust Rationality (mean)
            'emotion_rationality_mean',     # Emotion Rationality (mean)
            'cooperation_rationality_mean', # Cooperation Rationality (mean)
            'violation_mean',               # Violation Level (mean)
        ],
        
        'CHATBOT': [
            # --- Core Metrics ---
            'completion_rate',              # Completion Rate (%)
            'average_turns_to_target',      # Average Turns
            'profiling_accuracy',           # Profiling Accuracy (mean)
            'stage_completeness_rate',      # Stage Completeness Rate (%)
            'trust_mean',                   # Trust Level (mean)
            'cooperation_mean',             # Cooperation Level (mean)
            'emotion_mean',                 # Emotion Value (mean)

            # --- Variation Metrics ---
            'trust_variation_mean',         # Trust Variation (mean)
            'cooperation_variation_mean',   # Cooperation Variation (mean)
            'emotion_variation_mean',       # Emotion Variation (mean)
        ]
    }
}

# ==============================================================================

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

class CustomizableLogParser:
    def __init__(self, input_dir: str, step_start: int = 1, step_end: int = 11):
        """
        Initialize parser
        
        Args:
            input_dir: Input directory path
            step_start: Starting step (inclusive), default is 1
            step_end: Ending step (inclusive), default is 11
        """
        self.input_dir = Path(input_dir)
        self.step_start = step_start
        self.step_end = step_end
        self.aggregate_size = 4  # Default value, will be adjusted based on actual situation
        self.colors = PLOT_CONFIG['COLOR_PALETTE']
        self.detected_batch_size = None  # Store detected batch size
        
    def detect_batch_size(self, prefix: str) -> int:
        """
        Detect batch size by analyzing step information in filenames
        """
        pattern = str(self.input_dir / f"{prefix}_val_metric_step*.jsonl")
        files = glob.glob(pattern)
        
        if len(files) < 2:
            return 1  # Default to 1 if less than 2 files
        
        # Extract all step values within specified range
        steps = []
        for file_path in files:
            try:
                # Extract step number from filename
                filename = Path(file_path).stem
                step_str = filename.split('step')[-1]
                step = int(step_str)
                
                # Only consider steps within range
                if self.step_start <= step <= self.step_end:
                    steps.append(step)
            except:
                continue
        
        if len(steps) < 2:
            return 1
        
        # Calculate differences between consecutive steps
        steps.sort()
        diffs = [steps[i+1] - steps[i] for i in range(len(steps)-1)]
        
        # Use most common difference as batch size
        if diffs:
            batch_size = max(set(diffs), key=diffs.count)
            return batch_size
        
        return 1
        
    def load_data(self, prefix: str) -> pd.DataFrame:
        """
        Load data, only reading files within specified step range
        """
        pattern = str(self.input_dir / f"{prefix}_val_metric_step*.jsonl")
        all_files = glob.glob(pattern)
        
        if not all_files:
            print(f"⚠️  No {prefix} data files found: {pattern}")
            return pd.DataFrame()
        
        # Filter files within range
        files = []
        for file_path in all_files:
            try:
                filename = Path(file_path).stem
                step_str = filename.split('step')[-1]
                step = int(step_str)
                
                if self.step_start <= step <= self.step_end:
                    files.append(file_path)
            except:
                continue
        
        if not files:
            print(f"⚠️  No {prefix} data files found in step range [{self.step_start}, {self.step_end}]")
            return pd.DataFrame()
        
        data_list = []
        print(f"[{prefix.upper()}] Found {len(files)} files in range [step {self.step_start} - {self.step_end}], loading...")
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip(): continue
                        data = json.loads(line)
                        raw_metrics = data.get('metrics', {})
                        entry = {'step': data.get('step')}
                        
                        # Extract all numeric metrics
                        for k, v in raw_metrics.items():
                            if isinstance(v, (int, float)):
                                # Smart percentage conversion
                                if ('rate' in k or 'accuracy' in k) and -1.0 <= v <= 1.0:
                                    entry[k] = v * 100
                                else:
                                    entry[k] = v
                        # Add top-level metrics
                        if 'num_trajectories' in data:
                            entry['num_trajectories'] = data['num_trajectories']
                            
                        data_list.append(entry)
            except Exception as e:
                print(f"⚠️  Error reading {file_path}: {e}")
                
        df = pd.DataFrame(data_list)
        if not df.empty:
            df = df.sort_values('step').reset_index(drop=True)
            df['seq_id'] = range(1, len(df) + 1)
            
            available_metrics = [c for c in df.columns if c not in ['step', 'seq_id']]
            print(f"✅ [{prefix.upper()}] Data loaded successfully, {len(df)} records total.")
            
        return df

    def aggregate_steps(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Only aggregate when batch > 1"""
        if df_raw.empty: 
            return pd.DataFrame()
        
        # If batch_size = 1, return original data directly (with necessary columns added)
        if self.detected_batch_size == 1:
            df_result = df_raw.copy()
            df_result['aggregated_point'] = df_result['seq_id']
            return df_result
        
        # Aggregate when batch > 1
        df_raw['group_id'] = (df_raw['seq_id'] - 1) // self.aggregate_size
        df_agg = df_raw.groupby('group_id').mean(numeric_only=True)
        df_agg['aggregated_point'] = df_agg.index + 1
        return df_agg

    def _add_trend_line(self, ax, x, y, color):
        if len(x) > 1:
            try:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                slope = z[0]
                trend_symbol = "↗" if slope > 0 else "↘"
                ax.plot(x, p(x), linestyle='--', color=color, alpha=0.9, 
                       linewidth=2, label=f'Trend {trend_symbol} ({slope:.3f})')
            except Exception:
                pass

    def plot_selected_metrics(self, df_raw: pd.DataFrame, df_agg: pd.DataFrame, prefix: str, output_dir: str):
        if df_raw.empty: return
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. Get configuration list
        target_metrics = PLOT_CONFIG['METRICS_TO_PLOT'].get(prefix.upper(), [])
        
        # 2. Strict filtering: only plot metrics that exist in data
        valid_metrics = []
        for m in target_metrics:
            if m in df_raw.columns:
                valid_metrics.append(m)
            else:
                print(f"⚠️  [{prefix.upper()}] Metric '{m}' not found in data, skipped.")
        
        if not valid_metrics:
            print(f"❌ [{prefix.upper()}] No valid metrics to plot. Please check PLOT_CONFIG configuration.")
            return
        
        num_metrics = len(valid_metrics)
        batch_info = f"(Batch Size = {self.detected_batch_size})"
        step_range_info = f"[Step {self.step_start}-{self.step_end}]"
        print(f"📊 [{prefix.upper()}] Plotting {num_metrics} metrics... {batch_info} {step_range_info}")

        # 3. Dynamic layout
        cols = 3
        rows = math.ceil(num_metrics / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
        title_text = f'{prefix.upper()} Metrics Analysis {batch_info} {step_range_info}'
        fig.suptitle(title_text, fontsize=20, fontweight='bold', y=1.02)
        
        axes_flat = axes.flatten() if num_metrics > 1 else [axes]
        x_raw = df_raw['seq_id']
        
        # Decide plotting method based on batch size
        if self.detected_batch_size == 1:
            # Batch = 1: Only plot one line, no aggregation display
            x_plot = df_agg['aggregated_point']
            for i, metric in enumerate(valid_metrics):
                ax = axes_flat[i]
                color = self.colors[i % len(self.colors)]
                
                # Only plot main line
                ax.plot(x_plot, df_agg[metric], 'o-', color=color, 
                       linewidth=2.5, markersize=6, label='Data')
                # Trend Line
                self._add_trend_line(ax, x_plot, df_agg[metric], color)
                
                title = metric.replace('_', ' ').title()
                ax.set_title(title, fontweight='bold', fontsize=12, color='black')
                ax.grid(True, alpha=0.3)
                
                if i == 0:
                    ax.legend(loc='best', frameon=True)
        else:
            # Batch > 1: Display raw data + aggregated data
            x_agg_mapped = df_agg['aggregated_point'] * self.aggregate_size - (self.aggregate_size / 2) + 0.5
            
            for i, metric in enumerate(valid_metrics):
                ax = axes_flat[i]
                color = self.colors[i % len(self.colors)]
                
                # Raw Data
                ax.plot(x_raw, df_raw[metric], 'o-', color='gray', alpha=0.2, 
                       linewidth=1, markersize=3, label='Raw Step')
                # Aggregated Data
                ax.plot(x_agg_mapped, df_agg[metric], 'D-', color=color, 
                       linewidth=2.5, markersize=8, label='Aggregated')
                # Trend Line
                self._add_trend_line(ax, x_agg_mapped, df_agg[metric], color)
                
                title = metric.replace('_', ' ').title()
                ax.set_title(title, fontweight='bold', fontsize=12, color='black')
                ax.grid(True, alpha=0.3)
                
                if i == 0:
                    ax.legend(loc='best', frameon=True)

        # Hide extra subplots
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].axis('off')
            
        plt.tight_layout()
        
        # Include step range in filename
        save_path = f'{output_dir}/{prefix}_metrics_chart_step{self.step_start}-{self.step_end}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Chart saved: {save_path}")

    def save_reports(self, client_data, chatbot_data, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Include step range in filename
        excel_path = f'{output_dir}/full_metrics_report_step{self.step_start}-{self.step_end}.xlsx'
        txt_path = f'{output_dir}/full_metrics_report_step{self.step_start}-{self.step_end}.txt'
        
        # Save Excel
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            if chatbot_data['agg'] is not None:
                chatbot_data['agg'].to_excel(writer, sheet_name='Chatbot Aggregated', index=False)
            if client_data['agg'] is not None:
                client_data['agg'].to_excel(writer, sheet_name='Client Aggregated', index=False)
            if chatbot_data['raw'] is not None:
                chatbot_data['raw'].to_excel(writer, sheet_name='Chatbot Raw', index=False)
            if client_data['raw'] is not None:
                client_data['raw'].to_excel(writer, sheet_name='Client Raw', index=False)
        print(f"✓ Excel report saved: {excel_path}")
        
        # Save TXT
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\nFULL TRAINING METRICS REPORT\n" + "=" * 100 + "\n")
            f.write(f"Step Range: [{self.step_start}, {self.step_end}]\n")
            f.write(f"Detected Batch Size: {self.detected_batch_size}\n")
            f.write("=" * 100 + "\n\n")
            
            for name, data in [('CHATBOT', chatbot_data), ('CLIENT', client_data)]:
                f.write(f">>> {name} METRICS <<<\n")
                if data['agg'] is not None:
                    df = data['agg']
                    cols_to_show = [c for c in df.columns if c not in ['group_id', 'seq_id', 'aggregated_point']]
                    f.write("\n[Aggregated Data Summary (First 10 Rows)]\n")
                    f.write(df[cols_to_show].head(10).to_string(index=False))
                    f.write("\n\n[Statistical Analysis]\n")
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    numeric_cols = [c for c in numeric_cols if c not in ['group_id', 'seq_id', 'aggregated_point']]
                    f.write(df[numeric_cols].describe().round(4).to_string())
                    f.write("\n" + "-"*80 + "\n\n")
        print(f"✓ TXT report saved: {txt_path}")

    def run(self, output_dir: str = './outputs/evaluation/report'):
        print("="*60)
        print("Customizable Metrics Parser (Adaptive Batch Mode)")
        print(f"📍 Step Range: [{self.step_start}, {self.step_end}]")
        print("="*60)
        
        results = {'client': {'raw': None, 'agg': None}, 'chatbot': {'raw': None, 'agg': None}}
        
        # First detect batch size (using either client or chatbot)
        self.detected_batch_size = self.detect_batch_size('client')
        if self.detected_batch_size == 1:
            # If client detection is 1, check chatbot as well
            chatbot_batch = self.detect_batch_size('chatbot')
            if chatbot_batch > 1:
                self.detected_batch_size = chatbot_batch
        
        print(f"🔍 Detected Batch Size = {self.detected_batch_size}")
        
        # Set aggregation size based on batch size
        if self.detected_batch_size > 1:
            self.aggregate_size = self.detected_batch_size
            print(f"📦 Will aggregate every {self.aggregate_size} steps")
        else:
            print(f"📦 Batch Size = 1, no aggregation, plotting raw data directly")
        
        print("-" * 60)
        
        df_client = self.load_data('client')
        if not df_client.empty:
            df_client_agg = self.aggregate_steps(df_client)
            self.plot_selected_metrics(df_client, df_client_agg, 'client', output_dir)
            results['client']['raw'] = df_client
            results['client']['agg'] = df_client_agg
            
        print("-" * 30)
        
        df_chatbot = self.load_data('chatbot')
        if not df_chatbot.empty:
            df_chatbot_agg = self.aggregate_steps(df_chatbot)
            self.plot_selected_metrics(df_chatbot, df_chatbot_agg, 'chatbot', output_dir)
            results['chatbot']['raw'] = df_chatbot
            results['chatbot']['agg'] = df_chatbot_agg

        print("-" * 30)
        self.save_reports(results['client'], results['chatbot'], output_dir)
        print("\n✅ All tasks completed!")

if __name__ == "__main__":
    # ========== Usage Example ==========
    # Modify these paths to match your directory structure
    input_directory = "./outputs/evaluation/" 
    output_directory = "./outputs/evaluation/report"
    
    # ========== 🎯 Set step range here ==========
    STEP_START = 1    # Starting step (inclusive)
    STEP_END = 11      # Ending step (inclusive)
    # ============================================
    
    parser = CustomizableLogParser(
        input_dir=input_directory,
        step_start=STEP_START,
        step_end=STEP_END
    )
    parser.run(output_dir=output_directory)