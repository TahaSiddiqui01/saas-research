"""Chart generation utilities for agents."""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, List
from pathlib import Path


class ChartGenerator:
    """Generate charts and save them as images."""
    
    def __init__(self, output_dir: str = "output/charts"):
        """Initialize chart generator with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_bar_chart(
        self,
        data: Dict[str, float],
        title: str,
        xlabel: str,
        ylabel: str,
        filename: str
    ) -> str:
        """Create a bar chart from dictionary data."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        keys = list(data.keys())
        values = list(data.values())
        
        bars = ax.bar(keys, values, color='steelblue', alpha=0.7)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Rotate x-axis labels if needed
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom')
        
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def create_pie_chart(
        self,
        data: Dict[str, float],
        title: str,
        filename: str
    ) -> str:
        """Create a pie chart from dictionary data."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        keys = list(data.keys())
        values = list(data.values())
        
        ax.pie(values, labels=keys, autopct='%1.1f%%', startangle=90)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def create_line_chart(
        self,
        data: Dict[str, List[float]],
        title: str,
        xlabel: str,
        ylabel: str,
        filename: str
    ) -> str:
        """Create a line chart from dictionary data."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for label, values in data.items():
            ax.plot(range(len(values)), values, marker='o', label=label, linewidth=2)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def create_comparison_chart(
        self,
        categories: List[str],
        data_series: Dict[str, List[float]],
        title: str,
        xlabel: str,
        ylabel: str,
        filename: str
    ) -> str:
        """Create a grouped bar chart for comparison."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = range(len(categories))
        width = 0.8 / len(data_series)
        
        for i, (label, values) in enumerate(data_series.items()):
            offset = (i - len(data_series) / 2) * width + width / 2
            ax.bar([xi + offset for xi in x], values, width, label=label, alpha=0.7)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)

