import matplotlib.pyplot as plt
import numpy as np

def burnout_trend_chart(df):
    """Create enhanced burnout trend visualization"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data
    if 'burnout_score' not in df.columns:
        return fig
    
    scores = df['burnout_score'].values
    dates = range(len(scores))
    
    # Create gradient color based on risk level
    colors = []
    for score in scores:
        if score >= 70:
            colors.append('#DC2626')  # Red for high risk
        elif score >= 40:
            colors.append('#F59E0B')  # Orange for medium risk
        else:
            colors.append('#10B981')  # Green for low risk
    
    # Plot line
    ax.plot(dates, scores, color='#3B82F6', linewidth=2, alpha=0.7, label='Burnout Score')
    
    # Plot points with risk-based colors
    ax.scatter(dates, scores, c=colors, s=100, edgecolors='white', linewidth=2, zorder=5)
    
    # Fill under curve with gradient
    ax.fill_between(dates, scores, alpha=0.2, color='#3B82F6')
    
    # Add threshold lines
    ax.axhline(y=70, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='High Risk (70%)')
    ax.axhline(y=40, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='Moderate Risk (40%)')
    
    # Customize appearance
    ax.set_title('Burnout Risk Trend Over Time', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Session Number', fontsize=12)
    ax.set_ylabel('Burnout Risk (%)', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 105)
    
    # Add legend
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Add value labels on last point
    if len(scores) > 0:
        last_score = scores[-1]
        ax.annotate(f'{last_score}%', 
                   xy=(dates[-1], last_score), 
                   xytext=(10, 0), 
                   textcoords='offset points',
                   fontsize=11,
                   fontweight='bold',
                   color='#1E40AF')
    
    plt.tight_layout()
    return fig