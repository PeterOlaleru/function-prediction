import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

# Data for startup landscape
data = {
    'Startup Idea': ['Dark Proteome (Pharma)', 'Industrial Enzymes (SynBio)', 'VUS Classifier (Health)', 'Proteomics API (SaaS)'],
    'Regulatory Friction': [9, 4, 8, 2],  # 1-10 Scale
    'Technical Difficulty': [8, 7, 9, 5], # 1-10 Scale (relative to CAFA baseline)
    'Market Size ($B)': [50, 15, 8, 2]    # Bubble size
}

df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Create scatter plot
scatter = sns.scatterplot(
    data=df, 
    x='Regulatory Friction', 
    y='Technical Difficulty', 
    size='Market Size ($B)', 
    sizes=(200, 2000), 
    hue='Startup Idea',
    palette='viridis',
    alpha=0.7
)

# Labels and formatting
plt.title('Post-CAFA Startup Landscape', fontsize=14, fontweight='bold')
plt.xlabel('Regulatory Friction (Time to Revenue)', fontsize=12)
plt.ylabel('Technical Difficulty (Beyond CAFA)', fontsize=12)
plt.xlim(0, 10)
plt.ylim(0, 10)

# Annotate points
for i in range(df.shape[0]):
    plt.text(
        df['Regulatory Friction'][i]+0.2, 
        df['Technical Difficulty'][i], 
        df['Startup Idea'][i], 
        fontsize=10,
        weight='bold'
    )

# Quadrant lines
plt.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=5, color='gray', linestyle='--', alpha=0.5)

# Quadrant labels
plt.text(1, 9, 'High Tech / High Reg\n(Deep Tech)', color='gray')
plt.text(8, 1, 'Low Tech / High Reg\n(Compliance)', color='gray')
plt.text(1, 1, 'Low Tech / Low Reg\n(SaaS/Tools)', color='gray')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.tight_layout()

# Save the plot
output_path = Path('docs/startup_landscape.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_path}")
