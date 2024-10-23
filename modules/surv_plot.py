import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

## Discrete Numerics = PassengerId, SibSp, Parch
## Nominal = Sex, Embarked, Cabin, Ticket, Survived
## Ordinal = Pclass
## Continuous = Age, Fare

class sv_plot:
    def __init__(self, data):
        self.data = data

    def sv_countplot(self, features):
        """
        Create survival analysis countplots for specified features
        Args:
            features: list of column names to analyze
        """
        n_rows = (len(features) + 1) // 2  # 2 features per row
        
        # Create subplots
        fig, axes = plt.subplots(n_rows, 2, figsize=(12, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        # Create plots for each feature
        for idx, feature in enumerate(features):
            # Create countplot
            sns.countplot(data=self.data, 
                        x=feature,
                        hue='Survived',
                        ax=axes[idx],
                        palette='viridis')
            
            # Add title and labels
            axes[idx].set_title(f'Survival Distribution by {feature}')
            axes[idx].set_xlabel(feature)
            axes[idx].set_ylabel('Count')
            
            # Add percentage labels on bars
            total = len(self.data[feature])
            for p in axes[idx].patches:
                percentage = f'{100 * p.get_height() / total:.1f}%'
                x = p.get_x() + p.get_width() / 2
                y = p.get_height()
                axes[idx].annotate(percentage, (x, y), ha='center')
            
            # Rotate x-labels if they're long
            if self.data[feature].nunique() > 3:
                axes[idx].tick_params(axis='x', rotation=45)
                
            # Add legend with better labels
            axes[idx].legend(title='Survival Status', 
                            labels=['Did Not Survive', 'Survived'])
        
        # Remove empty subplots if any
        for idx in range(len(features), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.show()