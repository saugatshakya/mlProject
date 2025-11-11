"""
Dish Recommendation System - Analysis & Comparison
===================================================

This script compares the original implementation with the app_v2 implementation
and generates visualizations showing:
- Performance metrics
- Association rules quality
- Co-occurrence patterns
- Top recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add paths
sys.path.append('../app_v2')
sys.path.append('src')

from models.recommender import DishRecommender
from data.preprocessing import DishOrderPreprocessor

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def load_data():
    """Load the original data."""
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    
    data_path = Path('../data/data.csv')
    preprocessor = DishOrderPreprocessor()
    df = preprocessor.load_data(str(data_path))
    
    # Create transactions
    transactions = preprocessor.create_transactions(df, status_filter='Delivered')
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total orders: {len(df):,}")
    print(f"   Delivered orders: {len(transactions):,}")
    print(f"   Unique dishes: {len(preprocessor.dish_names):,}")
    
    # Calculate order statistics
    order_sizes = [len(t) for t in transactions]
    print(f"   Avg items/order: {np.mean(order_sizes):.2f}")
    print(f"   Multi-item orders: {sum(1 for s in order_sizes if s > 1):,} ({sum(1 for s in order_sizes if s > 1)/len(order_sizes):.1%})")
    
    return transactions, preprocessor


def train_original_model(transactions):
    """Train the original recommender model."""
    print("\n" + "="*60)
    print("TRAINING ORIGINAL MODEL")
    print("="*60)
    
    recommender = DishRecommender()
    recommender.fit(
        transactions=transactions,
        min_support=0.001,  # 0.1% - appears in at least 21 orders
        min_confidence=0.1   # 10%
    )
    
    return recommender


def analyze_association_rules(recommender):
    """Analyze quality of association rules."""
    print("\n" + "="*60)
    print("ASSOCIATION RULES ANALYSIS")
    print("="*60)
    
    if not recommender.association_rules:
        print("⚠️  No association rules generated")
        return
    
    df_rules = pd.DataFrame(recommender.association_rules)
    
    print(f"\n📈 Rules Statistics:")
    print(f"   Total rules: {len(df_rules):,}")
    print(f"   Avg confidence: {df_rules['confidence'].mean():.2%}")
    print(f"   Avg lift: {df_rules['lift'].mean():.2f}x")
    print(f"   Max lift: {df_rules['lift'].max():.2f}x")
    
    # Top rules
    print(f"\n🏆 Top 10 Rules by Lift:")
    top_rules = df_rules.nlargest(10, 'lift')[['antecedent', 'consequent', 'confidence', 'lift', 'count']]
    for i, row in top_rules.iterrows():
        print(f"   {i+1}. {row['antecedent'][:30]:<30} → {row['consequent'][:30]:<30} "
              f"(lift={row['lift']:.1f}x, conf={row['confidence']:.1%})")
    
    return df_rules


def create_visualizations(recommender, df_rules, output_dir='docs/figures'):
    """Generate comparison visualizations."""
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Confidence Distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1a. Confidence histogram
    axes[0, 0].hist(df_rules['confidence'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Confidence')
    axes[0, 0].set_ylabel('Number of Rules')
    axes[0, 0].set_title('Distribution of Rule Confidence')
    axes[0, 0].axvline(df_rules['confidence'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df_rules["confidence"].mean():.2%}')
    axes[0, 0].legend()
    
    # 1b. Lift histogram
    axes[0, 1].hist(df_rules['lift'], bins=30, color='coral', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Lift')
    axes[0, 1].set_ylabel('Number of Rules')
    axes[0, 1].set_title('Distribution of Rule Lift')
    axes[0, 1].axvline(df_rules['lift'].mean(), color='red', linestyle='--',
                       label=f'Mean: {df_rules["lift"].mean():.2f}x')
    axes[0, 1].legend()
    
    # 1c. Support histogram
    axes[1, 0].hist(df_rules['support']*100, bins=30, color='mediumseagreen', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Support (%)')
    axes[1, 0].set_ylabel('Number of Rules')
    axes[1, 0].set_title('Distribution of Rule Support')
    axes[1, 0].axvline(df_rules['support'].mean()*100, color='red', linestyle='--',
                       label=f'Mean: {df_rules["support"].mean():.2%}')
    axes[1, 0].legend()
    
    # 1d. Confidence vs Lift scatter
    axes[1, 1].scatter(df_rules['confidence'], df_rules['lift'], alpha=0.5, s=20)
    axes[1, 1].set_xlabel('Confidence')
    axes[1, 1].set_ylabel('Lift')
    axes[1, 1].set_title('Confidence vs Lift')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / '01_association_rules_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()
    
    # 2. Top Dishes by Support
    fig, ax = plt.subplots(figsize=(12, 8))
    
    support_df = pd.DataFrame([
        {'dish': dish, 'support': support}
        for dish, support in recommender.dish_support.items()
    ]).sort_values('support', ascending=False).head(20)
    
    ax.barh(range(len(support_df)), support_df['support']*100, color='steelblue', alpha=0.8)
    ax.set_yticks(range(len(support_df)))
    ax.set_yticklabels(support_df['dish'], fontsize=9)
    ax.set_xlabel('Support (% of orders)')
    ax.set_title('Top 20 Most Popular Dishes', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(support_df['support']*100):
        ax.text(v + 0.2, i, f'{v:.1f}%', va='center', fontsize=8)
    
    plt.tight_layout()
    output_path = output_dir / '02_top_dishes_support.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()
    
    # 3. Top Recommendations Example
    example_dishes = ['bageecha pizza', 'bone in jamaican grilled chicken', 
                      'peri peri fries', 'chilli cheese garlic bread']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, dish in enumerate(example_dishes):
        if dish in recommender.cooccurrence_matrix:
            recs = recommender.recommend(dish, top_n=10)
            
            if not recs.empty:
                axes[idx].barh(range(len(recs)), recs['confidence']*100, 
                              color='coral', alpha=0.8)
                axes[idx].set_yticks(range(len(recs)))
                axes[idx].set_yticklabels(recs['recommended_dish'].str[:35], fontsize=8)
                axes[idx].set_xlabel('Confidence (%)')
                axes[idx].set_title(f'Recommendations for: {dish.title()}', 
                                   fontsize=10, fontweight='bold')
                axes[idx].invert_yaxis()
                axes[idx].grid(True, axis='x', alpha=0.3)
                
                # Add value labels
                for i, v in enumerate(recs['confidence']*100):
                    axes[idx].text(v + 1, i, f'{v:.1f}%', va='center', fontsize=7)
    
    plt.tight_layout()
    output_path = output_dir / '03_example_recommendations.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()
    
    # 4. Implementation Comparison Table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    comparison_data = [
        ['Feature', 'Original (dish_recommend)', 'App V2 (app_v2)'],
        ['Data Parsing', 'Regex "\\d+ x Dish"', 'Comma/semicolon split'],
        ['Normalization', 'Lowercase, strip special chars', 'Basic strip()'],
        ['Min Support', '0.001 (0.1%)', '0.001 (0.1%)'],
        ['Min Confidence', '0.10 (10%)', '0.10 (10%)'],
        ['Association Rules', f'{len(df_rules):,} rules', 'Same algorithm'],
        ['Co-occurrence', 'Symmetric matrix', 'Symmetric matrix'],
        ['Recommendation', 'By lift & confidence', 'By lift & confidence'],
        ['Output Format', 'DataFrame', 'JSON/Dict'],
        ['Model Saving', 'CSV files (3)', 'Pickle file (1)'],
        ['API Interface', 'recommend(dish, top_n)', 'get_recommendations(dish)'],
        ['Filtering', 'Status filter (Delivered)', 'All transactions'],
    ]
    
    table = ax.table(cellText=comparison_data, cellLoc='left', loc='center',
                    colWidths=[0.25, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(comparison_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F2F2F2')
    
    ax.set_title('Implementation Comparison: Original vs App V2', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = output_dir / '04_implementation_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()
    
    # 5. Performance Summary
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    metrics_data = [
        ['Metric', 'Value'],
        ['Total Transactions', f'{recommender.total_transactions:,}'],
        ['Unique Dishes', f'{len(recommender.dish_support):,}'],
        ['Association Rules', f'{len(df_rules):,}'],
        ['Avg Confidence', f'{df_rules["confidence"].mean():.2%}'],
        ['Avg Lift', f'{df_rules["lift"].mean():.2f}x'],
        ['Max Lift', f'{df_rules["lift"].max():.2f}x'],
        ['Min Support Threshold', '0.1%'],
        ['Min Confidence Threshold', '10%'],
    ]
    
    table = ax.table(cellText=metrics_data, cellLoc='left', loc='center',
                    colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#70AD47')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(metrics_data)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F2F2F2')
    
    ax.set_title('Model Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = output_dir / '05_performance_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def main():
    """Run complete analysis and comparison."""
    print("\n" + "="*60)
    print("DISH RECOMMENDATION SYSTEM - ANALYSIS & COMPARISON")
    print("="*60)
    
    # Load data
    transactions, preprocessor = load_data()
    
    # Train model
    recommender = train_original_model(transactions)
    
    # Analyze rules
    df_rules = analyze_association_rules(recommender)
    
    if df_rules is not None and not df_rules.empty:
        # Create visualizations
        create_visualizations(recommender, df_rules)
    
    # Save model (optional)
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    recommender.save_model('models')
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nGenerated visualizations in: docs/figures/")
    print(f"Saved model files in: models/")


if __name__ == '__main__':
    main()
