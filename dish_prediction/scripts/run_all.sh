#!/bin/bash
# Master script to run complete dish prediction analysis

echo "================================================================================"
echo "DISH DEMAND PREDICTION - COMPLETE ANALYSIS"
echo "================================================================================"
echo ""

# Activate Python environment if needed
# source env/bin/activate

# Run complete pipeline
echo "Running complete pipeline..."
python src/pipeline.py --top-n 10

# Generate visualizations
echo ""
echo "Generating all visualizations..."
python src/visualization/generate_all_figures.py

echo ""
echo "================================================================================"
echo "✅ ANALYSIS COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  - data/processed/model_results.csv"
echo "  - reports/figures/*.png"
echo "  - reports/RESULTS.md"
echo ""
echo "================================================================================"
