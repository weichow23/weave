# Copyright (c) 2025 WEAVE Team
# SPDX-License-Identifier: Apache-2.0

import os
import json
import argparse
from collections import defaultdict
from config import WEAVE_DOMAIN

def analyze_data(args):
    try:
        # Initialize data structures to store metrics
        domain_scores = defaultdict(lambda: defaultdict(lambda: {"total": 0, "count": 0}))
        all_metrics = defaultdict(lambda: {"total": 0, "count": 0})
        
        # Process the metrics file line by line to handle JSONL format
        with open(args.metrics_file, 'r') as file:
            line_number = 0
            for line in file:
                line_number += 1
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                    
                    # Extract domain information
                    domain = item.get("domain", "unknown")
                    
                    # Find all possible scoring categories and process turn scores
                    for key in item:
                        if key.startswith("turn ") and isinstance(item[key], dict):
                            for category, category_data in item[key].items():
                                if isinstance(category_data, dict) and "score" in category_data:
                                    score = category_data["score"]
                                    domain_scores[domain][category]["total"] += score
                                    domain_scores[domain][category]["count"] += 1
                                    all_metrics[category]["total"] += score
                                    all_metrics[category]["count"] += 1
                                    
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse JSON at line {line_number}")
        
        # Calculate averages and prepare output
        results = {}
        
        # Initialize category results
        category_results = {
            "Science": {"categories": {}, "domains": {}, "AVG": "No data"},
            "Creation": {"categories": {}, "domains": {}, "AVG": "No data"},
            "Logic": {"categories": {}, "domains": {}, "AVG": "No data"},
            "Game": {"categories": {}, "domains": {}, "AVG": "No data"}
        }
        
        # Track totals for each main category
        category_totals = {cat: {"total": 0, "count": 0} for cat in category_results}
        overall_metrics = {}
        
        for metric, data in all_metrics.items():
            if data["count"] > 0:
                avg_score = data["total"] / data["count"]
                # Divide by 10 before storing
                overall_metrics[metric] = round(avg_score / 10, 3)
            else:
                overall_metrics[metric] = "No data"
        
        # Define the function for calculating the weighted average
        def calculate_weighted_avg(metrics_dict):
            # Check if the required metrics are included.
            has_key_metrics = all(k in metrics_dict for k in ["key_point", "visual_consistency", "image_quality"])
            has_accuracy = "accuracy" in metrics_dict
            
            if not has_key_metrics:
                # If the required key metrics are not available, return the average value.
                total = sum(score for score in metrics_dict.values() if isinstance(score, (int, float)))
                count = sum(1 for score in metrics_dict.values() if isinstance(score, (int, float)))
                if count > 0:
                    return round(total / count, 3)
                return "No data"
                
            # Ensure all required metrics are numerical.
            for key in ["key_point", "visual_consistency", "image_quality"]:
                if not isinstance(metrics_dict.get(key), (int, float)):
                    return "Incomplete data"
            
            if has_accuracy and not isinstance(metrics_dict.get("accuracy"), (int, float)):
                has_accuracy = False
                
            # Use different weights based on whether there is accuracy.
            if has_accuracy:
                # Weighting when accuracy is present
                weights = {
                    "key_point": 0.4,
                    "visual_consistency": 0.1,
                    "image_quality": 0.2,
                    "accuracy": 0.3
                }
                weighted_sum = sum(metrics_dict[k] * weights[k] for k in weights.keys() if k in metrics_dict)
                return round(weighted_sum, 3)
            else:
                # Weighting when there is no accuracy
                weights = {
                    "key_point": 0.5,
                    "visual_consistency": 0.2,
                    "image_quality": 0.3
                }
                weighted_sum = sum(metrics_dict[k] * weights[k] for k in weights.keys())
                return round(weighted_sum, 3)
        
        # Calculate the overall weighted average
        overall_avg = calculate_weighted_avg(overall_metrics)
        
        # Process domain scores and map to main categories
        for domain, categories in domain_scores.items():
            results[domain] = {"categories": {}, "AVG": 0}
            domain_metrics = {}
            
            # Find which main category this domain belongs to
            main_category = "Other"  # Default if not found
            for cat, domains in WEAVE_DOMAIN.items():
                if domain in domains:
                    main_category = cat
                    break
            
            # Process each scoring category
            for category, data in categories.items():
                if data["count"] > 0:
                    avg_score = data["total"] / data["count"]
                    # Divide by 10 before storing
                    score_value = round(avg_score / 10, 3)
                    results[domain]["categories"][category] = score_value
                    domain_metrics[category] = score_value
                    
                    # Add to main category totals if it belongs to one
                    if main_category in category_results:
                        if category not in category_results[main_category]["categories"]:
                            category_results[main_category]["categories"][category] = {"total": 0, "count": 0}
                        
                        category_results[main_category]["categories"][category]["total"] += data["total"]
                        category_results[main_category]["categories"][category]["count"] += data["count"]
                        
                        # Add to category totals for overall average
                        category_totals[main_category]["total"] += data["total"]
                        category_totals[main_category]["count"] += data["count"]
                else:
                    results[domain]["categories"][category] = "No data"
                    domain_metrics[category] = "No data"
            
            # Calculate domain-wide weighted average
            results[domain]["AVG"] = calculate_weighted_avg(domain_metrics)
            
            # Add domain avg to its main category if it's a valid number
            if isinstance(results[domain]["AVG"], (int, float)) and main_category in category_results and main_category not in ["Other"]:
                category_results[main_category]["domains"][domain] = results[domain]["AVG"]
        
        # Calculate the weighted average for the main categories
        for category_name, data in category_results.items():
            category_metrics = {}
            
            for metric_name, metric_value in data["categories"].items():
                if isinstance(metric_value, dict) and "total" in metric_value and "count" in metric_value and metric_value["count"] > 0:
                    category_metrics[metric_name] = round(metric_value["total"] / metric_value["count"] / 10, 3)
                elif isinstance(metric_value, (int, float)):
                    category_metrics[metric_name] = metric_value
            
            data["AVG"] = calculate_weighted_avg(category_metrics)
            
            for cat, cat_data in data["categories"].items():
                if isinstance(cat_data, dict) and "count" in cat_data and cat_data["count"] > 0:
                    # Divide by 10 before storing
                    data["categories"][cat] = round(cat_data["total"] / cat_data["count"] / 10, 3)
                elif not isinstance(cat_data, (int, float)):
                    data["categories"][cat] = "No data"
        
        final_results = {
            "domains": results,
            "categories": category_results,
            "overall": {
                "metrics": overall_metrics,
                "AVG": overall_avg
            }
        }
        
        # Save results to JSON file
        output_dir = os.path.dirname(args.metrics_file)
        output_file = os.path.join(output_dir, 'result.jsonl')
        with open(output_file, "w") as outfile:
            json.dump(final_results, outfile, indent=4)
        
        # Print results for main categories
        print("\nMain Category Scores:")
        print("=====================")
        
        for category, data in category_results.items():
            print(f"\nCategory: {category}")
            print("-" * (len(category) + 11))
            
            # Print domains in this category with their averages
            print("Domains:")
            if data["domains"]:
                for domain, score in data["domains"].items():
                    print(f"  {domain}: {score}")
            else:
                print()  # Empty line if no domains
            
            # Print category-specific metrics
            print("\nMetrics:")
            if data["categories"]:
                for metric, score in data["categories"].items():
                    if isinstance(score, (int, float)):
                        print(f"  {metric}: {score:.3f}")
                    else:
                        print(f"  {metric}: {score}")
            else:
                print()  # Empty line if no metrics
            
            # Print category AVG with note about weighting
            print(f"\nCategory AVG (Weighted): {data['AVG']}")
            print("-" * 30)
            
        # Print results for individual domains
        print("\nIndividual Domain Scores:")
        print("========================")
        
        for domain, data in results.items():
            print(f"\nDomain: {domain}")
            print("-" * (len(domain) + 9))
            
            for category, score in data["categories"].items():
                if isinstance(score, (int, float)):
                    print(f"{category}: {score:.3f}")
                else:
                    print(f"{category}: {score}")
            
            # Print domain AVG with note about weighting
            print(f"AVG (Weighted): {data['AVG']}")
        
        print("\nOverall Metrics Scores:")
        print("======================")
        
        for metric, score in overall_metrics.items():
            if isinstance(score, (int, float)):
                print(f"{metric}: {score:.3f}")
            else:
                print(f"{metric}: {score}")
        
        print(f"\nOverall AVG (Weighted): {overall_avg}")
        print("\nWeights used:")
        print("=============")
        print("Without 'accuracy':")
        print("  key_point: 50%")
        print("  visual_consistency: 20%")
        print("  image_quality: 30%")
        print("\nWith 'accuracy':")
        print("  key_point: 40%")
        print("  visual_consistency: 10%")
        print("  image_quality: 20%")
        print("  accuracy: 30%")
    
    except Exception as e:
        print(f"Error processing data: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize WEAVE evaluation results with weighted averages")
    parser.add_argument('--metrics_file', type=str, required=True, 
                        help='Path to the rover_metrics.jsonl file')
    
    args = parser.parse_args()
    analyze_data(args)