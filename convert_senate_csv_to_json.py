#!/usr/bin/env python3
"""
Convert Senate CSV ground truth to JSON labels format for PIE framework.
Extracts 6 key fields: birthdate, gender, race_ethnicity, committee_roles, religion, education
"""

import csv
import json
import os
import sys

def parse_education(education_str):
    """Parse education JSON string from CSV"""
    if not education_str or education_str.strip() == '':
        return []
    try:
        # Education comes as JSON array string in CSV (with escaped quotes)
        return json.loads(education_str)
    except:
        return []

def convert_csv_to_json(csv_path, output_path):
    """
    Convert Senate CSV ground truth to JSON labels format.
    
    Args:
        csv_path: Path to CSV file
        output_path: Path to output JSON file
    """
    labels = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            senator_id = row['senator_id']
            
            # Extract only the 6 fields we need
            label_entry = {
                'birthdate': row.get('birthdate', '').strip() or None,
                'gender': row.get('gender', '').strip() or None,
                'race_ethnicity': row.get('race_ethnicity', '').strip() or None,
                'committee_roles': row.get('committee_roles', '').strip() or None,
                'religion': row.get('religion', '').strip() or None,
                'education': parse_education(row.get('education', ''))
            }
            
            labels[senator_id] = label_entry
            count += 1
    
    # Write to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2)
    
    print(f"✓ Converted {count} senator records to {output_path}")
    return count

if __name__ == '__main__':
    csv_path = './external_data/ground_truth/senate_ground_truth_updated_manual.csv'
    output_path = './data/senator/labels.json'
    
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found at {csv_path}")
        sys.exit(1)
    
    count = convert_csv_to_json(csv_path, output_path)
    print(f"✓ JSON labels file created successfully at {output_path}")
    print(f"✓ Ready for {count} senator profiles")
