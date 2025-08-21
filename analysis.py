# analysis_data_prep.py
"""
Generate occurrence-level analysis log from image-level inference results.

This script aggregates image-level flowering predictions to occurrence-level 
predictions and combines with geographic and temporal data for trend analysis.

Logic: An occurrence is classified as flowering if ANY of its images is predicted as flowering.
Probability: Uses maximum probability across all images for an occurrence.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


def load_image_inference_log(log_path):
    """Load the image-level inference results"""
    print(f"Loading image inference log from: {log_path}")
    df = pd.read_csv(log_path)
    print(f"Loaded {len(df)} image predictions")
    print(f"Unique occurrences: {df['gbifID'].nunique()}")
    return df


def load_occurrence_data(occurrence_path):
    """Load occurrence data with geographic and temporal information"""
    print(f"Loading occurrence data from: {occurrence_path}")
    
    # Define columns we need for analysis
    columns_needed = [
        'gbifID', 'eventDate', 'year', 'month', 'day',
        'decimalLatitude', 'decimalLongitude', 
        'countryCode', 'stateProvince', 'county', 'locality',
        'continent', 'elevation'
    ]
    
    # Read the tab-separated file
    df = pd.read_csv(occurrence_path, sep='\t', low_memory=False)
    print(f"Loaded {len(df)} occurrence records")
    
    # Select only needed columns that exist
    available_columns = [col for col in columns_needed if col in df.columns]
    missing_columns = [col for col in columns_needed if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
    
    df_filtered = df[available_columns].copy()
    print(f"Selected {len(available_columns)} relevant columns")
    
    return df_filtered


def aggregate_to_occurrence_level(image_df):
    """
    Aggregate image-level predictions to occurrence-level
    
    Logic: FL if any image is FL, otherwise Non-FL
    Probability: Max probability across all images
    """
    print("Aggregating image predictions to occurrence level...")
    
    # Group by gbifID and aggregate
    occurrence_agg = image_df.groupby('gbifID').agg({
        'predicted_label': lambda x: 'FL' if 'FL' in x.values else 'Non-FL',
        'predicted_prob': ['max', 'mean', 'min', 'count']
    }).round(4)
    
    # Flatten column names
    occurrence_agg.columns = [
        'predicted_label_occurrence',
        'max_prob_flowering', 
        'mean_prob_flowering',
        'min_prob_flowering',
        'num_images'
    ]
    
    # Reset index to make gbifID a column
    occurrence_agg = occurrence_agg.reset_index()
    
    print(f"Aggregated to {len(occurrence_agg)} occurrences")
    print(f"Flowering occurrences: {(occurrence_agg['predicted_label_occurrence'] == 'FL').sum()}")
    print(f"Non-flowering occurrences: {(occurrence_agg['predicted_label_occurrence'] == 'Non-FL').sum()}")
    
    return occurrence_agg


def merge_with_occurrence_data(predictions_df, occurrence_df):
    """Merge predictions with geographic and temporal data"""
    print("Merging predictions with occurrence metadata...")
    
    # Ensure gbifID is same type in both dataframes
    predictions_df['gbifID'] = predictions_df['gbifID'].astype(str)
    occurrence_df['gbifID'] = occurrence_df['gbifID'].astype(str)
    
    # Merge on gbifID
    merged_df = predictions_df.merge(
        occurrence_df, 
        on='gbifID', 
        how='left'
    )
    
    print(f"Merged data: {len(merged_df)} records")
    print(f"Records with missing occurrence data: {merged_df.isnull().any(axis=1).sum()}")
    
    return merged_df


def clean_temporal_data(df):
    """Clean and standardize temporal data"""
    print("Cleaning temporal data...")
    
    # Convert year, month, day to numeric
    for col in ['year', 'month', 'day']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create a clean date column where possible
    if all(col in df.columns for col in ['year', 'month', 'day']):
        # Create date only where we have complete information
        date_mask = df[['year', 'month', 'day']].notna().all(axis=1)
        df.loc[date_mask, 'observation_date'] = pd.to_datetime(
            df.loc[date_mask, ['year', 'month', 'day']]
        ).dt.date
    
    # Create decade column for trend analysis
    if 'year' in df.columns:
        df['decade'] = (df['year'] // 10) * 10
    
    return df


def clean_geographic_data(df):
    """Clean and validate geographic data"""
    print("Cleaning geographic data...")
    
    # Convert coordinates to numeric
    if 'decimalLatitude' in df.columns:
        df['decimalLatitude'] = pd.to_numeric(df['decimalLatitude'], errors='coerce')
    if 'decimalLongitude' in df.columns:
        df['decimalLongitude'] = pd.to_numeric(df['decimalLongitude'], errors='coerce')
    
    # Flag records with valid coordinates
    if all(col in df.columns for col in ['decimalLatitude', 'decimalLongitude']):
        coord_mask = df[['decimalLatitude', 'decimalLongitude']].notna().all(axis=1)
        df['has_coordinates'] = coord_mask
        print(f"Records with valid coordinates: {coord_mask.sum()}")
    
    return df


def generate_summary_stats(df):
    """Generate summary statistics for the analysis dataset"""
    print("\n" + "="*60)
    print("ANALYSIS DATASET SUMMARY")
    print("="*60)
    
    # Basic counts
    print(f"Total occurrences: {len(df)}")
    print(f"Flowering occurrences: {(df['predicted_label_occurrence'] == 'FL').sum()}")
    print(f"Flowering percentage: {(df['predicted_label_occurrence'] == 'FL').mean()*100:.1f}%")
    
    # Temporal coverage
    if 'year' in df.columns:
        year_range = df['year'].dropna()
        if len(year_range) > 0:
            print(f"\nTemporal coverage:")
            print(f"  Year range: {year_range.min():.0f} - {year_range.max():.0f}")
            print(f"  Records with year data: {len(year_range)}")
    
    # Geographic coverage
    if 'has_coordinates' in df.columns:
        coord_records = df['has_coordinates'].sum()
        print(f"\nGeographic coverage:")
        print(f"  Records with coordinates: {coord_records}")
        print(f"  Countries represented: {df['countryCode'].nunique() if 'countryCode' in df.columns else 'N/A'}")
    
    # Image counts
    if 'num_images' in df.columns:
        print(f"\nImage statistics:")
        print(f"  Total images analyzed: {df['num_images'].sum()}")
        print(f"  Average images per occurrence: {df['num_images'].mean():.1f}")
        print(f"  Max images per occurrence: {df['num_images'].max()}")
    
    # Probability distribution
    if 'max_prob_flowering' in df.columns:
        print(f"\nProbability distribution:")
        print(f"  Mean max probability: {df['max_prob_flowering'].mean():.3f}")
        print(f"  Median max probability: {df['max_prob_flowering'].median():.3f}")
        print(f"  Min max probability: {df['max_prob_flowering'].min():.3f}")
        print(f"  Max max probability: {df['max_prob_flowering'].max():.3f}")


def main():
    """Main processing pipeline"""
    print("Arabidopsis Flowering Analysis Data Preparation")
    print("="*50)
    
    # File paths
    image_log_path = "image_occurrence_log.csv"
    occurrence_path = "Picture Stuff/occurrence.txt"
    output_path = "occurrence_analysis_log.csv"
    
    try:
        # Step 1: Load image inference results
        image_df = load_image_inference_log(image_log_path)
        
        # Step 2: Load occurrence metadata
        occurrence_df = load_occurrence_data(occurrence_path)
        
        # Step 3: Aggregate image predictions to occurrence level
        predictions_df = aggregate_to_occurrence_level(image_df)
        
        # Step 4: Merge with occurrence data
        analysis_df = merge_with_occurrence_data(predictions_df, occurrence_df)
        
        # Step 5: Clean temporal data
        analysis_df = clean_temporal_data(analysis_df)
        
        # Step 6: Clean geographic data
        analysis_df = clean_geographic_data(analysis_df)
        
        # Step 7: Reorder columns for analysis
        column_order = [
            'gbifID', 'predicted_label_occurrence', 'max_prob_flowering',
            'num_images', 'year', 'month', 'day', 'decade', 
            'decimalLatitude', 'decimalLongitude', 'has_coordinates',
            'countryCode', 'stateProvince', 'county', 'locality', 'continent',
            'eventDate', 'elevation', 'mean_prob_flowering', 'min_prob_flowering'
        ]
        
        # Keep only columns that exist
        existing_columns = [col for col in column_order if col in analysis_df.columns]
        additional_columns = [col for col in analysis_df.columns if col not in column_order]
        
        final_columns = existing_columns + additional_columns
        analysis_df = analysis_df[final_columns]
        
        # Step 8: Save the analysis dataset
        analysis_df.to_csv(output_path, index=False)
        print(f"\nAnalysis dataset saved to: {output_path}")
        
        # Step 9: Generate summary statistics
        generate_summary_stats(analysis_df)
        
        print(f"\n✅ Analysis data preparation completed successfully!")
        print(f"📊 Ready for trend analysis with {len(analysis_df)} occurrence records")
        
        return analysis_df
        
    except FileNotFoundError as e:
        print(f"❌ Error: Required file not found - {e}")
        print("Make sure the following files exist:")
        print(f"  - {image_log_path}")
        print(f"  - {occurrence_path}")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    analysis_df = main()
