# analysis_data_prep.py
"""
Generate occurrence-level analysis log from image-level inference results.

This script aggregates image-level flowering predictions to occurrence-level 
predictions and combines with geographic and temporal data for trend analysis.
An occurrence is classified as flowering if any of its images has flowers present 
and none of its images have siliques present

Probability: Uses maximum probability across all images for an occurrence for both 
flower and silique components separately.
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


def load_multimedia_data(multimedia_path):
    """Load multimedia data with image URLs"""
    print(f"Loading multimedia data from: {multimedia_path}")
    df = pd.read_csv(multimedia_path)
    print(f"Loaded {len(df)} multimedia records")
    
    # Keep only the relevant columns
    multimedia_clean = df[['gbifID', 'identifier']].copy()
    multimedia_clean = multimedia_clean.rename(columns={'identifier': 'image_url'})
    
    # Remove rows with empty URLs
    multimedia_clean = multimedia_clean.dropna(subset=['image_url'])
    multimedia_clean = multimedia_clean[multimedia_clean['image_url'].str.strip() != '']
    
    # Create sequential image numbers for each gbifID (1, 2, 3, etc.)
    multimedia_clean['image_sequence'] = multimedia_clean.groupby('gbifID').cumcount() + 1
    
    # Create image filename from gbifID and sequential number for matching
    multimedia_clean['image_filename'] = (
        multimedia_clean['gbifID'].astype(str) + '_' + 
        multimedia_clean['image_sequence'].astype(str) + '.jpg'
    )
    
    print(f"Processed multimedia data for {multimedia_clean['gbifID'].nunique()} unique occurrences")
    print(f"Total valid URLs: {len(multimedia_clean)}")
    
    return multimedia_clean


def merge_image_urls(image_df, multimedia_df):
    """Merge image URLs with inference results"""
    print("Merging image URLs with inference results...")
    
    # Merge on image_filename to get URLs
    merged_df = image_df.merge(
        multimedia_df[['image_filename', 'image_url']], 
        on='image_filename', 
        how='left'
    )
    
    # Check merge success
    url_matches = merged_df['image_url'].notna().sum()
    print(f"Successfully matched URLs for {url_matches}/{len(merged_df)} images ({url_matches/len(merged_df)*100:.1f}%)")
    
    if url_matches < len(merged_df):
        missing_urls = len(merged_df) - url_matches
        print(f"Warning: {missing_urls} images without URL matches")
    
    return merged_df


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
    
    NEW Logic: FL if ANY image has flowers present AND NONE have siliques present
    Otherwise: Non-FL
    """
    print("Aggregating image predictions to occurrence level...")
    print("Using new flowering definition: ANY flowers present AND NO siliques present")
    
    def determine_flowering_status(group):
        """Determine flowering status for an occurrence based on new rule"""
        has_any_flowers = group['flower_present'].any()  # Any image has flowers
        has_any_siliques = group['silique_present'].any()  # Any image has siliques
        
        # Flowering = has flowers AND no siliques
        is_flowering = has_any_flowers and not has_any_siliques
        return 'FL' if is_flowering else 'Non-FL'
    
    # Group by gbifID and aggregate
    occurrence_agg = image_df.groupby('gbifID').agg({
        'flower_probability': ['max', 'mean', 'min'],
        'silique_probability': ['max', 'mean', 'min'],
        'flower_present': ['any', 'sum'],  # any=has any flowers, sum=count of flower images
        'silique_present': ['any', 'sum'],  # any=has any siliques, sum=count of silique images
        'predicted_label': 'count'  # count = number of images
    }).round(4)
    
    # Flatten column names
    occurrence_agg.columns = [
        'max_flower_prob', 'mean_flower_prob', 'min_flower_prob',
        'max_silique_prob', 'mean_silique_prob', 'min_silique_prob', 
        'has_any_flowers', 'num_flower_images',
        'has_any_siliques', 'num_silique_images',
        'num_images'
    ]
    
    # Reset index to make gbifID a column
    occurrence_agg = occurrence_agg.reset_index()
    
    # Apply new flowering logic
    occurrence_agg['predicted_label_occurrence'] = occurrence_agg.apply(
        lambda row: 'FL' if (row['has_any_flowers'] and not row['has_any_siliques']) else 'Non-FL',
        axis=1
    )
    
    print(f"Aggregated to {len(occurrence_agg)} occurrences")
    print(f"Flowering occurrences (new rule): {(occurrence_agg['predicted_label_occurrence'] == 'FL').sum()}")
    print(f"Non-flowering occurrences: {(occurrence_agg['predicted_label_occurrence'] == 'Non-FL').sum()}")
    
    # Additional statistics for the new logic
    print(f"\nBreakdown by components:")
    print(f"  Occurrences with ANY flowers: {occurrence_agg['has_any_flowers'].sum()}")
    print(f"  Occurrences with ANY siliques: {occurrence_agg['has_any_siliques'].sum()}")
    print(f"  Occurrences with flowers AND siliques: {(occurrence_agg['has_any_flowers'] & occurrence_agg['has_any_siliques']).sum()}")
    print(f"  Occurrences with NEITHER: {(~occurrence_agg['has_any_flowers'] & ~occurrence_agg['has_any_siliques']).sum()}")
    
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
    
    # Note: Removed has_coordinates column as it's redundant 
    # (can check if lat/lon are not null when needed)
    if all(col in df.columns for col in ['decimalLatitude', 'decimalLongitude']):
        coord_mask = df[['decimalLatitude', 'decimalLongitude']].notna().all(axis=1)
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
    
    # New flowering rule breakdown
    if all(col in df.columns for col in ['has_any_flowers', 'has_any_siliques']):
        print(f"\nFlowering rule breakdown (ANY flowers AND NO siliques):")
        print(f"  Occurrences with flowers: {df['has_any_flowers'].sum()}")
        print(f"  Occurrences with siliques: {df['has_any_siliques'].sum()}")
        print(f"  Occurrences with flowers AND siliques: {(df['has_any_flowers'] & df['has_any_siliques']).sum()}")
        print(f"  Occurrences with NEITHER: {(~df['has_any_flowers'] & ~df['has_any_siliques']).sum()}")
        print(f"  Occurrences with flowers ONLY (flowering): {(df['has_any_flowers'] & ~df['has_any_siliques']).sum()}")
    
    # Temporal coverage
    if 'year' in df.columns:
        year_range = df['year'].dropna()
        if len(year_range) > 0:
            print(f"\nTemporal coverage:")
            print(f"  Year range: {year_range.min():.0f} - {year_range.max():.0f}")
            print(f"  Records with year data: {len(year_range)}")
    
    # Geographic coverage
    coord_records = 0
    if all(col in df.columns for col in ['decimalLatitude', 'decimalLongitude']):
        coord_records = df[['decimalLatitude', 'decimalLongitude']].notna().all(axis=1).sum()
    print(f"\nGeographic coverage:")
    print(f"  Records with coordinates: {coord_records}")
    print(f"  Countries represented: {df['countryCode'].nunique() if 'countryCode' in df.columns else 'N/A'}")
    
    # Image counts
    if 'num_images' in df.columns:
        print(f"\nImage statistics:")
        print(f"  Total images analyzed: {df['num_images'].sum()}")
        print(f"  Average images per occurrence: {df['num_images'].mean():.1f}")
        print(f"  Max images per occurrence: {df['num_images'].max()}")
        
        # Component image counts
        if 'num_flower_images' in df.columns:
            print(f"  Total flower images: {df['num_flower_images'].sum()}")
            print(f"  Average flower images per occurrence: {df['num_flower_images'].mean():.1f}")
        if 'num_silique_images' in df.columns:
            print(f"  Total silique images: {df['num_silique_images'].sum()}")
            print(f"  Average silique images per occurrence: {df['num_silique_images'].mean():.1f}")
    
    # Probability distribution for flowers
    if 'max_flower_prob' in df.columns:
        print(f"\nFlower probability distribution:")
        print(f"  Mean max flower probability: {df['max_flower_prob'].mean():.3f}")
        print(f"  Median max flower probability: {df['max_flower_prob'].median():.3f}")
        print(f"  Range: {df['max_flower_prob'].min():.3f} - {df['max_flower_prob'].max():.3f}")
    
    # Probability distribution for siliques
    if 'max_silique_prob' in df.columns:
        print(f"\nSilique probability distribution:")
        print(f"  Mean max silique probability: {df['max_silique_prob'].mean():.3f}")
        print(f"  Median max silique probability: {df['max_silique_prob'].median():.3f}")
        print(f"  Range: {df['max_silique_prob'].min():.3f} - {df['max_silique_prob'].max():.3f}")
    
    print("\n" + "="*60)
    print("REMOVED REDUNDANT COLUMNS:")
    print("- max_prob_flowering (same as max_flower_prob)")
    print("- mean_prob_flowering (same as mean_flower_prob)")  
    print("- min_prob_flowering (same as min_flower_prob)")
    print("- eventDate (same as observation_date)")
    print("- has_coordinates (redundant with lat/lon null check)")
    print("="*60)


def main():
    """Main processing pipeline"""
    print("Arabidopsis Flowering Analysis Data Preparation")
    print("="*50)
    
    # File paths
    image_log_path = "../data/image_occurrence_log.csv"
    multimedia_path = "../data/multimedia_human_observations_only.csv"
    occurrence_path = "../data/occurrence.txt"  # User should place their occurrence.txt file in data/ folder
    output_path = "../data/occurrence_analysis_log.csv"
    enhanced_image_log_path = "../data/image_occurrence_log_with_urls.csv"
    
    try:
        # Step 1: Load image inference results
        image_df = load_image_inference_log(image_log_path)
        
        # Step 2: Load multimedia data with URLs
        multimedia_df = load_multimedia_data(multimedia_path)
        
        # Step 3: Merge URLs with image data
        image_df_with_urls = merge_image_urls(image_df, multimedia_df)
        
        # Step 4: Save enhanced image log with URLs
        image_df_with_urls.to_csv(enhanced_image_log_path, index=False)
        print(f"Enhanced image log with URLs saved to: {enhanced_image_log_path}")
        
        # Step 5: Load occurrence metadata
        occurrence_df = load_occurrence_data(occurrence_path)
        
        # Step 6: Aggregate image predictions to occurrence level (using enhanced data)
        predictions_df = aggregate_to_occurrence_level(image_df_with_urls)
        
        # Step 7: Merge with occurrence data
        analysis_df = merge_with_occurrence_data(predictions_df, occurrence_df)
        
        # Step 8: Clean temporal data
        analysis_df = clean_temporal_data(analysis_df)
        
        # Step 9: Clean geographic data
        analysis_df = clean_geographic_data(analysis_df)
        
        # Step 10: Reorder columns for analysis (removed redundant columns)
        column_order = [
            'gbifID', 'predicted_label_occurrence', 
            'has_any_flowers', 'has_any_siliques', 'num_flower_images', 'num_silique_images',
            'max_flower_prob', 'mean_flower_prob', 'min_flower_prob',
            'max_silique_prob', 'mean_silique_prob', 'min_silique_prob',
            'num_images', 
            'year', 'month', 'day', 'decade', 
            'decimalLatitude', 'decimalLongitude',
            'countryCode', 'stateProvince', 'county', 'locality', 'continent',
            'elevation', 'observation_date'
        ]
        
        # Keep only columns that exist
        existing_columns = [col for col in column_order if col in analysis_df.columns]
        additional_columns = [col for col in analysis_df.columns if col not in column_order]
        
        final_columns = existing_columns + additional_columns
        analysis_df = analysis_df[final_columns]
        
        # Step 11: Save the analysis dataset
        analysis_df.to_csv(output_path, index=False)
        print(f"\nAnalysis dataset saved to: {output_path}")
        
        # Step 12: Generate summary statistics
        generate_summary_stats(analysis_df)
        
        print(f"\nAnalysis data preparation completed successfully!")
        print(f"Ready for trend analysis with {len(analysis_df)} occurrence records")
        print(f"Enhanced image log with URLs available at: {enhanced_image_log_path}")
        
        return analysis_df
        
    except FileNotFoundError as e:
        print(f"Error: Required file not found - {e}")
        print("Make sure the following files exist:")
        print(f"  - {image_log_path}")
        print(f"  - {multimedia_path}")
        print(f"  - {occurrence_path}")
        sys.exit(1)
        
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    analysis_df = main()
