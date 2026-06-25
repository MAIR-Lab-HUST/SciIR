import json
import os
import shutil
import numpy as np

# ================= Path configuration =================
ABSTRACTS_PATH = "scir_dataset/classified_abstracts6.json"
CAPTIONS_PATH = "scir_dataset/caption6.json"
OUTPUT_ROOT = "output_dataset6"  # Output root directory
IMAGES_SOURCE_DIR = "scir_dataset/filtered_images_6"  # Source images directory


# ================= Helpers =================

def normalize_path(p):
    """Normalize a path and use the basename as a unique identifier."""
    if not p:
        return ""
    return os.path.basename(p)


def count_terms(reasoning):
    """Count total number of terms across labels in reasoning."""
    if not reasoning:
        return 0
    total = 0
    keys = ["ScientificLaw", "EntityStructure", "ScientificProcess"]
    for k in keys:
        if reasoning.get(k) and isinstance(reasoning[k], dict) and "terms" in reasoning[k]:
            terms = reasoning[k]["terms"]
            if terms:
                total += len(terms)
    return total


def clean_reasoning(reasoning):
    """
    Clean reasoning: if a label has both empty terms and empty visualization,
    set that label's value to null for cleaner, more consistent outputs.
    """
    if not reasoning:
        return reasoning

    cleaned = reasoning.copy()
    keys = ["ScientificLaw", "EntityStructure", "ScientificProcess"]

    for k in keys:
        if k in cleaned and cleaned[k]:
            label_data = cleaned[k]
            # Check whether terms is empty
            terms = label_data.get("terms", [])
            terms_empty = not terms or len(terms) == 0

            # Check whether visualization is empty
            visualization = label_data.get("visualization", [])
            viz_empty = not visualization or len(visualization) == 0

            # If both are empty, set to null
            if terms_empty and viz_empty:
                cleaned[k] = None

    return cleaned


def load_json(path):
    if not os.path.exists(path):
        print(f"Error: File not found {path}")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ================= Main logic =================

def get_quartiles(data):
    """Compute first and third quartiles (Q1, Q3) for a list of values."""
    if not data:
        return 0, 0
    return np.percentile(data, 25), np.percentile(data, 75)


def filter_by_iqr(items, key_func):
    """
    Keep items whose key values fall within [Q1, Q3].
    """
    if not items:
        return []

    values = [key_func(item) for item in items]
    q1, q3 = get_quartiles(values)

    filtered = []
    for item in items:
        val = key_func(item)
        if q1-1 <= val <= q3+1:
            filtered.append(item)
    return filtered


def main():
    print("1. Loading datasets...")
    abstracts_data = load_json(ABSTRACTS_PATH)
    captions_data = load_json(CAPTIONS_PATH)

    if not abstracts_data or not captions_data:
        print("Data load failed. Exiting.")
        return

    # Track missing images
    missing_images = set()
    
    # Build caption index
    caption_map = {normalize_path(item['image_path']): item for item in captions_data}

    # Predefined label set of interest
    VALID_LABELS = ["ScientificLaw", "EntityStructure", "ScientificProcess"]

    # Initialize four groups
    groups = {
        "EntityStructure_ScientificLaw": [],
        "ScientificLaw_ScientificProcess": [],
        "EntityStructure_ScientificProcess": [],
        "All_Three": []
    }

    print("2. Grouping candidates by checking reasoning content...")
    processed_files = set()  # Prevent processing the same image multiple times

    for article in abstracts_data:
        for segment in article.get("segments", []):
            img_filename = segment.get("filename")

            # Try to locate a matching caption by filename or path
            caption_item = caption_map.get(img_filename)
            if not caption_item:
                caption_item = caption_map.get(normalize_path(segment.get("path", "")))

            if not caption_item:
                continue

            # De-duplication (some abstracts may reference the same image)
            if img_filename in processed_files:
                continue
            processed_files.add(img_filename)

            # ================= [Core change] =================
            # Do not rely on segment['labels']; instead check whether terms actually exist in reasoning.
            reasoning = caption_item.get("reasoning", {})
            current_labels = []

            for label in VALID_LABELS:
                label_data = reasoning.get(label)
                # A label is considered valid only if label data exists and terms is non-empty.
                if label_data and isinstance(label_data, dict):
                    terms = label_data.get("terms", [])
                    if terms and len(terms) > 0:
                        current_labels.append(label)

            # Sort to keep a stable group key order
            current_labels.sort()
            # ================= [End change] =================

            group_name = None
            if len(current_labels) == 2:
                group_name = f"{current_labels[0]}_{current_labels[1]}"
            elif len(current_labels) == 3:
                group_name = "All_Three"

            # Skip if not in these four groups
            if not group_name or group_name not in groups:
                continue

            # Extract required fields
            rendered_txt = caption_item.get("rendered_text_stage2", [])
            retained_txt = caption_item.get("retained_text_stage3", [])

            # Count total terms for later filtering
            term_count = count_terms(reasoning)

            # Clean reasoning for output display (set fully empty labels to null)
            cleaned_reasoning = clean_reasoning(reasoning)

            # Build merged object
            merged_obj = {
                "source_image_id": article.get("image_id"),
                "image_filename": img_filename,
                "original_path": segment.get("path"),
                "labels": current_labels,  # Labels inferred from actual content
                "reasoning": cleaned_reasoning,
                "term_count": term_count,
                "rendered_len": len(rendered_txt),
                "retained_len": len(retained_txt),
                "sci-RCoT": caption_item.get("sci-RCoT"),
                "science_abstract_prompt": caption_item.get("science_abstract_prompt"),
                "retained_text": retained_txt
            }

            groups[group_name].append(merged_obj)

    # Process each group
    print("3. Processing groups (Filtering & Splitting)...")

    for group_name, candidates in groups.items():
        print(f"\n--- Processing Group: {group_name} (Initial: {len(candidates)}) ---")
        if not candidates:
            continue

        # --- Filter step 1: term_count within [Q1, Q3] ---
        candidates_step1 = filter_by_iqr(candidates, lambda x: x['term_count'])
        print(f"   After Term Count Filter: {len(candidates_step1)}")

        if not candidates_step1:
            continue

        # --- Filter step 2: both rendered_text and retained_text lengths within [Q1, Q3] ---
        # Apply filters based on the distribution from step 1
        candidates_step2a = filter_by_iqr(candidates_step1, lambda x: x['rendered_len'])
        candidates_step2b = filter_by_iqr(candidates_step2a, lambda x: x['retained_len'])

        final_candidates = candidates_step2b
        print(f"   After Text Length Filters: {len(final_candidates)}")

        if not final_candidates:
            continue

        # --- Split step: Prompt vs CoT (median split) ---
        # Use the median on the final filtered samples
        term_counts = [x['term_count'] for x in final_candidates]
        median_val = np.median(term_counts)
        print(f"   Median Term Count (for split): {median_val:.2f}")

        prompt_ds = []
        cot_ds = []
        for item in final_candidates:
            # Remove helper stats fields before saving
            out_item = item.copy()
            del out_item['rendered_len']
            del out_item['retained_len']
            del out_item['term_count']

            if item['term_count'] < median_val:
                prompt_ds.append(out_item)
            else:
                cot_ds.append(out_item)

        print(f"   > Prompt: {len(prompt_ds)}, CoT: {len(cot_ds)}")

        # --- Save outputs ---
        sub_tasks = [("prompt", prompt_ds), ("CoT", cot_ds)]

        for sub_name, ds in sub_tasks:
            # If empty, skip.
            if not ds:
                continue

            # Path layout: OUTPUT_ROOT / GroupName / prompt_dataset / ...
            ds_folder_name = f"{sub_name}_dataset"
            base_dir = os.path.join(OUTPUT_ROOT, group_name, ds_folder_name)
            img_dir = os.path.join(base_dir, "images")
            os.makedirs(img_dir, exist_ok=True)

            # Verify images exist and drop records with missing images
            filtered_ds = []
            for item in ds:
                img_filename = item['image_filename']
                if img_filename:
                    # Build the source path from the configured image directory
                    src_path = os.path.join(IMAGES_SOURCE_DIR, img_filename)
                    if os.path.exists(src_path):
                        filtered_ds.append(item)
                    else:
                        print(f"   Warning: Image not found: {src_path}, removing record from dataset")
                        missing_images.add(img_filename)
                else:
                    # If filename is missing, keep the record (original behavior)
                    filtered_ds.append(item)
            
            # Skip if the filtered dataset becomes empty
            if not filtered_ds:
                print(f"   Skipping {sub_name}: no valid images found")
                continue
            
            # Save JSON (valid records only)
            save_json(filtered_ds, os.path.join(base_dir, f"{sub_name}_data.json"))

            # Copy images
            for item in filtered_ds:
                img_filename = item['image_filename']
                if img_filename:
                    src_path = os.path.join(IMAGES_SOURCE_DIR, img_filename)
                    shutil.copy(src_path, os.path.join(img_dir, img_filename))

    # If there are missing images, update source input files accordingly
    if missing_images:
        print(f"\n4. Updating source files to remove {len(missing_images)} missing image records...")
        
        # Update captions_data: remove records with missing images
        updated_captions = []
        for item in captions_data:
            img_filename = normalize_path(item.get('image_path', ''))
            if img_filename not in missing_images:
                updated_captions.append(item)
        
        # Update abstracts_data: remove segments with missing images
        for article in abstracts_data:
            if 'segments' in article:
                updated_segments = []
                for segment in article['segments']:
                    img_filename = segment.get('filename')
                    if img_filename not in missing_images:
                        updated_segments.append(segment)
                article['segments'] = updated_segments
        
        # Save updated files
        save_json(updated_captions, CAPTIONS_PATH)
        save_json(abstracts_data, ABSTRACTS_PATH)
        print(f"   Updated {CAPTIONS_PATH}: {len(captions_data)} -> {len(updated_captions)} records")
        print(f"   Updated {ABSTRACTS_PATH}: removed segments with missing images")
    
    print(f"\nDone! All outputs generated in {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()