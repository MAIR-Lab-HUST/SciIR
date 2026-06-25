import os
import base64
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from collections import defaultdict
import time
import threading
import tempfile
import traceback

# Initialize client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", ""),
    base_url=os.getenv("OPENAI_BASE_URL", ""),
)

# Classification system prompt
classification_prompt = """
You are a rigorous scientific image classifier. Your task is to perform multi-label, multi-dimensional relevance scoring (1-10) on the input image based on the following three dimensions.

Evaluation Dimensions and Key Points
ScientificLaw
Determine whether the image involves and presents elements related to disciplinary laws and constraints (e.g., valence states/bonds, scale relationships, visual cues of energy/momentum conservation). 
EntityStructure
Determine whether the image involves the structure and geometric relationships of scientific entities (e.g., morphology, connections, topology, and relative scale of molecules/lattices/cells/galaxies/instrument components). 
ScientificProcess
Determine whether the image presents process information (temporal evolution, causal chains, state transitions, reaction mechanisms) or has clear process clues (multi-stage labels, timelines).

Scoring Principles
Score Meaning: 1-10 represents the "relevance" of the dimension in the image, not expression intensity, correctness, or quality.
Scoring Anchors (Relevance):
1-2: Almost not involved
3-4: Sporadic clues, but very weak
5-6: Moderately relevant, some evidence
7-8: Strongly relevant, ample evidence
9-10: Dominantly relevant, core of the image

Evidence and Limitations
Answer based ONLY on visible and clear evidence in the image; do not speculate on invisible elements.
Dimension names limited to: ScientificLaw, EntityStructure, ScientificProcess.

Output Format (Strictly follow the JSON output below)
{
  "relevance": {
    "ScientificLaw": { "score": 0 },
    "EntityStructure": { "score": 0 },
    "ScientificProcess": { "score": 0 }
  }
}
"""

# Path configuration (avoid machine-specific absolute paths)
filtered_dir = os.getenv("FILTERED_DIR", "./filtered_images_2/filtered_images_2")
metadata_path = os.getenv("METADATA_PATH", "./filtered_images_2/updated_metadata_2.json")
output_metadata_path = os.getenv("OUTPUT_METADATA_PATH", "./classified_metadata_2.json")
cache_path = os.getenv("CACHE_PATH", "./classification_cache_2.json")

# Lock for concurrent cache writes
cache_lock = threading.Lock()

# Atomic JSON save (with retries)
def _atomic_save_json(obj, path, lock=None, retries=5, base_delay=0.3):
    dirn = os.path.dirname(path) or "."
    if lock:
        lock.acquire()
    try:
        for attempt in range(retries):
            fd = None
            tmp_path = None
            try:
                fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(path), dir=dirn, text=True)
                with os.fdopen(fd, "w", encoding="utf-8") as tf:
                    json.dump(obj, tf, ensure_ascii=False, indent=2)
                    tf.flush()
                    os.fsync(tf.fileno())
                os.replace(tmp_path, path)
                return
            except PermissionError as e:
                try:
                    if tmp_path and os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
                if attempt < retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                raise
            except Exception:
                try:
                    if tmp_path and os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
                raise
    finally:
        if lock:
            lock.release()

# Load classification cache (file may not exist)
if os.path.exists(cache_path):
    try:
        with open(cache_path, "r", encoding='utf-8') as f:
            classification_cache = json.load(f)
    except Exception as e:
        print(f"⚠️ Unable to read cache, starting empty cache: {e}")
        classification_cache = {}
else:
    classification_cache = {}


def parse_json_response(response_text):
    import json, re

    # Strip markdown wrapping
    response_text = response_text.strip()
    response_text = re.sub(r"^```(?:json)?", "", response_text)
    response_text = re.sub(r"```$", "", response_text)
    response_text = response_text.strip()

    # Try direct parse
    try:
        data = json.loads(response_text)
        # If dict and contains relevance, return it
        if isinstance(data, dict) and "relevance" in data:
            return data
    except Exception:
        pass

    # If direct parse fails, try extracting the outermost JSON block
    # Use greedy matching to take the largest block
    matches = re.findall(r"\{[\s\S]*\}", response_text)
    if not matches:
        return get_default_result()

    # Prefer the longest JSON fragment
    matches.sort(key=len, reverse=True)
    for m in matches:
        try:
            data = json.loads(m)
            if "relevance" in data:
                return data
            # Support nested layers, e.g., {"result": {"relevance": {...}}}
            for v in data.values():
                if isinstance(v, dict) and "relevance" in v:
                    return v
        except Exception:
            continue

    # Fallback to default
    return get_default_result()

def get_default_result():
    return {
        "relevance": {
            "ScientificConsistency": {"score": 0},
            "EntityStructure": {"score": 0},
            "ScientificProcess": {"score": 0}
        },
        "confidence": 0
    }


def generate_labels(relevance_data):
    """Generate labels by the scoring rule (score >= 7)."""
    labels = []
    for dim in ["ScientificConsistency", "EntityStructure",  "ScientificProcess"]:
        dim_info = relevance_data.get(dim, {})
        # Safety: ensure dict and has score
        if isinstance(dim_info, dict) and dim_info.get("score", 0) >= 7:
            labels.append(dim)
    print("DEBUG relevance_data:", json.dumps(relevance_data, indent=2, ensure_ascii=False))
    return labels


def classify_image(filename):
    """Classify a single image across multiple dimensions."""
    img_path = os.path.join(filtered_dir, filename)

    # Prefer reading from cache first (cache stores relevance + confidence)
    if filename in classification_cache and isinstance(classification_cache[filename], dict):
        print(f"📋 Loaded from cache: {filename}")
        cached_result = classification_cache[filename]
        result = {
            "relevance": cached_result["relevance"],
            "confidence": cached_result.get("confidence", 0),
            "labels": generate_labels(cached_result["relevance"])
        }
        return filename, result

    try:
        with open(img_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"❌ Unable to read image {filename}: {e}")
        return filename, None

    messages = [
        {"role": "system", "content": classification_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please rate this scientific image across multiple dimensions."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
        }
    ]

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="internvl3.5-latest",
                messages=messages,
                temperature=0
            )

            reply = response.choices[0].message.content.strip()
            print(f"🔍 Raw model output ({filename}):\n{reply}")
            result = parse_json_response(reply)

            # Ensure required fields exist
            if "relevance" not in result:
                result["relevance"] = get_default_result()["relevance"]
            if "confidence" not in result:
                result["confidence"] = 0

            # Generate labels in code
            labels = generate_labels(result["relevance"])

            # Save to cache (store only raw scoring, not labels)
            to_cache = {
                "relevance": result["relevance"],
                "confidence": result.get("confidence", 0)
            }

            # Persist successful result to disk cache (atomic save; don't cache failures)
            try:
                # Update in-memory cache first (for concurrent reads)
                classification_cache[filename] = to_cache
                _atomic_save_json(classification_cache, cache_path, lock=cache_lock, retries=5, base_delay=0.3)
            except Exception as e:
                # If writing cache fails, remove the in-memory entry so next run retries
                print(f"⚠️ Cache write failed ({filename}); will retry next run: {e}")
                traceback.print_exc()
                classification_cache.pop(filename, None)
                # Still return the result for this run, but it won't be cached
                full_result = {"relevance": to_cache["relevance"], "confidence": to_cache["confidence"], "labels": labels}
                return filename, full_result

            # Build the full return result
            full_result = {
                "relevance": result["relevance"],
                "confidence": result["confidence"],
                "labels": labels
            }

            print(f"✅ Classification complete: {filename} → Labels: {labels}")
            return filename, full_result

        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1}/{max_retries} failed: {filename} - {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"❌ Classification failed: {filename}")
                return filename, None


def main():
    print("=" * 60)
    print("🚀 Starting multi-dimensional scientific image classification task")
    print("=" * 60)

    # 1. Load metadata
    print("\n📂 Loading metadata file...")
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_list = json.load(f)
        print(f"✅ Successfully loaded {len(metadata_list)} metadata records")
    except Exception as e:
        print(f"❌ Unable to load metadata file: {e}")
        return

    # Map filename to metadata
    filename_to_metadata = {}
    for metadata in metadata_list:
        for segment in metadata.get("segments", []):
            filename = segment.get("filename")
            if filename:
                filename_to_metadata[filename] = (metadata, segment)

    # 2. Collect images to classify (skip cached)
    print("\n📊 Counting images to classify...")
    image_files = []
    for fname in os.listdir(filtered_dir):
        if fname.startswith("sciir") and fname.endswith(".png"):
            image_files.append(fname)

    total_images = len(image_files)
    # Submit tasks only for uncached files (cached ones are skipped)
    to_process = [f for f in image_files if f not in classification_cache]
    skipped = total_images - len(to_process)
    print(f"✅ Found {total_images} images; cached {skipped}; will process {len(to_process)}")

    # 3. Run classification in parallel (only uncached)
    print("\n🔍 Starting multi-threaded classification...")
    classification_results = {}

    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit uncached tasks
        future_to_filename = {
            executor.submit(classify_image, fname): fname
            for fname in to_process
        }

        # Collect completed tasks
        completed = 0
        for future in as_completed(future_to_filename):
            filename, result = future.result()
            completed += 1
            if result:
                classification_results[filename] = result
                labels_str = ", ".join(result.get("labels", []))
                print(f"[{completed}/{len(to_process)}] ✅ {filename} → {labels_str if labels_str else 'No labels'}")
            else:
                print(f"[{completed}/{len(to_process)}] ❌ {filename} → Classification failed")

    # Include cached entries in results for metadata update
    for fname, cached in classification_cache.items():
        classification_results.setdefault(fname, {
            "relevance": cached["relevance"],
            "confidence": cached.get("confidence", 0),
            "labels": generate_labels(cached["relevance"])
        })

    # 4. Update metadata: only add labels field
    print("\n📝 Updating labels field in metadata...")
    update_count = 0

    for metadata in metadata_list:
        for segment in metadata.get("segments", []):
            filename = segment.get("filename")
            if filename in classification_results:
                # Only add labels; do not add any relevance content
                segment["labels"] = classification_results[filename].get("labels", [])
                update_count += 1

    # 5. Save updated metadata
    print(f"\n💾 Saving updated metadata...")
    try:
        with open(output_metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, indent=2, ensure_ascii=False)
        print(f"✅ Updated labels for {update_count} segments")
        print(f"✅ Updated metadata saved to: {output_metadata_path}")
    except Exception as e:
        print(f"❌ Failed to save metadata: {e}")

    # 6. Save classification cache (write again for consistency)
    print("\n💾 Saving classification cache...")
    try:
        _atomic_save_json(classification_cache, cache_path, lock=cache_lock, retries=5, base_delay=0.3)
        print(f"✅ Classification cache saved to: {cache_path}")
    except Exception as e:
        print(f"⚠️ Failed to save cache: {e}")

    # 7. Summary statistics
    print("\n📊 Classification summary:")
    print("=" * 60)

    # Distribution of labels by dimension
    label_counts = defaultdict(int)
    label_combinations = defaultdict(int)

    for result in classification_results.values():
        labels = result.get("labels", [])
        for label in labels:
            label_counts[label] += 1
        if labels:
            label_combinations[tuple(sorted(labels))] += 1

    print("\nLabel distribution by dimension:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / max(1, len(classification_results))) * 100
        print(f"  {label}: {count} images ({percentage:.1f}%)")

    print("\nCommon label combinations (Top 10):")
    for combo, count in sorted(label_combinations.items(), key=lambda x: x[1], reverse=True)[:10]:
        combo_str = " + ".join(combo)
        percentage = (count / max(1, len(classification_results))) * 100
        print(f"  {combo_str}: {count} images ({percentage:.1f}%)")

    # Count images with no labels
    no_label_count = sum(1 for r in classification_results.values() if not r.get("labels"))
    if no_label_count > 0:
        print(f"\n⚠️ Images with no labels: {no_label_count} ({(no_label_count / max(1, len(classification_results))) * 100:.1f}%)")

    print("\n" + "=" * 60)
    print("🎉 Classification task complete!")
    print(f"✅ Successfully classified: {len(classification_results)}/{len(image_files)} images")
    print(f"✅ Output file: {output_metadata_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()