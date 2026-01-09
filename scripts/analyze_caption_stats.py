import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np

try:
    import nltk
except Exception:
    nltk = None

SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


def split_sentences(text, use_nltk=True):
    text = text.strip()
    if not text:
        return []
    if use_nltk and nltk is not None:
        try:
            sentences = nltk.sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]
        except LookupError:
            pass
        except Exception:
            pass
    parts = SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def estimate_token_count(text):
    if not text.strip():
        return 0
    return int(len(text.split()) * 1.3)


def build_token_counter():
    try:
        import clip
    except Exception:
        return None, "word_count_estimate"

    def count_tokens_batch(texts):
        if not texts:
            return []
        try:
            tokens = clip.tokenize(texts, truncate=True, context_length=1024)
        except TypeError:
            try:
                tokens = clip.tokenize(texts, truncate=True)
            except TypeError:
                tokens = clip.tokenize(texts)
        except Exception:
            return [estimate_token_count(t) for t in texts]
        return (tokens != 0).sum(dim=1).tolist()

    return count_tokens_batch, "clip.tokenize (batched)"


def analyze_captions(file_path, output_path, max_plot, clip_limit, batch_size, use_nltk):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if not isinstance(data, dict):
        print("Error: Expected a JSON object mapping ids to captions.")
        return

    token_counter, token_counter_name = build_token_counter()

    s1_counts = []
    s12_counts = []
    full_counts = []
    s1_batch = []
    s12_batch = []
    full_batch = []

    def count_tokens_batch(texts):
        return [estimate_token_count(t) for t in texts]

    if token_counter is None:
        token_counter = count_tokens_batch

    def flush_batches():
        if not s1_batch:
            return
        s1_counts.extend(token_counter(s1_batch))
        s12_counts.extend(token_counter(s12_batch))
        full_counts.extend(token_counter(full_batch))
        s1_batch.clear()
        s12_batch.clear()
        full_batch.clear()

    for key, caption in data.items():
        if not isinstance(caption, str):
            print(f"Warning: Value for key {key} is not a string.")
            continue

        sentences = split_sentences(caption, use_nltk=use_nltk)
        s1 = sentences[0] if sentences else ""
        s12 = " ".join(sentences[:2]) if sentences else ""

        s1_batch.append(s1)
        s12_batch.append(s12)
        full_batch.append(caption)

        if len(s1_batch) >= batch_size:
            flush_batches()

    flush_batches()

    if not s1_counts:
        print("No captions found or file is empty.")
        return

    mean_s1 = sum(s1_counts) / len(s1_counts)
    mean_s12 = sum(s12_counts) / len(s12_counts)
    pct_trunc = 100.0 * sum(1 for c in s12_counts if c > clip_limit) / len(s12_counts)
    max_full = max(full_counts)

    print(f"Token counter: {token_counter_name}")
    print(f"Mean S1: {mean_s1:.2f}")
    print(f"Mean S1+S2: {mean_s12:.2f}")
    print(f"Percentage of (S1+S2) > {clip_limit}: {pct_trunc:.2f}%")
    print(f"Max Full: {max_full}")

    bins = np.linspace(0, max_plot, 61)

    def clip_for_plot(values):
        return np.clip(values, 0, max_plot)

    plt.figure(figsize=(10, 6))
    plt.hist(clip_for_plot(s1_counts), bins=bins, density=True, histtype="step", linewidth=2, label="S1")
    plt.hist(clip_for_plot(s12_counts), bins=bins, density=True, histtype="step", linewidth=2, label="S1+S2")
    plt.hist(clip_for_plot(full_counts), bins=bins, density=True, histtype="step", linewidth=2, label="Full")

    plt.axvline(clip_limit, color="red", linestyle="--", linewidth=1.5, label="CLIP Context Limit")
    plt.xlabel("Token Count")
    plt.ylabel("Density")
    plt.xlim(0, max_plot)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze caption length distribution for TexVerse.")
    parser.add_argument("--input", default="../datasets/texverse/caption.json", help="Path to caption.json")
    parser.add_argument("--output", default="caption_length_distribution.png", help="Output figure path")
    parser.add_argument("--max-plot", type=int, default=300, help="Max X-axis token count")
    parser.add_argument("--clip-limit", type=int, default=77, help="CLIP context limit")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for tokenization")
    parser.add_argument("--no-nltk", action="store_true", help="Disable NLTK sentence splitting")
    args = parser.parse_args()

    analyze_captions(
        args.input,
        args.output,
        args.max_plot,
        args.clip_limit,
        args.batch_size,
        use_nltk=not args.no_nltk,
    )
