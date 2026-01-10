import argparse
import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    import nltk
except Exception:
    nltk = None

SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LONGCLIP_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "third_party", "Long-CLIP"))


def resolve_longclip_module(longclip_root):
    last_exc = None
    try:
        import longclip as longclip_module
        return longclip_module, "longclip"
    except Exception as exc:
        last_exc = exc

    candidates = []
    if longclip_root:
        candidates.append(longclip_root)
    if os.path.isdir(DEFAULT_LONGCLIP_ROOT) and DEFAULT_LONGCLIP_ROOT not in candidates:
        candidates.append(DEFAULT_LONGCLIP_ROOT)

    for root in candidates:
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            continue
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            from model import longclip as longclip_module
            return longclip_module, f"model.longclip ({root})"
        except Exception as exc:
            last_exc = exc

    raise RuntimeError(
        "longclip is not available; install longclip or set --longclip-root to your Long-CLIP repo "
        f"(default: {DEFAULT_LONGCLIP_ROOT})"
    ) from last_exc


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


def build_longclip_token_counter(longclip_limit, longclip_root):
    longclip_module, import_source = resolve_longclip_module(longclip_root)

    tokenizer = longclip_module.tokenize
    counter_name = f"longclip.tokenize (batched, {import_source})"

    def count_tokens_batch(texts):
        if not texts:
            return []
        context_length = max(longclip_limit, 1024)
        try:
            tokens = tokenizer(texts, truncate=True, context_length=context_length)
        except TypeError:
            try:
                tokens = tokenizer(texts, truncate=True)
            except TypeError:
                tokens = tokenizer(texts)
        except Exception:
            return [estimate_token_count(t) for t in texts]
        return (tokens != 0).sum(dim=1).tolist()

    return count_tokens_batch, counter_name


def analyze_captions(
    file_path,
    output_path,
    max_plot,
    clip_limit,
    longclip_limit,
    longclip_root,
    batch_size,
    use_nltk,
):
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
    try:
        longclip_counter, longclip_counter_name = build_longclip_token_counter(longclip_limit, longclip_root)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return

    s1_counts = []
    s12_counts = []
    full_counts = []
    full_longclip_counts = []
    s1_batch = []
    s12_batch = []
    full_batch = []

    def count_tokens_batch(texts):
        return [estimate_token_count(t) for t in texts]

    if token_counter is None:
        token_counter = count_tokens_batch
    if longclip_counter is None:
        longclip_counter = count_tokens_batch

    def flush_batches():
        if not s1_batch:
            return
        s1_counts.extend(token_counter(s1_batch))
        s12_counts.extend(token_counter(s12_batch))
        full_counts.extend(token_counter(full_batch))
        full_longclip_counts.extend(longclip_counter(full_batch))
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
    longclip_pct_trunc = (
        100.0 * sum(1 for c in full_longclip_counts if c > longclip_limit) / len(full_longclip_counts)
    )
    max_full_longclip = max(full_longclip_counts)

    print(f"Token counter: {token_counter_name}")
    print(f"LongCLIP token counter: {longclip_counter_name}")
    print(f"Mean S1: {mean_s1:.2f}")
    print(f"Mean S1+S2: {mean_s12:.2f}")
    print(f"Percentage of (S1+S2) > {clip_limit}: {pct_trunc}%")
    print(f"Max Full (CLIP): {max_full}")
    print(f"Percentage of Full (LongCLIP) > {longclip_limit}: {longclip_pct_trunc}%")
    print(f"Max Full (LongCLIP): {max_full_longclip}")

    bins = np.linspace(0, max_plot, 61)

    def clip_for_plot(values):
        return np.clip(values, 0, max_plot)

    plt.figure(figsize=(10, 6))
    plt.hist(clip_for_plot(s1_counts), bins=bins, density=True, histtype="step", linewidth=2, label="S1")
    plt.hist(clip_for_plot(s12_counts), bins=bins, density=True, histtype="step", linewidth=2, label="S1+S2")
    plt.hist(clip_for_plot(full_counts), bins=bins, density=True, histtype="step", linewidth=2, label="Full (CLIP)")
    plt.hist(
        clip_for_plot(full_longclip_counts),
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        label="Full (LongCLIP)",
    )

    plt.axvline(clip_limit, color="red", linestyle="--", linewidth=1.5, label="CLIP Context Limit")
    plt.axvline(longclip_limit, color="orange", linestyle="--", linewidth=1.5, label="LongCLIP Context Limit")
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
    parser.add_argument("--longclip-limit", type=int, default=248, help="LongCLIP context limit")
    parser.add_argument(
        "--longclip-root",
        default=DEFAULT_LONGCLIP_ROOT,
        help="Path to Long-CLIP repo (default: %(default)s)",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for tokenization")
    parser.add_argument("--no-nltk", action="store_true", help="Disable NLTK sentence splitting")
    args = parser.parse_args()

    analyze_captions(
        args.input,
        args.output,
        args.max_plot,
        args.clip_limit,
        args.longclip_limit,
        args.longclip_root,
        args.batch_size,
        use_nltk=not args.no_nltk,
    )
