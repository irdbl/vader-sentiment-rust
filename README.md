# vader-sentiment-rust

A high-performance Rust port of [VADER](https://github.com/cjhutto/vaderSentiment) (Valence Aware Dictionary and sEntiment Reasoner) with context-aware sarcasm detection and an expanded, Reddit-tuned lexicon.

Built for bulk processing of social media text at scale (5TB+ Reddit comment archives). Processes millions of comments per second on a single machine.

## What's different from upstream VADER

- **Sarcasm detection** — context-aware module that detects sarcasm by comparing parent/reply sentiment incongruity, surface markers (`/s` tags, idioms, scare quotes), and intensity signals
- **Expanded lexicon** — 300+ new entries tuned on labeled Reddit data across r/wallstreetbets, r/movies, r/gaming, r/stocks, r/television, and r/patientgamers
- **Recalibrated constants** — NORMALIZATION_ALPHA, existing word scores, and intensifier weights tuned via automated evaluation loop
- **Performance optimizations** — ~2x faster than the original Rust port (Aho-Corasick emoji matching, scratch buffers, SIMD punctuation counting)
- **Upstream bug fixes** — "no" negation, `least_check` logic, curly quote handling, single-char tokens, and more

## Usage

```toml
[dependencies]
vader_sentiment = { git = "https://github.com/irdbl/vader-sentiment-rust" }
```

### Basic

```rust
use vader_sentiment::SentimentIntensityAnalyzer;

let analyzer = SentimentIntensityAnalyzer::new();
let scores = analyzer.polarity_scores("VADER is smart, handsome, and funny.");

println!("compound: {}", scores.compound);  // 0.8316
println!("positive: {}", scores.positive);  // 0.7458
println!("neutral:  {}", scores.neutral);   // 0.2542
println!("negative: {}", scores.negative);  // 0.0
```

### Sarcasm-aware analysis

```rust
use vader_sentiment::SentimentIntensityAnalyzer;
use vader_sentiment::sarcasm::{SarcasmDetector, SarcasmConfig};

let analyzer = SentimentIntensityAnalyzer::new();
let detector = SarcasmDetector::new(&analyzer, SarcasmConfig::default());

let parent = "My dog just died. Worst day of my life.";
let reply = "Oh that's just wonderful news!";

let result = detector.detect(parent, reply);
println!("sarcasm probability: {:.2}", result.probability);  // ~0.65
println!("incongruity: {:.2}", result.incongruity);          // high
println!("surface score: {:.2}", result.surface_score);      // moderate
```

### High-throughput (reusable scratch buffers)

```rust
use vader_sentiment::{SentimentIntensityAnalyzer, Scratch};

let analyzer = SentimentIntensityAnalyzer::new();
let mut scratch = Scratch::new();

for text in &texts {
    let scores = analyzer.polarity_scores_with_scratch(text, &mut scratch);
    println!("{}: {:.4}", text, scores.compound);
}
```

### Parallel batch processing

```rust
let scores = vader_sentiment::polarity_scores_batch(&["Great!", "Terrible!", "Meh."]);
```

## What VADER handles

- Negation ("not good", "wasn't very good")
- Punctuation emphasis ("Good!!!")
- Capitalization for emphasis ("VERY SMART")
- Degree modifiers ("very", "kind of", "barely")
- Slang ("sux", "uber", "friggin", "goated", "mid")
- Emoticons (`:)`, `:D`, `:^(`)
- UTF-8 emoji
- Contrastive conjunctions ("good, but not great")
- Special case idioms ("the bomb", "bad ass", "kiss of death")
- Context-aware sarcasm (when parent comment is provided)

## Performance

Benchmarked on Apple M4:

| Benchmark | Time |
|---|---|
| Short text (7 words) | 445 ns |
| Long text (paragraph) | 4.2 us |
| Text with emoji | 2.0 us |
| 1K batch | 378 us |
| 10K batch | 3.8 ms |

At these speeds, sentiment analysis is effectively free in a bulk pipeline — I/O (decompression, JSON parsing) is the bottleneck, not VADER.

Key optimizations:
- `SentimentScores` struct instead of `HashMap` return
- Aho-Corasick SIMD-accelerated emoji matching
- `memchr` SIMD punctuation counting
- Scratch buffer API for zero-allocation hot paths
- ASCII fast-path bypassing emoji scan
- Compile-time punctuation bitset

Run benchmarks: `cargo bench`

## Tuning

The `tuning/` directory contains an automated lexicon tuning pipeline:

1. **`build_corpus.py`** — Samples Reddit comments from Pushshift zst dumps with parent context (two-pass: collect comments, then resolve parent bodies)
2. **`label_corpus.py`** — Labels sentiment using Claude (context-aware, includes parent comment in prompt)
3. **`evaluate` binary** — Evaluates VADER against labeled corpus (Pearson, MAE, accuracy), outputs per-comment detail with sarcasm scores
4. **`tune_vader.sh`** — Automated loop: runs Codex to edit lexicon/constants/sarcasm config, evaluates, keeps improvements, reverts regressions

Current evaluation on 2,991 labeled Reddit comments (6 subreddits):
- Pearson correlation: 0.36
- Mean absolute error: 0.40
- Accuracy (within 0.2): 0.32

These numbers reflect the inherent ceiling of lexicon-based approaches on informal social media text. For trend analysis at scale, the consistent bias means relative trends are reliable even though absolute per-comment scores are noisy.

## Citation

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. *Eighth International Conference on Weblogs and Social Media (ICWSM-14)*. Ann Arbor, MI, June 2014.

## License

MIT
