# vader-sentiment-rust

A high-performance Rust port of [VADER](https://github.com/cjhutto/vaderSentiment) (Valence Aware Dictionary and sEntiment Reasoner), a lexicon and rule-based sentiment analysis tool attuned to sentiments expressed in social media.

This fork includes upstream bug fixes from the original Python VADER and significant performance optimizations (~2x faster than the original Rust port).

## Usage

Add to your `Cargo.toml`:

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

### High-throughput (reusable scratch buffers)

```rust
use vader_sentiment::{SentimentIntensityAnalyzer, Scratch};

let analyzer = SentimentIntensityAnalyzer::new();
let mut scratch = Scratch::new();

let texts = vec!["Great movie!", "Terrible service.", "It was okay."];
for text in &texts {
    let scores = analyzer.polarity_scores_with_scratch(text, &mut scratch);
    println!("{}: {:.4}", text, scores.compound);
}
```

### Parallel batch processing

Enable the `parallel` feature for Rayon-powered batch analysis:

```toml
vader_sentiment = { git = "https://github.com/irdbl/vader-sentiment-rust", features = ["parallel"] }
```

```rust
let scores = vader_sentiment::polarity_scores_batch(&["Great!", "Terrible!", "Meh."]);
```

## What VADER handles

- Negation ("not good", "wasn't very good")
- Punctuation emphasis ("Good!!!")
- Capitalization for emphasis ("VERY SMART")
- Degree modifiers ("very", "kind of", "barely")
- Slang ("sux", "uber", "friggin")
- Emoticons (`:)`, `:D`, `:^(`)
- UTF-8 emoji (💘, 😁, 💋)
- Initialisms and acronyms ("lol")
- Contrastive conjunctions ("good, but not great")
- Special case idioms ("the bomb", "bad ass", "kiss of death")

## Performance

~2x faster than the original Rust port, benchmarked on Apple M4:

| Benchmark | Time |
|---|---|
| Short text (7 words) | 445 ns |
| Long text (paragraph) | 4.2 µs |
| Text with emoji | 2.0 µs |
| 1K batch | 378 µs |
| 10K batch | 3.8 ms |

Key optimizations:
- `SentimentScores` struct instead of `HashMap` return
- Aho-Corasick SIMD-accelerated emoji matching
- FxHash for all internal hash maps
- `memchr` SIMD punctuation counting
- Scratch buffer API for zero-allocation hot paths
- ASCII fast-path bypassing emoji scan
- Compile-time punctuation bitset

Run benchmarks: `cargo bench`

## Upstream fixes ported

Fixes from the original Python VADER that were missing in the Rust port:

- **"no" negation** — "no good" correctly produces negative sentiment
- **`least_check` logic** — fixed impossible `&&` condition
- **Special case idioms** — added "beating heart", "broken heart", "bus stop"
- **Single-char tokens** — no longer filtered out (matches Python behavior)
- **Strip punctuation threshold** — changed from `< 3` to `<= 2`
- **Curly quote negation** — "won\u{2019}t" handled like "won't"

## Citation

If you use VADER in your research, please cite:

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. *Eighth International Conference on Weblogs and Social Media (ICWSM-14)*. Ann Arbor, MI, June 2014.

## License

MIT
