use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use vader_sentiment::{Scratch, SentimentIntensityAnalyzer};

fn short_texts() -> Vec<&'static str> {
    vec![
        "VADER is smart, handsome, and funny.",
        "VADER is not smart, handsome, nor funny.",
        "Today SUX!",
        "Make sure you :) or :D today!",
        "Not bad at all",
    ]
}

fn long_text() -> &'static str {
    "The book was good. I really enjoyed reading it, especially the first few chapters \
     which were absolutely brilliant! However, the ending was somewhat disappointing \
     and I felt the author could have done a much better job wrapping things up. \
     Overall, it was a very decent read but not the best I've ever encountered. \
     The characters were compelling and the dialogue was mostly great, \
     though occasionally it felt a bit forced. Would I recommend it? \
     Absolutely, but with some reservations about the final act."
}

fn text_with_emojis() -> &'static str {
    "I love this movie 😀😀😀 it was so amazing 💘 best thing ever 😁"
}

fn bench_single_short(c: &mut Criterion) {
    let analyzer = SentimentIntensityAnalyzer::new();
    let mut scratch = Scratch::new();
    c.bench_function("single_short_text", |b| {
        b.iter(|| {
            analyzer.polarity_scores_with_scratch(
                black_box("VADER is smart, handsome, and funny!"),
                &mut scratch,
            )
        })
    });
}

fn bench_single_long(c: &mut Criterion) {
    let analyzer = SentimentIntensityAnalyzer::new();
    let mut scratch = Scratch::new();
    let text = long_text();
    c.bench_function("single_long_text", |b| {
        b.iter(|| analyzer.polarity_scores_with_scratch(black_box(text), &mut scratch))
    });
}

fn bench_with_emojis(c: &mut Criterion) {
    let analyzer = SentimentIntensityAnalyzer::new();
    let mut scratch = Scratch::new();
    let text = text_with_emojis();
    c.bench_function("text_with_emojis", |b| {
        b.iter(|| analyzer.polarity_scores_with_scratch(black_box(text), &mut scratch))
    });
}

fn bench_without_emojis(c: &mut Criterion) {
    let analyzer = SentimentIntensityAnalyzer::new();
    let mut scratch = Scratch::new();
    c.bench_function("text_without_emojis", |b| {
        b.iter(|| {
            analyzer.polarity_scores_with_scratch(black_box("The plot was good, but the characters are uncompelling and the dialog is not great."), &mut scratch)
        })
    });
}

fn bench_batch(c: &mut Criterion) {
    let analyzer = SentimentIntensityAnalyzer::new();
    let mut scratch = Scratch::new();
    let texts = short_texts();
    let batch_1k: Vec<&str> = texts.iter().cycle().take(1000).copied().collect();
    let batch_10k: Vec<&str> = texts.iter().cycle().take(10_000).copied().collect();

    let mut group = c.benchmark_group("batch_sequential");
    group.bench_with_input(BenchmarkId::new("1k", 1000), &batch_1k, |b, texts| {
        b.iter(|| {
            for text in texts {
                black_box(analyzer.polarity_scores_with_scratch(text, &mut scratch));
            }
        })
    });
    group.bench_with_input(BenchmarkId::new("10k", 10_000), &batch_10k, |b, texts| {
        b.iter(|| {
            for text in texts {
                black_box(analyzer.polarity_scores_with_scratch(text, &mut scratch));
            }
        })
    });
    group.finish();
}

fn bench_batch_api(c: &mut Criterion) {
    let analyzer = SentimentIntensityAnalyzer::new();
    let texts = short_texts();
    let batch_1k: Vec<&str> = texts.iter().cycle().take(1000).copied().collect();
    let batch_10k: Vec<&str> = texts.iter().cycle().take(10_000).copied().collect();
    let mut out = Vec::new();

    let mut group = c.benchmark_group("batch_api_sequential");
    group.bench_with_input(BenchmarkId::new("1k", 1000), &batch_1k, |b, texts| {
        b.iter(|| {
            analyzer.polarity_scores_batch_into(texts, &mut out);
            black_box(out.len());
        })
    });
    group.bench_with_input(BenchmarkId::new("10k", 10_000), &batch_10k, |b, texts| {
        b.iter(|| {
            analyzer.polarity_scores_batch_into(texts, &mut out);
            black_box(out.len());
        })
    });
    group.finish();
}

fn bench_batch_parallel(c: &mut Criterion) {
    let texts = short_texts();
    let batch_1k: Vec<&str> = texts.iter().cycle().take(1000).copied().collect();
    let batch_10k: Vec<&str> = texts.iter().cycle().take(10_000).copied().collect();

    let mut group = c.benchmark_group("batch_api_parallel");
    group.bench_with_input(BenchmarkId::new("1k", 1000), &batch_1k, |b, texts| {
        b.iter(|| {
            black_box(vader_sentiment::polarity_scores_batch(texts));
        })
    });
    group.bench_with_input(BenchmarkId::new("10k", 10_000), &batch_10k, |b, texts| {
        b.iter(|| {
            black_box(vader_sentiment::polarity_scores_batch(texts));
        })
    });
    group.finish();
}

fn bench_punctuation_emphasis(c: &mut Criterion) {
    let analyzer = SentimentIntensityAnalyzer::new();
    let mut scratch = Scratch::new();
    c.bench_function("punctuation_emphasis", |b| {
        let text = "This is amazing!!! Really??? Yes!!! Are you sure??? Absolutely!!!";
        b.iter(|| analyzer.polarity_scores_with_scratch(black_box(text), &mut scratch))
    });
}

fn bench_negation_heavy(c: &mut Criterion) {
    let analyzer = SentimentIntensityAnalyzer::new();
    let mut scratch = Scratch::new();
    c.bench_function("negation_heavy", |b| {
        b.iter(|| {
            analyzer.polarity_scores_with_scratch(black_box(
                "I don't think this isn't not a bad idea, but it won't never work without doubt"
            ), &mut scratch)
        })
    });
}

criterion_group!(
    benches,
    bench_single_short,
    bench_single_long,
    bench_with_emojis,
    bench_without_emojis,
    bench_batch,
    bench_batch_api,
    bench_batch_parallel,
    bench_punctuation_emphasis,
    bench_negation_heavy,
);
criterion_main!(benches);
