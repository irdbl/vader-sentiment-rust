//! Sarcasm detection via sentiment incongruity between context and reply.
//!
//! This module provides a lightweight sarcasm detector that works by comparing
//! VADER sentiment scores of a parent comment (context) and its reply. When
//! the reply's sentiment strongly contradicts the context's sentiment, this
//! signals possible sarcasm.
//!
//! The detector combines three signals:
//! 1. **Incongruity** — sentiment polarity flip between context and reply
//! 2. **Surface markers** — `/s` tags, scare quotes, sarcastic idioms
//! 3. **Intensity** — exaggerated positive language (exclamation marks, extreme compound)

use crate::SentimentIntensityAnalyzer;
pub use crate::SentimentScores;

/// Result of sarcasm detection on a context-reply pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SarcasmResult {
    /// Overall sarcasm probability, 0.0 (sincere) to 1.0 (sarcastic).
    pub probability: f64,
    /// Raw sentiment incongruity score (0.0–1.0).
    pub incongruity: f64,
    /// Surface marker detection score (0.0+).
    pub surface_score: f64,
    /// Exaggeration / intensity signal (0.0–1.0).
    pub intensity_score: f64,
    /// VADER sentiment of the context (parent comment).
    pub context_sentiment: SentimentScores,
    /// VADER sentiment of the reply.
    pub reply_sentiment: SentimentScores,
}

/// Tunable parameters for sarcasm detection.
#[derive(Debug, Clone)]
pub struct SarcasmConfig {
    /// Weight of sentiment incongruity in final probability.
    pub incongruity_weight: f64,
    /// Weight of surface markers in final probability.
    pub surface_weight: f64,
    /// Weight of exaggeration/intensity in final probability.
    pub intensity_weight: f64,
    /// Minimum |compound| to consider a text "opinionated".
    pub sentiment_threshold: f64,
    /// Bonus when `/s` tag is detected in reply.
    pub slash_s_bonus: f64,
    /// Bonus per scare-quoted phrase detected.
    pub scare_quote_bonus: f64,
    /// Bonus per sarcastic idiom match.
    pub idiom_bonus: f64,
    /// Bonus for excessive punctuation (3+ `!` or `?`).
    pub excessive_punct_bonus: f64,
    /// Bonus for ALL CAPS words in mixed-case text.
    pub all_caps_bonus: f64,
}

impl Default for SarcasmConfig {
    fn default() -> Self {
        SarcasmConfig {
            incongruity_weight: 0.65,
            surface_weight: 0.10,
            intensity_weight: 0.25,
            sentiment_threshold: 0.22,
            // Bonuses calibrated for surface_weight=1.0 in evaluator.
            // /s tag is strongest signal; scare quotes weakest (most are legit on Reddit).
            slash_s_bonus: 0.8,
            scare_quote_bonus: 0.06,
            idiom_bonus: 0.15,
            excessive_punct_bonus: 0.04,
            all_caps_bonus: 0.08,
        }
    }
}

/// Sarcastic idiom phrases (case-insensitive matching).
const SARCASTIC_IDIOMS: &[&str] = &[
    "yeah right",
    "oh great",
    "oh wonderful",
    "oh fantastic",
    "oh joy",
    "oh goody",
    "oh sure",
    "oh lovely",
    "what a surprise",
    "what a shock",
    "what a shocker",
    "how wonderful",
    "how lovely",
    "how nice",
    "how quaint",
    "how original",
    "how predictable",
    "thanks a lot",
    "big surprise",
    "no kidding",
    "no way",
    "real nice",
    "real classy",
    "sure thing",
    "tell me about it",
    "you don't say",
    "color me shocked",
    "shocking",
    "must be nice",
    "what could go wrong",
    "what could possibly go wrong",
    "what else could go wrong",
    "cool story",
    "cool story bro",
    "big whoop",
    "so brave",
    "slow clap",
    "go figure",
    "you must be fun at parties",
    "imagine that",
    "who knew",
    "smooth move",
    "nice going",
    "real smooth",
    "good luck with that",
    "thanks for nothing",
    "whatever you say",
    "thanks, i hate it",
    "just what we needed",
    "how convenient",
    "how fitting",
    "oh how nice",
    "bravo",
    "what a concept",
    "super helpful",
    "my favorite part",
    "love how",
    "i love how",
    "oh how convenient",
    "oh how fitting",
    "thanks obama",
    "captain obvious",
    "that'll be the day",
    "i'll believe it when i see it",
    "when pigs fly",
    "this is exactly why we can't have nice things",
    "oh wah",
    "because that makes sense",
    "because that makes total sense",
    "well that's helpful",
    "well thats helpful",
    "i bet that will go over well",
    "sure jan",
    "sure, jan",
    "give me a break",
    "hold my beer",
    "sounds legit",
    "that seems legit",
    "i'm sure that'll work",
    "im sure that'll work",
    "now we play the waiting game",
];

/// Sarcasm detector that wraps a `SentimentIntensityAnalyzer`.
pub struct SarcasmDetector<'a> {
    analyzer: &'a SentimentIntensityAnalyzer<'a>,
    config: SarcasmConfig,
}

impl<'a> SarcasmDetector<'a> {
    /// Create a new detector with default configuration.
    pub fn new(analyzer: &'a SentimentIntensityAnalyzer<'a>) -> Self {
        SarcasmDetector {
            analyzer,
            config: SarcasmConfig::default(),
        }
    }

    /// Create a new detector with custom configuration.
    pub fn with_config(analyzer: &'a SentimentIntensityAnalyzer<'a>, config: SarcasmConfig) -> Self {
        SarcasmDetector { analyzer, config }
    }

    /// Detect sarcasm in a reply given its parent context.
    pub fn detect(&self, context: &str, reply: &str) -> SarcasmResult {
        let context_sentiment = self.analyzer.polarity_scores(context);
        let reply_sentiment = self.analyzer.polarity_scores(reply);

        let incongruity = self.compute_incongruity(context_sentiment.compound, reply_sentiment.compound);
        let surface_score = self.detect_surface_markers(reply);
        let intensity_score = self.compute_intensity(reply_sentiment.compound, reply);

        let raw = self.config.incongruity_weight * incongruity
            + self.config.surface_weight * surface_score
            + self.config.intensity_weight * intensity_score;

        let probability = raw.clamp(0.0, 1.0);

        SarcasmResult {
            probability,
            incongruity,
            surface_score,
            intensity_score,
            context_sentiment,
            reply_sentiment,
        }
    }

    /// Detect sarcasm for a batch of (context, reply) pairs.
    pub fn detect_batch(&self, pairs: &[(&str, &str)]) -> Vec<SarcasmResult> {
        pairs.iter().map(|(ctx, reply)| self.detect(ctx, reply)).collect()
    }

    /// Compute sentiment incongruity between context and reply.
    ///
    /// High incongruity = reply sentiment strongly contradicts context sentiment.
    /// Returns 0.0–1.0.
    fn compute_incongruity(&self, context_compound: f64, reply_compound: f64) -> f64 {
        let threshold = self.config.sentiment_threshold;

        // Both must be "opinionated" (above threshold) and in opposite directions
        let context_opinionated = context_compound.abs() >= threshold;
        let reply_opinionated = reply_compound.abs() >= threshold;

        if !context_opinionated || !reply_opinionated {
            return 0.0;
        }

        // Check for polarity flip
        let is_flip = (context_compound < 0.0 && reply_compound > 0.0)
            || (context_compound > 0.0 && reply_compound < 0.0);

        if !is_flip {
            return 0.0;
        }

        // Magnitude: average of absolute compounds, capped at 1.0
        let magnitude = (context_compound.abs() + reply_compound.abs()) / 2.0;
        magnitude.min(1.0)
    }

    /// Detect surface-level sarcasm markers in the reply text.
    ///
    /// Returns an unbounded score (can exceed 1.0 with multiple markers).
    fn detect_surface_markers(&self, reply: &str) -> f64 {
        let mut score = 0.0;
        let lower = reply.to_lowercase();

        // /s tag — strong sarcasm signal
        if lower.contains("/s") {
            // Check it's likely a sarcasm tag, not part of a URL or markdown link
            for part in lower.split_whitespace() {
                if part == "/s"
                    || (part.ends_with("/s")
                        && !part.contains("://")
                        && !part.starts_with('(')
                        && !part.contains("](/s"))
                {
                    score += self.config.slash_s_bonus;
                    break;
                }
            }
        }

        // Scare quotes — count pairs of quotes around short phrases
        let scare_count = count_scare_quotes(reply);
        score += scare_count as f64 * self.config.scare_quote_bonus;

        // Sarcastic idioms
        for idiom in SARCASTIC_IDIOMS {
            if lower.contains(idiom) {
                score += self.config.idiom_bonus;
            }
        }

        // Excessive punctuation (3+ ! or 3+ ?)
        let excl_count = reply.bytes().filter(|&b| b == b'!').count();
        let quest_count = reply.bytes().filter(|&b| b == b'?').count();
        if excl_count >= 3 || quest_count >= 3 {
            score += self.config.excessive_punct_bonus;
        }

        // ALL CAPS words in mixed-case text
        if has_mixed_caps_words(reply) {
            score += self.config.all_caps_bonus;
        }

        score
    }

    /// Compute intensity/exaggeration signal from the reply.
    ///
    /// Extremely positive compound + exclamation marks = possible exaggeration.
    /// Returns 0.0–1.0.
    fn compute_intensity(&self, reply_compound: f64, reply: &str) -> f64 {
        // Only flag exaggerated positivity
        if reply_compound <= 0.8 {
            return 0.0;
        }

        let excl_count = reply.bytes().filter(|&b| b == b'!').count();
        let excl_factor = (excl_count as f64 * 0.1).min(0.5);

        let compound_excess = reply_compound - 0.8; // 0.0–0.2 range
        let compound_factor = compound_excess * 5.0; // scale to 0.0–1.0

        (compound_factor + excl_factor).min(1.0)
    }
}

/// Count scare-quoted phrases (words in quotes that aren't full sentences).
fn count_scare_quotes(text: &str) -> usize {
    let mut count = 0;
    let mut in_quote = false;
    let mut quote_start = 0;

    for (i, c) in text.char_indices() {
        if c == '"' {
            if in_quote {
                // End of quote — check if it's short (scare quote vs. full quote)
                let quoted_len = i - quote_start;
                if quoted_len > 0 && quoted_len <= 40 {
                    count += 1;
                }
                in_quote = false;
            } else {
                in_quote = true;
                quote_start = i + c.len_utf8();
            }
        }
    }

    count
}

/// Check if text has ALL-CAPS words mixed with lowercase words.
fn has_mixed_caps_words(text: &str) -> bool {
    let mut has_caps_word = false;
    let mut has_lower_word = false;

    for word in text.split_whitespace() {
        let alpha_count = word.bytes().filter(|b| b.is_ascii_alphabetic()).count();
        if alpha_count < 2 {
            continue;
        }

        let upper_count = word.bytes().filter(|b| b.is_ascii_uppercase()).count();
        let lower_count = word.bytes().filter(|b| b.is_ascii_lowercase()).count();

        if upper_count >= 2 && lower_count == 0 {
            has_caps_word = true;
        } else if lower_count > 0 {
            has_lower_word = true;
        }

        if has_caps_word && has_lower_word {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_detector() -> (SentimentIntensityAnalyzer<'static>, SarcasmConfig) {
        (SentimentIntensityAnalyzer::new(), SarcasmConfig::default())
    }

    #[test]
    fn test_obvious_sarcasm_with_incongruity() {
        let (analyzer, _) = make_detector();
        let detector = SarcasmDetector::new(&analyzer);

        // Negative context, positive reply = sarcasm signal
        let result = detector.detect(
            "This is terrible and awful, everything is broken",
            "Oh wow that's just wonderful and amazing! /s",
        );

        assert!(result.incongruity > 0.0, "expected incongruity, got {}", result.incongruity);
        assert!(result.probability > 0.3, "expected probability > 0.3, got {}", result.probability);
    }

    #[test]
    fn test_sincere_agreement() {
        let (analyzer, _) = make_detector();
        let detector = SarcasmDetector::new(&analyzer);

        // Both positive = no incongruity
        let result = detector.detect(
            "This movie is great, loved it",
            "I agree, it was really good!",
        );

        assert!(result.incongruity == 0.0, "expected no incongruity for agreement");
        assert!(result.probability < 0.3, "expected low probability for sincere reply");
    }

    #[test]
    fn test_slash_s_detection() {
        let (analyzer, _) = make_detector();
        let detector = SarcasmDetector::new(&analyzer);

        let result = detector.detect(
            "The servers are down again",
            "Great job team /s",
        );

        assert!(result.surface_score >= 0.8, "expected /s detection, got surface={}", result.surface_score);
    }

    #[test]
    fn test_scare_quotes() {
        assert_eq!(count_scare_quotes(r#"That's "totally" fine"#), 1);
        assert_eq!(count_scare_quotes(r#"He said "I love this" and "great work""#), 2);
        assert_eq!(count_scare_quotes(r#"No quotes here"#), 0);
    }

    #[test]
    fn test_sarcastic_idioms() {
        let (analyzer, _) = make_detector();
        let detector = SarcasmDetector::new(&analyzer);

        let result = detector.detect(
            "We shipped the bug to production",
            "Oh great, what a surprise",
        );

        // Should detect at least "oh great" and "what a surprise"
        assert!(result.surface_score >= 0.3, "expected idiom detection, got {}", result.surface_score);
    }

    #[test]
    fn test_mixed_caps_detection() {
        assert!(has_mixed_caps_words("this is GREAT work"));
        assert!(!has_mixed_caps_words("this is great work"));
        assert!(!has_mixed_caps_words("EVERYTHING IS GREAT"));
    }

    #[test]
    fn test_intensity_extreme_positive() {
        let (analyzer, _) = make_detector();
        let detector = SarcasmDetector::new(&analyzer);

        // Need multiple extreme-positive words to exceed the 0.8 compound threshold
        // with higher NORMALIZATION_ALPHA values (currently 32).
        let result = detector.detect(
            "The project failed completely",
            "AMAZING INCREDIBLE WONDERFUL SUPERB!!!",
        );

        assert!(result.intensity_score > 0.0, "expected intensity signal for extreme positive, got compound={}", result.reply_sentiment.compound);
    }

    #[test]
    fn test_neutral_context_no_incongruity() {
        let (analyzer, _) = make_detector();
        let detector = SarcasmDetector::new(&analyzer);

        // Neutral context → no incongruity regardless of reply
        let result = detector.detect(
            "I went to the store today",
            "That sounds absolutely terrible!",
        );

        assert_eq!(result.incongruity, 0.0, "neutral context should produce no incongruity");
    }

    #[test]
    fn test_batch_detection() {
        let (analyzer, _) = make_detector();
        let detector = SarcasmDetector::new(&analyzer);

        let pairs = vec![
            ("This is terrible", "Oh how wonderful /s"),
            ("Great movie!", "Yeah I loved it too"),
        ];

        let results = detector.detect_batch(&pairs);
        assert_eq!(results.len(), 2);
        assert!(results[0].probability > results[1].probability);
    }

    #[test]
    fn test_custom_config() {
        let (analyzer, _) = make_detector();
        let config = SarcasmConfig {
            incongruity_weight: 0.0,
            surface_weight: 1.0,
            intensity_weight: 0.0,
            ..SarcasmConfig::default()
        };
        let detector = SarcasmDetector::with_config(&analyzer, config);

        // With zero incongruity weight, only surface markers matter
        let result = detector.detect(
            "This is terrible",
            "Oh great /s",
        );

        assert!(result.surface_score > 0.0);
        // Probability should come entirely from surface markers
        assert!(result.probability > 0.0);
    }

    #[test]
    fn test_probability_clamped() {
        let (analyzer, _) = make_detector();
        let detector = SarcasmDetector::new(&analyzer);

        let result = detector.detect(
            "Everything about this is terrible awful horrible",
            "Oh wow that's just WONDERFUL and AMAZING!!! Incredible!!! /s yeah right what a surprise",
        );

        assert!(result.probability >= 0.0 && result.probability <= 1.0,
            "probability must be in [0,1], got {}", result.probability);
    }

    #[test]
    fn test_reverse_incongruity() {
        let (analyzer, _) = make_detector();
        let detector = SarcasmDetector::new(&analyzer);

        // Positive context, negative reply = also sarcasm signal
        let result = detector.detect(
            "This is wonderful and amazing, great job everyone!",
            "Yeah this is terrible, awful stuff",
        );

        assert!(result.incongruity > 0.0, "reverse polarity should also produce incongruity");
    }
}
