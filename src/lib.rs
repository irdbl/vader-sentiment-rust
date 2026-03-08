/**
 * If you use the VADER sentiment analysis tools, please cite:
 * Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for
 * Sentiment Analysis of Social Media Text. Eighth International Conference on
 * Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
 **/

use std::borrow::Cow;
use std::cmp::min;
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use aho_corasick::AhoCorasick;
use memchr::memchr_iter;
use unicase::UniCase;

#[cfg(test)]
mod tests;

// Empirically derived constants for scaling/amplifying sentiments
const B_INCR: f64 = 0.293;
const B_DECR: f64 = -0.293;

const C_INCR: f64 = 0.733;
const NEGATION_SCALAR: f64 = -0.740;

// Sentiment increases for text with question or exclamation marks
const QMARK_INCR: f64 = 0.180;
const EMARK_INCR: f64 = 0.292;

// Maximum amount of question or exclamation marks before their contribution to sentiment is
// disregarded
const MAX_EMARK: i32 = 4;
const MAX_QMARK: i32 = 3;
const MAX_QMARK_INCR: f64 = 0.96;

const NORMALIZATION_ALPHA: f64 = 15.0;

static RAW_LEXICON: &str = include_str!("resources/vader_lexicon.txt");
static RAW_EMOJI_LEXICON: &str = include_str!("resources/emoji_utf8_lexicon.txt");

// Punctuation bitset for O(1) lookup (all ASCII punctuation chars)
const PUNCT_BITSET: u128 = {
    let bytes = b"[!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~]";
    let mut bitset: u128 = 0;
    let mut i = 0;
    while i < bytes.len() {
        bitset |= 1u128 << bytes[i];
        i += 1;
    }
    bitset
};

#[inline]
fn is_punctuation(c: char) -> bool {
    let code = c as u32;
    code < 128 && (PUNCT_BITSET >> code) & 1 == 1
}

// --- Static data (LazyLock replaces lazy_static + maplit) ---

static NEGATION_TOKENS: LazyLock<HashSet<UniCase<&'static str>>> = LazyLock::new(|| {
    let words = [
        "aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
        "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
        "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
        "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
        "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
        "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
        "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
        "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite",
    ];
    let mut set = HashSet::with_capacity(words.len());
    for w in &words {
        set.insert(UniCase::new(*w));
    }
    set
});

static BOOSTER_DICT: LazyLock<HashMap<UniCase<&'static str>, f64>> = LazyLock::new(|| {
    let entries: &[(&str, f64)] = &[
        ("absolutely", B_INCR), ("amazingly", B_INCR), ("awfully", B_INCR),
        ("completely", B_INCR), ("considerable", B_INCR), ("considerably", B_INCR),
        ("decidedly", B_INCR), ("deeply", B_INCR), ("effing", B_INCR),
        ("enormous", B_INCR), ("enormously", B_INCR),
        ("entirely", B_INCR), ("especially", B_INCR), ("exceptional", B_INCR),
        ("exceptionally", B_INCR),
        ("extreme", B_INCR), ("extremely", B_INCR),
        ("fabulously", B_INCR), ("flipping", B_INCR), ("flippin", B_INCR),
        ("frackin", B_INCR), ("fracking", B_INCR),
        ("fricking", B_INCR), ("frickin", B_INCR), ("frigging", B_INCR),
        ("friggin", B_INCR), ("fully", B_INCR),
        ("fuckin", B_INCR), ("fucking", B_INCR), ("fuggin", B_INCR), ("fugging", B_INCR),
        ("greatly", B_INCR), ("hella", B_INCR), ("highly", B_INCR), ("hugely", B_INCR),
        ("incredible", B_INCR), ("incredibly", B_INCR), ("intensely", B_INCR),
        ("major", B_INCR), ("majorly", B_INCR), ("more", B_INCR), ("most", B_INCR),
        ("particularly", B_INCR),
        ("purely", B_INCR), ("quite", B_INCR), ("really", B_INCR), ("remarkably", B_INCR),
        ("so", B_INCR), ("substantially", B_INCR),
        ("thoroughly", B_INCR), ("total", B_INCR), ("totally", B_INCR),
        ("tremendous", B_INCR), ("tremendously", B_INCR),
        ("uber", B_INCR), ("unbelievably", B_INCR), ("unusually", B_INCR),
        ("utter", B_INCR), ("utterly", B_INCR),
        ("very", B_INCR),
        ("almost", B_DECR), ("barely", B_DECR), ("hardly", B_DECR),
        ("just enough", B_DECR),
        ("kind of", B_DECR), ("kinda", B_DECR), ("kindof", B_DECR), ("kind-of", B_DECR),
        ("less", B_DECR), ("little", B_DECR), ("marginal", B_DECR),
        ("marginally", B_DECR),
        ("occasional", B_DECR), ("occasionally", B_DECR), ("partly", B_DECR),
        ("scarce", B_DECR), ("scarcely", B_DECR), ("slight", B_DECR),
        ("slightly", B_DECR), ("somewhat", B_DECR),
        ("sort of", B_DECR), ("sorta", B_DECR), ("sortof", B_DECR), ("sort-of", B_DECR),
    ];
    let mut map = HashMap::with_capacity(entries.len());
    for &(w, v) in entries {
        map.insert(UniCase::new(w), v);
    }
    map
});

static SPECIAL_CASE_IDIOMS: LazyLock<HashMap<UniCase<&'static str>, f64>> = LazyLock::new(|| {
    let entries: &[(&str, f64)] = &[
        ("the shit", 3.0), ("the bomb", 3.0), ("bad ass", 1.5), ("badass", 1.5),
        ("bus stop", 0.0), ("yeah right", -2.0), ("kiss of death", -1.5),
        ("to die for", 3.0), ("beating heart", 3.1), ("broken heart", -2.9),
    ];
    let mut map = HashMap::with_capacity(entries.len());
    for &(w, v) in entries {
        map.insert(UniCase::new(w), v);
    }
    map
});

// Pre-split special case idioms into word sequences for zero-alloc matching
static SPECIAL_IDIOMS_SPLIT: LazyLock<Vec<(Vec<UniCase<&'static str>>, f64)>> = LazyLock::new(|| {
    SPECIAL_CASE_IDIOMS.iter().map(|(key, &val)| {
        let words: Vec<UniCase<&str>> = key.as_ref().split(' ').map(UniCase::new).collect();
        (words, val)
    }).collect()
});

pub static LEXICON: LazyLock<HashMap<UniCase<&'static str>, f64>> = LazyLock::new(|| {
    parse_raw_lexicon(RAW_LEXICON)
});

pub static EMOJI_LEXICON: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    parse_raw_emoji_lexicon(RAW_EMOJI_LEXICON)
});

// Aho-Corasick automaton for SIMD-accelerated emoji matching
static EMOJI_AC: LazyLock<(AhoCorasick, Vec<&'static str>)> = LazyLock::new(|| {
    let lines = RAW_EMOJI_LEXICON.trim_end_matches('\n').split('\n');
    let mut patterns = Vec::with_capacity(3600);
    let mut descriptions = Vec::with_capacity(3600);
    for line in lines {
        if line.is_empty() { continue; }
        let mut split = line.split('\t');
        let emoji = split.next().unwrap();
        let desc = split.next().unwrap();
        patterns.push(emoji);
        descriptions.push(desc);
    }
    let ac = AhoCorasick::builder()
        .match_kind(aho_corasick::MatchKind::LeftmostLongest)
        .build(&patterns)
        .unwrap();
    (ac, descriptions)
});

static STATIC_BUT: LazyLock<UniCase<&'static str>> = LazyLock::new(|| UniCase::new("but"));
static STATIC_THIS: LazyLock<UniCase<&'static str>> = LazyLock::new(|| UniCase::new("this"));
static STATIC_AT: LazyLock<UniCase<&'static str>> = LazyLock::new(|| UniCase::new("at"));
static STATIC_LEAST: LazyLock<UniCase<&'static str>> = LazyLock::new(|| UniCase::new("least"));
static STATIC_VERY: LazyLock<UniCase<&'static str>> = LazyLock::new(|| UniCase::new("very"));
static STATIC_WITHOUT: LazyLock<UniCase<&'static str>> = LazyLock::new(|| UniCase::new("without"));
static STATIC_DOUBT: LazyLock<UniCase<&'static str>> = LazyLock::new(|| UniCase::new("doubt"));
static STATIC_SO: LazyLock<UniCase<&'static str>> = LazyLock::new(|| UniCase::new("so"));
static STATIC_NEVER: LazyLock<UniCase<&'static str>> = LazyLock::new(|| UniCase::new("never"));
static STATIC_KIND: LazyLock<UniCase<&'static str>> = LazyLock::new(|| UniCase::new("kind"));
static STATIC_OF: LazyLock<UniCase<&'static str>> = LazyLock::new(|| UniCase::new("of"));
static STATIC_NO: LazyLock<UniCase<&'static str>> = LazyLock::new(|| UniCase::new("no"));
static STATIC_OR: LazyLock<UniCase<&'static str>> = LazyLock::new(|| UniCase::new("or"));
static STATIC_NOR: LazyLock<UniCase<&'static str>> = LazyLock::new(|| UniCase::new("nor"));

/// Sentiment scores returned by polarity_scores.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SentimentScores {
    pub compound: f64,
    pub positive: f64,
    pub negative: f64,
    pub neutral: f64,
}

/// Takes the raw text of the lexicon files and creates HashMaps
pub fn parse_raw_lexicon(raw_lexicon: &str) -> HashMap<UniCase<&str>, f64> {
    let lines: Vec<&str> = raw_lexicon.trim_end_matches('\n').split('\n').collect();
    let mut lex_dict = HashMap::with_capacity(lines.len());
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let mut split_line = line.split('\t');
        let word = split_line.next().unwrap();
        let val = split_line.next().unwrap();
        lex_dict.insert(UniCase::new(word), val.parse().unwrap());
    }
    lex_dict
}

pub fn parse_raw_emoji_lexicon(raw_emoji_lexicon: &str) -> HashMap<&str, &str> {
    let lines: Vec<&str> = raw_emoji_lexicon.trim_end_matches('\n').split('\n').collect();
    let mut emoji_dict = HashMap::with_capacity(lines.len());
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let mut split_line = line.split('\t');
        let word = split_line.next().unwrap();
        let desc = split_line.next().unwrap();
        emoji_dict.insert(word, desc);
    }
    emoji_dict
}

/// Stores tokens and useful info about text
struct ParsedText<'a> {
    tokens: Vec<UniCase<&'a str>>,
    has_mixed_caps: bool,
    punc_amplifier: f64,
}

impl<'a> ParsedText<'a> {
    fn from_text(text: &'a str) -> ParsedText<'a> {
        let tokens = ParsedText::tokenize(text);
        let has_mixed_caps = ParsedText::has_mixed_caps(&tokens);
        let punc_amplifier = ParsedText::get_punctuation_emphasis(text);
        ParsedText {
            tokens,
            has_mixed_caps,
            punc_amplifier,
        }
    }

    fn tokenize(text: &str) -> Vec<UniCase<&str>> {
        text.split_whitespace()
            .map(ParsedText::strip_punc_if_word)
            .map(UniCase::new)
            .collect()
    }

    // Removes punctuation from words, ie "hello!!!" -> "hello" and ",don't??" -> "don't"
    // Keeps most emoticons, ie ":^)" -> ":^)"
    fn strip_punc_if_word(token: &str) -> &str {
        let stripped = token.trim_matches(is_punctuation);
        if stripped.len() <= 2 {
            return token;
        }
        stripped
    }

    // Determines if message has a mix of both all caps and non all caps words
    fn has_mixed_caps<S: AsRef<str>>(tokens: &[S]) -> bool {
        let (mut has_caps, mut has_non_caps) = (false, false);
        for token in tokens.iter() {
            if is_all_caps(token.as_ref()) {
                has_caps = true;
            } else {
                has_non_caps = true;
            }
            if has_non_caps && has_caps {
                return true;
            }
        }
        false
    }

    // Uses empirical values to determine how the use of '?' and '!' contribute to sentiment
    // Uses memchr for SIMD-accelerated byte counting
    fn get_punctuation_emphasis(text: &str) -> f64 {
        let bytes = text.as_bytes();
        let emark_count = memchr_iter(b'!', bytes).count() as i32;
        let qmark_count = memchr_iter(b'?', bytes).count() as i32;

        let emark_emph = min(emark_count, MAX_EMARK) as f64 * EMARK_INCR;
        let mut qmark_emph = (qmark_count as f64) * QMARK_INCR;
        if qmark_count > MAX_QMARK {
            qmark_emph = MAX_QMARK_INCR;
        }
        qmark_emph + emark_emph
    }
}

// Checks if all letters in token are capitalized (matches Python's str.isupper())
// Single pass: returns true if at least one uppercase and no lowercase letters
#[inline]
fn is_all_caps(token: &str) -> bool {
    let mut has_upper = false;
    for &b in token.as_bytes() {
        if b.is_ascii_lowercase() {
            return false;
        }
        has_upper |= b.is_ascii_uppercase();
    }
    has_upper
}

// Checks if token is in the list of negation tokens
// Handles both ASCII apostrophe and Unicode right single quotation mark (U+2019)
#[inline]
fn is_negated(token: &UniCase<&str>) -> bool {
    if NEGATION_TOKENS.contains(token) {
        return true;
    }
    token.contains("n't") || token.contains("n\u{2019}t")
}

// Normalizes score between -1.0 and 1.0. Alpha value is expected upper limit for a score
fn normalize_score(score: f64) -> f64 {
    let norm_score = score / (score * score + NORMALIZATION_ALPHA).sqrt();
    if norm_score < -1.0 {
        return -1.0;
    } else if norm_score > 1.0 {
        return 1.0;
    }
    norm_score
}

// Checks how previous tokens affect the valence of the current token
// Uses pre-computed booster value to avoid HashMap lookup
#[inline]
fn scalar_inc_dec(token: &UniCase<&str>, booster: Option<f64>, valence: f64, has_mixed_caps: bool) -> f64 {
    let mut scalar = 0.0;
    if let Some(s) = booster {
        scalar = s;
        if valence < 0.0 {
            scalar *= -1.0;
        }
        if is_all_caps(token.as_ref()) && has_mixed_caps {
            if valence > 0.0 {
                scalar += C_INCR;
            } else {
                scalar -= C_INCR;
            }
        }
    }
    scalar
}

fn sum_sentiment_scores(scores: &[f64]) -> (f64, f64, u32) {
    let (mut pos_sum, mut neg_sum, mut neu_count) = (0f64, 0f64, 0);
    for &score in scores {
        if score > 0f64 {
            pos_sum += score + 1.0;
        } else if score < 0f64 {
            neg_sum += score - 1.0;
        } else {
            neu_count += 1;
        }
    }
    (pos_sum, neg_sum, neu_count)
}

pub struct SentimentIntensityAnalyzer<'a> {
    lexicon: &'a HashMap<UniCase<&'a str>, f64>,
}

impl<'a> SentimentIntensityAnalyzer<'a> {
    pub fn new() -> SentimentIntensityAnalyzer<'static> {
        SentimentIntensityAnalyzer {
            lexicon: &LEXICON,
        }
    }

    pub fn from_lexicon<'b>(lexicon: &'b HashMap<UniCase<&str>, f64>) ->
                                        SentimentIntensityAnalyzer<'b> {
        SentimentIntensityAnalyzer {
            lexicon,
        }
    }

    fn get_total_sentiment(&self, sentiments: &[f64], punct_emph_amplifier: f64) -> SentimentScores {
        let (mut neg, mut neu, mut pos, mut compound) = (0f64, 0f64, 0f64, 0f64);
        if !sentiments.is_empty() {
            let mut total_sentiment: f64 = sentiments.iter().sum();
            if total_sentiment > 0f64 {
                total_sentiment += punct_emph_amplifier;
            } else {
                total_sentiment -= punct_emph_amplifier;
            }
            compound = normalize_score(total_sentiment);

            let (mut pos_sum, mut neg_sum, neu_count) = sum_sentiment_scores(sentiments);

            if pos_sum > neg_sum.abs() {
                pos_sum += punct_emph_amplifier;
            } else if pos_sum < neg_sum.abs() {
                neg_sum -= punct_emph_amplifier;
            }

            let total = pos_sum + neg_sum.abs() + (neu_count as f64);
            pos = (pos_sum / total).abs();
            neg = (neg_sum / total).abs();
            neu = (neu_count as f64 / total).abs();
        }
        SentimentScores {
            negative: neg,
            neutral: neu,
            positive: pos,
            compound,
        }
    }

    pub fn polarity_scores(&self, text: &str) -> SentimentScores {
        let text = append_emoji_descriptions(text);
        let parsedtext = ParsedText::from_text(&text);
        let tokens = &parsedtext.tokens;

        // Pre-compute lexicon and booster values for all tokens (eliminates
        // redundant HashMap lookups in inner loops)
        let lex_vals: Vec<Option<f64>> = tokens.iter()
            .map(|t| self.lexicon.get(t).copied())
            .collect();
        let boost_vals: Vec<Option<f64>> = tokens.iter()
            .map(|t| BOOSTER_DICT.get(t).copied())
            .collect();

        let mut sentiments = Vec::with_capacity(tokens.len());

        for (i, word) in tokens.iter().enumerate() {
            if boost_vals[i].is_some() {
                sentiments.push(0f64);
            } else if i < tokens.len() - 1 && word == &*STATIC_KIND
                                  && tokens[i + 1] == *STATIC_OF {
                sentiments.push(0f64);
            } else {
                sentiments.push(self.sentiment_valence(&parsedtext, word, i, &lex_vals, &boost_vals));
            }
        }
        but_check(tokens, &mut sentiments);
        self.get_total_sentiment(&sentiments, parsedtext.punc_amplifier)
    }

    fn sentiment_valence(&self, parsed: &ParsedText, word: &UniCase<&str>, i: usize,
                         lex_vals: &[Option<f64>], boost_vals: &[Option<f64>]) -> f64 {
        let mut valence = 0f64;
        let tokens = &parsed.tokens;
        let word_valence = lex_vals[i];
        if let Some(wv) = word_valence {
            valence = wv;
            if is_all_caps(word.as_ref()) && parsed.has_mixed_caps {
                if valence > 0f64 {
                    valence += C_INCR;
                } else {
                    valence -= C_INCR
                }
            }
            for start_i in 0..3 {
                if i > start_i && lex_vals[i - start_i - 1].is_none() {
                    let j = i - start_i - 1;
                    let mut s = scalar_inc_dec(&tokens[j], boost_vals[j], valence, parsed.has_mixed_caps);
                    if start_i == 1 {
                        s *= 0.95;
                    } else if start_i == 2 {
                        s *= 0.9
                    }
                    valence += s;
                    valence = negation_check(valence, tokens, start_i, i);
                    if start_i == 2 {
                        valence = special_idioms_check(valence, tokens, i, boost_vals);
                    }
                }
            }
            valence = least_check(valence, tokens, i, lex_vals);
        }

        // "no" as current word: neutralize when followed by a lexicon word
        if *word == *STATIC_NO
            && i < tokens.len() - 1
            && lex_vals[i + 1].is_some() {
            valence = 0.0;
        }

        // "no" preceding current word: negate using raw lexicon valence
        if let Some(base) = word_valence {
            if (i > 0 && tokens[i - 1] == *STATIC_NO)
                || (i > 1 && tokens[i - 2] == *STATIC_NO)
                || (i > 2 && tokens[i - 3] == *STATIC_NO
                    && (tokens[i - 1] == *STATIC_OR || tokens[i - 1] == *STATIC_NOR)) {
                valence = base * NEGATION_SCALAR;
            }
        }

        valence
    }
}

// Removes emoji and appends their description using Aho-Corasick SIMD-accelerated matching
// Returns Cow::Borrowed when no emojis found (zero-alloc fast path)
fn append_emoji_descriptions(text: &str) -> Cow<'_, str> {
    let (ref ac, ref descriptions) = *EMOJI_AC;
    let mut iter = ac.find_iter(text);
    let first = match iter.next() {
        Some(m) => m,
        None => return Cow::Borrowed(text),
    };
    let mut result = String::with_capacity(text.len() + text.len() / 2);
    let mut last_end = 0;
    // Process first match
    result.push_str(&text[last_end..first.start()]);
    if !result.is_empty() && !result.ends_with(' ') {
        result.push(' ');
    }
    result.push_str(descriptions[first.pattern().as_usize()]);
    last_end = first.end();
    // Process remaining matches
    for mat in iter {
        result.push_str(&text[last_end..mat.start()]);
        if !result.is_empty() && !result.ends_with(' ') {
            result.push(' ');
        }
        result.push_str(descriptions[mat.pattern().as_usize()]);
        last_end = mat.end();
    }
    result.push_str(&text[last_end..]);
    Cow::Owned(result)
}

/// Check for specific patterns or tokens, and modify sentiment as needed
fn negation_check(valence: f64, tokens: &[UniCase<&str>], start_i: usize, i: usize) -> f64 {
    let mut valence = valence;
    if start_i == 0 {
        if is_negated(&tokens[i - start_i - 1]) {
            valence *= NEGATION_SCALAR;
        }
    } else if start_i == 1 {
        if tokens[i - 2] == *STATIC_NEVER &&
          (tokens[i - 1] == *STATIC_SO ||
           tokens[i - 1] == *STATIC_THIS) {
            valence *= 1.25
        } else if tokens[i - 2] == *STATIC_WITHOUT && tokens[i - 1] == *STATIC_DOUBT {
            valence *= 1.0
        } else if is_negated(&tokens[i - start_i - 1]) {
            valence *= NEGATION_SCALAR;
        }
    } else if start_i == 2 {
        if tokens[i - 3] == *STATIC_NEVER &&
           tokens[i - 2] == *STATIC_SO || tokens[i - 2] == *STATIC_THIS||
           tokens[i - 1] == *STATIC_SO || tokens[i - 1] == *STATIC_THIS {
            valence *= 1.25
        } else if tokens[i - 3] == *STATIC_WITHOUT &&
                  tokens[i - 2] == *STATIC_DOUBT ||
                  tokens[i - 1] == *STATIC_DOUBT {
            valence *= 1.0;
        } else if is_negated(&tokens[i - start_i - 1]) {
            valence *= NEGATION_SCALAR;
        }
    }
    valence
}

// If "but" is in the tokens, scales down the sentiment of words before "but" and
// adds more emphasis to the words after
fn but_check(tokens: &[UniCase<&str>], sentiments: &mut Vec<f64>) {
    match tokens.iter().position(|&s| s == *STATIC_BUT) {
        Some(but_index) => {
            for i in 0..sentiments.len() {
                if i < but_index {
                    sentiments[i] *= 0.5;
                } else if i > but_index {
                    sentiments[i] *= 1.5;
                }
            }
        },
        None => return,
    }
}

// Fixed: original had impossible `tokens[i-2] == AT && tokens[i-2] == VERY` condition.
// Python logic: if "least" precedes and is NOT in lexicon, negate — unless preceded by "at" or "very".
fn least_check(valence: f64, tokens: &[UniCase<&str>], i: usize, lex_vals: &[Option<f64>]) -> f64 {
    let mut valence = valence;
    if i > 1 && lex_vals[i - 1].is_none()
             && tokens[i - 1] == *STATIC_LEAST {
        if tokens[i - 2] != *STATIC_AT && tokens[i - 2] != *STATIC_VERY {
            valence *= NEGATION_SCALAR;
        }
    } else if i > 0 && lex_vals[i - 1].is_none()
                     && tokens[i - 1] == *STATIC_LEAST {
        valence *= NEGATION_SCALAR;
    }
    valence
}

// Zero-allocation special idioms check
// Uses pre-split idiom word sequences instead of joining tokens into strings
// Uses direct BOOSTER_DICT lookups instead of iterating all entries
fn special_idioms_check(valence: f64, tokens: &[UniCase<&str>], i: usize, boost_vals: &[Option<f64>]) -> f64 {
    assert!(i > 2);
    let mut valence = valence;
    let mut end_i = i + 1;

    if tokens.len() - 1 > i {
        end_i = min(i + 3, tokens.len());
    }

    let window = &tokens[(i - 3)..end_i];

    // Check special case idioms using pre-split word sequences (zero-alloc)
    for (idiom_words, val) in SPECIAL_IDIOMS_SPLIT.iter() {
        let n = idiom_words.len();
        if window.len() >= n {
            for w in window.windows(n) {
                if w.iter().zip(idiom_words.iter()).all(|(a, b)| a == b) {
                    valence = *val;
                    break;
                }
            }
        }
    }

    // Check previous 3 tokens using pre-computed booster values (O(1) array access)
    for j in (i - 3)..i {
        if let Some(val) = boost_vals[j] {
            valence += val;
        }
    }

    valence
}

// Rayon-powered batch analysis
#[cfg(feature = "parallel")]
pub fn polarity_scores_batch(texts: &[&str]) -> Vec<SentimentScores> {
    use rayon::prelude::*;
    let analyzer = SentimentIntensityAnalyzer::new();
    texts.par_iter()
        .map(|text| analyzer.polarity_scores(text))
        .collect()
}

pub mod demo;
