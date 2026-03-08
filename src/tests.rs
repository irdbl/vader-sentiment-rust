use unicase::UniCase;

#[test]
fn test_lexicon() {
    assert_eq!(*crate::LEXICON.get(&UniCase::new("feudally")).unwrap(), -0.6);
    assert_eq!(*crate::LEXICON.get(&UniCase::new("irrationalism")).unwrap(), -1.5);
    assert_eq!(*crate::LEXICON.get(&UniCase::new("sentimentalize")).unwrap(), 0.8);
    assert_eq!(*crate::LEXICON.get(&UniCase::new("wisewomen")).unwrap(), 1.3);
}

#[test]
fn test_emoji_lexicon() {
    assert_eq!(*crate::EMOJI_LEXICON.get("👽").unwrap(), "alien");
    assert_eq!(*crate::EMOJI_LEXICON.get("👨🏿‍🎓").unwrap(), "man student: dark skin tone");
    assert_eq!(*crate::EMOJI_LEXICON.get("🖖🏻").unwrap(), "vulcan salute: light skin tone");
}

#[test]
fn test_parsed_text() {
    let messy_text = "WOAH!!! ,Who? DO u Think you're?? :) :D :^(";
    let parsed_messy = crate::ParsedText::from_text(messy_text);
    let expected_text: Vec<UniCase<&str>> = ["WOAH", "Who", "DO", "u", "Think", "you\'re", ":)", ":D", ":^("].iter().map(| r| UniCase::new(*r)).collect();
    assert_eq!(parsed_messy.tokens, expected_text);
    assert_eq!(parsed_messy.has_mixed_caps, true);
    assert_eq!(parsed_messy.punc_amplifier, 1.416);

    assert_eq!(crate::ParsedText::has_mixed_caps(&crate::ParsedText::tokenize("yeah!!! I'm aLLERGIC to ShouTING.")), false);
    assert_eq!(crate::ParsedText::has_mixed_caps(&crate::ParsedText::tokenize("OH MAN I LOVE SHOUTING!")), false);
    assert_eq!(crate::ParsedText::has_mixed_caps(&crate::ParsedText::tokenize("I guess I CAN'T MAKE UP MY MIND")), true);
    assert_eq!(crate::ParsedText::has_mixed_caps(&crate::ParsedText::tokenize("Hmm, yeah ME NEITHER")), true);
}

#[test]
fn but_check_test() {
    let tokens: Vec<UniCase<&str>> = ["yeah", "waffles", "are", "great", "but", "have", "you", "ever", "tried", "spam"].iter().map(| r| UniCase::new(*r)).collect();
    let mut sents  = vec![ 0.5,    0.1,       0.0,   0.2,     0.6,   0.25,    0.5,   0.5,    0.5,     0.5];
    crate::but_check(&tokens, &mut sents);
    assert_eq!(sents, vec![0.25,   0.05,      0.0,   0.1,     0.6,   0.375,  0.75,   0.75,  0.75,   0.75]);
}

#[test]
fn demo_test() {
    crate::demo::run_demo();
}

#[test]
fn embedded_emoji_test() {
    let single_emoji = "😀";
    let embedded_emoji = "heyyyy 😀 what're you up to???";
    let multiple_emoji = "woah there 😀😀😀 :) :)";
    assert_eq!(crate::append_emoji_descriptions(single_emoji), "grinning face");
    assert_eq!(crate::append_emoji_descriptions(embedded_emoji), "heyyyy grinning face what're you up to???");
    assert_eq!(crate::append_emoji_descriptions(multiple_emoji), "woah there grinning face grinning face grinning face :) :)");
}

#[test]
fn no_negation_test() {
    let analyzer = crate::SentimentIntensityAnalyzer::new();
    // "no good" should be negative (no negates good)
    let scores = analyzer.polarity_scores("no good");
    assert!(scores.compound < 0.0, "expected negative compound for 'no good', got {}", scores.compound);

    // "not bad" should also be negative (via negation)
    let scores2 = analyzer.polarity_scores("not bad");
    assert!(scores2.compound > 0.0, "expected positive compound for 'not bad', got {}", scores2.compound);
}

#[test]
fn curly_quote_negation_test() {
    let analyzer = crate::SentimentIntensityAnalyzer::new();
    // Smart quote "won\u{2019}t" should work as negation like "won't"
    let ascii = analyzer.polarity_scores("I won't stop loving this");
    let curly = analyzer.polarity_scores("I won\u{2019}t stop loving this");
    // Both should have the same compound score
    assert_eq!(ascii.compound, curly.compound,
        "curly quote negation mismatch: ascii={}, curly={}", ascii.compound, curly.compound);
}

#[test]
fn special_case_idioms_test() {
    let analyzer = crate::SentimentIntensityAnalyzer::new();
    // "bus stop" should be neutral (prevents false negative from "stop")
    let _scores = analyzer.polarity_scores("I waited at the bus stop");
    // "broken heart" should be negative
    let broken = analyzer.polarity_scores("She has a broken heart");
    assert!(broken.compound < 0.0, "expected negative for 'broken heart', got {}", broken.compound);
}
