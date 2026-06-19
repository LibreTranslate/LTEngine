use once_cell::sync::Lazy;
use serde::Serialize;
use std::collections::HashMap;
use whatlang::{Detector, Lang};

const LANGS: &[(&str, &str, &str)] = &[
    ("en", "", "English"),
    ("sq", "", "Albanian"),
    ("ar", "", "Arabic"),
    ("az", "", "Azerbaijani"),
    ("eu", "", "Basque"),
    ("bn", "", "Bengali"),
    ("bg", "", "Bulgarian"),
    ("ca", "", "Catalan"),
    ("zh", "zh-Hans", "Chinese"),
    ("zt", "zh-Hant", "Chinese (traditional)"),
    ("cs", "", "Czech"),
    ("da", "", "Danish"),
    ("nl", "", "Dutch"),
    ("eo", "", "Esperanto"),
    ("et", "", "Estonian"),
    ("fi", "", "Finnish"),
    ("fr", "", "French"),
    ("gl", "", "Galician"),
    ("de", "", "German"),
    ("el", "", "Greek"),
    ("he", "", "Hebrew"),
    ("hi", "", "Hindi"),
    ("hu", "", "Hungarian"),
    ("id", "", "Indonesian"),
    ("ga", "", "Irish"),
    ("it", "", "Italian"),
    ("ja", "", "Japanese"),
    ("ko", "", "Korean"),
    ("lv", "", "Latvian"),
    ("lt", "", "Lithuanian"),
    ("ms", "", "Malay"),
    ("nb", "", "Norwegian"),
    ("fa", "", "Persian"),
    ("pl", "", "Polish"),
    ("pt", "", "Portuguese"),
    ("pb", "pt-BR", "Portuguese (Brazil)"),
    ("ro", "", "Romanian"),
    ("ru", "", "Russian"),
    ("sr", "", "Serbian"),
    ("sk", "", "Slovak"),
    ("sl", "", "Slovenian"),
    ("es", "", "Spanish"),
    ("sv", "", "Swedish"),
    ("tl", "", "Tagalog"),
    ("th", "", "Thai"),
    ("tr", "", "Turkish"),
    ("uk", "", "Ukrainian"),
    ("ur", "", "Urdu"),
    ("vi", "", "Vietnamese"),
];

#[derive(Serialize)]
pub struct Language {
    pub code: &'static str,
    pub name: &'static str,
    pub targets: &'static [&'static str],

    #[serde(skip)]
    pub lang_detect: Option<&'static Lang>,

    #[serde(skip)]
    pub internal_code: &'static str,
}

pub static LANGUAGES: Lazy<Vec<Language>> = Lazy::new(|| {
    // From whatlang names to our names
    let eng_name_map: HashMap<&'static str, &'static str> =
        [("Mandarin", "Chinese")].iter().cloned().collect();

    let mut lang_detect_map: HashMap<&'static str, &'static Lang> = HashMap::new();
    for lang in Lang::all() {
        let eng_name = lang.eng_name();
        lang_detect_map.insert(eng_name_map.get(eng_name).unwrap_or(&eng_name), lang);
    }

    LANGS
        .iter()
        .map(|&(code, alias, name)| {
            let targets: Vec<&str> = LANGS
                .iter()
                .filter_map(|&(c, a, _)| Some(if a != "" { a } else { c }))
                .collect();

            Language {
                code: if alias != "" { alias } else { code },
                name,
                targets: Box::leak(targets.into_boxed_slice()),
                lang_detect: lang_detect_map.get(name).map(|v| &**v),
                internal_code: code,
            }
        })
        .collect()
});

static LANGUAGES_MAP: Lazy<HashMap<&'static str, &'static Language>> = Lazy::new(|| {
    LANGUAGES
        .iter()
        .map(|lang| (lang.internal_code, lang))
        .collect()
});

static CODE_TO_INTERNAL_CODE_MAP: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    LANGUAGES
        .iter()
        .map(|lang| (lang.code, lang.internal_code))
        .collect()
});

pub fn get_language_from_code(code: &String) -> Option<&'static Language> {
    let code_str = code.as_str();
    let internal_code = CODE_TO_INTERNAL_CODE_MAP.get(code_str).unwrap_or(&code_str);
    LANGUAGES_MAP.get(internal_code).map(|v| &**v)
}

pub struct LangDetect {
    pub language: &'static Language,
    pub confidence: i32,
}

pub fn detect_lang(q: &String) -> LangDetect {
    let allowlist: Vec<Lang> = LANGUAGES
        .iter()
        .filter_map(|l| l.lang_detect.copied())
        .collect();

    let detector = Detector::with_allowlist(allowlist);
    if let Some(info) = detector.detect(q) {
        let lang = info.lang();
        let confidence = info.confidence();

        LANGUAGES
            .iter()
            .find(|l| l.lang_detect == Some(&lang))
            .map(|l| LangDetect {
                language: l,
                confidence: (confidence * 100.0) as i32,
            })
            .unwrap_or(LangDetect {
                language: &LANGUAGES[0],
                confidence: 0,
            })
    } else {
        LangDetect {
            language: &LANGUAGES[0],
            confidence: 0,
        }
    }
}
