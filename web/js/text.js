// Text helpers, ported from triage/preprocess.py so explanations and keyword
// matches behave the same in the browser and in the Python engine.

const TOKEN = /[a-z][a-z0-9]+/g;

export const STOPWORDS = new Set(`a about above after again against all also am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further had has
have having he her here hers herself him himself his how i if in into is it its itself just me more
most my myself no nor not now of off on once only or other our ours ourselves out over own same she
should so some such than that the their theirs them themselves then there these they this those
through to too under until up very was we were what when where which while who whom why will with
would you your yours yourself yourselves via using use used based new novel approach approaches
method methods paper propose proposed show shows results result present presents study work
towards toward study studies however thus therefore while well within without across among many
several various different one two three first second large small high low et al e g i e`.split(/\s+/));

export const rawTokens = text => (String(text || "").toLowerCase().match(TOKEN) || []);
export const tokenize = text => rawTokens(text).filter(t => !STOPWORDS.has(t));

/** Tiny suffix stripper so "models" matches "model", "learning" ≈ "learn". */
export function stem(word) {
  for (const suf of ["ings", "ing", "ies", "es", "s", "ed"]) {
    if (word.length > suf.length + 3 && word.endsWith(suf)) return word.slice(0, -suf.length) + (suf === "ies" ? "y" : "");
  }
  return word;
}

export const stemTokens = text => tokenize(text).map(stem);

/**
 * True if the phrase's content words occur in `text` in order, each within
 * `maxGap` words of the previous one (stem match). Phrases of 3+ words also
 * match their acronym ("retrieval augmented generation" → "RAG").
 * `text` may be a string or a textCache() of it (faster when testing many phrases).
 */
export function phraseInText(phrase, text, maxGap = 1) {
  const p = stemTokens(phrase);
  if (!p.length) return false;
  const cache = typeof text === "string" || text == null ? textCache(text) : text;
  const toks = cache.stems;
  let found = false;
  if (p.length === 1) found = toks.includes(p[0]);
  else {
    outer: for (let i = 0; i < toks.length; i++) {
      if (toks[i] !== p[0]) continue;
      let pos = i;
      for (const w of p.slice(1)) {
        const window = toks.slice(pos + 1, pos + 2 + maxGap);
        const j = window.indexOf(w);
        if (j < 0) continue outer;
        pos = pos + 1 + j;
      }
      found = true;
      break;
    }
  }
  if (found) return true;
  const words = rawTokens(phrase);
  if (words.length >= 3) {
    const acronym = words.map(w => w[0]).join("");
    return cache.raw.has(acronym);
  }
  return false;
}

/** Tokens of a text, prepared once for repeated phraseInText() calls. */
export function textCache(text) {
  const raw = rawTokens(text);
  return { stems: raw.map(stem), raw: new Set(raw) };
}

export function splitSentences(text) {
  return String(text || "").split(/(?<=[.!?])\s+(?=[A-Z0-9])/).map(s => s.trim()).filter(s => s.length > 20);
}

/** "a, b; c\nd" → ["a","b","c","d"], de-duplicated case-insensitively. */
export function parseList(text) {
  const seen = new Set();
  const out = [];
  for (const s of String(text || "").split(/[,;\n]/).map(x => x.trim())) {
    if (s && !seen.has(s.toLowerCase())) { seen.add(s.toLowerCase()); out.push(s); }
  }
  return out;
}

export const titleKey = title => String(title || "").toLowerCase().replace(/[^a-z0-9]+/g, "");
