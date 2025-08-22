// api/num-normalize.js
export const config = { runtime: 'nodejs' };

import wtnMod from 'words-to-numbers';
import ntwMod from 'number-to-words';

const wordsToNumbers = typeof wtnMod === 'function' ? wtnMod : wtnMod?.wordsToNumbers;
const toWords        = typeof ntwMod === 'function' ? ntwMod : ntwMod?.toWords;

function assertDeps() {
  if (typeof wordsToNumbers !== 'function') throw new Error('words-to-numbers import failed');
  if (typeof toWords !== 'function') throw new Error('number-to-words import failed');
}

// Remove quotes and trailing sentence punctuation
function stripEnds(s) {
  return (s || '')
    .trim()
    .replace(/^["'`]+|["'`]+$/g, '')
    .replace(/[\.。！？!?…，,；;：:]+$/u, '')
    .trim();
}

// Normalize for word parsing (keep hyphens)
function normalizeForWords(s) {
  return stripEnds(s)
    .toLowerCase()
    .replace(/[,]+/g, '')
    .replace(/\s*-\s*/g, '-')
    .replace(/\s+/g, ' ');
}

// Quick gate: only try words→numbers if string contains obvious number words
const NUMBER_WORDS = new Set([
  // smalls
  'zero','one','two','three','four','five','six','seven','eight','nine',
  'ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen',
  'seventeen','eighteen','nineteen',
  // tens
  'twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety',
  // magnitudes
  'hundred','thousand','million','billion',
  // filler sometimes used
  'and'
]);

function containsNumberWords(text) {
  const tokens = text.split(/[ \-]/).filter(Boolean);
  return tokens.some(t => NUMBER_WORDS.has(t));
}

function withinRange(n) {
  // clamp to a reasonable range to avoid scientific notation & absurd values
  return Number.isSafeInteger(n) && n >= 0 && n <= 999_999_999;
}

export default async function handler(req, res) {
  try {
    assertDeps();

    const origin = `http://${req.headers.host || 'localhost'}`;
    const url = new URL(req.url || '/', origin);
    const raw0 = (url.searchParams.get('text') || '');
    const raw  = stripEnds(raw0);

    let digitForm = null;
    let wordForm  = null;

    // empty -> nothing to do
    if (!raw) {
      res.setHeader('content-type', 'application/json');
      return res.status(200).send(JSON.stringify({ digitForm, wordForm }));
    }

    // A) digits -> words (handles "144." thanks to stripEnds)
    if (/^\d+$/.test(raw)) {
      const n = parseInt(raw, 10);
      if (withinRange(n)) {
        digitForm = String(n);
        try { wordForm = toWords(n).replace(/,/g, ''); } catch { wordForm = null; }
      }
      res.setHeader('content-type', 'application/json');
      return res.status(200).send(JSON.stringify({ digitForm, wordForm }));
    }

    // B) words -> digits (only if it contains legit number words)
    const text = normalizeForWords(raw);
    if (containsNumberWords(text)) {
      const converted = wordsToNumbers(text, { fuzzy: true });
      const maybe =
        (typeof converted === 'number' && Number.isFinite(converted)) ? converted
        : (/^\d+$/.test(String(converted)) ? parseInt(String(converted), 10) : null);

      if (maybe !== null && withinRange(maybe)) {
        digitForm = String(maybe);
        try { wordForm = toWords(maybe).replace(/,/g, ''); } catch { wordForm = null; }
      }
    }

    res.setHeader('content-type', 'application/json');
    return res.status(200).send(JSON.stringify({ digitForm, wordForm }));
  } catch (err) {
    res.setHeader('content-type', 'application/json');
    return res.status(200).send(JSON.stringify({
      digitForm: null,
      wordForm: null,
      error: String(err?.message || err)
    }));
  }
}
