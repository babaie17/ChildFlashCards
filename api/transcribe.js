// api/transcribe.js
export const config = { runtime: 'edge' };

export default async function handler(request) {
  try {
    // CORS (safe for same-origin too)
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: corsHeaders() });
    }
    if (request.method !== 'POST') {
      return json({ error: 'POST only', candidates: [] }, 405);
    }

    const form = await request.formData();
    const file = form.get('audio'); // Blob from client
    const language = (form.get('language') || 'en-US').toString();
    const provider = (form.get('provider') || 'azure').toString().toLowerCase();
    const debug = (form.get('debug') || '').toString() === '1';

    if (!file || typeof file.arrayBuffer !== 'function') {
      return json({ error: 'No audio uploaded', candidates: [] }, 400);
    }

    // Guard: tiny/empty uploads
    const size = Number(file.size || 0);
    const type = (file.type || '(none)').toString();
    if (size < 2000) {
      return json({ error: `Audio too small (${size} bytes). type=${type}`, candidates: [] }, 400);
    }

    // ---------- OpenAI path (1-best) ----------
    if (provider === 'openai') {
      const key = process.env.OPENAI_API_KEY;
      if (!key) return json({ error: 'OPENAI_API_KEY missing', candidates: [] }, 500);

      const t = (file.type || '');
      const name =
        t.includes('wav')  ? 'speech.wav' :
        t.includes('ogg')  ? 'speech.ogg' :
        t.includes('webm') ? 'speech.webm' : 'audio.bin';

      const fd = new FormData();
      fd.append('file', file, name);
      fd.append('language', toWhisperLang(language));   // force language for Whisper
      fd.append('model', 'gpt-4o-transcribe');

      const r = await fetch('https://api.openai.com/v1/audio/transcriptions', {
        method: 'POST',
        headers: { Authorization: `Bearer ${key}` },
        body: fd
      });
      const body = await safeBody(r);

      if (!r.ok) return json({ error: errString(body, 'OpenAI error'), candidates: [] }, r.status);

      // Normalize candidates (strip trailing punctuation if single token)
      let candidates = normalizeCandidates([(body?.text || '').trim()]);

      // zh homophones + tone; en homophones (via Node helper)
      const zh = await buildZhHomophones(request.url, candidates, language);
      const en = await buildEnHomophones(candidates, request.url, language);
      return json({ provider: 'openai', candidates, zhAugment: zh, enHomophones: en }, 200);
    }

    // ---------- Azure path (Top-5 via format=detailed) ----------
    const azKey = process.env.AZURE_SPEECH_KEY;
    const azRegion = process.env.AZURE_REGION || 'eastus';
    if (!azKey) return json({ error: 'AZURE_SPEECH_KEY missing', candidates: [] }, 500);

    // Prefer using the actual client MIME; normalize to common Azure-accepted types
    const blobType = (file.type || '').toLowerCase();
    const isWav  = blobType.includes('wav');
    const isWebm = blobType.includes('webm');
    const isOgg  = blobType.includes('ogg');

    const contentType = isWav
      ? 'audio/wav; codecs=audio/pcm; samplerate=16000' // explicit for WAV (mono 16 kHz)
      : isWebm
        ? 'audio/webm; codecs=opus'
        : isOgg
          ? 'audio/ogg; codecs=opus'
          : 'application/octet-stream';

    const endpoints = [
      'recognition/conversation/cognitiveservices/v1',
      'recognition/interactive/cognitiveservices/v1',
      'recognition/dictation/cognitiveservices/v1'
    ];

    let lastErr = null;

    for (const path of endpoints) {
      const url = `https://${azRegion}.stt.speech.microsoft.com/speech/${path}?language=${encodeURIComponent(language)}&format=detailed&profanity=raw`;

      const r = await fetch(url, {
        method: 'POST',
        headers: {
          'Ocp-Apim-Subscription-Key': azKey,
          'Content-Type': contentType,
          'Accept': 'application/json'
        },
        body: file // send Blob directly
      });

      const body = await safeBody(r);
      if (!r.ok) { lastErr = errString(body, 'Azure error'); continue; }

      let candidates = extractAzureCandidates(body)
        .map(s => (s || '').trim())
        .filter(Boolean)
        .slice(0, 5);

      candidates = strictLanguageFilter(candidates, language);
      candidates = normalizeCandidates(candidates); // strip trailing punctuation if single token

      if (candidates.length > 0) {
        const zh = await buildZhHomophones(request.url, candidates, language);
        const en = await buildEnHomophones(candidates, request.url, language);
        return json({
          provider: 'azure',
          endpoint: path.split('/')[1],
          contentType,
          candidates,
          zhAugment: zh,
          enHomophones: en
        }, 200);
      }

      lastErr = errString(body, 'No speech recognized');
    }

    return json({ provider: 'azure', error: lastErr || 'No speech recognized', candidates: [] }, 200);

  } catch (err) {
    return json({ error: String(err?.message || err), candidates: [] }, 500);
  }
}

/* ======================= zh homophones (server-side) ======================= */
// If language starts with zh and the top candidate is either:
// - exactly one Han character -> look up its pinyin (ignoring tone) and return all homophones from /pinyin-index/<base>.json
// - a single pinyin syllable (e.g., "hǎo"/"hao3"/"hao") -> return all chars for that base
// Adds toneLabel: "3", "2/4", or null.
async function buildZhHomophones(baseUrl, candidates, bcp47) {
  try {
    const primary = (bcp47 || '').split('-')[0].toLowerCase();
    if (primary !== 'zh') return null;

    let top = (candidates && candidates[0]) ? candidates[0].trim() : '';
    if (!top) return null;

    // Strip non-Han from ends so "你。" counts as "你"
    const hanOnly = [...top].filter(ch => /\p{Script=Han}/u.test(ch)).join('');
    if (hanOnly) top = hanOnly;

    const isSingleHan = [...top].filter(ch => /\p{Script=Han}/u.test(ch)).length === 1;
    const singlePinyin = detectSinglePinyin(top); // "hao", "hao3", "hǎo" -> normalized or null

    if (isSingleHan) {
      const ch = [...top].find(c => /\p{Script=Han}/u.test(c));
      const readings = await lookupHanziReadings(baseUrl, ch); // [{sound,tone,pretty}]
      const bases = [...new Set(readings.map(r => (r.sound || '').toLowerCase()).filter(Boolean))];
      const tones = [...new Set(readings.map(r => r.tone).filter(Boolean))]; // e.g., [3] or [2,4]
      const toneLabel = tones.length ? tones.join('/') : null;

      const homophonesSet = new Set();
      for (const b of bases) {
        const shard = await loadPinyinShard(baseUrl, b);
        for (const char of (shard[b] || [])) homophonesSet.add(char);
      }
      return {
        mode: 'singleChar',
        input: ch,
        bases,
        homophones: Array.from(homophonesSet),
        toneLabel
      };
    }

    if (singlePinyin) {
      const baseKey = singlePinyin.replace(/[1-5]$/,'');
      const shard = await loadPinyinShard(baseUrl, baseKey);
      const homophones = (shard[baseKey] || []).slice();
      const toneLabel = /[1-5]$/.test(singlePinyin) ? singlePinyin.slice(-1) : null;
      return {
        mode: 'singlePinyin',
        input: top,
        bases: [baseKey],
        homophones,
        toneLabel
      };
    }

    return null;
  } catch {
    return null;
  }
}

// Cache (module-scope) for performance across invocations
let HANZI_MAP = null;              // { "好":[{sound:"hao",tone:3,pretty:"hǎo"}], ... }
const SHARD_CACHE = new Map();     // "hao" -> { hao:[...], hao1:[...], ... }

async function lookupHanziReadings(baseUrl, ch) {
  if (!HANZI_MAP) {
    const url = new URL('/hanzi_to_pinyin.json', baseUrl).toString();
    const r = await fetch(url);
    if (!r.ok) return [];
    HANZI_MAP = await r.json();
  }
  return HANZI_MAP[ch] || [];
}

async function loadPinyinShard(baseUrl, base) {
  if (SHARD_CACHE.has(base)) return SHARD_CACHE.get(base);
  const url = new URL(`/pinyin-index/${base}.json`, baseUrl).toString();
  const r = await fetch(url);
  const obj = r.ok ? await r.json() : {};
  SHARD_CACHE.set(base, obj);
  return obj;
}

/* ======================= helpers ======================= */

// English homophones (English only): number words <-> digits + Datamuse
async function buildEnHomophones(candidates, baseUrl, bcp47) {
  try {
    let top = (candidates && candidates[0] || '').trim();
    if (!top || /\s/.test(top)) return null; // single token only

    const primary = (bcp47 || '').split('-')[0].toLowerCase();
    if (primary !== 'en') return null; // ONLY do this in English

    // Trim trailing punctuation from single tokens (e.g., "Two." -> "Two")
    if (!top.includes(' ')) {
      top = top.replace(/[\.。！？!?…，,；;：:]+$/u, '');
    }

    // Ask Node helper to normalize numbers both ways
    const normURL = new URL(`/api/num-normalize?text=${encodeURIComponent(top)}`, baseUrl).toString();
    let digitForm = null, wordForm = null;
    try {
      const r = await fetch(normURL, { headers: { 'accept': 'application/json' } });
      if (r.ok) {
        const j = await r.json();
        digitForm = j?.digitForm || null;
        wordForm  = j?.wordForm || null;
      }
    } catch {}

    // Decide which word to query at Datamuse
    let queryWord = null;
    if (wordForm) {
      queryWord = wordForm;                   // if number detected, query on the word form
    } else if (/^[a-z-]+$/i.test(top)) {
      queryWord = top.toLowerCase();          // otherwise, plain ASCII word
    }

    // Fetch homophones from Datamuse
    let datamuse = [];
    if (queryWord) {
      const url = `https://api.datamuse.com/words?rel_hom=${encodeURIComponent(queryWord)}&max=30`;
      const r = await fetch(url, { headers: { 'Accept': 'application/json' } });
      if (r.ok) {
        const items = await r.json();
        datamuse = (Array.isArray(items) ? items : [])
          .map(x => (x && x.word ? x.word : '').trim())
          .filter(Boolean);
      }
    }

    // Union datamuse + digit/word forms, excluding the top token itself
    const set = new Set(datamuse.map(w => w.toLowerCase()));
    if (wordForm)  set.add(wordForm.toLowerCase());
    if (digitForm) set.add(digitForm.toLowerCase());
    set.delete(top.toLowerCase());

    const homos = Array.from(set);
    if (homos.length === 0) return null;
    return { input: top, homophones: homos.slice(0, 30) };
  } catch {
    return null;
  }
}

// Strip trailing punctuation if the string is a single token (no spaces)
function stripTrailingPunctIfSingle(s) {
  if (!s) return s;
  const t = s.trim();
  if (t.includes(' ')) return t;
  return t.replace(/[\.。！？!?，,、；;：:…]+$/u, '');
}

// Apply normalization to a list of candidates
function normalizeCandidates(arr) {
  return (Array.isArray(arr) ? arr : [])
    .map(stripTrailingPunctIfSingle)
    .filter(Boolean);
}

// Detect a single pinyin syllable like "hao", "hǎo", "hao3"
function detectSinglePinyin(s) {
  const toneMap = {
    'ā':'a1','á':'a2','ǎ':'a3','à':'a4',
    'ē':'e1','é':'e2','ě':'e3','è':'e4',
    'ī':'i1','í':'i2','ǐ':'i3','ì':'i4',
    'ō':'o1','ó':'o2','ǒ':'o3','ò':'o4',
    'ū':'u1','ú':'u2','ǔ':'u3','ù':'u4',
    'ǖ':'v1','ǘ':'v2','ǚ':'v3','ǜ':'v4','ü':'v'
  };
  let t = (s||'').trim().toLowerCase();
  if (!t || t.includes(' ')) return null;
  t = t.replace(/[āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜü]/g, m => toneMap[m] || m);
  if (!/^[a-z]+[1-5]?$/.test(t)) return null;
  if (t.length > 6) return null;
  return t;
}

function looksLikeLatinOnly(s) {
  const hasLetter = /[A-Za-z]/.test(s);
  const hasCJK = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/u.test(s);
  return hasLetter && !hasCJK;
}
function strictLanguageFilter(cands, bcp47) {
  const primary = bcp47.split('-')[0].toLowerCase();
  if (['zh', 'ja', 'ko'].includes(primary)) {
    const filtered = cands.filter(t => !looksLikeLatinOnly(t));
    return filtered.length ? filtered : [];
  }
  return cands;
}

// Map BCP-47 to Whisper’s expected ISO-639
function toWhisperLang(bcp47) {
  const map = {
    'zh-CN': 'zh', 'zh-TW': 'zh',
    'ja-JP': 'ja', 'ko-KR': 'ko',
    'en-US': 'en', 'es-ES': 'es'
  };
  return map[bcp47] || bcp47.split('-')[0];
}

function corsHeaders() {
  return {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type'
  };
}

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { 'Content-Type': 'application/json', ...corsHeaders() }
  });
}

function errString(body, fallback = 'Unknown error') {
  if (!body) return fallback;
  if (typeof body === 'string') return body;
  const m = body?.error?.message || body?.message || body?.Message || body?.RecognitionStatus;
  if (typeof m === 'string') return m;
  try { return JSON.stringify(body); } catch { return fallback; }
}

// JSON if possible; otherwise return {message: rawText}
async function safeBody(r) {
  const ct = (r.headers.get('content-type') || '').toLowerCase();
  if (ct.includes('application/json')) return await r.json();
  const text = await r.text();
  return { message: text };
}

// Pull plausible text fields from Azure result shapes
function extractAzureCandidates(data) {
  let nbest = [];
  if (Array.isArray(data?.NBest)) {
    nbest = data.NBest;
  } else if (Array.isArray(data?.results) && Array.isArray(data.results[0]?.NBest)) {
    nbest = data.results[0].NBest;
  }

  const out = [];
  const fields = ['lexical', 'display', 'itn', 'maskedITN', 'transcript', 'NormalizedText', 'Display'];
  for (const item of nbest || []) {
    for (const f of fields) {
      const v = (item?.[f] || '').toString().trim();
      if (v) { out.push(v); break; }
    }
  }
  return out;
}


