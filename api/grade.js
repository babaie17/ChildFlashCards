// api/grade.js
export const config = { runtime: 'edge' };

export default async function handler(request) {
  const t0 = Date.now();
  try {
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: corsHeaders() });
    }
    if (request.method !== 'POST') {
      return json({ error: 'POST only' }, 405);
    }

    const form = await request.formData();
    const file = form.get('audio');                    // Blob
    const expected = (form.get('expected') || '').toString();
    const language = (form.get('language') || 'en-US').toString();
    const provider = (form.get('provider') || 'openai').toString().toLowerCase();

    if (!file || typeof file.arrayBuffer !== 'function') {
      return json({ error: 'No audio uploaded' }, 400);
    }
    if (!expected) {
      return json({ error: 'Missing expected' }, 400);
    }

    // ---------- 1) ASR ----------
    const asrStart = Date.now();
    let candidates = [];
    let asrProvider = provider;

    if (asrProvider === 'openai') {
      const key = process.env.OPENAI_API_KEY;
      if (!key) return json({ error: 'OPENAI_API_KEY missing' }, 500);

      const fd = new FormData();
      // Better filenames help OpenAI infer formats
      const t = (file.type || '');
      const name =
        t.includes('wav')  ? 'speech.wav' :
        t.includes('ogg')  ? 'speech.ogg' :
        t.includes('webm') ? 'speech.webm' : 'audio.bin';
      fd.append('file', file, name);
      fd.append('language', toWhisperLang(language));   // lock language
      fd.append('model', 'gpt-4o-transcribe');

      const r = await fetch('https://api.openai.com/v1/audio/transcriptions', {
        method: 'POST',
        headers: { Authorization: `Bearer ${key}` },
        body: fd
      });
      const body = await safeBody(r);
      if (!r.ok) return json({ error: errString(body, 'OpenAI ASR error') }, r.status);

      const text = (body?.text || '').trim();
      candidates = normalizeCandidates(text ? [text] : []);
    } else if (asrProvider === 'azure') {
      const azKey = process.env.AZURE_SPEECH_KEY;
      const azRegion = process.env.AZURE_REGION || 'eastus';
      if (!azKey) return json({ error: 'AZURE_SPEECH_KEY missing' }, 500);

      const blobType = (file.type || '').toLowerCase();
      const contentType =
        blobType.includes('ogg')  ? 'audio/ogg; codecs=opus' :
        blobType.includes('webm') ? 'audio/webm; codecs=opus' :
        blobType.includes('wav')  ? 'audio/wav; codecs=audio/pcm; samplerate=16000' :
                                    'application/octet-stream';

      const endpoints = [
        'recognition/conversation/cognitiveservices/v1',
        'recognition/interactive/cognitiveservices/v1',
        'recognition/dictation/cognitiveservices/v1'
      ];
      let success = false, lastErr = null;
      for (const path of endpoints) {
        const url = `https://${azRegion}.stt.speech.microsoft.com/speech/${path}?language=${encodeURIComponent(language)}&format=detailed&profanity=raw`;
        const r = await fetch(url, {
          method: 'POST',
          headers: {
            'Ocp-Apim-Subscription-Key': azKey,
            'Content-Type': contentType,
            'Accept': 'application/json',
          },
          body: file
        });
        const body = await safeBody(r);
        if (!r.ok) { lastErr = errString(body, 'Azure ASR error'); continue; }
        candidates = extractAzureCandidates(body).map(s => s.trim()).filter(Boolean).slice(0, 5);
        candidates = strictLanguageFilter(candidates, language);
        candidates = normalizeCandidates(candidates);
        if (candidates.length) { success = true; break; }
      }
      if (!success) return json({ error: errString(lastErr, 'No speech recognized') }, 200);
    } else {
      // fallback to openai if unknown
      asrProvider = 'openai';
    }
    const asrMs = Date.now() - asrStart;

    // ---------- helper lists (optional evidence for judge) ----------
    const zhAugment = await buildZhHomophones(request.url, candidates, language);
    const enHomophones = await buildEnHomophones(candidates, request.url, language);

    // ---------- 2) LLM Judge ----------
    const judgeStart = Date.now();
    const key = process.env.OPENAI_API_KEY;
    if (!key) return json({ error: 'OPENAI_API_KEY missing' }, 500);

    const sys =
`You are a strict but kind pronunciation judge for a language learning app.
Return STRICT JSON with keys: pass (boolean), score (0..1), tone (object or null), hint (string or null).
Rules:
- Consider the target "expected" and language.
- Use ASR candidates and helper lists as evidence.
- If language starts with zh, judge by Mandarin pronunciation (pinyin + tone). Be tone-aware if a single Hanzi is expected.
- If unsure, prefer pass=false and give a short, kind, actionable hint (<=80 chars).
- score: 1.0 = perfect match; 0.0 = unrelated.
- tone: { expected: '1|2|3|4|5|null', heard: '1|2|3|4|5|null', match: boolean } or null for non-tonal languages.`;

    const user = {
      language,
      expected,
      asr: {
        provider: asrProvider,
        candidates,
        zhAugment,
        enHomophones
      }
    };

    const r2 = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${key}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: 'gpt-4o-mini',
        response_format: { type: 'json_object' },
        temperature: 0.2,
        messages: [
          { role: 'system', content: sys },
          { role: 'user', content: JSON.stringify(user) }
        ]
      })
    });
    const body2 = await safeBody(r2);
    if (!r2.ok) return json({ error: errString(body2, 'OpenAI judge error') }, r2.status);

    let verdict = {};
    try {
      verdict = JSON.parse(body2?.choices?.[0]?.message?.content || '{}');
    } catch {}

    const pass = Boolean(verdict.pass);
    const score = clamp01(Number(verdict.score));
    const tone = normalizeTone(verdict.tone);
    const hint = typeof verdict.hint === 'string' ? verdict.hint.slice(0, 120) : null;

    const totalMs = Date.now() - t0;
    const judgeMs = Date.now() - judgeStart;

    return json({
      pass, score, tone, hint,
      transcriptTop: candidates[0] || '',
      candidates,
      zhAugment,
      enHomophones,
      timings: { asr_ms: asrMs, judge_ms: judgeMs, total_ms: totalMs },
      provider: asrProvider
    }, 200);

  } catch (err) {
    return json({ error: String(err?.message || err) }, 500);
  }
}

/* -------- helpers (same as your transcribe.js, trimmed) -------- */
function clamp01(x) { if (!Number.isFinite(x)) return 0; return Math.max(0, Math.min(1, x)); }
function normalizeTone(t) {
  if (!t || typeof t !== 'object') return null;
  const toS = v => (v===null||v===undefined) ? null : String(v);
  const exp = toS(t.expected); const heard = toS(t.heard);
  let match = t.match; if (typeof match !== 'boolean') match = (exp && heard) ? (exp === heard) : null;
  return { expected: exp, heard, match };
}
function corsHeaders() {
  return {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type'
  };
}
function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), { status, headers: { 'Content-Type': 'application/json', ...corsHeaders() } });
}
function errString(body, fallback='Unknown error') {
  if (!body) return fallback;
  if (typeof body === 'string') return body;
  const m = body?.error?.message || body?.message || body?.Message || body?.RecognitionStatus;
  if (typeof m === 'string') return m;
  try { return JSON.stringify(body); } catch { return fallback; }
}
async function safeBody(r) {
  const ct = (r.headers.get('content-type') || '').toLowerCase();
  if (ct.includes('application/json')) return await r.json();
  const text = await r.text();
  return { message: text };
}

// Strip trailing punctuation for single-token strings
function stripTrailingPunctIfSingle(s) {
  if (!s) return s;
  const t = s.trim();
  if (t.includes(' ')) return t;
  return t.replace(/[\.。！？!?，,、；;：:…]+$/u, '');
}
function normalizeCandidates(arr) {
  return (Array.isArray(arr) ? arr : []).map(stripTrailingPunctIfSingle).filter(Boolean);
}
function looksLikeLatinOnly(s) {
  const hasLetter = /[A-Za-z]/.test(s);
  const hasCJK = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/u.test(s);
  return hasLetter && !hasCJK;
}
function strictLanguageFilter(cands, bcp47) {
  const primary = bcp47.split('-')[0].toLowerCase();
  if (['zh','ja','ko'].includes(primary)) {
    const filtered = cands.filter(t => !looksLikeLatinOnly(t));
    return filtered.length ? filtered : [];
  }
  return cands;
}
function toWhisperLang(bcp47) {
  const map = { 'zh-CN':'zh', 'zh-TW':'zh', 'ja-JP':'ja', 'ko-KR':'ko', 'en-US':'en', 'es-ES':'es' };
  return map[bcp47] || bcp47.split('-')[0];
}

/* zh helpers */
let HANZI_MAP = null;
const SHARD_CACHE = new Map();
async function lookupHanziReadings(baseUrl, ch) {
  if (!HANZI_MAP) {
    const url = new URL('/hanzi_to_pinyin.json', baseUrl).toString();
    const r = await fetch(url); if (!r.ok) return [];
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
function detectSinglePinyin(s) {
  const toneMap = {'ā':'a1','á':'a2','ǎ':'a3','à':'a4','ē':'e1','é':'e2','ě':'e3','è':'e4','ī':'i1','í':'i2','ǐ':'i3','ì':'i4','ō':'o1','ó':'o2','ǒ':'o3','ò':'o4','ū':'u1','ú':'u2','ǔ':'u3','ù':'u4','ǖ':'v1','ǘ':'v2','ǚ':'v3','ǜ':'v4','ü':'v' };
  let t = (s||'').trim().toLowerCase();
  if (!t || t.includes(' ')) return null;
  t = t.replace(/[āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜü]/g, m => toneMap[m] || m);
  if (!/^[a-z]+[1-5]?$/.test(t)) return null;
  if (t.length > 6) return null;
  return t;
}
async function buildZhHomophones(baseUrl, candidates, bcp47) {
  try {
    const primary = (bcp47 || '').split('-')[0].toLowerCase();
    if (primary !== 'zh') return null;
    let top = (candidates && candidates[0]) ? candidates[0].trim() : '';
    if (!top) return null;
    const hanOnly = [...top].filter(ch => /\p{Script=Han}/u.test(ch)).join('');
    if (hanOnly) top = hanOnly;
    const isSingleHan = [...top].filter(ch => /\p{Script=Han}/u.test(ch)).length === 1;
    const singlePinyin = detectSinglePinyin(top);

    if (isSingleHan) {
      const ch = [...top].find(c => /\p{Script=Han}/u.test(c));
      const readings = await lookupHanziReadings(baseUrl, ch);
      const bases = [...new Set(readings.map(r => (r.sound || '').toLowerCase()).filter(Boolean))];
      const tones = [...new Set(readings.map(r => r.tone).filter(Boolean))];
      const toneLabel = tones.length ? tones.join('/') : null;

      const homophonesSet = new Set();
      for (const b of bases) {
        const shard = await loadPinyinShard(baseUrl, b);
        for (const char of (shard[b] || [])) homophonesSet.add(char);
      }
      return { mode:'singleChar', input: ch, bases, homophones: Array.from(homophonesSet), toneLabel };
    }
    if (singlePinyin) {
      const baseKey = singlePinyin.replace(/[1-5]$/,'');
      const shard = await loadPinyinShard(baseUrl, baseKey);
      const homophones = (shard[baseKey] || []).slice();
      const toneLabel = /[1-5]$/.test(singlePinyin) ? singlePinyin.slice(-1) : null;
      return { mode:'singlePinyin', input: top, bases:[baseKey], homophones, toneLabel };
    }
    return null;
  } catch { return null; }
}

/* English homophones (English only): number words <-> digits + Datamuse */
async function buildEnHomophones(candidates, baseUrl, bcp47) {
  try {
    let top = (candidates && candidates[0] || '').trim();
    if (!top || /\s/.test(top)) return null;
    const primary = (bcp47 || '').split('-')[0].toLowerCase();
    if (primary !== 'en') return null;
    // strip trailing punctuation for single tokens
    if (!top.includes(' ')) top = top.replace(/[\.。！？!?…，,；;：:]+$/u, '');

    // number duals via your helper
    let digitForm = null, wordForm = null;
    try {
      const normURL = new URL(`/api/num-normalize?text=${encodeURIComponent(top)}`, baseUrl).toString();
      const r = await fetch(normURL, { headers: { 'accept':'application/json' } });
      if (r.ok) { const j = await r.json(); digitForm = j?.digitForm || null; wordForm = j?.wordForm || null; }
    } catch {}

    // Datamuse
    let datamuse = [];
    const queryWord = wordForm ? wordForm : (/^[a-z-]+$/i.test(top) ? top.toLowerCase() : null);
    if (queryWord) {
      const url = `https://api.datamuse.com/words?rel_hom=${encodeURIComponent(queryWord)}&max=30`;
      const r = await fetch(url, { headers: { 'Accept':'application/json' } });
      if (r.ok) {
        const items = await r.json();
        datamuse = (Array.isArray(items) ? items : []).map(x => (x?.word || '').trim()).filter(Boolean);
      }
    }
    const set = new Set(datamuse.map(w => w.toLowerCase()));
    if (wordForm)  set.add(wordForm.toLowerCase());
    if (digitForm) set.add(digitForm.toLowerCase());
    set.delete(top.toLowerCase());
    const homos = Array.from(set);
    return homos.length ? { input: top, homophones: homos.slice(0, 30) } : null;
  } catch { return null; }
}

/* Azure shape parser */
function extractAzureCandidates(data) {
  let nbest = [];
  if (Array.isArray(data?.NBest)) {
    nbest = data.NBest;
  } else if (Array.isArray(data?.results) && Array.isArray(data.results[0]?.NBest)) {
    nbest = data.results[0].NBest;
  }
  const out = [];
  const fields = ['lexical','display','itn','maskedITN','transcript','NormalizedText','Display'];
  for (const item of nbest || []) {
    for (const f of fields) {
      const v = (item?.[f] || '').toString().trim();
      if (v) { out.push(v); break; }
    }
  }
  return out;
}
