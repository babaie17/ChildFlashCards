// api/assess.js
export const config = { runtime: 'edge' };

/**
 * POST multipart/form-data:
 *  - audio: Blob (wav/ogg/webm/mp3)  [<= 60s recommended]
 *  - expected: string                [target word/character]
 *  - language: BCP-47 (e.g., "en-US", "zh-CN")
 *
 * Env:
 *  - AZURE_SPEECH_KEY
 *  - AZURE_REGION (e.g., "eastus")
 *
 * Returns JSON:
 *  { pass:boolean, score:number(0..1), tone?:{expected,heard,match}|null, hint?:string|null, provider:"azure-assess" }
 */
export default async function handler(req) {
  try {
    if (req.method === 'OPTIONS') return new Response(null, { status: 204, headers: cors() });
    if (req.method !== 'POST') return json({ error: 'POST only' }, 405);

    const form = await req.formData();
    const file = form.get('audio');
    const expected = (form.get('expected') || '').toString();
    const language = (form.get('language') || 'en-US').toString();

    if (!file || typeof file.arrayBuffer !== 'function') return json({ error: 'No audio' }, 200);
    if (!expected) return json({ error: 'Missing expected' }, 200);

    const key = process.env.AZURE_SPEECH_KEY;
    const region = process.env.AZURE_REGION || 'eastus';
    if (!key) return json({ error: 'AZURE_SPEECH_KEY missing' }, 200);

    // Build Pronunciation Assessment header (base64-encoded JSON)
    // HundredMark => 0..100 score; Phoneme granularity => better hints/tones when available.
    const pa = {
      ReferenceText: expected,
      GradingSystem: 'HundredMark',
      Granularity: 'Phoneme',              // or "Word"
      Dimension: 'Comprehensive',          // "Accuracy" | "Fluency" | "Completeness" | "Prosody" | "Comprehensive"
      EnableProsodyAssessment: 'True'
    };
    const paHeader = btoa(JSON.stringify(pa));

    // Accept common browser MIME types; wav (PCM 16k) tends to work best.
    const contentType = pickContentType(file.type);

    // Use short-audio endpoints; try conversation then interactive as fallback.
    const base = `https://${region}.stt.speech.microsoft.com/speech/recognition`;
    const endpoints = [
      `${base}/conversation/cognitiveservices/v1?language=${encodeURIComponent(language)}&format=detailed&profanity=raw`,
      `${base}/interactive/cognitiveservices/v1?language=${encodeURIComponent(language)}&format=detailed&profanity=raw`
    ];

    let lastErr = null, best = null, rawResp = null;
    for (const url of endpoints) {
      const r = await fetch(url, {
        method: 'POST',
        headers: {
          'Ocp-Apim-Subscription-Key': key,
          'Content-Type': contentType,
          'Accept': 'application/json',
          'Pronunciation-Assessment': paHeader
        },
        body: file
      });
      const body = await safeBody(r);
      if (!r.ok) { lastErr = { status: r.status, body }; continue; }

      rawResp = body;
      const parsed = parseAssessment(body);
      if (parsed) { best = parsed; break; }
      lastErr = { status: r.status, body };
    }

    if (!best) {
      return json({
        error: 'Azure assessment error',
        diagnostics: lastErr || rawResp || null
      }, 200);
    }

    // Convert 0..100 to 0..1, choose pass threshold (tune to taste)
    const languagePrimary = (language.split('-')[0] || '').toLowerCase();
    const rawScore = clamp0to100(best.accuracy ?? 0);
    const score = +(rawScore / 100).toFixed(4);

    // Default thresholds: zh a bit stricter for single characters, en general
    const passThreshold = languagePrimary === 'zh' ? 0.80 : 0.78;
    let pass = score >= passThreshold;

    // Tone extraction (heuristic): if available in phoneme stream
    let tone = null;
    if (languagePrimary === 'zh') {
      const heardTone = guessToneFromPhones(best.words);
      const expectedTone = null; // (Optional) integrate your hanzi_to_pinyin map here if you want.
      if (heardTone || expectedTone) {
        tone = { expected: expectedTone, heard: heardTone, match: (expectedTone && heardTone) ? expectedTone === heardTone : null };
      }
    }

    // Simple friendly hint when not passing (use Azure sub-scores if present)
    const hint = pass ? null : buildHint(best, languagePrimary);

    return json({
      pass,
      score,
      tone,
      hint,
      provider: 'azure-assess'
    }, 200);

  } catch (e) {
    return json({ error: String(e?.message || e) }, 200);
  }
}

/* ---------------- Helpers ---------------- */
function cors() {
  return {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type'
  };
}
function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), { status, headers: { 'Content-Type': 'application/json', ...cors() } });
}
async function safeBody(r) {
  const ct = (r.headers.get('content-type') || '').toLowerCase();
  if (ct.includes('application/json')) return r.json();
  return { message: await r.text() };
}
function pickContentType(m) {
  const t = (m || '').toLowerCase();
  if (t.includes('wav'))  return 'audio/wav';                        // ideally PCM 16k
  if (t.includes('ogg'))  return 'audio/ogg; codecs=opus';
  if (t.includes('webm')) return 'audio/webm; codecs=opus';
  if (t.includes('mp3'))  return 'audio/mpeg';
  return 'application/octet-stream';
}
function clamp0to100(n) {
  const x = Number(n); if (!Number.isFinite(x)) return 0;
  return Math.max(0, Math.min(100, x));
}

// Parse the common Azure "format=detailed" shapes with PronunciationAssessment blocks
function parseAssessment(body) {
  // Shapes vary; normalize to first-best result
  let nbest = null;
  if (Array.isArray(body?.NBest) && body.NBest[0]) {
    nbest = body.NBest[0];
  } else if (Array.isArray(body?.results) && Array.isArray(body.results[0]?.NBest) && body.results[0].NBest[0]) {
    nbest = body.results[0].NBest[0];
  }
  if (!nbest) return null;

  const PA = nbest.PronunciationAssessment || {};
  const accuracy = PA.AccuracyScore ?? PA.OverallScore ?? 0; // sometimes OverallScore exists
  const words = Array.isArray(nbest.Words) ? nbest.Words : [];

  return { accuracy, words, raw: nbest };
}

// Heuristic: try to detect a Mandarin tone digit from phoneme stream if present
function guessToneFromPhones(words) {
  try {
    const s = JSON.stringify(words || []).toLowerCase();
    // crude: look for first standalone digit 1-5 (depends on locale/model details)
    const m = s.match(/[^0-9]([1-5])[^0-9]/);
    return m ? m[1] : null;
  } catch { return null; }
}

function buildHint(best, lang) {
  // Use sub-scores if Azure returns them (often AccuracyScore, FluencyScore, CompletenessScore)
  const PA = best?.raw?.PronunciationAssessment || {};
  const acc = Math.round(PA.AccuracyScore ?? 0);
  const flu = Math.round(PA.FluencyScore ?? 0);
  const comp = Math.round(PA.CompletenessScore ?? 0);

  if (lang === 'zh') {
    if (acc < 70) return '放慢一点，注意声母与韵母的发音';
    return '注意声调变化，再试一次';
  }
  // English default hints
  if (acc < 70) return 'Try slower and enunciate the vowel sound';
  if (flu < 70) return 'Try a steadier pace, less hesitation';
  if (comp < 70) return 'Say the whole word clearly';
  return 'Try a bit clearer and slower';
}
