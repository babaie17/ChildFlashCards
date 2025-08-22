// api/grade_direct.js
export const config = { runtime: 'edge' };

export default async function handler(req) {
  try {
    if (req.method !== 'POST') return json({ error: 'POST only' }, 405);
    const form = await req.formData();
    const file = form.get('audio');
    const expected = (form.get('expected') || '').toString();
    const language = (form.get('language') || 'en-US').toString();
    if (!file) return json({ error: 'No audio' }, 400);
    if (!expected) return json({ error: 'Missing expected' }, 400);

    // Read audio as base64
    const ab = await file.arrayBuffer();
    const b64 = base64FromArrayBuffer(ab);

    // Tight rubric to avoid 0/1 scoring only
    const sys = `You are a strict but kind pronunciation judge.
Return STRICT JSON: { "pass": boolean, "score": number, "tone": object|null, "hint": string|null }.
Scoring (continuous):
- 0.95–1.00: clear correct pronunciation of the intended target.
- 0.85–0.94: minor deviation but clearly correct target.
- 0.70–0.84: close but uncertain; likely minor mispronunciation.
- 0.40–0.69: somewhat related but likely wrong.
- 0.00–0.39: unrelated.
Chinese tone policy:
- If language starts with zh and target is a single Hanzi: base pinyin match = pass=true.
- Wrong tone => tone.match=false and deduct ~0.10 from score.
- tone object: {"expected":"1|2|3|4|5|null","heard":"1|2|3|4|5|null","match":boolean}.
Hints: <= 80 chars, actionable.`;

    const user = {
      language,
      expected
    };

    const payload = {
      // Prefer a small audio-capable model to keep cost down.
      model: "gpt-4o-audio-preview", // or the current audio-capable snapshot
      response_format: { type: "json_object" },
      temperature: 0.2,
      messages: [
        { role: "system", content: sys },
        {
          role: "user",
          content: [
            { type: "input_text", text: JSON.stringify(user) },
            { type: "input_audio", audio: { data: b64, format: mimeToFormat(file.type) } }
          ]
        }
      ]
    };

    const r = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${process.env.OPENAI_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    const body = await safeBody(r);
    if (!r.ok) return json({ error: errString(body, 'OpenAI audio judge error') }, r.status);
    let verdict = {};
    try { verdict = JSON.parse(body?.choices?.[0]?.message?.content || "{}"); } catch {}

    // final coercion
    const out = {
      pass: !!verdict.pass,
      score: clamp01(Number(verdict.score)),
      tone: normTone(verdict.tone),
      hint: (typeof verdict.hint === 'string' ? verdict.hint.slice(0, 120) : null)
    };
    return json(out, 200);
  } catch (e) {
    return json({ error: String(e?.message || e) }, 500);
  }
}

/* helpers */
function json(obj, status=200){ return new Response(JSON.stringify(obj),{status,headers:{'Content-Type':'application/json'}}); }
function clamp01(x){ if(!Number.isFinite(x)) return 0; return Math.max(0, Math.min(1, x)); }
function normTone(t){ if(!t||typeof t!=='object') return null; const s=v=>v==null?null:String(v); const exp=s(t.expected), heard=s(t.heard); const match=typeof t.match==='boolean'?t.match:(exp&&heard?exp===heard:null); return {expected:exp, heard, match}; }
async function safeBody(r){ const ct=(r.headers.get('content-type')||'').toLowerCase(); return ct.includes('json')?r.json():{ message: await r.text() }; }
function errString(b,f='Unknown error'){ if(!b) return f; if(typeof b==='string') return b; const m=b?.error?.message||b?.message; return typeof m==='string'?m: f; }
function base64FromArrayBuffer(ab){ let s=''; const bytes=new Uint8Array(ab); for (let i=0;i<bytes.length;i++) s+=String.fromCharCode(bytes[i]); return btoa(s); }
function mimeToFormat(m){ // map MIME to expected "format" token for input_audio
  const t=(m||'').toLowerCase();
  if (t.includes('wav')) return 'wav';
  if (t.includes('ogg')) return 'ogg';
  if (t.includes('webm')) return 'webm';
  if (t.includes('mp3')) return 'mp3';
  return 'wav';
}
