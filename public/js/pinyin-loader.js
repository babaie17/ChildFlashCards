// public/js/pinyin-loader.js
console.log('pinyin-loader loaded');

const shardCache = new Map();   // base syllable -> shard object
let hanziMap = null;            // { "你": [{sound:"ni", tone:3, pretty:"nǐ"}], ... }

async function ensureHanziMap() {
  if (hanziMap) return hanziMap;
  const r = await fetch('/hanzi_to_pinyin.json', { cache: 'force-cache' });
  hanziMap = r.ok ? await r.json() : {};
  return hanziMap;
}

// Normalize a pinyin string to "hao" or "hao3"
function toPinyinKey(s) {
  const toneMap = {
    'ā':'a1','á':'a2','ǎ':'a3','à':'a4',
    'ē':'e1','é':'e2','ě':'e3','è':'e4',
    'ī':'i1','í':'i2','ǐ':'i3','ì':'i4',
    'ō':'o1','ó':'o2','ǒ':'o3','ò':'o4',
    'ū':'u1','ú':'u2','ǔ':'u3','ù':'u4',
    'ǖ':'v1','ǘ':'v2','ǚ':'v3','ǜ':'v4','ü':'v'
  };
  let t = (s||'').trim().toLowerCase();
  t = t.replace(/[āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜü]/g, m => toneMap[m] || m);
  // keep "hao3" if present, else "hao"
  const withTone = /^[a-z]+[1-5]$/.test(t);
  return withTone ? t : t.replace(/[^a-z]/g,'');
}

// Fetch a shard and return candidates for key ("hao" or "hao3")
async function loadHanziForPinyin(key) {
  const base = key.replace(/[1-5]$/, '');
  if (!base) return [];
  if (!shardCache.has(base)) {
    const resp = await fetch(`/pinyin-index/${base}.json`, { cache: 'force-cache' });
    const obj = resp.ok ? await resp.json() : {};
    shardCache.set(base, obj);
  }
  const shard = shardCache.get(base) || {};
  return shard[key] || shard[base] || [];
}

/* ---------- public API (arrays) ---------- */
// From a recognized single Hanzi → union of all homophones across its readings (tone ignored)
async function homophonesFromHanzi(ch) {
  if (!ch) return [];
  await ensureHanziMap();
  const readings = Array.isArray(hanziMap[ch]) ? hanziMap[ch] : [];
  if (!readings.length) return [];
  const bases = [...new Set(readings.map(r => (r.sound||'').toLowerCase()).filter(Boolean))];
  const out = new Set();
  for (const b of bases) {
    const list = await loadHanziForPinyin(b);
    for (const c of list) out.add(c);
  }
  return [...out];
}

// From a recognized single pinyin syllable (e.g., "hǎo" or "hao3")
async function homophonesFromPinyin(syllable) {
  const key = toPinyinKey(syllable);
  if (!key) return [];
  return await loadHanziForPinyin(key);
}

/* ---------- optional helpers WITH tone ---------- */
// Same as homophonesFromHanzi, but also returns toneLabel, e.g., "3" or "2/4"
async function homophonesFromHanziWithTone(ch) {
  await ensureHanziMap();
  const homophones = await homophonesFromHanzi(ch);
  const readings = Array.isArray(hanziMap[ch]) ? hanziMap[ch] : [];
  const tones = [...new Set(readings.map(r => r.tone).filter(Boolean))];
  const toneLabel = tones.length ? tones.join('/') : null;
  return { homophones, toneLabel };
}

// Same as homophonesFromPinyin, but returns {homophones, toneLabel}
// toneLabel is parsed from the syllable if it ends with 1..5; otherwise null.
async function homophonesFromPinyinWithTone(syllable) {
  const key = toPinyinKey(syllable);
  if (!key) return { homophones: [], toneLabel: null };
  const homophones = await loadHanziForPinyin(key);
  const toneLabel = /[1-5]$/.test(key) ? key.slice(-1) : null;
  return { homophones, toneLabel };
}

async function getReadings(ch) {
  await ensureHanziMap();
  return Array.isArray(hanziMap[ch]) ? hanziMap[ch] : [];
}

window.pinyinLoader = {
  homophonesFromPinyin,
  homophonesFromHanzi,
  // optional helpers (keep UI decoupled):
  homophonesFromPinyinWithTone,
  homophonesFromHanziWithTone,
  getReadings
};
