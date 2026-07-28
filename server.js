"use strict";

const path = require("path");
const express = require("express");
require("dotenv").config();

// ---------------------------------------------------------------------------
// Configuration — every tuning knob lives here.
// ---------------------------------------------------------------------------
const PORT = process.env.PORT || 3000;
// HuggingFace Inference (OpenAI-compatible router). Set HF_TOKEN in .env.
const HF_TOKEN = (process.env.HF_TOKEN || process.env.HUGGINGFACE_API_KEY || "").trim();
const HF_BASE_URL = "https://router.huggingface.co/v1";
const MODEL = "Qwen/Qwen2.5-7B-Instruct"; // any text-generation model on the HF router
const REQUEST_TIMEOUT_MS = 120000; // per pass — full chain runs up to 4 sequential calls

// Sampling constants. Temperature stays below 0.5 per the project constraint;
// the penalties + system prompt are what actually push the model away from
// its most-probable / repeated word choices.
const SAMPLING = {
  temperature: 0.45,      // stays < 0.5 per the project constraint
  top_p: 0.92,
  frequency_penalty: 0.9, // discourage reusing tokens -> less repetitive vocabulary
  presence_penalty: 0.7,  // reward introducing fresh vocabulary -> higher perplexity
};

const MAX_INPUT_CHARS = 6000;

// Shared constraints applied across every pass in the chain.
const BASE_RULES = [
  "Preserve the meaning, facts, technical terminology, and language of the original.",
  "Keep every citation marker (for example [1], [2], [3,4]), acronym, and defined term exactly as written.",
  "Do NOT add, invent, or imitate citation markers or bracketed reference numbers unless they already appear in the input.",
  "Maintain correct grammar and punctuation. Do NOT insert sentence fragments, comma splices, or deliberate errors.",
  "NEVER use em dashes or en dashes (the '—' and '–' characters). Use a comma, colon, semicolon, or parentheses instead.",
  "Output ONLY the rewritten text — no pass labels, commentary, quotation wrappers, or explanations.",
].join("\n");

// 4-pass sequential humanizer chain. Each pass rewrites the previous pass output.
const HUMANIZER_CHAIN = [
  {
    id: 1,
    name: "Vocabulary & Perplexity",
    includePenalties: true,
    prompt: [
      "PASS 1 OF 4 — VOCABULARY AND PERPLEXITY.",
      "Rewrite the input to raise lexical unpredictability and eliminate repetitive, high-probability AI phrasing.",
      "",
      BASE_RULES,
      "",
      "Pass 1 focus:",
      "• Unpredictable word choice: replace predictable, high-probability word patterns with varied, natural human choices.",
      "• Vocabulary variety: eliminate repetitive adjectives, verbs, and stock phrasing — no word or connective should appear twice if a natural synonym exists.",
      "• Ban stock AI vocabulary: delve, underscore, leverage, utilize, facilitate, robust, comprehensive, pivotal, crucial, vital, paramount, nuanced, multifaceted, intricate, groundbreaking, transformative, innovative, seamless, holistic, overarching, landscape, tapestry, realm, paradigm, synergy, plethora, myriad, a testament to, plays a crucial role, serves as, sheds light on, in today's world, at its core, it is important to note.",
      "• Replace heavy adjectives (significant, substantial, remarkable, profound, extensive, considerable, notable, compelling, dynamic, cutting-edge) with simple, precise words — or drop them.",
      "• Ban overused AI transitions: furthermore, moreover, additionally, consequently, subsequently, notably, importantly, significantly, indeed, thus, hence, in addition, as a result, on the other hand, that said, it is worth noting, in this regard, to that end, ultimately, overall, in summary, in conclusion.",
      "• Prefer plain verbs (use, help, show, build) over ornate ones (utilize, facilitate, underscore).",
      "• Keep paragraph count and order stable in this pass — focus on words, not layout.",
    ].join("\n"),
  },
  {
    id: 2,
    name: "Sentence Structure & Burstiness",
    includePenalties: true,
    prompt: [
      "PASS 2 OF 4 — SENTENCE STRUCTURE AND BURSTINESS.",
      "Rewrite the input to break mechanical sentence patterns and create high rhythmic contrast.",
      "",
      BASE_RULES,
      "",
      "Pass 2 focus:",
      "• Structural variety: break repetitive grammatical patterns (Subject-Verb-Object loops, identical openers, parallel 'X does…, Y enables…, Z provides…' templates).",
      "• High burstiness: mix punchy short sentences (3–6 words) with standard medium sentences (13–22 words) and complex longer sentences (23+ words).",
      "• Never let three or more consecutive sentences stay in the same length band.",
      "• Vary how sentences open: subject-first, prepositional phrase, dependent clause, participial phrase, concrete detail.",
      "• Short sentences must remain grammatically complete — not fragments.",
      "• Connect sentences directly when possible; use only simple connectors (but, and, so, yet, still, also, then, because, while, when) when genuinely needed.",
      "• Do not reshape paragraphs yet — focus on sentence-level rhythm.",
    ].join("\n"),
  },
  {
    id: 3,
    name: "Tone, Voice & Personal Depth",
    includePenalties: false,
    prompt: [
      "PASS 3 OF 4 — TONE, VOICE, AND PERSONAL DEPTH.",
      "Rewrite the input so it reads like a knowledgeable peer explaining a concept directly to a colleague.",
      "",
      BASE_RULES,
      "",
      "Pass 3 focus:",
      "• Dynamic tone: shift away from an overly formal, flat academic posture into an approachable, peer-to-peer voice. Use relaxed phrasing where it fits. Occasional contractions (it's, don't, won't, can't) are fine when natural. Direct address ('you', 'we') is acceptable where it clarifies.",
      "• Personal perspective: inject realistic human uncertainty and hedging where judgment or uncertainty exists (arguably, it seems that, often, tends to, in most cases, usually, likely, may, appears to, is generally). Express a readable point of view — not a completely neutral encyclopedia observer.",
      "• Deep explanations: eliminate broad, generic textbook-style summaries ('X is a process that…', 'Y refers to…'). Ground concepts in sharp, practical, real-world context drawn from the source. Prefer specific scenarios and mechanisms over abstract overview. Do not invent new facts or statistics.",
      "• Stay credible: no slang, hype, exclamations, or rhetorical filler.",
      "• Preserve the source's ideas and logical order.",
    ].join("\n"),
  },
  {
    id: 4,
    name: "Layout & Natural Imperfection",
    includePenalties: false,
    prompt: [
      "PASS 4 OF 4 — LAYOUT AND NATURAL IMPERFECTION.",
      "This is the final pass. Polish the input into naturally imperfect, human-scannable prose.",
      "",
      BASE_RULES,
      "",
      "Pass 4 focus:",
      "• Organic asymmetry: allow subtle stylistic variations rather than hyper-polished, robotic perfection. Vary how ideas are introduced. Allow brief parenthetical side-thoughts and minor phrasing shifts between sections.",
      "• Paragraph diversity: break uniform paragraph sizes. Mix single-sentence impact lines with short two-to-three-sentence blocks and longer dense paragraphs. Never place two or more consecutive paragraphs of similar word count. Split dense blocks at thought shifts; merge tiny uniform ones and leave one standing alone for punch.",
      "• Understated formatting: avoid mechanical formatting templates. Use bold text, headers, and bullet points sparingly and only if the source already uses them. Prefer flowing prose. Never produce symmetrical outlines, mirrored section headers, or repeated bold-lede bullet patterns ('**Term**: definition' × N). Flatten over-structured source material into natural paragraphs.",
      "• Do not sanitize into unnatural perfection — imperfect flow is a feature as long as grammar and meaning stay clear.",
      "• Output ONLY the final rewritten text.",
    ].join("\n"),
  },
];

// ---------------------------------------------------------------------------
// Post-processing 1 — Dash removal.
// The model is told to avoid em/en dashes, but we also strip any that slip
// through as a hard guarantee. A dash between words becomes a comma; the
// cleanup passes fix any doubled or stranded punctuation left behind.
// ---------------------------------------------------------------------------
function stripDashes(text) {
  return text
    .replace(/\s*[—―–]\s*/g, ", ") // em / horizontal bar / en dash -> comma
    .replace(/ +- +/g, ", ")                        // a spaced hyphen used as a dash -> comma
    .replace(/,\s*,/g, ", ")                        // collapse doubled commas
    .replace(/\s+([.,;:!?])/g, "$1")                // no space before punctuation
    .replace(/,\s*([.;:!?])/g, "$1");               // drop a comma stranded before other punctuation
}

// If the ORIGINAL text contained no bracketed citations, remove any the model
// invented (it can be tempted to by the citation-rich style reference).
const CITATION_RE = /\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]/g;

function stripInventedCitations(text, original) {
  // Note: use a fresh, non-global regex for the test so lastIndex is not
  // carried between requests (a stateful /g regex would misfire).
  if (/\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]/.test(original)) return text;
  return text
    .replace(CITATION_RE, "")
    .replace(/\s+([.,;:!?])/g, "$1") // tidy space left before punctuation
    .replace(/[ \t]{2,}/g, " ")
    .trim();
}

// Some open models (e.g. Mixtral) occasionally leak special/control tokens
// such as </s> or <|...|> into the message content. Strip them out.
function stripModelTokens(text) {
  return text
    .replace(/<\/?s>/gi, "")            // <s> and </s>
    .replace(/<\|[^|]*\|>/g, "")        // <|endoftext|>, <|im_end|>, etc.
    .replace(/\[\/?INST\]/gi, "")       // [INST] / [/INST]
    .replace(/[ \t]{2,}/g, " ")         // collapse the double spaces left behind
    .trim();
}

// ---------------------------------------------------------------------------
// Post-processing 2 — Inconsistent, invisible Unicode spacing.
// Inserts one zero-width Unicode character after EVERY word. Which character
// is used is chosen at random per word, so the spacing is "inconsistent" at
// the byte level while staying invisible to a human reader.
//
// IMPORTANT: only characters that Microsoft Word (and other renderers) treat
// as non-rendering are used. These are all "Default_Ignorable_Code_Point"
// format characters that Word hides rather than drawing a placeholder box.
// The obscure U+2063 INVISIBLE SEPARATOR was removed because Word's default
// fonts lack a glyph for it and render it as a visible box.
// ---------------------------------------------------------------------------
const INVISIBLE_MARKS = [
  "​", // ZERO WIDTH SPACE
  "‌", // ZERO WIDTH NON-JOINER
  "‍", // ZERO WIDTH JOINER
  "﻿", // ZERO WIDTH NO-BREAK SPACE
];

function applyUnicodeSpacing(text) {
  // Match each run of non-whitespace (a "word", including trailing punctuation)
  // and append a randomly chosen invisible mark after it — no word is missed.
  return text.replace(/\S+/gu, (word) => {
    const mark = INVISIBLE_MARKS[Math.floor(Math.random() * INVISIBLE_MARKS.length)];
    return word + mark;
  });
}

function maxTokensFor(text) {
  return Math.min(4096, Math.max(256, Math.ceil(text.length / 3) + 200));
}

async function callModelPass({ systemPrompt, userText, includePenalties, signal }) {
  const body = {
    model: MODEL,
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: userText },
    ],
    temperature: SAMPLING.temperature,
    top_p: SAMPLING.top_p,
    max_tokens: maxTokensFor(userText),
    stream: false,
  };
  if (includePenalties) {
    body.frequency_penalty = SAMPLING.frequency_penalty;
    body.presence_penalty = SAMPLING.presence_penalty;
  }

  let upstream = await fetch(`${HF_BASE_URL}/chat/completions`, {
    method: "POST",
    signal,
    headers: {
      Authorization: `Bearer ${HF_TOKEN}`,
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify(body),
  });

  if (includePenalties && (upstream.status === 400 || upstream.status === 422)) {
    console.warn("Model rejected penalty params; retrying without them.");
    delete body.frequency_penalty;
    delete body.presence_penalty;
    upstream = await fetch(`${HF_BASE_URL}/chat/completions`, {
      method: "POST",
      signal,
      headers: {
        Authorization: `Bearer ${HF_TOKEN}`,
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(body),
    });
  }

  if (!upstream.ok) {
    const detail = await upstream.text();
    throw new Error(`upstream ${upstream.status}: ${detail}`);
  }

  const data = await upstream.json();
  const raw = data?.choices?.[0]?.message?.content?.trim();
  if (!raw) {
    throw new Error("empty response");
  }
  return stripModelTokens(raw);
}

async function runHumanizerChain(originalText, signal) {
  let current = originalText;

  for (const pass of HUMANIZER_CHAIN) {
    console.log(`  → Pass ${pass.id}/4: ${pass.name}`);
    try {
      current = await callModelPass({
        systemPrompt: pass.prompt,
        userText: current,
        includePenalties: pass.includePenalties,
        signal,
      });
    } catch (err) {
      const detail = err.message || String(err);
      throw new Error(`Pass ${pass.id} (${pass.name}) failed: ${detail}`);
    }
  }

  return current;
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------
const app = express();
app.use(express.json({ limit: "1mb" }));
app.use(express.static(path.join(__dirname, "public")));

app.post("/api/humanize", async (req, res) => {
  const text = typeof req.body?.text === "string" ? req.body.text.trim() : "";

  if (!text) {
    return res.status(400).json({ error: "Please provide some text to rephrase." });
  }
  if (text.length > MAX_INPUT_CHARS) {
    return res.status(400).json({
      error: `Text is too long (${text.length} chars). Limit is ${MAX_INPUT_CHARS}.`,
    });
  }
  if (!HF_TOKEN) {
    return res.status(500).json({
      error: "Server is missing HF_TOKEN. Add your HuggingFace access token to the .env file.",
    });
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS * HUMANIZER_CHAIN.length);

  try {
    const raw = await runHumanizerChain(text, controller.signal);

    // Clean invented citations -> strip dashes -> invisible spacing (final pass only).
    const cleaned = stripDashes(stripInventedCitations(raw, text));
    const result = applyUnicodeSpacing(cleaned);
    return res.json({ result });
  } catch (err) {
    if (err.name === "AbortError") {
      console.error("Humanizer chain timed out.");
      return res.status(504).json({
        error: "The 4-pass chain took too long. Try again or shorten the text.",
      });
    }
    console.error("Humanizer chain failed:", err.message || err);
    const passFailed = /Pass \d/.test(err.message || "");
    return res.status(502).json({
      error: passFailed
        ? `The language model failed during the chain (${err.message}). Please try again.`
        : "Could not reach the language model. Check your connection and try again.",
    });
  } finally {
    clearTimeout(timer);
  }
});

app.listen(PORT, () => {
  console.log(`\n  AI Humanizer running at  http://localhost:${PORT}`);
  if (!HF_TOKEN) {
    console.warn("  ⚠  HF_TOKEN is not set — add your HuggingFace token to .env before humanizing.\n");
  } else {
    console.log(`  Model: ${MODEL} (HuggingFace)  ·  4-pass chain  ·  temp ${SAMPLING.temperature}\n`);
  }
});
