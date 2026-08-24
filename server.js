"use strict";

const path = require("path");
const express = require("express");
require("dotenv").config();

// ---------------------------------------------------------------------------
// Configuration — every tuning knob lives here.
// ---------------------------------------------------------------------------
const PORT = process.env.PORT || 3000;
// NVIDIA NIM (OpenAI-compatible endpoint). Set NVIDIA_API_KEY in .env.
// Get a free key + free starter credits at https://build.nvidia.com
const HF_TOKEN = (process.env.NVIDIA_API_KEY || process.env.NVIDIA_KEY || "").trim();
const HF_BASE_URL = "https://integrate.api.nvidia.com/v1";
// MODEL is auto-detected at startup from NVIDIA's own /v1/models list (see
// detectModel() below) — you do NOT need to look up the model id yourself.
// This is only the fallback used if auto-detection fails for some reason.
let MODEL = "meta/llama-3.1-8b-instruct";

// Models we'd prefer if they're available (checked in this order against
// whatever the API actually returns). If none of these match, we just use
// the first model the API lists.
const PREFERRED_MODELS = [
  "meta/llama-3.1-8b-instruct",
  "meta/llama-3.1-70b-instruct",
  "qwen/qwen2.5-7b-instruct",
  "mistralai/mistral-7b-instruct-v0.3",
  "microsoft/phi-3-mini-4k-instruct",
];

// Per-pass timeout. There are now 12 single-focus passes instead of 4 broad
// ones, so each individual call is doing less work and should finish faster.
// 45s/pass * 12 passes = 9 min worst-case ceiling for the whole chain.
const REQUEST_TIMEOUT_MS = 45000;

// Sampling constants. Temperature stays below 0.5 per the project constraint.
// NOTE: a low, fixed temperature works AGAINST "raise unpredictability"
// (Rules 1-3). We can't remove the <0.5 constraint, so we lean harder on
// frequency/presence penalties for the lexical/structural rules instead —
// see RULES[].includePenalties below. If output still feels too uniform,
// this is the first knob to revisit (the constraint, not the prompts).
const SAMPLING = {
  temperature: 0.45,
  top_p: 0.92,
  frequency_penalty: 0.9, // discourage reusing tokens -> less repetitive vocabulary
  presence_penalty: 0.7,  // reward introducing fresh vocabulary -> higher perplexity
};

const MAX_INPUT_CHARS = 6000;

// ---------------------------------------------------------------------------
// Auto-rehumanize loop. Manually re-pasting the humanized output back through
// the pipeline a second time was measurably improving the "human" score
// (39% -> higher in testing), so the server now does that automatically and
// unconditionally: run the full 12-rule pipeline, take its output, run the
// FULL 12-rule pipeline on it again, exactly like pasting the result back in
// by hand — repeated HUMANIZE_PASSES times. No scoring, no early-stop
// guessing; every pass always runs the complete chain on the previous pass's
// output. humanize(text) -> output1 -> humanize(output1) -> output2 -> ...
// ---------------------------------------------------------------------------
const HUMANIZE_PASSES = Number(process.env.HUMANIZE_PASSES || 2);

// Content-loss guard (see runHumanizerPipeline below). If a rule's output
// drops a large chunk of the input (the model summarizing/skipping instead
// of rewriting everything), we retry that single rule once with a sharper
// instruction before giving up and keeping the pre-rule text untouched.
const MIN_LENGTH_RATIO = 0.65; // output must keep at least 65% of input word count

// Rules that must hold on every pass. Used for passes 1-8 and 10-12.
// Rule 9 (minor natural inconsistencies) uses a relaxed variant below,
// because "no fragments / no comma splices" directly contradicts the goal
// of that specific rule — see BASE_RULES_RELAXED.
const BASE_RULES_STRICT = [
  "Preserve the meaning, facts, technical terminology, and language of the original.",
  "Rewrite the ENTIRE input, start to finish. Do NOT skip, drop, condense, or summarize any sentence or paragraph, even long ones. The output must cover every part of the input, not just the easy parts.",
  "Keep every citation marker (for example [1], [2], [3,4]), acronym, and defined term exactly as written.",
  "Do NOT add, invent, or imitate citation markers or bracketed reference numbers unless they already appear in the input.",
  "Maintain correct grammar and punctuation. Do NOT insert sentence fragments, comma splices, or deliberate errors.",
  "NEVER use em dashes or en dashes (the '—' and '–' characters). Use a comma, colon, semicolon, or parentheses instead.",
  "Output ONLY the rewritten text — no pass labels, commentary, quotation wrappers, or explanations.",
].join("\n");

// Same as above, EXCEPT it deliberately permits the kind of small looseness
// real human writing has (an occasional fragment or comma splice used for
// effect). This is the one and only pass allowed to bend that rule.
const BASE_RULES_RELAXED = [
  "Preserve the meaning, facts, technical terminology, and language of the original.",
  "Rewrite the ENTIRE input, start to finish. Do NOT skip, drop, condense, or summarize any sentence or paragraph, even long ones. The output must cover every part of the input, not just the easy parts.",
  "Keep every citation marker (for example [1], [2], [3,4]), acronym, and defined term exactly as written.",
  "Do NOT add, invent, or imitate citation markers or bracketed reference numbers unless they already appear in the input.",
  "Grammar should stay clear and readable, but a genuine sentence fragment or comma splice is allowed here and there if it reads like something a careful human would actually write — never so much that it becomes confusing or sloppy.",
  "NEVER use em dashes or en dashes (the '—' and '–' characters). Use a comma, colon, semicolon, or parentheses instead.",
  "Output ONLY the rewritten text — no pass labels, commentary, quotation wrappers, or explanations.",
].join("\n");

// ---------------------------------------------------------------------------
// 12-rule sequential pipeline. Rule 1 -> Rule 2 -> ... -> Rule 12.
// Each rule is single-focus and receives ONLY the previous rule's output as
// input, so it never sees the earlier prompts — a true pipeline, not a
// bundle of instructions in one call.
// ---------------------------------------------------------------------------
const RULES = [
  {
    id: 1,
    name: "Break Predictable Word Choice",
    sourcePoint: "AI tends to generate the statistically most likely next word, making text highly predictable.",
    includePenalties: true,
    prompt: [
      "RULE 1 OF 12 — BREAK PREDICTABLE WORD CHOICE.",
      "AI text tends to pick the statistically most likely next word, which makes it read as predictable.",
      "",
      BASE_RULES_STRICT,
      "",
      "Your only job in this pass:",
      "• Wherever a word or phrase feels like the 'obvious' choice, replace it with a less predictable but equally natural alternative.",
      "• Do not touch sentence length, paragraph structure, tone, or formatting — that happens in later rules.",
      "• Keep the same number of sentences and their order.",
    ].join("\n"),
  },
  {
    id: 2,
    name: "Break Repetitive Grammatical Patterns",
    sourcePoint: "Similar sentence lengths and repetitive grammatical patterns indicate machine generation.",
    includePenalties: true,
    prompt: [
      "RULE 2 OF 12 — BREAK REPETITIVE GRAMMATICAL PATTERNS.",
      "Repetitive grammatical patterns (the same sentence shape used over and over, e.g. 'X does…, Y enables…, Z provides…') read as machine-generated.",
      "",
      BASE_RULES_STRICT,
      "",
      "Your only job in this pass:",
      "• Find sentences that share an identical grammatical template or opening structure and rebuild them with different structures.",
      "• Vary how sentences open: subject-first, prepositional phrase, dependent clause, participial phrase, concrete detail.",
      "• Do not change sentence length distribution — that happens in Rule 3. Focus only on structural pattern, not length.",
    ].join("\n"),
  },
  {
    id: 3,
    name: "Vary Sentence Length (Burstiness)",
    sourcePoint: "Human writing naturally alternates between short, medium, and long sentences. AI often produces consistent-length sentences.",
    includePenalties: true,
    prompt: [
      "RULE 3 OF 12 — VARY SENTENCE LENGTH (BURSTINESS).",
      "Human writing alternates between short, medium, and long sentences. AI text tends to sit at one consistent length.",
      "",
      BASE_RULES_STRICT,
      "",
      "Your only job in this pass:",
      "• Mix punchy short sentences (3-6 words), standard medium sentences (13-22 words), and complex longer sentences (23+ words).",
      "• Never leave three or more consecutive sentences in the same length band.",
      "• Short sentences must stay grammatically complete, not fragments (that specific allowance comes later, in Rule 9 only).",
      "• Do not reorganize paragraphs — that happens in Rule 10.",
    ].join("\n"),
  },
  {
    id: 4,
    name: "Eliminate Repetitive Transition Words",
    sourcePoint: "AI frequently reuses the same transition words (\"Furthermore,\" \"Moreover,\" \"Additionally,\" \"In conclusion,\" etc.)",
    includePenalties: true,
    prompt: [
      "RULE 4 OF 12 — ELIMINATE REPETITIVE TRANSITION WORDS.",
      "AI text leans on the same handful of transition words repeatedly.",
      "",
      BASE_RULES_STRICT,
      "",
      "Your only job in this pass:",
      "• Remove or replace: furthermore, moreover, additionally, consequently, subsequently, notably, importantly, significantly, indeed, thus, hence, in addition, as a result, on the other hand, that said, it is worth noting, in this regard, to that end, ultimately, overall, in summary, in conclusion.",
      "• Where a transition is genuinely needed, connect the sentences directly or use a simple connector instead (but, and, so, yet, still, also, then, because, while, when).",
      "• Many sentences don't need a transition word at all — cut it rather than swap it for a synonym.",
    ].join("\n"),
  },
  {
    id: 5,
    name: "Eliminate Repetitive Vocabulary & Adjectives",
    sourcePoint: "AI frequently reuses the same adjectives and stock vocabulary.",
    includePenalties: true,
    prompt: [
      "RULE 5 OF 12 — ELIMINATE REPETITIVE VOCABULARY AND ADJECTIVES.",
      "AI text reuses the same adjectives and stock words across a document.",
      "",
      BASE_RULES_STRICT,
      "",
      "Your only job in this pass:",
      "• Ban stock AI vocabulary: delve, underscore, leverage, utilize, facilitate, robust, comprehensive, pivotal, crucial, vital, paramount, nuanced, multifaceted, intricate, groundbreaking, transformative, innovative, seamless, holistic, overarching, landscape, tapestry, realm, paradigm, synergy, plethora, myriad, a testament to, plays a crucial role, serves as, sheds light on, in today's world, at its core, it is important to note.",
      "• Replace heavy adjectives (significant, substantial, remarkable, profound, extensive, considerable, notable, compelling, dynamic, cutting-edge) with simpler, precise words, or drop them.",
      "• No adjective, verb, or connective should appear twice in the text if a natural synonym exists.",
      "• Prefer plain verbs (use, help, show, build) over ornate ones (utilize, facilitate, underscore).",
    ].join("\n"),
  },
  {
    id: 6,
    name: "Break Consistent Academic Tone",
    sourcePoint: "AI often maintains a consistent academic tone throughout the document.",
    includePenalties: false,
    prompt: [
      "RULE 6 OF 12 — BREAK CONSISTENT ACADEMIC TONE.",
      "AI text tends to hold one flat, formal, academic register the entire way through.",
      "",
      BASE_RULES_STRICT,
      "",
      "Your only job in this pass:",
      "• Shift from an overly formal, flat academic posture into an approachable, peer-to-peer voice, like a knowledgeable colleague explaining something directly to another colleague.",
      "• Occasional contractions (it's, don't, won't, can't) are fine when natural.",
      "• Direct address ('you', 'we') is fine where it clarifies.",
      "• Stay credible: no slang, hype, exclamations, or rhetorical filler.",
    ].join("\n"),
  },
  {
    id: 7,
    name: "Add Opinion, Uncertainty & Personal Interpretation",
    sourcePoint: "Human writing typically includes opinions, uncertainty, or personal interpretation.",
    includePenalties: false,
    prompt: [
      "RULE 7 OF 12 — ADD OPINION, UNCERTAINTY, AND PERSONAL INTERPRETATION.",
      "Human writing carries a point of view. It hedges where things are genuinely uncertain instead of stating everything as flat fact.",
      "",
      BASE_RULES_STRICT,
      "",
      "Your only job in this pass:",
      "• Where the source expresses judgment, estimation, or something that isn't a hard fact, add realistic hedging: arguably, it seems that, often, tends to, in most cases, usually, likely, may, appears to, is generally.",
      "• Do not invent new facts, opinions, or claims that aren't implied by the source — only surface the uncertainty or interpretation that's already latent in it.",
      "• Do not overdo it: hedge only where it's genuinely warranted, not on every sentence.",
    ].join("\n"),
  },
  {
    id: 8,
    name: "Replace Broad Textbook Explanations with Concrete Detail",
    sourcePoint: "AI often explains concepts in a broad, textbook-like way.",
    includePenalties: false,
    prompt: [
      "RULE 8 OF 12 — REPLACE BROAD TEXTBOOK EXPLANATIONS WITH CONCRETE DETAIL.",
      "AI text explains concepts the way a textbook glossary does: broad, generic, definitional.",
      "",
      BASE_RULES_STRICT,
      "",
      "Your only job in this pass:",
      "• Find generic definitional phrasing ('X is a process that…', 'Y refers to…') and ground it in something sharper: a mechanism, a specific scenario, a concrete detail already implied by the source.",
      "• Prefer specific, practical framing over abstract overview.",
      "• Do not invent new facts, examples, or statistics that aren't grounded in the source material.",
    ].join("\n"),
  },
  {
    id: 9,
    name: "Introduce Minor Natural Inconsistencies",
    sourcePoint: "Real human writing usually contains minor inconsistencies.",
    includePenalties: false,
    prompt: [
      "RULE 9 OF 12 — INTRODUCE MINOR NATURAL INCONSISTENCIES.",
      "Real human writing is not perfectly uniform. It has small, natural rough edges: an occasional aside, a slightly informal turn of phrase, a sentence that doesn't quite match the polish of the one before it.",
      "",
      BASE_RULES_RELAXED,
      "",
      "Your only job in this pass:",
      "• Introduce a small number of genuine, natural imperfections: a brief parenthetical aside, a slightly less polished phrasing choice, an occasional short fragment used deliberately for effect.",
      "• This should feel like natural human unevenness, not sloppiness — meaning must stay completely clear.",
      "• Do not touch paragraph structure or formatting — that happens in Rules 10-11.",
      "• Use this sparingly: a handful of spots in the whole text, not every sentence.",
    ].join("\n"),
  },
  {
    id: 10,
    name: "Vary Paragraph Size & Structure",
    sourcePoint: "AI often creates paragraphs of similar size and structure.",
    includePenalties: false,
    prompt: [
      "RULE 10 OF 12 — VARY PARAGRAPH SIZE AND STRUCTURE.",
      "AI text tends to produce paragraphs that are all roughly the same length and shape.",
      "",
      BASE_RULES_STRICT,
      "",
      "Your only job in this pass:",
      "• Break uniform paragraph sizes. Mix single-sentence impact lines, short two-to-three-sentence blocks, and longer, denser paragraphs.",
      "• Never leave two or more consecutive paragraphs at a similar word count.",
      "• Split dense blocks at natural thought shifts; merge tiny uniform paragraphs where it reads better; let one short paragraph stand alone for punch if it earns it.",
      "• Keep the same overall content and order — only reshape where paragraph breaks fall.",
    ].join("\n"),
  },
  {
    id: 11,
    name: "Reduce Mechanical Formatting Consistency",
    sourcePoint: "Extremely consistent formatting can contribute (but is not decisive).",
    includePenalties: false,
    prompt: [
      "RULE 11 OF 12 — REDUCE MECHANICAL FORMATTING CONSISTENCY.",
      "Overly consistent formatting (mirrored headers, repeated bold-lede bullets, symmetrical outlines) reads as templated.",
      "",
      BASE_RULES_STRICT,
      "",
      "Your only job in this pass:",
      "• Use bold text, headers, and bullet points sparingly, and only if the source already relies on them structurally.",
      "• Prefer flowing prose over structured lists.",
      "• Flatten any '**Term**: definition' × N bullet patterns or mirrored section headers into natural paragraphs.",
      "• Never produce a fully symmetrical outline.",
    ].join("\n"),
  },
  {
    id: 12,
    name: "Final QA & Consistency Pass",
    sourcePoint: "Holistic check that Rules 1-11 actually held across the whole document.",
    includePenalties: false,
    prompt: [
      "RULE 12 OF 12 — FINAL QA AND CONSISTENCY PASS.",
      "This is the last pass. Read the text as a whole and fix anything that still slipped through from these 11 checks:",
      "1. Predictable/obvious word choice.",
      "2. Repetitive grammatical sentence patterns.",
      "3. Sentence lengths that don't alternate short/medium/long.",
      "4. Leftover transition words like furthermore/moreover/additionally/in conclusion.",
      "5. Repeated adjectives or stock AI vocabulary.",
      "6. Flat, overly formal academic tone.",
      "7. Missing opinion, hedging, or personal interpretation where warranted.",
      "8. Broad, textbook-style explanation instead of concrete detail.",
      "9. Suspiciously perfect, zero-inconsistency prose.",
      "10. Paragraphs that are all a similar size.",
      "11. Overly mechanical, symmetrical formatting.",
      "",
      BASE_RULES_STRICT,
      "",
      "Only make small corrective touch-ups where a check above still clearly fails. Do not do a full rewrite, and do not undo the natural imperfections added in Rule 9.",
    ].join("\n"),
  },
];

// ---------------------------------------------------------------------------
// Post-processing — hard guarantees that don't rely on the model complying.
// ---------------------------------------------------------------------------

// Dash removal. The model is told to avoid em/en dashes on every single pass,
// but we also strip any that slip through as a hard guarantee, after EVERY
// pass (not just at the end) so a stray dash from an early rule can't get
// treated as legitimate punctuation by a later rule in the chain.
function stripDashes(text) {
  return text
    .replace(/\s*[—―–]\s*/g, ", ") // em / horizontal bar / en dash -> comma
    .replace(/ +- +/g, ", ")       // a spaced hyphen used as a dash -> comma
    .replace(/,\s*,/g, ", ")       // collapse doubled commas
    .replace(/\s+([.,;:!?])/g, "$1")  // no space before punctuation
    .replace(/,\s*([.;:!?])/g, "$1"); // drop a comma stranded before other punctuation
}

// If the ORIGINAL text contained no bracketed citations, remove any the
// model invented anywhere in the 12-pass chain.
const CITATION_RE = /\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]/g;

function stripInventedCitations(text, original) {
  // Fresh, non-global regex for the test so lastIndex isn't carried between
  // requests (a stateful /g regex would misfire).
  if (/\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]/.test(original)) return text;
  return text
    .replace(CITATION_RE, "")
    .replace(/\s+([.,;:!?])/g, "$1")
    .replace(/[ \t]{2,}/g, " ")
    .trim();
}

// Some open models occasionally leak special/control tokens such as </s> or
// <|...|> into the message content. Strip them out after every pass.
function stripModelTokens(text) {
  return text
    .replace(/<\/?s>/gi, "")
    .replace(/<\|[^|]*\|>/g, "")
    .replace(/\[\/?INST\]/gi, "")
    .replace(/[ \t]{2,}/g, " ")
    .trim();
}

// Hard safety net for Rule 4. Even if the model leaves one in, strip these
// specific transition words/phrases as a guarantee rather than a hope.
const BANNED_TRANSITIONS_RE = new RegExp(
  "\\b(furthermore|moreover|additionally|consequently|subsequently|" +
    "in\\s+conclusion|in\\s+summary|as\\s+a\\s+result|on\\s+the\\s+other\\s+hand)" +
    "[,]?\\s*",
  "gi"
);

function stripBannedTransitions(text) {
  return text
    .replace(BANNED_TRANSITIONS_RE, (match, word, offset, full) => {
      // Only strip when it's being used as a sentence-opening transition
      // (i.e. right after start-of-text or a sentence-ending punctuation),
      // so we don't mangle the word if it legitimately appears mid-sentence.
      const before = full.slice(0, offset);
      const isSentenceStart = /(^\s*|[.!?]\s+)$/.test(before);
      return isSentenceStart ? "" : match;
    })
    .replace(/\n{2,}\s*\n/g, "\n\n")
    .trim();
}

// ---------------------------------------------------------------------------
// Lightweight, non-blocking quality metrics. This is the "verification" step
// the old version was missing entirely — it doesn't change the response, it
// just gives visibility into whether the chain actually did its job, logged
// to the console for the developer.
// ---------------------------------------------------------------------------
function analyzeText(text) {
  const sentences = text
    .split(/(?<=[.!?])\s+(?=[A-Z0-9"'(])/)
    .map((s) => s.trim())
    .filter(Boolean);
  const lengths = sentences.map((s) => s.split(/\s+/).filter(Boolean).length);

  const mean = lengths.reduce((a, b) => a + b, 0) / (lengths.length || 1);
  const variance =
    lengths.reduce((a, b) => a + (b - mean) ** 2, 0) / (lengths.length || 1);
  const stdev = Math.sqrt(variance);

  const band = (n) => (n <= 6 ? "short" : n <= 22 ? "medium" : "long");
  let longestSameBandRun = 1;
  let currentRun = 1;
  for (let i = 1; i < lengths.length; i++) {
    if (band(lengths[i]) === band(lengths[i - 1])) {
      currentRun++;
      longestSameBandRun = Math.max(longestSameBandRun, currentRun);
    } else {
      currentRun = 1;
    }
  }

  const paragraphs = text.split(/\n{2,}/).map((p) => p.trim()).filter(Boolean);
  const paraLengths = paragraphs.map((p) => p.split(/\s+/).filter(Boolean).length);

  const bannedHits = (text.match(BANNED_TRANSITIONS_RE) || []).length;

  return {
    sentenceCount: sentences.length,
    avgSentenceLength: Number(mean.toFixed(1)),
    sentenceLengthStdev: Number(stdev.toFixed(1)),
    longestSameLengthBandRun: longestSameBandRun,
    paragraphCount: paragraphs.length,
    paragraphLengths: paraLengths,
    bannedTransitionHits: bannedHits,
  };
}

function logQualityMetrics(metrics) {
  console.log("  ── Quality check (informational only) ──");
  console.log(`     sentences: ${metrics.sentenceCount}, avg length: ${metrics.avgSentenceLength} words, stdev: ${metrics.sentenceLengthStdev}`);
  if (metrics.longestSameLengthBandRun >= 3) {
    console.warn(`     ⚠ ${metrics.longestSameLengthBandRun} consecutive sentences share a length band (Rule 3 target: < 3)`);
  }
  console.log(`     paragraphs: ${metrics.paragraphCount}, lengths: [${metrics.paragraphLengths.join(", ")}]`);
  if (metrics.bannedTransitionHits > 0) {
    console.warn(`     ⚠ ${metrics.bannedTransitionHits} banned transition word(s) still present (auto-stripped as a safety net)`);
  }
}

// ---------------------------------------------------------------------------
// Runs the full 12-rule pipeline HUMANIZE_PASSES times, unconditionally.
// Pass 2 receives pass 1's output as its input, pass 3 receives pass 2's
// output, and so on — the exact same thing as pasting the humanized text
// back into the box and hitting "Humanize" again. No scoring, no early
// stopping. Returns the final text plus per-pass metrics purely for logging.
// ---------------------------------------------------------------------------
async function runAutoHumanizer(originalText, signal) {
  let current = originalText;
  const passReports = [];

  for (let pass = 1; pass <= HUMANIZE_PASSES; pass++) {
    console.log(`\n=== Humanize pass ${pass}/${HUMANIZE_PASSES} ===`);
    current = await runHumanizerPipeline(current, signal, `pass ${pass}`);

    // Cleanup after every full pass, not just the last one, so a stray
    // banned transition or invented citation from pass N never leaks into
    // pass N+1's input.
    current = stripBannedTransitions(stripInventedCitations(current, originalText));

    const metrics = analyzeText(current);
    logQualityMetrics(metrics);
    passReports.push({ pass, metrics });
  }

  return { text: current, passReports };
}

function maxTokensFor(text) {
  // Rewrites (hedging, added detail, longer sentences) can come out LONGER
  // than the input, so give real headroom above the input's own size —
  // being stingy here is what causes a pass to cut off mid-document and
  // silently drop the rest of the text. Bump the 6144 cap higher if your
  // NIM model supports a larger max_tokens and you still see truncation.
  return Math.min(6144, Math.max(512, Math.ceil(text.length / 2.5) + 400));
}

// Ask the API itself which models this key can actually use, and pick one —
// no manual lookup on a dashboard required. Called once at server startup.
async function detectModel() {
  if (!HF_TOKEN) return; // no key yet; keep the fallback, the /api/humanize route will error clearly

  try {
    const resp = await fetch(`${HF_BASE_URL}/models`, {
      headers: { Authorization: `Bearer ${HF_TOKEN}` },
    });

    if (!resp.ok) {
      console.warn(`  ⚠  Could not list models (status ${resp.status}). Using fallback: ${MODEL}`);
      return;
    }

    const data = await resp.json();
    const ids = (data?.data || []).map((m) => m.id).filter(Boolean);

    if (ids.length === 0) {
      console.warn(`  ⚠  Model list was empty. Using fallback: ${MODEL}`);
      return;
    }

    const preferredMatch = PREFERRED_MODELS.find((p) => ids.includes(p));
    MODEL = preferredMatch || ids[0];

    console.log(`  ✓ Auto-detected model: ${MODEL} (${ids.length} model${ids.length === 1 ? "" : "s"} available to this key)`);
  } catch (err) {
    console.warn(`  ⚠  Model auto-detection failed (${err.message}). Using fallback: ${MODEL}`);
  }
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

  // Defensive cleanup after EVERY pass, not just at the end of the chain,
  // so artifacts from one rule can't propagate into and confuse the next.
  return stripDashes(stripModelTokens(raw));
}

function wordCount(text) {
  return text.split(/\s+/).filter(Boolean).length;
}

// ---------------------------------------------------------------------------
// The pipeline itself: Rule 1 -> Rule 2 -> Rule 3 -> ... -> Rule 12.
// Each rule receives ONLY the previous rule's output as input.
//
// Content-loss guard: if a rule's output is suspiciously shorter than what
// went into it (the model summarizing or skipping a chunk instead of
// rewriting the whole thing — the "skips a huge part" problem), we retry
// that ONE rule once with an emphasized instruction. If it still comes back
// short, we keep the pre-rule text and move on, rather than silently
// letting the pipeline lose content on every later pass too.
// ---------------------------------------------------------------------------
async function runHumanizerPipeline(originalText, signal, label = "") {
  let current = originalText;

  for (const rule of RULES) {
    console.log(`  → ${label ? label + " · " : ""}Rule ${rule.id}/12: ${rule.name}`);
    const inputWords = wordCount(current);

    try {
      let output = await callModelPass({
        systemPrompt: rule.prompt,
        userText: current,
        includePenalties: rule.includePenalties,
        signal,
      });

      if (inputWords >= 20 && wordCount(output) < inputWords * MIN_LENGTH_RATIO) {
        console.warn(
          `     ⚠ Rule ${rule.id} output dropped from ${inputWords} to ${wordCount(output)} words, retrying with a stricter prompt`
        );
        const retryPrompt =
          rule.prompt +
          "\n\nIMPORTANT: your previous attempt at this left out part of the text. " +
          "Rewrite the FULL input this time, every sentence and paragraph, start to finish. Do not summarize or condense.";
        const retryOutput = await callModelPass({
          systemPrompt: retryPrompt,
          userText: current,
          includePenalties: rule.includePenalties,
          signal,
        });

        if (wordCount(retryOutput) >= inputWords * MIN_LENGTH_RATIO) {
          output = retryOutput;
        } else {
          console.warn(
            `     ⚠ Rule ${rule.id} still dropped content after retry (${wordCount(retryOutput)} words) — keeping pre-rule text for this rule.`
          );
          output = current; // skip this rule rather than lose content
        }
      }

      current = output;
    } catch (err) {
      const detail = err.message || String(err);
      throw new Error(`Rule ${rule.id} (${rule.name}) failed: ${detail}`);
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
      error: "Server is missing NVIDIA_API_KEY. Add your NVIDIA API key (from build.nvidia.com) to the .env file.",
    });
  }

  const controller = new AbortController();
  const timer = setTimeout(
    () => controller.abort(),
    REQUEST_TIMEOUT_MS * RULES.length * HUMANIZE_PASSES
  );

  try {
    const { text: raw, passReports } = await runAutoHumanizer(text, controller.signal);

    // Final dash cleanup (citation stripping + banned-transition stripping
    // already happen after every pass inside runAutoHumanizer).
    const result = stripDashes(raw);

    return res.json({
      result,
      passes: passReports.length, // how many full 12-rule passes actually ran
      passReports, // per-pass metrics, remove this field if you don't want it exposed to the client
    });
  } catch (err) {
    if (err.name === "AbortError") {
      console.error("Humanizer pipeline timed out.");
      return res.status(504).json({
        error: "The 12-rule pipeline took too long. Try again or shorten the text.",
      });
    }
    console.error("Humanizer pipeline failed:", err.message || err);
    const ruleFailed = /Rule \d/.test(err.message || "");
    return res.status(502).json({
      error: ruleFailed
        ? `The language model failed during the pipeline (${err.message}). Please try again.`
        : "Could not reach the language model. Check your connection and try again.",
    });
  } finally {
    clearTimeout(timer);
  }
});

async function start() {
  await detectModel();
  app.listen(PORT, () => {
    console.log(`\n  AI Humanizer running at  http://localhost:${PORT}`);
    if (!HF_TOKEN) {
      console.warn("  ⚠  NVIDIA_API_KEY is not set — add your NVIDIA API key to .env before humanizing.\n");
    } else {
      console.log(`  Model: ${MODEL} (NVIDIA NIM)  ·  12-rule pipeline  ·  temp ${SAMPLING.temperature}\n`);
    }
  });
}

start();