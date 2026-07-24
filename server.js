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
const REQUEST_TIMEOUT_MS = 90000; // guard against a hung upstream request

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

// A sample of the target writing style. The rewriter should MATCH this
// register and rhythm (formal academic prose), not copy its content.
const STYLE_SAMPLE = [
  "With the rapid increase in the world's population, electricity demand also increases. It is estimated that total energy demand at the end of 2020 will increase by 75% as compared to 2000 [1]. This increase may force utilities to rethink electricity generation and distribution in order to avoid unprecedented energy challenges. The utilities thus struggle to fulfill and manage the energy demand with smart generation with reduced carbon emissions. For this purpose, the traditional electric grid is evolving to a new smart grid (SG) [2]. In SG, advanced information and communication technologies provide flexibility to interact customers with utility [3,4]. Advanced metering infrastructure (AMI) equips each customer with smart meter whose major function is to gather energy demand information at customer premises and upload to the utility server [5]. According to [6,7], SG allows integration of renewable and distributed energy generation to diminish the effects of CO2 on environment and to reduce the energy consumption.",
  "",
  "Demand side management (DSM) is one of the key programs of SG to efficiently manage the energy demand of end users via real time information exchange between utility and consumer through AMI. These programs aim at enhancing grid reliability by reducing average peak load demand. So, utilities and customers can manage the energy generation and consumption through the implementation of DSM programs by providing incentives or encouraging the customers to participate in energy management programs. End users can take monitory benefits by shifting peak load during off peak hours by adopting different scheduling techniques.",
].join("\n");

const SYSTEM_PROMPT = [
  "You are an expert academic editor. Rewrite the user's text as formal, scholarly academic prose.",
  "Preserve the meaning, facts, technical terminology, and the language of the original. Keep every citation marker (for example [1], [2], [3,4]), acronym, and defined term (for example SG, AMI, DSM, PAR, CO2) exactly as written.",
  "CRITICAL: Do NOT add, invent, or imitate any citation markers, reference numbers, or bracketed numbers such as [1] or [2]. Use them ONLY if they already appear in the user's text. The style reference below contains citations; those belong to its content, so never copy or fabricate them.",
  "",
  "STYLE REFERENCE. Match the register, rhythm, formality, and academic voice of the passage below. Do NOT reuse its subject matter or copy its sentences; only emulate how it is written:",
  "-----",
  STYLE_SAMPLE,
  "-----",
  "",
  "Rules:",
  "",
  "1. Maintain a formal, objective, third-person academic tone throughout. No contractions, no slang, no casual asides, no first-person opinions, no rhetorical questions, no exclamations.",
  "",
  "2. Use precise, discipline-appropriate vocabulary. Prefer the exact technical term over a vague or informal substitute.",
  "",
  "3. Vary sentence length and construction so the writing does not read as machine-uniform: alternate concise declarative sentences with longer, complex or compound sentences, and vary how sentences open. Every sentence must be grammatically complete and correct.",
  "",
  "4. Do not repeat the same word, connective, or phrase. Vary transitions (for example 'however', 'moreover', 'consequently', 'in addition', 'as a result') and do not lean on any single one.",
  "",
  "5. Avoid clichés and stock AI phrasing ('in today's world', 'it is important to note', 'delve', 'a testament to', 'plays a crucial role', 'in conclusion').",
  "",
  "6. Keep the writing specific and substantive. Do not pad with generic filler or empty generalisations.",
  "",
  "7. Maintain correct, formal grammar and punctuation. Do NOT insert sentence fragments, comma splices, or deliberate errors.",
  "",
  "8. NEVER use em dashes or en dashes (the '—' and '–' characters). Use a comma, colon, semicolon, or parentheses instead.",
  "",
  "9. Preserve the paragraph structure of the source unless a change clearly improves clarity.",
  "",
  "Do not add headings, commentary, quotation marks, or explanations. Output ONLY the rewritten text.",
].join("\n");

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

  // Give the model room to match the input length (~1.4 tokens/word + headroom).
  const maxTokens = Math.min(4096, Math.max(256, Math.ceil(text.length / 3) + 200));

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  // Build and send one request. Penalty params can be omitted because some
  // HuggingFace providers reject them.
  const callModel = (includePenalties) => {
    const body = {
      model: MODEL,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: text },
      ],
      temperature: SAMPLING.temperature,
      top_p: SAMPLING.top_p,
      max_tokens: maxTokens,
      stream: false,
    };
    if (includePenalties) {
      body.frequency_penalty = SAMPLING.frequency_penalty;
      body.presence_penalty = SAMPLING.presence_penalty;
    }
    return fetch(`${HF_BASE_URL}/chat/completions`, {
      method: "POST",
      signal: controller.signal,
      headers: {
        Authorization: `Bearer ${HF_TOKEN}`,
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(body),
    });
  };

  try {
    let upstream = await callModel(true);
    // If the provider rejected the request over unsupported params, retry clean.
    if (upstream.status === 400 || upstream.status === 422) {
      console.warn("Model rejected penalty params; retrying without them.");
      upstream = await callModel(false);
    }

    if (!upstream.ok) {
      const detail = await upstream.text();
      console.error(`HuggingFace API error ${upstream.status}:`, detail);
      return res.status(502).json({
        error: `The language model returned an error (${upstream.status}). Please try again.`,
      });
    }

    const data = await upstream.json();
    const raw = data?.choices?.[0]?.message?.content?.trim();

    if (!raw) {
      return res.status(502).json({ error: "The model returned an empty response." });
    }

    // Clean model tokens -> drop invented citations -> strip dashes -> invisible spacing.
    const cleaned = stripDashes(stripInventedCitations(stripModelTokens(raw), text));
    const result = applyUnicodeSpacing(cleaned);
    return res.json({ result });
  } catch (err) {
    if (err.name === "AbortError") {
      console.error("NVIDIA API request timed out.");
      return res.status(504).json({
        error: "The model took too long to respond. Try again or shorten the text.",
      });
    }
    console.error("Request to NVIDIA API failed:", err);
    return res.status(502).json({
      error: "Could not reach the language model. Check your connection and try again.",
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
    console.log(`  Model: ${MODEL} (HuggingFace)  ·  temp ${SAMPLING.temperature}\n`);
  }
});
