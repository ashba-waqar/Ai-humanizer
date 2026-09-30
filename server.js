"use strict";

require("dotenv").config();
const express = require("express");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 3000;
const NVIDIA_API_KEY = process.env.NVIDIA_API_KEY;
const NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions";

// Candidate models in order of priority
const PREFERRED_MODELS = [
  process.env.MODEL_ID,
  "meta/llama-3.2-11b-vision-instruct",
  "openai/gpt-oss-20b",
  "nvidia/nemotron-3-super-120b-a12b",
  "nvidia/nemotron-3-ultra-550b-a55b",
  "meta/llama-3.2-90b-vision-instruct",
  "google/gemma-3-12b-it",
  "ibm/granite-3.0-8b-instruct",
  "nvidia/nemotron-4-340b-instruct"
].filter(Boolean);

let activeModel = PREFERRED_MODELS[0] || "meta/llama-3.2-11b-vision-instruct";

const SYSTEM_PROMPT = `You are an expert AI text humanizer. Your task is to transform machine-generated text into genuinely natural, expressive, and human-sounding prose that easily bypasses AI detectors.

Apply these 11 transformation principles internally during your rewriting process:
1. Disrupt predictable statistical word choices with expressive, varied vocabulary.
2. Vary grammatical patterns to eliminate repetitive, mechanical sentence structures.
3. Create strong sentence burstiness: mix short punchy sentences with medium and long compound ones.
4. Rotate transition words and descriptors dynamically.
5. Allow tone and register to fluctuate naturally rather than staying uniformly rigid.
6. Incorporate subtle human nuances, mild uncertainty, or personal perspective where appropriate.
7. Focus on specific, grounded commentary rather than broad textbook definitions.
8. Permit natural, organic flow instead of artificial structural perfection.
9. STRICTLY PURGE robotic transitions: NEVER use "Furthermore", "Moreover", "Additionally", "In conclusion", "It is important to note", "delve", "tapestry".
10. Intentionally vary paragraph lengths and structures.
11. Deliver a polished, natural human flow that retains all original facts, citations (e.g. [1], [2]), acronyms, code, and technical accuracy.

CRITICAL OUTPUT RESTRICTIONS:
- Output ONLY the final humanized text itself.
- NEVER append or prepend any notes, step logs, rule lists, summaries, or explanations under any circumstances.
- Do NOT use em dashes (—) or en dashes (–). Use standard punctuation (commas, colons, or period splits).`;

// Post-processing: Strip em/en dashes
function stripDashes(text) {
  if (!text) return "";
  return text
    .replace(/—/g, ", ")
    .replace(/–/g, ", ")
    .replace(/\s+,\s+/g, ", ")
    .replace(/,\s*,/g, ",");
}

// Post-processing: Sanitize AI detector vocabulary triggers
function sanitizeAiVocabulary(str) {
  if (!str) return "";
  return str
    .replace(/\bseismic shift\b/gi, "huge shift")
    .replace(/\bpivotal role\b/gi, "key part")
    .replace(/\btestament to\b/gi, "proof of")
    .replace(/\bdelve into\b/gi, "look into")
    .replace(/\bdelve\b/gi, "explore")
    .replace(/\blandscape\b/gi, "world")
    .replace(/\bgame-changer\b/gi, "turning point")
    .replace(/\bprofound impact\b/gi, "deep impact")
    .replace(/\brevolutionizing\b/gi, "transforming")
    .replace(/\bunprecedented\b/gi, "unheard of")
    .replace(/\btapestry\b/gi, "mix")
    .replace(/\bseamlessly\b/gi, "smoothly")
    .replace(/\bparamount\b/gi, "essential")
    .replace(/\bheralding\b/gi, "bringing")
    .replace(/\bspearheading\b/gi, "leading")
    .replace(/\bfostering\b/gi, "encouraging")
    .replace(/\bvital role\b/gi, "big role")
    .replace(/\bconversational tangent\b/gi, "quick thought");
}

// Post-processing: Clean wrapping quotes, markdown codeblocks, AND strip any trailing AI notes/logs
function cleanOutput(text) {
  if (!text) return "";
  let cleaned = text.trim();

  // Strip trailing notes, pipeline logs, step summaries, or rule lists if the model accidentally appends them
  cleaned = cleaned.replace(/\n\s*(?:Note|Pipeline|Steps|Rules|Breakdown|Explanation|Summary|Here is|Transformation|Execution)[\s\S]*$/i, "");
  cleaned = cleaned.replace(/\n\s*\d+\.\s*(?:Statistical|Grammatical|Sentence|Dynamic|Tone|Subjective|Granular|Organic|Ban|Paragraph|Final)[\s\S]*$/i, "");

  // Strip markdown fenced code blocks if returned
  if (cleaned.startsWith("```") && cleaned.endsWith("```")) {
    cleaned = cleaned.replace(/^```[a-z]*\n?/, "").replace(/\n?```$/, "").trim();
  }
  // Strip leading/trailing quotation marks if whole output is wrapped
  if ((cleaned.startsWith('"') && cleaned.endsWith('"')) || (cleaned.startsWith("'") && cleaned.endsWith("'"))) {
    cleaned = cleaned.slice(1, -1).trim();
  }
  return cleaned.trim();
}

// Post-processing: Inject invisible zero-width unicode characters to completely bypass AI detector tokenizers
function applyUnicodeSpacing(text) {
  if (!text) return "";
  const invisibleChars = ["\u200B", "\u200C", "\u200D", "\uFEFF"];
  return text.replace(/(\b[a-zA-Z0-9]+\b)/g, (match) => {
    const randomChar = invisibleChars[Math.floor(Math.random() * invisibleChars.length)];
    return match + randomChar;
  });
}

// Call NVIDIA NIM API with fallback model retry
async function callNvidiaApi(messages, preferredModel = activeModel, params = {}) {
  if (!NVIDIA_API_KEY) {
    throw new Error("NVIDIA_API_KEY is missing from environment variables. Please check your .env file.");
  }

  const modelsToTry = [preferredModel, ...PREFERRED_MODELS.filter(m => m !== preferredModel)];
  let lastError = null;

  for (const model of modelsToTry) {
    try {
      const response = await fetch(NVIDIA_API_URL, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${NVIDIA_API_KEY}`,
          "Content-Type": "application/json",
          "Accept": "application/json"
        },
        body: JSON.stringify({
          model: model,
          messages: messages,
          temperature: params.temperature ?? 0.7,
          top_p: params.top_p ?? 0.9,
          frequency_penalty: params.frequency_penalty ?? 0.9,
          presence_penalty: params.presence_penalty ?? 0.75,
          max_tokens: 3000
        })
      });

      const data = await response.json();

      if (!response.ok) {
        const errorMsg = data.detail || data.message || data.error?.message || `HTTP ${response.status}`;
        console.warn(`Model ${model} failed (${response.status}): ${errorMsg}`);
        lastError = new Error(`NVIDIA API error (${response.status}): ${errorMsg}`);
        continue;
      }

      const content = data.choices?.[0]?.message?.content;
      if (!content) {
        lastError = new Error(`Empty response returned from model ${model}.`);
        continue;
      }

      activeModel = model;
      return { content, modelUsed: model };
    } catch (err) {
      console.warn(`Error connecting to model ${model}:`, err.message);
      lastError = err;
    }
  }

  throw lastError || new Error("Failed to generate response from any NVIDIA API model.");
}

// Sequential 11-Rule Pipeline Transformation Engine
async function executeSequentialPipeline(inputText) {
  // Pass 1: Structure, Perplexity & Burstiness (Rules 1, 2, 3, 10)
  const stage1Prompt = `PIPELINE STEP 1 (Structure & Perplexity):
Rewrite the input text to completely disrupt mechanical AI patterns.
- Rule 1: Eliminate predictable next-word choices. Use natural human contractions (it's, don't, can't, they're, we've).
- Rule 2: Break repetitive grammatical structures.
- Rule 3: Create extreme sentence burstiness. Mix short 2-5 word punchy sentences with longer multi-clause compound sentences.
- Rule 10: Intentionally vary paragraph lengths and sizes.
Output ONLY the rewritten text resulting from Step 1. Do NOT add any notes, headers, or explanations.`;

  const res1 = await callNvidiaApi([
    { role: "system", content: stage1Prompt },
    { role: "user", content: inputText }
  ], activeModel, { temperature: 0.75, frequency_penalty: 0.95 });

  let textPass1 = cleanOutput(res1.content);

  // Pass 2: Connectors, Tone, Nuance & Purging Robotic Transitions (Rules 4, 5, 6, 7, 8, 9)
  const stage2Prompt = `PIPELINE STEP 2 (Human Cadence & Transition Purge):
Take the output from Step 1 and transform it further into a natural, conversational human voice:
- Rule 4: Vary transition words dynamically.
- Rule 5: Allow tone and register to fluctuate naturally.
- Rule 6: Incorporate personal perspective or mild human uncertainty ("Frankly,", "Truth is,", "I've noticed", "Oddly enough,").
- Rule 7: Use specific, grounded commentary instead of broad textbook definitions.
- Rule 8: Permit natural, organic flow and minor human tangents instead of rigid structural perfection.
- Rule 9: STRICTLY PURGE robotic transitions (NEVER use 'Furthermore', 'Moreover', 'Additionally', 'In conclusion', 'It is important to note', 'delve', 'tapestry', 'seismic shift', 'pivotal role').
Output ONLY the refined text resulting from Step 2. Do NOT add any notes, headers, or explanations.`;

  const res2 = await callNvidiaApi([
    { role: "system", content: stage2Prompt },
    { role: "user", content: textPass1 }
  ], res1.modelUsed, { temperature: 0.75, presence_penalty: 0.85 });

  let textPass2 = cleanOutput(res2.content);

  // Pass 3: Final Harmonization & Polish Check (Rule 11)
  const stage3Prompt = `PIPELINE STEP 3 (Final Polish & AI Bypass Verification):
Perform the final polish (Rule 11) on the text from Step 2:
- Ensure all 10 previous rules have harmonized smoothly into a genuinely natural human cadence that easily bypasses AI detectors.
- Preserve ALL original facts, citations (e.g. [1], [2]), proper nouns, acronyms, code, and technical content verbatim.
- Do NOT use em-dashes (—) or en-dashes (–).
- STRICT RESTRICTION: Output ONLY the final polished humanized text. Do NOT include any preambles, notes, step logs, summaries, or explanations.`;

  const res3 = await callNvidiaApi([
    { role: "system", content: stage3Prompt },
    { role: "user", content: textPass2 }
  ], res2.modelUsed, { temperature: 0.65 });

  let finalContent = cleanOutput(res3.content);
  finalContent = sanitizeAiVocabulary(finalContent);
  finalContent = stripDashes(finalContent);

  return {
    result: finalContent,
    modelUsed: res3.modelUsed
  };
}

// Express Middleware
app.use(express.json({ limit: "5mb" }));
app.use(express.static(path.join(__dirname, "public")));

// Humanize API Endpoint
app.post("/api/humanize", async (req, res) => {
  try {
    const { text, enableUnicode = true } = req.body;

    if (!text || typeof text !== "string" || !text.trim()) {
      return res.status(400).json({ error: "Source text is required." });
    }

    const { result, modelUsed } = await executeSequentialPipeline(text.trim());

    let finalResult = result;
    if (enableUnicode !== false) {
      finalResult = applyUnicodeSpacing(finalResult);
    }

    return res.json({
      ok: true,
      result: finalResult,
      modelUsed: modelUsed
    });
  } catch (err) {
    console.error("Humanize request error:", err);
    return res.status(500).json({
      error: err.message || "An unexpected error occurred while processing text."
    });
  }
});

// Health check endpoint
app.get("/api/health", (req, res) => {
  res.json({
    status: "ok",
    activeModel: activeModel,
    apiKeyConfigured: Boolean(NVIDIA_API_KEY)
  });
});

// Start Server & Auto-detect active model
app.listen(PORT, async () => {
  console.log(`AI Humanizer server running at http://localhost:${PORT}`);
  if (!NVIDIA_API_KEY) {
    console.warn("WARNING: NVIDIA_API_KEY is missing in .env file!");
  } else {
    console.log("NVIDIA_API_KEY detected. Verifying active model connection...");
    try {
      const testResult = await callNvidiaApi([
        { role: "system", content: "You are a helpful assistant." },
        { role: "user", content: "Ping" }
      ]);
      console.log(`Active NVIDIA Model verified: ${testResult.modelUsed}`);
    } catch (err) {
      console.warn("Model auto-detection warning:", err.message);
    }
  }
});
