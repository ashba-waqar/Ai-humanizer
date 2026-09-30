# Humanizer

A small full-stack app that takes text and **rephrases it into varied, natural, human-sounding prose** using an NVIDIA NIM language model. It keeps the same meaning while deliberately varying sentence structure and avoiding repeated or overly predictable word choices.

## How the constraints are met

Core sampling constraints:

| Constraint | Implementation |
| --- | --- |
| Temperature below 0.5 | `temperature: 0.45` |
| Avoid highly probable / repeated words | `frequency_penalty: 0.9`, `presence_penalty: 0.7`, plus explicit prompt rules |
| No em dashes | Prompt forbids em/en dashes, and `stripDashes()` removes any that slip through |
| Unicode spacing | `applyUnicodeSpacing()` appends one random **invisible zero-width** mark (ZWSP `U+200B`, ZWNJ `U+200C`, ZWJ `U+200D`, ZWNBSP `U+FEFF`) after **every** word — these are the characters Microsoft Word reliably hides |

### Academic style, with detector-aware variation

The rewriter targets a **formal academic register** (matching a `STYLE_SAMPLE` embedded in the prompt) while still avoiding the machine tells that are compatible with scholarly writing:

| Goal | How the prompt handles it |
| --- | --- |
| Match academic voice | Emulates the register/rhythm of an embedded `STYLE_SAMPLE` reference |
| Preserve citations & terms | Keeps `[1]`, `[2,3]`, acronyms (SG, AMI, DSM…) exactly |
| Predictability / low perplexity | Precise, specific word choice; avoid the most-likely generic option |
| Uniform sentence structure | Vary length and construction; alter how sentences open |
| Repetitive vocabulary | No reused transitions or phrasing; rotate connectives |
| Excessive transition words | Do not lean on any single connective; avoids "in conclusion" clichés |
| Generic filler | Stay specific and substantive |
| Formal tone & grammar | Objective third-person register; **correct** grammar (no fragments/slang/contractions) |

> Note: unlike a casual "humanizer", this build deliberately keeps formal grammar and tone because the target output is academic. The conversational/personal-voice rules were removed.

All sampling knobs and the prompt (including `STYLE_SAMPLE`) live at the top of [`server.js`](server.js) so they're easy to tune.

## Stack

- **Backend:** Node.js + Express — keeps the API key server-side (never exposed to the browser).
- **Frontend:** vanilla HTML/CSS/JS served from `public/`.
- **Model:** auto-detected at startup from **NVIDIA NIM** (OpenAI-compatible endpoint). Any chat model the key can invoke works — override with `MODEL_ID` in `.env`, or change `PREFERRED_MODELS` in `server.js`.

## Setup

1. Requires **Node.js 18+** (uses the built-in `fetch`).
2. Add your NVIDIA API key to `.env`:

   ```
   NVIDIA_API_KEY=nvapi-...
   ```

   Create a free key (with starter credits) at <https://build.nvidia.com>.

   The server auto-detects a usable model at startup: it lists the catalogue,
   then sends a tiny test call to each candidate until one answers, because
   NVIDIA lists models a given key cannot actually invoke. To pin one
   yourself, add `MODEL_ID=<model-id>` to `.env`.

3. Install dependencies and start:

   ```bash
   npm install
   npm start
   ```

4. Open <http://localhost:3000>.

## Usage

- Paste text into the **Source** panel, click **Humanize** (or press `Ctrl/Cmd + Enter`).
- The rewritten text appears in the **Humanized** panel; use **Copy** to grab it.
- Toggle light/dark with the button in the header.

## Project layout

```
server.js        Express server + POST /api/humanize
public/          Frontend (index.html, styles.css, app.js)
.env             NVIDIA_API_KEY, optional MODEL_ID (git-ignored)
```
