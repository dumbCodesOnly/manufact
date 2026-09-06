// ---------------------------------------------------------------------------
// connectors/delegate/agent/agent_delegate.js — read-only investigation loop.
//
// Lets Gemini run its OWN multi-step tool-use loop server-side (via Gemini
// function calling) to answer an open-ended question, instead of the
// calling model doing 5-10 separate manual tool round-trips. One
// delegate_agent call in, one synthesized answer out.
//
// SCOPE: every delegated function below is READ-ONLY. Gemini is never given
// a write-capable function here -- writes stay confined to the fixed
// GEMINI_NOTION_ROOT_PAGE_ID path in agent_tools.js, same isolation rule as
// delegate_research. This file only reaches into GitHub/Cloudflare/Notion's
// existing client-layer functions (not the MCP tool layer) to avoid
// round-tripping through the MCP server for its own internal calls.
//
// IMPORTANT -- INDEPENDENT FROM THE MCP-FACING TOOL DESCRIPTIONS:
// The `description` strings on FUNCTIONS below are what GEMINI sees during
// its own tool-calling loop. They are entirely separate from the
// server.tool(...) descriptions the CALLING MODEL (e.g. Claude) sees for
// read_file/get_file_tree/list_directory/etc. in connectors/github/files.js
// (and equivalents in other connectors/*/tools.js files). Editing one set
// does NOT affect the other -- they are different objects read by different
// models for different purposes.
// Concretely: connectors/github/files.js's read_file/get_file_tree descriptions
// carry "RULE for the calling model: ... use delegate_agent instead" text
// aimed at steering Claude away from manual multi-file loops. Do NOT copy
// that kind of "use delegate_agent instead" language onto github_read_file/
// github_get_file_tree/etc. below -- Gemini calling one of these FUNCTIONS
// *is* delegate_agent already running; a self-referential "delegate to
// delegate_agent" hint here would be nonsensical and could confuse Gemini
// into stalling instead of just calling the function. Keep these
// descriptions plain and factual, matching what they actually do.
//
// STEP CAP: HARD_MAX_STEPS bounds the loop regardless of the caller's
// max_steps argument -- both to bound Gemini API cost and because a
// synchronous madmcp tool call has to fit inside the hosting platform's
// request duration limit (a real constraint on Vercel -- see the Notion
// plan page for the "known constraint" note; unresolved as of writing).
// ---------------------------------------------------------------------------

import { randomUUID } from "node:crypto";
import { providerChat } from "../../llm/router.js";
import { formatCascadeLogLine } from "../../llm/cascade_log.js";
import { saveCheckpoint, loadCheckpoint, deleteCheckpoint, savePreCompactionResult, getPreCompactionResults, saveResultCacheEntry, getResultCacheEntries } from "./agent_checkpoint.js";
import { isRedisConfigured } from "../../shared/cooldown.js";
import { githubRequest } from "../../github/client.js";
import { readFileViaBlob } from "../../github/helpers.js";
import { extractRepoQualifier, fallbackCodeSearch } from "../../github/search.js";
import { queryTelemetry, toEpochMillis } from "../../cloudflare/observability.js";
import { cfAccountRequest } from "../../cloudflare/client.js";
import { context7Request } from "../../context7/client.js";
import { mem0Request } from "../../mem/client.js";
import { notionRequest, notionRichTextToString, notionPageTitle, notionDatabaseTitle, notionBlocksToText } from "../../notion/client.js";
import { DEFAULT_OWNER } from "../../../config.js";
import { getDelegateHooks } from "../provider_hooks.js";
import { selectPreambleVariant, appendTask } from "../shared/preamble.js";

const HARD_MAX_STEPS = 30;
export const HISTORY_FULL_DETAIL_STEPS = 3;
export const COMPACTION_CHAR_THRESHOLD = 500;

// Oversized-step guardrails (2026-09-02, plan.md Section 9 root cause fix):
// a single step batching many tool calls and/or large truncated file reads
// bloats that step's contents turn, which in turn makes the FOLLOWING
// outbound LLM call (sending that bloated history back to bai/Gemini) slow
// enough to routinely exceed Vercel's 300s hard function timeout -- a
// platform kill that bypasses this repo's own retry/dead-letter logic
// entirely (see agent_worker.js's retryCount comment). These two caps bound
// a single step's contribution to context size directly, independent of
// per-call truncation (sliceFileContentForModel's own 30000/100000-char
// limits), since per-call limits alone don't stop the model from batching
// many calls -- each individually within its own limit -- into one step.
// MAX_TOOL_CALLS_PER_STEP bounds fan-out (a step that batched 13 calls was
// the reproduction in plan.md Section 7); MAX_STEP_RESULT_CHARS bounds the
// aggregate payload size actually appended to `contents` regardless of call
// count. Both are deliberately conservative but not tiny -- ordinary steps
// (1-3 calls, one moderate file read) are far under either cap and are
// completely unaffected.
export const MAX_TOOL_CALLS_PER_STEP = 8;
// CONFIRMED UNSAFE AT 300000 (2026-09-02, plan.md §6 item 9 follow-up): the
// 60000 -> 100000 raise was live-timing-tested and confirmed comfortably
// safe (invocation gaps 15-26s, ~5-9% of Vercel's 300s ceiling). The
// further raise, 100000 -> 300000, was NOT confirmed safe -- two
// independent live reproductions (forcing step 1's aggregate payload past
// 300000 chars) each produced a genuine step-2 stall, root-caused against
// real Vercel runtime-error timestamps (not just an inferred QStash error
// string) to `Vercel Runtime Timeout Error: Task timed out after 300
// seconds`, route `/api/agent-worker`. See plan.md §6 item 9 for the full
// timestamped evidence from both runs.
// DECISION (2026-09-02): lowered to 270000 rather than re-testing 200000
// first -- there is zero live data anywhere between the confirmed-safe
// 100000 floor and the confirmed-unsafe 300000 ceiling, so 200000 needs
// the same live validation 270000 does before either can be trusted; pick
// the value closer to the original target and validate that one. This is
// an UNTESTED value pending the same live-timing validation 100000
// received before being trusted -- see plan.md §6 for the test plan and
// results once run. If 270000 also proves unsafe, fall back to the
// confirmed-safe 100000 floor rather than continuing to iterate upward on
// this constant without a different way to bound the outbound payload
// (e.g. compressing/summarizing large file reads instead of just capping
// raw chars).
export const MAX_STEP_RESULT_CHARS = 270000;
export const BULKY_TOOL_NAMES = new Set([
  "github_read_file",
  "github_get_file_at_commit",
  "github_get_file_tree",
  "cf_query_logs",
  "cf_workers_get_worker_code",
  "github_get_workflow_run_logs",
  "github_get_job_logs",
  "context7_get_library_docs",
  "notion_get_page",
  "github_get_pull_request",
  "github_search_issues",
  "github_diff_files",
  "github_search_code",
]);

// History compaction (2026-08-27): reduces token usage on long multi-step
// investigations by replacing bulky older tool results with short summary
// pointers in place.
//
// Hard constraints:
// 1. Do NOT delete, reorder, or remove any turns/messages from contents.
// 2. Do NOT break functionCall/functionResponse id pairing or role alternation.
// 3. Only edit the text inside an existing functionResponse part in place.
//
// Gated on getDelegateHooks(provider).historyCompactionEnabled (see
// connectors/delegate/provider_hooks.js) so non-opted-in providers (like
// default gemini) remain completely unaffected byte-for-byte -- which
// providers are opted in is bai's own concern (connectors/bai/
// delegate_hooks.js), not something this file branches on directly.

// `functionResponse.id` only needs to be unique WITHIN a single turn for Gemini's
// own call/response pairing -- nothing in this codebase or in Gemini's contract
// guarantees it stays unique ACROSS the whole multi-step run. `compactHistoryInPlace`
// currently keys `preCompactionResults` directly by that id via a plain `.set()` call;
// if two different steps' calls ever reused the same id, a later compaction would
// silently overwrite an earlier turn's saved text, permanently dropping it from
// `findUnverifiedClaims`'s coverage. This collision-safe helper stores a value
// under `id` normally, but if `id` already maps to a DIFFERENT existing value,
// stores the new value under a disambiguated key instead (`${id}#2`, etc.) rather
// than ever overwriting a differing prior entry.
function setPreCompactionResult(preCompactionResults, id, text) {
  if (!preCompactionResults.has(id) || preCompactionResults.get(id) === text) {
    preCompactionResults.set(id, text);
    return id;
  }
  let suffix = 2;
  let altKey = `${id}#${suffix}`;
  while (preCompactionResults.has(altKey) && preCompactionResults.get(altKey) !== text) {
    suffix++;
    altKey = `${id}#${suffix}`;
  }
  preCompactionResults.set(altKey, text);
  return altKey;
}

export async function compactHistoryInPlace(contents, currentStep, preCompactionResults, {
  fullDetailSteps = HISTORY_FULL_DETAIL_STEPS,
  charThreshold = COMPACTION_CHAR_THRESHOLD,
  bulkyTools = BULKY_TOOL_NAMES,
  provider,
  runId,
} = {}) {
  const isEnabled = provider ? getDelegateHooks(provider).historyCompactionEnabled : false;
  if (!isEnabled || !Array.isArray(contents)) return;

  // Side-store writes (addressing state-checkpoint bloat): every
  // first-time compaction below also persists its full text to Redis via
  // savePreCompactionResult, not just the in-memory `preCompactionResults`
  // Map -- meta's checkpoint blob only carries the Map, which is what made
  // the now-removed 2eea726 eviction "necessary" in the first place. The
  // writes are collected here and fired at the end (see the Promise.all
  // below) rather than awaited inline, so every synchronous mutation to
  // `contents`/`preCompactionResults` in the loop below still completes
  // before this function's first `await` -- callers (and every existing
  // test, none of which await this call) that only care about those
  // synchronous mutations keep working unchanged; only callers that want to
  // know the side-store writes have actually landed need to await this.
  const sideStoreWrites = [];

  let modelTurnStack = [];
  let stepIndex = 0;
  for (const turn of contents) {
    if (turn.role === "model") {
      modelTurnStack.push(turn);
    } else if (turn.role === "user" && Array.isArray(turn.parts)) {
      const hasFunctionResponses = turn.parts.some((p) => p.functionResponse);
      if (hasFunctionResponses) {
        stepIndex++;
        const recentModelTurn = modelTurnStack[modelTurnStack.length - 1];
        // At this point in the loop, `contents` holds completed response turns for steps `1..currentStep-1`.
        // `currentStep` represents the step number about to be attempted (1-indexed).
        // Since `contents` contains completed turns only up to `currentStep - 1`, we use
        // `(currentStep - 1)` as the basis to apply the "keep last `fullDetailSteps` steps" rule.
        if (stepIndex <= (currentStep - 1) - fullDetailSteps) {
          for (const part of turn.parts) {
            if (part.functionResponse && bulkyTools.has(part.functionResponse.name)) {
              const res = part.functionResponse.response;
              if (res && typeof res.result === "string" && res.result.length > charThreshold) {
                // Store original text before compacting -- collision-safe, see setPreCompactionResult above.
                const storedKey = setPreCompactionResult(preCompactionResults, part.functionResponse.id, res.result);
                if (runId) {
                  sideStoreWrites.push(savePreCompactionResult(runId, storedKey, res.result));
                }

                const originalLength = res.result.length;
                const toolName = part.functionResponse.name;
                let target = "";
                if (recentModelTurn && recentModelTurn.parts) {
                  const call = recentModelTurn.parts.find(p => p.functionCall?.id === part.functionResponse.id);
                  if (call) {
                    const args = call.functionCall.args || {};
                    target = args.path || args.pull_number || args.query || args.run_id || args.job_id || "";
                  }
                }
                const targetDisplay = target ? ` (${target})` : "";
                res.result = `[Earlier tool result compacted: ${toolName}${targetDisplay}, originally ${originalLength} chars — call the tool again if the exact content is needed; resultCache will serve it without a new network round trip.]`;
              }
            }
          }
        }
      }
    }
  }

  if (sideStoreWrites.length) await Promise.all(sideStoreWrites);
}

// 429 (rate limit) and 503 (overloaded/high demand) are the documented
// transient HTTP statuses -- see client.js's own model-fallback cascade,
// which deliberately only retries a different model on a 429 for the same
// reason. This also retries anything the adapter layer has explicitly
// flagged via `err.transient === true`, independent of `err.status` --
// that flag lets a provider adapter (e.g. bai's) mark its own errors as
// safe to retry without this file needing to know that provider's
// specific status-code conventions. Everything else (400 malformed
// request, 401/403 auth, 404 unknown model, or no err.status/err.transient
// at all -- e.g. "GEMINI_API_KEY is not set" thrown locally in client.js,
// or "Gemini returned no candidates" from a safety/recitation block) is a
// config or request problem that will reproduce identically on a resume,
// not something retrying fixes.
function isTransientGeminiError(err) {
  return err?.status === 429 || err?.status === 503 || err?.transient === true;
}

// Validates a model-supplied args object against a FUNCTIONS[] entry's own
// declared JSON schema (parameters.properties/required) before execute()
// ever runs. Added after live testing showed a model (GLM, but nothing
// about this is provider-specific -- both providers share this exact
// execute() path) passing a `repo` parameter to github_search_code, a tool
// whose schema has no such property (repo-scoping for that tool is meant
// to go inside the `query` string, per its own description) -- the extra
// property was silently ignored rather than rejected, so the call ran
// unscoped and returned irrelevant results with no signal for the model to
// self-correct, and it repeated the same mistake on every retry.
// Deliberately shallow (top-level property names only, not full JSON
// Schema validation of types/nested shapes) -- this is meant to catch the
// "model invented or omitted a parameter" class of mistake cheaply, not to
// replace real schema validation.
function validateFunctionArgs(fn, args) {
  const properties = fn.parameters?.properties || {};
  const required   = fn.parameters?.required || [];
  const allowedKeys   = Object.keys(properties);
  const providedKeys  = Object.keys(args || {});

  const missing = required.filter((k) => args?.[k] === undefined || args?.[k] === null || args?.[k] === "");
  const unknown = providedKeys.filter((k) => !allowedKeys.includes(k));

  if (!missing.length && !unknown.length) return null;

  const lines = [`Error: invalid arguments for ${fn.name} -- this call was NOT executed.`];
  if (missing.length) {
    lines.push(`Missing required parameter(s): ${missing.join(", ")}.`);
  }
  if (unknown.length) {
    lines.push(
      `Unknown parameter(s) not accepted by this tool: ${unknown.join(", ")}. ` +
      `Valid parameters for ${fn.name} are: ${allowedKeys.join(", ") || "(none)"}. ` +
      `Re-read this tool's own description for how to express what you intended -- ` +
      `some tools (e.g. github_search_code) expect scoping like "repo:owner/name" embedded ` +
      `directly inside a query string rather than as a separate parameter, and parameter shapes ` +
      `are NOT consistent across every tool in this list.`
    );
  }
  return lines.join(" ");
}

// Repeat-detection signature (fix for redundant/wasted tool calls found in
// live testing 2026-08-27 -- the redundant tool call dedup fix).
// Used by the stuck-loop guard below in place of the old raw
// `${name}:${JSON.stringify(args || {})}` signature, which had two gaps
// that let genuine repeats slip past the dedup cache:
//
// Gap 1 -- key-order sensitivity: JSON.stringify on a plain JS object
// serializes keys in insertion order, not sorted/canonical order. Two
// functionCall args with identical VALUES but different key order in
// Gemini's own emitted JSON (confirmed as a real, observed pattern in
// live-testing transcripts -- Gemini's key order for a given call is not
// guaranteed stable across turns) therefore produced two different
// signature strings and were never recognized as the same call. Fixed
// below by sorting keys before stringifying -- this is a pure JS-semantics
// fix (JSON.stringify key order depends only on insertion order, which is
// enough on its own to explain the gap), independent of the live-transcript
// evidence that motivated looking for it.
//
// Gap 2 -- no cross-tool/ref semantic equivalence: github_read_file(no ref
// or ref: "HEAD") and github_get_file_at_commit(commit: "HEAD") on the same
// path resolve to the exact same content (both are the tip of the default
// branch), but were treated as entirely distinct signatures -- different
// function names, and even different parameter names (ref vs commit) for
// the same concept. This is intentionally a LIGHTWEIGHT fix, not the full
// resolve-to-the-actual-default-branch-SHA normalization described in the
// original handoff: it only collapses the cheap-to-detect case of an
// omitted ref/commit or the literal string "HEAD" across those two
// functions for the same (owner, repo, path). It does NOT resolve a named
// branch (e.g. "main") that happens to currently equal the default branch
// to that same signature -- doing that correctly requires an extra API
// call to look up the default branch before every single dedup check
// (i.e. paying latency/cost on every call to occasionally catch a rarer
// redundant one), which is a bad trade for the common case. If this
// lightweight version doesn't catch enough real-world redundancy, revisit
// with the heavier default-branch-SHA resolution.
// Shared slicing logic for github_read_file/github_get_file_at_commit below.
// Mirrors connectors/github/files.js's sliceFileContent (the MCP-facing
// read_file's helper) but returns a plain string, not { content: [...] }
// blocks -- these execute() results feed back into GEMINI's own
// conversation as a function response, not out to the calling model, so
// there's no MCP content-block wrapping to match. Kept as a distinct copy
// rather than importing the other one: the two tool surfaces are
// deliberately decoupled (see the file-header note on why editing one
// FUNCTIONS entry must never assume it affects the other model's tools),
// and a shared import would blur that boundary for a few lines of logic.
//
// Default (no char_offset/char_limit): matches the old hard-30000-cutoff
// behavior in spirit, but replaces the old bare "...[truncated]" marker
// (which gave Gemini no way to ever see the rest) with an explicit total
// length and the exact char_offset to pass next.
function sliceFileContentForModel(content, path, { char_offset, char_limit }) {
  const total = content.length;

  if (char_offset === undefined && char_limit === undefined) {
    if (total <= 30000) return content;
    const slice     = content.slice(0, 30000);
    const remaining = total - 30000;
    return (
      `[File: ${path} | Total: ${total} chars | Offset: 0 | Returning: ${slice.length} chars | Remaining: ${remaining} chars]\n` +
      `Pass char_offset=30000 to this same function to keep reading.\n\n${slice}`
    );
  }

  const offset    = char_offset ?? 0;
  const safeLimit = Math.min(char_limit ?? 30000, 200000);
  const slice     = content.slice(offset, offset + safeLimit);
  const remaining = Math.max(0, total - offset - slice.length);
  const header    = `[File: ${path} | Total: ${total} chars | Offset: ${offset} | Returning: ${slice.length} chars | Remaining: ${remaining} chars]\n\n`;
  return header + slice;
}

const READ_FILE_SIGNATURE_FAMILY = new Set(["github_read_file", "github_get_file_at_commit"]);

function normalizedSignature(name, args) {
  const a = args || {};
  const sortedArgs = {};
  for (const key of Object.keys(a).sort()) sortedArgs[key] = a[key];

  if (READ_FILE_SIGNATURE_FAMILY.has(name)) {
    const refValue = "ref" in sortedArgs ? sortedArgs.ref : sortedArgs.commit;
    const isHeadLike = refValue === undefined || refValue === null || refValue === "" || refValue === "HEAD";
    if (isHeadLike) {
      const { owner, repo, path, char_offset, char_limit } = sortedArgs;
      // char_offset/char_limit MUST be part of this signature -- omitting them
      // was a real bug (found live, 2026-08-27): every paginated call to the
      // same file (offset 0, 30000, 60000, ...) collapsed to one identical
      // signature and got served the cached FIRST chunk back regardless of
      // the offset actually requested, silently corrupting a multi-step read
      // into N repeats of the same 30000-char slice.
      return `github_read_file_family:${JSON.stringify({ owner, repo, path, ref: "HEAD", char_offset: char_offset ?? null, char_limit: char_limit ?? null })}`;
    }
  }

  return `${name}:${JSON.stringify(sortedArgs)}`;
}

// Minimal line-based diff (LCS backtrace) -- good enough for investigation
// summaries, not a full unified-diff implementation. Capped at 2000 lines
// (so a huge file pair can't blow up the O(n*m) table) unless a specific
// range is requested via start_line/end_line (1-indexed).
function simpleLineDiff(aText, bText, { start_line, end_line } = {}) {
  const aFull = aText.split("\n");
  const bFull = bText.split("\n");

  const hasRange = start_line !== undefined || end_line !== undefined;

  if (!hasRange && (aFull.length > 2000 || bFull.length > 2000)) {
    if (aText === bText) return "(files identical)";
    return `(files differ — too large for full line diff: file A has ${aFull.length} lines, file B has ${bFull.length} lines. Pass start_line and end_line parameters to request a diff of a specific slice.)`;
  }

  let a = aFull;
  let b = bFull;

  if (hasRange) {
    const start = (start_line ?? 1) - 1;
    const end = (end_line ?? Math.max(a.length, b.length));
    a = a.slice(start, end);
    b = b.slice(start, end);
  }

  if (a.length > 2000 || b.length > 2000) {
    if (a.join("\n") === b.join("\n")) return "(files identical)";
    return `(selected slice too large for line diff — slice has ${a.length} vs ${b.length} lines. Use a narrower start_line/end_line range.)`;
  }

  const dp = Array.from({ length: a.length + 1 }, () => new Array(b.length + 1).fill(0));
  for (let i = a.length - 1; i >= 0; i--) {
    for (let j = b.length - 1; j >= 0; j--) {
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  const lines = [];
  let i = 0, j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) { i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) { lines.push(`-${a[i]}`); i++; }
    else { lines.push(`+${b[j]}`); j++; }
  }
  while (i < a.length) { lines.push(`-${a[i]}`); i++; }
  while (j < b.length) { lines.push(`+${b[j]}`); j++; }
  return lines.length ? lines.join("\n") : "(files identical)";
}

// ---------------------------------------------------------------------------
// Delegated function declarations -- Gemini's "tools" param (a subset of
// OpenAPI schema: type/properties/required, no $ref/oneOf/etc support).
// Each entry pairs the Gemini-facing declaration with a local `execute`
// that calls the real connector client function.
// ---------------------------------------------------------------------------

const FUNCTIONS = [
  {
    name: "github_read_file",
    description: "Read a file's contents from a GitHub repository. ALWAYS call with no char_offset/char_limit first -- this returns the whole file up to 30,000 chars in one call (the common case), or the first 30,000 chars plus the file's total length and the exact char_offset to continue from if it's longer -- there is no separate chunked/paginated tool, this same function does both. Only pass char_offset/char_limit once you actually have a reason to: either THIS function's own prior response on this SAME file already told you it's longer than 30,000 chars and gave you the offset to continue, or a github_search_code hit already reports a specific line deep inside a file you independently know is large. Do not guess an offset on a file whose real size you haven't confirmed -- if in doubt, call with no params.",
    parameters: {
      type: "object",
      properties: {
        owner:       { type: "string", description: `Repository owner (default "${DEFAULT_OWNER}" if omitted)` },
        repo:        { type: "string", description: "Repository name" },
        path:        { type: "string", description: "File path within the repo" },
        ref:         { type: "string", description: "Branch, tag, or commit SHA (default: repo default branch)" },
        char_offset: { type: "number", description: "Character offset to start reading from. Omit for default behavior (whole file, or its first 30,000 chars if longer)." },
        char_limit:  { type: "number", description: "Maximum number of characters to return (default 30000, max 100000). Ignored if char_offset is also omitted." },
      },
      required: ["repo", "path"],
    },
    execute: async ({ owner = DEFAULT_OWNER, repo, path, ref, char_offset, char_limit }) => {
      const content = await readFileViaBlob(owner, repo, path, ref);
      return sliceFileContentForModel(content, path, { char_offset, char_limit });
    },
  },
  {
    name: "github_get_file_tree",
    description: "Recursively list all files and folders in a GitHub repository.",
    parameters: {
      type: "object",
      properties: {
        owner: { type: "string", description: "Repository owner" },
        repo:  { type: "string", description: "Repository name" },
        ref:   { type: "string", description: "Branch, tag, or commit SHA (default: repo default branch)" },
      },
      required: ["owner", "repo"],
    },
    execute: async ({ owner, repo, ref }) => {
      let treeSha;
      if (ref) {
        try {
          const refData = await githubRequest(`/repos/${owner}/${repo}/git/ref/heads/${encodeURIComponent(ref)}`);
          treeSha = refData.object.sha;
        } catch { treeSha = ref; }
      } else {
        const repoData   = await githubRequest(`/repos/${owner}/${repo}`);
        const branchData = await githubRequest(`/repos/${owner}/${repo}/git/ref/heads/${repoData.default_branch}`);
        treeSha = branchData.object.sha;
      }
      const data = await githubRequest(`/repos/${owner}/${repo}/git/trees/${treeSha}?recursive=1`);
      return data.tree.map((item) => `${item.type === "tree" ? "dir " : "file"} ${item.path}`).join("\n");
    },
  },
  {
    name: "github_list_commits",
    description: "List recent commits on a branch in a GitHub repository.",
    parameters: {
      type: "object",
      properties: {
        owner:    { type: "string", description: `Repository owner (default "${DEFAULT_OWNER}" if omitted)` },
        repo:     { type: "string", description: "Repository name" },
        branch:   { type: "string", description: "Branch name (default: repo default branch)" },
        per_page: { type: "number", description: "Number of commits to return (default 20, max 100)" },
      },
      required: ["repo"],
    },
    execute: async ({ owner = DEFAULT_OWNER, repo, branch, per_page = 20 }) => {
      const query = new URLSearchParams({ per_page: String(Math.min(per_page, 100)) });
      if (branch) query.set("sha", branch);
      const data = await githubRequest(`/repos/${owner}/${repo}/commits?${query}`);
      return data.map((c) => `${c.sha.slice(0, 7)} — ${c.commit.message.split("\n")[0]} (${c.commit.author?.name}, ${c.commit.author?.date?.slice(0, 10)})`).join("\n");
    },
  },
  {
    name: "github_search_issues",
    description: "Search issues and pull requests across GitHub using GitHub's issue-search syntax (label:, is:issue, is:open, stars:>N, org:, repo:, -repo:, -org:, no:assignee, etc., combined with spaces as AND). Useful for cross-repo discovery like good-first-issue scanning -- github_read_file/github_get_file_tree only work within a single already-known repo.",
    parameters: {
      type: "object",
      properties: {
        query:    { type: "string", description: "GitHub issue-search query string, e.g. 'label:\"good first issue\" is:open is:issue no:assignee stars:>2000 -org:someorg'" },
        sort:     { type: "string", description: "Sort field: created, updated, or comments (default: best-match relevance)" },
        order:    { type: "string", description: "Sort order: asc or desc (default: desc)" },
        per_page: { type: "number", description: "Number of results to return, max 100 (default 20)" },
      },
      required: ["query"],
    },
    execute: async ({ query, sort, order = "desc", per_page = 20 }) => {
      let path = `/search/issues?q=${encodeURIComponent(query)}&order=${order}&per_page=${Math.min(per_page, 100)}`;
      if (sort) path += `&sort=${sort}`;
      const data = await githubRequest(path);
      if (!data.items?.length) return "No results found.";
      const lines = data.items.map((item) => {
        const kind = item.pull_request ? "PR" : "Issue";
        const labels = item.labels?.length ? ` [${item.labels.map((l) => l.name).join(", ")}]` : "";
        const assignee = item.assignee ? ` (assigned: ${item.assignee.login})` : " (unassigned)";
        return `${kind} #${item.number} [${item.state}] ${item.title}${labels}${assignee} -- ${item.repository_url.replace("https://api.github.com/repos/", "")} | created ${item.created_at.slice(0, 10)} | ${item.html_url}`;
      });
      const text = `Found ${data.total_count} total result(s), showing ${data.items.length}:\n${lines.join("\n")}`;
      return text.length > 20000 ? text.slice(0, 20000) + "\n...[truncated]" : text;
    },
  },
  {
    name: "cf_query_logs",
    description: "Query Cloudflare Workers Observability logs/traces/events for a time range.",
    parameters: {
      type: "object",
      properties: {
        timeframe_from: { type: "string", description: "Start of time range, ISO 8601 or epoch millis" },
        timeframe_to:   { type: "string", description: "End of time range, ISO 8601 or epoch millis" },
        script_name:    { type: "string", description: "Optional: scope to one Worker script" },
        limit:          { type: "number", description: "Max results (default ~100)" },
      },
      required: ["timeframe_from", "timeframe_to"],
    },
    execute: async ({ timeframe_from, timeframe_to, script_name, limit }) => {
      const data = await queryTelemetry({ timeframe_from, timeframe_to, script_name, limit });
      return JSON.stringify(data).slice(0, 30000);
    },
  },
  {
    name: "notion_get_page",
    description: "Read a Notion page's title and text content by page ID (read-only). Use this after notion_search finds a candidate page, to actually see what's on it -- notion_search only returns titles/ids, not content.",
    parameters: {
      type: "object",
      properties: {
        page_id: { type: "string", description: "Notion page ID, e.g. from notion_search results" },
      },
      required: ["page_id"],
    },
    execute: async ({ page_id }) => {
      const [page, blocksData] = await Promise.all([
        notionRequest(`/pages/${page_id}`),
        notionRequest(`/blocks/${page_id}/children?page_size=100`),
      ]);
      const title   = notionPageTitle(page);
      const blocks  = blocksData.results || [];
      const content = notionBlocksToText(blocks) || "(no content)";
      const hasMore = blocksData.has_more ? "\n[note: page has more than 100 blocks, only the first 100 are shown]" : "";
      const text = `# ${title}\n${content}${hasMore}`;
      return text.length > 20000 ? text.slice(0, 20000) + "\n...[truncated]" : text;
    },
  },
  {
    name: "notion_search",
    description: "Search pages and databases in the Notion workspace (read-only).",
    parameters: {
      type: "object",
      properties: {
        query:       { type: "string", description: "Search query string" },
        filter_type: { type: "string", description: "Restrict to 'page' or 'database' (optional)" },
        page_size:   { type: "number", description: "Number of results (default 10, max 100)" },
      },
      required: ["query"],
    },
    execute: async ({ query, filter_type, page_size = 10 }) => {
      const body = { query, page_size };
      if (filter_type) body.filter = { value: filter_type, property: "object" };
      const data = await notionRequest("/search", { method: "POST", body });
      if (!data.results?.length) return "No results found.";
      return data.results.map((r) => {
        const title = r.object === "page" ? notionPageTitle(r) : (notionDatabaseTitle(r) || "(untitled)");
        return `[${r.object}] ${title} — id: ${r.id}`;
      }).join("\n");
    },
  },
  {
    name: "notion_query_database",
    description: "Query rows from a Notion database (read-only), with an optional filter.",
    parameters: {
      type: "object",
      properties: {
        database_id: { type: "string", description: "Notion database ID" },
        page_size:   { type: "number", description: "Number of rows (default 20, max 100)" },
      },
      required: ["database_id"],
    },
    execute: async ({ database_id, page_size = 20 }) => {
      const data = await notionRequest(`/databases/${database_id}/query`, { method: "POST", body: { page_size } });
      if (!data.results?.length) return "No rows found.";
      return data.results.map((row) => {
        const props = Object.entries(row.properties || {}).map(([name, val]) => {
          if (val.type === "title") return `${name}: ${notionRichTextToString(val.title)}`;
          if (val.type === "rich_text") return `${name}: ${notionRichTextToString(val.rich_text)}`;
          return `${name}: ${JSON.stringify(val[val.type] ?? "")}`;
        }).join(" | ");
        return `- ${props}`;
      }).join("\n");
    },
  },

  // -- GitHub: issues / PRs --------------------------------------------
  {
    name: "github_get_issue",
    description: "Read a single GitHub issue's full body, labels, assignees, and (optionally) comments.",
    parameters: { type: "object", properties: {
      owner: { type: "string" }, repo: { type: "string" }, issue_number: { type: "number" },
      include_comments: { type: "boolean" },
    }, required: ["repo", "issue_number"] },
    execute: async ({ owner = DEFAULT_OWNER, repo, issue_number, include_comments = false }) => {
      const issue = await githubRequest(`/repos/${owner}/${repo}/issues/${issue_number}`);
      let text = `#${issue.number} [${issue.state}] ${issue.title}\nLabels: ${(issue.labels || []).map(l => l.name).join(", ") || "none"}\nAssignees: ${(issue.assignees || []).map(a => a.login).join(", ") || "none"}\n\n${issue.body || "(no body)"}`;
      if (include_comments && issue.comments > 0) {
        const comments = await githubRequest(`/repos/${owner}/${repo}/issues/${issue_number}/comments?per_page=50`);
        text += "\n\n--- comments ---\n" + comments.map(c => `${c.user?.login}: ${c.body}`).join("\n---\n");
      }
      return text.length > 20000 ? text.slice(0, 20000) + "\n...[truncated]" : text;
    },
  },
  {
    name: "github_list_pull_requests",
    description: "List pull requests in a repo, optionally filtered by state.",
    parameters: { type: "object", properties: {
      owner: { type: "string" }, repo: { type: "string" }, state: { type: "string", description: "open, closed, or all (default open)" }, per_page: { type: "number" },
    }, required: ["repo"] },
    execute: async ({ owner = DEFAULT_OWNER, repo, state = "open", per_page = 20 }) => {
      const data = await githubRequest(`/repos/${owner}/${repo}/pulls?state=${state}&per_page=${Math.min(per_page, 100)}`);
      return data.map(pr => `#${pr.number} [${pr.state}${pr.draft ? " draft" : ""}] ${pr.title} (${pr.head?.ref} -> ${pr.base?.ref}) by ${pr.user?.login}`).join("\n") || "No pull requests found.";
    },
  },
  {
    name: "github_get_pull_request",
    description: "Get a single pull request's details, optionally including comments, reviews, and commits.",
    parameters: { type: "object", properties: {
      owner: { type: "string" }, repo: { type: "string" }, pull_number: { type: "number" },
      include_comments: { type: "boolean" }, include_reviews: { type: "boolean" }, include_commits: { type: "boolean" },
    }, required: ["repo", "pull_number"] },
    execute: async ({ owner = DEFAULT_OWNER, repo, pull_number, include_comments, include_reviews, include_commits }) => {
      const pr = await githubRequest(`/repos/${owner}/${repo}/pulls/${pull_number}`);
      let text = `#${pr.number} [${pr.state}] ${pr.title}\n${pr.head?.ref} -> ${pr.base?.ref} by ${pr.user?.login}\nMergeable: ${pr.mergeable} (${pr.mergeable_state})\n\n${pr.body || "(no body)"}`;
      if (include_comments) {
        const c = await githubRequest(`/repos/${owner}/${repo}/issues/${pull_number}/comments?per_page=50`);
        text += "\n\n--- comments ---\n" + c.map(x => `${x.user?.login}: ${x.body}`).join("\n---\n");
      }
      if (include_reviews) {
        const r = await githubRequest(`/repos/${owner}/${repo}/pulls/${pull_number}/reviews?per_page=50`);
        text += "\n\n--- reviews ---\n" + r.map(x => `${x.user?.login}: ${x.state} -- ${x.body || "(no comment)"}`).join("\n");
      }
      if (include_commits) {
        const cm = await githubRequest(`/repos/${owner}/${repo}/pulls/${pull_number}/commits?per_page=50`);
        text += "\n\n--- commits ---\n" + cm.map(x => `${x.sha.slice(0, 7)} ${x.commit.message.split("\n")[0]}`).join("\n");
      }
      return text.length > 25000 ? text.slice(0, 25000) + "\n...[truncated]" : text;
    },
  },
  {
    name: "github_get_pr_comments",
    description: "Get the conversation comments on a pull request.",
    parameters: { type: "object", properties: { owner: { type: "string" }, repo: { type: "string" }, pull_number: { type: "number" }, per_page: { type: "number" } }, required: ["repo", "pull_number"] },
    execute: async ({ owner = DEFAULT_OWNER, repo, pull_number, per_page = 50 }) => {
      const data = await githubRequest(`/repos/${owner}/${repo}/issues/${pull_number}/comments?per_page=${Math.min(per_page, 100)}`);
      return data.map(c => `${c.user?.login} (${c.created_at?.slice(0, 10)}): ${c.body}`).join("\n---\n") || "No comments.";
    },
  },
  {
    name: "github_get_pr_reviews",
    description: "Get the formal reviews (approve/request-changes/comment) on a pull request.",
    parameters: { type: "object", properties: { owner: { type: "string" }, repo: { type: "string" }, pull_number: { type: "number" }, per_page: { type: "number" } }, required: ["repo", "pull_number"] },
    execute: async ({ owner = DEFAULT_OWNER, repo, pull_number, per_page = 50 }) => {
      const data = await githubRequest(`/repos/${owner}/${repo}/pulls/${pull_number}/reviews?per_page=${Math.min(per_page, 100)}`);
      return data.map(r => `${r.user?.login}: ${r.state} -- ${r.body || "(no comment)"}`).join("\n") || "No reviews.";
    },
  },
  {
    name: "github_get_pr_mergeability",
    description: "Check whether a pull request can be merged (mergeable state, conflicts). GitHub computes this async, so this retries briefly if the result isn't ready yet.",
    parameters: { type: "object", properties: { owner: { type: "string" }, repo: { type: "string" }, pull_number: { type: "number" } }, required: ["repo", "pull_number"] },
    execute: async ({ owner = DEFAULT_OWNER, repo, pull_number }) => {
      let pr;
      for (let i = 0; i < 3; i++) {
        pr = await githubRequest(`/repos/${owner}/${repo}/pulls/${pull_number}`);
        if (pr.mergeable !== null) break;
        await new Promise(r => setTimeout(r, 1000));
      }
      return `mergeable: ${pr.mergeable}\nmergeable_state: ${pr.mergeable_state}\nrebaseable: ${pr.rebaseable}`;
    },
  },

  // -- GitHub: CI / checks -----------------------------------------------
  {
    name: "github_get_check_runs",
    description: "Get CI check-run results (pass/fail dots) for a commit or ref.",
    parameters: { type: "object", properties: { owner: { type: "string" }, repo: { type: "string" }, ref: { type: "string" }, per_page: { type: "number" } }, required: ["repo", "ref"] },
    execute: async ({ owner = DEFAULT_OWNER, repo, ref, per_page = 50 }) => {
      const data = await githubRequest(`/repos/${owner}/${repo}/commits/${ref}/check-runs?per_page=${Math.min(per_page, 100)}`);
      return `${data.total_count} check run(s):\n` + data.check_runs.map(c => `${c.name}: ${c.status}/${c.conclusion}`).join("\n");
    },
  },
  {
    name: "github_get_combined_status",
    description: "Get the combined commit status (overall pass/fail/pending rollup) for a ref.",
    parameters: { type: "object", properties: { owner: { type: "string" }, repo: { type: "string" }, ref: { type: "string" } }, required: ["repo", "ref"] },
    execute: async ({ owner = DEFAULT_OWNER, repo, ref }) => {
      const data = await githubRequest(`/repos/${owner}/${repo}/commits/${ref}/status`);
      return `Overall state: ${data.state} (${data.total_count} statuses)\n` + (data.statuses || []).map(s => `${s.context}: ${s.state} -- ${s.description || ""}`).join("\n");
    },
  },
  {
    name: "github_list_workflow_runs",
    description: "List recent GitHub Actions workflow runs for a repo, optionally scoped to one workflow.",
    parameters: { type: "object", properties: {
      owner: { type: "string" }, repo: { type: "string" }, workflow_id: { type: "string" }, branch: { type: "string" }, status: { type: "string" }, per_page: { type: "number" },
    }, required: ["repo"] },
    execute: async ({ owner = DEFAULT_OWNER, repo, workflow_id, branch, status, per_page = 20 }) => {
      const qs = new URLSearchParams({ per_page: String(Math.min(per_page, 100)) });
      if (branch) qs.set("branch", branch);
      if (status) qs.set("status", status);
      const path = workflow_id ? `/repos/${owner}/${repo}/actions/workflows/${encodeURIComponent(workflow_id)}/runs?${qs}` : `/repos/${owner}/${repo}/actions/runs?${qs}`;
      const data = await githubRequest(path);
      return data.workflow_runs.map(r => `#${r.run_number} [${r.status}/${r.conclusion}] ${r.name} on ${r.head_branch} (${r.created_at?.slice(0, 10)}) -- run_id ${r.id}`).join("\n") || "No runs found.";
    },
  },
  {
    name: "github_get_workflow_run_logs",
    description: "Get a summary of a workflow run's jobs and steps (status/conclusion per step). For raw log text, use github_get_job_logs with a job_id from this result.",
    parameters: { type: "object", properties: { owner: { type: "string" }, repo: { type: "string" }, run_id: { type: "number" } }, required: ["repo", "run_id"] },
    execute: async ({ owner = DEFAULT_OWNER, repo, run_id }) => {
      const [run, jobsData] = await Promise.all([
        githubRequest(`/repos/${owner}/${repo}/actions/runs/${run_id}`),
        githubRequest(`/repos/${owner}/${repo}/actions/runs/${run_id}/jobs`),
      ]);
      let text = `Run #${run.run_number} [${run.status}/${run.conclusion}] ${run.name} on ${run.head_branch}\n\n`;
      text += jobsData.jobs.map(j => `Job ${j.id} "${j.name}": ${j.status}/${j.conclusion}\n` + (j.steps || []).map(s => `  - ${s.name}: ${s.status}/${s.conclusion}`).join("\n")).join("\n\n");
      return text.length > 20000 ? text.slice(0, 20000) + "\n...[truncated]" : text;
    },
  },
  {
    name: "github_get_job_logs",
    description: "Get raw log text for a specific workflow job (find the job_id via github_get_workflow_run_logs first, or pass job_name to look it up).",
    parameters: { type: "object", properties: {
      owner: { type: "string" }, repo: { type: "string" }, run_id: { type: "number" }, job_id: { type: "number" }, job_name: { type: "string" },
    }, required: ["repo", "run_id"] },
    execute: async ({ owner = DEFAULT_OWNER, repo, run_id, job_id, job_name }) => {
      let id = job_id;
      if (!id) {
        const jobsData = await githubRequest(`/repos/${owner}/${repo}/actions/runs/${run_id}/jobs`);
        const match = job_name ? jobsData.jobs.find(j => j.name === job_name) : jobsData.jobs[0];
        if (!match) return `No job found${job_name ? ` matching "${job_name}"` : ""}.`;
        id = match.id;
      }
      const logs = await githubRequest(`/repos/${owner}/${repo}/actions/jobs/${id}/logs`, { accept: "application/vnd.github+json" });
      const text = typeof logs === "string" ? logs : JSON.stringify(logs);
      return text.length > 25000 ? "...[truncated, showing tail]...\n" + text.slice(-25000) : text;
    },
  },

  // -- GitHub: repo metadata / discovery ----------------------------------
  {
    name: "github_list_issues",
    description: "List issues in a repo (excludes pull requests), optionally filtered by state/labels/assignee.",
    parameters: { type: "object", properties: {
      owner: { type: "string" }, repo: { type: "string" }, state: { type: "string" }, labels: { type: "string" }, assignee: { type: "string" }, per_page: { type: "number" },
    }, required: ["repo"] },
    execute: async ({ owner = DEFAULT_OWNER, repo, state = "open", labels, assignee, per_page = 20 }) => {
      const qs = new URLSearchParams({ state, per_page: String(Math.min(per_page, 100)) });
      if (labels) qs.set("labels", labels);
      if (assignee) qs.set("assignee", assignee);
      const data = await githubRequest(`/repos/${owner}/${repo}/issues?${qs}`);
      const issues = data.filter(i => !i.pull_request);
      return issues.map(i => `#${i.number} [${i.state}] ${i.title}`).join("\n") || "No issues found.";
    },
  },
  {
    name: "github_list_releases",
    description: "List releases in a repo.",
    parameters: { type: "object", properties: { owner: { type: "string" }, repo: { type: "string" }, per_page: { type: "number" } }, required: ["repo"] },
    execute: async ({ owner = DEFAULT_OWNER, repo, per_page = 10 }) => {
      const data = await githubRequest(`/repos/${owner}/${repo}/releases?per_page=${Math.min(per_page, 100)}`);
      return data.map(r => `${r.tag_name}${r.name ? ` (${r.name})` : ""} -- ${r.prerelease ? "prerelease" : r.draft ? "draft" : "release"}, published ${r.published_at?.slice(0, 10) || "n/a"}`).join("\n") || "No releases found.";
    },
  },
  {
    name: "github_list_tags",
    description: "List tags in a repo.",
    parameters: { type: "object", properties: { owner: { type: "string" }, repo: { type: "string" }, per_page: { type: "number" } }, required: ["repo"] },
    execute: async ({ owner = DEFAULT_OWNER, repo, per_page = 20 }) => {
      const data = await githubRequest(`/repos/${owner}/${repo}/tags?per_page=${Math.min(per_page, 100)}`);
      return data.map(t => `${t.name} -- ${t.commit?.sha?.slice(0, 7)}`).join("\n") || "No tags found.";
    },
  },
  {
    name: "github_list_contributors",
    description: "List contributors to a repo with commit counts.",
    parameters: { type: "object", properties: { owner: { type: "string" }, repo: { type: "string" }, per_page: { type: "number" } }, required: ["repo"] },
    execute: async ({ owner = DEFAULT_OWNER, repo, per_page = 20 }) => {
      const data = await githubRequest(`/repos/${owner}/${repo}/contributors?per_page=${Math.min(per_page, 100)}`);
      return data.map(c => `${c.login}: ${c.contributions} commits`).join("\n") || "No contributors found.";
    },
  },
  {
    name: "github_get_repo",
    description: "Get repo metadata: description, default branch, language, stars, topics, etc.",
    parameters: { type: "object", properties: { owner: { type: "string" }, repo: { type: "string" } }, required: ["repo"] },
    execute: async ({ owner = DEFAULT_OWNER, repo }) => {
      const r = await githubRequest(`/repos/${owner}/${repo}`);
      return `${r.full_name} (${r.visibility})\n${r.description || "(no description)"}\nDefault branch: ${r.default_branch} | Language: ${r.language} | Stars: ${r.stargazers_count} | Forks: ${r.forks_count} | Open issues: ${r.open_issues_count}\nTopics: ${(r.topics || []).join(", ") || "none"}\nURL: ${r.html_url}`;
    },
  },
  {
    name: "github_get_branch_protection",
    description: "Get branch protection rules for a branch (required checks, required reviews, etc.). Returns a note if the branch is unprotected or the caller lacks access.",
    parameters: { type: "object", properties: { owner: { type: "string" }, repo: { type: "string" }, branch: { type: "string" } }, required: ["repo", "branch"] },
    execute: async ({ owner = DEFAULT_OWNER, repo, branch }) => {
      try {
        const data = await githubRequest(`/repos/${owner}/${repo}/branches/${branch}/protection`);
        return JSON.stringify(data, null, 2).slice(0, 8000);
      } catch (err) {
        return `No accessible branch protection for "${branch}": ${err.message}`;
      }
    },
  },
  {
    name: "github_list_branches",
    description: "List branches in a repo.",
    parameters: { type: "object", properties: { owner: { type: "string" }, repo: { type: "string" } }, required: ["repo"] },
    execute: async ({ owner = DEFAULT_OWNER, repo }) => {
      const data = await githubRequest(`/repos/${owner}/${repo}/branches`);
      return data.map(b => `${b.name}${b.protected ? " (protected)" : ""}`).join("\n") || "No branches found.";
    },
  },
  {
    name: "github_get_repo_topics",
    description: "Get the topics/tags set on a repo.",
    parameters: { type: "object", properties: { owner: { type: "string" }, repo: { type: "string" } }, required: ["repo"] },
    execute: async ({ owner = DEFAULT_OWNER, repo }) => {
      const data = await githubRequest(`/repos/${owner}/${repo}/topics`, { accept: "application/vnd.github.mercy-preview+json" });
      return (data.names || []).join(", ") || "No topics set.";
    },
  },
  {
    name: "github_list_directory",
    description: "List files and folders at a specific path in a repo (non-recursive; use github_get_file_tree for the full recursive tree).",
    parameters: { type: "object", properties: { owner: { type: "string" }, repo: { type: "string" }, path: { type: "string" }, ref: { type: "string" } }, required: ["repo"] },
    execute: async ({ owner = DEFAULT_OWNER, repo, path = "", ref }) => {
      const qs = ref ? `?ref=${encodeURIComponent(ref)}` : "";
      const data = await githubRequest(`/repos/${owner}/${repo}/contents/${path}${qs}`);
      const entries = Array.isArray(data) ? data : [data];
      return entries.map(e => `${e.type === "dir" ? "dir " : "file"} ${e.path}`).join("\n") || "(empty)";
    },
  },

  // -- GitHub: commits / diffs / code search ------------------------------
  {
    name: "github_get_commit",
    description: "Get a commit's message, author, and changed files.",
    parameters: { type: "object", properties: { owner: { type: "string" }, repo: { type: "string" }, sha: { type: "string" } }, required: ["repo", "sha"] },
    execute: async ({ owner = DEFAULT_OWNER, repo, sha }) => {
      const c = await githubRequest(`/repos/${owner}/${repo}/commits/${sha}`);
      const files = (c.files || []).map(f => `  ${f.status} ${f.filename} (+${f.additions}/-${f.deletions})`).join("\n");
      return `${c.sha.slice(0, 7)} by ${c.commit.author?.name} on ${c.commit.author?.date?.slice(0, 10)}\n${c.commit.message}\n\nFiles changed:\n${files || "(none)"}`;
    },
  },
  {
    name: "github_get_file_at_commit",
    description: "Read a file's contents as it existed at a specific commit SHA. ALWAYS call with no char_offset/char_limit first -- returns the whole file up to 30,000 chars in one call, or the first 30,000 plus total length and the offset to continue if longer. Only pass char_offset/char_limit once you already know you need a specific window (this function's own prior truncation notice on this same file/commit, or a search hit pointing at a known-large file) -- do not guess an offset on a file whose real size you haven't confirmed.",
    parameters: { type: "object", properties: {
      owner: { type: "string" }, repo: { type: "string" }, path: { type: "string" }, commit: { type: "string" },
      char_offset: { type: "number", description: "Character offset to start reading from. Omit for default behavior." },
      char_limit: { type: "number", description: "Maximum characters to return (default 30000, max 100000). Ignored if char_offset is also omitted." },
    }, required: ["repo", "path", "commit"] },
    execute: async ({ owner = DEFAULT_OWNER, repo, path, commit, char_offset, char_limit }) => {
      const content = await readFileViaBlob(owner, repo, path, commit);
      return sliceFileContentForModel(content, path, { char_offset, char_limit });
    },
  },
  {
    name: "github_diff_files",
    description: "Compare the same file (or two different files) between two refs/branches/commits and return a line-based diff. If files exceed 2000 lines, pass start_line and end_line to diff a specific slice.",
    parameters: { type: "object", properties: {
      owner: { type: "string" }, repo: { type: "string" }, path: { type: "string", description: "File path (used for both sides unless base_path/head_path given)" },
      base_ref: { type: "string" }, head_ref: { type: "string" }, base_path: { type: "string" }, head_path: { type: "string" },
      start_line: { type: "number", description: "Optional 1-indexed start line for ranged diffs when files are large" },
      end_line: { type: "number", description: "Optional 1-indexed end line for ranged diffs when files are large" },
    }, required: ["repo", "path", "base_ref", "head_ref"] },
    execute: async ({ owner = DEFAULT_OWNER, repo, path, base_ref, head_ref, base_path, head_path, start_line, end_line }) => {
      const [a, b] = await Promise.all([
        readFileViaBlob(owner, repo, base_path || path, base_ref),
        readFileViaBlob(owner, repo, head_path || path, head_ref),
      ]);
      const diff = simpleLineDiff(a, b, { start_line, end_line });
      return diff.length > 20000 ? diff.slice(0, 20000) + "\n...[truncated]" : diff;
    },
  },
  {
    name: "github_search_code",
    description: "Search code across GitHub using GitHub's code-search syntax (e.g. 'foo repo:owner/name', 'extension:js useState'). If the query is scoped to a single repo via repo:owner/name and GitHub's index returns nothing (a known gap for private repos), or if `ref` is given (GitHub's index only ever covers the default branch), this automatically falls back to fetching that repo as a tarball and grepping it locally instead of just reporting no results.",
    parameters: { type: "object", properties: {
      query: { type: "string", description: "Search query, e.g. 'foo repo:owner/name' or 'extension:js useState'" },
      per_page: { type: "number", description: "Number of results to return, max 100 (default 20)" },
      ref: { type: "string", description: "Branch, tag, or commit SHA to search instead of the default branch. Requires a repo:owner/name qualifier in the query -- GitHub's search index only covers the default branch, so this always uses the local content-search fallback rather than the real API." },
    }, required: ["query"] },
    execute: async ({ query, per_page = 20, ref }) => {
      const scoped = extractRepoQualifier(query);

      if (ref) {
        if (!scoped) {
          return "Error: `ref` requires a repo:owner/name qualifier in the query -- GitHub's search index only covers the default branch, so a specific repo must be named for the branch-aware fallback to know what to fetch.";
        }
        let fb;
        try {
          fb = await fallbackCodeSearch({ ...scoped, query, per_page, ref });
        } catch (err) {
          return `Branch search failed: ${err?.message ?? String(err)}`;
        }
        if (fb?.matches.length) {
          const lines = fb.matches.map((m) => `${scoped.owner}/${scoped.repo}/${m.path}:${m.line} -- ${m.snippet}`);
          return `Searched ${scoped.owner}/${scoped.repo}@${ref} directly (GitHub's code-search index only covers the default branch) -- scanned ${fb.scanned} file(s)${fb.truncated ? ", capped -- repo has more" : ""}:\n${lines.join("\n")}`;
        }
        return `No results found on ${scoped.owner}/${scoped.repo}@${ref} (scanned ${fb?.scanned ?? 0} file(s)${fb?.truncated ? ", capped -- repo has more" : ""}).`;
      }

      const data = await githubRequest(`/search/code?q=${encodeURIComponent(query)}&per_page=${Math.min(per_page, 100)}`);
      if (data.items?.length) {
        const text = `Found ${data.total_count} total, showing ${data.items.length}:\n` + data.items.map(i => `${i.repository.full_name}: ${i.path}`).join("\n");
        return text.length > 15000 ? text.slice(0, 15000) + "\n...[truncated]" : text;
      }

      if (scoped) {
        const fb = await fallbackCodeSearch({ ...scoped, query, per_page }).catch(() => null);
        if (fb?.matches.length) {
          const lines = fb.matches.map((m) => `${scoped.owner}/${scoped.repo}/${m.path}:${m.line} -- ${m.snippet}`);
          return `GitHub's code-search index returned nothing for this repo (a known gap for private repos), so this used a direct content search instead (scanned ${fb.scanned} file(s)${fb.truncated ? ", capped -- repo has more" : ""}):\n${lines.join("\n")}`;
        }
        if (fb) {
          return `No results found. Also tried a direct content search of ${scoped.owner}/${scoped.repo} (GitHub's search index can return empty for private repos regardless of permissions) -- scanned ${fb.scanned} file(s)${fb.truncated ? " (capped, repo has more)" : ""}, no match.`;
        }
      }

      return "No results found.";
    },
  },

  // -- Cloudflare: Workers / D1 / KV / R2 / Hyperdrive ---------------------
  {
    name: "cf_workers_list",
    description: "List all Cloudflare Workers scripts in the account.",
    parameters: { type: "object", properties: {} },
    execute: async () => {
      const data = await cfAccountRequest("/workers/scripts");
      return (data || []).map(w => `${w.id} (modified ${w.modified_on?.slice(0, 10)})`).join("\n") || "No workers found.";
    },
  },
  {
    name: "cf_workers_get_worker",
    description: "Get settings/metadata for a single Cloudflare Worker.",
    parameters: { type: "object", properties: { scriptName: { type: "string" } }, required: ["scriptName"] },
    execute: async ({ scriptName }) => JSON.stringify(await cfAccountRequest(`/workers/scripts/${scriptName}/settings`), null, 2).slice(0, 8000),
  },
  {
    name: "cf_workers_get_worker_code",
    description: "Get the source code of a Cloudflare Worker.",
    parameters: { type: "object", properties: { scriptName: { type: "string" } }, required: ["scriptName"] },
    execute: async ({ scriptName }) => {
      const data = await cfAccountRequest(`/workers/scripts/${scriptName}`);
      const text = typeof data === "string" ? data : JSON.stringify(data);
      return text.length > 30000 ? text.slice(0, 30000) + "\n...[truncated]" : text;
    },
  },
  {
    name: "cf_d1_databases_list",
    description: "List D1 databases in the account.",
    parameters: { type: "object", properties: { name: { type: "string" } } },
    execute: async ({ name }) => {
      const qs = name ? `?name=${encodeURIComponent(name)}` : "";
      const data = await cfAccountRequest(`/d1/database${qs}`);
      return (data || []).map(d => `${d.name} -- ${d.uuid}`).join("\n") || "No databases found.";
    },
  },
  {
    name: "cf_d1_database_get",
    description: "Get details for a single D1 database.",
    parameters: { type: "object", properties: { database_id: { type: "string" } }, required: ["database_id"] },
    execute: async ({ database_id }) => JSON.stringify(await cfAccountRequest(`/d1/database/${database_id}`), null, 2).slice(0, 5000),
  },
  {
    name: "cf_kv_namespaces_list",
    description: "List KV namespaces in the account.",
    parameters: { type: "object", properties: {} },
    execute: async () => {
      const data = await cfAccountRequest("/storage/kv/namespaces");
      return (data || []).map(n => `${n.title} -- ${n.id}`).join("\n") || "No namespaces found.";
    },
  },
  {
    name: "cf_kv_namespace_get",
    description: "Get details for a single KV namespace.",
    parameters: { type: "object", properties: { namespace_id: { type: "string" } }, required: ["namespace_id"] },
    execute: async ({ namespace_id }) => JSON.stringify(await cfAccountRequest(`/storage/kv/namespaces/${namespace_id}`), null, 2).slice(0, 5000),
  },
  {
    name: "cf_r2_buckets_list",
    description: "List R2 buckets in the account.",
    parameters: { type: "object", properties: {} },
    execute: async () => {
      const data = await cfAccountRequest("/r2/buckets");
      return (data?.buckets || data || []).map(b => `${b.name} (created ${b.creation_date?.slice(0, 10) || "n/a"})`).join("\n") || "No buckets found.";
    },
  },
  {
    name: "cf_r2_bucket_get",
    description: "Get details for a single R2 bucket.",
    parameters: { type: "object", properties: { name: { type: "string" } }, required: ["name"] },
    execute: async ({ name }) => JSON.stringify(await cfAccountRequest(`/r2/buckets/${name}`), null, 2).slice(0, 5000),
  },
  {
    name: "cf_hyperdrive_configs_list",
    description: "List Hyperdrive configurations in the account.",
    parameters: { type: "object", properties: {} },
    execute: async () => {
      const data = await cfAccountRequest("/hyperdrive/configs");
      return (data || []).map(h => `${h.name} -- ${h.id}`).join("\n") || "No Hyperdrive configs found.";
    },
  },
  {
    name: "cf_hyperdrive_config_get",
    description: "Get details for a single Hyperdrive configuration.",
    parameters: { type: "object", properties: { hyperdrive_id: { type: "string" } }, required: ["hyperdrive_id"] },
    execute: async ({ hyperdrive_id }) => JSON.stringify(await cfAccountRequest(`/hyperdrive/configs/${hyperdrive_id}`), null, 2).slice(0, 5000),
  },
  {
    name: "cf_workers_observability_keys",
    description: "List available telemetry keys (log/trace/event fields) for a time range.",
    parameters: { type: "object", properties: { timeframe_from: { type: "string" }, timeframe_to: { type: "string" }, dataset: { type: "string" } }, required: ["timeframe_from", "timeframe_to"] },
    execute: async ({ timeframe_from, timeframe_to, dataset = "cloudflare-workers" }) => {
      const data = await cfAccountRequest("/workers/observability/telemetry/keys", { method: "POST", body: { dataset, timeframe: { from: toEpochMillis(timeframe_from), to: toEpochMillis(timeframe_to) } } });
      return JSON.stringify(data).slice(0, 8000);
    },
  },
  {
    name: "cf_workers_observability_values",
    description: "List the distinct values seen for a given telemetry key over a time range.",
    parameters: { type: "object", properties: {
      key: { type: "string" }, timeframe_from: { type: "string" }, timeframe_to: { type: "string" }, dataset: { type: "string" }, valueType: { type: "string", description: "string, boolean, or number (default string)" },
    }, required: ["key", "timeframe_from", "timeframe_to"] },
    execute: async ({ key, timeframe_from, timeframe_to, dataset = "cloudflare-workers", valueType = "string" }) => {
      const data = await cfAccountRequest("/workers/observability/telemetry/values", { method: "POST", body: { datasets: [dataset], key, type: valueType, timeframe: { from: toEpochMillis(timeframe_from), to: toEpochMillis(timeframe_to) } } });
      return JSON.stringify(data).slice(0, 8000);
    },
  },

  // -- Context7 -----------------------------------------------------------
  {
    name: "context7_search_library",
    description: "Search Context7's index for a library/framework by name to get its library ID.",
    parameters: { type: "object", properties: { libraryName: { type: "string" }, query: { type: "string" } }, required: ["libraryName", "query"] },
    execute: async ({ libraryName, query }) => {
      const data = await context7Request("/libs/search", { libraryName, query });
      return (data.results || []).map(r => `${r.id} -- ${r.title} (trust ${r.trustScore})`).join("\n") || "No libraries found.";
    },
  },
  {
    name: "context7_get_library_docs",
    description: "Fetch version-specific documentation and code examples for a library by its Context7 library ID (from context7_search_library).",
    parameters: { type: "object", properties: { libraryId: { type: "string" }, query: { type: "string" }, tokens: { type: "number" } }, required: ["libraryId", "query"] },
    execute: async ({ libraryId, query, tokens }) => {
      const data = await context7Request("/context", { libraryId, query, tokens });
      const text = typeof data === "string" ? data : (data.context || data.text || JSON.stringify(data));
      return text.length > 25000 ? text.slice(0, 25000) + "\n...[truncated]" : text;
    },
  },

  // -- Mem0 -----------------------------------------------------------------
  {
    name: "mem0_search",
    description: "Search memories in the Mem0 workspace using hybrid semantic + keyword retrieval.",
    parameters: { type: "object", properties: { query: { type: "string" }, limit: { type: "number" } }, required: ["query"] },
    execute: async ({ query, limit = 10 }) => {
      const data = await mem0Request("/v3/memories/search/", { method: "POST", body: { query, limit } });
      const results = data.results || data || [];
      return results.map(m => `[${m.score?.toFixed?.(2) ?? "?"}] ${m.memory || m.content}`).join("\n---\n") || "No memories found.";
    },
  },
  {
    name: "mem0_list",
    description: "List recent memories from the Mem0 workspace.",
    parameters: { type: "object", properties: { page_size: { type: "number" } } },
    execute: async ({ page_size = 20 }) => {
      const data = await mem0Request("/v3/memories/", { method: "POST", body: { page_size } });
      const results = data.results || data.memories || data || [];
      return results.map(m => `${m.id}: ${m.memory || m.content}`).join("\n") || "No memories found.";
    },
  },
  {
    name: "mem0_get",
    description: "Get the full content of a specific Mem0 memory by ID.",
    parameters: { type: "object", properties: { id: { type: "string" } }, required: ["id"] },
    execute: async ({ id }) => {
      const m = await mem0Request(`/v1/memories/${id}/`);
      return `${m.memory}\ncreated: ${m.created_at} | updated: ${m.updated_at}\nmetadata: ${JSON.stringify(m.metadata || {})}`;
    },
  },
  {
    name: "mem0_get_history",
    description: "Get the version/audit history of a Mem0 memory by ID.",
    parameters: { type: "object", properties: { id: { type: "string" } }, required: ["id"] },
    execute: async ({ id }) => {
      const data = await mem0Request(`/v1/memories/${id}/history/`);
      return (data || []).map(h => `${h.event} @ ${h.timestamp}: ${h.old_memory || ""} -> ${h.new_memory || ""}`).join("\n") || "No history found.";
    },
  },

  // -- Notion ---------------------------------------------------------------
  {
    name: "notion_get_database",
    description: "Get a Notion database's schema (title and property definitions) and basic info. Use this before notion_query_database to see what properties are available and their types.",
    parameters: { type: "object", properties: { database_id: { type: "string" } }, required: ["database_id"] },
    execute: async ({ database_id }) => {
      const data = await notionRequest(`/databases/${database_id}`);
      const title = notionDatabaseTitle(data);
      const propLines = Object.entries(data.properties || {}).map(([name, def]) => `  ${name}: ${def.type}`);
      return `# ${title}\nID: ${data.id}\nURL: ${data.url}\nCreated: ${data.created_time?.slice(0, 10)} | Last edited: ${data.last_edited_time?.slice(0, 10)}\n\nProperties:\n${propLines.join("\n") || "(none)"}`;
    },
  },
  {
    name: "notion_list",
    description: "List recent pages and/or databases in the Notion workspace, sorted by most recently edited first -- no search query needed. Use this (not notion_search) when the task is 'find the latest X' or 'what's changed recently in Notion' -- notion_search requires a keyword and doesn't guarantee recency ordering.",
    parameters: { type: "object", properties: {
      filter_type: { type: "string", description: "Restrict to 'page' or 'database' (optional, default both)" },
      page_size:   { type: "number", description: "Number of results (default 10, max 100)" },
    } },
    execute: async ({ filter_type, page_size = 10 }) => {
      const body = { query: "", sort: { direction: "descending", timestamp: "last_edited_time" }, page_size };
      if (filter_type) body.filter = { value: filter_type, property: "object" };
      const data = await notionRequest("/search", { method: "POST", body });
      if (!data.results?.length) return "No pages or databases found.";
      return data.results.map(r => {
        const title = r.object === "page" ? notionPageTitle(r) : (notionRichTextToString(r.title) || "(untitled)");
        return `[${r.object}] ${title} — id: ${r.id} — last edited ${r.last_edited_time?.slice(0, 16)}`;
      }).join("\n");
    },
  },
  {
    name: "notion_get_page_history",
    description: "Get the changelog/version history entries recorded on a Notion page (read-only; looks for logged changelog blocks, not Notion's native edit history).",
    parameters: { type: "object", properties: { page_id: { type: "string" } }, required: ["page_id"] },
    execute: async ({ page_id }) => {
      const data = await notionRequest(`/blocks/${page_id}/children?page_size=100`);
      const text = notionBlocksToText(data.results || []) || "(no content)";
      return text.length > 10000 ? text.slice(0, 10000) + "\n...[truncated]" : text;
    },
  },
];

const FUNCTION_DECLARATIONS = [{
  functionDeclarations: FUNCTIONS.map(({ name, description, parameters }) => ({ name, description, parameters })),
}];

// Tool-call-leakage backstop (bai-only, plan.md Section 18/20 fix): on a
// withholdTools turn (final step, stuck-loop force, or the verification
// pass -- all cases where NO real function call is structurally possible),
// the model can still emit ordinary text that IMITATES a function call
// instead of giving a real plain-text answer. Observed live twice on bai's
// forced-final step: run 6ea018d5 leaked XML-tag-shaped syntax
// (`<github_read_file><params>...`), run ab8afaa8 leaked bracket-marker-
// shaped syntax (`[Function call: github_read_file with ...]`). The
// existing `finishReason === "MALFORMED_FUNCTION_CALL"` check (see the loop
// below) does NOT catch this -- it only fires when the model returns no
// text at all; both leaks above had non-empty `answer` text, so they took
// the ordinary `if (answer) return finishRun(...)` path untouched.
//
// This is a mechanical, structural check (real FUNCTIONS[] names wrapped in
// call-shaped syntax), not a generic "contains markup" heuristic -- a
// legitimate answer that happens to show an XML example or a JSON snippet
// unrelated to this file's own tool names should never trip it. Only
// matches call-shaped wrappers (an XML tag, a "[Function call: ..."
// marker, or a `"name": "..."` JSON shape) around a name that is ACTUALLY
// one of this file's declared functions.
//
// WHITESPACE NORMALIZATION (fix, plan.md Section 21): the ORIGINAL version
// of these patterns captured `[\w-]*` only -- contiguous identifier chars,
// no spaces -- which looks right for a normal tag/call name. But the actual
// literal text captured in Section 18's run 6ea018d5 was
// `<githu b_read_file>`: the model didn't just wrap a real name in tag
// syntax, it garbled the name itself with an embedded space. Under the old
// pattern that only captures "githu" (stops at the space), which is not a
// real function name, so the backstop silently let that exact case through
// -- the one case it was written specifically to catch. Fixed by allowing
// internal whitespace inside the captured candidate (`[\w\s-]*?`, matched
// non-greedily up to the syntax's own closing token) and then stripping
// ALL whitespace from the captured text before comparing against
// knownFunctionNames, so `<githu b_read_file>` and `<github_read_file>`
// normalize identically. This does not increase the false-positive surface
// -- normalization only feeds into the knownFunctionNames.has(...) check,
// so it still only fires on an ACTUAL real tool name (mangled or not), not
// on arbitrary whitespace-containing text.
const TOOL_CALL_LEAKAGE_PATTERNS = [
  /<\/?\s*([a-zA-Z_][\w\s-]*?)\s*\/?>/g,                                          // XML-tag-shaped: <github_read_file>, </params>, or a space-mangled <githu b_read_file>
  /\[\s*function\s*call\s*:?\s*([a-zA-Z_][\w\s-]*?)(?=\s+with\b|\s*\]|,)/gi,       // bracket-shaped: [Function call: github_read_file ...] (also tolerates a mangled name before "with"/"]"/",")
  /"(?:name|function)"\s*:\s*"([a-zA-Z_][\w\s-]*?)"/gi,                           // JSON-shaped: {"name": "github_read_file", ...}
];

export function detectToolCallLeakage(text, knownFunctionNames) {
  for (const pattern of TOOL_CALL_LEAKAGE_PATTERNS) {
    pattern.lastIndex = 0;
    let m;
    while ((m = pattern.exec(text))) {
      const normalized = m[1].replace(/\s+/g, "");
      if (knownFunctionNames.has(normalized)) return normalized;
    }
  }
  return null;
}

// The actual set of real function names this file declares -- built once
// from FUNCTIONS[] itself (not hand-maintained separately), so it can never
// drift out of sync with the real tool list as FUNCTIONS[] grows/changes.
const FUNCTION_NAME_SET = new Set(FUNCTIONS.map((f) => f.name));

// SCOPE NOTE (2026-07-27): this file deliberately has NO web access (no
// web_fetch, no Google Search grounding) -- that lives entirely in
// connectors/exa/research_delegate.js, behind the separate delegate_research
// tool. Keeping the two apart is a security boundary, not just a UX split:
// this loop reads private GitHub/Notion/Cloudflare/Context7/Mem0 data, and
// research_delegate.js's Exa call reads untrusted public web content -- a single loop
// with both would let a malicious page or search result Gemini encounters
// mid-investigation try to talk the model into leaking whatever it just
// read from those private systems (e.g. via a crafted outbound fetch to an
// attacker-controlled URL). Neither loop can do that, because neither ever
// has both capabilities available at once. Do NOT re-add web_fetch or a
// google_search tool here -- add web capability to research_delegate.js instead.

const SYSTEM_PREAMBLE =
  "You are a read-only investigation agent. Use the available functions to gather whatever " +
  "information you need to answer the task fully, calling as many as necessary across multiple " +
  "turns. When you have enough information, respond with a final plain-text answer and no further " +
  "function calls. Be specific and cite what you found (file paths, commit SHAs, log entries, page " +
  "titles) rather than speculating.\n\n" +
  "IMPORTANT -- no visible reasoning outside the answer itself: do not narrate your reasoning, plans, " +
  "or step-by-step thinking in the visible text of any turn -- neither alongside a function call (just " +
  "make the call, with no accompanying explanation of why) nor as a preamble before your final answer " +
  "(lead with the answer itself, not a walkthrough of how you got there). Every turn's visible text is " +
  "resent in full on every subsequent step for the rest of this investigation, so a reasoning preamble " +
  "repeated turn after turn compounds into wasted tokens without adding anything a caller reading only " +
  "the final answer needs. Think through the problem as needed, but only write down the function call " +
  "itself or the concluding answer -- not the thinking that produced it.\n\n" +
  "IMPORTANT -- cross-check, don't just aggregate: when the task touches more than one source " +
  "(e.g. a GitHub PR's status vs. a Notion tracking page, or a repo file vs. what a database row " +
  "claims), actively look for contradictions between them rather than reporting each source's claim " +
  "in isolation. A thing that LOOKS current, open, or resolved in one source can be stale or wrong " +
  "according to another -- if your task plan touches multiple sources for related claims, check them " +
  "against each other before answering, and call out any discrepancy explicitly (including which " +
  "source you consider more authoritative and why) rather than picking one silently.\n\n" +
  "IMPORTANT -- respect scope, don't let same-named symbols bleed across files: when a question is " +
  "about whether something is used, referenced, or defined WITHIN A SPECIFIC FILE OR SCOPE (e.g. an " +
  "unused-import lint warning, which is always per-file), only evidence found in THAT exact file or " +
  "scope counts. A same-named function/variable being called somewhere else in the repo -- even in a " +
  "file that imports it from the same source module -- does NOT mean it's used in the file the question " +
  "is actually about; each file's own import/declaration is independent. Before calling a usage claim a " +
  "'false positive' or asserting something IS used, quote the exact call site (file + line/snippet) " +
  "inside the specific scope in question. If you can't produce that quote from within the scope asked " +
  "about, say plainly that no such usage was found there, rather than pointing to usage elsewhere as if " +
  "it answered the question.\n\n" +
  "IMPORTANT -- re-scan your OWN retrieved text before writing a verdict word (consistent, fixed, " +
  "resolved, stale, up-to-date, matches, etc.): a long tool-use run compresses many turns of raw " +
  "file/page content into one final summary, and that compression step is itself a separate inference " +
  "that can pattern-match toward a comfortable verdict even when the contradicting text is sitting " +
  "unused in your own transcript. If your task asks you to check whether something is stale, " +
  "inconsistent, or still-accurate, before writing the verdict go back through EVERY piece of raw " +
  "content you fetched (not just the ones that confirm your leaning) and check it against the specific " +
  "claim in the question -- do not let a majority of confirming sources outvote a single contradicting " +
  "one you already retrieved. If you find a contradiction this way, quote it and flag it explicitly " +
  "even if most of what you found points the other way.\n\n" +
  "IMPORTANT -- a full/direct read outranks a narrower or derived result for the SAME fact: when a " +
  "complete, direct read of a file or page (github_read_file, github_get_file_at_commit, notion_get_page, " +
  "etc.) and a narrower or derived result about the same thing (a github_search_code snippet, a grep hit, " +
  "a mem0_search match) disagree, trust the full/direct read -- it is the more authoritative source, even " +
  "if the narrower result was fetched more recently in this conversation. A search snippet only shows the " +
  "matching line(s) out of context and can miss surrounding logic (a conditional, a comment, a different " +
  "code path) that changes what the match actually means; a full read does not have that limitation. " +
  "Do not let a later, narrower result override an earlier, complete one just because it came later.";

// bai-specific early reinforcement (plan.md Section 24 follow-up): every note
// version tried on the FINAL-STEP turn alone (elaborated / removed /
// simplified-restored) has produced a different failure shape on bai --
// most recently, the model narrating an intended tool action ("Fetching
// those now...") on a turn where no tools exist at all, despite the
// final-step note saying so explicitly. Hypothesis: by the time that note
// arrives, it's competing against many turns of established tool-calling
// momentum with nothing earlier in the conversation to counter it. This
// adds one short, early mention -- in turn 1, before any such momentum
// exists -- that a tool-less forced final turn is coming, so it isn't a
// surprise sprung only at the moment tools are withdrawn.
//
// Kept deliberately short: per this same investigation, MORE elaboration
// on the final-step note itself has correlated with new failure shapes,
// not fewer (Sections 16/18/20/21), so this errs toward minimal wording
// rather than a fuller explanation. Gemini-only runs have never exhibited
// any of these failure shapes, so this addendum is bai-only -- see
// buildSystemPreamble below -- to avoid changing a prompt surface that
// isn't broken for the other provider.
const BAI_PREAMBLE_ADDENDUM =
  "\n\nNOTE: at some point before you must answer, tool access will be withdrawn for one forced " +
  "final turn. When that happens, answer immediately in plain text -- do not describe what you would " +
  "fetch or do next, since there will be nothing left to fetch.";

// Trimmed variant (A/B test, see the handoff for this work): cuts 3 of the
// 6 SYSTEM_PREAMBLE paragraphs -- cross-check-sources, re-scan-before-
// verdict, and full/direct-read-outranks-narrower -- keeping only the core
// framing, the no-visible-reasoning rule, and the scope-bleed rule. The
// rationale for the cut is that those 3 paragraphs' guidance is arguably
// duplicated by VERIFICATION_PROMPT's forced second pass (below) plus the
// mechanical checks (extractMechanicalClaims/lineIsVerbatimInToolResults),
// which grep raw tool output rather than trusting prose self-policing.
// Open risk: if VERIFICATION_PROMPT doesn't fully substitute for always-on
// synthesis-time framing, or if that pass ever gets skipped (stuck-loop
// force, certain resumes), this variant could have no backstop for what
// those 3 paragraphs were preventing -- exactly what the A/B test below is
// meant to surface with real data instead of guessing.
const SYSTEM_PREAMBLE_TRIMMED =
  "You are a read-only investigation agent. Use the available functions to gather whatever " +
  "information you need to answer the task fully, calling as many as necessary across multiple " +
  "turns. When you have enough information, respond with a final plain-text answer and no further " +
  "function calls. Be specific and cite what you found (file paths, commit SHAs, log entries, page " +
  "titles) rather than speculating.\n\n" +
  "IMPORTANT -- no visible reasoning outside the answer itself: do not narrate your reasoning, plans, " +
  "or step-by-step thinking in the visible text of any turn -- neither alongside a function call nor " +
  "as a preamble before your final answer. Only write down the function call itself or the concluding " +
  "answer.\n\n" +
  "IMPORTANT -- respect scope, don't let same-named symbols bleed across files: when a question is " +
  "about whether something is used, referenced, or defined WITHIN A SPECIFIC FILE OR SCOPE, only " +
  "evidence found in THAT exact file or scope counts. A same-named function/variable used elsewhere " +
  "in the repo does NOT mean it's used in the file the question is about. Before calling a usage " +
  "claim a 'false positive' or asserting something IS used, quote the exact call site inside the " +
  "specific scope in question. If you can't produce that quote, say plainly that no such usage was " +
  "found there.";

// preambleVariant selects which base preamble a run uses -- "verbose"
// (SYSTEM_PREAMBLE, the existing default) or "trimmed"
// (SYSTEM_PREAMBLE_TRIMMED). This is an A/B test knob threaded through
// runInvestigation and, as of this change, also exposed on the MCP-facing
// delegate_agent tool's Zod schema (agent_tools.js) so the calling model
// can run the same task under both variants side-by-side for comparison.
// NOTE: this is a deliberate departure from the original A/B test plan,
// which called for keeping this OUT of the calling model's control
// (human/test-harness-only) specifically to avoid confounding results --
// if the model picks the variant per task, differences in outcome can
// reflect which tasks got which variant rather than the preamble itself.
// Revisit before treating any comparison done this way as a rigorous
// result; tighten back to harness-only once exploratory testing is done.
// Defaults to "verbose" so every existing caller is unaffected.
function buildSystemPreamble(provider, preambleVariant = "verbose") {
  return selectPreambleVariant({
    base: SYSTEM_PREAMBLE,
    trimmed: SYSTEM_PREAMBLE_TRIMMED,
    variant: preambleVariant,
    provider,
    addenda: { bai: BAI_PREAMBLE_ADDENDUM },
  });
}

// Mechanical-claim extraction (2026-08-27, fix for the confident-wrong-
// constant failure mode found in live testing, runs 3-4: the model asserted `HARD_MAX_STEPS` gated a
// condition when the real gate was `cappedSteps`, confidently and with no
// hedge, and the existing text-only VERIFICATION_PROMPT ("if you're not
// certain, re-check") did not catch it because the model WAS certain -- just
// wrong. "Are you sure?" cannot fix miscalibrated confidence; "can you paste
// the exact line?" can. This turns that into a checkable, structural gate
// instead of a request the model can silently skip.
//
// Deliberately a cheap regex pre-filter, not a real parser: it flags two
// shapes of mechanical claim worth double-checking --
//   (a) SCREAMING_SNAKE_CASE identifiers (constants/thresholds -- exactly
//       the shape of the HARD_MAX_STEPS/cappedSteps confusion), and
//       camelCase/PascalCase multi-word identifiers of the kind used for
//       variables/functions in this codebase, and
//   (b) backtick-quoted inline code spans, since the model already uses
//       those for exactly this class of claim in its own answers (see this
//       file's own comments/prompts for the convention).
// False positives (an English word that happens to be ALL_CAPS, a backtick
// span that's genuinely just a value already quoted correctly) are fine --
// the cost of over-flagging is one extra check against text already in the
// transcript, not a wrong answer.
const IDENTIFIER_CLAIM_PATTERN = /\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b|\b[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b|`[^`\n]+`/g;

function extractMechanicalClaims(answerText) {
  const claims = new Set();
  for (const m of answerText.matchAll(IDENTIFIER_CLAIM_PATTERN)) {
    const raw = m[0].startsWith("`") ? m[0].slice(1, -1) : m[0];
    // Skip trivially short/common matches that are noise, not claims worth
    // re-verifying (e.g. a stray 2-letter camelCase-looking fragment).
    if (raw.length >= 4) claims.add(raw);
  }
  return [...claims];
}

// Scans `contents` for functionResponse parts whose text has already been
// replaced by compactHistoryInPlace's marker string, and returns the ids of
// those that `preCompactionResults` (the in-memory Map) doesn't have text
// for -- compaction only ever replaces the TEXT of a functionResponse part,
// never its `id` (see compactHistoryInPlace's hard constraints), so the id
// is always still readable straight off `contents` even once compacted.
// This is the exact "ids needed for this pass" set that the side-store
// fallback batches a single MGET across, instead of fetching speculatively
// or one id at a time.
function findCompactedIdsMissingFromMap(contents, preCompactionResults) {
  const missing = new Set();
  for (const turn of contents) {
    if (turn.role !== "user" || !Array.isArray(turn.parts)) continue;
    for (const part of turn.parts) {
      if (!part.functionResponse) continue;
      const result = part.functionResponse.response?.result;
      if (typeof result === "string" && result.startsWith("[Earlier tool result compacted:")) {
        const id = part.functionResponse.id;
        if (id && !preCompactionResults.has(id)) missing.add(id);
      }
    }
  }
  return [...missing];
}

// Checks each extracted claim against the RAW tool-result text already
// gathered this run (functionResponse parts in `contents`) -- deliberately
// NOT against the model's own prior turns/text, since matching against its
// own earlier assertions would just let a wrong claim "verify" itself by
// citing itself. A claim is "verified" only if it appears verbatim in
// something an actual tool returned.
//
// `runId` (optional -- omitting it preserves the old Map-only behavior,
// e.g. for the hand-built-array unit tests that don't go through Redis)
// enables a side-store fallback for any compacted id `preCompactionResults`
// doesn't have text for -- a real gap the normal compact-then-verify-same-run
// flow doesn't hit, but which a caller reconstructing state from a
// checkpoint's id-only `preCompactionResultIds` without first re-running
// compactHistoryInPlace would. Batches every missing id into one MGET rather
// than fetching per id.
export async function findUnverifiedClaims(claims, contents, preCompactionResults = new Map(), runId = null) {
  const currentRawToolText = contents
    .flatMap((turn) => turn.parts || [])
    .filter((p) => p.functionResponse)
    .map((p) => p.functionResponse.response?.result || "")
    .join("\n");
  let preCompactionToolText = Array.from(preCompactionResults.values()).join("\n");
  if (runId) {
    const missingIds = findCompactedIdsMissingFromMap(contents, preCompactionResults);
    if (missingIds.length) {
      const fetched = await getPreCompactionResults(runId, missingIds);
      if (fetched.size) preCompactionToolText += "\n" + Array.from(fetched.values()).join("\n");
    }
  }
  const allToolText = currentRawToolText + "\n" + preCompactionToolText;
  return claims.filter((claim) => !allToolText.includes(claim));
}

// Conditional/comparison-expression claims (2026-08-27, fix for a Run 5
// gap: extractMechanicalClaims/findUnverifiedClaims
// check whether each individual identifier TOKEN appears verbatim in raw
// tool output, but not whether the specific COMBINATION/relationship
// asserted between two real tokens (e.g. "step is compared against
// max_steps - 1") matches what the source actually shows. Both `step` and
// `max_steps` are real identifiers that really do appear in the file, so a
// token-level check passes even when the composed expression citing them
// together is fabricated -- confirmed live, Run 5: the model asserted
// `step < max_steps - 1` when the real code is `step < cappedSteps`, and
// both `step` and `max_steps` are genuine tokens present in fetched source;
// only the RELATIONSHIP between them was invented.
//
// This flags claims shaped like a conditional/comparison expression --
// containing a comparison/logical operator (<, >, <=, >=, ===, !==, &&, ||)
// either inside a backtick-quoted span or as bare identifier-op-identifier
// text -- since these need stricter treatment than a bare identifier check:
// not "does this token appear somewhere", but "quote me the EXACT source
// line this composed expression came from" (see the LINE_QUOTE mechanism
// below), because a token-level verbatim check cannot catch a fabricated
// relationship between two real, individually-verifiable tokens.
const CONDITIONAL_CLAIM_PATTERN = /`[^`\n]*(?:<=|>=|===|!==|&&|\|\||[<>])[^`\n]*`|\b[a-zA-Z_$][\w$]*\s*(?:<=|>=|===|!==|&&|\|\||[<>])\s*[\w$.]+(?:\s*[-+]\s*\w+)?/g;

function extractConditionalClaims(answerText) {
  const claims = new Set();
  for (const m of answerText.matchAll(CONDITIONAL_CLAIM_PATTERN)) {
    const raw = (m[0].startsWith("`") ? m[0].slice(1, -1) : m[0]).trim();
    if (raw.length >= 4) claims.add(raw);
  }
  return [...claims];
}

// Mechanical (NOT LLM-judged) check that a model-quoted source line is a
// literal substring of the raw tool-result text already gathered this run.
// Deliberately a plain JS .includes() call, per the research backing this
// fix: self-verification via LLM judgment is weak, mechanical/structural
// verification against ground truth is what actually catches fabrication --
// asking the model itself "is this line real?" would just be another
// LLM-judgment call with the same miscalibrated-confidence failure mode
// this whole mechanism exists to route around.
//
// Now also checks pre-compaction text to support long-running investigations
// where historical tool results have been compacted.
//
// `runId` (optional, same contract as findUnverifiedClaims above):
// side-store fallback, batched via one MGET, for any compacted id
// `preCompactionResults` doesn't have text for.
export async function lineIsVerbatimInToolResults(quotedLine, contents, preCompactionResults = new Map(), runId = null) {
  const currentRawToolText = contents
    .flatMap((turn) => turn.parts || [])
    .filter((p) => p.functionResponse)
    .map((p) => p.functionResponse.response?.result || "")
    .join("\n");
  let preCompactionToolText = Array.from(preCompactionResults.values()).join("\n");
  if (runId) {
    const missingIds = findCompactedIdsMissingFromMap(contents, preCompactionResults);
    if (missingIds.length) {
      const fetched = await getPreCompactionResults(runId, missingIds);
      if (fetched.size) preCompactionToolText += "\n" + Array.from(fetched.values()).join("\n");
    }
  }
  const allToolText = currentRawToolText + "\n" + preCompactionToolText;
  return allToolText.includes(quotedLine.trim());
}

// Parses `LINE_QUOTE: <text>` markers (see the structural line-quote ask in
// the verification prompt below) out of a model response -- the exact-line
// quotes the model was asked to produce for each flagged conditional/
// comparison claim, one per line, in a fixed format specifically so they
// can be parsed and checked programmatically rather than trusting the model
// to have actually done the check just because it says so in prose.
const LINE_QUOTE_PATTERN = /^LINE_QUOTE:\s*(.+)$/gm;

function extractLineQuotes(answerText) {
  const quotes = [];
  for (const m of answerText.matchAll(LINE_QUOTE_PATTERN)) {
    quotes.push(m[1].trim());
  }
  return quotes;
}

// Strips LINE_QUOTE: marker lines out of a final answer before it's ever
// returned to a caller -- they're an internal verification artifact for
// this loop to parse, not something a caller asked for or should see.
function stripLineQuoteMarkers(answerText) {
  return answerText.replace(LINE_QUOTE_PATTERN, "").replace(/\n{3,}/g, "\n\n").trim();
}

// One-time self-check appended after the model's first draft final answer
// (see the verification-pass logic in the loop below). Targets a specific,
// observed failure (2026-08-27: gave a verifiably wrong answer that
// appears to have trusted a later, narrower github_search_code result
// over the complete file it had already read) that the SYSTEM_PREAMBLE
// rules above are meant to prevent DURING synthesis -- this is the backstop
// for when they don't: a forced second pass, after the draft answer already
// exists as concrete text to check claim-by-claim, rather than trusting the
// first synthesis attempt to have applied its own instructions correctly
// under a single pass.
//
// Tools stay available this turn (see withholdTools in the loop below,
// 2026-08-27 fix) -- an earlier no-tools version of this prompt asked the
// model to catch its own mechanical mistakes purely by re-reading scrollback
// from memory, which is the same failure mode in miniature: a live test
// ("Live verification test", runs 2-3) found it confidently
// re-asserting the WRONG one of two similar constants during a no-tools
// verification pass, rather than catching the error. Explicitly telling the
// model to re-fetch when unsure, instead of just re-reading its own
// transcript, is the actual fix.
const VERIFICATION_PROMPT =
  "[SYSTEM NOTE -- verification pass] Before your answer above is treated as final, check it against " +
  "the evidence you already gathered in this conversation. Go back through the RAW tool results already " +
  "in this conversation -- not your own summary of them -- and confirm every specific factual claim in " +
  "your answer (file paths, line numbers, function/variable names, log entries, statuses, dates, " +
  "verdicts like 'consistent' or 'stale') is directly supported by something you actually retrieved. " +
  "You have tool access again this turn -- if you are not certain a claim is correct from what's " +
  "already in the transcript (for example, you are recalling a variable/constant name or an exact " +
  "condition rather than seeing it verbatim above), re-read the specific file or re-run the specific " +
  "search rather than guessing from memory. Do not rely on remembering scrollback further back in this " +
  "conversation when you can just look again. If a full/direct read of a file or page conflicts with a " +
  "narrower or derived result (a search snippet, a grep match) that your answer relied on, the " +
  "full/direct read is the more authoritative source -- prefer it and correct your answer accordingly. " +
  "If you find anything unsupported or contradicted, fix it now. Once you are done checking, respond " +
  "with the corrected final answer (or the same answer, if it already holds up under this check) as " +
  "plain text with no further function calls.";

// Runs the investigation loop. Returns { answer, steps, transcript, runId,
// failed? } where transcript is a human-readable log of each function call
// made (for the Notion write in tools.js) and steps is how many model turns
// it took.
//
// CHECKPOINTING: after every step that completes its function calls, the
// NEW turns added this step are appended to Redis under a per-run UUID
// (see checkpoint.js's fix #5 -- append-delta, not a full-array overwrite;
// write cost is O(turns added this step), not O(conversation so far).
// stepsDone/transcript/task and fix #4's repeat-tracking state are small
// and get rewritten in full each time, which is cheap regardless of run
// length). If the NEXT geminiChat() call
// then fails (429/503/network blip -- exactly what killed a run in testing
// on 2026-07-25), the already-completed steps are not lost: the caller gets
// them back plus `runId`, and can pass `resume_run_id` on a follow-up call
// to continue the same conversation from where it left off instead of
// re-running (and re-paying for) steps 1..N again. Redis is best-effort
// (see checkpoint.js) -- if it's unavailable, resumption just isn't
// possible, same as before this existed; a failure still returns whatever
// transcript was gathered in-memory this call.
export async function runInvestigation({ task, max_steps = 20, resume_run_id, provider, model, maxOutputTokens, singleStep = false, preambleVariant = "verbose" }) {
  // The provider actually in effect for this run -- the caller-supplied one
  // on a fresh run, or the one restored from a resumed checkpoint (see
  // `checkpoint.provider || provider` below). A checkpointed `contents`
  // array is shaped for whichever provider originally produced it (Gemini's
  // shape either way, per connectors/glm/adapter.js -- but GLM-originated
  // turns may carry provider-specific quirks the adapter round-trips
  // faithfully rather than normalizing away), so resuming on a DIFFERENT
  // provider than the one that started the run is not just a preference
  // mismatch -- it risks silently corrupting the conversation. Threaded
  // through every providerChat/saveCheckpoint call below exactly like
  // effectiveTask is.
  let effectiveProvider = provider;
  // Same reasoning as effectiveProvider: which preamble variant a run uses
  // is fixed at run start (fresh or seeded) and must not silently change on
  // a resume just because a later caller passed a different value -- a run
  // resumed mid-conversation under a DIFFERENT preamble than the one its
  // earlier turns were built with would be an inconsistent conversation,
  // not just a preference mismatch. Threaded through saveCheckpoint below
  // exactly like effectiveProvider/effectiveTask.
  let effectivePreambleVariant = preambleVariant;
  // Same reasoning as effectiveProvider directly above: a caller-supplied
  // model on a fresh run, or the one restored from a resumed checkpoint so
  // a resume can't silently switch models mid-conversation. Undefined is a
  // valid value throughout (providerChat/glmChat fall back to GLM_MODEL),
  // so no special-casing is needed beyond mirroring effectiveProvider's
  // pattern.
  let effectiveModel = model;
  // Same pattern again for the per-turn output-token cap.
  let effectiveMaxOutputTokens = maxOutputTokens;
  // The run's TRUE overall step ceiling -- distinct from cappedSteps (this
  // particular invocation's own loop bound). For a normal synchronous
  // caller (or a fresh run) the two are the same value. They diverge for a
  // worker-driven singleStep resume (agent_worker.js), which deliberately
  // passes a shrunk per-call bound (stepsDone + 1, so the shared loop takes
  // exactly one step) that must NOT be mistaken for "the run's real last
  // step" when deciding whether to withhold tools -- see isFinalStep below.
  // Restored from the checkpoint on a singleStep resume; established fresh
  // (or updated, per the documented "resuming with max_steps sets a new
  // ceiling" behavior) otherwise. Persisted in every checkpoint write below
  // so it survives a resume.
  let effectiveOverallMaxSteps;
  let cappedSteps;

  let runId = resume_run_id;
  let contents;
  let transcript;
  let startStep;
  // The task text actually in effect for this run -- the caller-supplied
  // one on a fresh run, or the one restored from a resumed checkpoint.
  // Tracked (and persisted in every checkpoint below) so callers/tools.js
  // can log/title a resumed run without needing the caller to re-supply
  // task text the loop itself ignores on resume.
  let effectiveTask = task;
  // Stuck-loop detection (fix #4, 2026-07-27): repeatCounts tracks how many
  // times each exact (function name + JSON-stringified args) signature has
  // been called THIS RUN, persisted across resumes (see checkpoint.js) so a
  // resumed run doesn't forget what it already tried. resultCache holds the
  // actual result text per signature, keyed the same way.
  //
  // FIX (2026-08-31, debug/resultcache-not-persisted-on-resume): this
  // comment used to say resultCache was deliberately NOT persisted, on the
  // theory that a resumed run's repeat call would just re-execute once more
  // -- a correctness no-op, not worth the checkpoint weight. That held for a
  // rare, exceptional resume, but
  // agent_worker.js's QStash self-chaining path calls runInvestigation with
  // resume_run_id + singleStep: true ONCE PER STEP as its normal unit of
  // execution -- resultCache was therefore wiped every single step under
  // ordinary async operation, and any repeat call spanning a step boundary
  // silently re-executed for real (repeatCounts correctly reported
  // isRepeat: true, but the serve-from-cache check needs BOTH isRepeat and
  // resultCache.has(signature), and the latter was always false in a fresh
  // invocation).
  //
  // Fixed via the same side-store pattern as preCompactionResults
  // (savePreCompactionResult/getPreCompactionResult below): a signature's
  // result text is written once, the first time it's computed, to its own
  // Redis key (saveResultCacheEntry) -- never inlined into the checkpoint's
  // `meta` blob, which would reopen the exact unbounded-growth problem
  // preCompactionResults' own side-store fix was for. Unlike
  // preCompactionResults (which restores every aged-out entry up front via
  // compactHistoryInPlace, since each one WILL be needed for verification),
  // resultCache entries are fetched lazily, on demand, right before
  // executing each step's function calls -- only for signatures repeatCounts
  // already knows are a repeat AND that this fresh invocation's empty
  // in-memory Map doesn't have yet, batched via one MGET per step rather
  // than one round trip per call (see the pre-pass right before the
  // `Promise.all(functionCalls.map(...))` call below). `resultCacheIds` in
  // the checkpoint's meta blob (ids only, mirroring preCompactionResultIds)
  // exists purely for deleteCheckpoint's GC sweep, not for restoring
  // in-memory state on resume.
  // consecutiveAllRepeatSteps counts how many steps IN A ROW consisted
  // ENTIRELY of repeat calls -- the real stuck-loop signal (a single repeat
  // mixed with new calls is normal exploration, not a stuck loop).
  let repeatCounts = new Map();
  let resultCache = new Map();
  let preCompactionResults = new Map();
  let consecutiveAllRepeatSteps = 0;
  // Verification pass (2026-08-27, see VERIFICATION_PROMPT's comment above
  // for the specific failure it targets): true once the model has produced
  // a draft final answer and been sent back for one no-tools self-check
  // round before that answer is trusted. Persisted across resumes (below)
  // so a run that dies mid-verification -- e.g. the verification call
  // itself hits a transient 429/503 -- resumes into the verification turn
  // again rather than silently re-entering normal tool-use and re-drafting
  // a whole new answer from scratch.
  let pendingVerification = false;
  // Bounds the NEW structural line-quote recheck (see LINE_QUOTE mechanism
  // above) to exactly one extra round, same single-fire pattern as
  // pendingVerification -- once a corrective round has been sent for a
  // failed line-quote check, whatever comes back is accepted as final
  // regardless of whether it still has issues, so this can never loop more
  // than one extra step beyond the existing verification pass. Persisted
  // across resumes below for the same reason pendingVerification is.
  let structuralRecheckUsed = false;
  // How many entries of `contents` have already been pushed to the Redis
  // checkpoint list (fix #5) -- saveCheckpoint only ever needs the SLICE
  // added since the last checkpoint, not the whole array, so this cursor is
  // what makes that possible without checkpoint.js needing to diff arrays
  // itself.
  let contentsCheckpointedUpTo = 0;

  const checkpoint = resume_run_id ? await loadCheckpoint(resume_run_id) : null;
  // Async delegate_agent (Scenario A/B groundwork): a checkpoint
  // whose last save recorded status "done" already has a final answer sitting
  // in Redis (see the completion path near the end of this function, which
  // now PERSISTS a done checkpoint instead of deleting it, specifically so
  // this branch has something to read) -- return it directly rather than
  // re-entering the loop, which would otherwise treat `contents` as still
  // mid-conversation and either try to take more (nonsensical, unbounded-cost)
  // steps or misbehave against startStep/cappedSteps math that was never
  // designed for an already-finished run. This is also what makes
  // resume_run_id usable as a poll handle for a background/worker-driven run
  // (see agent_worker.js): polling a finished run is now a cheap Redis read,
  // not a re-run.
  if (checkpoint && checkpoint.status === "done") {
    return {
      answer: checkpoint.finalAnswer,
      steps: checkpoint.stepsDone,
      transcript: checkpoint.transcript,
      runId: resume_run_id,
      task: checkpoint.task,
      failed: false,
    };
  }
  if (checkpoint) {
    contents = checkpoint.contents;
    transcript = checkpoint.transcript;
    startStep = checkpoint.stepsDone + 1;
    // Every entry loadCheckpoint returned in `contents` was already RPUSHed
    // to Redis in a prior call -- nothing new to push until this run adds
    // more turns, so the cursor starts at the end of what was loaded.
    contentsCheckpointedUpTo = contents.length;
    // Same reasoning as `checkpoint.task || task` below -- once a run is
    // past step 1, the checkpoint's own record of which provider started it
    // is authoritative, not whatever the caller passes on a resume call.
    // Checkpoints saved before the provider field existed won't have it;
    // fall back to whatever the caller passed (may be undefined, which
    // providerChat/router.js treats as "gemini") rather than erroring.
    effectiveProvider = checkpoint.provider || provider;
    effectiveModel = checkpoint.model || model;
    effectiveMaxOutputTokens = checkpoint.maxOutputTokens || maxOutputTokens;
    // Checkpoints saved before this field existed won't have it -- fall
    // back to whatever the caller passed (may be undefined, treated as
    // "verbose" by buildSystemPreamble's own default) rather than erroring,
    // same defensive pattern as every other field restored here.
    effectivePreambleVariant = checkpoint.preambleVariant || preambleVariant;
    // Maps aren't JSON-serializable, so saveCheckpoint stores repeatCounts
    // as a plain object and this reconstructs the Map on load. Checkpoints
    // saved before fix #4 existed won't have this field -- fall back to an
    // empty Map rather than erroring, same defensive pattern as
    // `checkpoint.task || task` below.
    repeatCounts = new Map(Object.entries(checkpoint.repeatCounts || {}));
    // Checkpoints saved before this field existed won't have it -- fall back
    // to an empty Map rather than erroring, same defensive pattern as repeatCounts.
    preCompactionResults = new Map(Object.entries(checkpoint.preCompactionResults || {}));
    consecutiveAllRepeatSteps = checkpoint.consecutiveAllRepeatSteps || 0;
    // Checkpoints saved before this field existed won't have it -- default
    // to false (normal tool-use resumes as before), same defensive pattern
    // as every other field restored here.
    pendingVerification = checkpoint.pendingVerification || false;
    // Same defensive pattern -- checkpoints saved before this field existed
    // won't have it -- default to false (no structural recheck round used
    // yet), same as pendingVerification above.
    structuralRecheckUsed = checkpoint.structuralRecheckUsed || false;
    // Prefer the checkpoint's own record of the original task -- `task` is
    // genuinely ignored on a live resume (see file header), so this is the
    // only reliable source once a run is past step 1. Checkpoints saved
    // before this field existed won't have it; fall back to whatever the
    // caller passed (may be undefined) rather than erroring.
    effectiveTask = checkpoint.task || task;
    // `saveCheckpoint`'s `contents` Redis list is append-only (only new turns since the last save are ever RPUSHed -- see agent_checkpoint.js's own header comment on this) -- it never rewrites a turn already in Redis, even though `compactHistoryInPlace` mutates old turns' functionResponse text in place, in memory only. So a turn that was compacted before a crash comes back from `loadCheckpoint` in its ORIGINAL, uncompacted form, and without re-compacting immediately here, this resumed run's very first `providerChat` call later in the loop would re-send that full, uncompacted history -- exactly the token-bloat failure this feature exists to prevent, and exactly the scenario (the `bai` provider exhausting its API keys mid-run) that makes a caller reach for `resume_run_id` in the first place.
    // This is idempotent and safe to call unconditionally on every resume: entries already compacted before the crash get their `preCompactionResults` entry re-set to the same original text (now restored, uncompacted, from Redis) and get re-compacted to the same marker text -- no data loss, no double-compaction artifacts -- and entries that hadn't yet aged past the recent-detail window last time may now cross it if `stepsDone` has grown since, so this also correctly catches those.
    // `startStep` is "the step about to run" for this resumed call, exactly matching what the main loop's own call site passes on every other iteration (`step`, not `step - 1`) — passing `startStep - 1` (completed-steps count) here was itself an off-by-one against `compactHistoryInPlace`'s actual formula, which would silently under-compact by one step immediately after a resume (the very case FIX A exists to correctly handle), re-sending one more full-size turn than intended on the resumed run's first providerChat call.
    await compactHistoryInPlace(contents, startStep, preCompactionResults, { provider: effectiveProvider, runId });
  } else if (resume_run_id && !task) {
    // A resume WAS requested but its checkpoint didn't load -- expired past
    // the 1-hour TTL, Redis unavailable (checkpoint.js is deliberately
    // fail-open, see its header), or an invalid/typo'd runId -- AND there is
    // no task to fall back on either. This must NEVER be silently treated as
    // "no resume was requested" and fall through to a fresh run: that
    // previously produced a conversation seeded with `Task: undefined` (task
    // is genuinely ignored on a live resume, so callers legitimately omit
    // it), and the model burned several steps hunting blind for context
    // instead of investigating (found via the 2026-07-26 checkpoint-miss
    // test). Fail loudly and distinctly instead, so the caller can tell
    // "your resume target is gone" apart from any other failure.
    //
    // If a task WAS provided alongside a resume_run_id that fails to load,
    // this branch is skipped and the fresh-run branch below runs instead --
    // a legitimate defensive-caller pattern (passing the task as a fallback
    // even on a resume call), kept intentionally per the fix plan.
    throw new Error(
      isRedisConfigured()
        ? `resume_run_id "${resume_run_id}" has no live checkpoint -- it may have expired (1 hour TTL) or the id may be wrong. ` +
          `There is no saved task to resume from. Start a new investigation by calling again with a task and no resume_run_id.`
        : `resume_run_id "${resume_run_id}" has no live checkpoint -- and Redis is NOT configured in this environment ` +
          `(UPSTASH_REDIS_REST_URL/UPSTASH_REDIS_REST_TOKEN unset or unreachable), so no checkpoint could ever have been saved to resume from, ` +
          `regardless of the runId or how recently the original call failed. Retrying resume_run_id again will not help -- ` +
          `start a new investigation with a task instead, and expect that a future transient failure won't be resumable either until Redis is configured.`
    );
  } else {
    // Either no resume_run_id was given, or one was given with its checkpoint
    // missing but a `task` supplied as a fallback (see branch above) --
    // start a fresh run either way. Requires a real `task` (the caller-facing
    // tool in tools.js already guards against a missing task on a
    // non-resumable call, so `task` is trustworthy here).
    runId = randomUUID();
    contents = [{ role: "user", parts: [{ text: appendTask(buildSystemPreamble(effectiveProvider, effectivePreambleVariant), task) }] }];
    transcript = [];
    startStep = 1;
  }

  // Establish this call's own loop bound (cappedSteps) and the run's true
  // ceiling (effectiveOverallMaxSteps) now that startStep/checkpoint are
  // known. singleStep (agent_worker.js's one-step-per-invocation resume)
  // deliberately does NOT let this call's own max_steps redefine the run's
  // real ceiling -- it restores that ceiling from the checkpoint instead,
  // and bounds only THIS call's loop to a single iteration. Every other
  // caller (a fresh run, or a manual synchronous resume) keeps the existing,
  // documented behavior: max_steps sets/updates the real ceiling directly.
  if (singleStep) {
    if (!checkpoint) {
      // Should not happen in practice -- the `resume_run_id && !task` branch
      // above already throws a clearer error when a resume's checkpoint
      // fails to load. Guarded here too so a future caller of singleStep
      // without task/resume_run_id set correctly fails loudly instead of
      // silently mis-capping an investigation that was never actually
      // resumed from anything.
      throw new Error(`singleStep resume requested for runId "${resume_run_id}" but no live checkpoint was found.`);
    }
    // Checkpoints seeded/saved before this field existed won't have it --
    // fall back to the tool's own documented default (20) rather than
    // erroring or leaving the run's real ceiling undefined.
    effectiveOverallMaxSteps = checkpoint.overallMaxSteps || Math.min(20, HARD_MAX_STEPS);
    cappedSteps = startStep;
  } else {
    cappedSteps = Math.min(max_steps, HARD_MAX_STEPS);
    effectiveOverallMaxSteps = cappedSteps;
  }

  // Resuming with a max_steps ceiling that's already been met or exceeded
  // by the checkpoint's own stepsDone (e.g. a checkpoint has 5 completed
  // steps and the caller resumes with max_steps: 2) -- there's no budget
  // left to take even one more step. Don't fall into the loop-and-fall-
  // through path below: that unconditionally deletes the checkpoint via
  // deleteCheckpoint(runId) once the loop exits, which would throw away a
  // still-good, still-resumable checkpoint for no reason (the loop body
  // simply never executes when startStep > cappedSteps), and the generic
  // step-cap message doesn't explain that anything was actually completed.
  // Leave the checkpoint alone -- it's still resumable with a higher
  // max_steps -- and say so explicitly instead.
  if (checkpoint && startStep > cappedSteps) {
    return {
      answer: `(This run already completed ${startStep - 1} step(s), which meets or exceeds the requested max_steps of ${cappedSteps} -- no new steps were taken this call. The checkpoint has NOT been discarded. Call delegate_agent again with resume_run_id: "${runId}" and a higher max_steps to continue, or treat the ${transcript.length} tool call(s) below as the result so far.)`,
      steps: startStep - 1,
      transcript,
      runId,
      task: effectiveTask,
      failed: true,
    };
  }

  for (let step = startStep; step <= cappedSteps; step++) {
    // On the final allowed step, withhold the function-calling tools
    // entirely instead of just reminding the model to wrap up: a text-only
    // reminder wasn't reliable enough on its own (found via the 2026-07-26
    // test -- the model spent its very last step on another tool call
    // anyway, and the run hit the cap with zero synthesized answer, not
    // even an incomplete one). Without `tools` in the request body, Gemini
    // structurally cannot return a functionCall part here, so this step is
    // guaranteed to be a real text-answer attempt rather than another read.
    const isFinalStep = step === effectiveOverallMaxSteps;
    // Stuck-loop forced-answer (fix #4): once 3 consecutive steps have
    // consisted ENTIRELY of repeat calls (consecutiveAllRepeatSteps, updated
    // at the end of each step below), withhold tools the same way the final
    // step already does -- a text-only SYSTEM NOTE alone wasn't trusted to
    // reliably stop a model that keeps re-issuing the same call (same
    // lesson as isFinalStep's own history, see its comment above), so this
    // reuses that structural fix instead of a new mechanism.
    const stuckLoopForce = consecutiveAllRepeatSteps >= 3;
    // Verification pass (see VERIFICATION_PROMPT above): deliberately does
    // NOT withhold tools, unlike isFinalStep/stuckLoopForce. Those two
    // withhold tools to force a stop; this one exists to catch the model
    // trusting a wrong or misremembered mechanical detail (a variable name,
    // a threshold, which of two similar-looking things gates a condition),
    // and the fix for "misremembered" is letting it look again, not asking
    // it to recall harder from a long scrollback transcript -- that's the
    // same failure mode in a smaller box. Confirmed live (2026-08-27, see
    // "Live verification test" runs 2-3): a no-tools verification
    // pass confidently asserted the wrong constant (HARD_MAX_STEPS instead
    // of the actual cappedSteps) gated a condition, i.e. it re-affirmed a
    // wrong answer from memory instead of catching it. Tool access lets the
    // model re-read the actual line instead of guessing which of two
    // constants it half-remembers is the real one.
    const withholdTools = isFinalStep || stuckLoopForce;
    // bai-only, forced-final-step-only reasoning_effort override (plan.md
    // Section 25 fix): mitigates the reasoning-token-budget-exhaustion
    // failure mode isolated via test-bai-timeout.sh, where the forced
    // no-tools final call can spend nearly all of max_tokens on internal
    // reasoning before ever writing a visible answer. Scoped to isFinalStep
    // specifically -- not the broader withholdTools (which also covers
    // stuckLoopForce and the verification pass) -- since that's the one
    // call site this investigation actually reproduced and root-caused;
    // see connectors/bai/client.js's baiChat for the actual retry logic
    // this pairs with (isReasoningBudgetExhausted).
    const reasoningEffort = getDelegateHooks(effectiveProvider).getReasoningEffort(isFinalStep);
    let candidate;
    try {
      candidate = await providerChat(contents, { provider: effectiveProvider, tools: withholdTools ? undefined : FUNCTION_DECLARATIONS, model: effectiveModel, maxOutputTokens: effectiveMaxOutputTokens, reasoningEffort });
      const cascadeLog = formatCascadeLogLine(candidate, { step, fallbackModel: effectiveModel });
      if (cascadeLog) transcript.push(cascadeLog);
    } catch (err) {
      // The step-1..N-1 work already happened and is real -- don't throw it
      // away. Persist it (redundant with the save at the end of the prior
      // iteration, but cheap and safe) and hand the caller everything they
      // need to resume instead of restarting. newContents is usually empty
      // here (this failure happens before this step's model turn is ever
      // pushed to `contents`) -- saveCheckpoint just re-writes the small
      // meta blob in that case, which is exactly the O(delta) behavior fix
      // #5 is for.
      await saveCheckpoint(runId, {
        newContents: contents.slice(contentsCheckpointedUpTo),
        transcript,
        stepsDone: step - 1,
        task: effectiveTask,
        repeatCounts: Object.fromEntries(repeatCounts),
        preCompactionResults: Object.fromEntries(preCompactionResults),
        resultCache: Object.fromEntries(resultCache),
        consecutiveAllRepeatSteps,
        provider: effectiveProvider,
        model: effectiveModel,
        maxOutputTokens: effectiveMaxOutputTokens,
        preambleVariant: effectivePreambleVariant,
        pendingVerification,
        overallMaxSteps: effectiveOverallMaxSteps,
      });
      const errMessage = err?.message ?? String(err);
      const redisOk = isRedisConfigured();
      const resumeHint = isTransientGeminiError(err)
        ? (redisOk
            ? ` ${transcript.length} tool call(s) already completed this run are saved. Call delegate_agent again with resume_run_id: "${runId}" to continue from here instead of starting over. Checkpoint expires in 1 hour.`
            : ` ${transcript.length} tool call(s) were completed this run, but Redis is NOT configured in this environment, so nothing was actually saved -- resume_run_id: "${runId}" will NOT work no matter how soon you retry. ` +
              `The completed tool calls are listed in this run's transcript/Notion log (if log_to_notion was set) for manual reference, but the only way to continue is a fresh call with the full task text.`)
        : ` This does not look like a transient error (not a 429/503) -- resuming with resume_run_id: "${runId}" will likely reproduce the same failure, so check the underlying cause (e.g. GEMINI_API_KEY, request format, safety/recitation block) before retrying. The ${transcript.length} tool call(s) already completed are still saved if you want to resume anyway${redisOk ? "" : " (though note: Redis is NOT configured in this environment, so nothing was actually saved regardless)"}.`;
      return {
        answer: `(Gemini call failed on step ${step}: ${errMessage} --${resumeHint})`,
        steps: step - 1,
        transcript,
        runId,
        task: effectiveTask,
        failed: true,
      };
    }

    const parts = candidate.content?.parts || [];
    const allFunctionCalls = parts.filter((p) => p.functionCall);

    // Oversized-step guardrail #1 (count cap, see MAX_TOOL_CALLS_PER_STEP's
    // definition above): only the first MAX_TOOL_CALLS_PER_STEP calls the
    // model batched into this turn are actually executed. Every function
    // call in a turn still needs a functionResponse (Gemini/bai's contract
    // requires one per functionCall.id, regardless of whether we chose to
    // run it) -- deferredFunctionCalls get a synthetic, non-executed
    // response below instead of silently vanishing, so the model sees
    // exactly why the rest didn't run and can re-request them next step.
    // Oversized-step caps below (MAX_TOOL_CALLS_PER_STEP, MAX_STEP_RESULT_CHARS,
    // and the final-step brevity note) are scoped to provider "bai" only --
    // they were added to fix bai-specific 300s platform-timeout failures
    // (plan.md Sections 9-16), never observed on Gemini in any repro run.
    // Gemini must not have its tool-call batching or result sizing changed
    // by a fix aimed at a different provider's failure mode.
    const applyOversizedStepCaps = effectiveProvider === "bai";
    const functionCalls = applyOversizedStepCaps ? allFunctionCalls.slice(0, MAX_TOOL_CALLS_PER_STEP) : allFunctionCalls;
    const deferredFunctionCalls = applyOversizedStepCaps ? allFunctionCalls.slice(MAX_TOOL_CALLS_PER_STEP) : [];

    if (!functionCalls.length) {
      const answer = parts.map((p) => p.text || "").join("").trim();

      // First arrival at a draft final answer, with tool access still
      // available this turn (not already a forced no-tools turn) and steps
      // left in the budget: don't trust it at face value yet -- push it
      // back for one additional no-tools self-check turn first. See
      // VERIFICATION_PROMPT's comment above for the specific, observed
      // failure this targets (a later, narrower search result trusted over
      // a complete file already read earlier in the same run). If no steps
      // remain, or this turn was ALREADY a forced no-tools turn (final
      // step, stuck-loop, or the verification pass itself -- all captured
      // by withholdTools), fall through to the ordinary return below
      // instead: there's no budget left to check, or this IS the check.
      // `!pendingVerification` here (new alongside the withholdTools change
      // above) is what keeps this a ONE-TIME check: now that verification
      // turns keep tool access, a draft answer produced BY the verification
      // turn would otherwise satisfy this same condition again and loop
      // back into another verification round indefinitely. The step budget
      // (step < cappedSteps) still bounds worst-case cost, but this makes
      // the intent explicit -- verify once, then trust the result.
      if (answer && !withholdTools && !pendingVerification && step < cappedSteps) {
        // Mechanical-claim citation check (see extractMechanicalClaims'
        // comment above): identify specific identifier/constant/threshold-
        // shaped claims in the draft that do NOT appear verbatim in any raw
        // tool result gathered so far, and name them explicitly in the
        // verification prompt rather than leaving "check your claims" as a
        // generic instruction the model can satisfy by just re-reading its
        // own draft and feeling confident about it again.
        const mechanicalClaims = extractMechanicalClaims(answer);
        const unverifiedClaims = await findUnverifiedClaims(mechanicalClaims, contents, preCompactionResults, runId);
        const claimNote = unverifiedClaims.length
          ? `\n\n[SPECIFIC ITEMS TO CHECK] The following identifier(s)/snippet(s) in your draft answer do not appear verbatim in any raw tool result you've fetched so far this run: ${unverifiedClaims.map((c) => `"${c}"`).join(", ")}. For EACH one: re-read or re-search the specific file/location it's claimed to come from and confirm it exact-matches what's actually there, THEN either (a) keep the claim only if you can now quote it verbatim from a fresh tool result, or (b) correct it if the fresh read shows something different. Do not restate any of these unchanged based on memory or on the fact that you already wrote it once -- that is exactly the failure mode this check exists to catch. (Note: some of these may be false positives -- ordinary words, or claims already correctly quoted -- but each one still needs a fresh check, not a guess about which category it's in.)`
          : "";
        // Structural line-quote ask (see CONDITIONAL_CLAIM_PATTERN's comment
        // above): a token-presence check is not enough for a claim that
        // COMBINES two or more real identifiers into a relationship (an
        // operator between them) -- the model must instead quote the exact
        // literal source line for each, in a fixed, parseable format, so
        // round 2 below can verify the quote itself via a plain string
        // .includes() rather than trusting the model's own assertion that
        // it double-checked.
        const conditionalClaims = extractConditionalClaims(answer);
        const conditionalNote = conditionalClaims.length
          ? `\n\n[STRUCTURAL LINE-QUOTE CHECK] Your draft answer contains conditional/comparison expression(s) that a simple identifier check cannot verify, because the individual tokens in them can be real even when the RELATIONSHIP between those tokens is wrong: ${conditionalClaims.map((c) => `"${c}"`).join(", ")}. For EACH one: find the exact single source line it is based on (re-reading the file if you need to) and reproduce that line's ACTUAL, LITERAL text -- copied exactly, not paraphrased, reformatted, or reconstructed from memory -- on its own line in your response, in this exact format: LINE_QUOTE: <the exact literal source line>. Provide one LINE_QUOTE line for each conditional/comparison expression listed above, in addition to your corrected final answer text.`
          : "";
        contents.push({ role: "model", parts });
        contents.push({ role: "user", parts: [{ text: VERIFICATION_PROMPT + claimNote + conditionalNote }] });
        pendingVerification = true;
        await saveCheckpoint(runId, {
          newContents: contents.slice(contentsCheckpointedUpTo),
          transcript,
          stepsDone: step,
          task: effectiveTask,
          repeatCounts: Object.fromEntries(repeatCounts),
          preCompactionResults: Object.fromEntries(preCompactionResults),
          resultCache: Object.fromEntries(resultCache),
          consecutiveAllRepeatSteps,
          provider: effectiveProvider,
          model: effectiveModel,
          maxOutputTokens: effectiveMaxOutputTokens,
          preambleVariant: effectivePreambleVariant,
          pendingVerification,
          structuralRecheckUsed,
          overallMaxSteps: effectiveOverallMaxSteps,
        });
        contentsCheckpointedUpTo = contents.length;
        continue;
      }

      // Round 2: this is the model's response to the verification pass
      // above (pendingVerification already true). Before trusting it,
      // mechanically check any LINE_QUOTE: lines it produced against the raw
      // tool-result text already gathered -- NOT another LLM judgment call,
      // a plain string .includes() (see lineIsVerbatimInToolResults). If any
      // quoted "exact" line can't actually be found verbatim, the model
      // fabricated or misremembered it even while claiming to have checked,
      // so send exactly ONE corrective round naming which quotes failed and
      // accept whatever comes back after that as final -- bounded by
      // structuralRecheckUsed so this can never loop more than once beyond
      // the existing verification pass, consistent with pendingVerification's
      // own single-fire guard.
      if (answer && pendingVerification && !structuralRecheckUsed && step < cappedSteps) {
        const quotedLines = extractLineQuotes(answer);
        // lineIsVerbatimInToolResults is async (side-store fallback), so
        // resolve every lookup first via Promise.all, then filter on the
        // resolved results -- Array.prototype.filter can't await inside its
        // predicate.
        const verbatimChecks = await Promise.all(quotedLines.map((q) => lineIsVerbatimInToolResults(q, contents, preCompactionResults, runId)));
        const badQuotes = quotedLines.filter((_, i) => !verbatimChecks[i]);
        if (badQuotes.length) {
          const correctionNote =
            `[STRUCTURAL LINE-QUOTE CHECK FAILED] The following line(s) you quoted as exact source text could NOT be found verbatim in any raw tool result already in this conversation: ${badQuotes.map((q) => `"${q}"`).join(", ")}. This means at least one quoted line was reconstructed, paraphrased, or misremembered rather than copied exactly, which means the conditional/comparison claim it was meant to support has NOT actually been confirmed. Re-read the actual file or location again right now, copy the REAL line character-for-character (do not paraphrase, reformat whitespace, or reconstruct from memory), and give your corrected final answer along with a corrected LINE_QUOTE: <line> for each expression you're re-checking. This is the last verification round -- give your best, fully corrected final answer this time.`;
          contents.push({ role: "model", parts });
          contents.push({ role: "user", parts: [{ text: correctionNote }] });
          structuralRecheckUsed = true;
          await saveCheckpoint(runId, {
            newContents: contents.slice(contentsCheckpointedUpTo),
            transcript,
            stepsDone: step,
            task: effectiveTask,
            repeatCounts: Object.fromEntries(repeatCounts),
            preCompactionResults: Object.fromEntries(preCompactionResults),
            resultCache: Object.fromEntries(resultCache),
            consecutiveAllRepeatSteps,
            provider: effectiveProvider,
            model: effectiveModel,
            maxOutputTokens: effectiveMaxOutputTokens,
            preambleVariant: effectivePreambleVariant,
            pendingVerification,
            structuralRecheckUsed,
            overallMaxSteps: effectiveOverallMaxSteps,
          });
          contentsCheckpointedUpTo = contents.length;
          continue;
        }
      }

      // Async delegate_agent (groundwork): persist a "done" checkpoint
      // (status + finalAnswer) here instead of unconditionally deleting the
      // checkpoint the way this used to. A resume_run_id caller polling a
      // background/worker-driven run (see the checkpoint.status === "done"
      // short-circuit near the top of this function) needs SOMETHING to read
      // once the run finishes -- deleting it the instant it completes meant
      // there was never a window in which a "done" status could be observed.
      // `newContents: []` -- the conversation array itself isn't needed once
      // a run is done (only finalAnswer/steps/transcript/task are read back
      // by the short-circuit above), so there's no reason to pay the cost of
      // pushing whatever tail of `contents` hasn't been checkpointed yet.
      // Still expires via the normal CHECKPOINT_TTL_SECONDS (1 hour) like any
      // other checkpoint -- this is a bounded-lifetime record for polling,
      // not a permanent store, same contract as every other checkpoint.
      const finishRun = async (result) => {
        await saveCheckpoint(runId, {
          newContents: [],
          transcript: result.transcript,
          stepsDone: result.steps,
          task: result.task,
          repeatCounts: Object.fromEntries(repeatCounts),
          preCompactionResults: Object.fromEntries(preCompactionResults),
          resultCache: Object.fromEntries(resultCache),
          consecutiveAllRepeatSteps,
          provider: effectiveProvider,
          model: effectiveModel,
          maxOutputTokens: effectiveMaxOutputTokens,
          preambleVariant: effectivePreambleVariant,
          pendingVerification,
          structuralRecheckUsed,
          overallMaxSteps: effectiveOverallMaxSteps,
          status: "done",
          finalAnswer: result.answer,
        });
        // Every finishRun call site is a successful completion (the failure
        // paths elsewhere in this function return their own `failed: true`
        // shape directly, without going through finishRun) -- explicitly
        // set `failed: false` rather than leaving it undefined, so callers
        // that assert on this field (e.g. an integration test resuming a
        // run) see a real boolean instead of relying on undefined/false
        // being treated the same by `if (result.failed)` checks downstream.
        return { failed: false, ...result };
      };
      if (answer) {
        // Strip internal LINE_QUOTE: markers before ever returning an answer
        // to a caller -- they're a parseable artifact for THIS loop's own
        // structural check, not something a caller asked for or should see.
        const cleanedAnswer = stripLineQuoteMarkers(answer);
        // Tool-call-leakage backstop (bai-only, see detectToolCallLeakage's
        // comment above): only checked on a withholdTools turn (final step/
        // stuck-loop/verification), the exact shape both observed leaks
        // (6ea018d5, ab8afaa8) occurred on, and only for bai -- Gemini has
        // never exhibited this and must not be affected by a bai-specific
        // backstop.
        if (applyOversizedStepCaps && withholdTools) {
          const leakedToolName = detectToolCallLeakage(cleanedAnswer, FUNCTION_NAME_SET);
          if (leakedToolName) {
            return finishRun({
              answer: `(The model attempted to invoke the "${leakedToolName}" tool as text on this turn -- no tools were available (see the SYSTEM NOTE reason above), so nothing was executed and no real answer was produced. This is a known failure mode on bai's forced-final step (plan.md Section 18/20) -- retry with a higher max_steps so a real tool-enabled step is available, or narrow the task. Raw (unusable) model output follows for reference:\n\n${cleanedAnswer})`,
              steps: step, transcript, runId, task: effectiveTask, failed: true,
            });
          }
        }
        return finishRun({ answer: cleanedAnswer, steps: step, transcript, runId, task: effectiveTask });
      }
      if (!answer) {
        // MALFORMED_FUNCTION_CALL on a no-tools turn specifically means:
        // this step had NO tools in the request (isFinalStep/stuckLoopForce/
        // pendingVerification all withhold them, see withholdTools above),
        // but the model tried to make a function call anyway -- Gemini
        // rejects that as malformed rather than falling back to text.
        // Observed concretely with max_steps: 1 on a task that genuinely
        // needed a file read: the model had no way to answer without a
        // tool, no tools were offered, and the result was this opaque
        // finishReason with zero explanation of why (2026-07-26 stress
        // test). Surface the actual cause instead of just the raw enum
        // value, since "try a higher max_steps" is the fix and the caller
        // has no way to infer that from "MALFORMED_FUNCTION_CALL" alone.
        const starvationNote = withholdTools
          ? (isFinalStep
              ? (candidate.finishReason === "MALFORMED_FUNCTION_CALL"
                  ? ` This was the final allowed step, which never includes tools (so the model can only answer in plain text here) -- but the model attempted a function call anyway, which Gemini rejects as malformed when no tools are available. This almost always means the task genuinely requires at least one tool call and max_steps (${effectiveOverallMaxSteps}) left no tool-enabled steps to make it in. Retry with a higher max_steps (at least 2, ideally the default of 6 for anything non-trivial).`
                  : ` This was the final allowed step (no tools available), and the model stopped with an empty response instead of a real answer -- no fabricated/garbled output, but also no usable content. This means max_steps (${effectiveOverallMaxSteps}) was not enough room for the model to actually finish the task, and it gave up rather than attempting a rushed or invented answer. Retry with a higher max_steps, or narrow the task so it fits within the current budget.`)
              : pendingVerification
              ? (candidate.finishReason === "MALFORMED_FUNCTION_CALL"
                  ? ` This was the verification pass (no tools offered on purpose -- see VERIFICATION_PROMPT), but the model attempted a function call anyway, which Gemini rejects as malformed when no tools are available. The draft answer from the step before this one was never returned to the caller as a result -- treat this run as having produced no usable answer, and consider retrying with a higher max_steps in case the verification turn simply needed more room.`
                  : ` This was the verification pass (no tools offered on purpose), and the model stopped with an empty response rather than a corrected answer. The draft answer from the step before this one was never returned to the caller -- treat this run as having produced no usable answer, and consider retrying with a higher max_steps.`)
              : (candidate.finishReason === "MALFORMED_FUNCTION_CALL"
                  ? ` This step had no tools available because ${consecutiveAllRepeatSteps} consecutive steps consisted entirely of repeat calls (same function + arguments already tried this run) -- fix #4's stuck-loop guard forces a text-only answer the same way the final step does, but the model attempted a function call anyway, which Gemini rejects as malformed when no tools are available. The task likely needs to be narrowed or rephrased so it doesn't require repeating the same information-gathering calls.`
                  : ` This step had no tools available because ${consecutiveAllRepeatSteps} consecutive steps consisted entirely of repeat calls, and the model stopped with an empty response instead of an answer. The task likely needs to be narrowed or rephrased so it doesn't require repeating the same information-gathering calls.`))
          : "";
        // Explicitly failed: an empty/no-answer result is NOT a real
        // completion regardless of finishReason -- finishRun defaults
        // failed to false unless overridden, which previously let a run
        // that produced literally no usable content (confirmed live,
        // 2026-09-02, run 18ed8a96: finishReason "stop", empty text, no
        // timeout, no leakage) get recorded and returned as a success. An
        // honest empty answer is preferable to fabricated/garbled output,
        // but it is still not a result the caller should treat as done.
        return finishRun({ answer: `(Gemini stopped without a final answer -- finishReason: ${candidate.finishReason || "unknown"})${starvationNote}`, steps: step, transcript, runId, task: effectiveTask, failed: true });
      }
      return finishRun({ answer, steps: step, transcript, runId, task: effectiveTask });
    }

    // Record the model's turn (including its functionCall parts) before
    // executing anything, so the conversation history stays accurate even
    // if a function call below throws.
    contents.push({ role: "model", parts });

    const responseParts = [];
    try {
      // PARALLELIZED (2026-07-26, confirmed via live show_transcript testing
      // that Gemini routinely batches several independent calls into one
      // turn -- e.g. file tree + commit list + issue list all landing in the
      // same step). These calls were previously await'd one at a time in a
      // for-loop for no real reason: within a single turn, Gemini already
      // committed to every one of these calls before seeing ANY of their
      // results, so none of them can depend on another's output -- executing
      // them concurrently changes wall-clock time only, not what information
      // is available to what call. Cross-step sequencing (the real
      // plan->act->observe->re-plan loop) is untouched: that dependency
      // chain lives between steps, not within one.
      //
      // Results are collected here and then pushed to transcript/
      // responseParts below in ORIGINAL (input) order, not completion order --
      // so the transcript and the conversation history sent back to Gemini
      // are byte-for-byte the same shape they'd be under sequential
      // execution, just produced faster. functionResponse.id (not array
      // position) is what actually threads each result back to its call on
      // Gemini's side, so reordering here would be safe even without this,
      // but keeping input order makes the transcript's own readability not
      // regress either.
      //
      // NOTE ON "BLIND" BATCHING: calls sharing a step number are, by
      // definition, decided without seeing each other's results -- that was
      // true before this change too (sequential execution didn't feed call
      // N's result to call N+1's args; Gemini had already written both calls
      // in the same turn). This just makes that pre-existing fact match the
      // wall-clock reality instead of an execution order that only
      // coincidentally looked sequential.
      //
      // RATE-LIMIT NOTE: connectors/github/client.js has its own burst-safe
      // throttle queue (scheduleThrottled) specifically built to absorb
      // concurrent GitHub calls, so parallelizing those is fully safe.
      // Notion (connectors/notion/client.js) and Mem0 (connectors/mem/
      // client.js) have no equivalent throttle/retry/backoff -- a step that
      // batches several Notion or Mem0 calls together is now more likely to
      // trip those APIs' own rate limits than under sequential execution.
      // Not a correctness risk (every call below is already individually
      // try/caught into an error string, same as before), just a new-ish
      // source of noisier per-call failures under heavier batching that's
      // worth watching for in practice rather than something this change
      // guards against.
      // Fix for the 2026-08-31 resultcache-not-persisted-on-resume bug (see
      // this loop's own comment above): a fresh runInvestigation invocation's
      // in-memory `resultCache` Map starts empty regardless of whether this
      // is a resume, but `repeatCounts` IS restored from the
      // checkpoint -- so before executing anything below, identify every
      // signature this step's calls need that repeatCounts already knows is
      // a repeat but the local Map doesn't have text for yet, and fetch all
      // of them from the Redis side-store in ONE round trip (MGET) rather
      // than a per-call fetch inside the Promise.all below. A no-op (empty
      // fetch list) on every step of a fresh, never-resumed run, and on any
      // step of a resumed run whose repeats were already re-fetched earlier
      // in the same invocation.
      if (runId) {
        const missingCachedSignatures = [];
        for (const part of functionCalls) {
          const { name, args } = part.functionCall;
          const signature = normalizedSignature(name, args);
          if (repeatCounts.has(signature) && !resultCache.has(signature)) {
            missingCachedSignatures.push(signature);
          }
        }
        if (missingCachedSignatures.length) {
          const fetched = await getResultCacheEntries(runId, missingCachedSignatures);
          for (const [sig, text] of fetched) resultCache.set(sig, text);
        }
      }

      // Side-store writes for this step's freshly-computed (non-cached)
      // results, collected here and awaited once after the Promise.all below
      // -- same pattern compactHistoryInPlace uses for its own side-store
      // writes, so a synchronous mutation to resultCache/repeatCounts inside
      // the map callback below never waits on a Redis round trip before the
      // next call in the same step can proceed.
      const resultCacheWrites = [];

      const results = await Promise.all(functionCalls.map(async (part) => {
        const { name, args, id } = part.functionCall;
        // Stuck-loop detection (fix #4): a signature identifies an exact
        // repeat of a call already made this run. `isRepeat` reflects
        // whether this signature has been SEEN before (checked before the
        // increment below); repeatCounts itself is incremented regardless
        // of whether it's a repeat, purely for observability/debugging --
        // only the boolean matters to the stuck-loop logic further down.
        const signature = normalizedSignature(name, args);
        const isRepeat = repeatCounts.has(signature);
        repeatCounts.set(signature, (repeatCounts.get(signature) || 0) + 1);

        let resultText;
        let servedFromCache = false;
        if (isRepeat && resultCache.has(signature)) {
          // Exact repeat -- don't re-execute at all, just return what this
          // same call returned last time. This is the free win: no network
          // call, no wasted budget, regardless of whether the run as a
          // whole turns out to be stuck (see allRepeatsThisStep below). On a
          // resumed/singleStep invocation, resultCache.has(signature) being
          // true here means the pre-pass above already fetched it from the
          // side-store -- this branch doesn't need to know or care whether
          // the hit came from this same in-memory run or a prior invocation.
          resultText = resultCache.get(signature);
          servedFromCache = true;
        } else {
          const fn = FUNCTIONS.find((f) => f.name === name);
          if (!fn) {
            resultText = `Error: unknown function "${name}".`;
          } else {
            const validationError = validateFunctionArgs(fn, args || {});
            if (validationError) {
              // Caught before execute() ever runs -- no network/API call spent
              // on a request that was already known to be malformed, and the
              // model gets a specific, actionable correction fed back as this
              // call's result instead of a silently-wrong or empty answer.
              resultText = validationError;
            } else {
              try {
                resultText = await fn.execute(args || {});
              } catch (err) {
                resultText = `Error: ${err?.message ?? String(err)}`;
              }
            }
          }
          // Defensive: every FUNCTIONS[].execute() is expected to return a
          // string. Guard against a future one accidentally returning
          // something else (object, undefined, etc.) so this can't throw
          // mid-transcript and take down the whole step -- see the outer
          // catch below for why that matters.
          if (typeof resultText !== "string") {
            resultText = `Error: ${name} returned a non-string result (${typeof resultText}); this is a bug in the function's execute().`;
          }
          resultCache.set(signature, resultText);
          // Persist this fresh result to the side-store too (fix for the
          // 2026-08-31 bug) -- without this, a resumed/singleStep invocation
          // later in the SAME run would still have nothing to fetch in the
          // pre-pass above, and would silently re-execute the repeat exactly
          // as before this fix.
          if (runId) resultCacheWrites.push(saveResultCacheEntry(runId, signature, resultText));
        }
        return { name, args, id, resultText, isRepeat, servedFromCache };
      }));
      if (resultCacheWrites.length) await Promise.all(resultCacheWrites);

      // Oversized-step guardrail #2 (aggregate size cap, see
      // MAX_STEP_RESULT_CHARS's definition above): even with the count cap
      // above, a handful of large truncated file reads can still add up to
      // a huge combined payload for this one step. This caps only what
      // actually gets appended to `contents`/responseParts (the outbound
      // context sent to the LLM on the NEXT call) -- resultCache/the Redis
      // side-store already have the FULL text from the execution above, and
      // the transcript's own 300-char preview below is untouched, so
      // nothing is lost: a truncated call can be re-requested (with
      // char_offset if applicable) in a later step. Applied in array order,
      // which is call order within the step, so earlier calls in a step
      // keep priority over later ones once the cap is hit.
      if (applyOversizedStepCaps) {
        let stepResultCharsUsed = 0;
        for (const r of results) {
          if (stepResultCharsUsed >= MAX_STEP_RESULT_CHARS) {
            r.resultText = `[Result withheld -- this step's combined tool-result size already reached the ${MAX_STEP_RESULT_CHARS}-char per-step cap from earlier calls in the same step. Re-request this specific call in your next step.]`;
          } else if (stepResultCharsUsed + r.resultText.length > MAX_STEP_RESULT_CHARS) {
            const allowed = MAX_STEP_RESULT_CHARS - stepResultCharsUsed;
            r.resultText = `${r.resultText.slice(0, allowed)}\n...[truncated -- this step's combined tool-result cap of ${MAX_STEP_RESULT_CHARS} chars was reached; re-request the remainder (e.g. via char_offset) in a future step]`;
            stepResultCharsUsed += allowed;
          } else {
            stepResultCharsUsed += r.resultText.length;
          }
        }
      }
      // Gemini (applyOversizedStepCaps === false): no per-step result-size cap --
      // results are appended to contents/responseParts at their full size, exactly
      // as before this session's bai-specific fixes.

      for (const r of results) {
        const cacheNote = r.servedFromCache ? " [CACHED -- identical call already made this run, not re-executed]" : "";
        transcript.push(`[step ${step}] ${r.name}(${JSON.stringify(r.args || {})})${cacheNote} -> ${r.resultText.length > 300 ? r.resultText.slice(0, 300) + "…" : r.resultText}`);
        // Gemini 3 (current generateContent contract, verified 2026-07-25): function-result
        // turns go back with role "user" (NOT "function" -- that was the older doc convention
        // and is rejected by Gemini 3 models), and functionResponse.id echoes the model's
        // original functionCall.id so the API can thread multi-call turns correctly.
        responseParts.push({ functionResponse: { name: r.name, id: r.id, response: { result: r.resultText } } });
      }

      // Every call the model batched into this turn -- including the ones
      // this step chose not to execute (deferredFunctionCalls, see the
      // count-cap comment above) -- still needs a functionResponse or the
      // next outbound call to bai/Gemini will reject the malformed turn.
      // These are synthetic (no execute(), no cache, not counted toward
      // repeatCounts/stuck-loop detection) and deliberately short, so they
      // add negligible size back to the very payload this guardrail exists
      // to bound.
      for (const part of deferredFunctionCalls) {
        const { name, args, id } = part.functionCall;
        const resultText = `Error: step call limit reached (max ${MAX_TOOL_CALLS_PER_STEP} tool calls per step). This call was not executed -- request it again in a future step. Batching fewer, more targeted calls per step keeps each step fast and avoids platform timeouts.`;
        transcript.push(`[step ${step}] ${name}(${JSON.stringify(args || {})}) [DEFERRED -- step call limit reached, not executed]`);
        responseParts.push({ functionResponse: { name, id, response: { result: resultText } } });
      }

      // Stuck-loop bookkeeping (fix #4): only counts as a stuck step if
      // EVERY call this step was an exact repeat -- see isRepeat's comment
      // above for why a partial repeat doesn't count.
      const allRepeatsThisStep = results.length > 0 && results.every((r) => r.isRepeat);
      consecutiveAllRepeatSteps = allRepeatsThisStep ? consecutiveAllRepeatSteps + 1 : 0;
      if (consecutiveAllRepeatSteps === 2) {
        // Earlier, softer nudge -- same two-steps-ahead pattern as the
        // step-budget reminder below, giving the model a chance to steer
        // away before the hard stop one step down.
        responseParts.push({
          text: `[SYSTEM NOTE: you're re-requesting information you already have -- the last 2 steps consisted entirely of repeat calls (same function + arguments as something already tried this run). Either try a different angle (a different file, query, or function) or answer now with what you've got.]`,
        });
      } else if (consecutiveAllRepeatSteps >= 3) {
        // Matches reality: withholdTools (computed at the top of the loop)
        // will be true next iteration because consecutiveAllRepeatSteps >= 3
        // here, so the next turn genuinely won't have tools available.
        responseParts.push({
          text: `[SYSTEM NOTE: 3 consecutive steps have consisted entirely of repeat calls. The next turn will NOT include any tools -- you must answer now in plain text with whatever you've already found, since repeating the same calls further will not surface new information.]`,
        });
      }
    } catch (err) {
      // Belt-and-suspenders: nothing inside the loop above should throw past
      // its own per-call try/catch or the typeof guard anymore, but if
      // something still does (a bug in a future function, an unexpected
      // JSON.stringify(args) failure on a circular/exotic args shape, etc.),
      // don't let it escape runInvestigation and land in tools.js's generic
      // catch, which has no runId to offer -- that would silently lose this
      // step's (and any prior steps') completed work. Checkpoint what's
      // already done (this step's model turn was already pushed to
      // `contents` above) and return the same resumable-failure shape as a
      // geminiChat failure.
      await saveCheckpoint(runId, {
        newContents: contents.slice(contentsCheckpointedUpTo),
        transcript,
        stepsDone: step - 1,
        task: effectiveTask,
        repeatCounts: Object.fromEntries(repeatCounts),
        preCompactionResults: Object.fromEntries(preCompactionResults),
        resultCache: Object.fromEntries(resultCache),
        consecutiveAllRepeatSteps,
        provider: effectiveProvider,
        model: effectiveModel,
        maxOutputTokens: effectiveMaxOutputTokens,
        preambleVariant: effectivePreambleVariant,
        pendingVerification,
        overallMaxSteps: effectiveOverallMaxSteps,
      });
      const errMessage = err?.message ?? String(err);
      return {
        answer: `(Unexpected error while processing step ${step}'s function calls: ${errMessage} -- ${transcript.length} tool call(s) already completed this run are saved. Call delegate_agent again with resume_run_id: "${runId}" to continue from here instead of starting over. Checkpoint expires in 1 hour.)`,
        steps: step - 1,
        transcript,
        runId,
        task: effectiveTask,
        failed: true,
      };
    }
    // Step-budget reminder (added after the 2026-07-26 resume-truncation
    // bug): SYSTEM_PREAMBLE and the task's own formatting instructions only
    // ever appear once, in turn 1 -- by the last couple of steps before
    // cappedSteps, those instructions are many turns back in a long tool-use
    // history, and a model under a tight remaining budget has an incentive
    // to produce SOME answer rather than none, which can mean quietly
    // dropping the originally requested format/exhaustiveness. Surfacing the
    // remaining-step count explicitly turns a silent quality regression into
    // an honest one: the model is told to say it couldn't finish, rather
    // than presenting a rushed, incomplete answer as if it were complete.
    const remainingAfterThisStep = effectiveOverallMaxSteps - step;
    if (remainingAfterThisStep === 2) {
      // Earlier, softer nudge -- gives the model a chance to steer toward
      // synthesis before the hard cutoff two notes down, instead of only
      // finding out at the last possible moment.
      responseParts.push({
        text: `[SYSTEM NOTE: only 2 step(s) remain after this one. Start wrapping up -- prioritize synthesizing what you've already found over opening new lines of investigation.]`,
      });
    } else if (remainingAfterThisStep <= 1) {
      // When remainingAfterThisStep is 0, the NEXT turn is the final step,
      // which is called with no tools at all (see isFinalStep above) -- so
      // this note can say so as a fact, not just a suggestion to wrap up.
      // Output-generation-timeout fix (plan.md Section 11/15, prompt-side
      // half): this final-step note used to only say "describe what's
      // missing, rather than presenting a partial answer as if it were
      // complete" -- true, but easy to misread as "don't be brief", which is
      // backwards. A model asked for an exhaustive answer (every file, every
      // field, a full line-by-line comparison) with no tools left has every
      // incentive to attempt the full thing anyway rather than admit it
      // can't -- and on this step there is no next step to fall back to, no
      // maxOutputTokens cap set by default (router.js, both gemini and bai),
      // and generation time alone can exceed Vercel's 300s hard function
      // timeout with nothing salvaged (confirmed live twice: plan.md
      // Section 11, Section 15's runId 124a76f8, both bai).
      // The added sentences make the SHORT-answer instruction explicit and
      // separate from the existing don't-pretend-it's-complete instruction,
      // which stays as-is -- those are two different asks (be honest about
      // gaps; also don't try to out-write your budget) that were previously
      // collapsed into one sentence a model could satisfy by writing a long,
      // honest, but still unbounded answer.
      // SIMPLIFIED (2026-09-02, reverting most of Sections 16/18/20/21's
      // bai-specific elaboration): three successive rounds of adding a
      // brevity instruction, then increasingly detailed anti-leakage
      // wording (XML tags -> brackets -> delimiter-free examples)
      // correlated with the leakage MUTATING into a new syntax variant each
      // time, not stopping -- while Gemini, which has only ever gotten this
      // same short, example-free sentence below, has never leaked once.
      // Hypothesis: the elaboration itself -- length, and especially citing
      // literal example tool-call syntax right before a turn where the
      // model can't call a tool -- may have been priming the exact
      // completion pattern it was meant to suppress, on a small/fast model
      // with weaker instruction-following under a dense multi-clause
      // final-turn prompt. Testing the plainest possible instruction first
      // rather than continuing to whack-a-mole new syntax shapes. If
      // platform-timeout (Section 11/15) or leakage recurs without the
      // extra wording, that's real evidence the elaboration was earning its
      // keep and this should be revisited -- not assumed either way without
      // a fresh live test.
      const noToolsNote = remainingAfterThisStep === 0
        ? " The next turn will NOT include any tools -- a function call is not possible; you must answer in plain text now, not describe what you would fetch or do next."
        : "";
      responseParts.push({
        text: `[SYSTEM NOTE: only ${remainingAfterThisStep} step(s) remain before this investigation is forced to stop.${noToolsNote} Separately from the length of your response: if you cannot fully complete the task -- including any specific format requested -- say so explicitly and describe what's missing, rather than presenting a partial or reformatted answer as if it were complete. Before you write your verdict, scroll back through the raw content you already fetched this run (not just your impression of it) and confirm nothing you retrieved contradicts what you're about to claim -- a contradiction sitting unused in your own transcript is a miss, not a non-finding.]`,
      });
    }

    // Compact older bulky tool results in contents before appending this step's turns
    // (or before saving checkpoint / sending next turn), only if provider opted in.
    // We pass a preCompactionResults map to store text that gets compacted,
    // so it remains available for findUnverifiedClaims. Passing runId also lets
    // compactHistoryInPlace side-store each newly-compacted result's full text in
    // Redis (addressing state-checkpoint bloat), not just this in-memory Map
    // -- awaited so a checkpoint saved right after this line never races ahead
    // of the side-store writes it corresponds to.
    await compactHistoryInPlace(contents, step, preCompactionResults, { provider: effectiveProvider, runId });

    contents.push({ role: "user", parts: responseParts });

    // Checkpoint after every fully-completed step, so a failure on the NEXT
    // Gemini call (or a hosting-platform timeout) doesn't lose this one.
    // newContents/contentsCheckpointedUpTo implement fix #5 (append-delta
    // instead of overwrite-whole-blob): only the turns added THIS step (the
    // model's turn + the function-response turn, normally 2 entries) are
    // pushed, not the whole conversation so far -- write cost is O(delta),
    // not O(total run length).
    await saveCheckpoint(runId, {
      newContents: contents.slice(contentsCheckpointedUpTo),
      transcript,
      stepsDone: step,
      task: effectiveTask,
      repeatCounts: Object.fromEntries(repeatCounts),
      preCompactionResults: Object.fromEntries(preCompactionResults),
      resultCache: Object.fromEntries(resultCache),
      consecutiveAllRepeatSteps,
      provider: effectiveProvider,
      model: effectiveModel,
      maxOutputTokens: effectiveMaxOutputTokens,
      preambleVariant: effectivePreambleVariant,
      // Always false here in practice: this checkpoint fires only after a
      // step that made function calls, and a verification-pass turn never
      // reaches this branch (withholdTools forces it to text-only, so it
      // either returns from the !functionCalls.length branch above or, on
      // the MALFORMED_FUNCTION_CALL edge case, returns early with an
      // error) -- included explicitly so the persisted checkpoint always
      // states this field rather than silently omitting it on this path.
      pendingVerification,
      overallMaxSteps: effectiveOverallMaxSteps,
    });
    contentsCheckpointedUpTo = contents.length;
  }

  // Async delegate_agent (groundwork): reaching THIS caller's
  // max_steps ceiling is not necessarily the end of the run -- a caller
  // (notably the QStash worker, which deliberately calls with
  // max_steps: stepsDone + 1 to take exactly one step per invocation, see
  // agent_worker.js) may come back with a higher ceiling, or resume the
  // same runId again, and expects the checkpoint to still be there. The
  // per-step saveCheckpoint call above already left status "running" with a
  // fresh lastStepAt for this exact case, so there's nothing to write here
  // -- ONLY delete/finalize when cappedSteps has hit HARD_MAX_STEPS, since
  // that's a real ceiling no future call can ever raise (max_steps is
  // clamped to it unconditionally at the top of this function), so nothing
  // would ever be gained by keeping the checkpoint around for that case.
  // Previously this unconditionally deleted the checkpoint on EVERY
  // max_steps exhaustion, silently discarding real progress the moment a
  // caller happened to under-budget a call -- not just a Scenario B
  // prerequisite, an existing gap this also fixes.
  if (cappedSteps >= HARD_MAX_STEPS) {
    const result = {
      answer: `(Investigation stopped after reaching the hard step cap of ${HARD_MAX_STEPS} without a final answer -- the task likely needs to be narrowed.)`,
      steps: cappedSteps, transcript, runId, task: effectiveTask,
    };
    // Same finishRun shape as the mid-loop completion path above (that one
    // is a locally-scoped closure inside the for-loop, out of reach here --
    // this is a deliberate small duplication rather than hoisting it, since
    // hoisting would require threading `step`-scoped locals through as
    // params for a call site used exactly once).
    await saveCheckpoint(runId, {
      newContents: [], transcript, stepsDone: cappedSteps, task: effectiveTask,
      repeatCounts: Object.fromEntries(repeatCounts), preCompactionResults: Object.fromEntries(preCompactionResults),
      resultCache: Object.fromEntries(resultCache),
      consecutiveAllRepeatSteps,
      provider: effectiveProvider, model: effectiveModel, maxOutputTokens: effectiveMaxOutputTokens, preambleVariant: effectivePreambleVariant,
      pendingVerification, structuralRecheckUsed, overallMaxSteps: effectiveOverallMaxSteps, status: "done", finalAnswer: result.answer,
    });
    return result;
  }
  return {
    answer: `(Investigation stopped after reaching the requested max_steps of ${cappedSteps} without a final answer -- the task is not finished. The checkpoint has NOT been discarded: call delegate_agent again with resume_run_id: "${runId}" and a higher max_steps (up to the hard cap of ${HARD_MAX_STEPS}) to continue.)`,
    steps: cappedSteps, transcript, runId, task: effectiveTask, failed: true,
  };
}

// Seeds a fresh checkpoint (status "running", stepsDone 0, no steps taken
// yet) WITHOUT running any part of the investigation loop -- used by
// agent_tools.js's async-start path (Scenario B) to return a runId
// immediately and let the QStash worker (agent_worker.js) take step 1 in
// the background, rather than this call itself blocking on step 1
// synchronously before returning (which would defeat the "returns almost
// immediately" goal for a task whose very first step is itself slow).
//
// Deliberately duplicates the small fresh-run setup at the top of
// runInvestigation (a UUID + the initial SYSTEM_PREAMBLE/task turn) rather
// than calling into runInvestigation with max_steps: 0 -- runInvestigation's
// loop (`for (step = startStep; step <= cappedSteps; ...)`) does simply
// never execute when cappedSteps < startStep, which looks like it would
// work for a zero-step call, but the ONLY existing early-return path that
// covers cappedSteps < startStep (the `checkpoint && startStep > cappedSteps`
// guard) assumes a checkpoint ALREADY EXISTS to read runId/contents/
// transcript back off of -- reaching it on a genuinely fresh call (no
// resume_run_id, no checkpoint yet) would require either throwing (no task-
// less-resume fallback path exists for a non-resume, zero-step call) or
// restructuring that guard to cover a second, differently-shaped caller. A
// small, explicit duplication of the ~4-line fresh-run setup here is lower-
// risk than bending that guard's contract to serve both callers.
export async function seedRun({ task, provider, model, maxOutputTokens, max_steps = 20, preambleVariant = "verbose" }) {
  const runId = randomUUID();
  const contents = [{ role: "user", parts: [{ text: `${buildSystemPreamble(provider, preambleVariant)}\n\nTask: ${task}` }] }];
  // Seeds the run's TRUE overall step ceiling (see runInvestigation's
  // effectiveOverallMaxSteps for the full rationale) -- this is what lets
  // the QStash worker's later singleStep resumes (agent_worker.js) know
  // when they've actually reached the run's real last step, instead of
  // mistaking their own artificially-shrunk per-call max_steps for it.
  const overallMaxSteps = Math.min(max_steps, HARD_MAX_STEPS);
  await saveCheckpoint(runId, {
    newContents: contents,
    transcript: [],
    stepsDone: 0,
    task,
    repeatCounts: {},
    preCompactionResults: {},
    resultCache: {},
    consecutiveAllRepeatSteps: 0,
    provider,
    model,
    maxOutputTokens,
    preambleVariant,
    pendingVerification: false,
    structuralRecheckUsed: false,
    overallMaxSteps,
    status: "running",
  });
  return runId;
}
