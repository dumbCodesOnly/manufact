// ---------------------------------------------------------------------------
// connectors/delegate/agent/agent_checkpoint.js — Redis-backed checkpointing for
// delegate_agent's multi-step loop, so a run that dies partway through
// (Gemini 503/429, network blip, function timeout) doesn't lose every tool
// call it already made.
//
// STORAGE SHAPE (fix #5, 2026-07-27 -- append-delta instead of overwrite-
// whole-blob): the conversation `contents` array is the part of loop state
// that grows every step and can get large (tool outputs up to ~30k chars
// each) -- it lives in its own Redis LIST, and callers only ever RPUSH the
// turns added since the last checkpoint (see saveCheckpoint's `newContents`
// param), not the whole array. Write cost is therefore O(delta per step),
// not O(total conversation so far). Everything else (transcript, stepsDone,
// task, and fix #4's repeat-signature tracking state) stays small and cheap
// regardless of run length, so it's kept as one JSON blob under a separate
// key -- no benefit to splitting that up further.
//
// SAME FAIL-OPEN CONTRACT AS cooldown.js: if Redis isn't configured or a
// call fails, every function here no-ops / returns null. A missing Redis
// must never be the reason an investigation can't run -- it only means a
// failure can't be resumed, same as before this file existed.
// ---------------------------------------------------------------------------

import { getRedis } from "../../shared/cooldown.js";

const CHECKPOINT_KEY_PREFIX = "gemini:checkpoint:";
// A checkpoint only needs to survive long enough for the caller to retry
// with resume_run_id -- not to become a permanent store.
const CHECKPOINT_TTL_SECONDS = 3600;

function contentsKey(runId) {
  return `${CHECKPOINT_KEY_PREFIX}${runId}:contents`;
}
function metaKey(runId) {
  return `${CHECKPOINT_KEY_PREFIX}${runId}:meta`;
}
// Deliberately NOT under CHECKPOINT_KEY_PREFIX -- kept as its own top-level
// `precompact:` namespace, so a future GC pass can batch-delete
// `precompact:{runId}:*` independently of contents/meta.
function precompactKey(runId, id) {
  return `precompact:${runId}:${id}`;
}
// Same reasoning as precompactKey above, mirrored for resultCache (fix for
// the 2026-08-31 resultcache-not-persisted-on-resume bug -- see
// saveResultCacheEntry/getResultCacheEntries below). Its own
// top-level `resultcache:` namespace, independent of both
// CHECKPOINT_KEY_PREFIX and precompactKey's own namespace, so a GC pass can
// batch-delete `resultcache:{runId}:*` on its own.
function resultCacheKey(runId, signature) {
  return `resultcache:${runId}:${signature}`;
}

// Persists loop state after a step completes:
//   - newContents: ONLY the turn(s) added to `contents` since the last
//     saveCheckpoint call for this runId (may be an empty array -- e.g. a
//     geminiChat failure that happens before any new turn was pushed --
//     in which case the list simply isn't touched this call, only meta is).
//     The caller (agent_delegate.js) is responsible for tracking which slice of
//     its in-memory `contents` array is new; this function has no way to
//     know that on its own since it never sees the full array.
//   - transcript/stepsDone/task/repeatCounts/consecutiveAllRepeatSteps/preCompactionResults: the
//     small stuff, always written in full (cheap regardless of run length).
//   - status/lastStepAt (added for async delegate_agent work,
//     Scenario A `waitUntil` and Scenario B QStash self-chaining -- both
//     reuse this same meta shape): `status` is one of "running" | "done" |
//     "failed", defaulting to "running" when omitted so every existing
//     synchronous call site keeps working unchanged. `lastStepAt` is an
//     epoch-ms timestamp of THIS save, always set here (not left to the
//     caller) so every checkpoint write freshens it -- this is what lets
//     delegate_agent's poll path in agent_tools.js tell a genuinely-still-
//     running background worker apart from one whose chain silently died
//     (stale lastStepAt -> fall back to synchronous resume instead of
//     polling forever). Do not compute lastStepAt in a caller instead of
//     here -- a caller-supplied value could be stale by the time the actual
//     Redis write lands, defeating the freshness check.
// Fails open -- never throws.
export async function saveCheckpoint(runId, { newContents = [], transcript, stepsDone, task, repeatCounts, consecutiveAllRepeatSteps, preCompactionResults, resultCache, provider, model, maxOutputTokens, preambleVariant, pendingVerification, structuralRecheckUsed, overallMaxSteps, status = "running", finalAnswer, stepStartedAt }) {
  const client = getRedis();
  if (!client) return;
  try {
    const ops = [];
    if (newContents.length) {
      ops.push(client.rpush(contentsKey(runId), ...newContents.map((c) => JSON.stringify(c))));
      // EXPIRE (not a per-SET `ex` option, since RPUSH has no TTL param of
      // its own) re-armed on every push so the list's TTL tracks the meta
      // key's, rather than being set once and left to whatever it was at
      // list-creation time.
      ops.push(client.expire(contentsKey(runId), CHECKPOINT_TTL_SECONDS));
    }
    // finalAnswer (added alongside status/lastStepAt for async
    // delegate_agent work): only ever set by agent_delegate.js's completion
    // paths, once a run is genuinely done -- this is what lets a
    // resume_run_id poll of a "done" checkpoint (agent_delegate.js's
    // short-circuit right after loadCheckpoint) return the actual answer
    // instead of just a status flag with nothing to show for it. Always
    // included in the meta blob (even as undefined on every "running" save)
    // rather than conditionally spread in, so the shape of a saved
    // checkpoint doesn't vary step-to-step -- consistent with every other
    // field here.
    // Step 3 fix (addressing state-checkpoint bloat): store only the IDS
    // that have a side-store entry, not the text -- see savePreCompactionResult
    // below, and agent_delegate.js's resume-restore comment for why an
    // empty-of-text Map on load loses nothing (compactHistoryInPlace
    // re-derives every still-aged-out id's real text straight from the
    // pristine `contents` loadCheckpoint returns, before anything else runs).
    // This is still O(total ids compacted so far), same as every other field
    // in this blob (meta is rewritten in full every call) -- the win is a far
    // smaller per-entry cost (one id vs. up to tens of thousands of chars of
    // text), not a change in growth order.
    const preCompactionResultIds = Array.isArray(preCompactionResults)
      ? preCompactionResults
      : (preCompactionResults instanceof Map
          ? [...preCompactionResults.keys()]
          : Object.keys(preCompactionResults || {}));
    // Fix for the 2026-08-31 resultcache-not-persisted-on-resume bug: ids
    // only, same rationale as preCompactionResultIds above -- the actual
    // result text lives in its own side-store key (saveResultCacheEntry),
    // never inlined here. This list exists purely so deleteCheckpoint can
    // GC every resultcache:{runId}:* key a run wrote; nothing restores
    // in-memory resultCache state from this list on load (see
    // agent_delegate.js's lazy fetch-on-demand for why that's unnecessary).
    const resultCacheIds = Array.isArray(resultCache)
      ? resultCache
      : (resultCache instanceof Map
          ? [...resultCache.keys()]
          : Object.keys(resultCache || {}));
    const meta = JSON.stringify({ transcript, stepsDone, task, repeatCounts, consecutiveAllRepeatSteps, preCompactionResultIds, resultCacheIds, provider, model, maxOutputTokens, preambleVariant, pendingVerification, structuralRecheckUsed, overallMaxSteps, status, finalAnswer, stepStartedAt: stepStartedAt ?? null, lastStepAt: Date.now() });
    ops.push(client.set(metaKey(runId), meta, { ex: CHECKPOINT_TTL_SECONDS }));
    await Promise.all(ops);
  } catch {
    // best-effort -- see file header
  }
}

// Loads a previously saved checkpoint, or null if missing/expired/Redis is
// unavailable/either stored value doesn't parse. Reconstructs `contents` by
// concatenating every entry in the list (LRANGE 0 -1) -- this is the one
// place read cost is still O(total run length), but it only happens once
// per resume, not once per step (see file header).
//
// A genuine exception here (network blip, malformed JSON, etc.) is logged
// as a warning before returning null -- distinct from the ordinary "key
// doesn't exist" case (empty list / null meta), which is expected and
// silent. Both cases still return null to the caller (agent_delegate.js can't do
// anything different with either -- see its header), so this doesn't
// change behavior, only observability: without it, a Redis outage and an
// expired checkpoint look identical in the logs.
export async function loadCheckpoint(runId) {
  const client = getRedis();
  if (!client) return null;
  try {
    const [rawList, rawMeta] = await Promise.all([
      client.lrange(contentsKey(runId), 0, -1),
      client.get(metaKey(runId)),
    ]);
    // Meta missing means there's nothing usable here at all (expired, never
    // existed, or a partial/corrupted write) -- same as the old single-key
    // "raw == null" check.
    //
    // rawList (contents) being EMPTY, on the other hand, is no longer on its
    // own a sign of "nothing to resume" -- it used to be, back when every
    // checkpoint write was mid-loop and therefore always had at least one
    // turn already pushed. Two legitimate cases now produce an empty list
    // alongside real meta: (a) a "done" checkpoint (see agent_delegate.js's
    // finishRun/hard-cap-finalize paths, added for async
    // delegate_agent groundwork), which deliberately skips re-pushing
    // `contents` since nothing reads it once a run is finished -- only
    // finalAnswer/steps/transcript/task matter for a poll; and (b) a task
    // that's answered directly on step 1 with zero tool calls, which never
    // pushes a functionResponse turn at all before finishing. Neither case
    // is missing/expired/corrupted -- meta alone is the source of truth for
    // whether a checkpoint exists; `contents` is reconstructed as whatever
    // was actually saved (possibly empty), not treated as a required field.
    if (rawMeta == null) return null;
    // Upstash's client auto-parses JSON-looking values in some SDK versions
    // and returns a raw string in others -- guard both, same as the old
    // single-key version did.
    const contents = (rawList || []).map((entry) => (typeof entry === "string" ? JSON.parse(entry) : entry));
    const meta = typeof rawMeta === "string" ? JSON.parse(rawMeta) : rawMeta;
    return { contents, ...meta };
  } catch (err) {
    console.warn(`loadCheckpoint(${runId}) failed -- treating as no checkpoint:`, err?.message ?? err);
    return null;
  }
}

// Side-store for a single compacted-away tool result's full original text
// (fix for the preCompactionResults checkpoint-bloat issue). saveCheckpoint's
// `meta` blob is rewritten in full on every call (unlike `contents`, which is
// append-delta via RPUSH -- see this file's header), so storing every
// compacted result's full text inside `preCompactionResults` there means
// unbounded per-step write cost on exactly the long `bai` runs history
// compaction targets.
// This writes the text ONCE, to its own key, the first time a given id is
// compacted -- compactHistoryInPlace (agent_delegate.js) is responsible for
// only calling this on first-time compaction of an id, not on every step
// that id happens to still be in the aged-out window.
// Fails open -- never throws, same contract as every other function here.
export async function savePreCompactionResult(runId, id, text) {
  const client = getRedis();
  if (!client) return;
  try {
    await client.set(precompactKey(runId, id), text, { ex: CHECKPOINT_TTL_SECONDS });
  } catch {
    // best-effort -- see file header
  }
}

// Batched fetch-on-demand for compacted results' full text.
// findUnverifiedClaims and lineIsVerbatimInToolResults (agent_delegate.js)
// read primarily from the in-memory `preCompactionResults` Map, which the
// normal compact-then-verify-same-run flow keeps fully populated -- this is
// the side-store fallback for the id(s) that Map doesn't have (e.g. state
// reconstructed from a checkpoint's `preCompactionResultIds` without a
// preceding `compactHistoryInPlace` recompaction pass).
// Takes the full list of ids a verification pass needs and does ONE round
// trip (MGET) for all of them, not a GET per id -- a naive per-id fetch is
// a real added latency cost on exactly the long `bai` runs compacting 200+
// results, which is what this feature targets in the first place.
// Fails open -- returns an empty Map on any error or missing Redis, same
// contract as every other function in this file. Never throws.
export async function getPreCompactionResults(runId, ids) {
  const idList = [...new Set((ids || []).filter(Boolean))];
  const found = new Map();
  if (!idList.length) return found;
  const client = getRedis();
  if (!client) return found;
  try {
    const values = await client.mget(...idList.map((id) => precompactKey(runId, id)));
    idList.forEach((id, i) => {
      if (values[i] != null) found.set(id, values[i]);
    });
  } catch {
    // best-effort -- see file header
  }
  return found;
}

// Side-store for a single tool call's cached result text, keyed by the
// same normalized-call signature agent_delegate.js's resultCache Map uses
// in memory (fix for the 2026-08-31 resultcache-not-persisted-on-resume
// bug). Mirrors savePreCompactionResult immediately above in shape and
// contract (write once, on first computation of a given key; fails open;
// never throws), but keyed by call signature instead of a functionResponse
// id, and written on EVERY fresh (non-cached) tool call agent_delegate.js
// executes -- not just ones that later get compacted -- since any of them
// could turn out to be repeated across a resume/step boundary.
export async function saveResultCacheEntry(runId, signature, text) {
  const client = getRedis();
  if (!client) return;
  try {
    await client.set(resultCacheKey(runId, signature), text, { ex: CHECKPOINT_TTL_SECONDS });
  } catch {
    // best-effort -- see file header
  }
}

// Batched fetch-on-demand for cached result text (fix for the 2026-08-31
// resultcache-not-persisted-on-resume bug). Mirrors getPreCompactionResults
// immediately above: a fresh runInvestigation invocation's in-memory
// resultCache Map starts empty regardless of whether this is a resume, so
// before executing a step's function calls, agent_delegate.js collects every
// signature that repeatCounts (which IS restored from the checkpoint)
// already knows is a repeat but the local Map doesn't have yet, and fetches
// all of them here in ONE round trip (MGET) rather than one GET per
// signature.
// Fails open -- returns an empty Map on any error or missing Redis, same
// contract as every other function in this file. Never throws.
export async function getResultCacheEntries(runId, signatures) {
  const sigList = [...new Set((signatures || []).filter(Boolean))];
  const found = new Map();
  if (!sigList.length) return found;
  const client = getRedis();
  if (!client) return found;
  try {
    const values = await client.mget(...sigList.map((sig) => resultCacheKey(runId, sig)));
    sigList.forEach((sig, i) => {
      if (values[i] != null) found.set(sig, values[i]);
    });
  } catch {
    // best-effort -- see file header
  }
  return found;
}

// Deletes a checkpoint once a run finishes (a final answer, or the model
// stops issuing function calls) -- nothing left to resume. Clears the
// contents/meta keys, plus every precompact:{runId}:* side-store key
// savePreCompactionResult wrote during the run -- those live outside
// CHECKPOINT_KEY_PREFIX (see precompactKey's comment), so they wouldn't be
// swept up by deleting contents/meta alone and would otherwise just sit
// until their own CHECKPOINT_TTL_SECONDS TTL expires.
// The id list isn't passed in -- it's read back from meta's own
// preCompactionResultIds (fetched before meta is deleted), since that's
// the only record of which ids exist; this deliberately avoids relying on
// Redis KEYS/SCAN (not guaranteed available/cheap on every provider this
// runs against -- same fail-open, provider-agnostic contract as the rest
// of this file).
export async function deleteCheckpoint(runId) {
  const client = getRedis();
  if (!client) return;
  try {
    const rawMeta = await client.get(metaKey(runId));
    const meta = rawMeta ? (typeof rawMeta === "string" ? JSON.parse(rawMeta) : rawMeta) : null;
    const ids = (meta && meta.preCompactionResultIds) || [];
    // Same reasoning as `ids` above, mirrored for resultCache (fix for the
    // 2026-08-31 resultcache-not-persisted-on-resume bug) -- resultCacheIds
    // lives outside CHECKPOINT_KEY_PREFIX in its own `resultcache:`
    // namespace (see resultCacheKey), so it needs its own explicit sweep
    // here too, same as precompact:{runId}:* does.
    const resultCacheIds = (meta && meta.resultCacheIds) || [];
    await Promise.all([
      client.del(contentsKey(runId)),
      client.del(metaKey(runId)),
      ...ids.map((id) => client.del(precompactKey(runId, id))),
      ...resultCacheIds.map((sig) => client.del(resultCacheKey(runId, sig))),
    ]);
  } catch {
    // best-effort
  }
}
