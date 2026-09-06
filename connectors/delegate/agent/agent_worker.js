// ---------------------------------------------------------------------------
// connectors/delegate/agent/agent_worker.js — QStash-invoked HTTP endpoint that
// advances a delegate_agent run one step at a time in the background
// (Scenario B: self-chaining QStash worker).
//
// Reuses runInvestigation's EXISTING synchronous loop one call at a time
// (resume_run_id + singleStep: true) rather than a bespoke single-step
// function extracted from it -- see the "Progress log" entry for
// step 5 for why the literal per-step extraction the design originally
// called for turned out to be unnecessary, and riskier than this reuse,
// given how much interacting fix history lives inside that loop body.
//
// IMPORTANT: this MUST be singleStep: true, not max_steps: stepsDone + 1.
// The latter looks equivalent (both bound this call to exactly one step)
// but is NOT: runInvestigation derives isFinalStep from the run's real
// overall step ceiling, which max_steps: stepsDone + 1 would overwrite
// with this call's own artificially-shrunk bound on every single
// invocation -- withholding tools from every worker-driven step
// regardless of how many the run was actually meant to have (see the
// 2026-08-28 production-bug entry in the progress log). singleStep
// instead restores the true ceiling from the checkpoint (set once, at
// seedRun time) and only bounds THIS call's own loop to one iteration.
//
// SECURITY: this endpoint is PUBLICLY reachable (QStash calls it over the
// open internet to invoke it), unlike the MCP tool surface which sits
// behind requireMcpKey/requireAllowedIp in server.js. Every request's
// signature is verified via qstash_client.js's verifyQStashSignature
// BEFORE any checkpoint is touched -- an unsigned or invalid request is
// rejected outright and never reaches runInvestigation.
//
// IDEMPOTENCY: QStash retries failed deliveries automatically, so the SAME
// {runId, afterStep} message can arrive more than once. `afterStep` records
// the checkpoint's stepsDone at the moment THIS message was published -- if
// the live checkpoint has already moved past that (another invocation of
// the same message already advanced it), this invocation no-ops instead of
// double-executing a Gemini turn for a step that's already done.
//
// DEAD-LETTER / RETRY BOUND (step 8): `retryCount` bounds
// consecutive same-step failures (a step that completes without advancing
// stepsDone -- e.g. a transient 429/503 that runInvestigation's own
// per-step try/catch already turned into a `{ failed: true }` result rather
// than a thrown error) so a permanently-broken run (bad config, a
// non-transient error) can't re-chain into itself forever. After
// AGENT_WORKER_MAX_CONSECUTIVE_FAILURES consecutive failures on the same
// step, the chain stops and the checkpoint is finalized as "failed" with
// the last error as its stored answer, rather than silently retrying at
// QStash's (and this account's Gemini quota's) expense indefinitely.
// ---------------------------------------------------------------------------

import { randomUUID } from "node:crypto";
import { runInvestigation } from "./agent_delegate.js";
import { loadCheckpoint, saveCheckpoint } from "./agent_checkpoint.js";
import { publishAgentStep, verifyQStashSignature } from "../qstash_client.js";
import { AGENT_WORKER_MAX_CONSECUTIVE_FAILURES, DEBUG_AGENT_WORKER } from "../../../config.js";

// Gated wrapper for the agent-worker[<id>]: invocation-id debug logging added
// 2026-09-01 (commit 2aad526) to diagnose a worker-chain stall. Default OFF
// (DEBUG_AGENT_WORKER) once that stall was diagnosed as sustained B.AI
// rate-limiting driving the existing retry/re-chain path, not a
// concurrent-duplicate idempotency bug -- see config.js's DEBUG_AGENT_WORKER
// comment for the full history. Kept, not deleted, so it can be flipped
// back on (DEBUG_AGENT_WORKER=true) quickly if a similar stall resurfaces.
function debugLog(message) {
  if (DEBUG_AGENT_WORKER) console.log(message);
}

// Express handler for POST /api/agent-worker (registered in server.js).
// Always responds 200 for a request that was validly signed and reached a
// definite outcome (no-op, chained, dead-lettered, rechain-failed) -- QStash
// treats any non-2xx as a delivery failure and retries, which is only what
// we want for genuine transient failures INSIDE runInvestigation (already
// handled by the re-chain-with-same-afterStep path below, not by asking
// QStash's own retry to redeliver this HTTP call). Only signature failures
// and a missing runId return non-2xx, since those are the two cases where
// nothing meaningful was done and re-delivery (or rejection) is the
// correct QStash-level behavior.
export async function handleAgentWorker(req, res) {
  const signature = req.get("Upstash-Signature");
  // server.js's express.json() is configured with a `verify` callback that
  // stashes the raw request body buffer on req.rawBody specifically so this
  // handler can verify QStash's signature against the EXACT bytes QStash
  // signed -- re-serializing req.body (already JSON.parsed by express.json)
  // is not guaranteed to byte-for-byte match what QStash originally sent
  // (key order, whitespace) and would make signature verification flaky in
  // a way that's hard to reproduce. Falling back to a re-stringified body
  // only protects against req.rawBody being unexpectedly absent (e.g. a
  // future change to server.js's body-parser config); it will simply fail
  // verification (fail closed) rather than silently accept in that case.
  const rawBody = req.rawBody ? req.rawBody.toString("utf8") : JSON.stringify(req.body || {});

  const verified = await verifyQStashSignature({ signature, body: rawBody });
  if (!verified) {
    console.warn("agent-worker: rejected request with missing/invalid QStash signature");
    return res.status(401).json({ error: "invalid or missing QStash signature" });
  }

  const { runId, afterStep = 0, retryCount = 0 } = req.body || {};
  if (!runId) {
    return res.status(400).json({ error: "missing runId" });
  }

  // Invocation id (debugging aid, 2026-09-01): a fresh id per HTTP
  // invocation of this handler, distinct from runId/afterStep which are
  // shared across retries/redeliveries of the "same" logical step. Lets
  // log lines distinguish "one invocation ran long" from "two separate
  // invocations (a QStash redelivery, or a genuine concurrent duplicate)
  // both processed the same runId+afterStep" -- something timestamps
  // alone on the checkpoint can't tell apart. Log-only; no behavior change.
  // Moved above the dead-letter check below (2026-09-02) so debugLog calls
  // in that block can reference it -- previously only defined further down.
  const invocationId = randomUUID();
  debugLog(`agent-worker[${invocationId}]: entry runId=${runId} afterStep=${afterStep} retryCount=${retryCount}`);

  // Dead-letter blind spot fix (plan.md Section 9, priority #3): `retryCount`
  // above only reflects failures THIS handler lived long enough to observe
  // and persist (via the publishAgentStep re-chain call below, threading
  // newRetryCount into the next message's body). A Vercel platform timeout
  // kills the function mid-`runInvestigation` -- before any of that code
  // runs -- so a step that dies this way can NEVER increment `retryCount`,
  // and AGENT_WORKER_MAX_CONSECUTIVE_FAILURES can never catch it. QStash
  // itself, however, has no blind spot here: it retries the exact same
  // message (same body, including this stale `retryCount`) whenever the
  // HTTP call doesn't return within its own Max HTTP Response Duration, and
  // stamps every such redelivery with an `Upstash-Retried` header --
  // "0" on the first delivery attempt of a given message, incrementing by 1
  // on each subsequent QStash-initiated redelivery -- entirely independent
  // of whatever this repo's own code did or didn't get to run. Reading that
  // header gives a true count of "how many times has QStash already
  // redelivered THIS specific message" regardless of whether any of those
  // prior attempts ever reached the point of updating our own counter.
  const qstashRetried = Number(req.get("Upstash-Retried") ?? 0) || 0;
  // The larger of the two is the more accurate picture of how stuck this
  // step really is: retryCount catches ordinary (non-timeout) failures that
  // DID complete and re-chain normally; qstashRetried catches the timeout
  // case retryCount structurally cannot. Neither alone is complete.
  const effectiveRetryCount = Math.max(retryCount, qstashRetried);

  if (effectiveRetryCount >= AGENT_WORKER_MAX_CONSECUTIVE_FAILURES) {
    // Dead-letter BEFORE attempting another (likely doomed, and
    // potentially another ~300s-costing) runInvestigation call -- if
    // QStash has already redelivered this exact message
    // AGENT_WORKER_MAX_CONSECUTIVE_FAILURES times, further attempts are
    // vanishingly unlikely to be the one that finally finishes, and every
    // attempt in the meantime is real time and quota spent on a step this
    // run has already effectively failed on. Checked here (before the
    // idempotency/checkpoint-status checks below even load) so this can't
    // itself be skipped by racing another invocation's status update --
    // worst case this fires once redundantly if a genuinely-successful
    // concurrent invocation is *also* in flight, which the checkpoint
    // status check just below already treats as a safe no-op.
    const checkpointForDeadLetter = await loadCheckpoint(runId);
    if (checkpointForDeadLetter && checkpointForDeadLetter.status === "running" && checkpointForDeadLetter.stepsDone === afterStep) {
      await saveCheckpoint(runId, {
        newContents: [],
        transcript: checkpointForDeadLetter.transcript,
        stepsDone: checkpointForDeadLetter.stepsDone,
        task: checkpointForDeadLetter.task,
        repeatCounts: checkpointForDeadLetter.repeatCounts,
        consecutiveAllRepeatSteps: checkpointForDeadLetter.consecutiveAllRepeatSteps,
        provider: checkpointForDeadLetter.provider,
        model: checkpointForDeadLetter.model,
        maxOutputTokens: checkpointForDeadLetter.maxOutputTokens,
        preambleVariant: checkpointForDeadLetter.preambleVariant,
        pendingVerification: checkpointForDeadLetter.pendingVerification,
        structuralRecheckUsed: checkpointForDeadLetter.structuralRecheckUsed,
        status: "failed",
        finalAnswer: `Investigation stopped after QStash redelivered step ${checkpointForDeadLetter.stepsDone + 1} ${qstashRetried} time(s) without a completed attempt (likely repeated platform-level execution timeouts on an oversized step) -- this is caught by Upstash-Retried header inspection, not this worker's own in-process retryCount, since a hard platform timeout kills the function before retryCount can ever be incremented.`,
        stepStartedAt: null,
      });
      console.error(`agent-worker: runId ${runId} dead-lettered via Upstash-Retried inspection (qstashRetried=${qstashRetried}, bodyRetryCount=${retryCount}) on step ${checkpointForDeadLetter.stepsDone + 1}`);
      debugLog(`agent-worker[${invocationId}]: exit status=dead-lettered reason=qstash-retried-threshold runId=${runId} afterStep=${afterStep} qstashRetried=${qstashRetried}`);
      return res.status(200).json({ status: "dead-lettered", reason: "qstash-retried-threshold", steps: checkpointForDeadLetter.stepsDone });
    }
    // Checkpoint already moved on (finished, or another invocation already
    // handled this step) -- fall through to the ordinary idempotency checks
    // below, which will correctly no-op on it.
  }

  const checkpoint = await loadCheckpoint(runId);
  if (!checkpoint) {
    // Expired past the 1-hour TTL, never existed, or Redis is unreachable
    // (agent_checkpoint.js fails open on reads) -- nothing to advance. Not
    // an error from QStash's point of view (it did its job delivering the
    // message); the run itself is just gone.
    console.warn(`agent-worker: no live checkpoint for runId ${runId} -- dropping message`);
    debugLog(`agent-worker[${invocationId}]: exit status=no-op reason=checkpoint-missing runId=${runId} afterStep=${afterStep}`);
    return res.status(200).json({ status: "no-op", reason: "checkpoint missing or expired" });
  }

  if (checkpoint.status !== "running") {
    // Already finished ("done" via a genuine answer or the hard-cap
    // finalize path, or "failed" via this same file's dead-letter path
    // below) by a prior invocation -- do not touch it again. Also covers a
    // duplicate/late redelivery of a message whose run has since completed.
    debugLog(`agent-worker[${invocationId}]: exit status=no-op reason=status-not-running runId=${runId} afterStep=${afterStep} checkpointStatus=${checkpoint.status}`);
    return res.status(200).json({ status: "no-op", reason: `checkpoint status is "${checkpoint.status}", not "running"` });
  }

  if (checkpoint.stepsDone !== afterStep) {
    // Idempotency guard: this message's view of the world (afterStep) no
    // longer matches the live checkpoint -- another invocation (a QStash
    // redelivery racing this one, most likely) already advanced it.
    // Re-executing here would double-take a step Gemini already took.
    debugLog(`agent-worker[${invocationId}]: exit status=no-op reason=stale-afterStep runId=${runId} afterStep=${afterStep} liveStepsDone=${checkpoint.stepsDone}`);
    return res.status(200).json({ status: "no-op", reason: `stepsDone (${checkpoint.stepsDone}) != afterStep (${afterStep}) -- already advanced by another invocation` });
  }

  // Heartbeat write (step 1): write stepStartedAt timestamp onto the checkpoint
  // via saveCheckpoint, INSIDE the existing afterStep/stepsDone idempotency guard.
  await saveCheckpoint(runId, {
    transcript: checkpoint.transcript,
    stepsDone: checkpoint.stepsDone,
    task: checkpoint.task,
    repeatCounts: checkpoint.repeatCounts,
    consecutiveAllRepeatSteps: checkpoint.consecutiveAllRepeatSteps,
    preCompactionResults: checkpoint.preCompactionResultIds,
    resultCache: checkpoint.resultCacheIds,
    provider: checkpoint.provider,
    model: checkpoint.model,
    maxOutputTokens: checkpoint.maxOutputTokens,
    preambleVariant: checkpoint.preambleVariant,
    pendingVerification: checkpoint.pendingVerification,
    structuralRecheckUsed: checkpoint.structuralRecheckUsed,
    overallMaxSteps: checkpoint.overallMaxSteps,
    status: checkpoint.status,
    finalAnswer: checkpoint.finalAnswer,
    stepStartedAt: Date.now(),
  });
  debugLog(`agent-worker[${invocationId}]: heartbeat written, entering runInvestigation runId=${runId} afterStep=${afterStep}`);

  let result;
  try {
    result = await runInvestigation({ resume_run_id: runId, singleStep: true });
  } catch (err) {
    // Belt-and-suspenders: runInvestigation is designed to catch its own
    // failures internally and return a `{ failed: true }` result rather
    // than throw (see its own file header) -- but if something still
    // escapes it (a bug, an exotic error shape), treat it exactly like an
    // ordinary same-step failure below instead of letting it crash this
    // endpoint, which would strand the chain with no re-chain AND no
    // dead-letter finalization either.
    result = { steps: checkpoint.stepsDone, failed: true, answer: `(agent-worker: unexpected error advancing runId ${runId}: ${err?.message ?? String(err)})` };
  }
  debugLog(`agent-worker[${invocationId}]: runInvestigation returned steps=${result.steps} failed=${!!result.failed} runId=${runId} afterStep=${afterStep}`);

  if (result.failed) {
    console.error(`agent-worker: runId ${runId} step failed: ${result.answer}`);
  }

  const advanced = result.steps > afterStep;
  // Uses effectiveRetryCount (max of body retryCount and the
  // Upstash-Retried header, computed above) rather than plain retryCount,
  // so a step that failed via QStash-level redeliveries before ever
  // reaching this line still counts those redeliveries toward the
  // threshold below, not just whatever this repo's own code previously
  // managed to persist.
  const newRetryCount = advanced ? 0 : effectiveRetryCount + 1;

  const latest = await loadCheckpoint(runId);
  if (!latest || latest.status !== "running") {
    // Finished this step (a genuine final answer, or the hard-cap finalize
    // path inside runInvestigation) -- both already persist status "done"
    // themselves. Nothing more to chain.
    debugLog(`agent-worker[${invocationId}]: exit status=${latest?.status || "gone"} runId=${runId} afterStep=${afterStep} steps=${result.steps}`);
    return res.status(200).json({ status: latest?.status || "gone", steps: result.steps });
  }

  if (!advanced && newRetryCount >= AGENT_WORKER_MAX_CONSECUTIVE_FAILURES) {
    // Dead-letter (step 8): this exact step has now failed
    // AGENT_WORKER_MAX_CONSECUTIVE_FAILURES times in a row without ever
    // advancing stepsDone. A genuinely transient 429/503 succeeds well
    // before this many attempts -- QStash's own delivery-retry backoff
    // already spaces re-chain publishes out in practice -- so this is
    // treated as a permanent failure. Finalize the checkpoint as "failed"
    // so a poller (agent_tools.js's resume_run_id path) gets a definitive
    // answer instead of a chain that silently stopped re-publishing with
    // no record of why.
    await saveCheckpoint(runId, {
      newContents: [],
      transcript: latest.transcript,
      stepsDone: latest.stepsDone,
      task: latest.task,
      repeatCounts: latest.repeatCounts,
      consecutiveAllRepeatSteps: latest.consecutiveAllRepeatSteps,
      provider: latest.provider,
      model: latest.model,
      maxOutputTokens: latest.maxOutputTokens,
      preambleVariant: latest.preambleVariant,
      pendingVerification: latest.pendingVerification,
      structuralRecheckUsed: latest.structuralRecheckUsed,
      status: "failed",
      finalAnswer: `Investigation stopped after ${AGENT_WORKER_MAX_CONSECUTIVE_FAILURES} consecutive failures on step ${latest.stepsDone + 1}: ${result.answer}`,
      stepStartedAt: null,
    });
    console.error(`agent-worker: runId ${runId} dead-lettered after ${newRetryCount} consecutive failures on step ${latest.stepsDone + 1}`);
    debugLog(`agent-worker[${invocationId}]: exit status=dead-lettered runId=${runId} afterStep=${afterStep} steps=${latest.stepsDone}`);
    return res.status(200).json({ status: "dead-lettered", steps: latest.stepsDone });
  }

  // Still running and under the retry cap (whether this step succeeded and
  // there's more work to do, or it failed but hasn't hit the dead-letter
  // threshold yet) -- re-chain: publish the next worker invocation with the
  // FRESH stepsDone/retryCount so the next message's idempotency/dead-letter
  // checks are accurate.
  try {
    await publishAgentStep({ runId, afterStep: latest.stepsDone, retryCount: newRetryCount });
  } catch (err) {
    // The step itself either succeeded or failed-but-still-under-the-cap --
    // only the re-chain PUBLISH failed here. The checkpoint is left in a
    // perfectly valid "running" state with a fresh lastStepAt (saved by
    // runInvestigation's own per-step checkpoint write, not by this file),
    // so agent_tools.js's stale-lastStepAt fallback (see its "Tool behavior
    // change" branching) will correctly detect this as a broken chain once
    // lastStepAt goes stale and resume the loop synchronously on the next
    // poll -- this is exactly the scenario that fallback exists for, so a
    // failed publish here is not a silent stranding, just a slower recovery
    // path than the chain continuing on its own.
    console.error(`agent-worker: failed to re-chain runId ${runId} after step ${latest.stepsDone}: ${err?.message ?? String(err)}`);
    debugLog(`agent-worker[${invocationId}]: exit status=step-ok-rechain-failed runId=${runId} afterStep=${afterStep} steps=${latest.stepsDone}`);
    return res.status(200).json({ status: "step-ok-rechain-failed", steps: latest.stepsDone });
  }

  debugLog(`agent-worker[${invocationId}]: exit status=chained runId=${runId} afterStep=${afterStep} steps=${latest.stepsDone}`);
  return res.status(200).json({ status: "chained", steps: latest.stepsDone });
}

// ---------------------------------------------------------------------------
// handleAgentWorkerFailure (plan.md Section 13) -- Express handler for
// POST /api/agent-worker-failure, the QStash failureCallback target
// configured in publishAgentStep (qstash_client.js). QStash calls THIS
// endpoint -- as its own separately-signed message, not a redelivery of the
// original one -- exactly once, the moment it gives up on delivering the
// original agent-worker message (immediately, now that publishAgentStep
// sets retries: QSTASH_STEP_RETRIES = 0, rather than after its old
// default ~40min/3-retry budget).
//
// WHY THIS EXISTS: the in-process dead-letter check earlier in this file
// (effectiveRetryCount via the Upstash-Retried header) can only run INSIDE
// a live invocation of THIS worker. A step that hard-times-out on every
// single QStash delivery attempt means no further invocation of the
// ORIGINAL endpoint ever arrives once QStash's retry budget is spent --
// that check never gets a chance to fire, and the checkpoint is left at
// status:"running" forever with nothing to catch it (the live blind spot
// confirmed in Section 12). This callback is the
// only signal QStash gives for that terminal state, so it's the only place
// this specific gap can be closed.
//
// Body shape is QStash's failure-callback JSON (see
// https://upstash.com/docs/qstash/features/callbacks#what-is-a-failure-callback),
// NOT the original {runId, afterStep, retryCount} shape agent-worker
// receives directly -- the original request body survives base64-encoded
// under `sourceBody`.
export async function handleAgentWorkerFailure(req, res) {
  const signature = req.get("Upstash-Signature");
  // Same rawBody/signature-verification rationale as handleAgentWorker
  // above -- this is itself a QStash-originated request (a callback is
  // just a new signed message), so it goes through the same fail-closed
  // verifyQStashSignature check before anything else happens.
  const rawBody = req.rawBody ? req.rawBody.toString("utf8") : JSON.stringify(req.body || {});
  const verified = await verifyQStashSignature({ signature, body: rawBody });
  if (!verified) {
    console.warn("agent-worker-failure: rejected request with missing/invalid QStash signature");
    return res.status(401).json({ error: "invalid or missing QStash signature" });
  }

  const { retried, maxRetries } = req.body || {};
  let sourcePayload;
  try {
    sourcePayload = JSON.parse(Buffer.from(req.body?.sourceBody || "", "base64").toString("utf8"));
  } catch (err) {
    console.error(`agent-worker-failure: could not decode/parse sourceBody: ${err?.message ?? err}`);
    return res.status(200).json({ status: "no-op", reason: "unparseable sourceBody" });
  }

  const { runId, afterStep = 0 } = sourcePayload || {};
  if (!runId) {
    return res.status(200).json({ status: "no-op", reason: "sourceBody missing runId" });
  }

  const checkpoint = await loadCheckpoint(runId);
  if (!checkpoint || checkpoint.status !== "running" || checkpoint.stepsDone !== afterStep) {
    // Already finished, expired, or moved on -- most likely because the
    // organic in-process retry path (agent_tools.js's stale-lastStepAt
    // fallback, or a same-step retryCount re-chain that actually
    // succeeded) already recovered this run before QStash's own delivery
    // attempt to the ORIGINAL message finally gave up. This callback is
    // then stale, not authoritative -- no-op rather than clobbering
    // whatever more-current state the checkpoint is already in.
    debugLog(`agent-worker-failure: no-op runId=${runId} afterStep=${afterStep} (checkpoint not in the stalled state this callback describes)`);
    return res.status(200).json({ status: "no-op", reason: "checkpoint not in the stalled state this callback describes" });
  }

  // Same explicit-field-list shape as the Upstash-Retried-header dead-letter
  // save earlier in this file (NOT a blind spread -- agent_checkpoint.js
  // splits `contents` into its own Redis LIST, so a meta-only write needs
  // newContents: [] to avoid re-appending already-persisted content).
  await saveCheckpoint(runId, {
    newContents: [],
    transcript: checkpoint.transcript,
    stepsDone: checkpoint.stepsDone,
    task: checkpoint.task,
    repeatCounts: checkpoint.repeatCounts,
    consecutiveAllRepeatSteps: checkpoint.consecutiveAllRepeatSteps,
    provider: checkpoint.provider,
    model: checkpoint.model,
    maxOutputTokens: checkpoint.maxOutputTokens,
    preambleVariant: checkpoint.preambleVariant,
    pendingVerification: checkpoint.pendingVerification,
    structuralRecheckUsed: checkpoint.structuralRecheckUsed,
    status: "failed",
    finalAnswer: `Investigation stopped: QStash exhausted its own delivery budget (retried ${retried ?? "?"}/${maxRetries ?? "?"}) on step ${checkpoint.stepsDone + 1} without ever getting a response -- almost always a platform-level execution timeout repeating on the same oversized step. Reported via QStash's failureCallback, since a step that times out on every delivery attempt means no further worker invocation ever arrives to run this file's own in-process dead-letter check.`,
    stepStartedAt: null,
  });
  console.error(`agent-worker-failure: runId ${runId} dead-lettered via QStash failureCallback (retried=${retried}, maxRetries=${maxRetries}) on step ${checkpoint.stepsDone + 1}`);
  return res.status(200).json({ status: "dead-lettered", reason: "qstash-failure-callback", steps: checkpoint.stepsDone });
}
