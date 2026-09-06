#!/usr/bin/env node
// ---------------------------------------------------------------------------
// scripts/preamble-ab-metrics.js -- TEMPORARY debug tooling for the
// delegate_agent SYSTEM_PREAMBLE "verbose" vs "trimmed" A/B test.
//
// DELETE THIS FILE once the A/B test is done -- it's a throwaway debugging
// aid, not a supported part of the codebase (no tests, not wired into
// server.js, not documented anywhere else). Item 6 of the A/B test handoff:
// "discuss/build whatever pull-per-variant-stats-from-transcripts tooling
// is needed."
//
// USAGE:
//   node scripts/preamble-ab-metrics.js <runId> [<runId> ...]
//
// Reads each runId's checkpoint directly out of Redis (via
// agent_checkpoint.js's loadCheckpoint -- same fail-open contract: a
// missing/expired runId just gets skipped with a warning, not a crash) and
// prints:
//   - per-run metrics (variant, steps, verification pass fired + whether it
//     actually changed the answer, structural recheck used, stuck-loop
//     force evidence, tool-call-leakage evidence, final conversation size
//     as a token-cost proxy)
//   - an aggregate summary grouped by preambleVariant
//
// IMPORTANT CAVEATS (read before trusting a number below):
//   1. Checkpoints have a 1-hour TTL and "done" checkpoints are only kept
//      around for that same TTL (see agent_checkpoint.js) -- this only
//      works run-to-run, right after each test run, not for historical
//      analysis days later. If you want a durable record, copy this
//      script's printed output somewhere yourself.
//   2. "final conversation size (chars)" is a LOWER-BOUND proxy for
//      resent-token cost, not the true cumulative total -- it's the size of
//      the conversation's FINAL state (after compaction, if the provider
//      has it enabled), not the sum of every step's resend. A run that
//      took more steps resent its early turns more times than this number
//      reflects. Use it for a same-task, same-step-count comparison, not
//      as an absolute cost figure.
//   3. Stuck-loop-force detection only catches evidence still present in
//      the final `contents` array -- if compactHistoryInPlace ever starts
//      compacting SYSTEM NOTE text (it currently doesn't -- it only
//      compacts bulky tool results), this would under-count. Not a
//      concern today, just noted in case that changes.
//   4. Verification-pass "did it change anything" is a plain string
//      comparison (draft vs. corrected answer, LINE_QUOTE markers
//      stripped) -- a whitespace-only or trivial rewording counts as
//      "changed" the same as a substantive fix. Read the printed diff
//      snippet yourself before treating a run as a "real" correction --
//      see the handoff's own caution: log what changed, not just
//      pass/fail, because a bad first draft silently fixed by
//      verification still looks like a clean success if you only check
//      final answers.
// ---------------------------------------------------------------------------

import { loadCheckpoint } from "../connectors/delegate/agent/agent_checkpoint.js";

const VERIFICATION_ANCHOR = "[SYSTEM NOTE -- verification pass]";
const STRUCTURAL_RECHECK_ANCHOR = "[STRUCTURAL LINE-QUOTE CHECK FAILED]";
const STUCK_LOOP_SOFT_ANCHOR = "the last 2 steps consisted entirely of repeat calls";
const STUCK_LOOP_HARD_ANCHOR = "3 consecutive steps have consisted entirely of repeat calls";
// Matches the exact wording agent_delegate.js's tool-call-leakage backstop
// emits when it fires (see detectToolCallLeakage's caller) -- string match
// on the emitted message, not a re-implementation of the detector itself.
const LEAKAGE_ANCHOR = "attempted to invoke the";
// Duplicated from agent_delegate.js's own (unexported) LINE_QUOTE_PATTERN --
// deliberately not exported from that file for a throwaway debug script to
// import; small enough to copy.
const LINE_QUOTE_PATTERN = /^LINE_QUOTE:\s*(.+)$/gm;
function stripLineQuoteMarkers(text) {
  return text.replace(LINE_QUOTE_PATTERN, "").replace(/\n{3,}/g, "\n\n").trim();
}

function turnText(turn) {
  return (turn.parts || []).map((p) => p.text || "").join("").trim();
}

export function analyzeRun(runId, checkpoint) {
  const { contents = [], ...meta } = checkpoint;

  const modelTurns = contents.filter((t) => t.role === "model");
  const userTextTurns = contents.filter(
    (t) => t.role === "user" && Array.isArray(t.parts) && t.parts.some((p) => p.text)
  );

  // Verification pass: find the user turn whose text starts with the
  // anchor, then compare the model turn immediately before it (the draft)
  // against the model turn immediately after (the correction/reaffirmation).
  let verificationFired = false;
  let verificationChangedAnswer = null; // null = n/a, true/false once known
  let draftAnswer = null;
  let correctedAnswer = null;
  const verificationTurnIndex = contents.findIndex(
    (t) => t.role === "user" && Array.isArray(t.parts) && t.parts.some((p) => p.text?.startsWith(VERIFICATION_ANCHOR))
  );
  if (verificationTurnIndex !== -1) {
    verificationFired = true;
    const draftTurn = contents[verificationTurnIndex - 1];
    const correctedTurn = contents[verificationTurnIndex + 1];
    if (draftTurn?.role === "model") draftAnswer = stripLineQuoteMarkers(turnText(draftTurn));
    if (correctedTurn?.role === "model") correctedAnswer = stripLineQuoteMarkers(turnText(correctedTurn));
    if (draftAnswer !== null && correctedAnswer !== null) {
      verificationChangedAnswer = draftAnswer !== correctedAnswer;
    }
  }

  const structuralRecheckFired = userTextTurns.some((t) => turnText(t).includes(STRUCTURAL_RECHECK_ANCHOR));
  const stuckLoopSoftNudge = userTextTurns.some((t) => turnText(t).includes(STUCK_LOOP_SOFT_ANCHOR))
    || contents.some((t) => (t.parts || []).some((p) => p.text?.includes(STUCK_LOOP_SOFT_ANCHOR)));
  const stuckLoopForced = contents.some((t) => (t.parts || []).some((p) => p.text?.includes(STUCK_LOOP_HARD_ANCHOR)));
  const leakageEvidence = (meta.finalAnswer || "").includes(LEAKAGE_ANCHOR);

  const finalConversationChars = JSON.stringify(contents).length;

  return {
    runId,
    preambleVariant: meta.preambleVariant || "verbose (unset/legacy)",
    provider: meta.provider || "gemini",
    model: meta.model || "(default)",
    status: meta.status,
    stepsDone: meta.stepsDone,
    overallMaxSteps: meta.overallMaxSteps,
    verificationFired,
    verificationChangedAnswer,
    draftAnswer,
    correctedAnswer,
    structuralRecheckFired,
    structuralRecheckUsedFlag: meta.structuralRecheckUsed || false,
    stuckLoopSoftNudge,
    stuckLoopForced,
    leakageEvidence,
    modelTurnCount: modelTurns.length,
    finalConversationChars,
  };
}

function printRun(r) {
  console.log(`\n=== runId: ${r.runId} ===`);
  console.log(`  variant:              ${r.preambleVariant}`);
  console.log(`  provider/model:       ${r.provider} / ${r.model}`);
  console.log(`  status:               ${r.status}`);
  console.log(`  steps:                ${r.stepsDone} / ${r.overallMaxSteps} (cap)`);
  console.log(`  model turns:          ${r.modelTurnCount}`);
  console.log(`  verification fired:   ${r.verificationFired}`);
  if (r.verificationFired) {
    console.log(`    -> changed answer:  ${r.verificationChangedAnswer}`);
    if (r.verificationChangedAnswer) {
      console.log(`    -> draft (first 200 chars):     ${JSON.stringify((r.draftAnswer || "").slice(0, 200))}`);
      console.log(`    -> corrected (first 200 chars): ${JSON.stringify((r.correctedAnswer || "").slice(0, 200))}`);
    }
  }
  console.log(`  structural recheck:   fired=${r.structuralRecheckFired} (meta flag: ${r.structuralRecheckUsedFlag})`);
  console.log(`  stuck-loop soft nudge (2 in a row): ${r.stuckLoopSoftNudge}`);
  console.log(`  stuck-loop forced (3+ in a row):    ${r.stuckLoopForced}`);
  console.log(`  tool-call leakage evidence:         ${r.leakageEvidence}`);
  console.log(`  final conversation size (chars, proxy for resent-token cost): ${r.finalConversationChars}`);
}

function printAggregate(results) {
  const byVariant = new Map();
  for (const r of results) {
    if (!byVariant.has(r.preambleVariant)) byVariant.set(r.preambleVariant, []);
    byVariant.get(r.preambleVariant).push(r);
  }
  console.log(`\n=== AGGREGATE (grouped by preambleVariant) ===`);
  for (const [variant, rows] of byVariant) {
    const n = rows.length;
    const avg = (fn) => (rows.reduce((sum, r) => sum + (fn(r) ? 1 : 0), 0) / n * 100).toFixed(0);
    const avgNum = (fn) => (rows.reduce((sum, r) => sum + (fn(r) || 0), 0) / n).toFixed(1);
    console.log(`\n  ${variant} (n=${n}):`);
    console.log(`    verification fired:            ${avg((r) => r.verificationFired)}%`);
    console.log(`    verification changed answer:   ${avg((r) => r.verificationChangedAnswer)}% (of runs where it fired: ${
      rows.filter((r) => r.verificationFired).length ? (rows.filter((r) => r.verificationChangedAnswer).length / rows.filter((r) => r.verificationFired).length * 100).toFixed(0) : "n/a"
    }%)`);
    console.log(`    structural recheck fired:      ${avg((r) => r.structuralRecheckFired)}%`);
    console.log(`    stuck-loop forced:             ${avg((r) => r.stuckLoopForced)}%`);
    console.log(`    tool-call leakage evidence:    ${avg((r) => r.leakageEvidence)}%`);
    console.log(`    avg steps taken:               ${avgNum((r) => r.stepsDone)}`);
    console.log(`    avg final conversation chars:  ${avgNum((r) => r.finalConversationChars)}`);
  }
}

async function main() {
  const runIds = process.argv.slice(2);
  if (!runIds.length) {
    console.error("Usage: node scripts/preamble-ab-metrics.js <runId> [<runId> ...]");
    process.exit(1);
  }

  const results = [];
  for (const runId of runIds) {
    const checkpoint = await loadCheckpoint(runId);
    if (!checkpoint) {
      console.warn(`\n(!) No live checkpoint for runId "${runId}" -- expired (1hr TTL), never existed, or Redis unreachable. Skipping.`);
      continue;
    }
    const r = analyzeRun(runId, checkpoint);
    results.push(r);
    printRun(r);
  }

  if (results.length > 1) printAggregate(results);
  if (!results.length) {
    console.error("\nNo runs could be analyzed -- nothing to report.");
    process.exit(1);
  }
}

// Only run as a CLI when this file is executed directly (`node
// scripts/preamble-ab-metrics.js ...`) -- not when imported as a module
// (e.g. by test/preamble-ab-metrics.test.js importing analyzeRun), which
// would otherwise call process.exit() during a test run.
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((err) => {
    console.error("preamble-ab-metrics.js failed:", err);
    process.exit(1);
  });
}
