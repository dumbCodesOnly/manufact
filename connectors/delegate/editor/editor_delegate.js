// ---------------------------------------------------------------------------
// connectors/delegate/editor/editor_delegate.js -- core agent loop for delegate_editor ("Limited GitHub write access for agent").
//
// Adapts connectors/delegate/designer/designer_delegate.js's runDesignAgent shape --
// same multi-step Gemini function-calling loop, same checkpoint/resume
// contract (guardrail #7: reuse designer_checkpoint.js's shape, here via
// connectors/delegate/editor/editor_checkpoint.js, a same-shape sibling with its own
// Redis key prefix), same stuck-loop/repeat-detection and final-step tool
// withholding fixes -- but:
//
//   - THREE tools (read_file/write_file/validate), same shape as
//     designer_delegate.js's own three-tool loop -- validate is wired
//     in here via editor_tool_functions.js's sibling module
//     editor_validate.js, capped per-file the same way
//     FRONTEND_MAX_VALIDATE_CALLS caps designer_delegate.js's validate()
//     (see EDITOR_MAX_VALIDATE_CALLS below). validate is opt-in, same as
//     designer_delegate.js: the model chooses whether to call it before a
//     write, write_file itself doesn't force it.
//   - Backed by connectors/github/editor_tool_functions.js's general
//     Contents-API read_file/write_file (guardrails #2/#3/#4 already
//     enforced AT THAT LAYER -- see that file's own header), not
//     designer_tool_functions.js's frontend-only helper.
//   - NEW: guardrail #6, per-run and per-file write caps
//     (EDITOR_MAX_FILES_PER_RUN / EDITOR_MAX_WRITES_PER_FILE), enforced
//     inside write_file's execute() closure below, before writeFile() is
//     even called -- bounds the blast radius of a stuck or misbehaving loop
//     independently of (and in addition to) the stuck-loop repeat detection
//     carried over from designer_delegate.js, since a loop that keeps
//     writing DIFFERENT files/paths each step would never trip repeat
//     detection at all.
//   - Guardrail #2 (default-branch refusal) is checked once, up front, via
//     editor_tool_functions.js's assertNotDefaultBranch -- same "look it up
//     live, never trust the caller" posture designer_delegate.js uses via
//     a raw githubRequest call; here we just reuse the tool layer's own
//     exported helper instead of duplicating the lookup.
//   - No create_pull_request/merge_pull_request in this tool's own function
//     set (guardrail #8) -- enforced structurally, by the FUNCTIONS array
//     below simply never including them, same as designer_delegate.js's
//     three-tool array never including anything outside its own scope.
//
// NOT YET WIRED TO AN MCP TOOL: exports runEditorAgent as a plain function,
// unit-testable independently of any server.tool(...) registration
// -- same "build the loop, unit test it independently" posture
// designer_delegate.js's own header describes for its step.
// ---------------------------------------------------------------------------

import { randomUUID } from "node:crypto";
import { providerChat } from "../../llm/router.js";
import { formatCascadeLogLine } from "../../llm/cascade_log.js";
// Reused, provider-agnostic verification helpers from the read-only
// investigation loop (see that file's own header comments for the full
// rationale/failure-mode evidence behind each). Both are pure functions
// over (answerText) / (claims, contents) and carry no bai-specific or
// investigation-specific assumptions -- extractMechanicalClaims just
// regexes identifier/backtick-quoted shapes out of a draft answer, and
// findUnverifiedClaims just checks those strings against the raw
// functionResponse text already sitting in `contents`. Deliberately NOT
// importing detectToolCallLeakage/extractConditionalClaims/
// lineIsVerbatimInToolResults here -- the former is a bai-only backstop
// for a failure mode never observed on Gemini, and the latter two target a
// different failure shape (a fabricated RELATIONSHIP between two real
// tokens) than the one this file's own guard below is for (a fabricated
// WRITE that never happened at all).
import { extractMechanicalClaims, findUnverifiedClaims } from "../agent/agent_delegate.js";
import { readFile, writeFile, assertNotDefaultBranch } from "../../github/editor_tool_functions.js";
import { validateByExtension } from "../../github/editor_validate.js";
import { saveCheckpoint, loadCheckpoint } from "./editor_checkpoint.js";
import { isRedisConfigured } from "../../shared/cooldown.js";
import {
  EDITOR_DEFAULT_STEPS,
  EDITOR_HARD_MAX_STEPS,
  EDITOR_MAX_FILES_PER_RUN,
  EDITOR_MAX_WRITES_PER_FILE,
  EDITOR_MAX_VALIDATE_CALLS,
} from "../../../config.js";
import { appendTask, buildEditorPreamble } from "../shared/preamble.js";

// Same reasoning as connectors/delegate/agent/agent_delegate.js's
// isTransientGeminiError / designer_delegate.js's copy of it: only 429
// (rate limit) and 503 (overloaded) are worth resuming past -- everything
// else reproduces identically on a resume.
function isTransientGeminiError(err) {
  return err?.status === 429 || err?.status === 503 || err?.transient === true;
}

// ---------------------------------------------------------------------------
// Writes-vs-claim guard (fix for the 2026-09-06/07 raffle-app Stars-payment
// incident: a run served entirely by the fallback model "gemini-3.5-flash-lite"
// took 9 read_file steps, wrote NOTHING, then produced a confident, detailed
// completion report -- specific function names, specific files claimed
// updated -- none of which existed in the actual diff. Confirmed via
// diff_files against main: zero differences. Root cause: this loop's
// completion path used to trust the model's own final text unconditionally,
// with zero cross-check against `writtenFiles`, the exact ground-truth list
// already sitting in scope at that point).
//
// Modeled directly on agent_delegate.js's own verification pass (see that
// file's VERIFICATION_PROMPT/pendingVerification for the general pattern
// and the live-testing evidence behind it), but narrower and pointed at
// this loop's own failure mode rather than ported wholesale:
//   - agent_delegate.js verifies claims about RETRIEVED DATA (does a quoted
//     fact/identifier appear verbatim in tool output already gathered).
//   - This guard verifies claims about ACTIONS TAKEN (does a claimed write
//     appear in writtenFiles, the run's own append-only write log) --
//     a check agent_delegate.js has no reason to need, since it never
//     writes anything.
// Both flow into the SAME single-fire pendingVerification mechanism below
// (one extra round, tools re-enabled, then whatever comes back is final --
// see agent_delegate.js's own comments for why a no-tools self-check was
// tried first and found insufficient: a model asked to double-check purely
// from memory just re-asserts its own mistake with equal confidence).
//
// Deliberately does NOT also port extractConditionalClaims/
// lineIsVerbatimInToolResults/detectToolCallLeakage -- those target
// different failure shapes (a fabricated relationship between two real
// facts; bai-specific text-mimicking-a-function-call) neither observed nor
// relevant to this incident. See this file's import comment for the same
// scoping note.
function buildEditorVerificationPrompt({ answer, contents, writtenFiles }) {
  const mechanicalClaims = extractMechanicalClaims(answer);
  return findUnverifiedClaims(mechanicalClaims, contents).then((unverifiedClaims) => {
    const writeLogLine = writtenFiles.length
      ? `This run has written to the following file(s) so far: ${writtenFiles.join(", ")}.`
      : `This run has NOT written to any file yet -- writtenFiles is empty.`;
    const writeLogNote =
      `[WRITE LOG CHECK] ${writeLogLine} If your answer above describes specific code changes (a function added, ` +
      `a handler wired up, a file refactored, a value updated) as already done, every such claim must correspond ` +
      `to an actual write_file call already reflected in the write log above -- not a plan, not what you intended ` +
      `to do, not what a read_file call showed could be done. If you described a change whose file is not in that ` +
      `list, that change has NOT been made: either call write_file now to actually make it (you still have tool ` +
      `access this turn), or rewrite your final answer to say plainly it was not completed and why, instead of ` +
      `reporting it as done.`;
    const claimNote = unverifiedClaims.length
      ? `\n\n[SPECIFIC ITEMS TO CHECK] The following identifier(s)/snippet(s) in your draft answer do not appear ` +
        `verbatim in any tool result (read_file/write_file/validate output) gathered so far this run: ` +
        `${unverifiedClaims.map((c) => `"${c}"`).join(", ")}. For EACH one: re-read the specific file it's claimed ` +
        `to come from and confirm it exact-matches what's actually there (or actually write it, if it was meant to ` +
        `be a change you made), THEN either keep the claim only if you can now back it with a fresh, real tool ` +
        `result, or correct it. Do not restate any of these unchanged based on memory or on the fact that you ` +
        `already wrote it once.`
      : "";
    return (
      `[SYSTEM NOTE -- verification pass] Before your answer above is treated as final, check it against the ` +
      `write log and tool results already produced in this run -- not your own summary of them. You have tool ` +
      `access again this turn. Once you are done checking, respond with the corrected final answer (or the same ` +
      `answer, if it already holds up under this check) as plain text with no further function calls.\n\n` +
      writeLogNote + claimNote
    );
  });
}

// buildSystemPreamble's actual text now lives in ../shared/preamble.js as
// buildEditorPreamble({ owner, repo, branch }) -- see that file's header
// for why this was relocated (not unified with designer's own preamble)
// there. Called directly at this file's own call sites below.

// Builds the two function declarations + their execute() closures for one
// run. owner/repo/branch are captured here, NOT exposed as parameters the
// model can set -- same fencing rationale as designer_delegate.js's
// buildFunctions (guardrail #1).
function buildFunctions({ owner, repo, branch, writtenFiles, writesPerFile, validateCounts }) {
  const FUNCTIONS = [
    {
      name: "read_file",
      description: "Read a file's current content on this run's branch, together with its blob sha (useful as base_sha for write_file).",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "File path within the repo, relative to repo root." },
        },
        required: ["path"],
      },
      execute: async ({ path }) => {
        const result = await readFile(owner, repo, path, branch);
        return `sha: ${result.sha}\n\n${result.content}`;
      },
    },
    {
      name: "write_file",
      description: "Write a file on this run's branch. Exactly one of `content` (full overwrite / create) or `replacements` (find/replace operations) is required.",
      parameters: {
        type: "object",
        properties: {
          path:    { type: "string", description: "File path within the repo, relative to repo root." },
          content: { type: "string", description: "Full new file content (mutually exclusive with replacements)" },
          replacements: {
            type: "array",
            description: "List of find/replace operations, applied sequentially (mutually exclusive with content). Requires the file to already exist.",
            items: {
              type: "object",
              properties: {
                find:    { type: "string" },
                replace: { type: "string" },
              },
              required: ["find", "replace"],
            },
          },
          base_sha: { type: "string", description: "Optional: the sha returned by a prior read_file call on this exact path, to detect concurrent changes." },
          message:  { type: "string", description: "Commit message (optional -- a reasonable default is used if omitted)" },
        },
        required: ["path"],
      },
      execute: async ({ path, content, replacements, base_sha, message }) => {
        // Guardrail #6, checked BEFORE writeFile() is called at all -- a
        // stuck loop that keeps writing different paths would never trip
        // the stuck-loop repeat detection below, so this cap has to be
        // independent of that mechanism, not layered only on top of it.
        const alreadyTouched = writtenFiles.includes(path);
        if (!alreadyTouched && writtenFiles.length >= EDITOR_MAX_FILES_PER_RUN) {
          return `Error: this run has already touched ${writtenFiles.length} distinct file(s), which is this run's per-run cap (EDITOR_MAX_FILES_PER_RUN=${EDITOR_MAX_FILES_PER_RUN}). Writing "${path}" would exceed it -- finish up with the files already touched (${writtenFiles.join(", ")}), or explain what's left undone.`;
        }
        const priorWrites = writesPerFile.get(path) || 0;
        if (priorWrites >= EDITOR_MAX_WRITES_PER_FILE) {
          return `Error: "${path}" has already been written ${priorWrites} time(s) this run, which is this run's per-file cap (EDITOR_MAX_WRITES_PER_FILE=${EDITOR_MAX_WRITES_PER_FILE}). Proceed without further writes to this file.`;
        }

        try {
          const result = await writeFile(owner, repo, path, { content, replacements, baseSha: base_sha, branch, message });
          writesPerFile.set(path, priorWrites + 1);
          if (!alreadyTouched) writtenFiles.push(path);
          if (result.noop) {
            return `No-op: "${path}" content already matches what you submitted -- nothing was committed.`;
          }
          return `Wrote ${result.path} (commit ${result.commitSha.slice(0, 7)}, new sha ${result.sha}).`;
        } catch (err) {
          // Conflict/policy errors are a normal, expected outcome the
          // model should react to (re-read, adjust, retry) -- returning
          // the message as a regular string result (rather than throwing)
          // is what lets the loop's existing error-string convention carry
          // it back to the model as a next-turn input, same as any other
          // tool result. Note: a rejected write (policy or conflict) does
          // NOT count against the write caps above -- nothing was actually
          // committed, so charging the cap for it would penalize the model
          // for correctly discovering a boundary.
          return `Error: ${err.message}`;
        }
      },
    },
    {
      name: "validate",
      description: "Syntax-check content against its file type (by extension) before writing. Capped per file path -- see the system instructions. Some extensions (.md, .txt) have no syntax to check and will always report valid.",
      parameters: {
        type: "object",
        properties: {
          path:    { type: "string", description: "File path (used only to determine which validator to run, by extension)" },
          content: { type: "string", description: "Content to validate" },
        },
        required: ["path", "content"],
      },
      execute: async ({ path, content }) => {
        const count = validateCounts.get(path) || 0;
        if (count >= EDITOR_MAX_VALIDATE_CALLS) {
          return `Error: validate() has already been called ${count} time(s) for "${path}", which is this run's per-file cap (${EDITOR_MAX_VALIDATE_CALLS}). Proceed without further validation of this file, or write it and reconsider your approach if it's still not right.`;
        }
        validateCounts.set(path, count + 1);
        const result = await validateByExtension(path, content);
        if (result.skipped) return `Unvalidated: ${result.skipped}`;
        return result.valid ? "Valid -- no syntax errors found." : `Invalid:\n${result.errors.join("\n")}`;
      },
    },
  ];

  const declarations = [{
    functionDeclarations: FUNCTIONS.map(({ name, description, parameters }) => ({ name, description, parameters })),
  }];

  return { FUNCTIONS, declarations };
}

// Runs the write-capable general-editor agent loop. Returns
// { answer, steps, transcript, runId, writtenFiles, task, failed? } -- same
// overall shape as designer_delegate.js's runDesignAgent / delegate_agent's
// runInvestigation, so the eventual MCP-facing tool can follow the
// same resume_run_id/failed-response conventions those already use.
//
// On a fresh call, owner/repo/branch/task are required; on a resume
// (resume_run_id set), they're restored from the checkpoint and any passed
// values are ignored -- same resume contract as designer_delegate.js (see
// its comments for why `task` specifically must never be trusted over the
// checkpoint's own record of it on a live resume).
export async function runEditorAgent({ owner, repo, branch, task, max_steps = EDITOR_DEFAULT_STEPS, resume_run_id, singleStep = false, provider }) {
  // The run's TRUE overall step ceiling -- distinct from cappedSteps (this
  // particular invocation's own loop bound). For a fresh run or a manual
  // synchronous resume the two are the same value. They diverge for a
  // worker-driven singleStep resume (the upcoming editor_worker.js), which
  // deliberately passes a shrunk per-call bound (startStep, so the shared
  // loop takes exactly one step) that must NOT be mistaken for "the run's
  // real last step" when deciding whether to withhold tools -- see
  // isFinalStep below. Restored from the checkpoint on a singleStep resume;
  // established fresh (or updated, on a manual resume with a new max_steps)
  // otherwise. Persisted in every checkpoint write below so it survives a
  // resume -- same split as connectors/delegate/agent/agent_delegate.js's
  // runInvestigation.
  let effectiveOverallMaxSteps;
  let cappedSteps;

  let runId = resume_run_id;
  let contents;
  let transcript;
  let startStep;
  let writtenFiles;
  let writesPerFile;
  let validateCounts;
  let effectiveOwner = owner;
  let effectiveRepo = repo;
  let effectiveBranch = branch;
  let effectiveTask = task;
  // The provider actually in effect for this run -- the caller-supplied one
  // on a fresh run, or the one restored from a resumed checkpoint (see
  // `checkpoint.provider || provider` below). Same reasoning as
  // connectors/delegate/agent/agent_delegate.js's runInvestigation: resuming on a
  // DIFFERENT provider than the one that started the run risks corrupting
  // the conversation shape, not just a preference mismatch. Threaded
  // through every providerChat/saveCheckpoint call below exactly like
  // effectiveTask is.
  let effectiveProvider = provider;
  // Stuck-loop detection -- same shape as designer_delegate.js's copy of
  // connectors/delegate/agent/agent_delegate.js's fix #4. See that file's comments
  // for the full reasoning; unchanged here.
  let repeatCounts = new Map();
  let resultCache = new Map();
  let consecutiveAllRepeatSteps = 0;
  // Tracks whether ANY step this run was served by a Gemini fallback model
  // (see connectors/gemini/client.js's cascade -- a 429/503/network error on
  // the primary model/key silently drops to GEMINI_FALLBACK_MODELS, e.g.
  // "gemini-3.5-flash-lite"). formatCascadeLogLine already logs this into
  // `transcript` per-step, but that's a side log a caller has to know to
  // read -- the incident this file's writes-vs-claim guard fixes involved
  // EVERY step being served by a weak fallback model with the caller only
  // discovering that fact by manually reading the transcript after the
  // fact. Surfaced directly on the returned result below so a caller can
  // treat "answer came from a fallback model" as a first-class signal to
  // weigh, without needing to parse transcript strings.
  let fallbackModelUsed = null;
  // Writes-vs-claim verification pass (see buildEditorVerificationPrompt's
  // header comment above for the incident/rationale) -- true once the model
  // has produced a draft final answer and been sent back for one no-fresh-
  // trust self-check round before that answer is persisted as done. Single-
  // fire, same pattern/reasoning as agent_delegate.js's own pendingVerification:
  // bounds this to exactly one extra step regardless of what comes back on
  // the second pass, and is persisted across resumes so a run that dies
  // mid-verification (e.g. a transient 429/503 on the verification call
  // itself) resumes back into the verification turn rather than silently
  // re-drafting a whole new answer from scratch.
  let pendingVerification = false;

  const checkpoint = resume_run_id ? await loadCheckpoint(resume_run_id) : null;

  // Async delegate_editor: a checkpoint whose
  // last save recorded status "done" already has a final answer sitting in
  // Redis (see the completion path near the end of this function). Return
  // it directly rather than re-entering the loop, which would otherwise
  // treat `contents` as still mid-conversation and either try to take more
  // (nonsensical) steps against a finished run, or -- worse -- silently
  // re-call the model on the same contents and produce a second, possibly
  // different "final answer" for a run that already committed its changes
  // and reported a result. This is also what makes resume_run_id usable as
  // a cheap poll handle for a background/worker-driven run: polling a
  // finished run is now a Redis read, not a re-run. Mirrors
  // connectors/delegate/agent/agent_delegate.js's runInvestigation, which has the
  // same short-circuit for the same reason.
  if (checkpoint && checkpoint.status === "done") {
    return {
      answer: checkpoint.finalAnswer,
      steps: checkpoint.stepsDone,
      transcript: checkpoint.transcript,
      runId: resume_run_id,
      task: checkpoint.task,
      writtenFiles: checkpoint.writtenFiles || [],
      failed: false,
    };
  }

  if (checkpoint) {
    contents = checkpoint.contents;
    transcript = checkpoint.transcript;
    startStep = checkpoint.stepsDone + 1;
    writtenFiles = checkpoint.writtenFiles || [];
    writesPerFile = new Map(Object.entries(checkpoint.writesPerFile || {}));
    validateCounts = new Map(Object.entries(checkpoint.validateCounts || {}));
    effectiveOwner = checkpoint.owner;
    effectiveRepo = checkpoint.repo;
    effectiveBranch = checkpoint.branch;
    effectiveTask = checkpoint.task;
    // Same reasoning as effectiveTask directly above -- once a run is past
    // step 1, the checkpoint's own record of which provider started it is
    // authoritative, not whatever the caller passes on a resume call.
    // Checkpoints saved before this field existed won't have it; fall back
    // to whatever the caller passed (may be undefined, which providerChat
    // treats as "gemini") rather than erroring.
    effectiveProvider = checkpoint.provider || provider;
    repeatCounts = new Map(Object.entries(checkpoint.repeatCounts || {}));
    consecutiveAllRepeatSteps = checkpoint.consecutiveAllRepeatSteps || 0;
    // Checkpoints saved before this field existed won't have it -- default
    // to false (normal tool-use resumes as before), same defensive pattern
    // as every other field restored here.
    pendingVerification = checkpoint.pendingVerification || false;
    fallbackModelUsed = checkpoint.fallbackModelUsed || null;
  } else if (resume_run_id) {
    // Same "fail loudly and distinctly" reasoning as designer_delegate.js --
    // this loop has no task-optional fallback path either, so there's no
    // ambiguous case to accommodate: always an error.
    throw new Error(
      isRedisConfigured()
        ? `resume_run_id "${resume_run_id}" has no live checkpoint -- it may have expired (1 hour TTL) or the id may be wrong. Start a new run with owner/repo/branch/task instead.`
        : `resume_run_id "${resume_run_id}" has no live checkpoint -- and Redis is NOT configured in this environment, so no checkpoint could ever have been saved. Start a new run with owner/repo/branch/task instead.`
    );
  } else if (singleStep) {
    // singleStep always implies a resume in practice (the worker's whole
    // purpose is chaining one-step resumes) -- the branch above already
    // throws a clearer, more specific error when resume_run_id was given but
    // its checkpoint failed to load. Guard here too so a future caller
    // invoking singleStep with neither a resume_run_id nor a live checkpoint
    // fails loudly instead of silently mis-capping a run that was never
    // actually resumed from anything.
    throw new Error("singleStep was requested but no live checkpoint was found to resume from.");
  } else {
    if (!owner || !repo || !branch || !task) {
      throw new Error("owner, repo, branch, and task are all required on a fresh call (not resuming).");
    }

    // Guardrail #2, checked once up front, before the loop starts --
    // reuses editor_tool_functions.js's own live lookup rather than
    // duplicating it.
    await assertNotDefaultBranch(owner, repo, branch);

    runId = randomUUID();
    contents = [{ role: "user", parts: [{ text: appendTask(buildEditorPreamble({ owner, repo, branch }), task) }] }];
    transcript = [];
    startStep = 1;
    writtenFiles = [];
    writesPerFile = new Map();
    validateCounts = new Map();
  }

  // Establish this call's own loop bound (cappedSteps) and the run's true
  // ceiling (effectiveOverallMaxSteps) now that startStep/checkpoint are
  // known. singleStep (the editor worker's one-step-per-invocation resume)
  // deliberately does NOT let this call's own max_steps redefine the run's
  // real ceiling -- it restores that ceiling from the checkpoint instead,
  // and bounds only THIS call's loop to a single iteration. Every other
  // caller (a fresh run, or a manual synchronous resume) keeps the existing,
  // documented behavior: max_steps sets/updates the real ceiling directly.
  if (singleStep) {
    // Checkpoints seeded/saved before this field existed (or a fresh run
    // that reaches here, which can't happen given the branch above, but
    // kept as a safe fallback) won't have it -- fall back to this call's own
    // max_steps rather than leaving the run's real ceiling undefined.
    effectiveOverallMaxSteps = checkpoint.overallMaxSteps || Math.min(max_steps, EDITOR_HARD_MAX_STEPS);
    cappedSteps = startStep;
  } else {
    cappedSteps = Math.min(max_steps, EDITOR_HARD_MAX_STEPS);
    effectiveOverallMaxSteps = cappedSteps;
  }

  const { FUNCTIONS, declarations } = buildFunctions({
    owner: effectiveOwner, repo: effectiveRepo, branch: effectiveBranch, writtenFiles, writesPerFile, validateCounts,
  });

  if (checkpoint && startStep > cappedSteps) {
    return {
      answer: `(This run already completed ${startStep - 1} step(s), which meets or exceeds the requested max_steps of ${cappedSteps} -- no new steps were taken this call. The checkpoint has NOT been discarded. Call again with resume_run_id: "${runId}" and a higher max_steps to continue.)`,
      steps: startStep - 1,
      transcript,
      runId,
      task: effectiveTask,
      writtenFiles,
      failed: true,
    };
  }

  const saveState = (stepsDone) => saveCheckpoint(runId, {
    contents,
    transcript,
    stepsDone,
    task: effectiveTask,
    owner: effectiveOwner,
    repo: effectiveRepo,
    branch: effectiveBranch,
    writtenFiles,
    writesPerFile: Object.fromEntries(writesPerFile),
    validateCounts: Object.fromEntries(validateCounts),
    repeatCounts: Object.fromEntries(repeatCounts),
    consecutiveAllRepeatSteps,
    overallMaxSteps: effectiveOverallMaxSteps,
    provider: effectiveProvider,
    pendingVerification,
    fallbackModelUsed,
  });

  for (let step = startStep; step <= cappedSteps; step++) {
    // Withhold tools on the final step, same reasoning/mechanism as
    // designer_delegate.js: structurally forces a plain-text answer instead
    // of an unexecuted function call. Compared against effectiveOverallMaxSteps
    // (the run's TRUE ceiling), not cappedSteps (this call's own loop bound) --
    // on a singleStep resume the two differ, and isFinalStep must reflect
    // whether this is really the run's last step, not just this call's only
    // step. See effectiveOverallMaxSteps's own comment above.
    const isFinalStep = step === effectiveOverallMaxSteps;
    const stuckLoopForce = consecutiveAllRepeatSteps >= 3;
    const withholdTools = isFinalStep || stuckLoopForce;

    let candidate;
    try {
      candidate = await providerChat(contents, { tools: withholdTools ? undefined : declarations, provider: effectiveProvider });
      const cascadeLog = formatCascadeLogLine(candidate, { step });
      if (cascadeLog) transcript.push(cascadeLog);
      if (candidate._fallbackModelUsed) fallbackModelUsed = candidate._fallbackModelUsed;
    } catch (err) {
      await saveState(step - 1);
      const redisOk = isRedisConfigured();
      const resumeHint = isTransientGeminiError(err)
        ? (redisOk
            ? ` ${transcript.length} tool call(s) already completed this run are saved. Call again with resume_run_id: "${runId}" to continue instead of starting over. Checkpoint expires in 1 hour.`
            : ` ${transcript.length} tool call(s) were completed this run, but Redis is NOT configured, so nothing was actually saved -- resume_run_id: "${runId}" will NOT work. Re-run from scratch with the full task text.`)
        : ` This does not look like a transient error (not a 429/503) -- resuming will likely reproduce the same failure. Check the underlying cause before retrying.`;
      return {
        answer: `(Gemini call failed on step ${step}: ${err?.message ?? String(err)} --${resumeHint})`,
        steps: step - 1,
        transcript,
        runId,
        task: effectiveTask,
        writtenFiles,
        failed: true,
      };
    }

    const parts = candidate.content?.parts || [];
    const functionCalls = parts.filter((p) => p.functionCall);

    if (withholdTools && functionCalls.length) {
      await saveState(step - 1);
      const reason = stuckLoopForce
        ? `the agent appeared stuck repeating the same call(s) for ${consecutiveAllRepeatSteps} consecutive steps, so tools were withheld to force a plain-text answer instead of continuing to loop`
        : `the model attempted a function call on the final step, where no tools are available`;
      return {
        answer: `(Run stopped after reaching the step cap of ${cappedSteps}: ${reason}, so it was discarded rather than executed -- the task may need to be narrowed, or more steps requested up to the hard cap of ${EDITOR_HARD_MAX_STEPS}. ${transcript.length} tool call(s) already completed this run are saved. Call again with resume_run_id: "${runId}" to continue instead of starting over. Checkpoint expires in 1 hour.)`,
        steps: step - 1,
        transcript,
        runId,
        task: effectiveTask,
        writtenFiles,
        failed: true,
      };
    }

    if (!functionCalls.length) {
      const answer = parts.map((p) => p.text || "").join("").trim();
      if (!answer) {
        await saveState(step - 1);
        const starvationNote = withholdTools && candidate.finishReason === "MALFORMED_FUNCTION_CALL"
          ? (stuckLoopForce
              ? ` Tools were withheld this step because the agent appeared stuck repeating the same call(s) for ${consecutiveAllRepeatSteps} consecutive steps, but the model attempted a function call anyway, which Gemini rejects as malformed when no tools are available.`
              : ` This was the final allowed step, which never includes tools -- but the model attempted a function call anyway, which Gemini rejects as malformed when no tools are available. This usually means the task needed more steps than max_steps (${cappedSteps}) allowed. Retry with a higher max_steps.`)
          : "";
        return {
          answer: `(Gemini stopped without a final answer -- finishReason: ${candidate.finishReason || "unknown"})${starvationNote} ${transcript.length} tool call(s) already completed this run are saved. Call again with resume_run_id: "${runId}" to continue instead of starting over. Checkpoint expires in 1 hour.`,
          steps: step - 1,
          transcript,
          runId,
          task: effectiveTask,
          writtenFiles,
          failed: true,
        };
      }
      // Writes-vs-claim verification pass -- see buildEditorVerificationPrompt's
      // header comment for the incident this exists to fix. Fires at most
      // once per run (guarded by !pendingVerification): a draft answer that
      // arrives with tool access still available (not already a forced
      // no-tools turn) and step budget left gets sent back for one
      // corrective round BEFORE it's trusted, checking it against
      // writtenFiles and any tool-result text already gathered. Tools stay
      // ENABLED this turn (unlike isFinalStep/stuckLoopForce, which
      // deliberately withhold them to force a stop) -- same reasoning as
      // agent_delegate.js's own verification pass: a model asked to
      // self-check purely from memory just re-asserts its own mistake with
      // equal confidence, but tool access lets it actually re-read the file
      // or make the write it claimed, rather than guess.
      if (!withholdTools && !pendingVerification && step < cappedSteps) {
        const verificationPrompt = await buildEditorVerificationPrompt({ answer, contents, writtenFiles });
        contents.push({ role: "model", parts });
        contents.push({ role: "user", parts: [{ text: verificationPrompt }] });
        pendingVerification = true;
        await saveState(step);
        continue;
      }

      // Persist a "done" checkpoint (status + finalAnswer) here instead of
      // deleting it -- a resume_run_id caller polling a background/worker-
      // driven run needs SOMETHING to read once the run finishes, and
      // deleting it the instant it completes would mean there's never a
      // window in which a "done" status could be observed. Still expires via
      // the normal checkpoint TTL like any other checkpoint -- this is a
      // bounded-lifetime record for polling, not a permanent store. Mirrors
      // connectors/delegate/agent/agent_delegate.js's finishRun, which made the same
      // change and no longer calls deleteCheckpoint on its success path
      // either.
      await saveCheckpoint(runId, {
        contents,
        transcript,
        stepsDone: step,
        task: effectiveTask,
        owner: effectiveOwner,
        repo: effectiveRepo,
        branch: effectiveBranch,
        writtenFiles,
        writesPerFile: Object.fromEntries(writesPerFile),
        validateCounts: Object.fromEntries(validateCounts),
        repeatCounts: Object.fromEntries(repeatCounts),
        consecutiveAllRepeatSteps,
        overallMaxSteps: effectiveOverallMaxSteps,
        provider: effectiveProvider,
        pendingVerification,
        fallbackModelUsed,
        status: "done",
        finalAnswer: answer,
      });
      return { answer, steps: step, transcript, runId, task: effectiveTask, writtenFiles, fallbackModelUsed, failed: false };
    }

    contents.push({ role: "model", parts });

    let responseParts;
    try {
      // Parallelized for the same reason as designer_delegate.js: every
      // call batched into one model turn was decided without seeing any
      // of the others' results. write_file is NEVER cache-served (has a
      // real side effect -- a commit); read_file is safe to cache-serve on
      // an exact repeat, same distinction designer_delegate.js draws.
      const CACHEABLE_TOOLS = new Set(["read_file", "validate"]);

      const results = await Promise.all(functionCalls.map(async (part) => {
        const { name, args, id } = part.functionCall;
        const signature = `${name}:${JSON.stringify(args || {})}`;
        const priorCount = repeatCounts.get(signature) || 0;
        const isRepeat = priorCount > 0;
        repeatCounts.set(signature, priorCount + 1);

        const fn = FUNCTIONS.find((f) => f.name === name);
        let resultText;
        let servedFromCache = false;
        if (isRepeat && CACHEABLE_TOOLS.has(name) && resultCache.has(signature)) {
          resultText = resultCache.get(signature);
          servedFromCache = true;
        } else if (!fn) {
          resultText = `Error: unknown function "${name}".`;
        } else {
          try {
            resultText = await fn.execute(args || {});
          } catch (err) {
            resultText = `Error: ${err?.message ?? String(err)}`;
          }
        }
        if (typeof resultText !== "string") {
          resultText = `Error: ${name} returned a non-string result (${typeof resultText}); this is a bug in its execute().`;
        }
        if (!servedFromCache && CACHEABLE_TOOLS.has(name)) {
          resultCache.set(signature, resultText);
        }
        return { name, args, id, resultText, isRepeat, servedFromCache };
      }));

      // Invalidate any cached read_file result for a path this step just
      // wrote to successfully -- same reasoning as designer_delegate.js:
      // without this, a confirming re-read right after a write would be
      // served stale pre-write content.
      for (const r of results) {
        if (r.name === "write_file" && !r.resultText.startsWith("Error:") && r.args?.path) {
          const readSignature = `read_file:${JSON.stringify({ path: r.args.path })}`;
          resultCache.delete(readSignature);
          repeatCounts.delete(readSignature);
        }
      }

      const allRepeatsThisStep = results.length > 0 && results.every((r) => r.isRepeat);
      consecutiveAllRepeatSteps = allRepeatsThisStep ? consecutiveAllRepeatSteps + 1 : 0;

      responseParts = results.map((r) => {
        const cacheNote = r.servedFromCache ? " [served from cache -- identical call already made this run]" : "";
        transcript.push(`[step ${step}] ${r.name}(${JSON.stringify(r.args || {})})${cacheNote} -> ${r.resultText.length > 300 ? r.resultText.slice(0, 300) + "…" : r.resultText}`);
        return { functionResponse: { name: r.name, id: r.id, response: { result: r.resultText } } };
      });
    } catch (err) {
      await saveState(step - 1);
      return {
        answer: `(Unexpected error while processing step ${step}'s function calls: ${err?.message ?? String(err)} -- ${transcript.length} tool call(s) already completed this run are saved. Call again with resume_run_id: "${runId}" to continue.)`,
        steps: step - 1,
        transcript,
        runId,
        task: effectiveTask,
        writtenFiles,
        failed: true,
      };
    }

    if (consecutiveAllRepeatSteps === 2) {
      responseParts.push({
        text: "[SYSTEM NOTE: the last 2 steps consisted entirely of calls identical to ones already made this run. One more step like that and tools will be withheld to force a plain-text answer instead. If you're re-reading to double-check, that's fine once -- but if you're retrying the same write and getting the same result, stop and explain what's blocking it instead of repeating the call.]",
      });
    }

    // Halfway-point nudge: fires exactly once, on the step that lands at
    // the run's true halfway mark, and only if nothing has been written
    // yet -- a budget-aware check-in rather than upfront pressure not to
    // over-read (see buildSystemPreamble, which now explicitly permits
    // reading as long as needed). Compared against effectiveOverallMaxSteps
    // (the run's TRUE ceiling), not cappedSteps, for the same reason
    // isFinalStep above does -- correct across a singleStep/async resume,
    // where cappedSteps is only this call's own shrunk loop bound.
    const halfwayPoint = Math.ceil(effectiveOverallMaxSteps / 2);
    if (step === halfwayPoint && writtenFiles.length === 0) {
      responseParts.push({
        text: `[SYSTEM NOTE: you're ${halfwayPoint} step(s) into a ${effectiveOverallMaxSteps}-step budget and haven't written any files yet. If you've gathered enough context, start making the actual edits now -- you still have steps left, but not unlimited ones.]`,
      });
    }

    const remainingAfterThisStep = cappedSteps - step;
    if (remainingAfterThisStep === 1) {
      responseParts.push({
        text: "[SYSTEM NOTE: only 1 step remains after this one, and the step after that has NO tools available. Finish any in-progress write now if the file is ready, or explain what's left undone -- do not leave a task half-written without saying so.]",
      });
    } else if (remainingAfterThisStep === 0) {
      responseParts.push({
        text: "[SYSTEM NOTE: the next turn will NOT include any tools -- you must answer now in plain text summarizing what you changed (or didn't, and why) rather than attempting another function call.]",
      });
    }

    contents.push({ role: "user", parts: responseParts });
    await saveState(step);
  }

  // Defensive fallback only -- see designer_delegate.js's identical comment
  // for why this should no longer be reachable in normal operation.
  return {
    answer: `(Run stopped after reaching the step cap of ${cappedSteps} without a final answer -- the task may need to be narrowed, or more steps requested up to the hard cap of ${EDITOR_HARD_MAX_STEPS}. ${transcript.length} tool call(s) already completed this run are saved. Call again with resume_run_id: "${runId}" to continue instead of starting over. Checkpoint expires in 1 hour.)`,
    steps: cappedSteps,
    transcript,
    runId,
    task: effectiveTask,
    writtenFiles,
    failed: true,
  };
}

// Seeds a fresh checkpoint (status "running", stepsDone 0, no steps taken
// yet) WITHOUT running any part of the editor loop -- mirrors
// connectors/delegate/agent/agent_delegate.js's seedRun. Meant for the upcoming
// editor_tools.js async-start path (Step 7) to return a run_id immediately
// and let the editor worker (Step 4) take step 1 in the background, rather
// than this call itself blocking on step 1 synchronously before returning.
//
// Runs assertNotDefaultBranch up front -- same guardrail #2 check
// runEditorAgent's own fresh-call branch already does -- BEFORE ever
// writing a checkpoint, so an invalid branch never gets a run_id at all:
// there would be nothing useful to resume, and a caller polling a run_id
// that can never succeed is worse than an immediate, synchronous rejection.
//
// Deliberately duplicates the small fresh-run setup already inside
// runEditorAgent (a UUID + the initial system-preamble turn) rather than
// calling into runEditorAgent with max_steps: 0, for the same reason
// seedRun gives for its own equivalent duplication: runEditorAgent's loop
// simply never executes when cappedSteps < startStep, but the only existing
// early-return path that covers that case (`checkpoint && startStep >
// cappedSteps`) assumes a checkpoint ALREADY EXISTS -- reaching it on a
// genuinely fresh, zero-step call would mean either throwing or bending
// that guard's contract to serve a second, differently-shaped caller. A
// small, explicit duplication of the fresh-run setup here is lower-risk.
//
// Unlike editor_checkpoint.js's other callers, this writes the whole-blob
// shape directly (contents/writtenFiles/writesPerFile/validateCounts etc.)
// rather than through runEditorAgent's saveState closure, since there is no
// loop-scoped closure to reuse here -- the shape matches saveState's own
// object exactly, just with the zero-step initial values.
export async function seedEditorRun({ owner, repo, branch, task, max_steps = EDITOR_DEFAULT_STEPS, provider }) {
  if (!owner || !repo || !branch || !task) {
    throw new Error("owner, repo, branch, and task are all required to seed a new run.");
  }

  await assertNotDefaultBranch(owner, repo, branch);

  const runId = randomUUID();
  const contents = [{ role: "user", parts: [{ text: appendTask(buildEditorPreamble({ owner, repo, branch }), task) }] }];
  // Seeds the run's TRUE overall step ceiling (see runEditorAgent's
  // effectiveOverallMaxSteps for the full rationale) -- this is what lets
  // the editor worker's later singleStep resumes know when they've reached
  // the run's real last step, instead of mistaking their own artificially
  // shrunk per-call max_steps for it.
  const overallMaxSteps = Math.min(max_steps, EDITOR_HARD_MAX_STEPS);
  await saveCheckpoint(runId, {
    contents,
    transcript: [],
    stepsDone: 0,
    task,
    owner,
    repo,
    branch,
    writtenFiles: [],
    writesPerFile: {},
    validateCounts: {},
    repeatCounts: {},
    consecutiveAllRepeatSteps: 0,
    overallMaxSteps,
    provider,
    status: "running",
  });
  return runId;
}
