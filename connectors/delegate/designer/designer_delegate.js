// ---------------------------------------------------------------------------
// connectors/delegate/designer/designer_delegate.js — delegate_designer's tool-calling agent
// loop (issue #61, Notion "madmcp-delegate-designer-v2-plan" design doc,
// step 2: "Agent loop wiring").
//
// Adapts the delegate_agent loop's runInvestigation loop -- multi-step
// Gemini function-calling, not a single one-shot completion -- but restricted
// to exactly three tools (read_file / write_file / validate, all from
// connectors/frontend/designer_tool_functions.js, built in step 1) instead of
// delegate_agent's large read-only cross-system surface, and WRITE-capable
// rather than read-only.
//
// SCOPE FENCING, ENFORCED AT THE TOOL LAYER (per issue #61 -- "not just
// prompt instructions"): owner/repo/branch are fixed for the whole run and
// are NOT parameters the model can set via a function call -- the FUNCTIONS
// closures below bind them from runDesignAgent's own arguments, so there is
// no code path for the model to redirect a read/write at a different
// repo/branch than the one this run was started against. Extension fencing
// (FRONTEND_ALLOWED_EXTENSIONS) is enforced one level down, inside
// designer_tool_functions.js's readFile/writeFile themselves -- not repeated here.
// Default-branch refusal is checked once up front (connectors/delegate/designer/
// designer_tools.js).
//
// NOT YET WIRED TO AN MCP TOOL: this file exports runDesignAgent as a plain
// function, unit-testable independently (mirrors step 1's "build the tools
// layer, unit test each independently of the agent loop" -- this step does
// the analogous thing one level up: build the loop, unit test it
// independently of any server.tool(...) registration). MCP registration
// is step 5 in the design doc, not this step.
// ---------------------------------------------------------------------------

import { randomUUID } from "node:crypto";
import { providerChat } from "../../llm/router.js";
import { readFile, writeFile, validate as validateFile } from "../../frontend/designer_tool_functions.js";
import { saveCheckpoint, loadCheckpoint, deleteCheckpoint } from "./designer_checkpoint.js";
import { isRedisConfigured } from "../../shared/cooldown.js";
import { githubRequest } from "../../github/client.js";
import {
  FRONTEND_ALLOWED_EXTENSIONS,
  FRONTEND_DEFAULT_STEPS,
  FRONTEND_HARD_MAX_STEPS,
  FRONTEND_MAX_VALIDATE_CALLS,
} from "../../../config.js";
import { appendTask } from "../shared/preamble.js";

// Same reasoning as the agent loop's isTransientGeminiError:
// only 429 (rate limit) and 503 (overloaded) are worth resuming past --
// everything else (bad request, auth, missing key) will reproduce
// identically on a resume.
function isTransientGeminiError(err) {
  return err?.status === 429 || err?.status === 503 || err?.transient === true;
}

function buildSystemPreamble({ owner, repo, branch }) {
  return (
    "You are a frontend/UI design agent working inside ONE fixed repository and branch. You may " +
    `read and write files with these extensions only: ${FRONTEND_ALLOWED_EXTENSIONS.join(", ")}. ` +
    `Repository: ${owner}/${repo}. Branch: ${branch} (already confirmed to not be the default branch).\n\n` +
    "You have three tools:\n" +
    "- read_file(path): reads a file's current content on this branch, together with its blob sha. " +
    "Always read a file before patching it -- write_file's patch mode requires the exact sha the " +
    "content was read from.\n" +
    "- write_file(path, content OR patch, base_sha, message): writes a file. Give `content` for a full " +
    "overwrite, or `patch` (a list of {find, replace} operations, each `find` must appear exactly once) " +
    "to edit part of a file you already read. `base_sha` is required for patch mode, and required for " +
    "content mode too whenever you are replacing a file you already read (omit it only when creating a " +
    "brand-new file that doesn't exist yet). If a write is rejected as a conflict, the file changed " +
    "since you read it -- re-read it and retry, don't assume your version is still current.\n" +
    "- validate(path, content): syntax-checks content against its file type before you write it. Not " +
    "free of limits -- capped per file, so don't call it more than genuinely useful; a couple of passes " +
    "per file is normal, looping it dozens of times is not.\n\n" +
    "Work iteratively: read what you need, make changes, validate before writing when it's cheap to do " +
    "so, write, and confirm the result makes sense. When the task is fully done, respond with a final " +
    "plain-text summary of what you changed and no further function calls."
  );
}

// Builds the three function declarations + their execute() closures for one
// run. owner/repo/branch are captured here, NOT exposed as parameters the
// model can set -- see file header.
function buildFunctions({ owner, repo, branch, validateCounts, writtenFiles }) {
  const FUNCTIONS = [
    {
      name: "read_file",
      description: "Read a file's current content on this run's branch, together with its blob sha (needed for write_file's base_sha).",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: `File path within the repo. Must end in one of: ${FRONTEND_ALLOWED_EXTENSIONS.join(", ")}` },
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
      description: "Write a file on this run's branch. Exactly one of `content` (full overwrite) or `patch` (find/replace operations) is required. `base_sha` is required for `patch`, and required for `content` too unless this is a brand-new file with no prior read_file call.",
      parameters: {
        type: "object",
        properties: {
          path:    { type: "string", description: `File path within the repo. Must end in one of: ${FRONTEND_ALLOWED_EXTENSIONS.join(", ")}` },
          content: { type: "string", description: "Full new file content (mutually exclusive with patch)" },
          patch:   {
            type: "array",
            description: "List of find/replace operations, applied sequentially (mutually exclusive with content)",
            items: {
              type: "object",
              properties: {
                find:    { type: "string" },
                replace: { type: "string" },
              },
              required: ["find", "replace"],
            },
          },
          base_sha: { type: "string", description: "The sha returned by a prior read_file call on this exact path" },
          message:  { type: "string", description: "Commit message (optional -- a reasonable default is used if omitted)" },
        },
        required: ["path"],
      },
      execute: async ({ path, content, patch, base_sha, message }) => {
        try {
          const result = await writeFile(owner, repo, path, { content, patch, baseSha: base_sha, branch, message });
          writtenFiles.push(result.path);
          return `Wrote ${result.path} (commit ${result.commitSha.slice(0, 7)}, new sha ${result.sha}).`;
        } catch (err) {
          // Conflict errors (designer_tool_functions.js's `.conflict = true`) are a
          // normal, expected outcome the model should react to (re-read,
          // re-diff, retry) -- per the design doc, NOT a hard tool failure.
          // Returning the message as a regular string result (rather than
          // throwing) is what lets the loop's existing error-string
          // convention carry it back to the model as a next-turn input,
          // same as any other tool result.
          return `Error: ${err.message}`;
        }
      },
    },
    {
      name: "validate",
      description: "Syntax-check content against its file type (by extension) before writing. Capped per file path -- see the system instructions.",
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
        if (count >= FRONTEND_MAX_VALIDATE_CALLS) {
          return `Error: validate() has already been called ${count} time(s) for "${path}", which is this run's per-file cap (${FRONTEND_MAX_VALIDATE_CALLS}). Proceed without further validation of this file, or write it and reconsider your approach if it's still not right.`;
        }
        validateCounts.set(path, count + 1);
        const result = await validateFile(path, content);
        return result.valid ? "Valid -- no syntax errors found." : `Invalid:\n${result.errors.join("\n")}`;
      },
    },
  ];

  const declarations = [{
    functionDeclarations: FUNCTIONS.map(({ name, description, parameters }) => ({ name, description, parameters })),
  }];

  return { FUNCTIONS, declarations };
}

// Runs the write-capable design agent loop. Returns
// { answer, steps, transcript, runId, writtenFiles, task, failed? } --
// same overall shape as the agent loop's runInvestigation,
// so a future MCP-facing tool (step 5) can follow the same
// resume_run_id/failed-response conventions delegate_agent already uses.
//
// On a fresh call, owner/repo/branch/task are required; on a resume
// (resume_run_id set), they're restored from the checkpoint and any passed
// values are ignored, matching the agent loop's own resume
// contract (see its comments for why `task` specifically must never be
// trusted over the checkpoint's own record of it on a live resume).
export async function runDesignAgent({ owner, repo, branch, task, max_steps = FRONTEND_DEFAULT_STEPS, resume_run_id }) {
  const cappedSteps = Math.min(max_steps, FRONTEND_HARD_MAX_STEPS);

  let runId = resume_run_id;
  let contents;
  let transcript;
  let startStep;
  let validateCounts;
  let writtenFiles;
  let effectiveOwner = owner;
  let effectiveRepo = repo;
  let effectiveBranch = branch;
  let effectiveTask = task;
  // Stuck-loop detection (mirrors the agent loop's fix #4):
  // repeatCounts tracks how many times each exact (function name + JSON-
  // stringified args) signature has been called THIS RUN, persisted across
  // resumes so a resumed run doesn't forget what it already tried.
  // resultCache holds the actual result text per signature -- deliberately
  // NOT persisted in the checkpoint (same reasoning as agent_delegate.js: keeps
  // checkpoint writes small; a resumed run re-executing one exact-repeat
  // call and re-caching it is a correctness no-op). consecutiveAllRepeatSteps
  // counts how many steps IN A ROW consisted ENTIRELY of repeat calls -- a
  // single repeat mixed with new calls is normal exploration, not stuck.
  let repeatCounts = new Map();
  let resultCache = new Map();
  let consecutiveAllRepeatSteps = 0;

  const checkpoint = resume_run_id ? await loadCheckpoint(resume_run_id) : null;

  if (checkpoint) {
    contents = checkpoint.contents;
    transcript = checkpoint.transcript;
    startStep = checkpoint.stepsDone + 1;
    validateCounts = new Map(Object.entries(checkpoint.validateCounts || {}));
    writtenFiles = checkpoint.writtenFiles || [];
    effectiveOwner = checkpoint.owner;
    effectiveRepo = checkpoint.repo;
    effectiveBranch = checkpoint.branch;
    effectiveTask = checkpoint.task;
    // Checkpoints saved before this fix existed won't have these fields --
    // fall back to empty/zero rather than erroring, same defensive pattern
    // as validateCounts/writtenFiles above.
    repeatCounts = new Map(Object.entries(checkpoint.repeatCounts || {}));
    consecutiveAllRepeatSteps = checkpoint.consecutiveAllRepeatSteps || 0;
  } else if (resume_run_id) {
    // A resume was requested but its checkpoint didn't load (expired past
    // the 1-hour TTL, Redis unavailable, or an invalid/typo'd runId). Same
    // "fail loudly and distinctly" reasoning as the agent loop's
    // agent_delegate.js -- silently falling through to a fresh run here would
    // require owner/repo/branch/task to have been re-supplied anyway (this
    // loop, unlike agent_delegate.js, has no task-optional fallback path), so
    // there's no ambiguous case to accommodate -- always an error.
    throw new Error(
      isRedisConfigured()
        ? `resume_run_id "${resume_run_id}" has no live checkpoint -- it may have expired (1 hour TTL) or the id may be wrong. Start a new run with owner/repo/branch/task instead.`
        : `resume_run_id "${resume_run_id}" has no live checkpoint -- and Redis is NOT configured in this environment, so no checkpoint could ever have been saved. Start a new run with owner/repo/branch/task instead.`
    );
  } else {
    if (!owner || !repo || !branch || !task) {
      throw new Error("owner, repo, branch, and task are all required on a fresh call (not resuming).");
    }

    let repoInfo;
    try {
      repoInfo = await githubRequest(`/repos/${owner}/${repo}`);
    } catch (err) {
      throw new Error(`Failed to look up ${owner}/${repo}: ${err.message}`, { cause: err });
    }
    if (branch === repoInfo.default_branch) {
      throw new Error(`Refusing to run: "${branch}" is ${owner}/${repo}'s default branch. Create/use a feature branch instead -- this agent never writes directly to the default branch.`);
    }

    runId = randomUUID();
    contents = [{ role: "user", parts: [{ text: appendTask(buildSystemPreamble({ owner, repo, branch }), task) }] }];
    transcript = [];
    startStep = 1;
    validateCounts = new Map();
    writtenFiles = [];
  }

  const { FUNCTIONS, declarations } = buildFunctions({
    owner: effectiveOwner, repo: effectiveRepo, branch: effectiveBranch, validateCounts, writtenFiles,
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
    validateCounts: Object.fromEntries(validateCounts),
    writtenFiles,
    repeatCounts: Object.fromEntries(repeatCounts),
    consecutiveAllRepeatSteps,
  });

  for (let step = startStep; step <= cappedSteps; step++) {
    // Withhold tools on the final step so the model is structurally forced
    // to answer in plain text instead of attempting one more function call
    // that never gets to run -- same fix the agent loop's agent_delegate.js
    // applies for the identical reason (a text-only reminder alone wasn't
    // reliable enough there either).
    const isFinalStep = step === cappedSteps;
    // Same withholding, second trigger: 3 consecutive steps that were
    // ENTIRELY repeat calls means the model is stuck re-trying the same
    // thing rather than making progress -- force a plain-text answer now
    // instead of letting it burn the rest of the step budget the same way.
    // Mirrors the agent loop's agent_delegate.js fix #4; unlike that file, a
    // stuck loop here can't be quietly served from cache indefinitely
    // because write_file is never cache-served (see below) -- it would
    // otherwise keep spending real GitHub API calls, not just wasted model
    // turns.
    const stuckLoopForce = consecutiveAllRepeatSteps >= 3;
    const withholdTools = isFinalStep || stuckLoopForce;

    let candidate;
    try {
      candidate = await providerChat(contents, { tools: withholdTools ? undefined : declarations });
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

    // Guard against a step where tools were withheld (final step OR stuck-
    // loop force): the model can't legitimately act here, but Gemini does
    // not always reject an attempted function call API-side (the
    // MALFORMED_FUNCTION_CALL path below covers when it does -- sometimes
    // it just returns a function call anyway despite no tools being
    // declared). Discard it unexecuted rather than running it, or the
    // "no tools means nothing acts" guarantee this withholding exists for
    // doesn't actually hold.
    if (withholdTools && functionCalls.length) {
      await saveState(step - 1);
      const reason = stuckLoopForce
        ? `the agent appeared stuck repeating the same call(s) for ${consecutiveAllRepeatSteps} consecutive steps, so tools were withheld to force a plain-text answer instead of continuing to loop`
        : `the model attempted a function call on the final step, where no tools are available`;
      return {
        answer: `(Run stopped after reaching the step cap of ${cappedSteps}: ${reason}, so it was discarded rather than executed -- the task may need to be narrowed, or more steps requested up to the hard cap of ${FRONTEND_HARD_MAX_STEPS}. ${transcript.length} tool call(s) already completed this run are saved. Call again with resume_run_id: "${runId}" to continue instead of starting over. Checkpoint expires in 1 hour.)`,
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
        // No text and no function calls -- the model didn't actually
        // finish. Keep the checkpoint alive (don't delete it) and mark
        // this failed so the caller gets a real, usable resume_run_id --
        // matching the resume contract used everywhere else in this loop
        // (Gemini call errors, mid-step processing errors). Previously
        // this path deleted the checkpoint and omitted `failed`, which is
        // why a run stopping here never actually surfaced a usable
        // resume_run_id despite the message implying resumability.
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
      await deleteCheckpoint(runId);
      return { answer, steps: step, transcript, runId, task: effectiveTask, writtenFiles };
    }

    contents.push({ role: "model", parts });

    let responseParts;
    try {
      // Parallelized for the same reason as the agent loop's agent_delegate.js:
      // every call batched into one model turn was decided without seeing
      // any of the others' results, so awaiting them concurrently changes
      // only wall-clock time, not what information was available to what
      // call. Unlike agent_delegate.js's read-only tool set, write_file has a
      // real side effect (a commit) -- but two write_file calls in the same
      // batched turn would already be targeting different paths in any
      // sane model plan (the model has no way to make a second write
      // depend on the first write's result within the same turn either
      // way), so this doesn't introduce a new ordering hazard beyond what
      // agent_delegate.js already accepts for its own batched calls.
      // Only read_file and validate are safe to serve from cache on an exact repeat -- both are pure reads with no side effect, so serving a cached result changes nothing about what actually happened. write_file is NEVER cache-served: it has a real side effect (a commit), and silently skipping that on a "repeat" call would let the model believe a write happened when it didn't -- exactly the kind of silent-mismatch bug this whole file's other fixes exist to prevent. An identical write_file call is instead just executed again for real (its repeat count still contributes to stuck-loop detection below, so repeatedly retrying the same failing write still gets caught and stopped -- it just isn't executed from cache while that's happening).
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

      // Invalidate any cached read_file result for a path this step just wrote to successfully. Without this, a read_file call AFTER a write_file to the same path -- an entirely reasonable thing for the model to do, e.g. to confirm what actually landed -- would be treated as an "exact repeat" of an earlier read_file call on that path (the cache key is just the path, not the content) and served stale PRE-write content instead of what is actually on the branch now. Also clears repeatCounts for that signature, not just the cache entry: otherwise the next read would still be flagged isRepeat (misclassifying a genuinely-necessary re-read as a stuck-loop repeat) even though it is forced to execute for real. A failed write_file (its result starts with "Error:") changed nothing on the branch, so it does NOT invalidate anything.
      for (const r of results) {
        if (r.name === "write_file" && !r.resultText.startsWith("Error:") && r.args?.path) {
          const readSignature = `read_file:${JSON.stringify({ path: r.args.path })}`;
          resultCache.delete(readSignature);
          repeatCounts.delete(readSignature);
        }
      }

      // Stuck-loop tracking: this step counts as "all repeat" only if EVERY call in it was already seen before -- a step that mixes a repeat with a genuinely new call is normal exploration (e.g. re-reading one file while reading a second file for the first time), not a stuck loop.
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

  // Defensive fallback only -- with the isFinalStep guard above, the loop
  // should always return from inside its final iteration (either a real
  // answer, a discarded final-step function call, or a starved no-answer
  // response), so falling out of the for loop here should no longer be
  // reachable in normal operation. Kept as a safety net in case future
  // changes reintroduce a path that falls through, using the same
  // keep-checkpoint-alive-and-mark-failed resume contract as every other
  // stopped-without-an-answer path in this file.
  return {
    answer: `(Run stopped after reaching the step cap of ${cappedSteps} without a final answer -- the task may need to be narrowed, or more steps requested up to the hard cap of ${FRONTEND_HARD_MAX_STEPS}. ${transcript.length} tool call(s) already completed this run are saved. Call again with resume_run_id: "${runId}" to continue instead of starting over. Checkpoint expires in 1 hour.)`,
    steps: cappedSteps,
    transcript,
    runId,
    task: effectiveTask,
    writtenFiles,
    failed: true,
  };
}
