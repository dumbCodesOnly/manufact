// ---------------------------------------------------------------------------
// connectors/delegate/editor/editor_tools.js -- MCP tool registration for delegate_editor
// ("Limited GitHub write access for delegate_agent (non-default-branch only)").
//
// Thin wrapper around runEditorAgent (editor_delegate.js), same shape as
// connectors/delegate/designer/designer_tools.js's delegate_designer wrapper: same
// resume_run_id / max_steps / show_transcript conventions, same
// writtenFiles/transcript response shaping (guardrail #9's audit trail --
// the transcript/writtenFiles reporting IS the audit trail, per the
// note that mandatory Notion logging was considered and dropped as
// redundant with this).
//
// GATED BEHIND EDITOR_AGENT_ENABLED (the rollout flag):
// register() below is a no-op unless the flag is "true", so this tool
// simply doesn't exist on the MCP surface until a human flips it on
// deliberately -- same "disable without a revert" posture as
// DELEGATE_AGENT_ASYNC elsewhere in this repo. Flip via env var, not code,
// once the test suite and a manual smoke test both look right.
//
// Tool description is deliberately as explicit about scope limits as
// delegate_agent's own description is about being read-only -- the calling
// model needs to know, from the description alone, that this tool can only
// write to a non-default branch it doesn't get to choose past guardrails #1/#2,
// before it ever calls it.
// ---------------------------------------------------------------------------

import { z } from "zod";
import { runEditorAgent, seedEditorRun } from "./editor_delegate.js";
import { loadCheckpoint } from "./editor_checkpoint.js";
import { publishEditorStep, isEditorQStashConfigured } from "../qstash_client.js";
import {
  DEFAULT_OWNER,
  EDITOR_ALLOWED_EXTENSIONS,
  EDITOR_ALLOWED_PATH_PREFIXES,
  EDITOR_DEFAULT_STEPS,
  EDITOR_HARD_MAX_STEPS,
  EDITOR_MAX_FILES_PER_RUN,
  EDITOR_MAX_WRITES_PER_FILE,
  EDITOR_AGENT_ENABLED,
  EDITOR_AGENT_ASYNC,
  EDITOR_ASYNC_POLL_FRESH_SECONDS,
  EDITOR_ASYNC_STEP_DEAD_SECONDS,
} from "../../../config.js";

function scopeSummary() {
  const pathPart = EDITOR_ALLOWED_PATH_PREFIXES.length
    ? `; restricted to paths under: ${EDITOR_ALLOWED_PATH_PREFIXES.join(", ")}`
    : "; no additional path restriction beyond the deny list";
  return `${EDITOR_ALLOWED_EXTENSIONS.join(", ")}${pathPart}`;
}

export function register(server) {
  // Rollout gate: this tool is not registered at all --
  // not "registered but refuses calls" -- unless the flag is on. A caller
  // (or a prompt-injected instruction) can't discover or invoke a tool
  // that was never added to the server's tool list in the first place.
  if (!EDITOR_AGENT_ENABLED) return;

  server.tool(
    "delegate_editor",
    "TRIGGERS: general-purpose repo edits on a feature branch -- docs, config, backend code, tests, etc. -- that aren't frontend " +
      "HTML/CSS/JSX/Vue (use delegate_designer for those) and aren't a read-only investigation (use delegate_agent for those).\n" +
      "IS: WRITE TOOL, bounded agentic loop (default " + EDITOR_DEFAULT_STEPS + " steps, hard cap " + EDITOR_HARD_MAX_STEPS + ") with three tools of its own (read_file/write_file/validate) -- commits to the repo in the same call. Returns the agent's own final text summary plus which files it wrote, not the generated code inline.\n" +
      "SCOPE, ENFORCED AT THE TOOL LAYER (not just prompt instructions): reads/writes fenced to extensions " + scopeSummary() + ". " +
      "A hard deny list (independent of the allowlist) always blocks .github/workflows/**, connectors/security.js, and GitHub-App auth files, and blocks package.json's scripts/dependencies fields specifically -- regardless of extension. " +
      "This run may touch at most " + EDITOR_MAX_FILES_PER_RUN + " distinct file(s), and write any single file at most " + EDITOR_MAX_WRITES_PER_FILE + " time(s).\n" +
      "PREREQUISITE: `branch` MUST already exist and MUST NOT be the repo's default branch -- checked live against the GitHub API before any tool call, never trusted from the argument alone. No branch yet -> call create_branch first.\n" +
      "CANNOT open, approve, or merge pull requests -- this tool has no create_pull_request/merge_pull_request in its own function set, structurally, not just by convention. A human reviews the branch's diff afterward; this tool only gets it ready for that review.\n" +
      "RESUME: failed/partial run -> response includes resume_run_id -> pass back to continue from the last completed step instead of restarting.\n" +
      "ASYNC -- DO NOT HAMMER THIS: a fresh call may return a run_id immediately while the run keeps stepping server-side in the background. Check status with resume_run_id ONCE, then STOP -- go do other work (other tasks, other tool calls, or just respond to the user) and check back later. A single step realistically takes several seconds to tens of seconds end-to-end (QStash delivery plus the step itself), so calling resume_run_id again right away, or more than once in the same turn, does not finish it any faster -- it only burns a call for the same 'still running' answer. Only a resume_run_id call returns the final result, so you do need to check back eventually -- just not immediately, and not repeatedly.\n" +
      "POLLING vs PUSHING (async/QStash mode only -- see below): to just check progress on a resume_run_id, call with NO max_steps -- this is always read-only, even in the rare case where the background worker chain has stalled (you'll get a stalled status instead of a stored answer, never a silent extra step or write). Only pass max_steps when you actually want to advance the run further right now (raising the ceiling on a resumed run, or nudging a stalled one forward) -- an explicit max_steps is what authorizes real work (including further commits) to happen on that call. This distinction only applies when the async worker is configured; in synchronous mode there is no separate poll state at all -- every resume_run_id call continues the loop immediately regardless of max_steps, same as always.",
    {
      owner:           z.string().optional().describe(`Repository owner. Defaults to "${DEFAULT_OWNER}" if omitted.`),
      repo:            z.string().optional().describe("Repository name. Not needed when resuming (resume_run_id carries it)."),
      branch:          z.string().optional().describe("Branch to work on. Must already exist and MUST NOT be the repo's default branch. Not needed when resuming."),
      task:            z.string().optional().describe("What to change, described with enough detail for the agent to act without asking anything back -- it can't. Ignored when resume_run_id resolves to a live checkpoint (the original task from that run is reused). Optional ONLY when resume_run_id is given and its checkpoint is still live; required otherwise."),
      max_steps:       z.number().optional().describe(`Max agent steps before being forced to answer (default ${EDITOR_DEFAULT_STEPS}, hard cap ${EDITOR_HARD_MAX_STEPS} regardless of this value). LEAVE UNSET on a fresh call -- there's no reliable way to size this upfront from the task description alone, and an undersized guess just causes the run to hit the cap mid-task before it ever gets to write. Only pass an explicit value when RESUMING a run that already came back reporting it hit its step cap (the failed run's own reported step count, plus the transcript, is real evidence for how many more steps are actually needed -- use that, not a fresh guess). On a resumed run this is the new ceiling, not additional steps on top of what's already done.`),
      resume_run_id:   z.string().optional().describe("A runId returned from a previous failed/partial delegate_editor call. If its checkpoint is still live (1 hour TTL), continues that run's conversation instead of starting fresh."),
      show_transcript: z.boolean().optional().describe("Include the full step-by-step tool-call transcript in the response, even on a successful run (default: false). On a failed/partial run the transcript is always shown regardless of this flag."),
      provider: z.enum(["gemini"]).optional()
        .describe(`Which provider runs the editor loop (default and only supported value: "gemini"). RESUME RULE: if resume_run_id resolves to a checkpoint that recorded a provider, that recorded provider is always used and this argument is ignored.`),
    },
    async ({ owner = DEFAULT_OWNER, repo, branch, task, max_steps, resume_run_id, show_transcript = false, provider }) => {
      // Same "task is only genuinely optional when resuming a live
      // checkpoint" reasoning as delegate_designer/delegate_agent's own
      // handlers.
      if (!task && !resume_run_id) {
        return {
          content: [{ type: "text", text: "Missing required argument: task must be provided unless resuming a live checkpoint via resume_run_id." }],
          isError: true,
        };
      }
      if (max_steps !== undefined && (!Number.isInteger(max_steps) || max_steps < 1)) {
        return {
          content: [{ type: "text", text: `Invalid max_steps: ${max_steps}. Must be a positive integer (at least 1); the hard cap is ${EDITOR_HARD_MAX_STEPS} regardless of a larger value.` }],
          isError: true,
        };
      }

      // Distinct from runEditorAgent's own internal `max_steps =
      // EDITOR_DEFAULT_STEPS` default (applied one layer down, not here) --
      // the stale-checkpoint poll branch below needs to tell "caller
      // explicitly wants to push the run forward" apart from "caller is
      // just checking status and max_steps happens to be undefined", so it
      // reads this BEFORE any defaulting happens. Same distinction
      // connectors/delegate/agent/agent_tools.js draws for delegate_agent.
      const maxStepsProvided = max_steps !== undefined;

      // Async delegate_editor: gated behind BOTH the
      // rollout flag and QStash actually being reachable -- if
      // off/unconfigured, every branch below is skipped and this falls
      // straight through to today's fully-synchronous runEditorAgent call
      // further down, unchanged. Mirrors agent_tools.js's asyncEnabled gate.
      const asyncEnabled = EDITOR_AGENT_ASYNC === "qstash" && isEditorQStashConfigured();

      if (asyncEnabled && !resume_run_id) {
        // Fresh async start: seed a checkpoint (zero steps taken, no writes
        // made yet) and hand step 1 onward off to the QStash worker,
        // returning almost immediately instead of blocking on the whole
        // run -- this is the entire point of the async path.
        let runId;
        try {
          runId = await seedEditorRun({ owner, repo, branch, task, max_steps, provider });
          await publishEditorStep({ runId, afterStep: 0 });
        } catch (err) {
          return { content: [{ type: "text", text: `Failed to start async run: ${err?.message ?? String(err)}` }], isError: true };
        }
        return {
          content: [{ type: "text", text:
            `Run started in the background (run_id: ${runId}). It will keep stepping on its own -- call delegate_editor again with resume_run_id: "${runId}" (task not needed) to poll for progress or the final answer.` }],
        };
      }

      if (asyncEnabled && resume_run_id) {
        const checkpoint = await loadCheckpoint(resume_run_id);
        if (checkpoint && checkpoint.status === "failed") {
          // Dead-lettered by editor_worker.js after repeated same-step
          // failures -- a definitive, non-resumable outcome. Return it
          // directly rather than letting runEditorAgent try to resume a
          // run that was deliberately given up on.
          const writtenNote = checkpoint.writtenFiles?.length ? `\n\nFiles written before the failure: ${checkpoint.writtenFiles.join(", ")}` : "";
          return {
            content: [{ type: "text", text: `Run failed permanently (run_id: ${resume_run_id}) after repeated errors on the same step: ${checkpoint.finalAnswer || "(no error detail saved)"}${writtenNote}` }],
            isError: true,
          };
        }
        if (checkpoint && checkpoint.status === "running") {
          const now = Date.now();
          const stepStartedAt = checkpoint.stepStartedAt || 0;
          const lastStepAt = checkpoint.lastStepAt || 0;

          let isFresh;
          let ageMs;

          if (stepStartedAt > 0) {
            const stepStartedAgeMs = now - stepStartedAt;
            if (stepStartedAgeMs > EDITOR_ASYNC_STEP_DEAD_SECONDS * 1000) {
              // Exceeded long ceiling: treat as genuine stall/crash case, NOT as fresh.
              isFresh = false;
              ageMs = stepStartedAgeMs;
            } else {
              // Within long ceiling: in-flight step counts as fresh.
              isFresh = true;
              ageMs = stepStartedAgeMs;
            }
          } else {
            ageMs = now - lastStepAt;
            isFresh = ageMs < EDITOR_ASYNC_POLL_FRESH_SECONDS * 1000;
          }

          const writtenNote = checkpoint.writtenFiles?.length ? `\n\nFiles written so far: ${checkpoint.writtenFiles.join(", ")}` : "";

          if (isFresh) {
            // Fresh lastStepAt or stepStartedAt -- the background worker
            // chain is still actively stepping. Poll-only: report progress
            // WITHOUT taking a step (and therefore without making any
            // further writes) ourselves, so a poll can never race the
            // worker or trigger an unintended commit.
            return {
              content: [{ type: "text", text:
                `Still running (run_id: ${resume_run_id}) -- ${checkpoint.stepsDone} step(s) completed so far. Last activity ${Math.round(ageMs / 1000)}s ago. Call again with the same resume_run_id to keep polling.${writtenNote}` }],
            };
          }
          // Stale activity -- either lastStepAt is old, or stepStartedAt
          // exceeded EDITOR_ASYNC_STEP_DEAD_SECONDS. Same reasoning as
          // agent_tools.js: only fall through to a synchronous resume
          // (which can make further writes) if the caller explicitly
          // passed max_steps -- that's the one signal that distinguishes
          // "just checking progress" from "I want this to make more
          // progress right now". A plain status check on a stalled run
          // must never be able to trigger a commit as a side effect of bad
          // timing.
          if (!maxStepsProvided) {
            return {
              content: [{ type: "text", text:
                `Run appears stalled (run_id: ${resume_run_id}) -- ${checkpoint.stepsDone} step(s) completed, no activity in ${Math.round(ageMs / 1000)}s (the background worker chain may have broken). ` +
                `Call delegate_editor again with resume_run_id: "${resume_run_id}" and an explicit max_steps to resume synchronously from where it left off.${writtenNote}` }],
            };
          }
          // Explicit max_steps given -- caller wants to push forward: fall
          // through to the ordinary synchronous runEditorAgent call below,
          // which resumes the loop IN THIS CALL.
        }
        // checkpoint missing (expired/never existed), or status "done" --
        // fall through to runEditorAgent below, which now short-circuits a
        // "done" checkpoint into a cheap stored-answer read (see
        // editor_delegate.js), and already produces its existing clear
        // error for a missing one.
      }

      let result;
      try {
        result = await runEditorAgent({ owner, repo, branch, task, max_steps, resume_run_id, provider });
      } catch (err) {
        return { content: [{ type: "text", text: `delegate_editor failed: ${err?.message ?? String(err)}` }], isError: true };
      }

      const writtenNote = result.writtenFiles?.length
        ? `\n\nFiles written: ${result.writtenFiles.join(", ")}`
        : "";
      const transcriptBlock = result.transcript?.length && (result.failed || show_transcript)
        ? `\n\n${result.failed ? "Tool calls completed before the failure" : "Tool call transcript"}:\n${result.transcript.join("\n")}`
        : "";

      return {
        content: [{ type: "text", text: `${result.answer}${writtenNote}${transcriptBlock}\n\n(${result.steps} step(s) taken)` }],
        isError: !!result.failed,
      };
    }
  );
}
