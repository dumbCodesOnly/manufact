import { z } from "zod";
import { runInvestigation, seedRun } from "./agent_delegate.js";
import { loadCheckpoint } from "./agent_checkpoint.js";
import { publishAgentStep, isQStashConfigured } from "../qstash_client.js";
import { doCreatePage } from "../../notion/tools.js";
import { GEMINI_NOTION_ROOT_PAGE_ID, DELEGATE_AGENT_ASYNC, AGENT_ASYNC_POLL_FRESH_SECONDS, AGENT_ASYNC_STEP_DEAD_SECONDS } from "../../../config.js";

export function register(server) {

  server.tool(
    "delegate_agent",
    "DOES: Open-ended, multi-step READ-ONLY investigation across GitHub/Notion/Cloudflare -- Gemini runs its own server-side loop (bounded by max_steps) reading files/trees/commits/logs/pages across as many turns as needed, cross-checks claims BETWEEN sources, flags discrepancies, returns one synthesized answer.\n" +
    "RULE: default choice for multi-file or open-ended investigation -- prefer over manual read_file/get_file_tree/list_directory loops UNLESS you need exactly one named file.\n" +
    "NOT: web access -> use delegate_research (task param, wide mode) instead. NOT: any write -> read-only by design.\n" +
    "USE FOR: e.g. 'why is CI failing on PR #42', 'summarize what changed in this repo over the last week' -- cases needing 5-10+ manual cross-system calls otherwise.\n" +
    "RESUME: failed/partial run -> response includes resume_run_id -> pass back to continue from last completed step instead of restarting.\n" +
    "ASYNC, DO NOT POLL REPEATEDLY: a fresh call may return a run_id immediately while work continues server-side. Check status with resume_run_id ONCE, then stop -- space out polls instead of hammering it; polling faster does not finish it faster. If still running, end your turn (do other work, or wait for the user's next message) and resume later. Only a resume_run_id call returns the final answer, so you must check back eventually -- just not repeatedly in the same turn.\n" +
    "POLLING vs PUSHING (async/QStash mode only -- see below): to just check progress on a resume_run_id, call with NO max_steps -- this is always read-only, even in the rare case where the background worker chain has stalled (you'll get a 'stalled' status instead of a stored answer, never a silent extra step). Only pass max_steps when you actually want to advance the investigation further right now (raising the ceiling on a resumed run, or nudging a stalled one forward) -- an explicit max_steps is what authorizes real work to happen on that call. This distinction only applies when the async worker is configured; in synchronous mode there is no separate poll state at all -- every resume_run_id call continues the loop immediately regardless of max_steps, same as always.",
    {
      task:          z.string().optional().describe("The investigation task/question, described with enough context (repo names, time ranges, etc.) for Gemini to act without needing to ask you anything back -- it can't. Ignored when resume_run_id resolves to a live checkpoint (the original task from that run is reused). Optional ONLY when resume_run_id is given and its checkpoint is still live; required otherwise -- omitting it on a fresh run (no resume_run_id, or an expired one) returns an error rather than silently proceeding with no task."),
      max_steps:     z.number().optional().describe("Max tool-use turns Gemini gets before being forced to answer (default 20, hard cap 30 regardless of this value). LEAVE UNSET on a fresh call -- there's no reliable way to size this upfront from the task description alone, and an undersized guess just causes the investigation to hit the cap before it's actually done. Only pass an explicit value when RESUMING a run that already came back reporting it hit its step cap (the failed run's own reported step count, plus its transcript, is real evidence for how many more steps are needed -- use that, not a fresh guess). On a resumed run this is the new ceiling, not additional steps on top of what's already done. IN ASYNC/QSTASH MODE: leave unset to just poll a resume_run_id for status -- omitting max_steps guarantees the call is read-only (never drives a step), even if the background worker chain has stalled; pass it explicitly only when you want this call to actually push the investigation forward. In synchronous mode this distinction doesn't apply -- every resume_run_id call runs synchronously regardless."),
      log_to_notion: z.boolean().optional().describe("Whether to log the task, step-by-step tool calls, and final answer as a page under the Gemini section of Notion (default: false). Write always targets the fixed Gemini root page. ASYNC CAVEAT: not persisted across calls -- the initial fire-and-forget start call ignores this and returns before logging ever runs, so it must be passed again as true on the resume_run_id call(s) that actually retrieve the final answer, or nothing gets logged."),
      resume_run_id: z.string().optional().describe("A runId returned from a previous failed/partial delegate_agent call. If its checkpoint is still live (1 hour TTL), continues that run's conversation instead of starting fresh."),
      show_transcript: z.boolean().optional().describe("Include the full step-by-step tool-call transcript in the response, even on a successful run (default: false). Useful for debugging what Gemini actually called and in what order/grouping -- e.g. checking whether independent calls were batched into the same step. On a failed/partial run the transcript is only included if this flag is explicitly true."),
      provider: z.enum(["gemini"]).optional()
        .describe(`Which provider runs the investigation loop (default and only supported value: "gemini").`),
      model: z.string().optional()
        .describe(`Override the specific Gemini model to use (default: GEMINI_MODEL from config). ` +
          `WARNING -- CASCADE DISABLED: passing a model different from the default skips GEMINI_FALLBACK_MODELS entirely -- only the requested model is tried, so a 429/503 on it fails the call instead of cascading to another model. ` +
          `RESUME RULE: if resume_run_id resolves to a checkpoint that recorded a model, that recorded model is always used and this argument is ignored. If the checkpoint has no recorded model, this argument is used as a fallback instead of erroring.`),
      maxOutputTokens: z.number().optional()
        .describe(`Caps the per-turn (not whole-conversation) output token budget for each Gemini call in the investigation loop. Default: none set (Gemini's own API default applies, no cap sent). Raise this if answers are getting cut off mid-response. ` +
          `RESUME RULE: same as model -- if resume_run_id resolves to a checkpoint that recorded a value, that recorded value is always used and this argument is ignored. If the checkpoint has no recorded value, this argument is used as a fallback instead of erroring.`),
      // preambleVariant intentionally removed from the exposed schema (A/B
      // test concluded, see Notion writeup) -- the calling model can no
      // longer pick a variant. The internal plumbing that threads
      // preambleVariant through seedRun/runInvestigation/checkpoints below
      // is left intact deliberately (per handoff notes) in case we want to
      // re-test later; it just always receives `undefined` from this tool
      // now, so the shared default in preamble.js / agent_delegate.js
      // ("trimmed") applies.
    },
    async ({ task, max_steps: rawMaxSteps, log_to_notion = false, resume_run_id, show_transcript = false, provider, model, maxOutputTokens, preambleVariant }) => {
      // Distinct from the old `max_steps = 20` default-via-destructuring:
      // maxStepsProvided records whether the CALLER actually passed a value,
      // separately from the effective number used once defaulted -- the
      // stale-checkpoint poll branch below needs to tell "caller explicitly
      // wants to push the investigation forward" apart from "caller is just
      // checking status and happened to get 20 by default". See that
      // branch's comment for why this distinction matters.
      const maxStepsProvided = rawMaxSteps !== undefined;
      const max_steps = maxStepsProvided ? rawMaxSteps : 20;

      // task is only genuinely optional when resuming a live checkpoint --
      // runInvestigation ignores task entirely in that branch (it rebuilds
      // `contents` straight from the saved checkpoint). On a fresh run (no
      // resume_run_id, or one whose checkpoint already expired), there is no
      // saved task to fall back on, so fail loudly here rather than letting
      // runInvestigation start a conversation with an undefined task.
      if (!task && !resume_run_id) {
        return {
          content: [{ type: "text", text: "Missing required argument: task must be provided unless resuming a live checkpoint via resume_run_id." }],
          isError: true,
        };
      }

      // max_steps has no floor in its Zod type (z.number().optional() accepts
      // 0, negatives, and non-integers), but runInvestigation's loop is a
      // `for (step = startStep; step <= cappedSteps; ...)` that simply never
      // executes when cappedSteps < startStep -- silently "succeeding" with
      // zero Gemini calls made and a confusing "reached the step cap of 0"
      // answer instead of surfacing that the input itself was invalid.
      // Checked against maxStepsProvided (the caller's raw input), not just
      // "defined", since max_steps is always defined now post-default.
      if (maxStepsProvided && (!Number.isInteger(max_steps) || max_steps < 1)) {
        return {
          content: [{ type: "text", text: `Invalid max_steps: ${max_steps}. Must be a positive integer (at least 1); the hard cap is 30 regardless of a larger value.` }],
          isError: true,
        };
      }

      // Same reasoning as max_steps's guard above: z.number().optional() has
      // no floor of its own, and a non-positive value here would produce a
      // confusing provider-level error (or, worse, a silently truncated-to-
      // nothing response) instead of a clear rejection at the boundary.
      if (maxOutputTokens !== undefined && (!Number.isInteger(maxOutputTokens) || maxOutputTokens < 1)) {
        return {
          content: [{ type: "text", text: `Invalid maxOutputTokens: ${maxOutputTokens}. Must be a positive integer (at least 1).` }],
          isError: true,
        };
      }

      // Async delegate_agent (Scenario B groundwork):
      // gated behind BOTH the rollout flag and QStash actually being
      // reachable -- if either is off/unconfigured, every branch below is
      // skipped and this falls straight through to today's fully-
      // synchronous runInvestigation call further down, unchanged.
      const asyncEnabled = DELEGATE_AGENT_ASYNC === "qstash" && isQStashConfigured();

      if (asyncEnabled && !resume_run_id) {
        // Fresh async start: seed a checkpoint (zero steps taken) and hand
        // step 1 onward off to the QStash worker, returning almost
        // immediately instead of blocking on the whole investigation --
        // this is the entire point of Scenario B.
        let runId;
        try {
          runId = await seedRun({ task, provider, model, maxOutputTokens, max_steps, preambleVariant });
          await publishAgentStep({ runId, afterStep: 0 });
        } catch (err) {
          return { content: [{ type: "text", text: `Failed to start async investigation: ${err?.message ?? String(err)}` }], isError: true };
        }
        return {
          content: [{ type: "text", text:
            `Investigation started in the background (run_id: ${runId}). It will keep stepping on its own -- call delegate_agent again with resume_run_id: "${runId}" (task not needed) to poll for progress or the final answer.` }],
        };
      }

      if (asyncEnabled && resume_run_id) {
        const checkpoint = await loadCheckpoint(resume_run_id);
        if (checkpoint && checkpoint.status === "failed") {
          // Dead-lettered by agent_worker.js after repeated
          // same-step failures -- a definitive, non-resumable outcome.
          // Return it directly rather than letting runInvestigation try to
          // resume a run that was deliberately given up on.
          return {
            content: [{ type: "text", text: `Investigation failed permanently (run_id: ${resume_run_id}) after repeated errors on the same step: ${checkpoint.finalAnswer || "(no error detail saved)"}` }],
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
            if (stepStartedAgeMs > AGENT_ASYNC_STEP_DEAD_SECONDS * 1000) {
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
            isFresh = ageMs < AGENT_ASYNC_POLL_FRESH_SECONDS * 1000;
          }

          if (isFresh) {
            // Fresh lastStepAt or stepStartedAt -- the background worker chain is still
            // actively stepping. Poll-only: report progress WITHOUT taking
            // a step ourselves, so a poll can never race the worker.
            return {
              content: [{ type: "text", text:
                `Still running (run_id: ${resume_run_id}) -- ${checkpoint.stepsDone} step(s) completed so far. Last activity ${Math.round(ageMs / 1000)}s ago. Call again with the same resume_run_id to keep polling.` +
                (checkpoint.transcript?.length ? `\n\nTool calls so far:\n${checkpoint.transcript.join("\n")}` : "") }],
            };
          }
          // Stale activity -- either lastStepAt is old, or stepStartedAt exceeded AGENT_ASYNC_STEP_DEAD_SECONDS.
          //
          // Fix (2026-08-31): this used to fall through to a synchronous
          // runInvestigation call unconditionally, regardless of whether the
          // caller actually wanted to push the run forward. That meant a
          // caller doing nothing but a routine status check -- resume_run_id
          // with no max_steps, exactly what "just polling" looks like -- could
          // silently trigger real additional steps (up to the default max_steps
          // of 20) the moment the background worker chain happened to be
          // stale, with no visible difference in the request from an
          // intentional "push this forward" call. A poll should never be able
          // to drive real work as a side effect of bad luck in timing.
          //
          // Now: only fall through to a synchronous resume if the caller
          // explicitly passed max_steps -- that's the one signal that
          // distinguishes "just checking progress" from "I want this to make
          // more progress right now". A plain status check on a stalled run
          // instead reports the stall and tells the caller how to push it
          // forward, rather than doing so unasked. This does NOT reintroduce
          // the "run can be stranded" risk the original fallback existed to
          // prevent -- the checkpoint is untouched either way, and the very
          // next call with an explicit max_steps resumes it synchronously
          // exactly as before.
          if (!maxStepsProvided) {
            return {
              content: [{ type: "text", text:
                `Investigation appears stalled (run_id: ${resume_run_id}) -- ${checkpoint.stepsDone} step(s) completed, no activity in ${Math.round(ageMs / 1000)}s (the background worker chain may have broken). ` +
                `Call delegate_agent again with resume_run_id: "${resume_run_id}" and an explicit max_steps to resume the investigation synchronously from where it left off.` +
                (checkpoint.transcript?.length ? `\n\nTool calls so far:\n${checkpoint.transcript.join("\n")}` : "") }],
            };
          }
          // Explicit max_steps given -- caller wants to push forward: fall
          // through to the ordinary synchronous runInvestigation call below,
          // which resumes the loop IN THIS CALL.
        }
        // checkpoint missing (expired/never existed), or status "done" --
        // fall through to runInvestigation below, which already handles
        // both correctly: "done" is a cheap stored-answer read (no re-
        // execution), and "missing" produces its existing clear error.
      }

      let result;
      try {
        result = await runInvestigation({ task, max_steps, resume_run_id, provider, model, maxOutputTokens, preambleVariant });
      } catch (err) {
        return { content: [{ type: "text", text: `Investigation failed: ${err?.message ?? String(err)}` }], isError: true };
      }

      // On a resumed run, `task` may be undefined here (a fresh run always has
      // it, per the guard above) -- runInvestigation returns the effective
      // task text it actually used (the caller-supplied one, or the one
      // restored from the checkpoint) so logging/titling never has to guess.
      const effectiveTask = task || result.task || "(resumed run)";

      let notionNote = "";
      if (log_to_notion) {
        try {
          const logged = await doCreatePage({
            parent_id:   GEMINI_NOTION_ROOT_PAGE_ID,
            parent_type: "page",
            title:       `${result.failed ? "investigate (partial): " : "investigate: "}${effectiveTask.slice(0, 80)}`,
            content:     `Task: ${effectiveTask}\n\nrunId: ${result.runId}${result.failed ? " (resumable)" : ""}\n\nSteps taken: ${result.steps}\n\nTool calls:\n${result.transcript.join("\n") || "(none)"}\n\nAnswer:\n${result.answer}`,
            one_off:     true,
          });
          notionNote = `\n\n(Logged to Notion: ${logged.url})`;
        } catch (err) {
          notionNote = `\n\n(\u26a0\ufe0f Notion logging failed: ${err.message})`;
        }
      }

      // Fix: when a run fails/is partial, if show_transcript is not explicitly true,
      // return a COMPACT structured summary instead of the full transcript:
      // - step count reached
      // - short description of what failed (error message / reason)
      // - resume_run_id if resumable
      // - omit full transcript unless show_transcript is explicitly true.
      if (result.failed) {
        const resumeLine = result.runId ? `resume_run_id: "${result.runId}"` : "not resumable";
        const transcriptBlock = show_transcript && result.transcript?.length
          ? `\n\nTool calls completed before failure:\n${result.transcript.join("\n")}`
          : "";
        const compactText =
          `Investigation failed or partial after ${result.steps} step(s).\n` +
          `Reason/Error: ${result.answer}\n` +
          `Resumable: ${resumeLine}${transcriptBlock}${notionNote}`;
        return { content: [{ type: "text", text: compactText }], isError: true };
      }

      const transcriptBlock = result.transcript?.length && show_transcript
        ? `\n\nTool call transcript:\n${result.transcript.join("\n")}`
        : "";

      return { content: [{ type: "text", text: `${result.answer}${transcriptBlock}\n\n(${result.steps} step(s) taken)${notionNote}` }], isError: false };
    }
  );
}
