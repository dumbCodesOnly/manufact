import { describe, it, expect } from "vitest";
import { analyzeRun } from "../scripts/preamble-ab-metrics.js";

// TEMPORARY -- delete alongside scripts/preamble-ab-metrics.js once the
// verbose/trimmed A/B test is done. Exercises analyzeRun's parsing logic
// against synthetic checkpoint shapes (no live Redis needed) so the
// throwaway metrics script can be trusted before pointing it at real runs.

function modelTurn(text) {
  return { role: "model", parts: [{ text }] };
}
function userTurn(text) {
  return { role: "user", parts: [{ text }] };
}

describe("preamble-ab-metrics.js: analyzeRun", () => {
  it("reports no verification/recheck/stuck-loop when a run has none of it", () => {
    const checkpoint = {
      contents: [
        userTurn("You are a read-only investigation agent...\n\nTask: do the thing"),
        modelTurn("The final answer is X."),
      ],
      status: "done",
      stepsDone: 1,
      overallMaxSteps: 20,
      preambleVariant: "trimmed",
      provider: "gemini",
      finalAnswer: "The final answer is X.",
    };
    const r = analyzeRun("run-1", checkpoint);
    expect(r.preambleVariant).toBe("trimmed");
    expect(r.verificationFired).toBe(false);
    expect(r.structuralRecheckFired).toBe(false);
    expect(r.stuckLoopForced).toBe(false);
    expect(r.leakageEvidence).toBe(false);
  });

  it("detects a verification pass that changed the answer", () => {
    const checkpoint = {
      contents: [
        userTurn("Task: do the thing"),
        modelTurn("Draft answer: HARD_MAX_STEPS gates the condition."),
        userTurn("[SYSTEM NOTE -- verification pass] Before your answer above is treated as final..."),
        modelTurn("Corrected answer: cappedSteps gates the condition, not HARD_MAX_STEPS."),
      ],
      status: "done",
      stepsDone: 2,
      overallMaxSteps: 20,
      preambleVariant: "verbose",
      finalAnswer: "Corrected answer: cappedSteps gates the condition, not HARD_MAX_STEPS.",
    };
    const r = analyzeRun("run-2", checkpoint);
    expect(r.verificationFired).toBe(true);
    expect(r.verificationChangedAnswer).toBe(true);
    expect(r.draftAnswer).toContain("HARD_MAX_STEPS gates");
    expect(r.correctedAnswer).toContain("cappedSteps gates");
  });

  it("detects a verification pass that did NOT change the answer", () => {
    const checkpoint = {
      contents: [
        userTurn("Task: do the thing"),
        modelTurn("The answer is Y."),
        userTurn("[SYSTEM NOTE -- verification pass] Before your answer above is treated as final..."),
        modelTurn("The answer is Y."),
      ],
      status: "done",
      stepsDone: 2,
      overallMaxSteps: 20,
      preambleVariant: "verbose",
      finalAnswer: "The answer is Y.",
    };
    const r = analyzeRun("run-3", checkpoint);
    expect(r.verificationFired).toBe(true);
    expect(r.verificationChangedAnswer).toBe(false);
  });

  it("detects a hard stuck-loop force and a structural recheck", () => {
    const checkpoint = {
      contents: [
        userTurn("Task: do the thing"),
        modelTurn("some draft"),
        userTurn("[SYSTEM NOTE: 3 consecutive steps have consisted entirely of repeat calls. The next turn will NOT include any tools...]"),
        modelTurn("LINE_QUOTE: const x = 1;\nFinal answer text."),
        userTurn("[STRUCTURAL LINE-QUOTE CHECK FAILED] The following line(s)..."),
        modelTurn("Corrected final answer text."),
      ],
      status: "done",
      stepsDone: 5,
      overallMaxSteps: 6,
      preambleVariant: "trimmed",
      structuralRecheckUsed: true,
      finalAnswer: "Corrected final answer text.",
    };
    const r = analyzeRun("run-4", checkpoint);
    expect(r.stuckLoopForced).toBe(true);
    expect(r.structuralRecheckFired).toBe(true);
    expect(r.structuralRecheckUsedFlag).toBe(true);
  });

  it("detects tool-call-leakage evidence from the finalAnswer text", () => {
    const checkpoint = {
      contents: [userTurn("Task: x"), modelTurn("garbled output")],
      status: "failed",
      stepsDone: 3,
      overallMaxSteps: 6,
      preambleVariant: "verbose",
      finalAnswer: `(The model attempted to invoke the "github_read_file" tool as text on this turn...)`,
    };
    const r = analyzeRun("run-5", checkpoint);
    expect(r.leakageEvidence).toBe(true);
  });

  it("defaults preambleVariant label for legacy checkpoints with no recorded variant", () => {
    const checkpoint = {
      contents: [userTurn("Task: x"), modelTurn("answer")],
      status: "done",
      stepsDone: 1,
      overallMaxSteps: 20,
      finalAnswer: "answer",
    };
    const r = analyzeRun("run-6", checkpoint);
    expect(r.preambleVariant).toBe("verbose (unset/legacy)");
  });
});
