import { describe, it, expect, vi, beforeEach } from "vitest";

// Covers async delegate_editor groundwork: singleStep
// correctness, and specifically the singleStep vs. max_steps: stepsDone + 1
// distinction -- a single-hop test wouldn't catch a
// bug where overallMaxSteps silently falls out of the checkpoint the
// moment a second worker invocation reloads it, so this exercises TWO
// chained singleStep resumes (simulating two worker hops) and confirms
// isFinalStep/tool-withholding is still correct on the second hop.
//
// Unlike test/editor-delegate.test.js (which mocks editor_checkpoint.js
// with a plain in-memory Map), this wires up a real fake Redis (same
// whole-blob shape as editor_checkpoint.js itself -- get/set/del only, no
// list+meta split) so it exercises genuine save/resume/done round trips
// end to end, matching what the editor worker actually depends on.

function makeFakeRedis() {
  const strings = new Map();
  return {
    async set(key, val) { strings.set(key, val); return "OK"; },
    async get(key) { return strings.has(key) ? strings.get(key) : null; },
    async del(key) { strings.delete(key); return 1; },
  };
}

const fakeRedis = makeFakeRedis();
vi.mock("../connectors/shared/cooldown.js", () => ({
  getRedis: () => fakeRedis,
  isRedisConfigured: () => true,
}));

const mockProviderChat = vi.fn();
vi.mock("../connectors/llm/router.js", () => ({
  providerChat: mockProviderChat,
}));

const mockReadFile = vi.fn();
const mockWriteFile = vi.fn();
const mockAssertNotDefaultBranch = vi.fn(async () => ({ default_branch: "main" }));
vi.mock("../connectors/github/editor_tool_functions.js", () => ({
  readFile: (...args) => mockReadFile(...args),
  writeFile: (...args) => mockWriteFile(...args),
  assertNotDefaultBranch: (...args) => mockAssertNotDefaultBranch(...args),
}));

vi.mock("../connectors/github/editor_validate.js", () => ({
  validateByExtension: vi.fn(async () => ({ valid: true })),
}));

const OWNER = "allocsys";
const REPO = "madmcp";
const BRANCH = "feature-branch";

function functionCallCandidate(name, args, id = "call_1") {
  return {
    content: { role: "model", parts: [{ functionCall: { name, args, id } }] },
    finishReason: "STOP",
  };
}

function textCandidate(text) {
  return { content: { role: "model", parts: [{ text }] }, finishReason: "STOP" };
}

describe("editor_delegate.js — single-step resume chaining (async delegate_editor groundwork)", () => {
  let runEditorAgent, seedEditorRun, loadCheckpoint;

  beforeEach(async () => {
    vi.resetModules();
    vi.clearAllMocks();
    mockAssertNotDefaultBranch.mockResolvedValue({ default_branch: "main" });
    ({ runEditorAgent, seedEditorRun } = await import("../connectors/delegate/editor/editor_delegate.js"));
    ({ loadCheckpoint } = await import("../connectors/delegate/editor/editor_checkpoint.js"));
  });

  it("seedEditorRun makes no provider calls and produces a loadable checkpoint with stepsDone: 0 and the run's real overallMaxSteps", async () => {
    const runId = await seedEditorRun({ owner: OWNER, repo: REPO, branch: BRANCH, task: "seed only", max_steps: 8 });
    expect(mockProviderChat).not.toHaveBeenCalled();
    const cp = await loadCheckpoint(runId);
    expect(cp.status).toBe("running");
    expect(cp.stepsDone).toBe(0);
    expect(cp.overallMaxSteps).toBe(8);
  });

  it("seedEditorRun runs assertNotDefaultBranch before ever writing a checkpoint", async () => {
    mockAssertNotDefaultBranch.mockRejectedValueOnce(new Error('refusing to write to allocsys/madmcp\'s default branch ("main")'));
    await expect(
      seedEditorRun({ owner: OWNER, repo: REPO, branch: "main", task: "bad branch" })
    ).rejects.toThrow(/refusing to write/i);
  });

  it("a resume with singleStep: true advances the checkpoint by exactly one step", async () => {
    const runId = await seedEditorRun({ owner: OWNER, repo: REPO, branch: BRANCH, task: "multi-step task", max_steps: 5 });
    mockProviderChat.mockResolvedValueOnce(functionCallCandidate("write_file", { path: "a.md", content: "x" }));
    mockWriteFile.mockResolvedValue({ path: "a.md", sha: "s", commitSha: "c1234567", noop: false });

    const result = await runEditorAgent({ resume_run_id: runId, singleStep: true });

    expect(result.steps).toBe(1);
    expect(mockProviderChat).toHaveBeenCalledTimes(1);
    const cp = await loadCheckpoint(runId);
    expect(cp.stepsDone).toBe(1);
    expect(cp.status).toBe("running");
  });

  it("THE KEY REGRESSION CHECK: two chained singleStep resumes preserve overallMaxSteps (and therefore correct isFinalStep/tool-withholding) on the second hop", async () => {
    // Seed a run whose real ceiling is 2 steps -- step 2 is the run's
    // TRUE final step, where tools must be withheld. A bug that lets
    // overallMaxSteps fall out of the checkpoint after the first
    // singleStep save (Step 2's exact bug) would instead derive
    // isFinalStep from this call's own artificially-shrunk startStep,
    // and NOT withhold tools on hop 2 -- letting a function call slip
    // through where the plan says it must be forced into a plain-text
    // answer instead.
    const runId = await seedEditorRun({ owner: OWNER, repo: REPO, branch: BRANCH, task: "two-step task", max_steps: 2 });

    // Hop 1: a function call -- tools are NOT withheld on step 1 (not the
    // final step).
    mockProviderChat.mockResolvedValueOnce(functionCallCandidate("write_file", { path: "a.md", content: "x" }));
    mockWriteFile.mockResolvedValue({ path: "a.md", sha: "s", commitSha: "c1234567", noop: false });

    const hop1 = await runEditorAgent({ resume_run_id: runId, singleStep: true });
    expect(hop1.steps).toBe(1);

    const cpAfterHop1 = await loadCheckpoint(runId);
    expect(cpAfterHop1.overallMaxSteps).toBe(2); // NOT dropped/overwritten after the first save

    // Hop 2 (the run's real final step): confirm tools WERE withheld by
    // asserting providerChat was called with tools: undefined.
    mockProviderChat.mockResolvedValueOnce(textCandidate("done"));

    const hop2 = await runEditorAgent({ resume_run_id: runId, singleStep: true });
    expect(hop2.steps).toBe(2);
    expect(hop2.failed).toBeFalsy();

    const [, secondCallOptions] = mockProviderChat.mock.calls[1];
    expect(secondCallOptions.tools).toBeUndefined();

    const cpAfterHop2 = await loadCheckpoint(runId);
    expect(cpAfterHop2.status).toBe("done");
    expect(cpAfterHop2.finalAnswer).toBe("done");
  });

  it("if a singleStep resume's own call had been mis-derived from startStep (the bug this guards against), tools would wrongly remain available on the true final step -- explicitly confirm the withheld-tools branch fires by having the model attempt a function call anyway on hop 2, which must be discarded, not executed", async () => {
    const runId = await seedEditorRun({ owner: OWNER, repo: REPO, branch: BRANCH, task: "two-step task, model misbehaves", max_steps: 2 });

    mockProviderChat.mockResolvedValueOnce(functionCallCandidate("write_file", { path: "a.md", content: "x" }));
    mockWriteFile.mockResolvedValue({ path: "a.md", sha: "s", commitSha: "c1234567", noop: false });
    await runEditorAgent({ resume_run_id: runId, singleStep: true });

    // Hop 2: model attempts a function call even though tools were
    // withheld -- must be discarded (result.failed) rather than executed,
    // proving withholdTools really took effect on the real final step.
    mockProviderChat.mockResolvedValueOnce(functionCallCandidate("write_file", { path: "b.md", content: "y" }));

    const hop2 = await runEditorAgent({ resume_run_id: runId, singleStep: true });
    expect(hop2.failed).toBe(true);
    expect(hop2.answer).toMatch(/final step/i);
    // write_file must NOT have been called a second time for b.md -- the
    // attempted call was discarded, not executed.
    expect(mockWriteFile).toHaveBeenCalledTimes(1);
  });

  it("polling a done checkpoint via runEditorAgent short-circuits into a cheap stored-answer read without re-invoking the model", async () => {
    const runId = await seedEditorRun({ owner: OWNER, repo: REPO, branch: BRANCH, task: "one-step task", max_steps: 5 });
    mockProviderChat.mockResolvedValueOnce(textCandidate("the final answer"));

    const first = await runEditorAgent({ resume_run_id: runId, singleStep: true });
    expect(first.answer).toBe("the final answer");
    expect(mockProviderChat).toHaveBeenCalledTimes(1);

    const polled = await runEditorAgent({ resume_run_id: runId });
    expect(polled.answer).toBe("the final answer");
    expect(mockProviderChat).toHaveBeenCalledTimes(1); // not called again
  });

  it("a completed run's checkpoint is still loadable (status: done) immediately after runEditorAgent returns -- no test relies on the old deleteCheckpoint behavior", async () => {
    mockProviderChat.mockResolvedValueOnce(textCandidate("finished"));
    // Writes-vs-claim verification pass: a fresh synchronous run (not
    // singleStep) has tool access and step budget left on its first draft
    // answer, so it gets one corrective round before being trusted -- see
    // editor_delegate.js's buildEditorVerificationPrompt/pendingVerification.
    mockProviderChat.mockResolvedValueOnce(textCandidate("finished"));
    const result = await runEditorAgent({ owner: OWNER, repo: REPO, branch: BRANCH, task: "fresh synchronous run" });
    expect(result.answer).toBe("finished");

    const cp = await loadCheckpoint(result.runId);
    expect(cp).not.toBeNull();
    expect(cp.status).toBe("done");
    expect(cp.finalAnswer).toBe("finished");
  });

  it("preserves per-run/per-file write caps (writtenFiles/writesPerFile) correctly across a singleStep resume, not just stepsDone", async () => {
    const runId = await seedEditorRun({ owner: OWNER, repo: REPO, branch: BRANCH, task: "write cap carryover", max_steps: 5 });
    mockProviderChat.mockResolvedValueOnce(functionCallCandidate("write_file", { path: "a.md", content: "x" }));
    mockWriteFile.mockResolvedValue({ path: "a.md", sha: "s", commitSha: "c1234567", noop: false });
    await runEditorAgent({ resume_run_id: runId, singleStep: true });

    const cp = await loadCheckpoint(runId);
    expect(cp.writtenFiles).toEqual(["a.md"]);
    expect(cp.writesPerFile).toEqual({ "a.md": 1 });

    // Second hop: write the SAME file again -- confirms writesPerFile
    // carried over (count becomes 2), not reset to a fresh Map.
    mockProviderChat.mockResolvedValueOnce(functionCallCandidate("write_file", { path: "a.md", content: "y" }));
    await runEditorAgent({ resume_run_id: runId, singleStep: true });

    const cp2 = await loadCheckpoint(runId);
    expect(cp2.writtenFiles).toEqual(["a.md"]);
    expect(cp2.writesPerFile).toEqual({ "a.md": 2 });
  });
});
