// ---------------------------------------------------------------------------
// test/editor-delegate.test.js
//
// Coverage for connectors/delegate/editor/editor_delegate.js's runEditorAgent loop,
// focused on the guardrails that are specific to THIS loop rather than
// already covered by test/editor-tool-functions.test.js (guardrails
// #2/#3/#4, enforced at the tool layer) or test/editor-policy.test.js
// (guardrails #3/#4's allow/deny logic in isolation):
//
//   - Guardrail #6 (per-run / per-file write caps), enforced inside
//     write_file's execute() closure BEFORE writeFile() is ever called --
//     this is the one guardrail that lives in editor_delegate.js itself,
//     not editor_tool_functions.js, so it needs its own test here.
//   - Guardrail #8 (no create_pull_request/merge_pull_request in this
//     loop's own function set) -- this must be a TESTED structural fact,
//     not an assumed one.
//   - Guardrail #2 (default-branch refusal), checked once up front before
//     the loop starts -- editor_tool_functions.test.js already covers
//     assertNotDefaultBranch in isolation; this confirms runEditorAgent
//     actually calls it before doing anything else on a fresh run.
//
// providerChat and the tool-functions module are mocked -- same style/
// boundary as test/frontend-agent-loop.test.js for the sibling
// delegate_designer loop.
// ---------------------------------------------------------------------------

import { describe, it, expect, vi, beforeEach } from "vitest";

vi.mock("../connectors/llm/router.js", () => ({
  providerChat: vi.fn(),
}));

vi.mock("../connectors/github/editor_tool_functions.js", () => ({
  readFile: vi.fn(),
  writeFile: vi.fn(),
  assertNotDefaultBranch: vi.fn(async () => ({ default_branch: "main" })),
}));

vi.mock("../connectors/github/editor_validate.js", () => ({
  validateByExtension: vi.fn(async () => ({ valid: true })),
}));

vi.mock("../connectors/shared/cooldown.js", () => ({
  isRedisConfigured: vi.fn(() => true),
}));

const fakeCheckpoints = vi.hoisted(() => new Map());
vi.mock("../connectors/delegate/editor/editor_checkpoint.js", () => ({
  saveCheckpoint: vi.fn(async (runId, state) => { fakeCheckpoints.set(runId, state); }),
  loadCheckpoint: vi.fn(async (runId) => fakeCheckpoints.get(runId) ?? null),
  deleteCheckpoint: vi.fn(async (runId) => { fakeCheckpoints.delete(runId); }),
}));

import { providerChat } from "../connectors/llm/router.js";
import { writeFile, assertNotDefaultBranch } from "../connectors/github/editor_tool_functions.js";
import { runEditorAgent } from "../connectors/delegate/editor/editor_delegate.js";

const OWNER = "allocsys";
const REPO = "madmcp";
const BRANCH = "feature-branch";

function functionCallCandidate(calls) {
  return { content: { role: "model", parts: calls.map(({ name, args }, i) => ({ functionCall: { name, args, id: `${name}-${i}` } })) } };
}

function textCandidate(text) {
  return { content: { role: "model", parts: [{ text }] } };
}

beforeEach(() => {
  vi.clearAllMocks();
  fakeCheckpoints.clear();
  assertNotDefaultBranch.mockResolvedValue({ default_branch: "main" });
});

describe("guardrail #2 -- default-branch refusal checked before the loop starts", () => {
  it("calls assertNotDefaultBranch before making any providerChat call on a fresh run", async () => {
    assertNotDefaultBranch.mockRejectedValueOnce(new Error('refusing to write to allocsys/madmcp\'s default branch ("main")'));

    await expect(
      runEditorAgent({ owner: OWNER, repo: REPO, branch: "main", task: "do something" })
    ).rejects.toThrow(/refusing to write/i);

    expect(assertNotDefaultBranch).toHaveBeenCalledWith(OWNER, REPO, "main");
    expect(providerChat).not.toHaveBeenCalled();
  });
});

describe("guardrail #8 -- no PR-opening/merging capability in this loop's own function set", () => {
  it("never exposes create_pull_request or merge_pull_request as a callable function, even if the model tries to call one", async () => {
    // The model attempts to call a tool this loop was never given -- the
    // loop's own FUNCTIONS.find(...) lookup returns undefined for it, same
    // as any other unknown function name, and the run is not derailed.
    providerChat.mockResolvedValueOnce(functionCallCandidate([{ name: "merge_pull_request", args: { pull_number: 1 } }]));
    providerChat.mockResolvedValueOnce(textCandidate("Could not merge -- that tool isn't available to me."));
    // Writes-vs-claim verification pass: the first draft plain-text answer
    // (above) arrives with tool access still available and budget left, so
    // it is sent back for exactly one corrective round before being
    // trusted (see editor_delegate.js's buildEditorVerificationPrompt/
    // pendingVerification) -- this third mocked response is that round's
    // reply, re-affirming the same answer as final.
    providerChat.mockResolvedValueOnce(textCandidate("Confirmed -- could not merge, that tool isn't available to me."));

    const result = await runEditorAgent({ owner: OWNER, repo: REPO, branch: BRANCH, task: "merge my PR" });

    expect(result.failed).toBeFalsy();
    expect(result.transcript.join("\n")).toMatch(/unknown function "merge_pull_request"/i);
    expect(providerChat).toHaveBeenCalledTimes(3);
  });

  it("only declares read_file, write_file, and validate to the model -- structurally, not just by not calling the others", async () => {
    providerChat.mockResolvedValueOnce(textCandidate("done"));
    await runEditorAgent({ owner: OWNER, repo: REPO, branch: BRANCH, task: "no-op task" });

    const [, options] = providerChat.mock.calls[0];
    const declaredNames = options.tools[0].functionDeclarations.map((d) => d.name).sort();
    expect(declaredNames).toEqual(["read_file", "validate", "write_file"]);
    expect(declaredNames).not.toContain("create_pull_request");
    expect(declaredNames).not.toContain("merge_pull_request");
  });
});

describe("guardrail #6 -- per-run and per-file write caps", () => {
  it("rejects a write to a NEW file once EDITOR_MAX_FILES_PER_RUN distinct files have already been touched, without calling writeFile", async () => {
    // Touch files up to the cap (default EDITOR_MAX_FILES_PER_RUN=15) via
    // repeated single-write steps, then attempt one more distinct file.
    // To keep the test fast/independent of the exact configured cap value,
    // drive the loop with a small max_steps and assert the CAP MESSAGE
    // shape rather than hardcoding the numeric default -- if a write past
    // the cap is attempted, execute()'s own guard must fire before
    // writeFile() is called at all.
    const { EDITOR_MAX_FILES_PER_RUN } = await import("../config.js");

    // Step 1..N: write N distinct files, one per step, up to the cap.
    for (let i = 0; i < EDITOR_MAX_FILES_PER_RUN; i++) {
      providerChat.mockResolvedValueOnce(
        functionCallCandidate([{ name: "write_file", args: { path: `file-${i}.md`, content: "x" } }])
      );
    }
    // One more step: attempt a (N+1)th distinct file -- must be rejected
    // without ever reaching writeFile().
    providerChat.mockResolvedValueOnce(
      functionCallCandidate([{ name: "write_file", args: { path: "one-too-many.md", content: "x" } }])
    );
    providerChat.mockResolvedValueOnce(textCandidate("stopped"));

    writeFile.mockResolvedValue({ path: "x", content: "x", sha: "s", commitSha: "c1234567", diff: null, created: true, noop: false });

    const result = await runEditorAgent({
      owner: OWNER, repo: REPO, branch: BRANCH, task: "write many files",
      max_steps: EDITOR_MAX_FILES_PER_RUN + 2,
    });

    // writeFile was actually invoked exactly EDITOR_MAX_FILES_PER_RUN times
    // -- the (N+1)th attempt never reached it.
    expect(writeFile).toHaveBeenCalledTimes(EDITOR_MAX_FILES_PER_RUN);
    expect(result.transcript.join("\n")).toMatch(/per-run cap \(EDITOR_MAX_FILES_PER_RUN=/i);
    expect(result.writtenFiles).toHaveLength(EDITOR_MAX_FILES_PER_RUN);
  });

  it("rejects a write to the SAME file once EDITOR_MAX_WRITES_PER_FILE writes have already succeeded, without calling writeFile again", async () => {
    const { EDITOR_MAX_WRITES_PER_FILE } = await import("../config.js");

    for (let i = 0; i < EDITOR_MAX_WRITES_PER_FILE; i++) {
      providerChat.mockResolvedValueOnce(
        functionCallCandidate([{ name: "write_file", args: { path: "same.md", content: `v${i}` } }])
      );
    }
    providerChat.mockResolvedValueOnce(
      functionCallCandidate([{ name: "write_file", args: { path: "same.md", content: "one-more" } }])
    );
    providerChat.mockResolvedValueOnce(textCandidate("stopped"));

    writeFile.mockResolvedValue({ path: "same.md", content: "x", sha: "s", commitSha: "c1234567", diff: null, created: false, noop: false });

    const result = await runEditorAgent({
      owner: OWNER, repo: REPO, branch: BRANCH, task: "write same file repeatedly",
      max_steps: EDITOR_MAX_WRITES_PER_FILE + 2,
    });

    expect(writeFile).toHaveBeenCalledTimes(EDITOR_MAX_WRITES_PER_FILE);
    expect(result.transcript.join("\n")).toMatch(/per-file cap \(EDITOR_MAX_WRITES_PER_FILE=/i);
  });

  it("does NOT count a rejected (policy/conflict) write against either cap, since nothing was actually committed", async () => {
    providerChat.mockResolvedValueOnce(
      functionCallCandidate([{ name: "write_file", args: { path: "denied.js", content: "x" } }])
    );
    providerChat.mockResolvedValueOnce(textCandidate("done"));

    writeFile.mockRejectedValueOnce(new Error('path "denied.js" matches deny pattern'));

    const result = await runEditorAgent({ owner: OWNER, repo: REPO, branch: BRANCH, task: "try a denied path" });

    expect(result.writtenFiles).toEqual([]);
    expect(result.transcript.join("\n")).toMatch(/deny pattern/i);
  });
});
