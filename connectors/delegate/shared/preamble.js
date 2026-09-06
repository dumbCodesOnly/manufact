// ---------------------------------------------------------------------------
// connectors/delegate/shared/preamble.js — shared system-preamble/system-note
// building blocks for the delegate_* subsystems (agent, editor, designer).
//
// EXTRACTED 2026-09-06 as part of the preamble-management refactor (see the
// project handoff doc). Prior to this, each subsystem's own *_delegate.js
// duplicated small pieces of preamble-construction logic independently:
//   - agent_delegate.js's buildSystemPreamble(provider, preambleVariant) had
//     a verbose/trimmed variant switch plus a bai-only addendum, inlined
//     directly in that file.
//   - editor_delegate.js and designer_delegate.js's own buildSystemPreamble
//     ({ owner, repo, branch, task }) each independently ended their own
//     template string with the exact same "\n\nTask: ${task}" join.
//   - connectors/exa/research_delegate.js (delegate_research's wide mode)
//     has NO preamble at all -- a single direct Exa /answer call, no system
//     prompt construction of any kind -- so it does not use this module.
//
// UPDATE (2026-09-06): this module now also holds the actual prompt TEXT
// for all three write/investigate subsystems (buildAgentPreamble /
// buildEditorPreamble / buildDesignerPreamble below), not just the shared
// structural helpers -- the point of this file is to be the one place all
// delegate_* preamble content lives, so someone managing these prompts has
// one file to look in rather than three.
//
// IMPORTANT -- this is a relocation, NOT a unification: each builder below
// is its own distinct function with its own wording, moved verbatim from
// its subsystem's *_delegate.js. Nothing about what one subsystem tells its
// model has changed or converged with another. Concretely: editor's
// write_file exposes a `replacements` field for find/replace edits;
// designer's write_file exposes the same underlying concept under a
// DIFFERENT field name, `patch` -- both preserved as-is below. Editor's
// preamble explicitly states its per-run/per-file write caps
// (EDITOR_MAX_FILES_PER_RUN/EDITOR_MAX_WRITES_PER_FILE); designer's does
// not mention any equivalent caps -- also preserved as-is. Designer's
// preamble explicitly lists FRONTEND_ALLOWED_EXTENSIONS; editor's does not
// state an extension allowlist to the model at all -- same, preserved.
// These are real, currently-live behavioral differences between the
// subsystems' prompts. If a future change wants to actually converge any
// of this wording (e.g. unify replacements/patch naming), that is a
// separate, deliberate decision to make on its own -- not something this
// file's existence implies or should be used to justify quietly.
// ---------------------------------------------------------------------------

import {
  EDITOR_MAX_FILES_PER_RUN,
  EDITOR_MAX_WRITES_PER_FILE,
  FRONTEND_ALLOWED_EXTENSIONS,
} from "../../../config.js";

/**
 * Selects between a base preamble and an optional trimmed variant, then
 * appends an optional per-provider addendum. Mirrors agent_delegate.js's
 * existing buildSystemPreamble(provider, preambleVariant) exactly: same
 * verbose-by-default behavior, same "only providers with an entry in
 * `addenda` get anything appended" pattern (today, only "bai" has one, via
 * BAI_PREAMBLE_ADDENDUM, which stays defined in agent_delegate.js itself --
 * it's that subsystem's own content, not shared structure).
 *
 * Generalized (not agent-specific) so editor/designer could opt into the
 * same verbose/trimmed A/B mechanism later without reimplementing this
 * selection logic a second time -- neither does today.
 *
 * @param {object} opts
 * @param {string} opts.base - the default ("verbose") preamble text.
 * @param {string} [opts.trimmed] - an alternate, shorter preamble text. If
 *   omitted, `variant: "trimmed"` silently falls back to `base` (same
 *   defensive fallback pattern used elsewhere in this codebase for a field
 *   an older checkpoint/caller doesn't have) rather than throwing.
 * @param {"verbose"|"trimmed"} [opts.variant="verbose"]
 * @param {string} [opts.provider] - the delegate provider (e.g. "gemini",
 *   "bai"). Only used to look up `addenda`; omit if the caller's subsystem
 *   has no per-provider addenda at all.
 * @param {Record<string,string>} [opts.addenda] - provider -> addendum text
 *   to append to whichever base/trimmed text was selected. A provider with
 *   no matching entry gets no addendum.
 * @returns {string}
 */
export function selectPreambleVariant({ base, trimmed, variant = "verbose", provider, addenda = {} }) {
  const body = variant === "trimmed" && trimmed ? trimmed : base;
  const addendum = provider ? addenda[provider] : undefined;
  return addendum ? body + addendum : body;
}

/**
 * Appends a task description to a preamble body using the exact separator
 * already in use at every existing call site: a blank line, then
 * "Task: <task>". agent_delegate.js currently does this inline at its own
 * call site (`${buildSystemPreamble(...)}\n\nTask: ${task}`); editor_delegate.js
 * and designer_delegate.js currently do it inline at the END of their own
 * buildSystemPreamble template strings. All three produce byte-identical
 * output for the same (body, task) pair -- this just gives that one join a
 * single canonical home instead of three independent copies of the same
 * string concatenation.
 *
 * @param {string} preambleBody
 * @param {string} task
 * @returns {string}
 */
export function appendTask(preambleBody, task) {
  return `${preambleBody}\n\nTask: ${task}`;
}
