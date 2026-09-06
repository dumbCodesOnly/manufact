// ---------------------------------------------------------------------------
// connectors/notion/tools.js
// ---------------------------------------------------------------------------

import { z } from "zod";
import { NOTION_INDEX_DATABASE_ID, NOTION_SYNC_PARENT_PAGE_ID } from "../../config.js";
import {
  notionRequest, notionPageTitle, notionDatabaseTitle, notionRichTextToString,
  notionBlocksToText, buildMarkerBlocks, statusMarkerBlock, entityMarkerBlock, notionBlockPlainText, parseMarkers,
  buildChangelogEntryText, isChangelogEntryText,
  buildRelationBlocks, parseRelationBlocks,
  buildSyncStartText, buildSyncEndText, buildSyncRangeBlocks, findSyncRange, textBlock,
  buildCheckpointRangeBlocks, findCheckpointRange, buildCheckpointStartText,
} from "./client.js";
import { findLinkCandidates, extractTags } from "./linking.js";

const STATUS_VALUES = ["open", "resolved", "superseded"];

// ---------------------------------------------------------------------------
// Slug-like-title guard (2026-08-07 bug fix -- workspace audit found 6
// confirmed empty orphan pages: madmcp-cloudflare-workers-migration-plan,
// joblead-liliana-model-n8n, jobreq-liliana-model-n8n-detail,
// candidate-release-plz-release-plz-2130, laborx-scenium-94796,
// madmcp-generate-lockfile-workflow-recreation-2026-07-26). Root cause: the
// caller typed the STRING THEY MEANT AS entity_id into `title` instead,
// leaving entity_id unset (one_off either omitted or set true either way --
// the existing "entity_id or one_off" guard doesn't catch this shape at
// all, since one_off:true alone already satisfies it). Every confirmed
// orphan's title was lowercase, hyphen-separated, no spaces -- i.e. it was
// literally an entity_id, just in the wrong field. This regex mirrors that
// shape: 2+ lowercase/digit/hyphen segments, hyphen-joined, nothing else.
// Deliberately requires 2+ hyphens (not 1) so ordinary short hyphenated
// titles a human might actually type ("follow-up", "day-1") don't trip it --
// every real orphan had 3+ segments.
const SLUG_LIKE_TITLE = /^[a-z0-9]+(?:-[a-z0-9]+){2,}$/;

// ---------------------------------------------------------------------------
// Dedup/upsert lookup (2026-07-17, gap #1; database rewrite 2026-07-24 --
// see mem0 entity_id: madmcp-notion-connector-gaps-roadmap).
//
// FIRST FIX 2026-07-17: the original implementation leaned on notion_search
// to find candidate pages by entity_id text. Live testing confirmed that's
// fundamentally broken -- Notion's search index has real lag, and searching
// for an entity_id string immediately after creating that page (the most
// common dedup scenario) reliably returns zero results. Fixed by reading a
// dedicated index page's own blocks directly (uncached, no search lag).
//
// SECOND FIX 2026-07-24: the page-based index inherited a new gap it
// documented at the time -- /blocks/{id}/children pagination caps a single
// page's readable blocks at 100, so an index page with more than ~100
// tracked entities would silently stop finding older entries. A real Notion
// database queried via /databases/{id}/query with a filter on EntityId is
// just as immediately-consistent (no search-index lag either way, since
// this never goes through notion_search) but isn't bound by that 100-block
// limit -- database queries paginate independently of page block counts.
export async function findPageByEntityId(entity_id) {
  let rows;
  try {
    const data = await notionRequest(`/databases/${NOTION_INDEX_DATABASE_ID}/query`, {
      method: "POST",
      body: { filter: { property: "EntityId", rich_text: { equals: entity_id } }, page_size: 1 },
    });
    rows = data.results || [];
  } catch (err) {
    // Fail loudly rather than silently falling back to nothing found --
    // silently treating "index unreachable" as "no duplicate exists" would
    // just reintroduce the exact bug this fix is for.
    throw new Error(`Notion entity index database (${NOTION_INDEX_DATABASE_ID}) is unreachable, so entity_id dedup can't be verified: ${err.message}. Fix NOTION_INDEX_DATABASE_ID / the database's sharing settings before creating entity-tracked pages.`, { cause: err });
  }
  if (!rows.length) return null;
  const row = rows[0];
  const page_id = notionRichTextToString(row.properties?.PageId?.rich_text || []);
  if (!page_id) return null;
  try {
    const page = await notionRequest(`/pages/${page_id}`);
    const blocksData = await notionRequest(`/blocks/${page_id}/children?page_size=20`);
    const markers = parseMarkers(blocksData.results || []);
    return { pageId: page_id, title: notionPageTitle(page), url: page.url, markers };
  } catch {
    // Stale index row (target page deleted/archived outside these tools) --
    // treat as not-found so a fresh page can be created, rather than
    // erroring out on a dangling reference.
    return null;
  }
}

// Records a new entity_id -> page_id mapping as a row in the Entity Index
// database. Best-effort: if this fails, the page itself was still created
// successfully, so we don't throw -- but the caller surfaces the failure in
// its response text since it means the NEXT dedup check for this entity_id
// won't find it. Unlike the old page-based index, this has no pagination
// gap -- database rows aren't capped the way a single page's blocks are.
async function appendIndexEntry({ entity_id, page_id, url, tags }) {
  try {
    await notionRequest("/pages", {
      method: "POST",
      body: {
        parent: { database_id: NOTION_INDEX_DATABASE_ID },
        properties: {
          Name:     { title: [{ text: { content: entity_id } }] },
          EntityId: { rich_text: [{ text: { content: entity_id } }] },
          PageId:   { rich_text: [{ text: { content: page_id } }] },
          Url:      { url: url || null },
          Tags:     { multi_select: (tags || []).map((name) => ({ name })) },
        },
      },
    });
    return null;
  } catch (err) {
    return err.message;
  }
}

// ---------------------------------------------------------------------------
// Backfill/repair path (2026-07-24, index reset -- see Notion plan page
// "Entity Index Migration"). notion_create_page/appendIndexEntry always
// index the page THEY just created -- there was no way to add an index row
// for a page that already exists elsewhere (e.g. after the index database
// was reset empty and existing entity_id-marked pages needed backfilling).
// Reuses appendIndexEntry unchanged, same idempotent skip-if-present check
// as doCreatePage, just without also creating a page.
export async function upsertIndexEntry({ entity_id, page_id, url, tags }) {
  const existing = await findPageByEntityId(entity_id);
  if (existing) return { skipped: true, existingId: existing.pageId };
  const error = await appendIndexEntry({ entity_id, page_id, url, tags: tags || [] });
  return { skipped: false, error };
}

// ---------------------------------------------------------------------------
// Shared create-page logic (2026-07-17, gap #6 -- see mem0 entity_id:
// madmcp-notion-connector-gaps-roadmap). Extracted out of notion_create_page's
// handler so notion_create_pages_batch can reuse the exact same dedup +
// marker + index-recording behavior per item, mirroring how mem/tools.js's
// mem0_add and mem0_add_batch share logic. Returns a plain result object
// instead of an MCP content block -- callers format the response.
// 2026-07-18: NOT a hard entity_id requirement -- forcing entity_id on
// every page would pollute the index with genuine scratch/one-off content
// (test pages, quick notes) that was never meant to be deduped or tracked,
// which defeats the index's purpose and doesn't match the same tradeoff
// mem0_add already makes (entity_id optional there too, for the same
// reason). Instead: require an EXPLICIT choice. Omitting entity_id AND
// one_off is the actual failure mode worth catching -- someone forgetting
// to track a thing that should be tracked -- so that case now throws
// instead of silently creating an untracked page. Passing one_off: true is
// the deliberate opt-out for real one-offs.
export async function doCreatePage({ parent_id, parent_type, title, content, entity_id, status, relations, one_off, properties }) {
  if (!entity_id && !one_off) {
    throw new Error(`Refusing to create "${title}" without a tracking decision -- pass either entity_id (if this represents an ongoing/stable thing that should be deduped and indexed) or one_off: true (if it's genuinely disposable, e.g. a scratch note or test page). This is a deliberate choice, not a bug -- see notion_create_page's entity_id and one_off param descriptions.`);
  }
  // See SLUG_LIKE_TITLE comment above -- this fires regardless of one_off,
  // since the observed bug happened with one_off both set and unset. A
  // title this shaped is almost never a real human-readable title.
  if (!entity_id && SLUG_LIKE_TITLE.test(title)) {
    throw new Error(`Refusing to create a page titled "${title}" -- this looks like an entity_id (lowercase, hyphen-separated, no spaces), not a human-readable title, and entity_id is unset. This is almost always a mistake: pass this exact string as entity_id instead, and give "title" an actual readable name (e.g. title: "Cloudflare Workers migration plan", entity_id: "${title}"). If "${title}" is genuinely, deliberately meant to be the literal page title, pass entity_id explicitly (it can be any string, including this same one) to confirm that's intentional -- entity_id is still required or one_off must be true per the check above.`);
  }
  if (entity_id) {
    const existing = await findPageByEntityId(entity_id);
    if (existing) {
      return { skipped: true, entity_id, existingId: existing.pageId, existingTitle: existing.title, existingUrl: existing.url };
    }
  }

  // Deterministic (no-LLM, no-mem0) related-page detection -- see
  // linking.js header comment and Notion plan page (entity_id:
  // plan-notion-autolink-heuristic). Best-effort: a failure here (e.g.
  // Notion search unreachable) should never block page creation, since this
  // is a convenience layer on top of an otherwise-complete create call.
  let linkCandidates = { strong: [], medium: [] };
  try {
    linkCandidates = await findLinkCandidates({ title, content });
  } catch {
    // swallow -- see comment above
  }
  const explicitRelations = relations || [];
  const explicitTargets   = new Set(explicitRelations.map((r) => r.to_entity_id));
  const autoRelations = linkCandidates.strong
    .filter((c) => c.entity_id && c.entity_id !== entity_id && !explicitTargets.has(c.entity_id))
    .map((c) => ({ to_entity_id: c.entity_id, relation: "relates_to" }));
  const mergedRelations = [...explicitRelations, ...autoRelations];

  const parent         = parent_type === "database" ? { database_id: parent_id } : { page_id: parent_id };
  const pageProperties = parent_type === "database"
    ? { Name:  { title: [{ text: { content: title } }] }, ...(properties || {}) }
    : { title: { title: [{ text: { content: title } }] } };
  const markerBlocks   = buildMarkerBlocks({ entity_id, status });
  const relationBlocks = buildRelationBlocks(mergedRelations);
  const contentBlocks = content
    ? content.split("\n").filter(Boolean).map(textBlock)
    : [];
  const children = [...markerBlocks, ...relationBlocks, ...contentBlocks];

  // Notion's page-create endpoint rejects more than 100 children blocks in
  // a single call (live failure 2026-07-25: a long delegate_agent
  // transcript produced 199 blocks and got a 400 "body.children.length
  // should be <= 100"). Create the page with the first 100, then PATCH the
  // rest on afterward in further batches of <=100 -- same post-creation
  // append pattern appendChangelogEntry/replaceSyncedRange already use --
  // instead of silently truncating long content.
  const firstBatch = children.slice(0, 100);
  const remainingBatches = [];
  for (let i = 100; i < children.length; i += 100) {
    remainingBatches.push(children.slice(i, i + 100));
  }
  const data = await notionRequest("/pages", {
    method: "POST",
    body: { parent, properties: pageProperties, children: firstBatch },
  });

  // If a later batch fails (rate limit, network blip), the page ITSELF
  // still exists on Notion at this point -- letting the error propagate
  // immediately, before the index write below, would orphan it from the
  // dedup index. A retry with the same entity_id would then find nothing
  // and create a genuine duplicate page, exactly the failure mode the index
  // exists to prevent. So: capture the failure, still record the index
  // entry (the page really was created), then surface the partial-content
  // failure with the page's id/url so the caller can finish it via
  // notion_update_page instead of losing track of it.
  let batchError = null;
  for (const batch of remainingBatches) {
    try {
      await notionRequest(`/blocks/${data.id}/children`, { method: "PATCH", body: { children: batch } });
    } catch (err) {
      batchError = err;
      break;
    }
  }

  let indexError = null;
  if (entity_id) {
    indexError = await appendIndexEntry({ entity_id, page_id: data.id, url: data.url, tags: [...extractTags(content || "")] });
  }

  if (batchError) {
    throw new Error(
      `Page "${title}" was created (id: ${data.id}, url: ${data.url}) but is missing some content -- a later content batch failed: ${batchError.message}. ` +
      (entity_id
        ? `It IS recorded in the dedup index${indexError ? ` (though that index write also failed: ${indexError})` : ""}, so retrying notion_create_page with the same entity_id will find this page rather than creating a duplicate -- use notion_update_page (append_content) on id ${data.id} to add the missing content instead.`
        : `No entity_id was set, so there's no dedup protection -- check the page at the URL above before retrying, to avoid creating a duplicate, and use notion_update_page (append_content) on id ${data.id} to add the missing content.`)
    );
  }

  return { skipped: false, id: data.id, url: data.url, title, markerCount: markerBlocks.length, relationCount: relationBlocks.length, entity_id, status, indexError, linkCandidates, autoRelations };
}

// Sequential batch runner, mimicking Promise.allSettled's per-item
// {status, value|reason} shape so callers don't need to change their
// result-formatting code. NOT run in parallel -- BUG FOUND 2026-07-17 live
// testing: notion_create_pages_batch originally used Promise.allSettled,
// which let two items sharing the same entity_id both pass
// findPageByEntityId's dedup check concurrently (neither had written its
// index entry yet when the other checked), creating two pages for one
// entity_id in a single batch call. Running strictly in order guarantees
// each item's dedup check sees every earlier item's completed index write.
// Trades batch throughput for correctness -- acceptable at this tool's
// scale (personal/small-team usage, not high-volume bulk import).
async function runSequentially(items, fn) {
  const results = [];
  for (const item of items) {
    try {
      const value = await fn(item);
      results.push({ status: "fulfilled", value });
    } catch (reason) {
      results.push({ status: "rejected", reason });
    }
  }
  return results;
}

const EDITABLE_BLOCK_TYPES = ["paragraph", "heading_1", "heading_2", "heading_3", "bulleted_list_item", "numbered_list_item", "to_do"];

// ---------------------------------------------------------------------------
// Synced-range block replace (2026-07-18, mem0->Notion Sync Tool spec --
// see mem0 entity_id: mem0-notion-sync-tool-spec). See client.js's
// "Synced-range marker convention" comment for the marker format and why
// this exists (protecting manual edits from being clobbered by a re-sync).
// Same 100-block-page read limitation as findPageByEntityId/parseMarkers
// elsewhere in this file -- a range on a page with >100 total blocks may
// not be found; treated as not-found (append fresh range) rather than a
// silent corruption risk, same reasoning as findSyncRange's unterminated-
// range case.
export async function replaceSyncedRange({ page_id, contentLines, synced_at }) {
  const blocksData = await notionRequest(`/blocks/${page_id}/children?page_size=100`);
  const blocks = blocksData.results || [];
  const range = findSyncRange(blocks);

  if (!range) {
    const children = buildSyncRangeBlocks({ synced_at, contentLines });
    await notionRequest(`/blocks/${page_id}/children`, { method: "PATCH", body: { children } });
    return { action: "created", blockCount: children.length };
  }

  if (range.synced_at === synced_at) {
    return { action: "skipped", reason: `already up to date (mem0_synced_at: ${synced_at})` };
  }

  // Delete every block strictly between the markers -- never the markers
  // themselves, and never anything past the end marker.
  for (const blockId of range.innerBlockIds) {
    await notionRequest(`/blocks/${blockId}`, { method: "DELETE" });
  }

  // Insert new content right after the start marker via Notion's `after`
  // cursor, so it lands inside the range regardless of what (if anything)
  // sits below the end marker.
  const contentBlocks = (contentLines || []).filter(Boolean).map(textBlock);
  if (contentBlocks.length) {
    await notionRequest(`/blocks/${page_id}/children`, {
      method: "PATCH",
      body: { children: contentBlocks, after: range.startBlockId },
    });
  }

  // Update the start marker's own text in place with the new timestamp --
  // same single-block PATCH doUpdatePage uses for the status marker.
  await notionRequest(`/blocks/${range.startBlockId}`, {
    method: "PATCH",
    body: { paragraph: { rich_text: [{ type: "text", text: { content: buildSyncStartText(synced_at) } }] } },
  });

  return { action: "updated", removed: range.innerBlockIds.length, added: contentBlocks.length, previousSyncedAt: range.synced_at };
}

// Best-effort changelog append (gap #4) -- swallows its own errors rather
// than throwing, since a failed history write shouldn't roll back or block
// an otherwise-successful page update. Returns an error string (for the
// caller to optionally surface) or null on success.
async function appendChangelogEntry(page_id, summary) {
  try {
    await notionRequest(`/blocks/${page_id}/children`, {
      method: "PATCH",
      body: { children: [{
        object: "block", type: "paragraph",
        paragraph: { rich_text: [{ type: "text", text: { content: buildChangelogEntryText(summary) } }] },
      }] },
    });
    return null;
  } catch (err) {
    return err.message;
  }
}

// ---------------------------------------------------------------------------
// Shared update-page logic (2026-07-17, gap #6 -- mirrors doCreatePage above).
// Extracted out of notion_update_page's handler so notion_update_pages_batch
// can reuse the exact same title/append_content/archived/replacements/status
// behavior per item. Returns an array of result strings on success, or
// THROWS on any abort condition (ambiguous/missing replacement match,
// unsupported block type) -- the single-item tool catches this to preserve
// its existing isError response shape; the batch tool lets Promise.allSettled
// catch it per item, same pattern as mem/tools.js.
export async function doUpdatePage({ page_id, title, append_content, archived, replacements, status, entity_id, relations, properties }) {
  const results = [];
  // Unarchive (or a title-only change) runs first, same as before -- this
  // leaves the page editable for any block-level edits below. Archiving
  // (archived: true) is deliberately NOT handled here -- see the bottom of
  // this function for why it's deferred to run last.
  if (title !== undefined || archived === false || (properties !== undefined && archived !== true)) {
    const body = {};
    if (archived !== undefined) body.archived = archived;
    const propUpdates = {};
    if (title      !== undefined) propUpdates.title = { title: [{ text: { content: title } }] };
    if (properties !== undefined) Object.assign(propUpdates, properties);
    if (Object.keys(propUpdates).length) body.properties = propUpdates;
    const data = await notionRequest(`/pages/${page_id}`, { method: "PATCH", body });
    results.push(`Updated page "${notionPageTitle(data)}" (ID: ${data.id}).`);
  }
  if (append_content) {
    const children = append_content.split("\n").filter(Boolean).map(textBlock);
    await notionRequest(`/blocks/${page_id}/children`, { method: "PATCH", body: { children } });
    results.push(`Appended ${children.length} paragraph(s) to page.`);
  }
  if (replacements?.length) {
    const blocksData = await notionRequest(`/blocks/${page_id}/children?page_size=100`);
    const blocks = blocksData.results || [];
    for (const { find, replace } of replacements) {
      const matches = blocks.filter((b) => notionBlockPlainText(b) === find);
      const trunc = (s) => s.slice(0, 60) + (s.length > 60 ? "…" : "");
      if (matches.length === 0) {
        throw new Error(`Update aborted, nothing further written — "${trunc(find)}" was not found among this page's top-level blocks (first 100). It may be nested inside a toggle/column, or the page may have more than 100 blocks — re-check with notion_get_page.`);
      }
      if (matches.length > 1) {
        throw new Error(`Update aborted, nothing further written — "${trunc(find)}" matches ${matches.length} blocks, but must be unique. Include more surrounding context in "find" to disambiguate.`);
      }
      const block = matches[0];
      const type  = block.type;
      if (!EDITABLE_BLOCK_TYPES.includes(type)) {
        throw new Error(`Update aborted, nothing further written — matched block is type "${type}", which notion_update_page can't edit in place yet (supported: ${EDITABLE_BLOCK_TYPES.join(", ")}).`);
      }
      const patchBody = { [type]: { rich_text: [{ type: "text", text: { content: replace } }] } };
      if (type === "to_do") patchBody[type].checked = block.to_do?.checked ?? false;
      await notionRequest(`/blocks/${block.id}`, { method: "PATCH", body: patchBody });
      results.push(`Replaced block ("${trunc(find)}" → "${trunc(replace)}").`);
    }
  }
  if (status !== undefined) {
    const blocksData = await notionRequest(`/blocks/${page_id}/children?page_size=100`);
    const markers = parseMarkers(blocksData.results || []);
    if (markers.statusBlockId) {
      await notionRequest(`/blocks/${markers.statusBlockId}`, {
        method: "PATCH",
        body: statusMarkerBlock(status),
      });
      results.push(`Status updated to "${status}" (was "${markers.status}").`);
    } else {
      await notionRequest(`/blocks/${page_id}/children`, { method: "PATCH", body: { children: [statusMarkerBlock(status)] } });
      results.push(`Status marker added: "${status}" (page had none before).`);
    }
  }
  // entity_id (bug fix 2026-08-07, see doCreatePage/findPageByEntityId --
  // this is the missing counterpart to that: a supported way to CORRECT an
  // entity_id after creation that keeps the marker block and the Entity
  // Index database in sync, instead of a caller hand-editing the marker via
  // `replacements` and silently leaving the index stale/pointing nowhere.
  // Same marker-block-in-place pattern as `status` above.
  if (entity_id !== undefined) {
    const blocksData = await notionRequest(`/blocks/${page_id}/children?page_size=100`);
    const markers = parseMarkers(blocksData.results || []);
    const previousEntityId = markers.entity_id;
    if (markers.entityBlockId) {
      await notionRequest(`/blocks/${markers.entityBlockId}`, { method: "PATCH", body: { paragraph: entityMarkerBlock(entity_id).paragraph } });
    } else {
      await notionRequest(`/blocks/${page_id}/children`, { method: "PATCH", body: { children: [entityMarkerBlock(entity_id)] } });
    }
    // Re-fetch the page for its (stable) url -- appendIndexEntry needs it
    // and none of the branches above are guaranteed to have fetched it.
    const page = await notionRequest(`/pages/${page_id}`);
    const indexError = await appendIndexEntry({ entity_id, page_id, url: page.url, tags: [] });
    // NOTE: this does not delete/update the OLD entity_id's index row (if
    // any) -- best-effort, same tradeoff appendIndexEntry's own callers
    // already accept elsewhere in this file. The old row still resolves
    // lookups made against the old (now-wrong) entity_id to this page,
    // which is stale but not actively harmful; the bug this fixes is
    // specifically that lookups against the CORRECTED entity_id were
    // failing, and those now succeed immediately since appendIndexEntry
    // writes synchronously before this call returns.
    results.push(
      `Entity ID updated to "${entity_id}"${previousEntityId ? ` (was "${previousEntityId}")` : " (page had none before)"}.` +
      (indexError ? ` \u26a0\ufe0f index write failed: ${indexError} -- relation lookups against "${entity_id}" may still report dangling until this is retried.` : "")
    );
  }
  // relations REPLACES the existing set whole (not merged), same contract as
  // mem0_update's relations param. Requires reading current blocks to find
  // the existing relation blocks to remove -- reuses blocksData if a
  // replacements/status branch above already fetched it, to avoid a
  // redundant call.
  if (relations !== undefined) {
    const blocksData = await notionRequest(`/blocks/${page_id}/children?page_size=100`);
    const existingRelations = parseRelationBlocks(blocksData.results || []);
    for (const r of existingRelations) {
      await notionRequest(`/blocks/${r.blockId}`, { method: "DELETE" });
    }
    const newBlocks = buildRelationBlocks(relations);
    if (newBlocks.length) {
      await notionRequest(`/blocks/${page_id}/children`, { method: "PATCH", body: { children: newBlocks } });
    }
    results.push(`Relations replaced: ${existingRelations.length} removed, ${newBlocks.length} added.`);
  }
  // Archiving (archived: true) runs LAST, after append_content/
  // replacements/status/relations have all completed above -- confirmed via
  // live testing 2026-07-23 that Notion rejects block-level edits on an
  // already-archived page ("Can't edit block that is archived"), so doing
  // this step first (as the old code did, bundled with title) let the
  // archive silently succeed while a later block edit in the same call
  // threw, producing a confusing partial-success-then-error result. Title
  // is included here too if it wasn't already applied above, so a single
  // call with {title, archived: true} still sets both in one PATCH.
  if (archived === true) {
    const body = { archived: true };
    const propUpdates = {};
    if (title      !== undefined) propUpdates.title = { title: [{ text: { content: title } }] };
    if (properties !== undefined) Object.assign(propUpdates, properties);
    if (Object.keys(propUpdates).length) body.properties = propUpdates;
    const data = await notionRequest(`/pages/${page_id}`, { method: "PATCH", body });
    results.push(`Updated page "${notionPageTitle(data)}" (ID: ${data.id}).`);
  }
  // Skip the changelog write when this call archived the page -- Notion
  // rejects block edits on an already-archived page ("Can't edit block that
  // is archived"), confirmed via live testing 2026-07-17. Unarchiving
  // (archived: false) is fine since the page is editable again by then.
  if (results.length && archived !== true) {
    const changelogError = await appendChangelogEntry(page_id, results.join("; "));
    if (changelogError) results.push(`(\u26a0\ufe0f changelog entry not recorded: ${changelogError})`);
  }
  return results;
}

// ---------------------------------------------------------------------------
// Checkpoint range replace -- mirrors replaceSyncedRange below, but reads/
// writes the checkpoint markers (client.js's findCheckpointRange /
// buildCheckpointStartText) instead of the mem0 sync markers. Kept as its
// own function rather than parameterizing replaceSyncedRange, since the two
// callers (mem0 sync vs. checkpoint save) have historically diverged in
// small ways and mem0-specific skip/logging behavior in replaceSyncedRange
// shouldn't silently start applying to checkpoint saves or vice versa.
export async function replaceCheckpointRange({ page_id, contentLines, updated_at }) {
  const blocksData = await notionRequest(`/blocks/${page_id}/children?page_size=100`);
  const blocks = blocksData.results || [];
  const range = findCheckpointRange(blocks);

  if (!range) {
    const children = buildCheckpointRangeBlocks({ updated_at, contentLines });
    await notionRequest(`/blocks/${page_id}/children`, { method: "PATCH", body: { children } });
    return { action: "created", blockCount: children.length };
  }

  // Unlike replaceSyncedRange, always overwrite -- a checkpoint save is
  // meant to reflect "now", so there's no meaningful "already up to date"
  // skip case the way there is for a mem0 memory that hasn't changed.
  for (const blockId of range.innerBlockIds) {
    await notionRequest(`/blocks/${blockId}`, { method: "DELETE" });
  }

  const contentBlocks = (contentLines || []).filter(Boolean).map(textBlock);
  if (contentBlocks.length) {
    await notionRequest(`/blocks/${page_id}/children`, {
      method: "PATCH",
      body: { children: contentBlocks, after: range.startBlockId },
    });
  }

  await notionRequest(`/blocks/${range.startBlockId}`, {
    method: "PATCH",
    body: { paragraph: { rich_text: [{ type: "text", text: { content: buildCheckpointStartText(updated_at) } }] } },
  });

  return { action: "updated", removed: range.innerBlockIds.length, added: contentBlocks.length, previousUpdatedAt: range.updated_at };
}

// ---------------------------------------------------------------------------
// Checkpoint helper (Notion Session Checkpoint Tool - REVISION 3, 2026-09-04
// -- switched off the mem0 sync markers, which wrote confusing/inaccurate
// "SYNCED FROM MEM0" text on every checkpoint save even though this tool has
// nothing to do with mem0. See client.js's checkpoint marker convention.
// REDEPLOY TRIGGER: madmcp.vercel.app's alias got stuck on 554a108 (the
// client.js-only commit) instead of advancing to this commit -- this no-op
// comment forces a new deployment so the alias promotion re-runs.)
// ---------------------------------------------------------------------------
export async function doCheckpoint({ action, notes, replacements, append_notes }) {
  if (action === "save") {
    const existing = await findPageByEntityId("checkpoint-latest");
    const notesLines = (notes || "").split("\n");
    const updated_at = new Date().toISOString();

    if (!existing) {
      // Seed the checkpoint range directly in the page's initial content,
      // same reasoning as before: avoids a briefly rangeless page, and
      // replaceCheckpointRange always re-reads blocks first to check for an
      // existing range, which a brand-new page can never have.
      const contentText = [buildCheckpointStartText(updated_at), ...notesLines, "\u2705 End synced checkpoint"].join("\n");
      const created = await doCreatePage({
        parent_id: NOTION_SYNC_PARENT_PAGE_ID,
        parent_type: "page",
        title: "Session Checkpoint",
        entity_id: "checkpoint-latest",
        content: contentText,
      });
      return `Checkpoint saved successfully.\nURL: ${created.url}`;
    }

    await replaceCheckpointRange({ page_id: existing.pageId, contentLines: notesLines, updated_at });
    return `Checkpoint saved successfully.\nURL: ${existing.url}`;
  } else if (action === "load") {
    const existing = await findPageByEntityId("checkpoint-latest");
    if (!existing) {
      return "No checkpoint found.";
    }
    const blocksData = await notionRequest(`/blocks/${existing.pageId}/children?page_size=100`);
    const blocks = blocksData.results || [];
    const range = findCheckpointRange(blocks);
    if (!range) {
      return "No checkpoint found.";
    }
    const blockMap = new Map(blocks.map((b) => [b.id, b]));
    const innerBlocks = range.innerBlockIds.map((id) => blockMap.get(id)).filter(Boolean);
    const notesContent = notionBlocksToText(innerBlocks);
    return notesContent || "(empty checkpoint)";
  } else if (action === "update") {
    // Targeted edit path -- avoids replaceCheckpointRange's delete-every-
    // inner-block-then-recreate behavior, which is wasteful (and racks up
    // real Notion API calls) when a session just wants to tweak or extend
    // an existing checkpoint rather than replace it wholesale.
    const existing = await findPageByEntityId("checkpoint-latest");
    if (!existing) {
      throw new Error("No checkpoint found to update -- use action: \"save\" first to create one.");
    }
    if (!replacements?.length && !append_notes) {
      throw new Error("action: \"update\" requires at least one of 'replacements' or 'append_notes' -- otherwise there's nothing to update. Use action: \"save\" for a full rewrite, or action: \"load\" to just read the current content.");
    }
    const blocksData = await notionRequest(`/blocks/${existing.pageId}/children?page_size=100`);
    const blocks = blocksData.results || [];
    const range = findCheckpointRange(blocks);
    if (!range) {
      throw new Error("Checkpoint page exists but no checkpoint range was found on it (may exceed the 100-block read window) -- use action: \"save\" to recreate it cleanly.");
    }
    const blockMap = new Map(blocks.map((b) => [b.id, b]));
    const innerBlocks = range.innerBlockIds.map((id) => blockMap.get(id)).filter(Boolean);
    const results = [];
    const trunc = (s) => s.slice(0, 60) + (s.length > 60 ? "\u2026" : "");

    if (replacements?.length) {
      // Validate ALL replacements against the pre-write snapshot before
      // writing ANY of them (bug fix 2026-09-07 -- previously this loop
      // validated and PATCHed each replacement in the same iteration, so a
      // bad find later in the list threw AFTER earlier ones had already
      // been written to Notion, even though the error claimed "nothing
      // further written". That left checkpoints silently half-updated.
      // Block IDs don't change across these edits, so it's safe to resolve
      // every find against the same original innerBlocks snapshot up front.
      const resolved = replacements.map(({ find, replace }) => {
        const matches = innerBlocks.filter((b) => notionBlockPlainText(b) === find);
        if (matches.length === 0) {
          throw new Error(`Update aborted, nothing further written \u2014 "${trunc(find)}" was not found among the checkpoint's current lines. Use checkpoint (action: "load") to see current content, or action: "save" for a full rewrite.`);
        }
        if (matches.length > 1) {
          throw new Error(`Update aborted, nothing further written \u2014 "${trunc(find)}" matches ${matches.length} lines, but must be unique. Include more surrounding context in "find" to disambiguate.`);
        }
        const block = matches[0];
        await notionRequest(`/blocks/${block.id}`, {
          method: "PATCH",
          body: { paragraph: { rich_text: [{ type: "text", text: { content: replace } }] } },
        });
        results.push(`Replaced line ("${trunc(find)}" \u2192 "${trunc(replace)}").`);
      }
    }

    if (append_notes) {
      const newLines = append_notes.split("\n").filter(Boolean);
      const children = newLines.map(textBlock);
      if (children.length) {
        const afterId = range.innerBlockIds.length ? range.innerBlockIds[range.innerBlockIds.length - 1] : range.startBlockId;
        await notionRequest(`/blocks/${existing.pageId}/children`, {
          method: "PATCH",
          body: { children, after: afterId },
        });
        results.push(`Appended ${children.length} new line(s).`);
      }
    }

    // Bump the start marker's timestamp either way, so action: "load" and
    // the visible marker both reflect that the checkpoint has moved since
    // its last full save, even though this path never touched the marker
    // block for any other reason.
    const updated_at = new Date().toISOString();
    await notionRequest(`/blocks/${range.startBlockId}`, {
      method: "PATCH",
      body: { paragraph: { rich_text: [{ type: "text", text: { content: buildCheckpointStartText(updated_at) } }] } },
    });

    return `Checkpoint updated successfully (targeted edit, no full rewrite).\n${results.join("\n")}\nURL: ${existing.url}`;
  } else {
    throw new Error(`Invalid checkpoint action: "${action}" (expected "save", "load", or "update").`);
  }
}

export function register(server) {

  server.tool(
    "checkpoint",
    "Save, load, or update a handoff note for the CURRENT session so a fresh session can recover context — NOT a general-purpose notes tool. Uses a fixed global checkpoint entity ('checkpoint-latest'). 'save' fully rewrites the stored note (use for the first save in a session, or a genuine full replacement). 'update' makes a targeted edit instead of a full rewrite — use this for later checkpoints within the same session so each call doesn't delete and recreate every line.",
    {
      action:       z.enum(["save", "load", "update"]).describe("Action to perform: 'save' to fully (re)write the handoff notes, 'load' to retrieve them, 'update' to make a targeted edit (replacements and/or append_notes) without rewriting the whole checkpoint"),
      notes:        z.string().optional().describe("Freeform plain-text handoff notes to save (only used for action: 'save' — full rewrite)"),
      replacements: z.array(z.object({
        find:    z.string().describe("Exact plain text of an existing checkpoint line — must match exactly one line"),
        replace: z.string().describe("New plain text for that line"),
      })).optional().describe("Only used for action: 'update'. Targeted find/replace edits applied to specific existing lines in the checkpoint, instead of rewriting the whole note. Each 'find' must match exactly one current line — fails with nothing written on zero or multiple matches."),
      append_notes: z.string().optional().describe("Only used for action: 'update'. Plain-text lines to append after the checkpoint's existing content, without touching anything already there. Combine with 'replacements' in the same call if needed."),
    },
    async ({ action, notes, replacements, append_notes }) => {
      try {
        const text = await doCheckpoint({ action, notes, replacements, append_notes });
        return { content: [{ type: "text", text }] };
      } catch (err) {
        return { content: [{ type: "text", text: err.message }], isError: true };
      }
    }
  );

  server.tool(
    "notion_search",
    "DOES: searches pages and databases in your Notion workspace.\nRULE for the calling model: use this only for a single, targeted lookup. If you'll need to search and then read more than 2 pages, or the request asks you to understand, review, or summarize a whole area of the Notion workspace -- regardless of how it's phrased ('go through our notes on X', 'get up to speed on the workspace', 'dig into our docs', etc. all count) -- use delegate_agent instead of looping notion_search and notion_get_page manually.",
    {
      query:       z.string().describe("Search query string"),
      filter_type: z.enum(["page", "database"]).optional().describe("Filter results to only pages or only databases (default: both)"),
      page_size:   z.number().optional().describe("Number of results to return (default: 10, max: 100)"),
    },
    async ({ query, filter_type, page_size = 10 }) => {
      const body = { query, page_size };
      if (filter_type) body.filter = { value: filter_type, property: "object" };
      const data = await notionRequest("/search", { method: "POST", body });
      if (!data.results?.length) return { content: [{ type: "text", text: "No results found." }] };
      const lines = data.results.map((r) => {
        const title = r.object === "page"
          ? notionPageTitle(r)
          : (notionRichTextToString(r.title) || "(untitled)");
        return `[${r.object}] ${title}\n  ID: ${r.id}\n  URL: ${r.url || ""}`;
      });
      return { content: [{ type: "text", text: lines.join("\n\n") }] };
    }
  );

  server.tool(
    "notion_list",
    "List recent pages and/or databases in your Notion workspace, sorted by most recently edited first -- no search query needed. Use this (not notion_search) when the ask is 'get the latest entry/page' or 'what's new in Notion', since notion_search requires a keyword and doesn't guarantee recency ordering.",
    {
      filter_type: z.enum(["page", "database"]).optional().describe("Restrict results to only pages or only databases (default: both)"),
      page_size:   z.number().optional().describe("Number of results to return (default 20, max 100). Pass 1 to get just the single latest entry."),
    },
    async ({ filter_type, page_size = 20 }) => {
      const body = { query: "", sort: { direction: "descending", timestamp: "last_edited_time" }, page_size };
      if (filter_type) body.filter = { value: filter_type, property: "object" };
      const data = await notionRequest("/search", { method: "POST", body });
      if (!data.results?.length) return { content: [{ type: "text", text: "No pages or databases found." }] };
      const lines = data.results.map((r) => {
        const title = r.object === "page"
          ? notionPageTitle(r)
          : (notionRichTextToString(r.title) || "(untitled)");
        return `[${r.object}] ${title}\n  ID: ${r.id}\n  URL: ${r.url || ""}\n  Last edited: ${r.last_edited_time?.slice(0, 16)}`;
      });
      return { content: [{ type: "text", text: lines.join("\n\n") }] };
    }
  );

  server.tool(
    "notion_get_page",
    "DOES: gets a Notion page's properties and content blocks.\nRULE for the calling model: only call this directly for a single, specifically-named page whose ID you already have. If you'll need to read more than 2 pages, or the task involves understanding or reviewing a whole area of the workspace rather than one known page, use delegate_agent instead of looping notion_get_page across pages.",
    {
      page_id: z.string().describe("Notion page ID (UUID format, e.g. from notion_search)"),
      cursor:  z.string().optional().describe("Pagination cursor from a previous call's response (see the more-blocks note) -- fetches the next page of up to 100 blocks instead of starting over. Omit for the first call."),
    },
    async ({ page_id, cursor }) => {
      const blocksPath = `/blocks/${page_id}/children?page_size=100${cursor ? `&start_cursor=${encodeURIComponent(cursor)}` : ""}`;
      const [page, blocksData] = await Promise.all([
        notionRequest(`/pages/${page_id}`),
        notionRequest(blocksPath),
      ]);
      const title     = notionPageTitle(page);
      const allBlocks = blocksData.results || [];
      // Changelog entries (gap #4) are kept out of the normal content view --
      // they're an operational log, not page content -- surfaced instead via
      // notion_get_page_history. Filtered only from what's *shown* here, not
      // from the raw block count, since they still occupy real block slots.
      const blocks = allBlocks.filter((b) => !(b.type === "paragraph" && isChangelogEntryText(notionRichTextToString(b.paragraph?.rich_text || []))));
      const changelogCount = allBlocks.length - blocks.length;
      const content  = notionBlocksToText(blocks);
      const hasMore  = blocksData.has_more
        ? `\n\n⚠️ Page has more blocks — call notion_get_page again with cursor: "${blocksData.next_cursor}" to see the next page.`
        : "";
      const subPages     = blocks.filter((b) => b.type === "child_page").length;
      const subDatabases = blocks.filter((b) => b.type === "child_database").length;
      const childSummary = (subPages || subDatabases)
        ? `\n\n🔗 ${subPages} subpage(s), ${subDatabases} subdatabase(s) found — use notion_get_page on their IDs above to view them.`
        : "";
      const changelogNote = changelogCount ? `\n📜 ${changelogCount} changelog entr${changelogCount === 1 ? "y" : "ies"} on this page (this view) — use notion_get_page_history to see them.` : "";
      const markers      = parseMarkers(allBlocks);
      const markerLine   = (markers.entity_id || markers.status)
        ? `\n${markers.entity_id ? `Entity ID: ${markers.entity_id}` : ""}${markers.entity_id && markers.status ? " | " : ""}${markers.status ? `Status: ${markers.status}` : ""}`
        : "";
      // Relations (gap #5) -- resolve up to 5 outgoing relations to their
      // target's title/url via the same dedup index lookup findPageByEntityId
      // uses, so a person reading this doesn't have to manually chase each
      // to_entity_id. Capped at 5 to bound the extra API calls this costs
      // (each resolution is a full findPageByEntityId, itself 1-2 calls);
      // remaining relations are still listed, just unresolved.
      const relations = parseRelationBlocks(allBlocks);
      let relationsBlock = "";
      if (relations.length) {
        const toResolve = relations.slice(0, 5);
        const resolved = await Promise.all(toResolve.map(async (r) => {
          try {
            const target = await findPageByEntityId(r.to_entity_id);
            return target ? `  🔗 ${r.relation} -> ${r.to_entity_id} ("${target.title}", ${target.url})` : `  🔗 ${r.relation} -> ${r.to_entity_id} (not found -- dangling reference)`;
          } catch {
            return `  🔗 ${r.relation} -> ${r.to_entity_id} (couldn't resolve -- index unreachable)`;
          }
        }));
        const remaining = relations.length - toResolve.length;
        relationsBlock = `\n\nRelations:\n${resolved.join("\n")}${remaining ? `\n  … and ${remaining} more (not resolved, showing first 5)` : ""}`;
      }
      const text =
        `# ${title}\n` +
        `ID: ${page.id}\n` +
        `URL: ${page.url}\n` +
        `Created: ${page.created_time?.slice(0, 10)} | Last edited: ${page.last_edited_time?.slice(0, 10)}${markerLine}${changelogNote}\n\n` +
        (content || "(no content)") + hasMore + childSummary + relationsBlock;
      return { content: [{ type: "text", text }] };
    }
  );

  server.tool(
    "notion_get_page_history",
    "Get the version/change history of a Notion page -- every notion_update_page call recorded against it, with a summary of what changed and when. Notion's API has no native page-revision endpoint (unlike mem0_get_history, which wraps one), so this reads back the append-only changelog blocks notion_update_page writes on every successful change. Only covers changes made through these tools, not edits made directly in the Notion UI or by other integrations.",
    {
      page_id: z.string().describe("Notion page ID (UUID format, e.g. from notion_search)"),
      cursor:  z.string().optional().describe("Pagination cursor from a previous call, to see older history beyond the first 100 blocks scanned. Omit for the first call."),
    },
    async ({ page_id, cursor }) => {
      const blocksPath = `/blocks/${page_id}/children?page_size=100${cursor ? `&start_cursor=${encodeURIComponent(cursor)}` : ""}`;
      const blocksData = await notionRequest(blocksPath);
      const blocks = blocksData.results || [];
      const entries = blocks
        .filter((b) => b.type === "paragraph")
        .map((b) => notionRichTextToString(b.paragraph?.rich_text || []))
        .filter(isChangelogEntryText);
      const hasMore = blocksData.has_more
        ? `\n\n⚠️ More blocks exist beyond this page — call again with cursor: "${blocksData.next_cursor}" to scan further (older changelog entries, if any, may be further in).`
        : "";
      if (!entries.length) {
        return { content: [{ type: "text", text: `No changelog entries found on this page (within the blocks scanned).${hasMore}` }] };
      }
      return { content: [{ type: "text", text: entries.join("\n") + hasMore }] };
    }
  );

  server.tool(
    "notion_create_page",
    "Create a new Notion page inside a parent page or database. Pass entity_id to get upsert-style dedup protection (mirrors mem0_add): if a page already carries that entity_id marker, this refuses to create a duplicate and returns the existing page instead. Recommended whenever this page represents a stable, ongoing thing (a tracked PR, an issue, a recurring report) rather than a genuine one-off. When parent_type is 'database', pass `properties` to set real database column values (select/rich_text/url/etc) -- see notion_get_database first for the schema.",
    {
      parent_id:   z.string().describe("ID of the parent page or database"),
      parent_type: z.enum(["page", "database"]).describe("Whether the parent is a page or a database"),
      title:       z.string().describe("Title of the new page"),
      content:     z.string().optional().describe("Plain text content to add as paragraph blocks"),
      entity_id:   z.string().optional().describe("Optional stable identifier for the thing this page represents (e.g. 'pr-workers-sdk-14714'). BEFORE inventing a new one, use notion_search for an existing page on the same topic -- entity_id dedup only catches an EXACT marker match. If a page already exists with this entity_id, notion_create_page will NOT create a duplicate -- it returns the existing page's id/url/content instead, so you can call notion_update_page (append_content or replacements) on it instead of creating a new one. Stored as a visible '🔑 entity_id: ...' marker paragraph at the top of the page, since Notion pages outside a database have no real custom-property field to use instead."),
      status:      z.enum(STATUS_VALUES).optional().describe("Optional lifecycle status (open/resolved/superseded) for this page. Stored as a visible '🏷️ status: ...' marker paragraph, same convention as entity_id."),
      relations:   z.array(z.object({
        to_entity_id: z.string().describe("The entity_id of the other tracked page this one relates to"),
        relation:     z.string().describe("The relation type, e.g. 'blocks', 'depends_on', 'relates_to' -- free text"),
      })).optional().describe("Optional list of outgoing relations from this page's entity to others, e.g. [{to_entity_id:'bug-4', relation:'blocks'}]. Stored as visible '🔗 relation -> to_entity_id' marker paragraphs. Only outgoing relations are supported -- see notion_get_page's Relations section for resolved targets."),
      one_off:     z.boolean().optional().describe("Set true to explicitly opt this page OUT of entity_id tracking -- required if entity_id is omitted. This tool refuses to create a page without either entity_id or one_off: true, so omitting entity_id by accident (rather than on purpose) is caught immediately instead of silently producing an untracked, un-deduped page. Use for genuine one-offs: scratch notes, test pages, throwaway content that will never need dedup or update-in-place."),
      properties:  z.record(z.any()).optional().describe("Optional Notion database property VALUES to set when parent_type is 'database' (ignored for parent_type 'page', which has no custom properties). Keys are property names exactly as they appear in the database schema; values must be in Notion's property-value format, e.g. { \"Status\": { \"select\": { \"name\": \"open\" } }, \"Apply Link\": { \"url\": \"https://...\" }, \"Comp / Rate\": { \"rich_text\": [{ \"text\": { \"content\": \"$10-60/hr\" } }] } }. Call notion_get_database first to see available property names and types."),
    },
    async ({ parent_id, parent_type, title, content, entity_id, status, relations, one_off, properties }) => {
      let result;
      try {
        result = await doCreatePage({ parent_id, parent_type, title, content, entity_id, status, relations, one_off, properties });
      } catch (err) {
        return { content: [{ type: "text", text: err.message }], isError: true };
      }
      if (result.skipped) {
        return {
          content: [{
            type: "text",
            text:
              `Not creating — a page already exists for entity_id "${entity_id}" (id: ${result.existingId}, title: "${result.existingTitle}"). No duplicate was created.\n` +
              `URL: ${result.existingUrl}\n\n` +
              `Next step: call notion_get_page on this id to review current content, then notion_update_page (append_content or replacements) to update it instead of creating a new page.`,
          }],
        };
      }
      const indexNote = result.indexError
        ? `\n\n\u26a0\ufe0f Page created, but recording it in the dedup index failed: ${result.indexError}. Future notion_create_page calls with entity_id "${entity_id}" may not detect this page as a duplicate.`
        : "";
      const markerNote = result.markerCount ? ` (with ${entity_id ? "entity_id" : ""}${entity_id && status ? " + " : ""}${status ? "status" : ""} marker${result.markerCount > 1 ? "s" : ""})` : "";
      const autoLinkNote = result.autoRelations?.length
        ? `\n\n\ud83d\udd17 Auto-linked (identifier/cross-reference match): ${result.autoRelations.map((r) => r.to_entity_id).join(", ")}`
        : "";
      const unlinkableStrong = (result.linkCandidates?.strong || []).filter((c) => !c.entity_id);
      const unlinkedNote = unlinkableStrong.length
        ? `\n\n\ud83d\udd0e Strong match found but not auto-linked (candidate has no entity_id to attach a relation to): ${unlinkableStrong.map((c) => `"${c.title}" (${c.url}) -- ${c.reason}`).join("; ")}`
        : "";
      const candidateNote = result.linkCandidates?.medium?.length
        ? `\n\n\ud83e\udd14 Possible related page(s) (tag overlap, not auto-linked): ${result.linkCandidates.medium.map((c) => `"${c.title}" (${c.url})`).join("; ")}`
        : "";
      return { content: [{ type: "text", text: `Created Notion page "${title}"${markerNote}\nID: ${result.id}\nURL: ${result.url}${indexNote}${autoLinkNote}${unlinkedNote}${candidateNote}` }] };
    }
  );

  server.tool(
    "notion_create_pages_batch",
    "Create multiple Notion pages in a single call, to reduce round trips. Each item is created independently -- entity_id dedup, marker blocks, dedup-index recording, and database `properties` all apply per item exactly as in notion_create_page. One item failing (e.g. bad parent_id) does not block the others.",
    {
      items: z.array(z.object({
        parent_id:   z.string().describe("ID of the parent page or database"),
        parent_type: z.enum(["page", "database"]).describe("Whether the parent is a page or a database"),
        title:       z.string().describe("Title of the new page"),
        content:     z.string().optional().describe("Plain text content to add as paragraph blocks"),
        entity_id:   z.string().optional().describe("Optional stable identifier for this page -- see notion_create_page. If a page already exists with this entity_id, this item is skipped (not duplicated) and the existing id/url is reported instead."),
        status:      z.enum(STATUS_VALUES).optional().describe("Optional lifecycle status (open/resolved/superseded) -- see notion_create_page."),
        relations:   z.array(z.object({
          to_entity_id: z.string().describe("The entity_id of the other tracked page this one relates to"),
          relation:     z.string().describe("The relation type -- see notion_create_page"),
        })).optional().describe("Optional outgoing relations for this page -- see notion_create_page."),
        one_off:     z.boolean().optional().describe("Required if entity_id is omitted -- see notion_create_page."),
        properties:  z.record(z.any()).optional().describe("Optional database property values for this page -- see notion_create_page."),
      })).min(1).describe("List of pages to create"),
    },
    async ({ items }) => {
      const results = await runSequentially(items, doCreatePage);
      const lines = results.map((r, i) => {
        const label = items[i].title;
        if (r.status === "rejected") return `\u2717 [${i}] "${label}" — error: ${r.reason?.message || r.reason}`;
        const v = r.value;
        if (v.skipped) return `\u23ed [${i}] "${label}" — skipped, entity_id "${v.entity_id}" already exists (id: ${v.existingId}, title: "${v.existingTitle}").`;
        const idxNote = v.indexError ? ` \u26a0\ufe0f index record failed: ${v.indexError}` : "";
        return `\u2713 [${i}] "${label}" — id: ${v.id}${idxNote}`;
      });
      const created = results.filter((r) => r.status === "fulfilled" && !r.value.skipped).length;
      return { content: [{ type: "text", text: `${created}/${items.length} created.\n\n${lines.join("\n")}` }] };
    }
  );

  server.tool(
    "notion_create_database",
    "Create a new Notion database inside a parent page, with a given property schema. One-off/setup tool -- most workflows should use notion_create_page instead.",
    {
      parent_page_id: z.string().describe("ID of the parent page to create the database under"),
      title:          z.string().describe("Title of the new database"),
      properties:     z.record(z.any()).describe("Notion property schema object, e.g. { \"Name\": { \"title\": {} }, \"Status\": { \"select\": { \"options\": [{ \"name\": \"open\" }] } } }"),
    },
    async ({ parent_page_id, title, properties }) => {
      const data = await notionRequest("/databases", {
        method: "POST",
        body: {
          parent: { type: "page_id", page_id: parent_page_id },
          title: [{ type: "text", text: { content: title } }],
          properties,
        },
      });
      return { content: [{ type: "text", text: `Created database "${title}"\nID: ${data.id}\nURL: ${data.url}` }] };
    }
  );

  server.tool(
    "notion_get_database",
    "Get a Notion database's schema (title and property definitions) and basic info. Use this before notion_query_database or before notion_create_page with parent_type: 'database', to see what properties are available and their types.",
    {
      database_id: z.string().describe("Notion database ID (UUID format, e.g. from notion_search with filter_type: 'database')"),
    },
    async ({ database_id }) => {
      const data = await notionRequest(`/databases/${database_id}`);
      const title = notionDatabaseTitle(data);
      const propLines = Object.entries(data.properties || {}).map(([name, def]) => `  ${name}: ${def.type}`);
      const text = `# ${title}\nID: ${data.id}\nURL: ${data.url}\nCreated: ${data.created_time?.slice(0, 10)} | Last edited: ${data.last_edited_time?.slice(0, 10)}\n\nProperties:\n${propLines.join("\n") || "(none)"}`;
      return { content: [{ type: "text", text }] };
    }
  );

  server.tool(
    "notion_query_database",
    "Query rows from a Notion database, with an optional filter. Returns each row's properties in readable form. Call notion_get_database first to see available property names/types for building a filter.",
    {
      database_id: z.string().describe("Notion database ID"),
      filter:      z.record(z.any()).optional().describe("Optional Notion filter object, e.g. { property: 'EntityId', rich_text: { equals: 'some-id' } }"),
      page_size:   z.number().optional().describe("Number of rows to return (default 20, max 100)"),
      cursor:      z.string().optional().describe("Pagination cursor from a previous call's next_cursor, to fetch the next page of rows"),
    },
    async ({ database_id, filter, page_size = 20, cursor }) => {
      const body = { page_size };
      if (filter) body.filter = filter;
      if (cursor) body.start_cursor = cursor;
      const data = await notionRequest(`/databases/${database_id}/query`, { method: "POST", body });
      if (!data.results?.length) return { content: [{ type: "text", text: "No rows found." }] };
      const displayProp = (val) => {
        if (val.type === "title") return notionRichTextToString(val.title);
        if (val.type === "rich_text") return notionRichTextToString(val.rich_text);
        if (val.type === "url") return val.url || "";
        if (val.type === "select") return val.select?.name || "";
        if (val.type === "multi_select") return (val.multi_select || []).map((s) => s.name).join(",");
        if (val.type === "checkbox") return val.checkbox ? "true" : "false";
        if (val.type === "number") return String(val.number ?? "");
        return JSON.stringify(val[val.type] ?? "");
      };
      const lines = data.results.map((row) => {
        const props = Object.entries(row.properties || {}).map(([name, val]) => `${name}: ${displayProp(val)}`).join(" | ");
        return `- ${props}\n  (row id: ${row.id})`;
      });
      const hasMore = data.has_more ? `\n\n\u26a0\ufe0f More rows exist -- call again with cursor: "${data.next_cursor}" to see the next page.` : "";
      return { content: [{ type: "text", text: lines.join("\n") + hasMore }] };
    }
  );

  server.tool(
    "notion_update_database",
    "Update a Notion database's title, or archive/restore it. Use this instead of notion_update_page for database IDs -- databases live at a separate API endpoint from pages, so notion_update_page returns a 404 if given a database ID.",
    {
      database_id: z.string().describe("Notion database ID (UUID format, e.g. from notion_search with filter_type: 'database')"),
      title:       z.string().optional().describe("New title for the database"),
      archived:    z.boolean().optional().describe("Set true to archive (trash) the database, false to restore"),
    },
    async ({ database_id, title, archived }) => {
      const body = {};
      if (archived !== undefined) body.archived = archived;
      if (title    !== undefined) body.title    = [{ type: "text", text: { content: title } }];
      if (Object.keys(body).length === 0) {
        return { content: [{ type: "text", text: "No changes made." }] };
      }
      const data = await notionRequest(`/databases/${database_id}`, { method: "PATCH", body });
      return { content: [{ type: "text", text: `Updated database "${notionDatabaseTitle(data)}" (ID: ${data.id}).` }] };
    }
  );

  server.tool(
    "notion_update_page",
    "Update a Notion page's title, append text content to it, make a targeted in-place edit to an existing block (replacements), change its lifecycle status marker, or set real database column values via `properties` (select/rich_text/url/etc, if this page is a row in a database).",
    {
      page_id:        z.string().describe("Notion page ID to update"),
      title:          z.string().optional().describe("New title for the page"),
      append_content: z.string().optional().describe("Plain text to append as new paragraph blocks"),
      archived:       z.boolean().optional().describe("Set true to archive (trash) the page, false to restore"),
      replacements:   z.array(z.object({
        find:    z.string().describe("Exact plain text of an existing top-level block (paragraph, heading, list item, or to-do) -- must match exactly one block"),
        replace: z.string().describe("New plain text for that block"),
      })).optional().describe("List of find-and-replace operations for targeted in-place block edits, instead of appending new content. Each `find` must match exactly one of the page's top-level blocks (first 100) by plain text -- fails loudly (no changes made) on zero or multiple matches, same uniqueness rule as mem0_update's replacements and the github edit_file tool's `replacements` mode. Only text-style blocks can be edited this way (paragraph/heading/list-item/to-do); code blocks, subpages, etc. are not supported and will report an error instead of being silently skipped."),
      status:         z.enum(STATUS_VALUES).optional().describe("Set this page's lifecycle status (open/resolved/superseded). Updates the existing '🏷️ status: ...' marker block in place if one exists, or appends a new marker block if the page has none yet."),
      entity_id:      z.string().optional().describe("Correct or set this page's entity_id marker. Use this (not `replacements` on the marker text) whenever an entity_id was wrong or missing -- this updates the visible '🔑 entity_id: ...' marker block in place AND upserts the Entity Index database entry that notion_create_page's dedup check and notion_get_page's relation resolution both read from. Editing the marker text directly via `replacements` only changes what's visible on the page; it does NOT update the index, so other pages' relations pointing at the corrected entity_id will keep resolving as 'dangling' until this param is used instead."),
      relations:      z.array(z.object({
        to_entity_id: z.string().describe("The entity_id of the other entity this one relates to"),
        relation:     z.string().describe("The relation type, e.g. 'blocks', 'depends_on', 'relates_to' -- free text"),
      })).optional().describe("New outgoing relations for this page -- REPLACES the existing relation set whole (not merged). Omit to leave relations unchanged. Pass an empty array to clear all relations."),
      properties:     z.record(z.any()).optional().describe("Database property VALUES to set/update on this page (only meaningful if the page is a row in a database). Keys are property names exactly as they appear in the database schema; values must be in Notion's property-value format, e.g. { \"Status\": { \"select\": { \"name\": \"resolved\" } } }. Merged with any title change into a single PATCH. Call notion_get_database first to see available property names and types."),
    },
    async ({ page_id, title, append_content, archived, replacements, status, entity_id, relations, properties }) => {
      try {
        const results = await doUpdatePage({ page_id, title, append_content, archived, replacements, status, entity_id, relations, properties });
        return { content: [{ type: "text", text: results.join("\n") || "No changes made." }] };
      } catch (err) {
        return { content: [{ type: "text", text: err.message }], isError: true };
      }
    }
  );

  server.tool(
    "notion_sync_content",
    "Write content into a marked, machine-managed range on a Notion page, without disturbing anything a person has added elsewhere on the page. On first use, appends a new range (start marker + content + end marker) to the end of the page. On later calls with the same synced_at, does nothing (already up to date). On later calls with a different synced_at, replaces only the blocks between the markers -- content above the start marker or below the end marker is never read or touched. This is the low-level primitive behind mem0->Notion sync; call directly for testing, or to sync arbitrary external content into a page.",
    {
      page_id:     z.string().describe("Notion page ID to write the synced range onto"),
      content:     z.string().describe("Plain text content for the synced range, one paragraph block per newline-separated line"),
      synced_at:   z.string().describe("Version/timestamp identifying this content revision (e.g. an ISO timestamp or a source system's updated_at). If this matches what's already on the page, the call is a no-op."),
    },
    async ({ page_id, content, synced_at }) => {
      const contentLines = content.split("\n");
      let result;
      try {
        result = await replaceSyncedRange({ page_id, contentLines, synced_at });
      } catch (err) {
        return { content: [{ type: "text", text: err.message }], isError: true };
      }
      if (result.action === "created") return { content: [{ type: "text", text: `Created new synced range (${result.blockCount} blocks) on page ${page_id}.` }] };
      if (result.action === "skipped") return { content: [{ type: "text", text: `No changes made — ${result.reason}.` }] };
      return { content: [{ type: "text", text: `Synced range updated on page ${page_id}: ${result.removed} block(s) removed, ${result.added} added (was mem0_synced_at: ${result.previousSyncedAt}, now: ${synced_at}). Content above/below the markers was left untouched.` }] };
    }
  );

  server.tool(
    "notion_index_entries_add_batch",
    "Backfill/repair tool: add entries to the Entity Index database for pages that already exist and already carry an entity_id marker, but aren't yet recorded in the index (e.g. after the index was reset, or a write silently failed earlier). Skips (no duplicate row) any entity_id already indexed. NOT for normal page creation -- notion_create_page/notion_create_pages_batch already index automatically when you pass entity_id there; use this only to backfill pre-existing pages.",
    {
      items: z.array(z.object({
        entity_id: z.string().describe("The entity_id marker already present on the target page"),
        page_id:   z.string().describe("Notion page ID of the existing page this entity_id refers to"),
        url:       z.string().optional().describe("Notion URL of the page"),
        tags:      z.array(z.string()).optional().describe("Tags for this entity, if any (lowercase, no # prefix)"),
      })).min(1).describe("List of index entries to backfill"),
    },
    async ({ items }) => {
      const results = await runSequentially(items, upsertIndexEntry);
      const lines = results.map((r, i) => {
        const label = items[i].entity_id;
        if (r.status === "rejected") return `\u2717 [${i}] ${label} \u2014 ${r.reason?.message || r.reason}`;
        if (r.value.skipped) return `\u23ed [${i}] ${label} \u2014 already indexed (page ${r.value.existingId})`;
        if (r.value.error) return `\u26a0\ufe0f [${i}] ${label} \u2014 write failed: ${r.value.error}`;
        return `\u2713 [${i}] ${label} \u2014 indexed`;
      });
      const added = results.filter((r) => r.status === "fulfilled" && !r.value.skipped && !r.value.error).length;
      return { content: [{ type: "text", text: `${added}/${items.length} added.\n\n${lines.join("\n")}` }] };
    }
  );

  server.tool(
    "notion_update_pages_batch",
    "Update multiple Notion pages in a single call, to reduce round trips. Each item supports the same title/append_content/archived/replacements/status/properties behavior as notion_update_page. One item failing (e.g. an ambiguous replacement match) does not block the others.",
    {
      items: z.array(z.object({
        page_id:        z.string().describe("Notion page ID to update"),
        title:          z.string().optional().describe("New title for the page"),
        append_content: z.string().optional().describe("Plain text to append as new paragraph blocks"),
        archived:       z.boolean().optional().describe("Set true to archive (trash) the page, false to restore"),
        replacements:   z.array(z.object({
          find:    z.string().describe("Exact plain text of an existing top-level block -- must match exactly one block"),
          replace: z.string().describe("New plain text for that block"),
        })).optional().describe("Targeted find/replace edits for this page -- see notion_update_page for matching rules."),
        status:         z.enum(STATUS_VALUES).optional().describe("Set this page's lifecycle status -- see notion_update_page."),
        entity_id:      z.string().optional().describe("Correct or set this page's entity_id marker, reindexing it in the Entity Index database -- see notion_update_page."),
        relations:      z.array(z.object({
          to_entity_id: z.string().describe("The entity_id of the other entity this one relates to"),
          relation:     z.string().describe("The relation type -- see notion_update_page"),
        })).optional().describe("New outgoing relations for this page -- see notion_update_page (whole-set replace)."),
        properties:     z.record(z.any()).optional().describe("Database property values to set/update on this page -- see notion_update_page."),
      })).min(1).describe("List of page updates to apply"),
    },
    async ({ items }) => {
      const results = await runSequentially(items, doUpdatePage);
      const lines = results.map((r, i) => {
        const label = items[i].page_id;
        if (r.status === "rejected") return `\u2717 [${i}] ${label} — ${r.reason?.message || r.reason}`;
        return `\u2713 [${i}] ${label} — ${r.value.join("; ") || "no changes made"}`;
      });
      const succeeded = results.filter((r) => r.status === "fulfilled").length;
      return { content: [{ type: "text", text: `${succeeded}/${items.length} updated.\n\n${lines.join("\n")}` }] };
    }
  );
}
