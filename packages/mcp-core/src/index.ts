// mcp-handlers.ts — pure handler functions for the chart_review_state
// MCP server, decoupled from any specific MCP transport SDK.
//
// `mcp-tools.ts` adapts these to Anthropic's in-process
// @anthropic-ai/claude-agent-sdk via `tool()` + `createSdkMcpServer()`.
// `mcp-server/index.ts` adapts them to a stdio MCP server via
// @modelcontextprotocol/sdk for non-Anthropic providers (Codex, etc.).
//
// The handler signatures are stable across both adapters: each takes
// an explicit `McpSession` (patientId + task + sessionId) plus typed
// args, and returns a `CallToolResult` shaped to match the MCP wire
// protocol. Adapters only translate transport-specific message shapes.

import { z } from "zod";
import { parse as parseYaml } from "yaml";
import { findQuoteOffsetsImpl } from "@chart-review/find-quote-offsets";
import type { CompiledTask } from "@chart-review/tasks";
import {
  listNotes as listNotesFn,
  readNote as readNoteFn,
  readStructured as readStructuredFn,
} from "@chart-review/patients";
import { resolveRubricRoot } from "@chart-review/rubric";
import fs from "node:fs";
import path from "node:path";
import {
  applyUiAction,
  load as loadReviewState,
  loadOrCreate,
  type ReviewState,
  type UiAction,
} from "@chart-review/domain-review";
import { appendAuditEntry } from "@chart-review/audit-trail";
import { fieldApplicability, gateReferencedIds } from "@chart-review/contract-eval";

// ── public types ─────────────────────────────────────────────────────

/** CallToolResult shape mirrored from the MCP spec. The Anthropic SDK
 *  doesn't re-export this; we define it here so both adapters can
 *  return the same value. */
export type CallToolResult = {
  isError?: boolean;
  content: Array<{ type: "text"; text: string }>;
};

/** Per-(patient, task) session context. Adapters construct one of
 *  these from their transport-specific session info — env vars for
 *  the subprocess server, closure-captured args for the in-process
 *  Anthropic adapter. */
export interface McpSession {
  patientId: string;
  task: CompiledTask;
  sessionId: string;
  /** Per-session set of field_ids the agent has read via `read_criterion`
   *  (singular only — `read_criteria` is a compact overview that does not
   *  carry the body or the prefetched evidence, so it does not satisfy this).
   *  Consulted by the `set_field_assessment` gate for criteria that declare
   *  `uses.keyword_sets`. Lazily initialized on first read. */
  readCriterionIds?: Set<string>;
  /** Per-session cache of the real note filenames for this patient, from
   *  `listNotes`. Lazily populated the first time `read_note` / `read_notes`
   *  validates a filename. Used to catch the Nov-27 Qwen hallucination mode
   *  where the agent extrapolated sequential filenames (e.g.
   *  `043_2015-12-31_Emergency_Dept_MD_Progress_Note.txt`) that did not exist,
   *  burned the recursion budget retrying, and never started the rubric work. */
  knownNoteFilenames?: Set<string>;
}

/** Optional hook fired after every successful state mutation. The
 *  in-process Anthropic adapter wires this to the WebSocket
 *  broadcaster; the subprocess server passes a no-op (the main
 *  server discovers state changes by polling disk). */
export interface ReviewToolHooks {
  onStateUpdate(state: ReviewState): void;
}

const noopHooks: ReviewToolHooks = { onStateUpdate: () => {} };

// ── shared zod schema (re-exported for adapters) ─────────────────────

/** Flat evidence schema. Avoids z.union (Together-routed DeepSeek and
 *  some other providers reject `anyOf` at the parameter level). All
 *  fields optional; runtime discriminator (`source`) determines which
 *  ones are required. ensureEvidenceShape() does the runtime check. */
export const evidenceSchema = z.object({
  source: z.enum(["note", "omop", "structured"]),
  // Evidence is secondary metadata (a pointer to a note span or a structured
  // row, for display + the faithfulness check). It must NEVER reject on a benign
  // type mismatch, because a rejected set_field_assessment triggers a retry storm
  // that hangs the deepagents/langchain stack. So every field is BOTH:
  //   • `.nullish()` — agents serialize an absent field as explicit `null`
  //     (`.optional()` alone rejects null → the `unit:null` empty-run bug), and
  //   • `z.coerce.*` — agents send a numeric OMOP `row_id`/id where a string is
  //     declared (→ the `row_id` -32602 hang). Coercion normalizes 12345→"12345".
  // Source-specific requiredness is still enforced by ensureEvidenceShape below.
  note_id: z.coerce.string().nullish(),
  span_offsets: z
    .array(z.coerce.number().int().nonnegative())
    .length(2)
    .nullish(),
  verbatim_quote: z.coerce.string().nullish(),
  doc_type: z.coerce.string().nullish(),
  author_role: z.coerce.string().nullish(),
  // omop / structured fields
  table: z.coerce.string().nullish(),
  row_id: z.coerce.string().nullish(),
  concept_id: z.coerce.number().int().nullish(),
  concept_name: z.coerce.string().nullish(),
  value: z.unknown().nullish(),
  unit: z.coerce.string().nullish(),
  // common
  evidence_date: z.coerce.string().nullish(),
});

// ── shared helpers ───────────────────────────────────────────────────

/** Runtime discriminator. Throws if shape is wrong. */
export function ensureEvidenceShape(ev: any): asserts ev is {
  source: "note" | "omop" | "structured";
  [k: string]: unknown;
} {
  if (!ev || typeof ev !== "object" || !ev.source) {
    throw new Error("evidence missing");
  }
  if (ev.source === "note") {
    if (
      typeof ev.note_id !== "string" ||
      !Array.isArray(ev.span_offsets) ||
      ev.span_offsets.length !== 2 ||
      typeof ev.verbatim_quote !== "string" ||
      ev.verbatim_quote.length === 0
    ) {
      throw new Error(
        "note evidence requires note_id, span_offsets [start,end], and verbatim_quote",
      );
    }
  } else if (ev.source === "omop" || ev.source === "structured") {
    if (typeof ev.table !== "string" || ev.row_id === undefined) {
      throw new Error("omop/structured evidence requires table and row_id");
    }
  }
}

/** Validate every evidence row's source-specific shape and return
 *  them unchanged. Used by handlers that take an evidence array (or
 *  a single evidence) before constructing a UiAction. */
export function validatedEvidence<T extends Array<any> | undefined>(arr: T): T {
  if (!arr) return arr;
  for (const ev of arr as any[]) ensureEvidenceShape(ev);
  return arr;
}

/** Funnel every chat-agent action through one applyUiAction call.
 *  Returns a CallToolResult; emits a ui_action audit entry. Errors
 *  are translated to `{ok:false, ...}` payloads so the model can
 *  read them and try again. */
async function runAction(
  session: McpSession,
  hooks: ReviewToolHooks,
  action: UiAction,
  payloadSummary: () => string,
): Promise<CallToolResult> {
  const { patientId, task, sessionId } = session;
  try {
    const result = applyUiAction(
      patientId,
      task,
      "agent",
      `agent_${sessionId}`,
      action,
    );
    appendAuditEntry(
      { patientId, taskId: task.task_id, sessionId },
      {
        ts: new Date().toISOString(),
        session_id: sessionId,
        step_type: "ui_action",
        action_type: action.type,
        source: "agent",
        payload_summary: payloadSummary(),
        result_version: result.state.version,
        added_evidence_id: result.added_evidence_id,
        ...(action.type === "set_field_assessment" && {
          payload_field_id: (action.payload as { field_id?: string }).field_id,
          payload_answer: (action.payload as { answer?: unknown }).answer,
        }),
      },
    );
    hooks.onStateUpdate(result.state);
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            ok: true,
            action_type: action.type,
            version: result.state.version,
            warnings: result.warnings,
            added_evidence_id: result.added_evidence_id,
          }),
        },
      ],
    };
  } catch (e) {
    const code = (e as { code?: string }).code ?? "error";
    // Return the rejection as a NORMAL (non-isError) tool result. With
    // isError:true, langchain-mcp-adapters raises a ToolException that
    // deepagents' middleware re-raises — crashing the whole run instead of
    // letting the model react. As a recoverable result the model reads
    // {ok:false,...} and can retry (the documented intent of this funnel).
    const hint =
      code === "faithfulness_failed"
        ? "The verbatim_quote was NOT found in this note. If this fact came from a structured OMOP row (you read it via read_structured_data), re-submit the write with evidence {source:'omop', table, row_id} and NO note quote/note_id — a structured concept_name (diagnosis/drug/lab name) is NOT note text. Only if the wording genuinely appears in the note, call find_quote_offsets for the exact offsets and re-submit."
        : undefined;
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            ok: false,
            action_type: action.type,
            error_code: code,
            message: (e as Error).message,
            ...(hint ? { hint } : {}),
          }),
        },
      ],
    };
  }
}

/** Commit gate: before transitioning to `agent_complete`, verify
 *  every applicable, non-derived criterion has an assessment.
 *  Returns null on success, or `{ missing_criteria, unanswered_gate_deps? }`. */
export function checkCommitGate(
  task: CompiledTask,
  state: ReviewState,
): { missing_criteria: string[]; unanswered_gate_deps?: string[] } | null {
  const answers: Record<string, unknown> = {};
  for (const fa of state.field_assessments) {
    answers[fa.field_id] = fa.answer;
  }
  const missing: string[] = [];
  const unansweredGateDeps = new Set<string>();
  for (const field of task.fields) {
    if (field.derivation) continue;
    if (field.is_applicable_when) {
      const applicability = fieldApplicability(task, answers, field.id);
      if (applicability === "not_applicable") continue;
      if (applicability === "unknown") {
        const deps = gateReferencedIds(task, field.id);
        for (const dep of deps) {
          if (answers[dep] === undefined) unansweredGateDeps.add(dep);
        }
      }
    }
    const hasAssessment = state.field_assessments.some(
      (fa) => fa.field_id === field.id,
    );
    if (!hasAssessment) missing.push(field.id);
  }
  if (missing.length === 0) return null;
  const result: { missing_criteria: string[]; unanswered_gate_deps?: string[] } =
    { missing_criteria: missing };
  if (unansweredGateDeps.size > 0) {
    result.unanswered_gate_deps = [...unansweredGateDeps];
  }
  return result;
}

// ── the seven pure handler functions ─────────────────────────────────

/** Initialize per-session state. Should be called once before the
 *  first handler invocation; `loadOrCreate` ensures review_state.json
 *  exists so handlers don't race on first write. */
export function initSession(session: McpSession): void {
  loadOrCreate(session.patientId, session.task);
}

export interface SetFieldAssessmentArgs {
  field_id: string;
  answer?: unknown;
  confidence?: "low" | "medium" | "high";
  evidence?: unknown[];
  rationale?: string;
  edit_reason?:
    | "missed_evidence"
    | "misinterpreted"
    | "wrong_rule"
    | "criterion_ambiguous"
    | "other";
  edit_note?: string;
  override_of_agent?: boolean;
}

/**
 * Some tool-calling models (e.g. Qwen3) occasionally wrap the scalar answer in
 * a single-key object — `{ "<field_id>": "sarcoma" }` or `{ "answer": "..." }`
 * — instead of passing the bare value. That nested shape renders as
 * "[object Object]" in the UI and can't be scored. Unwrap a single-key
 * `{field_id|answer|value: scalar}` to the scalar; leave anything else as-is.
 */
function unwrapNestedAnswer(fieldId: string, answer: unknown, enumVals?: unknown[]): unknown {
  let a = answer;
  if (a && typeof a === "object" && !Array.isArray(a)) {
    const obj = a as Record<string, unknown>;
    const keys = Object.keys(obj);
    if (keys.length === 1 && (keys[0] === fieldId || keys[0] === "answer" || keys[0] === "value")) {
      a = obj[keys[0]];
    } else if ("value" in obj) {
      // typed-answer wrapper, e.g. { type: "boolean", value: false }
      a = obj.value;
    }
  }
  // Some models emit a JS boolean for a yes/no-style field. Map it to the
  // field's matching enum value, honoring whichever convention the rubric uses
  // (yes/no OR true/false); fall back to yes/no when there's no enum to match.
  if (typeof a === "boolean") {
    const wanted = a ? ["yes", "true"] : ["no", "false"];
    if (Array.isArray(enumVals)) {
      const match = enumVals.find((v) => wanted.includes(String(v).toLowerCase()));
      if (match !== undefined) return match;
    }
    return a ? "yes" : "no";
  }
  return a;
}

export async function setFieldAssessment(
  session: McpSession,
  args: SetFieldAssessmentArgs,
  hooks: ReviewToolHooks = noopHooks,
): Promise<CallToolResult> {
  const { override_of_agent, ...payload } = args;
  // The field's allowed answers (enum), used to normalize a raw boolean answer
  // to the right enum value for THIS field — generic across tasks/conventions.
  const field = (
    session.task.fields as
      | Array<{ field_id?: string; id?: string; answer_schema?: { enum?: unknown[] }; derivation?: string }>
      | undefined
  )?.find((f) => (f.field_id ?? f.id) === payload.field_id);
  // Derived fields are COMPUTED by the platform from their component leaves. An
  // agent must never set one directly — recompute would discard the value, and
  // it signals the agent is scoring instead of extracting. Reject with guidance
  // toward the components (defense-in-depth; list_criteria already hides these).
  if (field?.derivation) {
    // Return a NORMAL (non-isError) result — same convention as runAction's
    // catch below. `isError:true` makes langchain-mcp-adapters raise a
    // ToolException that deepagents re-raises, breaking the run; a plain
    // {ok:false} result lets the model read the message and commit the
    // components instead.
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            ok: false,
            action_type: "set_field_assessment",
            error_code: "derived_field_not_settable",
            message: `"${payload.field_id}" is a DERIVED field — the platform computes it from its component leaves, so you cannot set it directly. Do NOT score it. Commit the component fields it depends on instead (read its criterion, or the matching references/scoring/item-N file, for the exact components).`,
          }),
        },
      ],
    };
  }
  // ── read-criterion gate ──────────────────────────────────────────
  // For criteria that declare `uses.keyword_sets` (opt-in per criterion),
  // require the agent to have called `read_criterion(field_id)` at least
  // once in this session before accepting a `set_field_assessment` on that
  // field. The pre-retrieved evidence hits the agent needs to answer this
  // criterion ride on the `read_criterion` response — refusing without it
  // caught Sneha's Nov-24 pilot where the agent made 46 `set_field_assessment`
  // calls on a patient without ever reading the criterion bodies.
  // Backward-compatible: criteria without `uses.keyword_sets` are unaffected,
  // so other phenotypes (rucam, cp-depression, asthma, lung, malo, lcn,
  // cancer-diagnosis) are not gated. Same non-isError convention as above.
  if (criterionDeclaresKeywordSets(session, payload.field_id)) {
    if (!session.readCriterionIds?.has(payload.field_id)) {
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify({
              ok: false,
              action_type: "set_field_assessment",
              error_code: "read_criterion_required",
              message: `Call read_criterion("${payload.field_id}") before set_field_assessment — the response carries pre-retrieved evidence snippets you must consult before answering this criterion. read_criteria (batch) is a compact overview only; it does not satisfy this gate.`,
            }),
          },
        ],
      };
    }
  }
  const enumVals = field?.answer_schema?.enum;
  return runAction(
    session,
    hooks,
    {
      type: "set_field_assessment",
      payload: {
        ...payload,
        answer: unwrapNestedAnswer(payload.field_id, (payload as { answer?: unknown }).answer, enumVals),
        evidence: validatedEvidence(payload.evidence as any[] | undefined),
      } as any,
    },
    () =>
      `field_id=${args.field_id}${override_of_agent ? " override_of_agent=true" : ""}`,
  );
}

export async function getReviewState(
  session: McpSession,
): Promise<CallToolResult> {
  const state = loadReviewState(session.patientId, session.task.task_id);
  return {
    content: [
      {
        type: "text",
        text: JSON.stringify(state ?? { field_assessments: [] }),
      },
    ],
  };
}

export interface SetSummaryArgs {
  brief_summary?: string;
  key_conditions?: string[];
  uncertainties?: string[];
  evidence_files?: string[];
}

export async function setSummary(
  session: McpSession,
  args: SetSummaryArgs,
  hooks: ReviewToolHooks = noopHooks,
): Promise<CallToolResult> {
  return runAction(
    session,
    hooks,
    { type: "set_summary", payload: args },
    () => `keys=${Object.keys(args).join(",")}`,
  );
}

export interface RecommendKeywordsArgs {
  topic: string;
  direct_terms?: string[];
  aliases?: string[];
  abbreviations?: string[];
  behavioral_clues?: string[];
  treatment_terms?: string[];
  negation_patterns?: string[];
}

export async function recommendKeywords(
  session: McpSession,
  args: RecommendKeywordsArgs,
  hooks: ReviewToolHooks = noopHooks,
): Promise<CallToolResult> {
  return runAction(
    session,
    hooks,
    { type: "recommend_keywords", payload: args },
    () => `topic=${args.topic}`,
  );
}

export interface SelectEvidenceArgs {
  evidence: unknown;
  rationale?: string;
  category?: "supporting" | "contradicting" | "context";
  field_id?: string;
}

export async function selectEvidence(
  session: McpSession,
  args: SelectEvidenceArgs,
  hooks: ReviewToolHooks = noopHooks,
): Promise<CallToolResult> {
  return runAction(
    session,
    hooks,
    {
      type: "select_evidence",
      payload: {
        ...args,
        evidence: validatedEvidence([args.evidence as any])[0],
      } as any,
    },
    () =>
      `category=${args.category ?? "(none)"} note=${(args.evidence as any).note_id ?? (args.evidence as any).table}`,
  );
}

export interface FindQuoteOffsetsArgs {
  note_id: string;
  snippet: string;
}

export async function findQuoteOffsets(
  session: McpSession,
  args: FindQuoteOffsetsArgs,
): Promise<CallToolResult> {
  const r = findQuoteOffsetsImpl(session.patientId, args.note_id, args.snippet);
  return {
    isError: !r.ok,
    content: [{ type: "text", text: JSON.stringify(r) }],
  };
}

export async function setReviewStatus(
  session: McpSession,
  hooks: ReviewToolHooks = noopHooks,
): Promise<CallToolResult> {
  const state = loadReviewState(session.patientId, session.task.task_id);
  if (!state) {
    return {
      isError: true,
      content: [
        {
          type: "text",
          text: JSON.stringify({
            ok: false,
            error_code: "state_not_found",
            message:
              "Review state not found. Call set_field_assessment at least once first.",
          }),
        },
      ],
    };
  }
  const gateResult = checkCommitGate(session.task, state);
  if (gateResult) {
    const depHint =
      gateResult.unanswered_gate_deps &&
      gateResult.unanswered_gate_deps.length > 0
        ? ` Some criteria could not be evaluated because their gate dependencies (${gateResult.unanswered_gate_deps.join(", ")}) have not been answered yet — answer those leaf criteria first.`
        : "";
    return {
      isError: true,
      content: [
        {
          type: "text",
          text: JSON.stringify({
            ok: false,
            error_code: "incomplete_review",
            message: `Cannot mark review complete: ${gateResult.missing_criteria.length} criterion/criteria have no committed value. Commit values for every criterion before calling set_review_status.${depHint}`,
            missing_criteria: gateResult.missing_criteria,
            ...(gateResult.unanswered_gate_deps
              ? { unanswered_gate_deps: gateResult.unanswered_gate_deps }
              : {}),
          }),
        },
      ],
    };
  }
  return runAction(
    session,
    hooks,
    {
      type: "set_review_status",
      payload: { review_status: "agent_complete" },
    },
    () => "status=agent_complete",
  );
}

// ── read-side tools (added so agents stop shelling out for file reads) ──
//
// These are pure pass-throughs to @chart-review/patients helpers. They
// exist because Codex (no first-party Read/Glob tools) was reading
// notes/structured data via `cat`/`sed`/`for f in …` shell commands —
// each invocation echoes the whole file into the conversation history
// at every turn, blowing up token usage 7-10x.
//
// With these tools registered, Codex (and Claude when convenient) can
// fetch file content through structured calls instead. Tool results
// return pre-trimmed JSON; max_chars defaults to 8 KB so a single
// large note doesn't dominate context.

export const listNotesArgsSchema = z.object({});
export type ListNotesArgs = z.infer<typeof listNotesArgsSchema>;

export async function listNotesTool(
  session: McpSession,
  _args: ListNotesArgs,
): Promise<CallToolResult> {
  try {
    const notes = listNotesFn(session.patientId).map((n) => ({
      filename: n.filename,
      date: n.date,
      doctype: n.doctype,
    }));
    return {
      content: [{ type: "text", text: JSON.stringify({ ok: true, count: notes.length, notes }) }],
    };
  } catch (e) {
    return {
      isError: true,
      content: [{ type: "text", text: JSON.stringify({ ok: false, error: (e as Error).message }) }],
    };
  }
}

export const searchNotesArgsSchema = z.object({
  keyword: z.string().min(1),
  max_matches: z.number().int().positive().optional(),
  context_chars: z.number().int().nonnegative().optional(),
});
export type SearchNotesArgs = z.infer<typeof searchNotesArgsSchema>;

/** Case-insensitive substring search across all of the patient's notes.
 *  Returns {filename, offset, snippet} hits so the agent can jump to the
 *  relevant span on a long chart instead of reading every note. */
export async function searchNotesTool(
  session: McpSession,
  args: SearchNotesArgs,
): Promise<CallToolResult> {
  try {
    const kw = args.keyword.toLowerCase();
    const ctx = args.context_chars ?? 80;
    const cap = args.max_matches ?? 20;
    const hits: Array<{ filename: string; offset: number; snippet: string }> = [];
    for (const n of listNotesFn(session.patientId)) {
      if (hits.length >= cap) break;
      let text: string;
      try { text = readNoteFn(session.patientId, n.filename); } catch { continue; }
      const lc = text.toLowerCase();
      let i = lc.indexOf(kw);
      while (i !== -1 && hits.length < cap) {
        const start = Math.max(0, i - ctx);
        const end = Math.min(text.length, i + kw.length + ctx);
        hits.push({ filename: n.filename, offset: i,
                    snippet: (start > 0 ? "…" : "") + text.slice(start, end).replace(/\s+/g, " ") + (end < text.length ? "…" : "") });
        i = lc.indexOf(kw, i + kw.length);
      }
    }
    return {
      content: [{ type: "text", text: JSON.stringify({
        ok: true, keyword: args.keyword, match_count: hits.length, matches: hits,
      }) }],
    };
  } catch (e) {
    return {
      isError: true,
      content: [{ type: "text", text: JSON.stringify({ ok: false, error: (e as Error).message }) }],
    };
  }
}

export const readNoteArgsSchema = z.object({
  filename: z.string(),
  max_chars: z.number().int().positive().optional(),
  offset: z.number().int().nonnegative().optional(),
});
export type ReadNoteArgs = z.infer<typeof readNoteArgsSchema>;

const READ_NOTE_DEFAULT_MAX = 8192;

export async function readNoteTool(
  session: McpSession,
  args: ReadNoteArgs,
): Promise<CallToolResult> {
  const filename = args.filename.endsWith(".txt") ? args.filename : `${args.filename}.txt`;
  // Validate against list_notes before touching the filesystem. Catches
  // agent-fabricated filenames (the Nov-27 Qwen hallucination mode) with a
  // clear, prescriptive error rather than a bare ENOENT.
  const known = knownNoteFilenames(session);
  if (known.size > 0 && !known.has(filename)) {
    return {
      content: [{ type: "text", text: JSON.stringify(unknownNoteRefusal(filename)) }],
    };
  }
  let content: string;
  try {
    content = readNoteFn(session.patientId, filename);
  } catch (e) {
    return {
      isError: true,
      content: [{ type: "text", text: JSON.stringify({
        ok: false, error: `note read failed: ${(e as Error).message}`,
      }) }],
    };
  }
  const total = content.length;
  const offset = args.offset ?? 0;
  const max = args.max_chars ?? READ_NOTE_DEFAULT_MAX;
  const slice = content.slice(offset, offset + max);
  const truncated = offset + slice.length < total;
  return {
    content: [{ type: "text", text: JSON.stringify({
      ok: true,
      filename,
      total_chars: total,
      offset,
      returned_chars: slice.length,
      truncated,
      ...(truncated ? { next_offset: offset + slice.length } : {}),
      content: slice,
    }) }],
  };
}

// ── get_note_section ──────────────────────────────────────────────────
// Section-extraction note read: returns the note header + only the sections
// relevant to the active rubric, instead of the whole note. Much cheaper than
// read_note for long notes. Ported from RUCAM agent_v2's get_note_section.
//
// A cited quote pulled from a returned section still verifies against the FULL
// note bytes at the faithfulness gate (a section is a verbatim subset), so no
// faithfulness change is needed.

export const getNoteSectionArgsSchema = z.object({
  filename: z.string(),
  sections: z.array(z.string()).optional(),
});
export type GetNoteSectionArgs = z.infer<typeof getNoteSectionArgsSchema>;

// A line is a SECTION HEADER if it is ALL-CAPS (optionally ":") or Title-Case
// ending in ":" — and short (< 80 chars). Mirrors RUCAM's _SECTION_RE.
const GET_NOTE_SECTION_HEADER_RE =
  /^(?:[A-Z][A-Z0-9 /&-]{2,}:?$|[A-Z][a-zA-Z0-9 /&-]{2,}:)$/;

const GET_NOTE_SECTION_DEFAULT_TARGETS = [
  "assessment", "plan", "impression", "diagnos", "hospital course",
  "labs", "laboratory", "medications", "history of present illness",
  "hpi", "discharge diagnosis", "final diagnosis", "problem list",
];

export async function getNoteSectionTool(
  session: McpSession,
  args: GetNoteSectionArgs,
): Promise<CallToolResult> {
  const filename = args.filename.endsWith(".txt") ? args.filename : `${args.filename}.txt`;
  // Same unknown_note validation as read_note / read_notes.
  const known = knownNoteFilenames(session);
  if (known.size > 0 && !known.has(filename)) {
    return {
      content: [{ type: "text", text: JSON.stringify(unknownNoteRefusal(filename)) }],
    };
  }
  let content: string;
  try {
    content = readNoteFn(session.patientId, filename);
  } catch (e) {
    return {
      isError: true,
      content: [{ type: "text", text: JSON.stringify({
        ok: false, error: `note read failed: ${(e as Error).message}`,
      }) }],
    };
  }

  const targets = (args.sections ?? GET_NOTE_SECTION_DEFAULT_TARGETS).map((s) => s.toLowerCase());
  const lines = content.split("\n");

  // Always include the note header = the first 7 lines.
  const headerLines = lines.slice(0, 7);

  // Parse the body into sections keyed by their header line (":" stripped).
  // Lines before the first header land in "_preamble".
  const body = lines.slice(7);
  const parsed = new Map<string, string[]>();
  let current = "_preamble";
  parsed.set(current, []);
  for (const line of body) {
    const stripped = line.trim();
    if (stripped && stripped.length < 80 && GET_NOTE_SECTION_HEADER_RE.test(stripped)) {
      current = stripped.replace(/:$/, "");
      if (!parsed.has(current)) parsed.set(current, []);
    } else {
      parsed.get(current)!.push(line);
    }
  }

  // Collect matching sections as "[<HEADER>]\n<content>" blocks.
  const matched: string[] = [];
  for (const [name, secLines] of parsed) {
    if (targets.some((t) => name.toLowerCase().includes(t))) {
      const secContent = secLines.join("\n").trim();
      if (secContent) matched.push(`[${name}]\n${secContent}`);
    }
  }

  const headerText = headerLines.join("\n");
  const out = matched.length
    ? `${headerText}\n\n${matched.join("\n\n")}`
    : `${headerText}\n\n(No matching sections found — use read_note for full text)`;

  return {
    content: [{ type: "text", text: JSON.stringify({
      ok: true,
      filename,
      returned_chars: out.length,
      content: out,
    }) }],
  };
}

export const listStructuredDataArgsSchema = z.object({});
export type ListStructuredDataArgs = z.infer<typeof listStructuredDataArgsSchema>;

export async function listStructuredDataTool(
  session: McpSession,
  _args: ListStructuredDataArgs,
): Promise<CallToolResult> {
  // readStructured returns one object whose keys are mostly table names
  // (measurements, conditions, drug_exposures, …) plus a few scalar fields
  // (index_date). Pull row counts per table so the agent can pick which to
  // read in full; surface scalar fields directly rather than folding them
  // into the tables list as a fake 0-row table, which silently hid their
  // value from the agent (index_date was unreadable via any tool).
  try {
    const s = readStructuredFn(session.patientId);
    const tables = Object.entries(s)
      .filter(([, value]) => Array.isArray(value))
      .map(([name, rows]) => ({
        name,
        n_rows: (rows as unknown[]).length,
      }));
    return {
      content: [{
        type: "text",
        text: JSON.stringify({ ok: true, index_date: s.index_date ?? null, tables }),
      }],
    };
  } catch (e) {
    return {
      isError: true,
      content: [{ type: "text", text: JSON.stringify({ ok: false, error: (e as Error).message }) }],
    };
  }
}

export const readStructuredDataArgsSchema = z.object({
  table: z.string(),
  max_rows: z.number().int().positive().optional(),
});
export type ReadStructuredDataArgs = z.infer<typeof readStructuredDataArgsSchema>;

const READ_STRUCTURED_DEFAULT_MAX_ROWS = 200;

export async function readStructuredDataTool(
  session: McpSession,
  args: ReadStructuredDataArgs,
): Promise<CallToolResult> {
  try {
    const s = readStructuredFn(session.patientId);
    const raw = (s as unknown as Record<string, unknown>)[args.table];
    if (raw === undefined) {
      return {
        isError: true,
        content: [{ type: "text", text: JSON.stringify({
          ok: false, error: `unknown table '${args.table}'`,
          available: Object.keys(s),
        }) }],
      };
    }
    const rows = Array.isArray(raw) ? raw : [];
    const cap = args.max_rows ?? READ_STRUCTURED_DEFAULT_MAX_ROWS;
    const slice = rows.slice(0, cap);
    return {
      content: [{ type: "text", text: JSON.stringify({
        ok: true,
        table: args.table,
        total_rows: rows.length,
        returned_rows: slice.length,
        truncated: rows.length > slice.length,
        rows: slice,
      }) }],
    };
  } catch (e) {
    return {
      isError: true,
      content: [{ type: "text", text: JSON.stringify({ ok: false, error: (e as Error).message }) }],
    };
  }
}

// ── criterion-definition tools ─────────────────────────────────────────
// The skill bundle's references/criteria/<field_id>.md files describe each
// rubric criterion (its prompt, schema, derivation rules). Agents need
// them to know what to assess. Both Claude (via Read) and Codex (via shell
// cat/sed/for-loops) were paying ~10-15 file reads for these per run.
// Exposing typed list_criteria + read_criterion eliminates the shell path.

export const listCriteriaArgsSchema = z.object({});
export type ListCriteriaArgs = z.infer<typeof listCriteriaArgsSchema>;

export async function listCriteriaTool(
  session: McpSession,
  _args: ListCriteriaArgs,
): Promise<CallToolResult> {
  try {
    const dir = path.join(resolveRubricRoot(session.task.task_id), "references", "criteria");
    if (!fs.existsSync(dir)) {
      return { content: [{ type: "text", text: JSON.stringify({ ok: true, count: 0, criteria: [] }) }] };
    }
    // Derived fields (e.g. disease_extent) are computed by the platform from
    // their leaf inputs — the agent must NOT extract them. Hide them from the
    // criteria listing so the agent only commits leaf fields; otherwise it
    // wastes a commit on the derived field and tends to drop a real leaf.
    const derived = new Set(
      ((session.task.fields ?? []) as Array<{ field_id?: string; id?: string; derivation?: string }>)
        .filter((f) => f.derivation)
        .map((f) => f.field_id ?? f.id),
    );
    const criteria = fs.readdirSync(dir).sort()
      .filter((f) => f.endsWith(".md"))
      .map((f) => ({
        field_id: f.replace(/\.md$/, ""),
        filename: f,
      }))
      .filter((c) => !derived.has(c.field_id));
    return {
      content: [{ type: "text", text: JSON.stringify({ ok: true, count: criteria.length, criteria }) }],
    };
  } catch (e) {
    return {
      isError: true,
      content: [{ type: "text", text: JSON.stringify({ ok: false, error: (e as Error).message }) }],
    };
  }
}

// ── Path A: keyword-set pre-fetch ─────────────────────────────────────
//
// When a criterion's YAML frontmatter declares `uses.keyword_sets: [...]`,
// read_criterion / read_criteria pre-run `search_notes` for every term in
// each referenced set and attach the hits to the tool response as
// `pre_retrieved_evidence`. The LLM therefore sees concrete note snippets
// at the moment it asks for the criterion and cannot "skip-and-fabricate"
// the search step (the audit-log confabulation documented in platform
// issue #3). Criteria with no `uses:` block get no pre-fetch and the
// response shape is unchanged — opt-in per criterion.

const FRONTMATTER_RE = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/;

/** Lazy-populate `session.knownNoteFilenames` from `listNotes` and return it.
 *  Called once per session on the first `read_note` / `read_notes`; subsequent
 *  validations hit the cached set. Returns an empty set on resolution error
 *  so the gate is fail-open (validation must never block an agent when the
 *  underlying list call is itself failing). */
function knownNoteFilenames(session: McpSession): Set<string> {
  if (session.knownNoteFilenames) return session.knownNoteFilenames;
  const set = new Set<string>();
  try {
    for (const n of listNotesFn(session.patientId)) {
      if (n && typeof (n as { filename?: unknown }).filename === "string") {
        set.add((n as { filename: string }).filename);
      }
    }
  } catch {
    // leave empty — validation fail-open
  }
  session.knownNoteFilenames = set;
  return set;
}

/** Build the structured refusal that `read_note` / `read_notes` return when
 *  the agent asks for a filename not in the patient's note index. The
 *  message is deliberately prescriptive: the Nov-27 ACTS pilot showed that a
 *  bare "ENOENT" from the filesystem was not enough to break Qwen out of a
 *  sequential-pattern hallucination — the agent needs to be told explicitly
 *  that it cannot fabricate names from the ones it already saw. */
function unknownNoteRefusal(filename: string) {
  return {
    ok: false,
    error_code: "unknown_note",
    message:
      `No note named "${filename}" exists for this patient. Use filenames VERBATIM from the ` +
      `list_notes response — do NOT guess, auto-complete, or extrapolate names from patterns ` +
      `you see in other notes' filenames (dates, service types, sequential numbering). If you ` +
      `need to see the available filenames again, call list_notes.`,
  };
}

/** True when the criterion's markdown frontmatter declares
 *  `uses.keyword_sets`. Used by the `set_field_assessment` read-first gate
 *  to decide whether to require a prior `read_criterion` for this field.
 *  Returns false (unsafe-side-off) on any resolution / parse error — a
 *  missing or malformed criterion must not block the agent. */
function criterionDeclaresKeywordSets(session: McpSession, fieldId: string): boolean {
  if (!/^[a-zA-Z0-9_-]+$/.test(fieldId)) return false;
  try {
    const fp = path.join(
      resolveRubricRoot(session.task.task_id), "references", "criteria",
      `${fieldId}.md`,
    );
    if (!fs.existsSync(fp)) return false;
    const content = fs.readFileSync(fp, "utf8");
    return parseCriterionKeywordSetRefs(content).length > 0;
  } catch {
    return false;
  }
}

/** Extract `uses.keyword_sets` from a criterion markdown's YAML frontmatter.
 *  Returns `[]` if frontmatter is missing, malformed, or has no uses block. */
function parseCriterionKeywordSetRefs(markdown: string): string[] {
  const m = FRONTMATTER_RE.exec(markdown);
  if (!m) return [];
  let fm: unknown;
  try { fm = parseYaml(m[1]); } catch { return []; }
  if (!fm || typeof fm !== "object") return [];
  const uses = (fm as { uses?: unknown }).uses;
  if (!uses || typeof uses !== "object") return [];
  const ks = (uses as { keyword_sets?: unknown }).keyword_sets;
  if (!Array.isArray(ks)) return [];
  return ks.filter((x): x is string => typeof x === "string" && x.length > 0);
}

/** Per-criterion caps: enough to help a short-note chart, bounded enough
 *  to keep the tool response below ~8 KiB even when every term hits. */
const PREFETCH_MAX_HITS_PER_TERM = 3;
const PREFETCH_MAX_TOTAL_HITS = 40;

interface PrefetchedHit {
  term: string;
  filename: string;
  offset: number;
  snippet: string;
}

interface PrefetchedEvidence {
  keyword_sets_consulted: string[];
  keyword_sets_missing: string[];
  terms_searched: number;
  total_hits: number;
  hits: PrefetchedHit[];
  /** True when the (filename, offset) hit cap was reached and some hits were dropped. */
  truncated: boolean;
}

/** Resolve each referenced keyword_set id against the compiled operational
 *  layer and run `searchNotesTool` once per term. Hits that share
 *  (filename, offset) with an earlier hit are dropped — same chart span,
 *  different term — and the kept hit's `term` records the first term that
 *  surfaced it. */
async function prefetchKeywordHits(
  session: McpSession,
  keywordSetIds: string[],
): Promise<PrefetchedEvidence> {
  const consulted: string[] = [];
  const missing: string[] = [];
  const seen = new Set<string>();
  const hits: PrefetchedHit[] = [];
  let termsSearched = 0;
  let truncated = false;

  const operational = session.task.operational;
  const sets = operational?.keyword_sets ?? {};

  outer: for (const setId of keywordSetIds) {
    const set = sets[setId];
    if (!set) { missing.push(setId); continue; }
    consulted.push(setId);
    const terms = Array.isArray(set.terms) ? set.terms : [];
    for (const term of terms) {
      if (typeof term !== "string" || term.length === 0) continue;
      termsSearched++;
      const result = await searchNotesTool(session, {
        keyword: term,
        max_matches: PREFETCH_MAX_HITS_PER_TERM,
      });
      if (result.isError) continue;
      let parsed: { matches?: Array<{ filename: string; offset: number; snippet: string }> } = {};
      try { parsed = JSON.parse(result.content[0]?.text ?? "{}"); } catch { continue; }
      for (const m of parsed.matches ?? []) {
        if (hits.length >= PREFETCH_MAX_TOTAL_HITS) { truncated = true; break outer; }
        const key = `${m.filename}@${m.offset}`;
        if (seen.has(key)) continue;
        seen.add(key);
        hits.push({ term, filename: m.filename, offset: m.offset, snippet: m.snippet });
      }
    }
  }

  return {
    keyword_sets_consulted: consulted,
    keyword_sets_missing: missing,
    terms_searched: termsSearched,
    total_hits: hits.length,
    hits,
    truncated,
  };
}

export const readCriterionArgsSchema = z.object({
  field_id: z.string(),
});
export type ReadCriterionArgs = z.infer<typeof readCriterionArgsSchema>;

export async function readCriterionTool(
  session: McpSession,
  args: ReadCriterionArgs,
): Promise<CallToolResult> {
  try {
    if (!/^[a-zA-Z0-9_-]+$/.test(args.field_id)) {
      return {
        isError: true,
        content: [{ type: "text", text: JSON.stringify({
          ok: false, error: `invalid field_id: ${args.field_id}`,
        }) }],
      };
    }
    const fp = path.join(
      resolveRubricRoot(session.task.task_id), "references", "criteria",
      `${args.field_id}.md`,
    );
    if (!fs.existsSync(fp)) {
      return {
        isError: true,
        content: [{ type: "text", text: JSON.stringify({
          ok: false, error: `criterion '${args.field_id}' not found`,
        }) }],
      };
    }
    const content = fs.readFileSync(fp, "utf8");
    const refs = parseCriterionKeywordSetRefs(content);
    const payload: Record<string, unknown> = {
      ok: true, field_id: args.field_id, content,
    };
    if (refs.length > 0) {
      payload.pre_retrieved_evidence = await prefetchKeywordHits(session, refs);
    }
    // Record that the agent has read this criterion. The `set_field_assessment`
    // gate consults this set for criteria that declare `uses.keyword_sets`.
    (session.readCriterionIds ??= new Set()).add(args.field_id);
    return {
      content: [{ type: "text", text: JSON.stringify(payload) }],
    };
  } catch (e) {
    return {
      isError: true,
      content: [{ type: "text", text: JSON.stringify({ ok: false, error: (e as Error).message }) }],
    };
  }
}

// ── batch read tools ──────────────────────────────────────────────────
// One LLM turn per file read is wasteful — agents typically want to read
// all N notes or all M criteria as one "data fetch" step before reasoning.
// These bulk variants return everything in a single tool result, cutting
// total turn count (and therefore per-turn preamble overhead).

const BATCH_MAX = 50; // hard cap so the agent can't request 1000 at once

export const readNotesArgsSchema = z.object({
  filenames: z.array(z.string()).max(BATCH_MAX),
  max_chars_per_note: z.number().int().positive().optional(),
});
export type ReadNotesArgs = z.infer<typeof readNotesArgsSchema>;

export async function readNotesTool(
  session: McpSession,
  args: ReadNotesArgs,
): Promise<CallToolResult> {
  const per = args.max_chars_per_note ?? READ_NOTE_DEFAULT_MAX;
  const results: Array<Record<string, unknown>> = [];
  // Validate per-entry against list_notes before touching the filesystem.
  // Same unknown_note refusal as readNoteTool; per-entry so a mix of real
  // and fabricated names returns real notes for the real ones and a clear
  // refusal for the fabricated ones.
  const known = knownNoteFilenames(session);
  for (const raw of args.filenames) {
    const filename = raw.endsWith(".txt") ? raw : `${raw}.txt`;
    if (known.size > 0 && !known.has(filename)) {
      results.push({ filename, ...unknownNoteRefusal(filename) });
      continue;
    }
    try {
      const content = readNoteFn(session.patientId, filename);
      const total = content.length;
      const slice = content.slice(0, per);
      results.push({
        filename,
        total_chars: total,
        returned_chars: slice.length,
        truncated: total > slice.length,
        content: slice,
      });
    } catch (e) {
      results.push({
        filename,
        ok: false,
        error: (e as Error).message,
      });
    }
  }
  return {
    content: [{ type: "text", text: JSON.stringify({
      ok: true, count: results.length, notes: results,
    }) }],
  };
}

export const readCriteriaArgsSchema = z.object({
  field_ids: z.array(z.string()).max(BATCH_MAX),
});
export type ReadCriteriaArgs = z.infer<typeof readCriteriaArgsSchema>;

// Extract a compact header (prompt + answer_schema + uses.keyword_sets) from
// a criterion markdown's YAML frontmatter. Returns {} when frontmatter is
// missing or malformed — the agent still gets field_id back.
function parseCriterionHeader(markdown: string): {
  prompt?: string;
  answer_schema?: unknown;
  keyword_sets_declared?: string[];
} {
  const m = FRONTMATTER_RE.exec(markdown);
  if (!m) return {};
  let fm: unknown;
  try { fm = parseYaml(m[1]); } catch { return {}; }
  if (!fm || typeof fm !== "object") return {};
  const f = fm as Record<string, unknown>;
  const out: {
    prompt?: string;
    answer_schema?: unknown;
    keyword_sets_declared?: string[];
  } = {};
  if (typeof f.prompt === "string") out.prompt = f.prompt;
  if (f.answer_schema !== undefined) out.answer_schema = f.answer_schema;
  const uses = f.uses;
  if (uses && typeof uses === "object") {
    const ks = (uses as { keyword_sets?: unknown }).keyword_sets;
    if (Array.isArray(ks)) {
      const ids = ks.filter((x): x is string => typeof x === "string" && x.length > 0);
      if (ids.length > 0) out.keyword_sets_declared = ids;
    }
  }
  return out;
}

// `read_criteria` returns a COMPACT index only: field_id + prompt +
// answer_schema + which keyword_sets will fire on read_criterion. It does
// NOT return the criterion body and does NOT run the keyword-set prefetch.
//
// Why: the full body + full prefetch for N criteria blew past deepagents'
// per-tool-result size threshold. deepagents then offloaded the result to
// `/large_tool_results/<id>`, embedded a pointer in the transcript, and the
// agent read that file with offset=0 limit=100 in a loop until it hit the
// graph recursion limit. For the ACTS rubric (29 criteria) this was a hard
// fail: zero criterion bodies and zero prefetched evidence reached the model.
// Skipping prefetch in the batch path is also a big perf win — it drops the
// ~1,160 wasted search_notes calls per rubric bootstrap.
//
// Agent flow after this change:
//   1. list_criteria / read_criteria  → plan which fields to answer
//   2. read_criterion(field_id)       → full body + pre_retrieved_evidence
//   3. set_field_assessment           → using hits from step 2
export async function readCriteriaTool(
  session: McpSession,
  args: ReadCriteriaArgs,
): Promise<CallToolResult> {
  const dir = path.join(resolveRubricRoot(session.task.task_id), "references", "criteria");
  const results: Array<Record<string, unknown>> = [];
  for (const field_id of args.field_ids) {
    if (!/^[a-zA-Z0-9_-]+$/.test(field_id)) {
      results.push({ field_id, ok: false, error: `invalid field_id` });
      continue;
    }
    const fp = path.join(dir, `${field_id}.md`);
    if (!fs.existsSync(fp)) {
      results.push({ field_id, ok: false, error: `not found` });
      continue;
    }
    try {
      const content = fs.readFileSync(fp, "utf8");
      const header = parseCriterionHeader(content);
      results.push({ field_id, ...header });
    } catch (e) {
      results.push({ field_id, ok: false, error: (e as Error).message });
    }
  }
  return {
    content: [{ type: "text", text: JSON.stringify({
      ok: true,
      count: results.length,
      criteria: results,
      guidance: "Call read_criterion(field_id) to get the full extraction guidance for one criterion and its pre-retrieved evidence hits.",
    }) }],
  };
}
