import { describe, it, expect, vi, beforeAll, afterAll, beforeEach, afterEach } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

// ─────────────────────────────────────────────────────────────────────────────
// Suite 1 — upstream: hyphenated field_id validation (adherence step-table ids).
// Criterion filenames use the question-id convention (e.g. T2-StepTherapyMatch),
// which contains hyphens — the field_id validation must accept them, or the
// agent's mandated read_criterion(field_id="T2-StepTherapyMatch") call fails
// while list_criteria happily lists the same id.
// ─────────────────────────────────────────────────────────────────────────────

import {
  readCriterionTool,
  readCriteriaTool,
  listCriteriaTool,
  type McpSession,
} from "./index.js";

const CRITERION_ID = "T2-StepTherapyMatch";
const CRITERION_BODY = "# T2-StepTherapyMatch — stepwise-therapy reference\n\nStep 1 ...\n";

let hyphenRubricRoot: string;
let prevOverride: string | undefined;

const hyphenSession: McpSession = {
  patientId: "p1",
  task: { task_id: "asthma-adherence" } as any,
  sessionId: "s1",
} as any;

function parseResult(result: { content: Array<unknown> }) {
  return JSON.parse((result.content[0] as { text: string }).text);
}

beforeAll(() => {
  hyphenRubricRoot = fs.mkdtempSync(path.join(os.tmpdir(), "rubric-root-"));
  fs.mkdirSync(path.join(hyphenRubricRoot, "references", "criteria"), { recursive: true });
  fs.writeFileSync(
    path.join(hyphenRubricRoot, "references", "criteria", `${CRITERION_ID}.md`),
    CRITERION_BODY,
  );
  prevOverride = process.env.CHART_REVIEW_RUBRIC_ROOT;
  process.env.CHART_REVIEW_RUBRIC_ROOT = hyphenRubricRoot;
});

afterAll(() => {
  if (prevOverride === undefined) delete process.env.CHART_REVIEW_RUBRIC_ROOT;
  else process.env.CHART_REVIEW_RUBRIC_ROOT = prevOverride;
  fs.rmSync(hyphenRubricRoot, { recursive: true, force: true });
});

describe("read_criterion field_id validation", () => {
  it("lists the hyphenated criterion", async () => {
    const body = parseResult(await listCriteriaTool(hyphenSession, {}));
    expect(body.ok).toBe(true);
    expect(body.criteria.map((c: { field_id: string }) => c.field_id)).toContain(CRITERION_ID);
  });

  it("reads a hyphenated field_id (single)", async () => {
    const body = parseResult(await readCriterionTool(hyphenSession, { field_id: CRITERION_ID }));
    expect(body.ok).toBe(true);
    expect(body.content).toBe(CRITERION_BODY);
  });

  it("lists a hyphenated field_id via the compact read_criteria (no body)", async () => {
    // read_criteria is a COMPACT overview — field_id + header only, never
    // the criterion body. Agents call read_criterion per field for the body.
    const body = parseResult(await readCriteriaTool(hyphenSession, { field_ids: [CRITERION_ID] }));
    expect(body.ok).toBe(true);
    expect(body.criteria?.[0]?.field_id).toBe(CRITERION_ID);
    expect(body.criteria?.[0]?.content).toBeUndefined();
  });

  it("still rejects path traversal", async () => {
    const body = parseResult(await readCriterionTool(hyphenSession, { field_id: "../secrets" }));
    expect(body.ok).toBe(false);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Suite 2 — Path A: keyword-set pre-fetch.
// When a criterion declares `uses.keyword_sets`, read_criterion/read_criteria
// pre-run search_notes for every term and attach the hits as
// `pre_retrieved_evidence`. Verifies the opt-in contract, the dedup/cap
// behavior, and backward-compat (no uses:, response shape unchanged).
// ─────────────────────────────────────────────────────────────────────────────

const FAKE_NOTES: Record<string, string> = {
  "n1.txt": [
    "Progress note 2026-01-15",
    "PMH: former smoker, quit 2020. Pack-years 20.",
    "Denies current tobacco use.",
  ].join("\n"),
  "n2.txt": [
    "Progress note 2026-06-10",
    "Patient states he stopped smoking five years ago.",
    "Occasional cigar at social events.",
  ].join("\n"),
  "n3.txt": [
    "Progress note 2026-11-02",
    "Vitals stable. No acute concerns today.",
  ].join("\n"),
};

vi.mock("@chart-review/patients", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@chart-review/patients")>();
  return {
    ...actual,
    listNotes: vi.fn(() => Object.keys(FAKE_NOTES).map((filename) => ({
      filename, note_date: null, note_type: null,
    }))),
    readNote: vi.fn((_pid: string, filename: string) => {
      const txt = FAKE_NOTES[filename];
      if (txt === undefined) throw new Error(`no such note: ${filename}`);
      return txt;
    }),
  };
});

function makePrefetchSession(withKeywordSets: boolean): McpSession {
  const kw = withKeywordSets ? {
    kw_smoking_tobacco: {
      id: "kw_smoking_tobacco",
      description: "Smoking and tobacco terms",
      terms: ["smoker", "tobacco", "cigar", "pack-year"],
    },
  } : {};
  return {
    patientId: "p1",
    sessionId: "s1",
    task: {
      task_id: "acts",
      fields: [],
      operational: {
        keyword_sets: kw,
        code_sets: {},
        edge_cases: [],
        exemplars: {},
        note_type_filters: { filters: {} },
      },
    } as any,
  };
}

function writeCriterion(dir: string, fieldId: string, frontmatter: string, body = "") {
  const criteriaDir = path.join(dir, "references", "criteria");
  fs.mkdirSync(criteriaDir, { recursive: true });
  fs.writeFileSync(
    path.join(criteriaDir, `${fieldId}.md`),
    `---\n${frontmatter}\n---\n\n${body}\n`,
    "utf8",
  );
}

describe("readCriterionTool pre-fetch", () => {
  let tmp: string;
  let prevRubricRoot: string | undefined;

  beforeEach(() => {
    tmp = fs.mkdtempSync(path.join(os.tmpdir(), "acts-prefetch-"));
    prevRubricRoot = process.env.CHART_REVIEW_RUBRIC_ROOT;
    process.env.CHART_REVIEW_RUBRIC_ROOT = tmp;
  });

  afterEach(() => {
    if (prevRubricRoot === undefined) delete process.env.CHART_REVIEW_RUBRIC_ROOT;
    else process.env.CHART_REVIEW_RUBRIC_ROOT = prevRubricRoot;
    fs.rmSync(tmp, { recursive: true, force: true });
  });

  it("returns unchanged response when the criterion has no uses: block", async () => {
    writeCriterion(tmp, "no_uses_field",
      "field_id: no_uses_field\nprompt: test\nanswer_schema:\n  type: string",
      "# Criterion\nBody here.");
    const res = await readCriterionTool(makePrefetchSession(true), { field_id: "no_uses_field" });
    const body = parseResult(res);
    expect(body.ok).toBe(true);
    expect(body.field_id).toBe("no_uses_field");
    expect(body.content).toContain("Body here.");
    expect(body.pre_retrieved_evidence).toBeUndefined();
  });

  it("attaches pre_retrieved_evidence when uses.keyword_sets references a resolved set", async () => {
    writeCriterion(tmp, "smoking_status",
      "field_id: smoking_status\nprompt: What is the smoking status?\nanswer_schema:\n  enum: [current, former, never, unknown]\nuses:\n  keyword_sets:\n    - kw_smoking_tobacco",
      "# Criterion smoking_status");
    const res = await readCriterionTool(makePrefetchSession(true), { field_id: "smoking_status" });
    const body = parseResult(res);
    expect(body.ok).toBe(true);
    expect(body.pre_retrieved_evidence).toBeDefined();
    const ev = body.pre_retrieved_evidence;
    expect(ev.keyword_sets_consulted).toEqual(["kw_smoking_tobacco"]);
    expect(ev.keyword_sets_missing).toEqual([]);
    expect(ev.terms_searched).toBe(4);
    expect(ev.total_hits).toBeGreaterThan(0);
    for (const hit of ev.hits) {
      expect(typeof hit.term).toBe("string");
      expect(typeof hit.filename).toBe("string");
      expect(typeof hit.offset).toBe("number");
      expect(typeof hit.snippet).toBe("string");
    }
    const fnames = ev.hits.map((h: any) => h.filename);
    expect(fnames).toContain("n1.txt");
    // Deduplicated: same (filename, offset) appears at most once.
    const keys = ev.hits.map((h: any) => `${h.filename}@${h.offset}`);
    expect(new Set(keys).size).toBe(keys.length);
  });

  it("records missing keyword_set ids and keeps processing resolved ones", async () => {
    writeCriterion(tmp, "smoking_status",
      "field_id: smoking_status\nprompt: x\nanswer_schema:\n  type: string\nuses:\n  keyword_sets:\n    - kw_smoking_tobacco\n    - kw_does_not_exist",
      "");
    const res = await readCriterionTool(makePrefetchSession(true), { field_id: "smoking_status" });
    const body = parseResult(res);
    const ev = body.pre_retrieved_evidence;
    expect(ev.keyword_sets_consulted).toEqual(["kw_smoking_tobacco"]);
    expect(ev.keyword_sets_missing).toEqual(["kw_does_not_exist"]);
  });

  it("returns an empty hit list (not a missing field) when the chart has no matches", async () => {
    writeCriterion(tmp, "allergen",
      "field_id: allergen\nprompt: x\nanswer_schema:\n  type: array\nuses:\n  keyword_sets:\n    - kw_empty",
      "");
    const sess = makePrefetchSession(false);
    (sess.task.operational as any).keyword_sets = {
      kw_empty: {
        id: "kw_empty",
        description: "",
        terms: ["xyznotinanynote"],
      },
    };
    const res = await readCriterionTool(sess, { field_id: "allergen" });
    const ev = parseResult(res).pre_retrieved_evidence;
    expect(ev.keyword_sets_consulted).toEqual(["kw_empty"]);
    expect(ev.total_hits).toBe(0);
    expect(ev.hits).toEqual([]);
    expect(ev.truncated).toBe(false);
  });

  it("silently skips malformed uses and never throws", async () => {
    writeCriterion(tmp, "weird",
      "field_id: weird\nprompt: x\nanswer_schema:\n  type: string\nuses: not-an-object",
      "");
    const res = await readCriterionTool(makePrefetchSession(true), { field_id: "weird" });
    const body = parseResult(res);
    expect(body.ok).toBe(true);
    expect(body.pre_retrieved_evidence).toBeUndefined();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// readCriteriaTool is the COMPACT rubric overview. It must NOT return the
// criterion body and must NOT run the keyword-set prefetch — batching both
// across N criteria blew past deepagents' per-tool-result threshold and the
// runtime offloaded the result to /large_tool_results/<id>, which the agent
// then couldn't page past the first 100 lines (the 190-recursion-limit kill
// seen in the Nov-20 ACTS pilot). Prefetch is exclusively on read_criterion.
// ─────────────────────────────────────────────────────────────────────────────
describe("readCriteriaTool compact overview (no body, no prefetch)", () => {
  let tmp: string;
  let prevRubricRoot: string | undefined;

  beforeEach(() => {
    tmp = fs.mkdtempSync(path.join(os.tmpdir(), "acts-compact-batch-"));
    prevRubricRoot = process.env.CHART_REVIEW_RUBRIC_ROOT;
    process.env.CHART_REVIEW_RUBRIC_ROOT = tmp;
  });

  afterEach(() => {
    if (prevRubricRoot === undefined) delete process.env.CHART_REVIEW_RUBRIC_ROOT;
    else process.env.CHART_REVIEW_RUBRIC_ROOT = prevRubricRoot;
    fs.rmSync(tmp, { recursive: true, force: true });
  });

  it("returns prompt + answer_schema + keyword_sets_declared, but NO body and NO hits", async () => {
    writeCriterion(tmp, "smoking_status",
      "field_id: smoking_status\nprompt: What is the smoking status?\nanswer_schema:\n  enum: [current, former, never, unknown]\nuses:\n  keyword_sets:\n    - kw_smoking_tobacco",
      "# SHOULD NOT APPEAR IN BATCH RESPONSE\nFull body with extraction rules …");
    writeCriterion(tmp, "no_uses_field",
      "field_id: no_uses_field\nprompt: A plain field\nanswer_schema:\n  type: string",
      "# body");

    const res = await readCriteriaTool(makePrefetchSession(true), {
      field_ids: ["smoking_status", "no_uses_field"],
    });
    const body = parseResult(res);

    expect(body.ok).toBe(true);
    expect(body.count).toBe(2);
    expect(body.guidance).toContain("read_criterion");

    const bySmoking = body.criteria.find((c: any) => c.field_id === "smoking_status");
    expect(bySmoking.prompt).toBe("What is the smoking status?");
    expect(bySmoking.answer_schema).toEqual({ enum: ["current", "former", "never", "unknown"] });
    expect(bySmoking.keyword_sets_declared).toEqual(["kw_smoking_tobacco"]);
    expect(bySmoking.content).toBeUndefined();
    expect(bySmoking.pre_retrieved_evidence).toBeUndefined();

    const byPlain = body.criteria.find((c: any) => c.field_id === "no_uses_field");
    expect(byPlain.prompt).toBe("A plain field");
    expect(byPlain.answer_schema).toEqual({ type: "string" });
    expect(byPlain.keyword_sets_declared).toBeUndefined();
    expect(byPlain.content).toBeUndefined();
  });

  it("does not call search_notes while building the compact batch (no prefetch on this path)", async () => {
    // The point of compact read_criteria is that bootstrapping the rubric
    // costs zero search_notes calls. If prefetch ever gets re-wired on the
    // batch path, this assertion flips and we catch the regression.
    writeCriterion(tmp, "smoking_status",
      "field_id: smoking_status\nprompt: x\nanswer_schema:\n  type: string\nuses:\n  keyword_sets:\n    - kw_smoking_tobacco",
      "");

    const { readNote } = await import("@chart-review/patients") as any;
    const callsBefore = (readNote as ReturnType<typeof vi.fn>).mock.calls.length;
    await readCriteriaTool(makePrefetchSession(true), { field_ids: ["smoking_status"] });
    const callsAfter = (readNote as ReturnType<typeof vi.fn>).mock.calls.length;
    // search_notes internally reads notes via readNote. Zero hits was the
    // signal that prefetch didn't fire; equivalently, readNote was never
    // touched during compact read_criteria.
    expect(callsAfter).toBe(callsBefore);
  });

  it("keeps the batch response small (serialized size well below the offload threshold)", async () => {
    // Simulate a 29-criterion ACTS rubric with a long realistic body. The
    // compact response must stay small even when every criterion's body is
    // huge, because the body is deliberately omitted.
    const LONG_BODY = "# Criterion\n\n" + "Lorem ipsum dolor sit amet, ".repeat(400);
    const ids: string[] = [];
    for (let i = 0; i < 29; i++) {
      const fid = `field_${i}`;
      ids.push(fid);
      writeCriterion(tmp, fid,
        `field_id: ${fid}\nprompt: question ${i} about something\nanswer_schema:\n  type: integer\n  minimum: 0\n  maximum: 100\nuses:\n  keyword_sets:\n    - kw_${i}`,
        LONG_BODY);
    }
    const res = await readCriteriaTool(makePrefetchSession(true), { field_ids: ids });
    const serialized = (res.content[0] as { text: string }).text;
    const body = JSON.parse(serialized);
    expect(body.count).toBe(29);
    // 8 KiB is the per-note read-cap the platform uses to stay under
    // deepagents' large-tool-result offload; our batch overview is well
    // below even that.
    expect(serialized.length).toBeLessThan(8 * 1024);
  });
});
