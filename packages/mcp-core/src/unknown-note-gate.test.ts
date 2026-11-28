import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// ─────────────────────────────────────────────────────────────────────────────
// read_note / read_notes / get_note_section — unknown_note gate.
//
// Nov-27 ACTS pilot (pilot-iter_004, commit 3eba81a): patient …7288 hit the
// recursion limit at 190 after 83 read_notes calls with ZERO reads succeeding.
// Qwen3-32B saw early notes (001, 004, 007, 010) were all type
// `Emergency_Dept_MD_Progress_Note` and started FABRICATING sequential
// filenames (e.g. `043_2015-12-31_Emergency_Dept_MD_Progress_Note.txt`) that
// did not exist in the patient's notes dir. Each fabricated read returned a
// bare ENOENT, which did not break the agent out of the pattern.
//
// This gate catches it: read tools validate the filename against list_notes
// BEFORE touching the filesystem and return a prescriptive refusal
// (`error_code: "unknown_note"`) that tells the agent explicitly not to
// fabricate names. Opt-in by design: if list_notes resolution itself fails,
// the gate is fail-open (an empty set never blocks the agent).
// ─────────────────────────────────────────────────────────────────────────────

const FAKE_NOTES: Record<string, string> = {
  "001_2015-05-04_Emergency_Dept_MD_Progress_Note.txt":
    "Progress Note 2015-05-04\n\nASSESSMENT:\nChest pain, workup negative.",
  "002_2015-05-04_Chest_PA-Lat_XR.txt":
    "Chest X-ray: no acute cardiopulmonary findings.",
  "003_2015-05-04_ED_Discharge_Note.txt":
    "ED Discharge\n\nIMPRESSION:\nStable, discharge home.",
};

vi.mock("@chart-review/patients", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@chart-review/patients")>();
  return {
    ...actual,
    listNotes: vi.fn(() =>
      Object.keys(FAKE_NOTES).map((filename) => ({
        filename, note_date: null, note_type: null,
      })),
    ),
    readNote: vi.fn((_pid: string, filename: string) => {
      const txt = FAKE_NOTES[filename];
      if (txt === undefined) {
        throw new Error(
          `ENOENT: no such file or directory, open '/fake/${filename}'`,
        );
      }
      return txt;
    }),
  };
});

import {
  readNoteTool,
  readNotesTool,
  getNoteSectionTool,
  type McpSession,
} from "./index.js";

function makeSession(): McpSession {
  return {
    patientId: "p1",
    sessionId: "s1",
    task: {
      task_id: "acts",
      task_kind: "phenotype",
      source_document_sha: "x",
      fields: [],
    } as any,
  };
}

function parseResult(res: { content: Array<unknown> }) {
  return JSON.parse((res.content[0] as { text: string }).text);
}

describe("read_note unknown_note gate", () => {
  it("REFUSES a fabricated filename with error_code=unknown_note", async () => {
    const s = makeSession();
    const res = await readNoteTool(s, {
      // The exact hallucination shape from the Nov-27 pilot — a real index
      // (001-003 exist) collated with a wrong date + wrong service name.
      filename: "001_2099-01-01_Fabricated_Service.txt",
    } as any);
    const body = parseResult(res);
    expect(body.ok).toBe(false);
    expect(body.error_code).toBe("unknown_note");
    // Message must be prescriptive — ENOENT alone was not enough for Qwen.
    expect(body.message).toMatch(/list_notes/);
    expect(body.message).toMatch(/do NOT/i);
    expect(body.message).toMatch(/verbatim/i);
  });

  it("ACCEPTS a filename that is in the real list_notes response", async () => {
    const s = makeSession();
    const res = await readNoteTool(s, {
      filename: "001_2015-05-04_Emergency_Dept_MD_Progress_Note.txt",
    } as any);
    const body = parseResult(res);
    expect(body.ok).toBe(true);
    expect(body.content).toContain("Chest pain");
  });

  it("accepts a filename passed WITHOUT the .txt extension", async () => {
    const s = makeSession();
    const res = await readNoteTool(s, {
      filename: "002_2015-05-04_Chest_PA-Lat_XR",
    } as any);
    const body = parseResult(res);
    expect(body.ok).toBe(true);
    expect(body.content).toContain("Chest X-ray");
  });
});

describe("read_notes unknown_note gate (batch)", () => {
  it("returns per-entry refusal for fabricated names, keeps real reads working", async () => {
    const s = makeSession();
    const res = await readNotesTool(s, {
      filenames: [
        "001_2015-05-04_Emergency_Dept_MD_Progress_Note.txt",
        "043_2018-10-11_Fabricated_EKG.txt",
        "002_2015-05-04_Chest_PA-Lat_XR.txt",
      ],
    } as any);
    const body = parseResult(res);
    expect(body.ok).toBe(true);
    expect(body.count).toBe(3);
    expect(body.notes[0].content).toContain("Chest pain");
    expect(body.notes[1].ok).toBe(false);
    expect(body.notes[1].error_code).toBe("unknown_note");
    expect(body.notes[2].content).toContain("Chest X-ray");
  });

  it("caches the known-note set across tool calls within one session", async () => {
    // The point of caching is that we don't re-list on every read. Verify by
    // counting mock invocations of listNotes.
    const s = makeSession();
    const { listNotes } = await import("@chart-review/patients") as any;
    const before = (listNotes as ReturnType<typeof vi.fn>).mock.calls.length;
    await readNotesTool(s, {
      filenames: ["001_2015-05-04_Emergency_Dept_MD_Progress_Note.txt"],
    } as any);
    await readNotesTool(s, {
      filenames: ["002_2015-05-04_Chest_PA-Lat_XR.txt"],
    } as any);
    await readNoteTool(s, {
      filename: "003_2015-05-04_ED_Discharge_Note.txt",
    } as any);
    const after = (listNotes as ReturnType<typeof vi.fn>).mock.calls.length;
    // One listNotes call satisfied three reads.
    expect(after - before).toBe(1);
  });

  it("different SESSIONS don't share the cache", async () => {
    const s1 = makeSession();
    const s2 = makeSession();
    const { listNotes } = await import("@chart-review/patients") as any;
    const before = (listNotes as ReturnType<typeof vi.fn>).mock.calls.length;
    await readNoteTool(s1, { filename: "001_2015-05-04_Emergency_Dept_MD_Progress_Note.txt" } as any);
    await readNoteTool(s2, { filename: "001_2015-05-04_Emergency_Dept_MD_Progress_Note.txt" } as any);
    const after = (listNotes as ReturnType<typeof vi.fn>).mock.calls.length;
    expect(after - before).toBe(2);
  });
});

describe("get_note_section unknown_note gate", () => {
  it("REFUSES a fabricated filename the same way read_note does", async () => {
    const s = makeSession();
    const res = await getNoteSectionTool(s, {
      filename: "999_2099-01-01_Not_A_Real_Note.txt",
    } as any);
    const body = parseResult(res);
    expect(body.ok).toBe(false);
    expect(body.error_code).toBe("unknown_note");
    expect(body.message).toMatch(/list_notes/);
  });
});

describe("unknown_note gate is fail-open", () => {
  it("does NOT block reads when list_notes itself cannot resolve", async () => {
    // Simulate the list_notes call throwing: the cache populates to an empty
    // set, which means .size === 0 and the validation skips. The underlying
    // readNote call still runs — if the file is really there, we read it.
    const { listNotes } = await import("@chart-review/patients") as any;
    const orig = listNotes.getMockImplementation();
    listNotes.mockImplementationOnce(() => { throw new Error("mount down"); });
    const s = makeSession();
    const res = await readNoteTool(s, {
      filename: "001_2015-05-04_Emergency_Dept_MD_Progress_Note.txt",
    } as any);
    const body = parseResult(res);
    expect(body.ok).toBe(true);
    expect(body.content).toContain("Chest pain");
    listNotes.mockImplementation(orig);
  });
});
