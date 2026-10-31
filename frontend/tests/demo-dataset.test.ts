import assert from "node:assert/strict";
import test from "node:test";
import { DEMO_CSV_NAME, DEMO_CSV_PATH, fetchDemoDataset } from "../lib/demo-dataset";

test("official demo is fetched as a File for the ordinary upload path", async () => {
  const csv = "date,region,sales\n2026-08-01,North,123\n";
  let requested = "";
  const mockedFetch = (async (url: string) => {
    requested = url;
    return new Response(csv, { status: 200, headers: { "content-type": "text/csv" } });
  }) as typeof fetch;
  const file = await fetchDemoDataset(mockedFetch);
  assert.equal(requested, DEMO_CSV_PATH);
  assert.equal(file.name, DEMO_CSV_NAME);
  assert.equal(file.type, "text/csv");
  assert.equal(await file.text(), csv);
});

test("demo loader fails on missing or empty asset, without a fake dataset", async () => {
  await assert.rejects(
    fetchDemoDataset((async () => new Response("Missing", { status: 404 })) as typeof fetch),
    /HTTP 404/,
  );
  await assert.rejects(
    fetchDemoDataset((async () => new Response("", { status: 200 })) as typeof fetch),
    /empty/,
  );
});

test("demo loader reports network failures", async () => {
  await assert.rejects(
    fetchDemoDataset((async () => { throw new Error("connection reset"); }) as typeof fetch),
    /Unable to load the demo CSV/,
  );
});
