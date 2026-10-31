import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { createHash } from "node:crypto";

const official = resolve(process.cwd(), "../demo/vera_sales_campaign_demo.csv");
const publicCopy = resolve(process.cwd(), "public/demo/vera_sales_campaign_demo.csv");

function sha256(path) {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

const expected = sha256(official);
const actual = sha256(publicCopy);
if (expected !== actual) {
  process.stderr.write("Demo CSV differs from the versioned official dataset. Re-copy it from demo/.\n");
  process.exitCode = 1;
} else {
  process.stdout.write(`Demo dataset integrity OK: ${actual}\n`);
}
