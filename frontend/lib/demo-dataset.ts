export const DEMO_CSV_PATH = "/demo/vera_sales_campaign_demo.csv";
export const DEMO_CSV_NAME = "vera_sales_campaign_demo.csv";

/** Fetch the actual versioned public CSV; do not fabricate a backend dataset ID. */
export async function fetchDemoDataset(fetcher: typeof fetch = fetch): Promise<File> {
  let response: Response;
  try {
    response = await fetcher(DEMO_CSV_PATH, { cache: "no-store" });
  } catch {
    throw new Error("Unable to load the demo CSV. Check your connection and try again.");
  }
  if (!response.ok) throw new Error(`Unable to load the demo CSV (HTTP ${response.status}).`);
  const blob = await response.blob();
  if (!blob.size) throw new Error("The demo CSV is empty.");
  return new File([blob], DEMO_CSV_NAME, { type: "text/csv" });
}
