import puppeteer from 'puppeteer-core';
import fs from 'fs';
import path from 'path';

const SCREENSHOT_DIR = '/mnt/c/Users/DELL WORKSTATION/.gemini/antigravity/brain/51ddc667-0397-4574-ae1d-429c8451a7a5/qa_screenshots';
const CHROME_PATH = '/mnt/c/Program Files/Google/Chrome/Application/chrome.exe';
const APP_URL = 'http://localhost:3000';

async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function setReactTextarea(page, value) {
  await page.evaluate((val) => {
    const el = document.querySelector('textarea');
    if (!el) return;
    const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value')?.set;
    if (nativeSetter) {
      nativeSetter.call(el, val);
    } else {
      el.value = val;
    }
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
  await sleep(400);
}

async function clickButtonWithText(page, text) {
  const success = await page.evaluate((btnText) => {
    const buttons = Array.from(document.querySelectorAll('button'));
    const target = buttons.find((b) => b.textContent.includes(btnText) && !b.disabled);
    if (target) {
      target.scrollIntoView({ behavior: 'instant', block: 'center' });
      target.click();
      return true;
    }
    return false;
  }, text);
  if (!success) {
    throw new Error(`Enabled button containing text "${text}" was not found.`);
  }
}

async function waitForText(page, text, timeout = 25000) {
  await page.waitForFunction(
    (targetText) => document.body && document.body.innerText.includes(targetText),
    { timeout },
    text
  );
}

async function loadFreshDemo(page) {
  await page.goto(APP_URL, { waitUntil: 'networkidle2' });
  await sleep(600);
  await clickButtonWithText(page, 'Use demo dataset');
  await waitForText(page, 'DATASET READY');
  await sleep(400);
}

async function main() {
  if (!fs.existsSync(SCREENSHOT_DIR)) {
    fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
  }

  console.log('Starting Browser QA Suite with Chrome at:', CHROME_PATH);
  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: true,
    args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--window-size=1920,1080'],
  });

  const page = await browser.newPage();
  page.on('pageerror', (err) => console.error('PAGE ERROR:', err.message));

  try {
    // -------------------------------------------------------------
    // Scenario 1: Official Demo Success (4 tools, 4 charts, brief)
    // -------------------------------------------------------------
    console.log('\n--- Scenario 1: Official Demo Success ---');
    await page.setViewport({ width: 1920, height: 1080 });
    await loadFreshDemo(page);

    console.log('Clicking official question button...');
    await clickButtonWithText(page, 'Why did our sales decline');
    await sleep(400);

    console.log('Clicking "Run verified investigation"...');
    await clickButtonWithText(page, 'Run verified investigation');
    await waitForText(page, '03 / INVESTIGATION RESULT');
    await sleep(3000); // Allow SVG charts to render completely

    // Full 1920x1080 page
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, '01_desktop_1920_success_full.png'), fullPage: true });
    console.log('Captured 01_desktop_1920_success_full.png');

    // Zoom on Tool Orchestration
    const timelineEl = await page.$('#investigation-path');
    if (timelineEl) {
      await timelineEl.screenshot({ path: path.join(SCREENSHOT_DIR, '01_tool_orchestration.png') });
      console.log('Captured 01_tool_orchestration.png');
    }

    // Zoom on Visual Evidence
    const visualGrid = await page.$('#visual-evidence');
    if (visualGrid) {
      await visualGrid.screenshot({ path: path.join(SCREENSHOT_DIR, '01_campaign_charts.png') });
      console.log('Captured 01_campaign_charts.png');
    }

    // Viewport: 1440x900
    await page.setViewport({ width: 1440, height: 900 });
    await sleep(400);
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, '01_desktop_1440_success.png') });
    console.log('Captured 01_desktop_1440_success.png');

    // Viewport: 1366x768
    await page.setViewport({ width: 1366, height: 768 });
    await sleep(400);
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, '01_desktop_1366_success.png') });
    console.log('Captured 01_desktop_1366_success.png');

    // Viewport: 390px Mobile
    await page.setViewport({ width: 390, height: 844 });
    await sleep(400);
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, '01_mobile_390_success.png'), fullPage: true });
    console.log('Captured 01_mobile_390_success.png');

    // -------------------------------------------------------------
    // Scenario 2: Replan Scenario (ANOVA -> Kruskal-Wallis)
    // -------------------------------------------------------------
    console.log('\n--- Scenario 2: Replan Scenario ---');
    await page.setViewport({ width: 1440, height: 900 });
    await loadFreshDemo(page);
    await setReactTextarea(page, 'replan: Are sales different across regions?');
    await clickButtonWithText(page, 'Run verified investigation');
    await waitForText(page, 'Derived from:');
    await sleep(2500);
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, '02_replan_scenario.png'), fullPage: true });
    console.log('Captured 02_replan_scenario.png');

    // -------------------------------------------------------------
    // Scenario 3: Unresolved Evidence (Causal Needs More Analysis, No fallback)
    // -------------------------------------------------------------
    console.log('\n--- Scenario 3: Unresolved Evidence ---');
    await loadFreshDemo(page);
    await setReactTextarea(page, 'unresolved: Assess campaign without counterfactual controls');
    await clickButtonWithText(page, 'Run verified investigation');
    await waitForText(page, 'No final Decision Brief was produced');
    await sleep(2500);
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, '03_unresolved_scenario.png'), fullPage: true });
    console.log('Captured 03_unresolved_scenario.png');

    // -------------------------------------------------------------
    // Scenario 4: Failed Tool Execution
    // -------------------------------------------------------------
    console.log('\n--- Scenario 4: Failed Tool Execution ---');
    await loadFreshDemo(page);
    await setReactTextarea(page, 'fail: Analyze non-existent column');
    await clickButtonWithText(page, 'Run verified investigation');
    await waitForText(page, 'This call failed during execution');
    await sleep(2500);
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, '04_failed_tool_scenario.png'), fullPage: true });
    console.log('Captured 04_failed_tool_scenario.png');

    // -------------------------------------------------------------
    // Scenario 5: One-Tool Investigation
    // -------------------------------------------------------------
    console.log('\n--- Scenario 5: One-Tool Investigation ---');
    await loadFreshDemo(page);
    await setReactTextarea(page, 'one-tool: What is sales trend?');
    await clickButtonWithText(page, 'Run verified investigation');
    await waitForText(page, '03 / INVESTIGATION RESULT');
    await sleep(2500);
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, '05_one_tool_scenario.png'), fullPage: true });
    console.log('Captured 05_one_tool_scenario.png');

    // -------------------------------------------------------------
    // Scenario 6: Zero Visualizations
    // -------------------------------------------------------------
    console.log('\n--- Scenario 6: Zero Visualizations ---');
    await loadFreshDemo(page);
    await setReactTextarea(page, 'no-chart: Analyze variance shift');
    await clickButtonWithText(page, 'Run verified investigation');
    await waitForText(page, 'The API returned no verified chart specifications');
    await sleep(2500);
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, '06_zero_visualizations.png'), fullPage: true });
    console.log('Captured 06_zero_visualizations.png');

    // -------------------------------------------------------------
    // Scenario 8: Invalid CSV Upload
    // -------------------------------------------------------------
    console.log('\n--- Scenario 8: Invalid CSV Upload ---');
    await page.goto(APP_URL, { waitUntil: 'networkidle2' });
    await sleep(600);
    await page.evaluate(() => {
      const input = document.querySelector('input[type="file"]');
      if (input) {
        const file = new File([''], 'empty_dataset.csv', { type: 'text/csv' });
        const dt = new DataTransfer();
        dt.items.add(file);
        input.files = dt.files;
        input.dispatchEvent(new Event('change', { bubbles: true }));
      }
    });
    await sleep(1000);
    await waitForText(page, 'This CSV file is empty');
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, '08_invalid_csv.png') });
    console.log('Captured 08_invalid_csv.png');

    console.log('\n======================================================');
    console.log('ALL BROWSER QA SCENARIOS SUCCESSFULLY TESTED & CAPTURED!');
    console.log('======================================================');
  } catch (err) {
    console.error('Fatal Browser QA Error:', err);
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'fatal_error.png'), fullPage: true });
  } finally {
    await browser.close();
  }
}

main();
