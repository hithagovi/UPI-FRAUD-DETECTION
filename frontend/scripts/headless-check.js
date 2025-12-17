const puppeteer = require('puppeteer');
const fs = require('fs');

(async () => {
  const url = process.argv[2] || 'http://localhost:3000';
  const outScreenshot = process.argv[3] || 'frontend/headless-check.png';

  console.log(`Opening ${url} ...`);
  const browser = await puppeteer.launch({ args: ['--no-sandbox','--disable-setuid-sandbox'] });
  const page = await browser.newPage();

  try {
    await page.goto(url, { waitUntil: 'networkidle2', timeout: 30000 });

    // wait for Dashboard header or Login form
    const hasDashboard = await page.$eval('h1', el => el.textContent).catch(() => null);

    const pathname = await page.evaluate(() => location.pathname + location.search + location.hash);

    await page.screenshot({ path: outScreenshot, fullPage: true });

    console.log('Current path:', pathname);
    if (hasDashboard && hasDashboard.toLowerCase().includes('dashboard')) {
      console.log('Detected Dashboard page. Auth bypass appears to be working.');
    } else {
      console.log('Dashboard header not detected. Page may be login or another route.');
    }
    console.log('Screenshot saved to', outScreenshot);
    await browser.close();
    process.exit(0);
  } catch (err) {
    console.error('Headless check failed:', err.message || err);
    try { await page.screenshot({ path: outScreenshot, fullPage: true }); console.log('Screenshot saved to', outScreenshot); } catch(e){}
    await browser.close();
    process.exit(2);
  }
})();
