import asyncio
from pathlib import Path
from aiohttp import web
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True,args=["--disable-dev-shm-usage"])
        pages = asyncio.Queue()
        for _ in range(12):
            page = await browser.new_page(viewport={"width":108,"height":240},java_script_enabled=False)
            await page.route("http*://**",lambda route:route.abort())
            pages.put_nowait(page)
        async def render(request):
            src,dst = await request.json()
            page = await pages.get()
            try:
                await page.goto(Path(src).resolve().as_uri(),wait_until="domcontentloaded")
                await page.screenshot(path=dst)
                return web.Response(text="OK")
            finally:
                pages.put_nowait(page)
        app = web.Application()
        app.router.add_post("/",render)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner,"127.0.0.1",18901,backlog=256).start()
        await asyncio.Event().wait()
asyncio.run(main())
