

def setup_routes():
    try:
        from aiohttp import web
        from server import PromptServer

        async def do_something(request):
            """
            Wird vom Button per fetch() ausgelöst.
            """
            data = await request.json()
            print("Custom Route Triggered! Payload:", data)
            PromptServer.instance.send_sync(
                "custom.button.server_event",
                {"detail": f"Server hat die Route ausgeführt: {data}"}
            )
            return web.json_response({"status": "ok", "echo": data})
        
        async def site_prompt(request):
            #name = request.match_info.get('name', "Anonymous")
            text = "Test result"
            return web.Response(text=text)

        PromptServer.instance.app.router.add_post("/custom/buttonnode/do_something", do_something)
        PromptServer.instance.app.router.add_get("/nx/prompt", site_prompt)
        print("[ButtonNodeWithRoute] Custom route registered.")
    except Exception as e:
        print("Fehler beim Registrieren der Route:", e)