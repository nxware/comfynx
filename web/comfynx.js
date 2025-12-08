/**
 * https://docs.comfy.org/custom-nodes/js/javascript_overview
 */
import { app } from "../../scripts/app.js";
app.registerExtension({ 
	name: "comfynx_js",
	async setup() { 
		//alert("Setup complete!")
        console.log("NxComfy loaded");
	},
    async nodeCreated(node) {
        if (node.comfyClass !== "AddWatermark") return;
        console.log(node);
        window.mynode = node;
        // window.mynode.widgets[2].value = 45;
        // window.mynode.setDirtyCanvas(true, true)
        node.addWidget("button", "Call Server Route", null, async () => {
            console.log("Clicked");
            /*async*/app.api.fetchApi("/nx/btn", {
                method: "POST",
                body: JSON.stringify({ click: true })
            });
            app.extensionManager.toast.add({
                    severity: "success",
                    summary: "Route OK",
                    detail: "Server Route ausgeführt!",
                    life: 2500
            });
        });
        /*
        const container = node.dom.querySelector(".node-body") || node.dom;
        const btn = document.createElement("button");
        btn.textContent = "Call Server Route";
        btn.style.marginTop = "6px";
        btn.style.width = "100%";
        btn.addEventListener("click", async () => {
            try {
                const response = await app.api.fetchApi(
                "/custom/buttonnode/do_something",
                {
                    method: "POST",
                    body: JSON.stringify({ msg: "Button clicked!" })
                }
                );

                const result = await response.json();

                app.extensionManager.toast.add({
                    severity: "success",
                    summary: "Route OK",
                    detail: "Server Route ausgeführt!",
                    life: 2500
                });

                console.log("Route Response:", result);

            } catch (err) {
                app.extensionManager.toast.add({
                    severity: "error",
                    summary: "Fehler",
                    detail: err.toString(),
                    life: 3000
                });
            }
        });

        container.appendChild(btn);*/
    }
})