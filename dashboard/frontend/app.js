const healthState = document.getElementById("healthState");
const recordCount = document.getElementById("recordCount");
const rowsBody = document.getElementById("rows");
const refreshBtn = document.getElementById("refreshBtn");
const autoRefresh = document.getElementById("autoRefresh");

async function fetchJson(url) {
	const response = await fetch(url, {
		headers: { "Accept": "application/json" },
	});
	if (!response.ok) {
		throw new Error(`HTTP ${response.status}`);
	}
	return response.json();
}

function formatValue(value, digits = 6) {
	if (value === null || value === undefined) {
		return "-";
	}
	if (typeof value === "number") {
		return value.toFixed(digits);
	}
	return value;
}

function renderRows(items) {
	rowsBody.innerHTML = "";

	if (!items.length) {
		const tr = document.createElement("tr");
		const td = document.createElement("td");
		td.colSpan = 9;
		td.textContent = "No violation records available.";
		tr.appendChild(td);
		rowsBody.appendChild(tr);
		return;
	}

	for (const row of items) {
		const tr = document.createElement("tr");

		tr.innerHTML = `
			<td>${row.id ?? "-"}</td>
			<td>${row.license_plate ?? "-"}</td>
			<td>${row.first_seen ?? "-"}</td>
			<td>${row.last_seen ?? "-"}</td>
			<td>${formatValue(row.duration_sec, 2)}</td>
			<td>${formatValue(row.latitude, 6)}</td>
			<td>${formatValue(row.longitude, 6)}</td>
			<td>${row.video_source ?? "-"}</td>
			<td>${formatValue(row.confidence, 3)}</td>
		`;

		rowsBody.appendChild(tr);
	}
}

async function refreshHealth() {
	try {
		const health = await fetchJson("/api/health");
		healthState.textContent = health.ok ? "Online" : "Offline";
	} catch (err) {
		healthState.textContent = `Error: ${err.message}`;
	}
}

async function refreshViolations() {
	try {
		const payload = await fetchJson("/api/violations?limit=100");
		recordCount.textContent = payload.count ?? 0;
		renderRows(payload.items ?? []);
	} catch (err) {
		recordCount.textContent = "0";
		rowsBody.innerHTML = `<tr><td colspan="9">Failed to load data: ${err.message}</td></tr>`;
	}
}

async function refreshAll() {
	await Promise.all([refreshHealth(), refreshViolations()]);
}

refreshBtn.addEventListener("click", refreshAll);

setInterval(() => {
	if (autoRefresh.checked) {
		refreshAll();
	}
}, 3000);

refreshAll();
