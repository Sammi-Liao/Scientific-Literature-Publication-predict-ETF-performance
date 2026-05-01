let horizonChart;
let publicationChart;
let etfChart;

const industrySelect = document.getElementById("industry-select");
const statusEl = document.getElementById("status");
const cardsEl = document.getElementById("summary-cards");
const modelMetricsEl = document.getElementById("model-metrics");

const tickerLinks = {
  IHI: "https://www.ishares.com/us/products/239519/ishares-us-medical-devices-etf",
  XHE: "https://www.ssga.com/us/en/intermediary/etfs/spdr-sp-health-care-equipment-etf-xhe",
  IBB: "https://www.ishares.com/us/products/239699/ishares-biotechnology-etf",
  XBI: "https://www.ssga.com/us/en/intermediary/etfs/spdr-sp-biotech-etf-xbi",
};

const interactiveTableState = new WeakMap();

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "NA";
  return Number(value).toFixed(digits);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error || `Request failed: ${response.status}`);
  }
  return response.json();
}

function tickerTaskBreakdown(comparison, task, tickerOrder) {
  return tickerOrder.map((ticker) => {
    const rows = comparison.filter((row) => row.ticker === ticker && row.task === task);
    const lifts = rows.map((row) => Number(row.lit_helps)).filter((value) => !Number.isNaN(value));
    const nBetter = lifts.filter((value) => value > 0).length;
    const pctBetter = lifts.length ? (nBetter / lifts.length) * 100 : null;
    const avgLift = lifts.length ? lifts.reduce((sum, value) => sum + value, 0) / lifts.length : null;
    return { ticker, pctBetter, avgLift };
  });
}

function renderBreakdown(lines) {
  return lines
    .map(
      (line) => `
        <div class="breakdown-row">
          <span>${line.ticker}</span>
          <span>${fmt(line.pctBetter, 1)}% better · avg lift ${fmt(line.avgLift, 4)}</span>
        </div>`
    )
    .join("");
}

function renderTickerLinks(tickers) {
  return tickers
    .map((ticker) => {
      const url = tickerLinks[ticker];
      if (!url) return ticker;
      return `<a class="ticker-link" href="${url}" target="_blank" rel="noopener noreferrer">${ticker}</a>`;
    })
    .join(" / ");
}

function displayValue(row, col) {
  const value = col.format ? col.format(row[col.key]) : row[col.key];
  return value ?? "";
}

function mean(values) {
  const nums = values.map(Number).filter((value) => Number.isFinite(value));
  if (!nums.length) return null;
  return nums.reduce((sum, value) => sum + value, 0) / nums.length;
}

function horizonMeanScores(comparison, task) {
  const rows = comparison.filter(
    (row) => row.task === task && Number(row.horizon_weeks) >= 5 && Number(row.horizon_weeks) <= 8
  );
  return {
    baseline: mean(rows.map((row) => row.baseline_score)),
    literature: mean(rows.map((row) => row.lit_score)),
  };
}

function renderPublicationChart(literatureRows) {
  const rows = literatureRows
    .filter((row) => row.date)
    .sort((a, b) => String(a.date).localeCompare(String(b.date)));
  const labels = rows.map((row) => row.date);
  const data = rows.map((row) => Number(row.daily_pubs_30d_avg ?? row.daily_pubs));

  if (publicationChart) publicationChart.destroy();
  publicationChart = new Chart(document.getElementById("publication-chart"), {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "30-day avg publications",
          data,
          borderColor: "#2f6fed",
          backgroundColor: "rgba(47,111,237,0.12)",
          pointRadius: 0,
          tension: 0.25,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { position: "bottom" } },
      scales: {
        x: { ticks: { maxTicksLimit: 8, maxRotation: 0 } },
        y: { title: { display: true, text: "Publications" } },
      },
    },
  });
}

function renderEtfChart(marketRows, tickers) {
  const rowsByTicker = new Map(tickers.map((ticker) => [ticker, []]));
  marketRows.forEach((row) => {
    if (rowsByTicker.has(row.ticker) && row.date && row.adj_close !== null) {
      rowsByTicker.get(row.ticker).push(row);
    }
  });

  const labels = [
    ...new Set(
      marketRows
        .filter((row) => tickers.includes(row.ticker) && row.date)
        .map((row) => row.date)
    ),
  ].sort();
  const colors = ["#C9C8C5", "#10a37f"];
  const datasets = tickers.map((ticker, idx) => {
    const tickerRows = rowsByTicker.get(ticker).sort((a, b) => String(a.date).localeCompare(String(b.date)));
    const priceByDate = Object.fromEntries(tickerRows.map((row) => [row.date, Number(row.adj_close)]));
    return {
      label: ticker,
      data: labels.map((date) => {
        const price = priceByDate[date];
        return Number.isFinite(price) ? price : null;
      }),
      borderColor: colors[idx % colors.length],
      backgroundColor: idx === 0 ? "rgba(193,157,242,0.18)" : "rgba(16,163,127,0.12)",
      pointRadius: 0,
      tension: 0.25,
    };
  });

  if (etfChart) etfChart.destroy();
  etfChart = new Chart(document.getElementById("etf-chart"), {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      plugins: { legend: { position: "bottom" } },
      scales: {
        x: { ticks: { maxTicksLimit: 8, maxRotation: 0 } },
        y: { title: { display: true, text: "Adjusted close" } },
      },
    },
  });
}

function renderModelMetrics(comparison) {
  const classification = horizonMeanScores(comparison, "classification");
  const regression = horizonMeanScores(comparison, "regression");
  modelMetricsEl.innerHTML = `
    <article class="metric-card">
      <div>
        <div class="label">Mean Accuracy,<br>Horizons 5-8 Weeks</div>
        <div class="metric-note">Classification models, higher is better</div>
      </div>
      <div class="metric-pair">
        <div>
          <span>Baseline</span>
          <strong>${fmt(classification.baseline, 4)}</strong>
        </div>
        <div>
          <span>Literature</span>
          <strong>${fmt(classification.literature, 4)}</strong>
        </div>
      </div>
    </article>
    <article class="metric-card">
      <div>
        <div class="label">Mean RMSE,<br>Horizons 5-8 Weeks</div>
        <div class="metric-note">Regression models, lower is better</div>
      </div>
      <div class="metric-pair">
        <div>
          <span>Baseline</span>
          <strong>${fmt(regression.baseline, 4)}</strong>
        </div>
        <div>
          <span>Literature</span>
          <strong>${fmt(regression.literature, 4)}</strong>
        </div>
      </div>
    </article>
  `;
}

function renderCards(summary, comparison) {
  const byTask = Object.fromEntries(summary.overall.map((row) => [row.task, row]));
  const classification = byTask.classification || {};
  const regression = byTask.regression || {};
  const classificationBreakdown = tickerTaskBreakdown(comparison, "classification", summary.tickers);
  const regressionBreakdown = tickerTaskBreakdown(comparison, "regression", summary.tickers);
  cardsEl.innerHTML = `
    <article class="card">
      <div class="label">Industry</div>
      <div class="value">${summary.label}</div>
    </article>
    <article class="card">
      <div class="label">Tickers</div>
      <div class="value">${renderTickerLinks(summary.tickers)}</div>
    </article>
    <article class="card">
      <div class="label">% Classification Lit Models Better</div>
      <div class="value">${fmt(classification.pct_lit_better, 1)}%</div>
      <div class="breakdown">${renderBreakdown(classificationBreakdown)}</div>
    </article>
    <article class="card">
      <div class="label">% Regression Lit Models Better</div>
      <div class="value">${fmt(regression.pct_lit_better, 1)}%</div>
      <div class="breakdown">${renderBreakdown(regressionBreakdown)}</div>
    </article>
  `;
}

function renderHorizonChart(summary) {
  const labels = [...new Set(summary.by_horizon.map((row) => row.horizon_weeks))].sort((a, b) => a - b);
  const tasks = ["classification", "regression"];
  const datasets = tasks.map((task, idx) => ({
    label: task === "regression" ? "Regression RMSE reduction" : "Classification accuracy lift",
    data: labels.map((horizon) => {
      const row = summary.by_horizon.find((item) => item.task === task && item.horizon_weeks === horizon);
      return row ? row.avg_lit_help : null;
    }),
    borderColor: idx === 0 ? "#2f6fed" : "#10a37f",
    backgroundColor: idx === 0 ? "rgba(47,111,237,0.15)" : "rgba(16,163,127,0.15)",
    tension: 0.35,
  }));

  if (horizonChart) horizonChart.destroy();
  horizonChart = new Chart(document.getElementById("horizon-chart"), {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      plugins: { legend: { position: "bottom" } },
      scales: { x: { title: { display: true, text: "Horizon (weeks)" } } },
    },
  });
}

function renderTable(el, rows, columns, limit = 12) {
  const data = rows.slice(0, limit);
  if (!data.length) {
    el.innerHTML = "<tbody><tr><td>No cached rows yet.</td></tr></tbody>";
    return;
  }
  el.innerHTML = `
    <thead><tr>${columns.map((col) => `<th>${col.label}</th>`).join("")}</tr></thead>
    <tbody>
      ${data
        .map(
          (row) => `
        <tr>${columns
          .map((col) => `<td>${escapeHtml(displayValue(row, col))}</td>`)
          .join("")}</tr>`
        )
        .join("")}
    </tbody>
  `;
}

function compareValues(a, b) {
  const aNumber = Number(a);
  const bNumber = Number(b);
  if (Number.isFinite(aNumber) && Number.isFinite(bNumber)) {
    return aNumber - bNumber;
  }
  return String(a ?? "").localeCompare(String(b ?? ""), undefined, {
    numeric: true,
    sensitivity: "base",
  });
}

function uniqueFilterOptions(rows, col) {
  return [...new Set(rows.map((row) => displayValue(row, col)).filter((value) => value !== ""))]
    .sort(compareValues);
}

function renderFilterControl(rows, col, value) {
  if (col.filter === false) return "";
  if (col.filter === "select") {
    const options = uniqueFilterOptions(rows, col)
      .map((option) => `<option value="${escapeHtml(option)}">${escapeHtml(option)}</option>`)
      .join("");
    return `
      <select class="column-filter" data-filter-key="${col.key}">
        <option value="">All</option>
        ${options}
      </select>`;
  }
  return `
    <input
      class="column-filter"
      data-filter-key="${col.key}"
      placeholder="Filter"
      value="${escapeHtml(value)}"
    >`;
}

function renderInteractiveTable(el, rows, columns, focusKey = null) {
  if (!interactiveTableState.has(el)) {
    const defaultSort = columns.find((col) => col.defaultSort);
    interactiveTableState.set(el, {
      filters: {},
      sortKey: defaultSort ? defaultSort.key : null,
      sortDirection: defaultSort?.defaultSort || "asc",
    });
  }

  const state = interactiveTableState.get(el);
  const filteredRows = rows.filter((row) =>
    columns.every((col) => {
      if (col.filter === false) return true;
      const filter = (state.filters[col.key] || "").trim();
      if (!filter) return true;
      const value = String(displayValue(row, col));
      if (col.filter === "select") return value === filter;
      return value.toLowerCase().includes(filter.toLowerCase());
    })
  );
  const sortedRows = [...filteredRows].sort((a, b) => {
    if (!state.sortKey) return 0;
    const direction = state.sortDirection === "asc" ? 1 : -1;
    return compareValues(a[state.sortKey], b[state.sortKey]) * direction;
  });

  el.classList.add("interactive-table");
  el.innerHTML = `
    <caption class="table-caption">Showing ${sortedRows.length} of ${rows.length} rows</caption>
    <thead>
      <tr>
        ${columns
          .map((col) => {
            const isSorted = state.sortKey === col.key;
            const arrow = isSorted ? (state.sortDirection === "asc" ? " ↑" : " ↓") : "";
            return `<th><button class="sort-button" type="button" data-sort-key="${col.key}">${escapeHtml(
              col.label
            )}${arrow}</button></th>`;
          })
          .join("")}
      </tr>
      <tr class="filter-row">
        ${columns
          .map(
            (col) => `
          <th>
            ${renderFilterControl(rows, col, state.filters[col.key] || "")}
          </th>`
          )
          .join("")}
      </tr>
    </thead>
    <tbody>
      ${sortedRows
        .map(
          (row) => `
        <tr>${columns.map((col) => `<td>${escapeHtml(displayValue(row, col))}</td>`).join("")}</tr>`
        )
        .join("")}
    </tbody>
  `;

  el.querySelectorAll("[data-sort-key]").forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.dataset.sortKey;
      if (state.sortKey === key) {
        state.sortDirection = state.sortDirection === "asc" ? "desc" : "asc";
      } else {
        state.sortKey = key;
        state.sortDirection = "asc";
      }
      renderInteractiveTable(el, rows, columns);
    });
  });

  el.querySelectorAll("[data-filter-key]").forEach((control) => {
    const key = control.dataset.filterKey;
    control.value = state.filters[key] || "";
    control.addEventListener("input", () => {
      state.filters[key] = control.value;
      renderInteractiveTable(el, rows, columns, key);
    });
    control.addEventListener("change", () => {
      state.filters[key] = control.value;
      renderInteractiveTable(el, rows, columns, key);
    });
  });

  if (focusKey) {
    const input = el.querySelector(`[data-filter-key="${focusKey}"]`);
    if (input) {
      input.focus();
      if (input.setSelectionRange) {
        input.setSelectionRange(input.value.length, input.value.length);
      }
    }
  }
}

async function loadIndustry(industry) {
  statusEl.textContent = "";
  try {
    const [summary, granger, comparison, literature, market] = await Promise.all([
      fetchJson(`/api/summary?industry=${industry}`),
      fetchJson(`/api/granger?industry=${industry}&significant=true`),
      fetchJson(`/api/comparison?industry=${industry}`),
      fetchJson(`/api/literature?industry=${industry}`),
      fetchJson(`/api/market?industry=${industry}`),
    ]);
    renderPublicationChart(literature);
    renderEtfChart(market, summary.tickers);
    renderCards(summary, comparison);
    renderModelMetrics(comparison);
    renderHorizonChart(summary);
    renderTable(document.getElementById("granger-table"), granger, [
      { key: "ticker", label: "Ticker" },
      { key: "feature", label: "Feature" },
      { key: "best_weekly_lag", label: "Lag (days)" },
      { key: "min_weekly_pvalue", label: "p", format: (v) => fmt(v, 4) },
      { key: "weekly_fdr_pvalue", label: "FDR", format: (v) => fmt(v, 4) },
    ]);
    renderInteractiveTable(document.getElementById("comparison-table"), comparison, [
      { key: "ticker", label: "Ticker", filter: "select" },
      { key: "horizon_weeks", label: "Pred Horizon (Weeks)", filter: "select" },
      { key: "train_months", label: "Train (Months)", filter: "select" },
      { key: "task", label: "Task", filter: "select" },
      { key: "baseline_best_model", label: "Baseline", filter: "select" },
      { key: "baseline_score", label: "Baseline Score", format: (v) => fmt(v, 4), filter: false },
      { key: "lit_best_model", label: "Literature", filter: "select" },
      { key: "lit_score", label: "Literature Score", format: (v) => fmt(v, 4), filter: false },
      { key: "lit_helps", label: "Lift", format: (v) => fmt(v, 4), filter: false, defaultSort: "desc" },
    ]);
    statusEl.textContent = "";
  } catch (error) {
    statusEl.textContent = error.message;
    cardsEl.innerHTML = "";
    modelMetricsEl.innerHTML = "";
  }
}

industrySelect.addEventListener("change", (event) => loadIndustry(event.target.value));
loadIndustry(industrySelect.value);

