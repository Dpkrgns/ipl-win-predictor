document.addEventListener("DOMContentLoaded", () => {
  const battingTeam = document.querySelector("#batting-team");
  const bowlingTeam = document.querySelector("#bowling-team");
  const venue = document.querySelector("#venue");
  const target = document.querySelector('[name="target_score"]');
  const score = document.querySelector('[name="score"]');
  const overs = document.querySelector("#overs-completed");
  const wickets = document.querySelector('[name="wickets_lost"]');

  const labels = {
    batting: document.querySelector("#batting-team-label"),
    bowling: document.querySelector("#bowling-team-label"),
    venue: document.querySelector("#venue-label"),
    runs: document.querySelector("#preview-runs"),
    balls: document.querySelector("#preview-balls"),
    rate: document.querySelector("#preview-rate"),
  };

  function parseLegalBalls(value) {
    const text = String(value || "0").trim();
    const parts = text.split(".");
    const overCount = Number.parseInt(parts[0], 10) || 0;
    const ballCount = Number.parseInt(parts[1], 10) || 0;
    return Math.max(0, overCount * 6 + Math.min(ballCount, 5));
  }

  function updatePreview() {
    const targetScore = Number(target.value) || 0;
    const currentScore = Number(score.value) || 0;
    const ballsUsed = parseLegalBalls(overs.value);
    const ballsLeft = Math.max(120 - ballsUsed, 0);
    const runsLeft = Math.max(targetScore - currentScore, 0);
    const requiredRate = ballsLeft ? (runsLeft * 6) / ballsLeft : 99;

    labels.runs.textContent = runsLeft;
    labels.balls.textContent = ballsLeft;
    labels.rate.textContent = requiredRate.toFixed(2);
    labels.batting.textContent = battingTeam.value;
    labels.bowling.textContent = bowlingTeam.value;
    labels.venue.textContent = venue.value;
  }

  [battingTeam, bowlingTeam, venue, target, score, overs, wickets].forEach((field) => {
    field.addEventListener("input", updatePreview);
    field.addEventListener("change", updatePreview);
  });

  document.querySelector("#swap-teams").addEventListener("click", () => {
    const currentBatting = battingTeam.value;
    battingTeam.value = bowlingTeam.value;
    bowlingTeam.value = currentBatting;
    updatePreview();
  });

  const scenarios = {
    powerplay: { target: 196, score: 52, overs: 6, wickets: 1 },
    "mid-chase": { target: 178, score: 92, overs: 11.4, wickets: 3 },
    "final-over": { target: 164, score: 148, overs: 19.2, wickets: 6 },
  };

  document.querySelectorAll("[data-scenario]").forEach((button) => {
    button.addEventListener("click", () => {
      const scenario = scenarios[button.dataset.scenario];
      target.value = scenario.target;
      score.value = scenario.score;
      overs.value = scenario.overs;
      wickets.value = scenario.wickets;
      updatePreview();
    });
  });

  updatePreview();

  renderProbabilityGraph(window.PROBABILITY_TIMELINE || []);
  setupSimulator();
});

function setupSimulator() {
  const score = document.querySelector("#sim-score");
  const overs = document.querySelector("#sim-overs");
  const wickets = document.querySelector("#sim-wickets");
  const target = document.querySelector('[name="target_score"]');
  const outputScore = document.querySelector("#sim-score-output");
  const outputOvers = document.querySelector("#sim-overs-output");
  const outputWickets = document.querySelector("#sim-wickets-output");
  const result = document.querySelector("#sim-result");
  const runButton = document.querySelector("#run-simulation");
  if (!score || !overs || !wickets || !runButton) return;

  function syncOutputs() {
    const scoreLimit = (Number(target.value) || 300) + 30;
    score.max = Math.min(330, scoreLimit);
    if (Number(score.value) > scoreLimit) score.value = scoreLimit;
    outputScore.textContent = score.value;
    outputOvers.textContent = formatCricketOvers(Number(overs.value));
    outputWickets.textContent = wickets.value;
  }

  function formatCricketOvers(legalBalls) {
    return `${Math.floor(legalBalls / 6)}.${legalBalls % 6}`;
  }

  [score, overs, wickets, target].forEach((field) => field.addEventListener("input", syncOutputs));
  syncOutputs();

  runButton.addEventListener("click", async () => {
    const form = document.querySelector(".panel");
    const payload = {
      batting_team: form.querySelector('[name="batting_team"]').value,
      bowling_team: form.querySelector('[name="bowling_team"]').value,
      venue: form.querySelector('[name="venue"]').value,
      target_score: Number(form.querySelector('[name="target_score"]').value),
      score: Number(score.value),
      overs_completed: formatCricketOvers(Number(overs.value)),
      wickets_lost: Number(wickets.value),
    };
    runButton.disabled = true;
    runButton.querySelector("span").textContent = "Calculating...";
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Prediction unavailable.");
      result.classList.add("is-ready");
      result.innerHTML = `<span class="sim-result-kicker">SIMULATED WIN PROBABILITY</span><strong>${data.win_probability}<sup>%</sup></strong><p>${data.analysis.outlook}. ${data.analysis.key_signal}.</p>`;
    } catch (error) {
      result.classList.remove("is-ready");
      result.innerHTML = `<span class="sim-result-kicker">SIMULATOR STATUS</span><strong>--</strong><p>${error.message}</p>`;
    } finally {
      runButton.disabled = false;
      runButton.querySelector("span").textContent = "Simulate state";
    }
  });
}

function renderProbabilityGraph(timeline) {
  const graph = document.querySelector("#probability-graph");
  if (!graph || !timeline.length) return;

  const width = 620;
  const chart = { left: 38, right: 606, top: 18, bottom: 208 };
  const x = (over) => chart.left + (over / 20) * (chart.right - chart.left);
  const y = (probability) => chart.bottom - (probability / 100) * (chart.bottom - chart.top);
  const points = timeline.map((point) => `${x(point.over)},${y(point.probability)}`);
  const line = `M ${points.join(" L ")}`;
  const area = `${line} L ${x(timeline[timeline.length - 1].over)},${chart.bottom} L ${x(timeline[0].over)},${chart.bottom} Z`;
  const current = timeline.find((point) => point.is_current) || timeline[timeline.length - 1];

  const horizontalGrid = [0, 25, 50, 75, 100]
    .map((value) => `<line class="chart-grid-line" x1="${chart.left}" y1="${y(value)}" x2="${chart.right}" y2="${y(value)}"></line><text class="chart-y-label" x="2" y="${y(value) + 4}">${value}%</text>`)
    .join("");
  const verticalGrid = [0, 5, 10, 15, 20]
    .map((value) => `<line class="chart-vertical-line" x1="${x(value)}" y1="${chart.top}" x2="${x(value)}" y2="${chart.bottom}"></line>`)
    .join("");

  graph.innerHTML = `${horizontalGrid}${verticalGrid}<path class="chart-area" d="${area}"></path><path class="chart-line" d="${line}"></path><line class="chart-now-line" x1="${x(current.over)}" y1="${chart.top}" x2="${x(current.over)}" y2="${chart.bottom}"></line><circle class="chart-now-ring" cx="${x(current.over)}" cy="${y(current.probability)}" r="7"></circle><circle class="chart-now-dot" cx="${x(current.over)}" cy="${y(current.probability)}" r="3"></circle><text class="chart-now-label" x="${Math.min(x(current.over) + 9, 548)}" y="${Math.max(y(current.probability) - 11, 18)}">NOW ${current.probability}%</text>`;
}
