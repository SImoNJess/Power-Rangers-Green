document.addEventListener('DOMContentLoaded', () => {
  const daySelect   = document.getElementById('daySelect');
  const loadBtn     = document.getElementById('loadDayBtn');
  const downloadBtn = document.getElementById('downloadBtn');

  // Fetch available days
  fetch('/status/days')
    .then(r => r.json())
    .then(days => {
      daySelect.innerHTML = days
        .map(d => `<option value="${d}">${d}</option>`)
        .join('');
    })
    .catch(err => {
      console.error('Failed to load days:', err);
      daySelect.innerHTML = '<option>Error loading days</option>';
    });

  // Helper to build a chart
  function makeChart(ctx, label, color, yLabel) {
    return new Chart(ctx, {
      type: 'line',
      data: { labels: [], datasets: [{ label, data: [], borderColor: color, fill: false, tension: 0.1 }] },
      options: {
        scales: {
          x: { title: { display: true, text: 'Tick' } },
          y: { title: { display: true, text: yLabel } }
        },
        responsive: true,
        animation: false
      }
    });
  }

  // Instantiate empty charts
  const priceChart  = makeChart(
    document.getElementById('priceChart').getContext('2d'),
    'Buy Price', 'green', 'Price (p)'
  );
  // Add Sell Price to the same chart
  priceChart.data.datasets.push({
    label: 'Sell Price',
    data: [],
    borderColor: 'red',
    fill: false,
    tension: 0.1
  });

  const demandChart = makeChart(
    document.getElementById('demandChart').getContext('2d'),
    'Demand', 'blue', 'Demand (kW)'
  );

  // Load data for the selected day
  loadBtn.addEventListener('click', () => {
    const day = daySelect.value;
    if (!day) return;
    fetch(`/status/history?day=${day}`)
      .then(r => r.json())
      .then(records => {
        // Clear existing
        priceChart.data.labels = [];
        priceChart.data.datasets.forEach(ds => ds.data = []);
        demandChart.data.labels = [];
        demandChart.data.datasets[0].data = [];

        // Populate
        for (const rec of records) {
          const t = rec.tick;
          priceChart.data.labels.push(t);
          priceChart.data.datasets[0].data.push(rec.buy_price);
          priceChart.data.datasets[1].data.push(rec.sell_price);

          demandChart.data.labels.push(t);
          demandChart.data.datasets[0].data.push(rec.demand);
        }
        priceChart.update();
        demandChart.update();

        // Enable download
        downloadBtn.disabled = false;
      })
      .catch(err => {
        console.error('Failed to load history:', err);
      });
  });

  // Download TXT for the selected day
  downloadBtn.addEventListener('click', () => {
    const day = daySelect.value;
    if (!day) return;
    window.location = `/download/history?day=${day}`;
  });
});
