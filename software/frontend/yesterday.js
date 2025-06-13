document.addEventListener('DOMContentLoaded', () => {
  const daySelect   = document.getElementById('daySelect');
  const loadBtn     = document.getElementById('loadDayBtn');
  const downloadBtn = document.getElementById('downloadBtn');

//   // Fetch available days, this one loads ALL DAY, so slow
//  fetch('/status/days')
//   .then(r => r.json())
//   .then(days => {
//     console.log("Available days:", days);     // ← add this line
//     daySelect.innerHTML = days
//       .map(d => `<option value="${d}">${d}</option>`)
//       .join('');
//   })
//   .catch(err => {
//     console.error('Failed to load days:', err);
//     daySelect.innerHTML = '<option>Error loading days</option>';
//   });

// Fetch available days (changed to last 100 days as load all so sluggish)
  fetch('/status/history/days?limit=100')
  .then(r => r.json())
  .then(days => {
    // days is already at most 100, newest first
    daySelect.innerHTML = days
      .map(d => `<option value="${d}">${d}</option>`)
      .join('');
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

  const priceChart  = makeChart(
    document.getElementById('priceChart').getContext('2d'),
    'Buy Price', 'green', 'Price ($)'
  );
  priceChart.data.datasets.push({
    label: 'Sell Price',
    data: [],
    borderColor: 'red',
    fill: false,
    tension: 0.1
  });

  const demandChart = makeChart(
    document.getElementById('demandChart').getContext('2d'),
    'Demand', 'blue', 'Demand (W)'
  );

  // Load data for the selected day
  loadBtn.addEventListener('click', () => {
    const day = daySelect.value;
    if (!day) return;
    fetch(`/status/history?day=${day}`)
      .then(r => r.json())
      .then(records => {
        // clear existing charts
        irrChart.data.labels = [];
        irrChart.data.datasets[0].data = [];
        priceChart.data.labels = [];
        priceChart.data.datasets.forEach(ds => ds.data = []);
        demandChart.data.labels = [];
        demandChart.data.datasets[0].data = [];

        // populate
        for (const rec of records) {
          const t = rec.tick;
          irrChart.data.labels.push(rec.tick);
          irrChart.data.datasets[0].data.push(rec.irradiance);

          priceChart.data.labels.push(t);
          priceChart.data.datasets[0].data.push(rec.buy_price);
          priceChart.data.datasets[1].data.push(rec.sell_price);

          demandChart.data.labels.push(t);
          demandChart.data.datasets[0].data.push(rec.demand);
        }
        irrChart.update();
        priceChart.update();
        demandChart.update();

        // Enable download
        downloadBtn.disabled = false;
      })
      .catch(err => {
        console.error('Failed to load history:', err);
      });
  });

  const irrChart = makeChart(
  document.getElementById('irradianceChart').getContext('2d'),
  'Irradiance', 'orange', 'Irradiance (%)'
  );

  // Download TXT for the selected day
  downloadBtn.addEventListener('click', () => {
    const day = daySelect.value;
    if (!day) return;
    window.location = `/download/history?day=${day}`;
  });

  const downloadAllBtn = document.getElementById('downloadAllBtn');

  downloadAllBtn.addEventListener('click', () => {
    window.location = '/download/all';
  });

  // daySelect.innerHTML = days
  // .map(d => `<option value="${d}">Virtual Day ${d}</option>`)
  // .join('');
  // try converting day into real dates

});
