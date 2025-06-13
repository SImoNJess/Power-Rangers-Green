// dashboard.js

document.addEventListener('DOMContentLoaded', () => {
  // ── Element refs ───────────────────────────────────────────
  const tickEl   = document.getElementById('tick-value');
  const dayEl    = document.getElementById('day-value');
  const buyEl    = document.getElementById('buy-price-value');
  const sellEl   = document.getElementById('sell-price-value');
  const costEl   = document.getElementById('cost-value');

  // ── Extracted: update the action card (buy/sell/hold) ────────
  function updateActionCard(action_live) {
  const halfAction = action_live / 2;
  const card = document.getElementById('policy-action-value');
  let text, color;

  if (halfAction > 0) {
    text  = `${halfAction.toFixed(2)} ▶ STORE`;
    color = 'rgb(54,162,235)';
  } else if (halfAction < 0) {
    text  = `${halfAction.toFixed(2)} ▼ EXTRACT`;
    color = 'rgb(255,99,132)';
  } else {
    text  = 'HOLD';
    color = 'gray';
  }

  card.textContent = text;
  card.style.color = color;
}



  const d1sM = document.getElementById('d1-start-modal'),
        d1eM = document.getElementById('d1-end-modal'),
        d1En = document.getElementById('d1-energy'),
        d2sM = document.getElementById('d2-start-modal'),
        d2eM = document.getElementById('d2-end-modal'),
        d2En = document.getElementById('d2-energy'),
        d3sM = document.getElementById('d3-start-modal'),
        d3eM = document.getElementById('d3-end-modal'),
        d3En = document.getElementById('d3-energy');

  function animateValue(el, start, end, floatMode = false, duration = 800) {
    const range = end - start;
    let startTime = null;
    function step(ts) {
      if (!startTime) startTime = ts;
      const progress = Math.min((ts - startTime) / duration, 1);
      const val = start + range * progress;
      el.textContent = floatMode ? val.toFixed(2) : Math.floor(val);
      if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  // ── Deferrable modal fetch ───────────────────────────────────
  async function fetchDeferrables() {
    try {
      const loads = await fetch('/status/deferrables').then(r => r.json());
      loads.forEach((l, i) => {
        const start  = l.start_tick;
        const end    = l.end_tick;
        const joules = parseFloat(l.energy).toFixed(2);
        if (i === 0) { d1sM.textContent = start; d1eM.textContent = end; d1En.textContent = joules; }
        if (i === 1) { d2sM.textContent = start; d2eM.textContent = end; d2En.textContent = joules; }
        if (i === 2) { d3sM.textContent = start; d3eM.textContent = end; d3En.textContent = joules; }
      });
    } catch (err) {
      console.error('Failed to fetch deferrables', err);
    }
  }
  fetchDeferrables();
  setInterval(fetchDeferrables, 5000);

  // ── Chart boilerplate ──────────────────────────────────────
  const LOOK_BACK   = 60;
  const NO_DATA     = NaN;
  const staticTicks = Array.from({ length: LOOK_BACK }, (_, i) => i);

  function makeFixedChart(ctx, labelA, colorA, labelP, colorP, yLabel) {
    return new Chart(ctx, {
      type: 'line',
      data: {
        labels: staticTicks,
        datasets: [
          { label: labelA, data: Array(LOOK_BACK).fill(NO_DATA), borderColor: colorA, tension: 0.1, spanGaps: true, fill: false },
          { label: labelP, data: Array(LOOK_BACK).fill(NO_DATA), borderColor: colorP, borderDash: [5,5], tension: 0.1, spanGaps: true, fill: false }
        ]
      },
      options: {
        responsive: true, animation: false,
        interaction: { mode:'index', intersect:false },
        scales: {
          x: { title:{ display:true, text:'Tick' } },
          y: { title:{ display:true, text:yLabel }, beginAtZero:true }
        }
      }
    });
  }

  const sunChart     = makeFixedChart(
    document.getElementById('sunChart').getContext('2d'),
    'Actual Sun','orange','Predicted Sun','yellow','Irradiance'
  );
  const demandChart = makeFixedChart(
    document.getElementById('demandChart').getContext('2d'),
    'Actual Demand','blue','Predicted Demand','lightblue','Demand (W)'
  );
  const buyChart    = makeFixedChart(
    document.getElementById('buyChart').getContext('2d'),
    'Actual Buy','green','Predicted Buy','lightgreen','Buy Price ($)'
  );
  const sellChart   = makeFixedChart(
    document.getElementById('sellChart').getContext('2d'),
    'Actual Sell','red','Predicted Sell','pink','Sell Price ($)'
  );
  const pvChart     = new Chart(
    document.getElementById('pvChart').getContext('2d'),
    {
      type:'line',
      data:{ labels:staticTicks, datasets:[{
        label:'PV Power (W)',
        data:Array(LOOK_BACK).fill(NO_DATA),
        borderColor:'purple', tension:0.1, spanGaps:true, fill:false
      }]},
      options:{ responsive:true, animation:false,
        scales:{ x:{ title:{ display:true,text:'Tick' } }, y:{ title:{ display:true,text:'Power (W)' }, beginAtZero:true } }
      }
    }
  );



  // ── Policy Forecast ────────────────────────────────────────
  const policyCtx = document.getElementById('policyChart').getContext('2d');
  const futureActionChart = new Chart(policyCtx, {
    type:'bar',
    data:{ labels:staticTicks, datasets:[{
      label:'ΔAction',
      data:Array(LOOK_BACK).fill(0),
      backgroundColor:c=>{
        const v=c.dataset.data[c.dataIndex];
        return v<0?'rgba(255,99,132,0.6)':'rgba(54,162,235,0.6)';
      },
      borderColor:c=>{
        const v=c.dataset.data[c.dataIndex];
        return v<0?'rgba(255,99,132,1)':'rgba(54,162,235,1)';
      },
      borderWidth:1
    }]},
    options:{ responsive:true, animation:false,
      scales:{ x:{ title:{ display:true,text:'Tick Ahead' } }, y:{ title:{ display:true,text:'Action (Δ)' }, beginAtZero:false } }
    }
  });

  // ── Deferrable 1/2/3 (original BAR) ───────────────────────
  const defChart1 = new Chart(
    document.getElementById('deferralChart1').getContext('2d'),
    {
      type: 'bar',
      data: {
        labels: staticTicks,
        datasets: [{
          label: 'Deferrable 1 (J)',
          data: Array(LOOK_BACK).fill(0),
          backgroundColor: 'rgba(0,128,128,0.6)',
          borderColor:     'rgba(0,128,128,1)',
          borderWidth: 1
        }]
      },
      options: {
        responsive: true, animation: false,
        scales:{ x:{ title:{ display:true,text:'Tick' } }, y:{ title:{ display:true,text:'Energy (J)' }, beginAtZero:true } }
      }
    }
  );
  const defChart2 = new Chart(
    document.getElementById('deferralChart2').getContext('2d'),
    {
      type: 'bar',
      data: {
        labels: staticTicks,
        datasets: [{
          label: 'Deferrable 2 (J)',
          data: Array(LOOK_BACK).fill(0),
          backgroundColor: 'rgba(0,128,128,0.6)',
          borderColor:     'rgba(0,128,128,1)',
          borderWidth: 1
        }]
      },
      options: {
        responsive: true, animation: false,
        scales:{ x:{ title:{ display:true,text:'Tick' } }, y:{ title:{ display:true,text:'Energy (J)' }, beginAtZero:true } }
      }
    }
  );
  const defChart3 = new Chart(
    document.getElementById('deferralChart3').getContext('2d'),
    {
      type: 'bar',
      data: {
        labels: staticTicks,
        datasets: [{
          label: 'Deferrable 3 (J)',
          data: Array(LOOK_BACK).fill(0),
          backgroundColor: 'rgba(0,128,128,0.6)',
          borderColor:     'rgba(0,128,128,1)',
          borderWidth: 1
        }]
      },
      options: {
        responsive: true, animation: false,
        scales:{ x:{ title:{ display:true,text:'Tick' } }, y:{ title:{ display:true,text:'Energy (J)' }, beginAtZero:true } }
      }
    }
  );

  // Total demand chart
  const combinedChart = new Chart(
  document.getElementById('combinedChart').getContext('2d'),
  {
    type: 'line',
    data: {
      labels: staticTicks,
      datasets: [
        {
          label: 'Raw Total Demand (W)',
          data: Array(LOOK_BACK).fill(NO_DATA),
          borderColor: 'magenta',
          tension: 0.1,
          spanGaps: true,
          fill: false
        },
        {
          label: 'Modified Total Demand (W)',
          data: Array(LOOK_BACK).fill(NO_DATA),
          borderColor: 'cyan',
          borderDash: [5,5],
          tension: 0.1,
          spanGaps: true,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      animation: false,
      scales: {
        x: { title: { display: true, text: 'Tick' } },
        y: {
          title: { display: true, text: 'Power (W)' },
          beginAtZero: true,
          suggestedMax: 4.5
        }
      }
    }
  }
);




  // ── Shared state & initialization ──────────────────────────
  let prevTick = null;
  let globalHistory    = [],
      globalFutureSun  = [],
      globalFutureDem  = [],
      globalFutureBuy  = [],
      globalFutureSell = [],
      globalStartTick  = 0;

  let globalPolicyActions = [];
  async function initCharts() {
    try {
      // 1) full grid/history
      const { day, tick, history, cost, pv_power, connected_pv } =
        await fetch('/status/grid').then(r => r.json());
      globalHistory = history;

      animateValue(tickEl, prevTick||0, tick, false);
      animateValue(dayEl, parseInt(dayEl.textContent)||0, day, false);
      const last = history[history.length-1]||{};
      animateValue(buyEl, parseFloat(buyEl.textContent)||0, last.price_buy||0, true);
      animateValue(sellEl, parseFloat(sellEl.textContent)||0, last.price_sell||0, true);

      history.forEach(pt => {
        if(pt.tick<LOOK_BACK){
          sunChart.data.datasets[0].data[pt.tick]=pt.sun;
          demandChart.data.datasets[0].data[pt.tick]=pt.demand;
          buyChart.data.datasets[0].data[pt.tick]=pt.price_buy;
          sellChart.data.datasets[0].data[pt.tick]=pt.price_sell;
          pvChart.data.datasets[0].data[pt.tick]= pt.pv_power ?? NaN;
          combinedChart.data.datasets[0].data[pt.tick] = pt.combined;
        }
      });
      let carry = 0;
    history.forEach(pt => {
      if (pt.tick < LOOK_BACK) {
        const raw    = pt.combined + carry;
        const capped = Math.min(raw, 4);
        // dataset[1] --> “Modified Total Demand”
        combinedChart.data.datasets[1].data[pt.tick] = capped;
        carry = raw - capped;
      }
    });
      [sunChart,demandChart,buyChart,sellChart,pvChart,combinedChart].forEach(c=>c.update());

      // 2) predictions
      const { start_tick, future_buy, future_sell, future_demand, future_sun } =
        await fetch('/status/predict').then(r=>r.json());
      globalFutureSun=future_sun;
      globalFutureDem=future_demand;
      globalFutureBuy=future_buy;
      globalFutureSell=future_sell;
      globalStartTick=start_tick;

      future_sun.forEach((v,j)=>{
        const idx=start_tick+j;
        if(idx<LOOK_BACK) sunChart.data.datasets[1].data[idx]=v;
      });
      future_demand.forEach((v,j)=>{
        const idx=start_tick+j;
        if(idx<LOOK_BACK) demandChart.data.datasets[1].data[idx]=v;
      });
      future_buy.forEach((v,j)=>{
        const idx=start_tick+j;
        if(idx<LOOK_BACK) buyChart.data.datasets[1].data[idx]=v;
      });
      future_sell.forEach((v,j)=>{
        const idx=start_tick+j;
        if(idx<LOOK_BACK) sellChart.data.datasets[1].data[idx]=v;
      });
      [sunChart,demandChart,buyChart,sellChart].forEach(c=>c.update());


       // 3) policy forecast
    const liveState = history.map(pt =>
      [pt.sun, pt.demand, pt.price_buy, pt.price_sell]
    );
    const futureSeq = future_sun.map((s, j) =>
      [s, future_demand[j], future_buy[j], future_sell[j]]
    );
    const policyResp = await fetch('/status/policy', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ state_seq: liveState, future_seq: futureSeq })
    });
    const policy = await policyResp.json();
    globalPolicyActions = policy.future_actions;
    updateActionCard(policy.action_live);

    futureActionChart.data.datasets[0].data =
      policy.future_actions.map(a => a / 2);
    futureActionChart.update();

      // Deferrable‐load charts draw once per load/day
      const sched = await fetch('/status/schedule').then(r => r.json());
      sched.plans.forEach((plan, idx) => {
        const chart = [defChart1, defChart2, defChart3][idx];
        chart.data.datasets[0].data.fill(0);
        plan.forEach(([t, alloc]) => {
          if (t < LOOK_BACK) chart.data.datasets[0].data[t] = alloc;
        });
        chart.update();
      });


      costEl.textContent=cost.toFixed(2);
      prevTick=tick;
    }catch(err){
      console.error('initCharts error',err);
    }
  }

async function pollNewPoint() {
  console.log('⟳ pollNewPoint start; prevTick=', prevTick);
  try {
    // 1) Fetch live grid state (sun, demand, prices, cost)
    const {
      tick,
      sun,
      price_buy,
      price_sell,
      demand,
      pv_power,
      combined,
      connected_pv,
      cost
    } = await fetch('/status/grid').then(r => r.json());
    costEl.textContent = cost.toFixed(2);

       
    // 2) If we rolled over to a new day, redraw everything
    if (prevTick !== null && tick < prevTick) {
      await initCharts();
      prevTick = tick;
      return;
    }

    // 3) On a new tick
    if (tick !== prevTick) {
      if (tick < globalPolicyActions.length) {
        updateActionCard(globalPolicyActions[tick]);
      }
      animateValue(tickEl, prevTick || 0, tick, false);
      animateValue(buyEl, parseFloat(buyEl.textContent) || 0, price_buy, true);
      animateValue(sellEl, parseFloat(sellEl.textContent) || 0, price_sell, true);

      if (tick < LOOK_BACK) {
        sunChart.data.datasets[0].data[tick]    = sun;
        demandChart.data.datasets[0].data[tick] = demand;
        buyChart.data.datasets[0].data[tick]    = price_buy;
        sellChart.data.datasets[0].data[tick]   = price_sell;
        combinedChart.data.datasets[0].data[tick] = combined;
        {
     const prevCarry = combinedChart._carry || 0;
     const rawWithCarry = combined + prevCarry;
     const capped = Math.min(rawWithCarry, 4);
     combinedChart.data.datasets[1].data[tick] = capped;
     combinedChart._carry = rawWithCarry - capped;
     }
        [sunChart, demandChart, buyChart, sellChart, combinedChart].forEach(c => c.update());

        pvChart.data.datasets[0].data[tick] = connected_pv ? pv_power : NaN;
        pvChart.update();
      }

      prevTick = tick;
    }

  } catch (err) {
    console.error('pollNewPoint error', err);
  }
}

// Restart the loop
initCharts();
setInterval(pollNewPoint, 5000);


});
