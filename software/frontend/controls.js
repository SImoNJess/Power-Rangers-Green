// controls.js

document.addEventListener('DOMContentLoaded', () => {
  // Data for policy calls 
  let history       = [];
  let future_sun    = [];
  let future_demand = [];
  let future_buy    = [];
  let future_sell   = [];

  // Cache references to capacitor buttons
  const capButtons = Array.from(
    document.querySelectorAll("button[onclick^='sendCapCommand']")
  );

  // Auto-control state
  let autoControlEnabled = false;
  let controlPolicyActions = [];


  // Fetch and update control status every second
  async function fetchControl() {
    try {
      const {
        pv_power,
        voltage,
        irradiance,
        cap_energy,
        cap_SoC,
        cap_delta,
        led1_power,
        led2_power,
        led3_power,
        led4_power,
        auto_led1_power,
        auto_led2_power,
        auto_led3_power,
        auto_led4_power,
        import_power,
        export_power,
        tick,
        price_buy,
        price_sell,
        cost,
        mppt_enabled,
        action_live,
        connected
      } = await fetch('/status/control').then(r => r.json());

      // Auto LEDs + capacitor run every fetch (1 s) 
      if (autoControlEnabled) {
        // Send LED auto setpoints
        await sendOverride('led1_power', auto_led1_power);
        await sendOverride('led2_power', auto_led2_power);
        await sendOverride('led3_power', auto_led3_power);
        await sendOverride('led4_power', auto_led4_power);

        /// compute and send capacitor command from our preloaded policyActions
        let cmdVal = action_live;
        if (typeof tick === 'number' && controlPolicyActions[tick] !== undefined) {
          cmdVal = controlPolicyActions[tick];
        }
        // capAuto Logic
        let capCmd = 'H';
        if (cmdVal >  0) capCmd = 'S';
        else if (cmdVal <  0) capCmd = 'E';
        // high-Soc insurance
        if (cap_energy >= 45) capCmd = 'E';

        await sendCapCommand(capCmd);
        console.log(`Auto cap command (policy[${tick}]=${cmdVal}):`, capCmd);
        document.getElementById('cap-cmd-auto').textContent = capCmd;
      }

      // update in every second
      document.getElementById('pv-power').textContent     = pv_power.toFixed(2);
      document.getElementById('bus-voltage').textContent = voltage.toFixed(2);
      document.getElementById('irradiance').textContent =
        mppt_enabled ? '100% (MPPT)' : `${irradiance.toFixed(2)}%`;
      document.getElementById('cap-energy').textContent   = cap_energy.toFixed(2);
      document.getElementById('led1-power').textContent   = led1_power.toFixed(2);
      document.getElementById('led2-power').textContent   = led2_power.toFixed(2);
      document.getElementById('led3-power').textContent   = led3_power.toFixed(2);
      document.getElementById('led4-power').textContent   = led4_power.toFixed(2);
      document.getElementById('led1-auto').textContent    = auto_led1_power.toFixed(2);
      document.getElementById('led2-auto').textContent    = auto_led2_power.toFixed(2);
      document.getElementById('led3-auto').textContent    = auto_led3_power.toFixed(2);
      document.getElementById('led4-auto').textContent    = auto_led4_power.toFixed(2);
      document.getElementById('import-power').textContent = import_power.toFixed(2);
      document.getElementById('export-power').textContent = export_power.toFixed(2);
      document.getElementById('cap-SoC').textContent      = cap_SoC.toFixed(2);
      document.getElementById('cap-delta').textContent    = cap_delta.toFixed(2);
      document.getElementById('cost-value').textContent   = cost.toFixed(2);

      // toggle connection lights
      function toggleStatus(id, isOn) {
        const el = document.getElementById(id);
        if (!el) return;
        el.className = isOn
          ? 'w-4 h-4 rounded-full bg-green-500'
          : 'w-4 h-4 rounded-full bg-red-500';
      }

      toggleStatus('status-pv',   connected.PV);
      toggleStatus('status-dc',   connected.DC);
      toggleStatus('status-cap',  connected.CAP);
      toggleStatus('status-ext',  connected.EXT);
      toggleStatus('status-led1', connected.LED1);
      toggleStatus('status-led2', connected.LED2);
      toggleStatus('status-led3', connected.LED3);
      toggleStatus('status-led4', connected.LED4);

    } catch (err) {
      console.error('fetchControl error:', err);
    }
  }

  // Disable or enable manual UI (LED override + capacitor buttons)
  function setManualUi(enabled) {
    ['1','2','3','4'].forEach(i => {
      document.getElementById(`led${i}-input`).disabled        = !enabled;
      document.getElementById(`led${i}-override-btn`).disabled = !enabled;
    });
    capButtons.forEach(btn => btn.disabled = !enabled);
  }

  // Send capacitor command helper
  async function sendCapCommand(cmd) {
    await fetch('/controls/capacitor', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ command: cmd })
    });
  }
  window.sendCapCommand = sendCapCommand;

  // Override LED helper
  async function sendOverride(ledKey, value) {
    await fetch('/controls/override', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ [ledKey]: parseFloat(value) })
    });
  }

  // Auto control buttons
  const autoEnableBtn  = document.getElementById('auto-enable-btn');
  const autoDisableBtn = document.getElementById('auto-disable-btn');

  autoEnableBtn.disabled  = false;
  autoDisableBtn.disabled = true;
  setManualUi(true);

  autoEnableBtn.addEventListener('click', async () => {
    autoControlEnabled = true;
    autoEnableBtn.disabled   = true;
    autoDisableBtn.disabled  = false;
    setManualUi(false);

    // Prime history & forecast for policy calls & get full-day policy
    const grid = await fetch('/status/grid').then(r => r.json());
    history = grid.history;
    const pred = await fetch('/status/predict').then(r => r.json());
    future_sun    = pred.future_sun;
    future_demand = pred.future_demand;
    future_buy    = pred.future_buy;
    future_sell   = pred.future_sell;

    // fetch the full-day policy in one shot:
    try {
      const policy = await fetch('/status/policy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          state_seq: history.map(pt => [pt.sun, pt.demand, pt.price_buy, pt.price_sell]),
          future_seq: future_sun.map((s,i) =>
            [ s, future_demand[i], future_buy[i], future_sell[i] ]
          )
        })
      }).then(r => r.json());
      controlPolicyActions = policy.future_actions;
    } catch(err) {
      console.error('Failed to preload policy:', err);
     }

    // auto label for capacitor
    document.getElementById('cap-energy-auto-wrapper')
      .classList.remove('hidden');

    // LED auto labels
    ['1','2','3','4'].forEach(i => {
      document.getElementById(`led${i}-auto-wrapper`).classList.remove('hidden');
    });

    await fetchControl();
    console.log('Auto Control: ENABLED');
  });

  autoDisableBtn.addEventListener('click', () => {
    autoControlEnabled = false;
    autoDisableBtn.disabled = true;
    autoEnableBtn.disabled  = false;
    setManualUi(true);

    // Hide inline auto label for capacitor
    document.getElementById('cap-energy-auto-wrapper')
      .classList.add('hidden');

    // Hide LED auto labels
    ['1','2','3','4'].forEach(i => {
      document.getElementById(`led${i}-auto-wrapper`).classList.add('hidden');
    });
    console.log('Auto Control: DISABLED');
  });

  // Hook up LED override buttons
  function hookupOverride(ledNum) {
    const btn     = document.getElementById(`led${ledNum}-override-btn`);
    const input   = document.getElementById(`led${ledNum}-input`);
    const display = document.getElementById(`led${ledNum}-power`);

    btn.addEventListener('click', async () => {
      if (autoControlEnabled) return;
      const val = input.value;
      display.textContent = parseFloat(val).toFixed(2) + ' (pending…)';
      await sendOverride(`led${ledNum}_power`, val);
      await fetchControl();
    });
  }
  ['1','2','3','4'].forEach(hookupOverride);

  // MPPT enable/disable buttons as before...
  const enableBtn  = document.getElementById('mppt1-enable-btn');
  const disableBtn = document.getElementById('mppt1-disable-btn');

  enableBtn.addEventListener('click', async () => {
    await fetch('/controls/mppt', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ command: 1 }) });
    enableBtn.disabled  = true;
    disableBtn.disabled = false;
  });

  disableBtn.addEventListener('click', async () => {
    await fetch('/controls/mppt', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ command: 0 }) });
    disableBtn.disabled  = false;
    enableBtn.disabled = false;
  });

  fetchControl();
  setInterval(fetchControl, 1000);

});
