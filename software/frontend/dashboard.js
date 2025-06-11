// controls.js

document.addEventListener('DOMContentLoaded', () => {
  // ── 1) The 1 s poll to /status/control ─────────────────────────
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
        tick,         // this is what the server returns, not “current_tick”
        price_buy,
        price_sell,
        cost,         // server‐computed total cost
        mppt_enabled,
        connected     // { PV: bool, DC: bool, CAP: bool, EXT: bool, LED1: bool, … }
      } = await fetch('/status/control').then(r => r.json());

      // Update all the numeric fields
      document.getElementById('pv-power').textContent     = pv_power.toFixed(2);
      document.getElementById('bus-voltage').textContent = voltage.toFixed(2);
      // If MPPT override is active, force “100.00”, else show real irradiance
      document.getElementById('irradiance').textContent =
      mppt_enabled
      ? '100% (MPPT)'
      : `${irradiance.toFixed(2)}%`;
      document.getElementById('cap-energy').textContent   = cap_energy.toFixed(2);

      document.getElementById('led1-power').textContent   = led1_power.toFixed(2);
      document.getElementById('led2-power').textContent   = led2_power.toFixed(2);
      document.getElementById('led3-power').textContent   = led3_power.toFixed(2);
      document.getElementById('led4-power').textContent   = led4_power.toFixed(2);

      document.getElementById('led1-auto').textContent  = auto_led1_power.toFixed(2);
      document.getElementById('led2-auto').textContent  = auto_led2_power.toFixed(2);
      document.getElementById('led3-auto').textContent  = auto_led3_power.toFixed(2);
      document.getElementById('led4-auto').textContent  = auto_led4_power.toFixed(2);


      document.getElementById('import-power').textContent = import_power.toFixed(2);
      document.getElementById('export-power').textContent = export_power.toFixed(2);

      document.getElementById('cap-SoC').textContent      = cap_SoC.toFixed(2);
      document.getElementById('cap-delta').textContent    = cap_delta.toFixed(2);

      // ── NEW: Update Cost using server’s “cost” field ──────────
      document.getElementById('cost-value').textContent   = cost.toFixed(2);


      // ── 2) Toggle each connection‐light based on “connected” ────
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

    // ── If Auto Control is enabled, immediately send the auto setpoints ──
      if (autoControlEnabled) {
        // LED 1
        await sendOverride('led1_power', auto_led1_power);
        // LED 2
        await sendOverride('led2_power', auto_led2_power);
        // LED 3
        await sendOverride('led3_power', auto_led3_power);
        // LED 4
        await sendOverride('led4_power', auto_led4_power);
        console.log('Auto-setpoints pushed:', 
          auto_led1_power, auto_led2_power, auto_led3_power, auto_led4_power);
      }
    } catch (err) {
      console.error('fetchControl error:', err);
    }
  }


  function setManualUi(enabled) {
  ['1','2','3','4'].forEach(i => {
    document.getElementById(`led${i}-input`).disabled        = !enabled;
    document.getElementById(`led${i}-override-btn`).disabled = !enabled;
  });
}
  // ── 3) Keep your existing override helpers ─────────────────────
  async function sendCapCommand(cmd) {
    console.log("Sending capacitor command:", cmd);
    await fetch('/controls/capacitor', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ command: cmd })
    });
  }
  window.sendCapCommand = sendCapCommand;

  async function sendOverride(ledKey, value) {
    await fetch('/controls/override', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ [ledKey]: parseFloat(value) })
    });
  }


// ── Auto Control buttons ───────────────────
let autoControlEnabled = false;
const autoEnableBtn  = document.getElementById('auto-enable-btn');
const autoDisableBtn = document.getElementById('auto-disable-btn');

// initial UI state
autoEnableBtn.disabled  = false;
autoDisableBtn.disabled = true;
setManualUi(true);      // manual ON → auto OFF

autoEnableBtn.addEventListener('click', async () => {
  autoControlEnabled = true;
  autoEnableBtn.disabled   = true;
  autoDisableBtn.disabled  = false;
  setManualUi(false);

  // Unhide the “(Auto: …)” labels
  ['1','2','3','4'].forEach(i => {
    document.getElementById(`led${i}-auto-wrapper`)
            .classList.remove('hidden');
  });

  // 1) Refresh the auto values in the UI
  await fetchControl();

  // 2) Push them back to the real LEDs
  for (const i of ['1','2','3','4']) {
    const autoVal = parseFloat(
      document.getElementById(`led${i}-auto`).textContent
    );
    await sendOverride(`led${i}_power`, autoVal);
  }

  console.log('Auto Control: ENABLED and values published');
});


autoDisableBtn.addEventListener('click', () => {
  autoControlEnabled = false;
  autoDisableBtn.disabled = true;
  autoEnableBtn.disabled  = false;
  setManualUi(true);

  // HIDE all “Auto” wrappers
  ['1','2','3','4'].forEach(i => {
    document.getElementById(`led${i}-auto-wrapper`)
            .classList.add('hidden');
  });
});


  function hookupOverride(ledNum) {
    const btn     = document.getElementById(`led${ledNum}-override-btn`);
    const input   = document.getElementById(`led${ledNum}-input`);
    const display = document.getElementById(`led${ledNum}-power`);

    btn.addEventListener('click', async () => {
    if (autoControlEnabled) return;             // ignore clicks when auto ON
    const val = input.value;
    display.textContent = parseFloat(val).toFixed(2) + ' (pending…)';
    await sendOverride(`led${ledNum}_power`, val);
    await fetchControl();
  });

  }
  ['1','2','3','4'].forEach(hookupOverride);


  // ── 4) MPPT Enable/Disable buttons (send {command:1} or {command:0}) ─
  const enableBtn  = document.getElementById('mppt1-enable-btn');
  const disableBtn = document.getElementById('mppt1-disable-btn');

  enableBtn.addEventListener('click', async () => {
    await fetch('/controls/mppt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ command: 1 })
    });
    // After enabling, disable “Enable” button and enable “Disable”
    enableBtn.disabled  = true;
    disableBtn.disabled = false;
  });

  disableBtn.addEventListener('click', async () => {
    await fetch('/controls/mppt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ command: 0 })
    });
    // After disabling, disable “Disable” button and enable “Enable”
    disableBtn.disabled = true;
    enableBtn.disabled  = false;
  });


  // ── 5) Kick off the 1 s polling loop ─────────────────────────────
  fetchControl();
  setInterval(fetchControl, 1000);

}); // end DOMContentLoaded
