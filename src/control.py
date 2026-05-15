import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import sys
sys.path.insert(0, "src")


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SensorReading:
    """Raw sensor input at a single timepoint."""
    time_h:         float
    DO_pct:         Optional[float] = None
    pH:             Optional[float] = None
    temp_c:         Optional[float] = None
    stirrer:        Optional[float] = None
    feed_pump:      Optional[float] = None
    acid_pump:      Optional[float] = None
    # Sparse / manually measured
    OD660:          Optional[float] = None
    DCW_gL:         Optional[float] = None
    glycerol_gL:    Optional[float] = None
    methanol_gL:    Optional[float] = None
    L1_yield_mgL:   Optional[float] = None


@dataclass
class ControlOutput:
    """Decision output at each timestep."""
    time_h:             float
    phase:              str
    layer_used:         str        # "safety" | "rules" | "ml"
    action:             str        # "none" | "start_induction" | "feed_methanol"
                                   # "feed_glycerol" | "increase_stirrer"
                                   # "reduce_feed" | "harvest"
                                   # "alert_ph" | "alert_temp" | "alert_do"
    methanol_feed_pct:  float = 0.0
    glycerol_feed_pct:  float = 0.0
    stirrer_setpoint:   float = 700.0
    message:            str   = ""
    alerts:             list  = field(default_factory=list)
    ml_phase:           Optional[str]   = None
    ml_confidence:      Optional[float] = None
    L1_predicted:       Optional[float] = None
    carbon_ratio_pct:   Optional[float] = None   # methanol fraction 0–1


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — SAFETY (always active, never overridden)
# ══════════════════════════════════════════════════════════════════════════════

class SafetyLayer:
    """
    Hard limits that override everything else.
    Based on H. polymorpha cultivation constraints from paper.
    """

    LIMITS = {
        "pH":    (4.0,  6.5),
        "temp":  (25.0, 37.0),
        "DO":    (10.0, 105.0),
        # Biomass ceiling from paper: Replicate 2 hit OD 197 with lower yield
        # than Replicate 1 at OD 169 — oxygen limitation above ~180
        "OD660": (0.0,  180.0),
        "DCW":   (0.0,  50.0),   # g/L — plateau observed at ~46 g/L in batch 7
    }

    def check(self, reading: SensorReading) -> list:
        """Returns list of alert strings if any limit is violated."""
        alerts = []

        if reading.pH is not None:
            lo, hi = self.LIMITS["pH"]
            if reading.pH < lo:
                alerts.append(f"CRITICAL: pH too low ({reading.pH:.2f} < {lo})"
                               " — check acid pump")
            elif reading.pH > hi:
                alerts.append(f"CRITICAL: pH too high ({reading.pH:.2f} > {hi})"
                               " — check base pump")

        if reading.temp_c is not None:
            lo, hi = self.LIMITS["temp"]
            if reading.temp_c < lo:
                alerts.append(f"WARNING: temp low ({reading.temp_c:.1f}°C)")
            elif reading.temp_c > hi:
                alerts.append(f"CRITICAL: temp high ({reading.temp_c:.1f}°C)"
                               " — risk of protein denaturation")

        if reading.DO_pct is not None:
            lo, hi = self.LIMITS["DO"]
            if reading.DO_pct < lo:
                alerts.append(f"WARNING: DO critically low ({reading.DO_pct:.1f}%)"
                               " — increase stirrer or aeration")

        if reading.OD660 is not None:
            _, hi = self.LIMITS["OD660"]
            if reading.OD660 > hi:
                alerts.append(
                    f"WARNING: OD660 ceiling reached ({reading.OD660:.1f} > {hi})"
                    " — pause glycerol feed, oxygen may be limiting"
                )

        if reading.DCW_gL is not None:
            _, hi = self.LIMITS["DCW"]
            if reading.DCW_gL > hi:
                alerts.append(
                    f"WARNING: DCW ceiling ({reading.DCW_gL:.1f} g/L > {hi} g/L)"
                    " — biomass plateau, reduce glycerol feed rate"
                )

        return alerts

    def emergency_action(self, alerts: list) -> Optional[str]:
        """Returns emergency action if critical alert exists."""
        for a in alerts:
            if "CRITICAL" in a and "temp" in a.lower():
                return "stop_heating"
            if "CRITICAL" in a and "ph" in a.lower():
                return "adjust_ph"
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — DO-STAT METHANOL CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════

class DOStatController:
    """
    Online DO-stat feedback controller for methanol pump (Phase 1).

    Logic (H. polymorpha methanol metabolism):
      DO rises above setpoint  →  methanol depleted  →  pump ON
      DO falls below setpoint  →  bacteria consuming methanol  →  pump OFF
      Deadband prevents rapid toggling.
    """

    def __init__(self, do_setpoint: float = 40.0, deadband: float = 5.0,
                 pulse_pct: float = 1.0, min_interval_h: float = 0.2):
        self.do_setpoint    = do_setpoint
        self.deadband       = deadband
        self.pulse_pct      = pulse_pct
        self.min_interval_h = min_interval_h

        self._feeding      = False
        self._last_feed_h  = -999.0

    def step(self, do_pct: Optional[float],
             time_h: float) -> tuple[bool, float]:
        """
        Returns (feed_active, feed_pct).
        feed_active: whether methanol pump should be ON this step
        feed_pct:    pump output (0 or pulse_pct)
        """
        if do_pct is None:
            # No DO signal — hold current state
            return self._feeding, self.pulse_pct if self._feeding else 0.0

        hi = self.do_setpoint + self.deadband
        lo = self.do_setpoint - self.deadband
        dt = time_h - self._last_feed_h

        if do_pct > hi and dt >= self.min_interval_h:
            # DO high → methanol depleted → feed
            if not self._feeding:
                self._last_feed_h = time_h
            self._feeding = True
        elif do_pct < lo:
            # DO falling → bacteria actively consuming → stop
            self._feeding = False
        # inside deadband: maintain current state

        return self._feeding, self.pulse_pct if self._feeding else 0.0

    def reset(self):
        self._feeding     = False
        self._last_feed_h = -999.0

    @property
    def state(self) -> str:
        return "feeding" if self._feeding else "idle"


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — GLYCEROL FED-BATCH CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════

class GlycerolController:
    """
    Fed-batch glycerol pump controller for growth phase (Phase 2).

    Primary mode: maintain glycerol above setpoint via exponential feed
    profile that targets a constant specific growth rate (mu = target_mu).
    Equation: F = (mu_target * X * V) / (Ys/x * C_feed)

    Fallback mode (no online glycerol sensor): trigger on DO spike,
    which indicates glycerol depletion by bacteria switching to O2.
    """

    def __init__(self, setpoint_gL: float = 5.0, min_gL: float = 1.0,
                 target_mu: float = 0.12, yield_xs: float = 0.5,
                 feed_conc_gL: float = 500.0, volume_L: float = 2.0,
                 do_spike_threshold: float = 70.0):
        self.setpoint_gL        = setpoint_gL
        self.min_gL             = min_gL
        self.target_mu          = target_mu
        self.yield_xs           = yield_xs        # g-biomass / g-glycerol
        self.feed_conc_gL       = feed_conc_gL    # glycerol concentration in feed
        self.volume_L           = volume_L        # working volume
        self.do_spike_threshold = do_spike_threshold

    def step(self, reading: SensorReading) -> float:
        """Returns glycerol_feed_pct (0–100)."""
        if reading.glycerol_gL is not None:
            if reading.glycerol_gL >= self.setpoint_gL:
                return 0.0  # sufficient glycerol — no feed needed

            # Below setpoint: exponential feeding for constant growth rate
            if reading.DCW_gL is not None and reading.DCW_gL > 0:
                # F (L/h) = mu * X * V / (Ys * Cf)
                f_rate = (self.target_mu * reading.DCW_gL * self.volume_L
                          / (self.yield_xs * self.feed_conc_gL))
                return min(f_rate * 100.0, 100.0)
            return 30.0  # constant fallback when no biomass reading

        # No glycerol sensor — use DO spike as depletion proxy
        if reading.DO_pct is not None and reading.DO_pct > self.do_spike_threshold:
            return 40.0

        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — CARBON RATIO OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

# Base methanol:glycerol ratios per phase.
# Growth:     glycerol only (methanol toxic before MOX induction)
# Induction:  mostly methanol to activate AOX1, small glycerol supplement
# Production: methanol only (full induction)
PHASE_CARBON_RATIO: dict[str, dict[str, float]] = {
    "growth":     {"glycerol": 1.0, "methanol": 0.0},
    "induction":  {"glycerol": 0.3, "methanol": 0.7},
    "production": {"glycerol": 0.0, "methanol": 1.0},
    "harvest":    {"glycerol": 0.0, "methanol": 0.0},
}


class CarbonRatioOptimizer:
    """
    Phase 3: Methanol / glycerol carbon source ratio manager.

    Computes feed split for each phase. Dynamically adjusts when residual
    concentrations are available to avoid glycerol overflow or methanol toxicity.

    Both substrates are carbon sources; ratio affects:
      - Induction strength (methanol drives AOX1 promoter)
      - Growth/production balance
      - DO demand (methanol oxidation is O2-intensive)
    """

    def __init__(self, ratios: Optional[dict] = None):
        self.ratios = ratios or dict(PHASE_CARBON_RATIO)

    def split(self, total_feed_pct: float, phase: str,
              glycerol_residual: Optional[float] = None,
              methanol_residual: Optional[float] = None,
              ) -> tuple[float, float]:
        """
        Returns (glycerol_feed_pct, methanol_feed_pct).
        Adjusts base phase ratio if residual concentrations are known.
        """
        base = self.ratios.get(phase, {"glycerol": 0.0, "methanol": 1.0})
        gly_frac  = base["glycerol"]
        meth_frac = base["methanol"]

        # Glycerol residual high → reduce glycerol feed (avoid overflow)
        if glycerol_residual is not None and glycerol_residual > 10.0:
            excess    = min((glycerol_residual - 10.0) / 20.0, 1.0)
            shift     = gly_frac * excess * 0.5
            gly_frac  = max(0.0, gly_frac  - shift)
            meth_frac = min(1.0, meth_frac + shift)

        # Methanol residual high → reduce methanol feed (avoid toxicity >3 g/L)
        if methanol_residual is not None and methanol_residual > 3.0:
            excess    = min((methanol_residual - 3.0) / 5.0, 1.0)
            shift     = meth_frac * excess * 0.5
            meth_frac = max(0.0, meth_frac - shift)
            gly_frac  = min(1.0, gly_frac  + shift)

        return total_feed_pct * gly_frac, total_feed_pct * meth_frac

    def methanol_fraction(self, phase: str,
                          glycerol_residual: Optional[float] = None,
                          methanol_residual: Optional[float] = None) -> float:
        """Returns methanol fraction 0–1 for reporting."""
        _, m = self.split(1.0, phase, glycerol_residual, methanol_residual)
        return m

    def summary(self, phase: str,
                glycerol_residual: Optional[float] = None,
                methanol_residual: Optional[float] = None) -> str:
        g, m = self.split(100.0, phase, glycerol_residual, methanol_residual)
        return f"Gly:{g:.0f}%/MeOH:{m:.0f}%"


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — STRAIN AUTO-CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

STRAIN_PROFILES: dict[str, dict] = {
    "H. polymorpha": {
        "mu_max":               0.16,
        "do_setpoint":          40.0,
        "do_deadband":          5.0,
        "methanol_pulse_pct":   1.0,
        "min_induction_time_h": 20.0,
        "harvest_h":            120.0,
        "do_low_threshold":     20.0,
        "glycerol_setpoint_gL": 5.0,
        "target_mu":            0.12,
    },
    "P. pastoris": {
        "mu_max":               0.18,
        "do_setpoint":          30.0,
        "do_deadband":          8.0,
        "methanol_pulse_pct":   1.5,
        "min_induction_time_h": 16.0,
        "harvest_h":            96.0,
        "do_low_threshold":     15.0,
        "glycerol_setpoint_gL": 4.0,
        "target_mu":            0.15,
    },
}


class StrainAutoCalibrator:
    """
    Phase 4: Auto-calibrate all controllers given strain + target product.

    Lookup order:
      1. Known strain database (STRAIN_PROFILES)
      2. Claude API (claude-haiku) for unknown strains
      3. H. polymorpha defaults as last resort
    """

    def __init__(self):
        self.profiles       = dict(STRAIN_PROFILES)
        self.active_strain  = None
        self.active_product = None

    def setup(self, strain: str, target_product: str,
              rules_layer: "RuleBasedLayer") -> dict:
        """
        Auto-calibrate rules_layer for the given strain and product.
        Returns the resolved profile dict.
        """
        self.active_strain  = strain
        self.active_product = target_product

        profile = self._lookup(strain, target_product)
        self._apply(profile, rules_layer)

        print(f"\n  Strain auto-calibration: {strain} → {target_product}")
        for k, v in profile.items():
            print(f"    {k}: {v}")
        return profile

    def _lookup(self, strain: str, target_product: str) -> dict:
        key = next((k for k in self.profiles
                    if k.lower() in strain.lower()), None)
        if key:
            return dict(self.profiles[key])
        print(f"  Unknown strain '{strain}' — querying AI…")
        return self._ai_lookup(strain, target_product)

    def _ai_lookup(self, strain: str, product: str) -> dict:
        """Query Claude API for unknown strain parameters."""
        try:
            import anthropic
            import json
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                messages=[{"role": "user", "content": (
                    f"You are a fermentation process expert. "
                    f"For a fed-batch cultivation of {strain} targeting {product}, "
                    f"give recommended controller parameters as a JSON object with keys: "
                    f"mu_max, do_setpoint, do_deadband, methanol_pulse_pct, "
                    f"min_induction_time_h, harvest_h, do_low_threshold, "
                    f"glycerol_setpoint_gL, target_mu. "
                    f"Return ONLY the JSON object, no explanation."
                )}],
            )
            profile = json.loads(msg.content[0].text)
            print(f"  AI returned profile for {strain}")
            self.profiles[strain] = profile   # cache for this session
            return profile
        except Exception as e:
            print(f"  AI lookup failed ({e}) — using H. polymorpha defaults")
            return dict(STRAIN_PROFILES["H. polymorpha"])

    def _apply(self, profile: dict, rules_layer: "RuleBasedLayer"):
        """Push profile values into rules layer and its sub-controllers."""
        for attr in ("min_induction_time_h", "harvest_h",
                     "methanol_pulse_pct", "do_low_threshold"):
            if attr in profile:
                setattr(rules_layer, attr, profile[attr])

        if "do_setpoint" in profile:
            rules_layer._do_stat.do_setpoint = profile["do_setpoint"]
        if "do_deadband" in profile:
            rules_layer._do_stat.deadband = profile["do_deadband"]
        if "methanol_pulse_pct" in profile:
            rules_layer._do_stat.pulse_pct = profile["methanol_pulse_pct"]

        if "glycerol_setpoint_gL" in profile:
            rules_layer._glycerol_ctrl.setpoint_gL = profile["glycerol_setpoint_gL"]
        if "target_mu" in profile:
            rules_layer._glycerol_ctrl.target_mu = profile["target_mu"]


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — RULE-BASED CONTROL (integrates Phases 1–3)
# ══════════════════════════════════════════════════════════════════════════════

class RuleBasedLayer:
    """
    Rule-based controller with integrated DO-stat (Phase 1),
    glycerol fed-batch (Phase 2), and carbon ratio management (Phase 3).

    Phases: growth → induction → production → harvest
    """

    def __init__(self,
                 mu_induction_threshold: float = 0.02,
                 glycerol_depletion_gL:  float = 1.0,
                 min_induction_time_h:   float = 20.0,
                 methanol_interval_h:    float = 1.4,   # fallback only
                 methanol_pulse_pct:     float = 1.0,
                 do_low_threshold:       float = 20.0,
                 stirrer_max_rpm:        float = 700.0,
                 harvest_h:             float = 120.0,
                 do_setpoint:           float = 40.0,
                 do_deadband:           float = 5.0,
                 ):

        self.mu_induction_threshold = mu_induction_threshold
        self.glycerol_depletion_gL  = glycerol_depletion_gL
        self.min_induction_time_h   = min_induction_time_h
        self.methanol_interval_h    = methanol_interval_h
        self.methanol_pulse_pct     = methanol_pulse_pct
        self.do_low_threshold       = do_low_threshold
        self.stirrer_max_rpm        = stirrer_max_rpm
        self.harvest_h              = harvest_h

        # Phase 1-3 sub-controllers
        self._do_stat       = DOStatController(do_setpoint=do_setpoint,
                                               deadband=do_deadband,
                                               pulse_pct=methanol_pulse_pct)
        self._glycerol_ctrl = GlycerolController()
        self._ratio_opt     = CarbonRatioOptimizer()

        # Internal state
        self._phase             = "growth"
        self._last_methanol_h   = -999.0
        self._induction_start_h = None
        self._mu_history        = []
        self._do_history        = []

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        self._phase = value

    def _update_history(self, reading: SensorReading):
        if reading.DO_pct is not None:
            self._do_history.append((reading.time_h, reading.DO_pct))
            self._do_history = self._do_history[-20:]

    def _do_spike_detected(self) -> bool:
        """Rapid DO rise — carbon source depletion signal."""
        if len(self._do_history) < 4:
            return False
        recent  = self._do_history[-4:]
        do_rise = recent[-1][1] - recent[0][1]
        return do_rise > 20.0

    def _should_induce(self, reading: SensorReading,
                       mu: Optional[float]) -> tuple[bool, str]:
        if reading.time_h < self.min_induction_time_h:
            return False, ""

        if reading.glycerol_gL is not None:
            if reading.glycerol_gL < self.glycerol_depletion_gL:
                return True, f"glycerol depleted ({reading.glycerol_gL:.2f} g/L)"

        if mu is not None:
            if mu < self.mu_induction_threshold:
                return True, f"mu dropped ({mu:.4f} h-1)"

        if self._do_spike_detected():
            return True, "DO spike detected (carbon depletion)"

        return False, ""

    def step(self, reading: SensorReading,
             mu: Optional[float] = None) -> ControlOutput:
        """Process one timestep and return control decision."""

        self._update_history(reading)
        out = ControlOutput(
            time_h           = reading.time_h,
            phase            = self._phase,
            layer_used       = "rules",
            action           = "none",
            stirrer_setpoint = reading.stirrer or 700.0,
        )

        # ── GROWTH ────────────────────────────────────────────────────────────
        if self._phase == "growth":
            # Phase 2: glycerol pump control
            gly_pct = self._glycerol_ctrl.step(reading)
            if gly_pct > 0:
                out.glycerol_feed_pct = gly_pct
                out.action = "feed_glycerol"

            # Phase 3: growth = glycerol only, 0% methanol
            out.carbon_ratio_pct = 0.0

            # Check induction
            should, reason = self._should_induce(reading, mu)
            if should:
                self._phase             = "induction"
                self._induction_start_h = reading.time_h
                out.phase               = "induction"
                out.action              = "start_induction"
                out.glycerol_feed_pct   = 0.0   # stop glycerol at induction
                out.message = f"Induction start at {reading.time_h:.1f}h — {reason}"
                self._do_stat.reset()   # fresh DO-stat for induction phase
            else:
                gly_msg = f" | Gly:{gly_pct:.1f}%" if gly_pct > 0 else ""
                out.message = (f"Growth — mu={mu:.4f}" if mu else "Growth") + gly_msg

        # ── INDUCTION ─────────────────────────────────────────────────────────
        elif self._phase == "induction":
            # Phase 1: DO-stat methanol
            feeding, raw_meth = self._do_stat.step(reading.DO_pct, reading.time_h)

            # Phase 3: induction split (glycerol 30 / methanol 70 by default)
            gly_pct, meth_pct = self._ratio_opt.split(
                raw_meth, "induction",
                reading.glycerol_gL, reading.methanol_gL,
            )
            out.methanol_feed_pct = meth_pct
            out.glycerol_feed_pct = gly_pct
            out.carbon_ratio_pct  = self._ratio_opt.methanol_fraction(
                "induction", reading.glycerol_gL, reading.methanol_gL)

            if feeding:
                out.action  = "feed_methanol"
                do_str = f"DO={reading.DO_pct:.1f}%" if reading.DO_pct else "DO=?"
                out.message = (f"Induction DO-stat: {do_str} → feed "
                               f"| {self._ratio_opt.summary('induction', reading.glycerol_gL, reading.methanol_gL)}")
            else:
                do_str = f"DO={reading.DO_pct:.1f}%" if reading.DO_pct else "no DO"
                out.message = f"Induction: DO-stat idle ({do_str})"

            # Advance to production after 2h of induction
            if (self._induction_start_h and
                    reading.time_h - self._induction_start_h > 2.0):
                self._phase          = "production"
                out.phase            = "production"
                out.carbon_ratio_pct = self._ratio_opt.methanol_fraction(
                    "production", reading.glycerol_gL, reading.methanol_gL)
                out.message          = (f"→ Production at {reading.time_h:.1f}h"
                                        + (" | DO-stat feeding" if feeding else
                                           " | DO-stat idle"))

        # ── PRODUCTION ────────────────────────────────────────────────────────
        elif self._phase == "production":
            # Phase 1: DO-stat methanol (full production — no glycerol base)
            feeding, raw_meth = self._do_stat.step(reading.DO_pct, reading.time_h)

            # Phase 3: production = methanol only (unless residual glycerol high)
            gly_pct, meth_pct = self._ratio_opt.split(
                raw_meth, "production",
                reading.glycerol_gL, reading.methanol_gL,
            )
            out.methanol_feed_pct = meth_pct
            out.glycerol_feed_pct = gly_pct
            out.carbon_ratio_pct  = self._ratio_opt.methanol_fraction(
                "production", reading.glycerol_gL, reading.methanol_gL)

            if feeding:
                out.action  = "feed_methanol"
                do_str = f"DO={reading.DO_pct:.1f}%" if reading.DO_pct else "DO=?"
                out.message = (f"Production DO-stat: {do_str} → feed "
                               f"| {self._ratio_opt.summary('production', reading.glycerol_gL, reading.methanol_gL)}")

            # DO low → increase stirrer for O2 transfer
            if (reading.DO_pct is not None and
                    reading.DO_pct < self.do_low_threshold):
                new_rpm = min((reading.stirrer or 700) + 50, self.stirrer_max_rpm)
                out.stirrer_setpoint = new_rpm
                if not feeding:
                    out.action = "increase_stirrer"
                do_str = f"DO={reading.DO_pct:.1f}%"
                out.message += f" | {do_str} low → stirrer {new_rpm:.0f} rpm"

            # Harvest
            if reading.time_h >= self.harvest_h:
                self._phase = "harvest"
                out.phase   = "harvest"
                out.action  = "harvest"
                out.message = f"Harvest at {reading.time_h:.1f}h"

        # ── HARVEST ───────────────────────────────────────────────────────────
        elif self._phase == "harvest":
            out.action            = "harvest"
            out.methanol_feed_pct = 0.0
            out.glycerol_feed_pct = 0.0
            out.carbon_ratio_pct  = None
            out.message           = "Harvesting — all feeds stopped"

        return out

    def update_thresholds(self, mu_max: float, glycerol0: float):
        """Update thresholds based on adaptive calibration."""
        self.mu_induction_threshold = 0.15 * mu_max
        self.glycerol_depletion_gL  = 0.04 * glycerol0
        print(f"  Rules updated: mu_thresh={self.mu_induction_threshold:.4f}, "
              f"glycerol_thresh={self.glycerol_depletion_gL:.2f} g/L")


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — ML ADVISORY
# ══════════════════════════════════════════════════════════════════════════════

class MLAdvisoryLayer:
    """
    ML layer that advises the rule-based controller.
    Only activates after calibration window.
    Only overrides rules if confidence > threshold.
    """

    def __init__(self, confidence_threshold: float = 0.85,
                 min_active_time_h: float = 12.0):
        self.confidence_threshold = confidence_threshold
        self.min_active_time_h    = min_active_time_h
        self._classifier          = None
        self._predictor           = None
        self._active              = False

    def load_models(self, model_dir: str = "outputs/models"):
        """Load trained ML models if available."""
        from pathlib import Path
        import joblib
        import sys
        sys.path.insert(0, "src")

        clf_path  = Path(model_dir) / "phase_classifier.pkl"
        pred_path = Path(model_dir) / "yield_predictor.pkl"

        try:
            import ml_model  # noqa — needed for joblib unpickle
            if clf_path.exists():
                self._classifier = joblib.load(clf_path)
                print("  ML: phase classifier loaded")
            if pred_path.exists():
                self._predictor  = joblib.load(pred_path)
                print("  ML: yield predictor loaded")
            self._active = self._classifier is not None
        except Exception as e:
            print(f"  ML: could not load models ({e}) — falling back to rules")

    def advise(self, reading: SensorReading,
               sparse_history: pd.DataFrame) -> dict:
        """
        Returns ML advice dict:
        { phase, confidence, L1_predicted, override }
        override=True means ML is confident enough to override rules
        """
        advice = {
            "phase":        None,
            "confidence":   0.0,
            "L1_predicted": None,
            "override":     False,
        }

        if not self._active:
            return advice
        if reading.time_h < self.min_active_time_h:
            return advice
        if sparse_history is None or sparse_history.empty:
            return advice

        try:
            from ml_model import engineer_features
            feats = engineer_features(sparse_history)

            if self._classifier is not None:
                result = self._classifier.predict(feats)
                if result is not None and not result.empty:
                    nearest = result.iloc[(result.time_h -
                                          reading.time_h).abs().argsort()[:1]]
                    phase  = nearest.phase_pred.values[0]
                    conf   = float(nearest.confidence.values[0])
                    advice["phase"]      = phase
                    advice["confidence"] = conf
                    advice["override"]   = conf >= self.confidence_threshold

            if self._predictor is not None:
                pred = self._predictor.predict(feats)
                if pred is not None and not pred.empty:
                    latest = pred.iloc[-1]
                    advice["L1_predicted"] = float(latest.L1_pred)

        except Exception as e:
            print(f"  ML advise error: {e}")

        return advice


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4 — ADAPTIVE CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveCalibration:
    """
    Self-calibrates controller thresholds from early run data.
    Makes the system scale-agnostic.
    """

    def __init__(self, window_h: float = 12.0):
        self.window_h    = window_h
        self.mu_max      = 0.16    # default from batch 7
        self.glycerol0   = 25.0
        self.x0          = 0.5
        self.calibrated  = False

    def calibrate(self, sparse_df: pd.DataFrame,
                  rules_layer: RuleBasedLayer):
        """Calibrate from first window_h of a new run."""
        early = sparse_df[sparse_df.time_h <= self.window_h].copy()

        if "DCW_gL" in early.columns:
            dcw = early.DCW_gL.dropna()
            if len(dcw) >= 2:
                dt      = early.time_h.diff().dropna()
                mu_vals = (np.log(dcw / dcw.shift(1)) / dt).replace(
                    [np.inf, -np.inf], np.nan).dropna()
                if not mu_vals.empty:
                    self.mu_max = min(float(mu_vals.quantile(0.9)), 0.25)
                self.x0 = float(dcw.iloc[0])

        if "glycerol_gL" in early.columns:
            g = early.glycerol_gL.dropna()
            if not g.empty:
                self.glycerol0 = float(g.iloc[0])

        self.calibrated = True
        print(f"\n  Calibration complete:")
        print(f"    mu_max     = {self.mu_max:.4f} h-1")
        print(f"    glycerol0  = {self.glycerol0:.2f} g/L")
        print(f"    x0         = {self.x0:.2f} g/L")

        rules_layer.update_thresholds(self.mu_max, self.glycerol0)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTROLLER — orchestrates all layers
# ══════════════════════════════════════════════════════════════════════════════

class BioreactorController:
    """
    Full layered controller.

    Usage:
        ctrl = BioreactorController()
        ctrl.strain_setup("H. polymorpha", "HPV52 L1")  # Phase 4 auto-calibrate
        ctrl.load_models()                               # optional ML
        ctrl.calibrate(early_data)                       # data-driven calibration
        output = ctrl.step(reading, mu, sparse_history)
    """

    def __init__(self, **rule_kwargs):
        self.safety           = SafetyLayer()
        self.rules            = RuleBasedLayer(**rule_kwargs)
        self.ml               = MLAdvisoryLayer()
        self.calibration      = AdaptiveCalibration()
        self.strain_calibrator = StrainAutoCalibrator()
        self._history         = []

    def strain_setup(self, strain: str, target_product: str) -> dict:
        """Phase 4: auto-calibrate all controllers for a strain and product."""
        return self.strain_calibrator.setup(strain, target_product, self.rules)

    def load_models(self, model_dir: str = "outputs/models"):
        self.ml.load_models(model_dir)

    def calibrate(self, sparse_df: pd.DataFrame):
        self.calibration.calibrate(sparse_df, self.rules)

    def step(self, reading: SensorReading,
             mu: Optional[float] = None,
             sparse_history: Optional[pd.DataFrame] = None) -> ControlOutput:
        """Main control step — runs all layers in order."""

        # ── Layer 1: Safety ───────────────────────────────────────────────────
        alerts    = self.safety.check(reading)
        emergency = self.safety.emergency_action(alerts)

        if emergency:
            out = ControlOutput(
                time_h     = reading.time_h,
                phase      = self.rules.phase,
                layer_used = "safety",
                action     = emergency,
                alerts     = alerts,
                message    = f"SAFETY OVERRIDE: {emergency}",
            )
            self._history.append(out)
            return out

        # ── Layer 2: Rules (Phases 1-3 integrated) ───────────────────────────
        out        = self.rules.step(reading, mu)
        out.alerts = alerts

        # ── Layer 3: ML advisory ──────────────────────────────────────────────
        if sparse_history is not None:
            advice = self.ml.advise(reading, sparse_history)
            out.ml_phase      = advice["phase"]
            out.ml_confidence = advice["confidence"]
            out.L1_predicted  = advice["L1_predicted"]

            # ML override — only if confident AND different from rules
            # Never override harvest — that is a hard stop
            if (advice["override"] and
                    advice["phase"] is not None and
                    advice["phase"] != self.rules.phase and
                    self.rules.phase != "harvest"):
                print(f"  ML override at {reading.time_h:.1f}h: "
                      f"{self.rules.phase} → {advice['phase']} "
                      f"(conf={advice['confidence']:.2f})")
                self.rules.phase = advice["phase"]
                out.phase        = advice["phase"]
                out.layer_used   = "ml"

        self._history.append(out)
        return out

    def get_history(self) -> pd.DataFrame:
        """Return full decision history as DataFrame."""
        return pd.DataFrame([{
            "time_h":             o.time_h,
            "phase":              o.phase,
            "layer_used":         o.layer_used,
            "action":             o.action,
            "methanol_feed_pct":  o.methanol_feed_pct,
            "glycerol_feed_pct":  o.glycerol_feed_pct,
            "carbon_ratio_pct":   o.carbon_ratio_pct,
            "stirrer_setpoint":   o.stirrer_setpoint,
            "message":            o.message,
            "ml_phase":           o.ml_phase,
            "ml_confidence":      o.ml_confidence,
            "L1_predicted":       o.L1_predicted,
            "alerts":             "; ".join(o.alerts) if o.alerts else "",
        } for o in self._history])


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def simulate_batch(sparse_df: pd.DataFrame,
                   cont_df:   Optional[pd.DataFrame] = None,
                   load_ml:   bool = True,
                   strain:    str  = "H. polymorpha",
                   product:   str  = "HPV52 L1",
                   ) -> pd.DataFrame:
    """
    Run full simulation of a batch through the layered controller.
    Returns DataFrame of all control decisions.
    """
    from features import compute_growth_rate, continuous_features

    ctrl = BioreactorController()

    # Phase 4: strain auto-calibration
    ctrl.strain_setup(strain, product)

    if load_ml:
        ctrl.load_models()

    # Data-driven calibration (overrides strain defaults with actual data)
    ctrl.calibrate(sparse_df)

    # Compute growth rates
    gr = compute_growth_rate(sparse_df)
    cf = continuous_features(cont_df) if cont_df is not None else None

    timeline = sparse_df.merge(
        gr[["time_h", "mu_per_h"]], on="time_h", how="left"
    ).sort_values("time_h").reset_index(drop=True)

    sparse_history = pd.DataFrame()

    for _, row in timeline.iterrows():
        sparse_history = pd.concat(
            [sparse_history, pd.DataFrame([row])], ignore_index=True
        )

        do_val   = None
        ph_val   = None
        stir_val = None
        if cf is not None:
            nearby = cf[abs(cf.time_h - row.time_h) < 0.5]
            if not nearby.empty:
                do_val   = float(nearby.DO_pct.mean())  \
                           if "DO_pct"  in nearby else None
                ph_val   = float(nearby.pH.mean())      \
                           if "pH"      in nearby else None
                stir_val = float(nearby.stirrer.mean()) \
                           if "stirrer" in nearby else None

        reading = SensorReading(
            time_h       = float(row.time_h),
            DO_pct       = do_val,
            pH           = ph_val,
            temp_c       = 30.0,
            stirrer      = stir_val,
            OD660        = float(row.OD660)        if pd.notna(row.OD660)        else None,
            DCW_gL       = float(row.DCW_gL)       if pd.notna(row.DCW_gL)       else None,
            glycerol_gL  = float(row.glycerol_gL)  if "glycerol_gL"  in row.index
                           and pd.notna(row.glycerol_gL)  else None,
            methanol_gL  = float(row.methanol_gL)  if "methanol_gL"  in row.index
                           and pd.notna(row.methanol_gL)  else None,
            L1_yield_mgL = float(row.L1_yield_mgL) if "L1_yield_mgL" in row.index
                           and pd.notna(row.L1_yield_mgL) else None,
        )

        mu = float(row.mu_per_h) if pd.notna(row.mu_per_h) else None
        ctrl.step(reading, mu=mu, sparse_history=sparse_history)

    history = ctrl.get_history()

    history = history.merge(
        sparse_df[["time_h", "OD660", "DCW_gL",
                   "glycerol_gL", "methanol_gL", "L1_yield_mgL"]]
        .rename(columns=lambda c: c if c == "time_h" else f"{c}_actual"),
        on="time_h", how="left"
    )

    return history


if __name__ == "__main__":
    from loader import load_batch, load_sparse_samples

    print("=" * 60)
    print("  Bioreactor Controller — Layered Simulation")
    print("=" * 60)

    sparse = load_sparse_samples("data/result-batch_7.xlsx")
    sheets = load_batch("data/result-batch_7.xlsx")
    cont   = list(sheets.values())[0]
    cont   = cont[cont.phase == "run"].copy()

    sim = simulate_batch(sparse, cont, load_ml=True,
                         strain="H. polymorpha", product="HPV52 L1")

    cols = ["time_h", "phase", "layer_used", "action",
            "methanol_feed_pct", "glycerol_feed_pct", "carbon_ratio_pct",
            "ml_phase", "ml_confidence", "L1_predicted",
            "L1_yield_mgL_actual", "alerts"]
    print("\nSimulation results:")
    print(sim[[c for c in cols if c in sim.columns]].to_string())

    sim.to_csv("outputs/processed_data/batch7_layered_simulation.csv",
               index=False)
    print("\nSaved -> outputs/processed_data/batch7_layered_simulation.csv")
