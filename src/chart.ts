// The orbit chart: equatorial plan and meridional elevation of the hole, drawn like an engraved plate.
// Radial scale is logarithmic so the horizon and an observer at 1000 M fit on one sheet.
import { ergosphere, horizonInner, horizonOuter, isco, photonOrbit } from "./kerr.ts";

export interface ChartState {
  a: number;
  diskIn: number;
  diskOut: number;
  r: number;
  theta: number;
  phi: number;
  trail: [number, number, number][]; // r, theta, phi
}

const INK = "#2b2118", RED = "#8a2f1d", FAINT = "rgba(43,33,24,0.35)";

export function drawChart(cv: HTMLCanvasElement, s: ChartState) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const g = cv.getContext("2d")!;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const half = w / 2;
  const R = Math.min(half, h) * 0.42;
  const rMax = Math.max(s.r * 1.25, s.diskOut * 1.1, 12);
  const k = R / Math.log1p(rMax);
  const map = (r: number) => k * Math.log1p(r);

  const panel = (cx: number, cy: number, side: boolean, label: string) => {
    g.save();
    g.translate(cx, cy);
    g.font = "11px 'IBM Plex Mono', monospace";
    g.fillStyle = INK;
    g.textAlign = "center";
    g.fillText(label, 0, -R - 12);
    // graticule
    g.strokeStyle = FAINT;
    g.lineWidth = 0.5;
    for (const rr of [1, 10, 100, 1000]) {
      if (rr > rMax) break;
      g.beginPath();
      g.arc(0, 0, map(rr), 0, 2 * Math.PI);
      g.setLineDash([1, 3]);
      g.stroke();
      g.setLineDash([]);
      g.fillStyle = FAINT;
      g.fillText(`${rr}M`, map(rr) * Math.SQRT1_2 + 10, -map(rr) * Math.SQRT1_2);
    }
    g.beginPath(); g.moveTo(-R, 0); g.lineTo(R, 0); g.moveTo(0, -R); g.lineTo(0, R); g.stroke();

    // disk
    g.fillStyle = "rgba(138,47,29,0.18)";
    g.strokeStyle = RED;
    g.lineWidth = 1;
    if (side) {
      g.fillRect(map(s.diskIn), -1.5, map(s.diskOut) - map(s.diskIn), 3);
      g.fillRect(-map(s.diskOut), -1.5, map(s.diskOut) - map(s.diskIn), 3);
    } else {
      g.beginPath();
      g.arc(0, 0, map(s.diskOut), 0, 2 * Math.PI);
      g.arc(0, 0, map(s.diskIn), 0, 2 * Math.PI, true);
      g.fill();
      hatch(g, map(s.diskIn), map(s.diskOut));
    }
    // ergosphere
    g.strokeStyle = INK;
    g.setLineDash([4, 3]);
    g.beginPath();
    for (let i = 0; i <= 120; i++) {
      const th = (i / 120) * 2 * Math.PI;
      const re = side ? ergosphere(s.a, Math.cos(th)) : ergosphere(s.a, 0);
      const x = map(re) * Math.sin(th), y = -map(re) * Math.cos(th);
      if (i === 0) g.moveTo(x, y); else g.lineTo(x, y);
    }
    g.stroke();
    g.setLineDash([]);
    // photon orbits & ISCO (equatorial)
    if (!side) {
      const rings: [number, number[]][] = [[photonOrbit(s.a, true), [2, 2]], [photonOrbit(s.a, false), [2, 2]], [isco(s.a), [6, 2, 1, 2]]];
      for (const [r, dash] of rings) {
        g.setLineDash(dash);
        g.beginPath();
        g.arc(0, 0, map(r), 0, 2 * Math.PI);
        g.stroke();
      }
      g.setLineDash([]);
    }
    // horizons
    g.fillStyle = "#0d0a07";
    g.beginPath();
    g.arc(0, 0, map(horizonOuter(s.a)), 0, 2 * Math.PI);
    g.fill();
    if (s.a !== 0) {
      g.strokeStyle = "rgba(239,230,208,0.5)";
      g.beginPath();
      g.arc(0, 0, map(horizonInner(s.a)), 0, 2 * Math.PI);
      g.stroke();
    }
    // spin arrow
    g.strokeStyle = RED;
    g.fillStyle = RED;
    if (side) {
      g.beginPath(); g.moveTo(0, 0); g.lineTo(0, -map(3) * Math.sign(s.a || 1)); g.stroke();
    } else if (s.a !== 0) {
      g.beginPath();
      g.arc(0, 0, map(horizonOuter(s.a)) + 4, -0.3, -1.2 * Math.sign(s.a), s.a > 0);
      g.stroke();
    }

    // trail and observer
    const pt = (r: number, th: number, ph: number): [number, number] =>
      side ? [map(r) * Math.sin(th) * (Math.cos(ph) >= 0 ? 1 : -1), -map(r) * Math.cos(th)] : [map(r * Math.sin(th)) * Math.cos(ph), -map(r * Math.sin(th)) * Math.sin(ph)];
    g.strokeStyle = RED;
    g.lineWidth = 1;
    g.beginPath();
    s.trail.forEach(([r, th, ph], i) => { const [x, y] = pt(r, th, ph); if (i === 0) g.moveTo(x, y); else g.lineTo(x, y); });
    g.stroke();
    const [ox, oy] = pt(s.r, s.theta, s.phi);
    g.fillStyle = RED;
    g.beginPath(); g.arc(ox, oy, 3.5, 0, 2 * Math.PI); g.fill();
    g.strokeStyle = INK;
    g.beginPath(); g.arc(ox, oy, 6.5, 0, 2 * Math.PI); g.stroke();
    g.restore();
  };
  panel(half / 2, h / 2 + 6, false, "PLAN · EQUATORIAL");
  panel(half + half / 2, h / 2 + 6, true, "ELEVATION · MERIDIONAL");
}

function hatch(g: CanvasRenderingContext2D, r0: number, r1: number) {
  g.save();
  g.beginPath();
  g.arc(0, 0, r1, 0, 2 * Math.PI);
  g.arc(0, 0, r0, 0, 2 * Math.PI, true);
  g.clip();
  g.strokeStyle = "rgba(138,47,29,0.45)";
  g.lineWidth = 0.6;
  for (let x = -r1; x < r1; x += 4) { g.beginPath(); g.moveTo(x, -r1); g.lineTo(x + r1, r1); g.stroke(); }
  g.restore();
}
