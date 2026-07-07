"""Generate auto-updating assets for the status slides.

Reads every production summary (phonon/scripts/out/prod/<study>/summary.csv)
that exists and writes
  - slide_numbers.tex : \\newcommand macros for every headline number used in
    status_slides.tex, falling back to the dense-reference values (marked
    "dense") for studies whose production rerun has not landed yet,
  - fig/ratio_vs_T.pdf and fig/ratio_vs_L.pdf : combined cross-system panels.

Re-run this script whenever new production results land; the deck then
rebuilds with the fresh numbers and plots without touching the .tex.
Cheap (single process, seconds) -- safe to run next to production jobs.
"""

from __future__ import annotations

import csv
import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROD = REPO / "phonon" / "scripts" / "out" / "prod"
FIG = HERE / "fig"
FIG.mkdir(exist_ok=True)

STUDIES = ["cnt33", "cnt80", "sinw_d5a", "sinw_d11a", "srtio3", "sifilm"]
LABELS = {
    "cnt33": "CNT(3,3)",
    "cnt80": "CNT(8,0)",
    "sinw_d5a": "SiNW d5a",
    "sinw_d11a": "SiNW d11a",
    "srtio3": r"SrTiO$_3$",
    "sifilm": "Si film",
}
# Dense-reference fallbacks (document/src/results.tex) used until the
# production rerun of a study lands.
DENSE_FALLBACK = {
    "cnt80": ("0.775 (L1, 300\\,K)", "dense"),
    "sinw_d5a": ("0.942 (30\\,K, $\\lambda{=}0.3$)", "dense"),
    "sinw_d11a": ("0.997 ($\\lambda{=}0.3$)", "dense"),
    "srtio3": ("--", "pending"),
    "sifilm": ("0.52--0.38 (1.2--3.1\\,nm)", "dense"),
}


def read_summary(study: str) -> list[dict]:
    path = PROD / study / "summary.csv"
    if not path.is_file():
        return []
    with open(path) as fh:
        return [row for row in csv.DictReader(fh)]


def fnum(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def main() -> None:
    data = {s: read_summary(s) for s in STUDIES}
    macros: dict[str, str] = {}
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    macros["GenStamp"] = stamp

    # --- cnt33 headline numbers -------------------------------------------
    cnt = data["cnt33"]
    tsweep = sorted(
        (r for r in cnt if r["sweep"] == "temperature"), key=lambda r: float(r["t_mean"])
    )
    lsweep = sorted(
        (r for r in cnt if r["sweep"] == "length"), key=lambda r: int(r["n_slabs"])
    )
    if tsweep:
        macros["cntRatioLowT"] = fnum(float(tsweep[0]["ratio"]))
        macros["cntRatioRT"] = fnum(float(tsweep[-1]["ratio"]))
        macros["cntTLow"] = f"{float(tsweep[0]['t_mean']):.0f}"
        macros["cntGballRT"] = fnum(float(tsweep[-1]["G_ball_W_per_m2_K"]) / 1e8, 1)
        macros["cntConsRT"] = f"{float(tsweep[-1]['lead_conservation']):.0e}"
    if lsweep:
        macros["cntLadder"] = " / ".join(fnum(float(r["ratio"])) for r in lsweep)
        macros["cntLadderL"] = "/".join(r["n_slabs"] for r in lsweep)

    # --- per-study one-line status + headline ratio ----------------------
    # LaTeX macro names must be letters-only.
    MACRO_NAME = {
        "cnt33": "statCntThreeThree",
        "cnt80": "statCntEightZero",
        "sinw_d5a": "statSinwDFive",
        "sinw_d11a": "statSinwDEleven",
        "srtio3": "statSrtio",
        "sifilm": "statSifilm",
    }
    for study in STUDIES:
        key = MACRO_NAME[study]
        rows = data[study]
        if rows:
            n = len(rows)
            best = min(rows, key=lambda r: abs(float(r.get("t_mean") or 300) - 300))
            macros[key] = (
                f"{fnum(float(best['ratio']))} "
                f"({float(best['t_mean']):.0f}\\,K) -- {n} pts (prod)"
            )
        else:
            val, src = DENSE_FALLBACK.get(study, ("--", "pending"))
            macros[key] = f"{val} ({src})" if val != "--" else "running\\dots"

    # --- status table ------------------------------------------------------
    rows_tex = []
    for study in STUDIES:
        rows = data[study]
        npts = str(len(rows)) if rows else "--"
        state = "done" if rows else "running"
        if rows:
            rt = min(rows, key=lambda r: abs(float(r.get("t_mean") or 300) - 300))
            head = fnum(float(rt["ratio"]))
        else:
            head, _ = DENSE_FALLBACK.get(study, ("--", ""))
            head = head.split(" ")[0]
        rows_tex.append(f"    {LABELS[study]} & {npts} & {head} & {state} \\\\")
    macros["ProdStatusRows"] = "\n".join(rows_tex)

    # --- combined plots ----------------------------------------------------
    plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

    fig1, ax = plt.subplots(figsize=(4.4, 3.1))
    any_t = False
    for study in STUDIES:
        pts = sorted(
            (r for r in data[study] if r["sweep"] == "temperature"),
            key=lambda r: float(r["t_mean"]),
        )
        if len(pts) >= 2:
            ax.plot(
                [float(r["t_mean"]) for r in pts],
                [float(r["ratio"]) for r in pts],
                "o-",
                label=LABELS[study],
            )
            any_t = True
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"$G_{\mathrm{anh}}/G_{\mathrm{ball}}$")
    ax.set_ylim(top=1.0)
    ax.grid(alpha=0.3)
    if any_t:
        ax.legend(fontsize=9)
    fig1.tight_layout()
    fig1.savefig(FIG / "ratio_vs_T.pdf")

    fig2, ax = plt.subplots(figsize=(4.4, 3.1))
    any_l = False
    for study in STUDIES:
        pts = sorted(
            (r for r in data[study] if r["sweep"] == "length"),
            key=lambda r: int(r["n_slabs"]),
        )
        if len(pts) >= 2:
            ax.plot(
                [int(r["n_slabs"]) for r in pts],
                [float(r["ratio"]) for r in pts],
                "s-",
                label=LABELS[study],
            )
            any_l = True
    ax.set_xlabel("Device length (transport cells)")
    ax.set_ylabel(r"$G_{\mathrm{anh}}/G_{\mathrm{ball}}$")
    ax.grid(alpha=0.3)
    if any_l:
        ax.legend(fontsize=9)
    fig2.tight_layout()
    fig2.savefig(FIG / "ratio_vs_L.pdf")

    # --- write macros ------------------------------------------------------
    # Defaults so the deck always compiles even with no data at all.
    defaults = {
        "cntRatioLowT": "0.981",
        "cntRatioRT": "0.878",
        "cntTLow": "30",
        "cntGballRT": "5.1",
        "cntConsRT": "1e-04",
        "cntLadder": "--",
        "cntLadderL": "--",
    }
    for k, v in defaults.items():
        macros.setdefault(k, v)

    with open(HERE / "slide_numbers.tex", "w") as fh:
        fh.write(f"% auto-generated by make_slide_assets.py on {stamp} -- do not edit\n")
        for name, val in macros.items():
            if name == "ProdStatusRows":
                fh.write(f"\\newcommand{{\\{name}}}{{%\n{val}\n}}\n")
            else:
                fh.write(f"\\newcommand{{\\{name}}}{{{val}}}\n")

    done = [s for s in STUDIES if data[s]]
    print(f"assets written ({stamp}); studies with production data: {done}")


if __name__ == "__main__":
    main()
