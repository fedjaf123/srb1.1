import argparse
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Config:
    lookback_days: int = 56
    preoos_days: int = 28
    min_overlap_days: int = 14
    n_controls: int = 5
    min_controls: int = 2
    ewma_alpha: float = 0.2
    promo_threshold: float = 0.8
    promo_min_consecutive_days: int = 2
    control_min_corr: float = 0.1
    control_min_control_avg: float = 0.05


def _to_datetime_series(values: pd.Series) -> pd.Series:
    if values is None:
        return pd.Series(dtype="datetime64[ns]")
    dt = pd.to_datetime(values, errors="coerce", dayfirst=False)
    return dt


def _ensure_date_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(pd.NaT, index=df.index)
    dt = _to_datetime_series(df[col])
    return dt.dt.date


def _date_range(start: date, end: date) -> list[date]:
    days = (end - start).days
    return [start + timedelta(days=i) for i in range(days + 1)]


def load_kartice_events(events_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(events_csv, encoding="utf-8")
    if df.empty:
        return df
    df["SKU"] = df["SKU"].astype(str).str.strip()
    df = df.loc[df["SKU"].astype(str).str.len() > 0]
    df["Datum"] = pd.to_datetime(df["Datum"], errors="coerce").dt.date
    df = df.loc[df["Datum"].notna()]

    def num(name: str) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0)
        return pd.Series(0, index=df.index)

    df["Prijem_kolicina"] = num("Prijem kolicina")
    df["Izdavanje_kolicina"] = num("Izdavanje kolicina")
    df["Stanje_kolicina"] = num("Stanje zaliha kolicina")
    df["Smer"] = df.get("Smer", pd.Series("", index=df.index)).astype(str).str.strip().str.upper()
    df["Broj"] = df.get("Broj", pd.Series("", index=df.index)).astype(str).str.strip()

    df["is_sale"] = (df["Smer"] == "IZDAVANJE") & (df["Izdavanje_kolicina"] > 0)
    df["is_return"] = (df["Smer"] == "POVRAT") | (df["Izdavanje_kolicina"] < 0)
    return df


def build_daily_from_kartice(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    rows = []
    for sku, group in events.groupby("SKU"):
        group = group.sort_values(["Datum", "Broj"])
        start = group["Datum"].min()
        end = group["Datum"].max()
        dates = _date_range(start, end)

        # EOD stock: take last known stock per day, then forward-fill.
        eod = (
            group.groupby("Datum", as_index=False)
            .tail(1)[["Datum", "Stanje_kolicina"]]
            .set_index("Datum")["Stanje_kolicina"]
        )
        eod = eod.reindex(dates).ffill().fillna(0)

        sales = group.loc[group["is_sale"]].groupby("Datum")["Izdavanje_kolicina"].sum()
        returns = (
            group.loc[group["is_return"]]
            .assign(ret=lambda g: g["Izdavanje_kolicina"].abs())
            .groupby("Datum")["ret"]
            .sum()
        )

        for d in dates:
            gross = float(sales.get(d, 0.0))
            ret = float(returns.get(d, 0.0))
            net = gross - ret
            stock = float(eod.loc[d]) if d in eod.index else 0.0
            rows.append(
                {
                    "date": d.isoformat(),
                    "sku": sku,
                    "stock_eod_qty": stock,
                    "oos_flag": 1 if stock <= 0 else 0,
                    "gross_sales_qty": gross,
                    "return_qty": ret,
                    "net_sales_qty": net,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["date_dt"] = pd.to_datetime(out["date"], errors="coerce")
    return out


def load_receipts_summary(receipts_summary_csv: Path) -> pd.DataFrame:
    if not receipts_summary_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(receipts_summary_csv, encoding="utf-8")
    if df.empty:
        return df
    df["SKU"] = df["SKU"].astype(str).str.strip()
    df = df.loc[df["SKU"].astype(str).str.len() > 0]
    df["prvi_verifikovan_dt"] = pd.to_datetime(df.get("prvi_verifikovan"), errors="coerce").dt.date
    return df[["SKU", "poslato_sum", "pristiglo_sum", "prvi_verifikovan_dt"]].copy()


def _sql_date_expr(field: str) -> str:
    # Keep as raw string; we'll parse in pandas because SP exports are often dd.mm.yyyy.
    return f"COALESCE(o.{field}, '')"


def load_sp_price_daily(
    db_path: Path,
    date_field: str,
    start_date: date | None,
    end_date: date | None,
) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    try:
        expr = _sql_date_expr(date_field)
        where = [
            "oi.product_code IS NOT NULL AND TRIM(oi.product_code) != ''",
            f"TRIM({expr}) != ''",
            "o.id = oi.order_id",
        ]
        params: list[Any] = []
        # Exclude cancelled/returned-ish statuses; keep it simple and consistent.
        where.append("(o.status IS NULL OR (lower(o.status) NOT LIKE '%otkaz%' AND lower(o.status) NOT LIKE '%vrac%'))")

        query = (
            "SELECT "
            f"{expr} AS d_raw, "
            "TRIM(oi.product_code) AS sku, "
            "COALESCE(oi.qty, 0) AS qty, "
            "COALESCE(oi.cod_amount, 0) AS cod_amount, "
            "COALESCE(oi.discount, 0) AS item_discount, "
            "COALESCE(oi.extra_discount, 0) AS order_discount "
            "FROM order_items oi JOIN orders o ON o.id = oi.order_id "
            f"WHERE {' AND '.join(where)}"
        )
        rows = conn.execute(query, params).fetchall()
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["date_raw", "sku", "qty", "cod_amount", "item_discount", "order_discount"])
    dt = pd.to_datetime(df["date_raw"], errors="coerce", dayfirst=True)
    df["date"] = dt.dt.date
    df = df.loc[df["date"].notna()]
    if start_date:
        df = df.loc[df["date"] >= start_date]
    if end_date:
        df = df.loc[df["date"] <= end_date]
    df["sku"] = df["sku"].astype(str).str.strip()

    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0)
    df["cod_amount"] = pd.to_numeric(df["cod_amount"], errors="coerce").fillna(0)
    df["item_discount"] = pd.to_numeric(df["item_discount"], errors="coerce").fillna(0).clip(lower=0, upper=100)
    df["order_discount"] = pd.to_numeric(df["order_discount"], errors="coerce").fillna(0).clip(lower=0, upper=100)

    # Order discount applies first, then item discount.
    df["net_unit_price"] = df["cod_amount"] * (1 - df["order_discount"] / 100.0) * (1 - df["item_discount"] / 100.0)
    df["net_value"] = df["net_unit_price"] * df["qty"]

    df["has_discount"] = (df["order_discount"] > 0) | (df["item_discount"] > 0)

    agg = (
        df.groupby(["date", "sku"], dropna=False)
        .agg(
            sp_qty=("qty", "sum"),
            sp_net_value=("net_value", "sum"),
            sp_discount_share=("has_discount", "mean"),
            sp_avg_order_discount=("order_discount", "mean"),
            sp_avg_item_discount=("item_discount", "mean"),
        )
        .reset_index()
    )
    agg["sp_unit_net_price"] = agg["sp_net_value"] / agg["sp_qty"].replace({0: math.nan})
    agg["sp_unit_net_price"] = agg["sp_unit_net_price"].fillna(0)
    agg["date"] = agg["date"].apply(lambda d: d.isoformat())
    return agg


def detect_promos(price_daily: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if price_daily.empty:
        return pd.DataFrame()
    df = price_daily.copy()
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.loc[df["date_dt"].notna()]

    results = []
    for sku, g in df.groupby("sku"):
        g = g.sort_values("date_dt")
        flags = (g["sp_discount_share"].fillna(0) >= cfg.promo_threshold).astype(int).tolist()
        dates = g["date_dt"].dt.date.tolist()
        start_idx = None
        streak = 0
        for i, f in enumerate(flags):
            if f:
                streak += 1
                if start_idx is None and streak >= cfg.promo_min_consecutive_days:
                    start_idx = i - (cfg.promo_min_consecutive_days - 1)
            else:
                if start_idx is not None:
                    end_idx = i - 1
                    seg = g.iloc[start_idx : end_idx + 1]
                    results.append(
                        {
                            "sku": sku,
                            "promo_start": dates[start_idx].isoformat(),
                            "promo_end": dates[end_idx].isoformat(),
                            "avg_discount_share": float(seg["sp_discount_share"].mean()),
                            "avg_order_discount": float(seg["sp_avg_order_discount"].mean()),
                            "avg_item_discount": float(seg["sp_avg_item_discount"].mean()),
                        }
                    )
                start_idx = None
                streak = 0
        if start_idx is not None:
            end_idx = len(flags) - 1
            seg = g.iloc[start_idx : end_idx + 1]
            results.append(
                {
                    "sku": sku,
                    "promo_start": dates[start_idx].isoformat(),
                    "promo_end": dates[end_idx].isoformat(),
                    "avg_discount_share": float(seg["sp_discount_share"].mean()),
                    "avg_order_discount": float(seg["sp_avg_order_discount"].mean()),
                    "avg_item_discount": float(seg["sp_avg_item_discount"].mean()),
                }
            )

    return pd.DataFrame(results)


def ewma_baseline(series: pd.Series, oos_flag: pd.Series, alpha: float) -> pd.Series:
    # series and oos_flag are aligned by date order.
    ewma = None
    baseline = []
    for qty, oos in zip(series.tolist(), oos_flag.tolist()):
        if oos == 0:
            ewma = qty if ewma is None else (alpha * qty + (1 - alpha) * ewma)
        baseline.append(0.0 if ewma is None else float(ewma))
    return pd.Series(baseline, index=series.index)


def _pearson_corr(a: pd.Series, b: pd.Series) -> float | None:
    if len(a) < 2:
        return None
    if a.std(ddof=0) == 0 or b.std(ddof=0) == 0:
        return None
    return float(a.corr(b))


def _find_oos_intervals(df: pd.DataFrame) -> list[tuple[date, date]]:
    # df has date_dt and oos_flag sorted.
    intervals = []
    in_oos = False
    start = None
    last = None
    for d, flag in zip(df["date_dt"].dt.date.tolist(), df["oos_flag"].tolist()):
        if flag == 1 and not in_oos:
            in_oos = True
            start = d
        if flag == 1:
            last = d
        if flag == 0 and in_oos:
            intervals.append((start, last))
            in_oos = False
            start = None
            last = None
    if in_oos and start and last:
        intervals.append((start, last))
    return intervals


def apply_control_group_baseline(
    daily: pd.DataFrame,
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adds demand_baseline_qty, method_used, confidence_score, lost_sales_qty.
    Returns (daily_out, audit_df).
    """
    if daily.empty:
        return daily, pd.DataFrame()

    daily = daily.copy()
    daily["date_dt"] = pd.to_datetime(daily["date"], errors="coerce")
    daily = daily.loc[daily["date_dt"].notna()]

    # Pivot for fast access.
    pivot_qty = daily.pivot(index="date_dt", columns="sku", values="net_sales_qty").fillna(0.0)
    pivot_oos = daily.pivot(index="date_dt", columns="sku", values="oos_flag").fillna(0).astype(int)

    baselines = {}
    methods = {}
    confidences = {}
    audits = []

    for sku in pivot_qty.columns.tolist():
        sku_series = pivot_qty[sku]
        sku_oos = pivot_oos[sku]
        base_ewma = ewma_baseline(sku_series, sku_oos, cfg.ewma_alpha)

        baseline = base_ewma.copy()
        method = pd.Series(["EWMA_FALLBACK"] * len(sku_series), index=sku_series.index)
        conf = pd.Series([0.2] * len(sku_series), index=sku_series.index)

        df_sku = pd.DataFrame({"date_dt": sku_series.index, "oos_flag": sku_oos.values})
        intervals = _find_oos_intervals(df_sku.sort_values("date_dt"))
        for start, end in intervals:
            look_start = start - timedelta(days=cfg.lookback_days)
            look_end = start - timedelta(days=1)
            pre_start = start - timedelta(days=cfg.preoos_days)
            pre_end = start - timedelta(days=1)

            look_mask = (pivot_qty.index.date >= look_start) & (pivot_qty.index.date <= look_end)
            pre_mask = (pivot_qty.index.date >= pre_start) & (pivot_qty.index.date <= pre_end)
            oos_mask = (pivot_qty.index.date >= start) & (pivot_qty.index.date <= end)

            if not look_mask.any() or not pre_mask.any():
                continue

            # Candidate controls: in-stock during entire OOS interval for target.
            in_stock_during_oos = (pivot_oos.loc[oos_mask, :] == 0).all(axis=0)
            in_stock_during_oos = in_stock_during_oos[in_stock_during_oos.index != sku]
            candidates = in_stock_during_oos[in_stock_during_oos].index.tolist()
            if not candidates:
                continue

            corr_scores = []
            for cand in candidates:
                both_instock = (pivot_oos.loc[look_mask, sku] == 0) & (pivot_oos.loc[look_mask, cand] == 0)
                if int(both_instock.sum()) < cfg.min_overlap_days:
                    continue
                a = pivot_qty.loc[look_mask, sku].loc[both_instock]
                b = pivot_qty.loc[look_mask, cand].loc[both_instock]
                c = _pearson_corr(a, b)
                if c is None or c < cfg.control_min_corr:
                    continue
                corr_scores.append((cand, c, int(both_instock.sum())))

            if not corr_scores:
                continue

            corr_scores.sort(key=lambda x: x[1], reverse=True)
            top = corr_scores[: cfg.n_controls]

            # Calibration ratios on pre window.
            controls = []
            for cand, cval, overlap_days in top:
                both_instock = (pivot_oos.loc[pre_mask, sku] == 0) & (pivot_oos.loc[pre_mask, cand] == 0)
                if not both_instock.any():
                    continue
                target_avg = float(pivot_qty.loc[pre_mask, sku].loc[both_instock].mean())
                control_avg = float(pivot_qty.loc[pre_mask, cand].loc[both_instock].mean())
                if control_avg < cfg.control_min_control_avg:
                    continue
                ratio = target_avg / control_avg if control_avg else None
                if ratio is None or not math.isfinite(ratio):
                    continue
                controls.append(
                    {
                        "control_sku": cand,
                        "corr": float(cval),
                        "overlap_days": int(overlap_days),
                        "ratio": float(ratio),
                    }
                )

            if len(controls) < cfg.min_controls:
                continue

            # Predict baseline during OOS days.
            preds = []
            for dtt in pivot_qty.index[oos_mask]:
                vals = []
                for c in controls:
                    cand = c["control_sku"]
                    vals.append(c["ratio"] * float(pivot_qty.at[dtt, cand]))
                preds.append((dtt, float(pd.Series(vals).median())))

            for dtt, bval in preds:
                baseline.at[dtt] = bval
                method.at[dtt] = "CONTROL_GROUP"
                avg_corr = sum(c["corr"] for c in controls) / len(controls)
                avg_overlap = sum(c["overlap_days"] for c in controls) / len(controls)
                conf_val = 0.0
                conf_val += 0.55 * max(0.0, min(1.0, (avg_corr - cfg.control_min_corr) / (1 - cfg.control_min_corr)))
                conf_val += 0.25 * min(1.0, math.log(1 + len(controls)) / math.log(1 + cfg.n_controls))
                conf_val += 0.20 * min(1.0, avg_overlap / max(cfg.preoos_days, 1))
                conf.at[dtt] = float(max(0.0, min(1.0, conf_val)))

            audits.append(
                {
                    "sku": sku,
                    "oos_start": start.isoformat(),
                    "oos_end": end.isoformat(),
                    "controls": json.dumps(controls, ensure_ascii=False),
                }
            )

        baselines[sku] = baseline
        methods[sku] = method
        confidences[sku] = conf

    # Unpivot back.
    base_df = pd.DataFrame(baselines).stack().reset_index()
    base_df.columns = ["date_dt", "sku", "demand_baseline_qty"]
    method_df = pd.DataFrame(methods).stack().reset_index()
    method_df.columns = ["date_dt", "sku", "method_used"]
    conf_df = pd.DataFrame(confidences).stack().reset_index()
    conf_df.columns = ["date_dt", "sku", "confidence_score"]

    merged = daily.merge(base_df, on=["date_dt", "sku"], how="left").merge(method_df, on=["date_dt", "sku"], how="left").merge(conf_df, on=["date_dt", "sku"], how="left")
    merged["demand_baseline_qty"] = pd.to_numeric(merged["demand_baseline_qty"], errors="coerce").fillna(0.0)
    merged["confidence_score"] = pd.to_numeric(merged["confidence_score"], errors="coerce").fillna(0.0)
    merged["method_used"] = merged["method_used"].fillna("EWMA_FALLBACK")
    merged["lost_sales_qty"] = (merged["demand_baseline_qty"] - merged["net_sales_qty"]).clip(lower=0)
    merged.loc[merged["oos_flag"] == 0, "lost_sales_qty"] = 0.0
    return merged, pd.DataFrame(audits)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build per-SKU daily metrics (OOS + demand baseline + lost sales).")
    parser.add_argument("--events", type=Path, default=Path(r"Kalkulacije_kartice_art\izlaz\kartice_events.csv"))
    parser.add_argument("--receipts-summary", type=Path, default=Path(r"Kalkulacije_kartice_art\izlaz\sp_prijemi_summary.csv"))
    parser.add_argument("--db", type=Path, default=Path(r"SRB1.0 - Copy.db"))
    parser.add_argument("--sp-date-field", type=str, default="picked_up_at", choices=["picked_up_at", "created_at", "delivered_at"])
    parser.add_argument("--out", type=Path, default=Path(r"Kalkulacije_kartice_art\izlaz"))
    parser.add_argument("--start", type=str, default="")
    parser.add_argument("--end", type=str, default="")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config override.")
    args = parser.parse_args()

    cfg = Config()
    if args.config and args.config.exists():
        data = json.loads(args.config.read_text(encoding="utf-8"))
        cfg = Config(**{**cfg.__dict__, **{k: data[k] for k in data if k in cfg.__dict__}})

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else None
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else None

    args.out.mkdir(parents=True, exist_ok=True)

    events = load_kartice_events(args.events)
    daily = build_daily_from_kartice(events)
    if daily.empty:
        print("No kartice events to process.")
        return 0

    if start_date:
        daily = daily.loc[pd.to_datetime(daily["date"]) >= pd.to_datetime(start_date)]
    if end_date:
        daily = daily.loc[pd.to_datetime(daily["date"]) <= pd.to_datetime(end_date)]

    receipts = load_receipts_summary(args.receipts_summary)
    if not receipts.empty:
        daily = daily.merge(receipts.rename(columns={"SKU": "sku"}), on="sku", how="left")
        daily["verified_available_flag"] = 1
        mask = daily["prvi_verifikovan_dt"].notna()
        if mask.any():
            daily.loc[mask, "verified_available_flag"] = (
                pd.to_datetime(daily.loc[mask, "date"]).dt.date >= daily.loc[mask, "prvi_verifikovan_dt"]
            ).astype(int)
    else:
        daily["verified_available_flag"] = 1

    price_daily = load_sp_price_daily(args.db, args.sp_date_field, start_date, end_date)
    if not price_daily.empty:
        daily = daily.merge(price_daily, left_on=["date", "sku"], right_on=["date", "sku"], how="left")
    else:
        daily["sp_unit_net_price"] = 0.0
        daily["sp_discount_share"] = 0.0
        daily["sp_qty"] = 0.0
        daily["sp_net_value"] = 0.0

    for col in [
        "sp_unit_net_price",
        "sp_discount_share",
        "sp_qty",
        "sp_net_value",
        "sp_avg_order_discount",
        "sp_avg_item_discount",
    ]:
        if col in daily.columns:
            daily[col] = pd.to_numeric(daily[col], errors="coerce").fillna(0.0)

    # Baseline + lost sales from qty series.
    daily, audit = apply_control_group_baseline(daily, cfg)

    # Don't count lost sales before first verified receipt (if known).
    if "verified_available_flag" in daily.columns:
        daily.loc[daily["verified_available_flag"] == 0, ["demand_baseline_qty", "lost_sales_qty"]] = 0.0

    daily["lost_sales_value_est"] = daily["lost_sales_qty"] * pd.to_numeric(daily["sp_unit_net_price"], errors="coerce").fillna(0.0)

    promos = detect_promos(price_daily, cfg)

    out_daily = args.out / "sku_daily_metrics.csv"
    keep_cols = [
        "date",
        "sku",
        "stock_eod_qty",
        "oos_flag",
        "verified_available_flag",
        "gross_sales_qty",
        "return_qty",
        "net_sales_qty",
        "demand_baseline_qty",
        "lost_sales_qty",
        "lost_sales_value_est",
        "sp_unit_net_price",
        "sp_discount_share",
        "sp_qty",
        "sp_net_value",
        "method_used",
        "confidence_score",
        "poslato_sum",
        "pristiglo_sum",
        "prvi_verifikovan_dt",
    ]
    keep = [c for c in keep_cols if c in daily.columns]
    daily[keep].sort_values(["sku", "date"]).to_csv(out_daily, index=False, encoding="utf-8")
    print(f"wrote: {out_daily}")

    out_audit = args.out / "sku_controls_audit.csv"
    if not audit.empty:
        audit.to_csv(out_audit, index=False, encoding="utf-8")
        print(f"wrote: {out_audit}")

    out_promos = args.out / "sku_promo_periods.csv"
    if not promos.empty:
        promos.to_csv(out_promos, index=False, encoding="utf-8")
        print(f"wrote: {out_promos}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
