import argparse
import csv
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

try:
    import pdfplumber
except Exception:  # pragma: no cover - optional dependency
    pdfplumber = None


DATE_RE = re.compile(r"\b(\d{2}\.\d{2}\.\d{4})\b")
DOC_RE = re.compile(r"^([A-Z]{1,3}-[0-9A-Z-]+)")
NUM_RE = re.compile(r"[-+]?\d{1,3}(?:\.\d{3})*(?:,\d+)?|[-+]?\d+(?:,\d+)?")


@dataclass
class StockEvent:
    date: datetime
    qty: float
    doc: str


def _normalize(text: str) -> str:
    text = str(text).strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text


def _num_from_token(token: str) -> float:
    token = token.replace(".", "").replace(",", ".")
    try:
        return float(token)
    except ValueError:
        return 0.0


def _extract_sku_from_text(text: str) -> str:
    match = re.search(r"\(([A-Za-z0-9-]+)\)", text)
    if match:
        return match.group(1).strip()
    return _normalize(text)


def _parse_date(value) -> datetime | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip().rstrip(".")
    if not text:
        return None
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def extract_kalkulacije(
    excel_path: Path, output_dir: Path, sheet_name: str = "Minimax"
) -> tuple[Path, Path]:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    orig_cols = list(df.columns)
    norm_map = {c: _normalize(c) for c in orig_cols}

    reverse = {}
    for orig, norm in norm_map.items():
        reverse.setdefault(norm, orig)

    def get_col(norm_name: str) -> str | None:
        return reverse.get(norm_name)

    vp_col = get_col("VP")
    if vp_col is None:
        raise ValueError("Missing 'VP' column")

    vp_norm = df[vp_col].astype(str).str.replace(" ", "").str.upper()
    calc = df.loc[vp_norm == "PS"].copy()

    want_norm = [
        "VP",
        "Broj",
        "Datum",
        "Opis",
        "Artikal",
        "Kolicina",
        "Nabavna vrednost",
        "Prodajna vrednost",
        "Prodajna cena",
        "U skladiste",
        "ID prometa u zalihama",
    ]
    missing = [n for n in want_norm if get_col(n) is None]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    cols = {n: get_col(n) for n in want_norm}
    calc = calc[[cols[n] for n in want_norm]].copy()
    calc.columns = want_norm

    calc["Marza"] = calc["Prodajna vrednost"] - calc["Nabavna vrednost"]
    calc["Broj"] = calc["Broj"].astype(str).str.strip()
    calc["Artikal"] = calc["Artikal"].astype(str).str.strip()
    calc["Opis"] = calc["Opis"].astype(str).str.strip()
    calc["U skladiste"] = calc["U skladiste"].astype(str).str.strip()
    calc["SKU"] = calc["Artikal"].apply(_extract_sku_from_text)

    out_detail = output_dir / "kalkulacije_marza.csv"
    calc.to_csv(out_detail, index=False)

    # Aggregation by SKU + article name
    agg = (
        calc.assign(
            Kolicina=calc["Kolicina"].fillna(0),
            Nabavna=calc["Nabavna vrednost"].fillna(0),
            Prodajna=calc["Prodajna vrednost"].fillna(0),
            Marza=calc["Marza"].fillna(0),
        )
        .groupby(["SKU", "Artikal"], dropna=False)
        .agg(
            kolicina_sum=("Kolicina", "sum"),
            nabavna_sum=("Nabavna", "sum"),
            prodajna_sum=("Prodajna", "sum"),
            marza_sum=("Marza", "sum"),
            doc_count=("Broj", "nunique"),
            first_date=("Datum", "min"),
            last_date=("Datum", "max"),
        )
        .reset_index()
    )

    out_agg = output_dir / "kalkulacije_marza_agregat.csv"
    agg.to_csv(out_agg, index=False)
    return out_detail, out_agg


def extract_zero_intervals(pdf_path: Path, output_dir: Path) -> Path:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is required for PDF parsing")

    current_article = None
    current_sku = None
    events: dict[str, list[StockEvent]] = {}
    article_names: dict[str, str] = {}

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("ARTIKAL:"):
                    current_article = line.replace("ARTIKAL:", "", 1).strip()
                    current_sku = _extract_sku_from_text(current_article)
                    if not current_sku:
                        current_sku = current_article
                    article_names[current_sku] = current_article
                    events.setdefault(current_sku, [])
                    continue

                if current_sku is None:
                    continue

                doc_match = DOC_RE.match(line)
                date_match = DATE_RE.search(line)
                if not doc_match or not date_match:
                    continue

                nums = NUM_RE.findall(line)
                if len(nums) < 2:
                    continue

                qty = _num_from_token(nums[-2])
                doc = doc_match.group(1)
                date = datetime.strptime(date_match.group(1), "%d.%m.%Y")
                events[current_sku].append(StockEvent(date=date, qty=qty, doc=doc))

    rows = []
    for sku, items in events.items():
        if not items:
            continue
        items.sort(key=lambda x: x.date)
        prev_qty = None
        zero_start = None
        zero_start_doc = None
        for event in items:
            qty = event.qty
            if prev_qty is None:
                prev_qty = qty
                if qty <= 0:
                    zero_start = event.date
                    zero_start_doc = event.doc
                continue

            if prev_qty > 0 and qty <= 0 and zero_start is None:
                zero_start = event.date
                zero_start_doc = event.doc
            elif prev_qty <= 0 and qty > 0 and zero_start is not None:
                days = (event.date - zero_start).days
                rows.append(
                    {
                        "SKU": sku,
                        "Artikal": article_names.get(sku, ""),
                        "Zero od": zero_start.date().isoformat(),
                        "Zero do": event.date.date().isoformat(),
                        "Dani bez zalihe": days,
                        "Dokument start": zero_start_doc,
                        "Dokument kraj": event.doc,
                    }
                )
                zero_start = None
                zero_start_doc = None
            prev_qty = qty

        if zero_start is not None:
            rows.append(
                {
                    "SKU": sku,
                    "Artikal": article_names.get(sku, ""),
                    "Zero od": zero_start.date().isoformat(),
                    "Zero do": "",
                    "Dani bez zalihe": "",
                    "Dokument start": zero_start_doc,
                    "Dokument kraj": "",
                }
            )

    out_path = output_dir / "kartice_zero_intervali.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "SKU",
                "Artikal",
                "Zero od",
                "Zero do",
                "Dani bez zalihe",
                "Dokument start",
                "Dokument kraj",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def _iter_sales_files(sales_path: Path) -> list[Path]:
    if sales_path.is_file():
        return [sales_path]
    if not sales_path.exists():
        return []
    return sorted([p for p in sales_path.glob("*.xlsx") if p.is_file()])


def _load_sales_rows(sales_path: Path) -> pd.DataFrame:
    frames = []
    files = _iter_sales_files(sales_path)
    for path in files:
        xls = pd.ExcelFile(path)
        for sheet in xls.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet)
            if df.empty:
                continue
            norm_cols = {c: _normalize(c) for c in df.columns}
            df = df.rename(columns=norm_cols)
            if "Sifra proizvoda" not in df.columns or "Kolicina proizvoda" not in df.columns:
                continue
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_zero_intervals(path: Path) -> dict[str, list[tuple[datetime, datetime | None]]]:
    intervals: dict[str, list[tuple[datetime, datetime | None]]] = {}
    if not path.exists():
        return intervals
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sku = (row.get("SKU") or "").strip()
            if not sku:
                continue
            start = _parse_date(row.get("Zero od"))
            end = _parse_date(row.get("Zero do"))
            if start is None:
                continue
            intervals.setdefault(sku, []).append((start, end))
    return intervals


def _overlap_days(
    start: datetime, end: datetime, zero_intervals: list[tuple[datetime, datetime | None]]
) -> int:
    range_start = start
    range_end = end + timedelta(days=1)
    total = 0
    for zero_start, zero_end in zero_intervals:
        z_start = zero_start
        z_end = zero_end + timedelta(days=1) if zero_end else range_end
        overlap_start = max(range_start, z_start)
        overlap_end = min(range_end, z_end)
        if overlap_end > overlap_start:
            total += (overlap_end - overlap_start).days
    return total


def build_sales_availability(
    sales_path: Path, zero_csv: Path, output_dir: Path, status_filter: str | None
) -> Path | None:
    sales = _load_sales_rows(sales_path)
    if sales.empty:
        return None

    if status_filter and "Status" in sales.columns:
        status_norm = sales["Status"].astype(str).apply(_normalize).str.strip().str.lower()
        sales = sales.loc[status_norm == status_filter.strip().lower()]

    sales["SKU"] = sales["Sifra proizvoda"].astype(str).str.strip()
    sales["Kolicina"] = pd.to_numeric(sales["Kolicina proizvoda"], errors="coerce").fillna(0)
    def _num_col(name: str) -> pd.Series:
        if name in sales.columns:
            return pd.to_numeric(sales[name], errors="coerce").fillna(0)
        return pd.Series(0, index=sales.index)

    otkup = _num_col("Otkup proizvoda")
    popust = _num_col("Popust proizvoda")
    sales["Net value"] = (otkup - popust) * sales["Kolicina"]

    date_col = None
    for candidate in ("Datum isporuke", "Datum kreiranja", "Datum"):
        if candidate in sales.columns:
            date_col = candidate
            break
    if date_col is None:
        return None

    sales["Datum"] = sales[date_col].apply(_parse_date)
    sales = sales.loc[sales["Datum"].notna()]
    if sales.empty:
        return None

    zero_intervals = _load_zero_intervals(zero_csv)
    rows = []
    for sku, group in sales.groupby("SKU"):
        total_qty = group["Kolicina"].sum()
        first_date = group["Datum"].min().date()
        last_date = group["Datum"].max().date()
        total_days = (last_date - first_date).days + 1
        unavailable_days = _overlap_days(
            datetime.combine(first_date, datetime.min.time()),
            datetime.combine(last_date, datetime.min.time()),
            zero_intervals.get(sku, []),
        )
        available_days = max(total_days - unavailable_days, 0)
        avg_daily_qty = total_qty / available_days if available_days else 0
        lost_qty = avg_daily_qty * unavailable_days if unavailable_days else 0
        total_net = group["Net value"].sum()
        avg_net = total_net / available_days if available_days else 0
        lost_net = avg_net * unavailable_days if unavailable_days else 0
        rows.append(
            {
                "SKU": sku,
                "Total prodato": total_qty,
                "Prvi datum": first_date.isoformat(),
                "Zadnji datum": last_date.isoformat(),
                "Ukupno dana": total_days,
                "Dani bez zalihe": unavailable_days,
                "Dani dostupno": available_days,
                "Prosek dnevno": round(avg_daily_qty, 4),
                "Ukupno neto": round(total_net, 2),
                "Prosek neto": round(avg_net, 4),
                "Procjena gubitka neto": round(lost_net, 2),
                "Procena izgubljeno": round(lost_qty, 4),
            }
        )

    out_path = output_dir / "prodaja_avg_i_gubitak.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def write_top_lists(agg_path: Path, sales_path: Path, output_dir: Path) -> list[Path]:
    outputs = []
    if agg_path.exists():
        df = pd.read_csv(agg_path)
        if "marza_sum" in df.columns:
            top_marza = df.sort_values("marza_sum", ascending=False).head(20)
            out_marza = output_dir / "top_marza.csv"
            top_marza.to_csv(out_marza, index=False)
            outputs.append(out_marza)

    if sales_path.exists():
        df = pd.read_csv(sales_path)
        if "Dani bez zalihe" in df.columns:
            top_zero = df.sort_values("Dani bez zalihe", ascending=False).head(20)
            out_zero = output_dir / "top_nedostupnost.csv"
            top_zero.to_csv(out_zero, index=False)
            outputs.append(out_zero)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract kalkulacije and kartice intervals into CSVs."
    )
    parser.add_argument(
        "--excel",
        type=Path,
        default=Path(
            r"C:\Users\HOME\Desktop\Srbija1.0 aplikacija\Kalkulacije_kartice_art\Minimax111.xlsx"
        ),
        help="Path to Minimax Excel export with kalkulacije",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path(
            r"C:\Users\HOME\Desktop\Srbija1.0 aplikacija\Kalkulacije_kartice_art\Kartica_20260119_070347.pdf"
        ),
        help="Path to kartica PDF export",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            r"C:\Users\HOME\Desktop\Srbija1.0 aplikacija\Kalkulacije_kartice_art\izlaz"
        ),
        help="Output directory",
    )
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Skip PDF parsing for kartice",
    )
    parser.add_argument(
        "--skip-excel",
        action="store_true",
        help="Skip Excel parsing for kalkulacije",
    )
    parser.add_argument(
        "--sales",
        type=Path,
        default=Path(
            r"C:\Users\HOME\Desktop\Srbija1.0 aplikacija\SP-Narudzbe"
        ),
        help="Path to sales Excel file or folder (SP-Narudzbe)",
    )
    parser.add_argument(
        "--sales-status",
        type=str,
        default="Isporuceno",
        help="Filter sales by status (normalized, optional)",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    detail_path = args.out / "kalkulacije_marza.csv"
    agg_path = args.out / "kalkulacije_marza_agregat.csv"
    if not args.skip_excel:
        detail_path, agg_path = extract_kalkulacije(args.excel, args.out)
        print(f"kalkulacije: {detail_path}")
        print(f"kalkulacije agregat: {agg_path}")

    if not args.skip_pdf:
        zero_path = extract_zero_intervals(args.pdf, args.out)
        print(f"kartice zero: {zero_path}")
    else:
        zero_path = args.out / "kartice_zero_intervali.csv"

    sales_out = build_sales_availability(
        args.sales, zero_path, args.out, args.sales_status
    )
    if sales_out:
        print(f"prodaja avg: {sales_out}")
        top_outputs = write_top_lists(agg_path, sales_out, args.out)
        for path in top_outputs:
            print(f"top: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
